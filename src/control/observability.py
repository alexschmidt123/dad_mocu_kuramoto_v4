"""Objective-observability diagnostic: do probe histories change terminal u_ctrl?"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.control.banks import extract_U_bank
from src.control.cuda_control import CudaControlEngine
from src.control.generate import control_banks_certified
from src.control.posterior_ctrl import (
    normalize_log_weights,
    posterior_ess,
    posterior_safe_u_ctrl,
)
from src.control.terminal_rule import observe_with_keyed_noise
from src.control.u_req import ControlSpec
from src.data import lookup_action_y
from src.rollout import RandomSelector, update_log_weights
from src.run_context import ExperimentRun, load_experiment_run
from src.swing_equation_ode.design import build_catalog, build_simulator
from src.swing_equation_ode.simulator import system_mk
from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables


@dataclass(frozen=True)
class ObservabilityGateConfig:
    enabled: bool = True
    num_rollouts: int = 1000
    seed: int = 1234
    min_unique_terminal_controls: int = 3
    min_terminal_control_std: float = 0.01
    min_fraction_changed_from_prior: float = 0.05
    min_true_safety_rate: float = 1.0
    min_posterior_mean_u_spearman: float = 0.10
    require_real_better_than_shuffled: bool = True

    @classmethod
    def from_cfg(cls, cfg: Any) -> ObservabilityGateConfig:
        raw = dict(getattr(cfg, "raw", {}).get("objective_observability") or {})
        return cls(
            enabled=bool(raw.get("enabled", True)),
            num_rollouts=int(raw.get("num_rollouts", 1000)),
            seed=int(raw.get("seed", 1234)),
            min_unique_terminal_controls=int(raw.get("min_unique_terminal_controls", 3)),
            min_terminal_control_std=float(raw.get("min_terminal_control_std", 0.01)),
            min_fraction_changed_from_prior=float(
                raw.get("min_fraction_changed_from_prior", 0.05)
            ),
            min_true_safety_rate=float(raw.get("min_true_safety_rate", 1.0)),
            min_posterior_mean_u_spearman=float(
                raw.get("min_posterior_mean_u_spearman", 0.10)
            ),
            require_real_better_than_shuffled=bool(
                raw.get("require_real_better_than_shuffled", True)
            ),
        )


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation; NaN if undefined."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size != x.size:
        return float("nan")
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        r = spearmanr(x, y)
        return float(r.correlation)
    except Exception:
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])


def verify_observability_prerequisites(run: ExperimentRun) -> None:
    """Fail fast if probe/control banks or config are missing (do not generate)."""
    if not run.exp_dir.is_dir():
        raise FileNotFoundError(f"experiment directory missing: {run.exp_dir}")
    cfg_path = run.exp_dir / "run_config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"resolved configuration missing: {cfg_path}")
    for split in ("train", "test"):
        probe = run.data_path / f"{split}.json"
        if not probe.is_file():
            raise FileNotFoundError(f"probe bank missing: {probe}")
    if not run.train_systems:
        raise RuntimeError("posterior particle bank empty (train systems)")
    missing_u = [i for i, s in enumerate(run.train_systems) if "u_req" not in s]
    if missing_u:
        raise RuntimeError(
            f"control U-bank missing on train systems (e.g. index {missing_u[0]}). "
            "Run: python -m src.cli generate-control-bank --config <config>"
        )
    side = run.data_path / "train_control_bank.json"
    if not side.is_file():
        raise FileNotFoundError(f"control U-bank sidecar missing: {side}")
    certified, detail = control_banks_certified(run.data_path)
    if not certified:
        raise RuntimeError(
            "control U-bank not certified (oracle/u_max/particle safety must be 1.0). "
            f"Detail: {detail}"
        )
    test_missing = [i for i, s in enumerate(run.test_systems) if "u_req" not in s]
    if test_missing:
        raise RuntimeError(
            f"test systems missing u_req (e.g. index {test_missing[0]}); "
            "regenerate control bank."
        )


def _posterior_mean_U(U: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    u = np.asarray(U, dtype=np.float64)
    return float(np.sum(w * u))


def run_diagnostic_rollout(
    *,
    system: dict[str, Any],
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    horizon: int,
    n_actions: int,
    sigma_y: float,
    alpha: float,
    rng: np.random.Generator,
    update_posterior: bool = True,
    forced_sequence: list[int] | None = None,
    forced_observations: list[float] | None = None,
    theta_test_id: int = -1,
    margin: float = 0.0,
    u_grid=None,
    global_seed: int | None = None,
    rollout_id: int = 0,
    use_keyed_noise: bool = True,
) -> dict[str, Any]:
    """
    One diagnostic history. Probe sampler is uniform without replacement.

    Observation noise matches the named Random / pilot path when
    ``use_keyed_noise`` is True (default): keyed Gaussian noise keyed by
    (global_seed, theta_id, rollout_id, step). Set ``use_keyed_noise=False``
    only for legacy banked-lookup diagnostics.
    """
    log_w = np.asarray(table_support.log_p0, dtype=np.float64).copy()
    w0 = normalize_log_weights(log_w)
    u_prior = float(
        posterior_safe_u_ctrl(U_support, w0, alpha, margin=margin, u_grid=u_grid)
    )

    used: set[int] = set()
    seq: list[int] = []
    y_list: list[float] = []
    u_path: list[float] = [u_prior]
    ess_path: list[float] = [float(posterior_ess(w0))]
    selector = RandomSelector(n_actions=n_actions)

    for t in range(horizon):
        weights = normalize_log_weights(log_w)
        if forced_sequence is not None:
            a = int(forced_sequence[t])
            if a in used:
                raise ValueError(f"forced action {a} already used")
        else:
            a = int(selector.select(used=used, rng=rng))

        if forced_observations is not None:
            y = float(forced_observations[t])
        elif use_keyed_noise and global_seed is not None:
            y = observe_with_keyed_noise(
                system,
                a,
                sigma_y=sigma_y,
                global_seed=int(global_seed),
                theta_id=int(theta_test_id),
                rollout_id=int(rollout_id),
                step=t,
            )
        else:
            y = float(lookup_action_y(system, a))

        if update_posterior:
            centres = y_sim_last_step_from_tables(table_support, [a])
            log_w = update_log_weights(log_w, y, centres, sigma_y)

        seq.append(a)
        y_list.append(y)
        used.add(a)
        weights = normalize_log_weights(log_w)
        u_path.append(
            float(
                posterior_safe_u_ctrl(
                    U_support, weights, alpha, margin=margin, u_grid=u_grid
                )
            )
        )
        ess_path.append(float(posterior_ess(weights)))

    weights = normalize_log_weights(log_w)
    u_final = float(u_path[-1])
    u_req = float(system["u_req"]) if "u_req" in system else float("nan")
    return {
        "theta_test_id": int(theta_test_id),
        "sequence": seq,
        "y_obs": y_list,
        "u_ctrl_path": u_path,
        "ess_path": ess_path,
        "u_ctrl_prior": u_prior,
        "u_ctrl_final": u_final,
        "u_req_true": u_req,
        "posterior_mean_U": _posterior_mean_U(U_support, weights),
        "weights": weights,
        "log_weights": log_w,
        "changed": abs(u_final - u_prior) > 1e-12,
        "reduced": u_final < u_prior - 1e-12,
        "increased": u_final > u_prior + 1e-12,
    }


def evaluate_true_safety(
    control_engine: CudaControlEngine,
    system: dict[str, Any],
    u_ctrl: float,
) -> dict[str, float]:
    M, K = system_mk(system, control_engine.N)
    return control_engine.evaluate_one(M, K, float(u_ctrl))


def evaluate_gate(
    summary: dict[str, Any],
    gate: ObservabilityGateConfig,
) -> dict[str, Any]:
    checks = {
        "unique_final_u_ctrl_count": (
            int(summary["unique_final_u_ctrl_count"]) >= gate.min_unique_terminal_controls,
            int(summary["unique_final_u_ctrl_count"]),
            gate.min_unique_terminal_controls,
        ),
        "final_u_ctrl_std": (
            float(summary["final_u_ctrl_std"]) >= gate.min_terminal_control_std,
            float(summary["final_u_ctrl_std"]),
            gate.min_terminal_control_std,
        ),
        "fraction_changed_from_prior": (
            float(summary["fraction_changed_from_prior"])
            >= gate.min_fraction_changed_from_prior,
            float(summary["fraction_changed_from_prior"]),
            gate.min_fraction_changed_from_prior,
        ),
        "true_safety_rate": (
            float(summary["true_safety_rate"]) >= gate.min_true_safety_rate,
            float(summary["true_safety_rate"]),
            gate.min_true_safety_rate,
        ),
        "real_spearman": (
            float(summary["real_spearman"]) >= gate.min_posterior_mean_u_spearman
            if np.isfinite(summary["real_spearman"])
            else False,
            float(summary["real_spearman"]),
            gate.min_posterior_mean_u_spearman,
        ),
    }
    if gate.require_real_better_than_shuffled:
        checks["real_better_than_shuffled"] = (
            float(summary["real_spearman"]) > float(summary["shuffled_spearman"])
            if np.isfinite(summary["real_spearman"])
            and np.isfinite(summary["shuffled_spearman"])
            else False,
            float(summary["real_spearman"]),
            float(summary["shuffled_spearman"]),
        )

    failed = [k for k, (ok, *_rest) in checks.items() if not ok]
    return {
        "passed": len(failed) == 0,
        "failed_checks": failed,
        "checks": {
            k: {"passed": bool(ok), "value": val, "threshold": thr}
            for k, (ok, val, thr) in checks.items()
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _write_report_md(path: Path, summary: dict[str, Any], gate_result: dict[str, Any]) -> None:
    lines = [
        "# Objective observability report",
        "",
        f"**Gate: {'PASS' if gate_result['passed'] else 'FAIL'}**",
        "",
        "## Summary",
        "",
        f"- prior terminal control: `{summary['prior_u_ctrl']}`",
        f"- unique final controls: `{summary['unique_final_u_ctrl_count']}`",
        f"- final u_ctrl mean / std: `{summary['final_u_ctrl_mean']:.6f}` / `{summary['final_u_ctrl_std']:.6f}`",
        f"- final u_ctrl min / median / max: "
        f"`{summary['final_u_ctrl_min']:.6f}` / `{summary['final_u_ctrl_median']:.6f}` / "
        f"`{summary['final_u_ctrl_max']:.6f}`",
        f"- fraction changed / reduced / increased from prior: "
        f"`{summary['fraction_changed_from_prior']:.4f}` / "
        f"`{summary['fraction_reduced_from_prior']:.4f}` / "
        f"`{summary['fraction_increased_from_prior']:.4f}`",
        f"- true-system safety rate: `{summary['true_safety_rate']:.4f}`",
        f"- mean excess control: `{summary['mean_excess_control']:.6f}`",
        f"- real Spearman (posterior mean U vs u_req): `{summary['real_spearman']:.4f}`",
        f"- shuffled Spearman: `{summary['shuffled_spearman']:.4f}`",
        f"- no-update check passed: `{summary['no_update_check_passed']}`",
        "",
        "## Gate checks",
        "",
    ]
    for name, c in gate_result["checks"].items():
        lines.append(
            f"- `{name}`: {'PASS' if c['passed'] else 'FAIL'} "
            f"(value={c['value']}, threshold={c['threshold']})"
        )
    if gate_result["failed_checks"]:
        lines.extend(["", f"Failed: `{', '.join(gate_result['failed_checks'])}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_plots(out_dir: Path, summary: dict[str, Any], stepwise: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    finals = np.asarray(summary.get("final_u_ctrl_values", []), dtype=np.float64)
    if finals.size:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(finals, bins=min(20, max(5, len(set(finals.tolist())))), color="#2c5f7c", edgecolor="white")
        ax.axvline(summary["prior_u_ctrl"], color="#c44e52", ls="--", label="prior")
        ax.set_xlabel("final u_ctrl")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "final_u_ctrl_hist.png", dpi=120)
        plt.close(fig)

    if stepwise:
        steps = [r["step"] for r in stepwise]
        stds = [r["u_ctrl_std"] for r in stepwise]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(steps, stds, marker="o", color="#2c5f7c")
        ax.set_xlabel("probe step t")
        ax.set_ylabel("std of u_ctrl(h_t)")
        ax.set_title("Stepwise terminal-control variation")
        fig.tight_layout()
        fig.savefig(plots / "stepwise_u_ctrl_std.png", dpi=120)
        plt.close(fig)


def check_objective_observability(
    exp_dir: Path | str,
    *,
    project_root: Path | None = None,
    num_rollouts: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Run the objective-observability diagnostic and gate.

    Does not train or evaluate named methods (dad/myopic/fixed/random).
    """
    from src.config import repo_root

    root = project_root or repo_root()
    run = load_experiment_run(Path(exp_dir), root)
    verify_observability_prerequisites(run)

    gate = ObservabilityGateConfig.from_cfg(run.cfg)
    n_roll = int(num_rollouts if num_rollouts is not None else gate.num_rollouts)
    rng_seed = int(seed if seed is not None else gate.seed)
    if n_roll <= 0:
        raise ValueError("num_rollouts must be positive")

    control_spec = ControlSpec.from_cfg(run.cfg)
    from src.control.safety_calibration import resolve_terminal_rule

    term_rule = resolve_terminal_rule(run.exp_dir, control_spec)
    # Apply calibrated alpha/margin for this diagnostic (same rule as train/eval).
    control_spec = ControlSpec(
        alpha=term_rule.alpha,
        safety_margin=term_rule.margin,
        rocof_limit_hz_s=control_spec.rocof_limit_hz_s,
        delta_f_nadir_hz=control_spec.delta_f_nadir_hz,
        profile=control_spec.profile,
        contingency=control_spec.contingency,
        u_candidates=control_spec.u_candidates,
        myopic_hypothetical=control_spec.myopic_hypothetical,
        fixed_exhaustive_threshold=control_spec.fixed_exhaustive_threshold,
        fixed_noise_replicas=control_spec.fixed_noise_replicas,
        fixed_greedy_restarts=control_spec.fixed_greedy_restarts,
        T_obs_sec=control_spec.T_obs_sec,
        ode_dt=control_spec.ode_dt,
        fs_hz=control_spec.fs_hz,
    )
    catalog = build_catalog(run.cfg)
    n_actions = len(catalog)
    horizon = int(run.cfg.step_number)
    if horizon < 1:
        raise ValueError("experiment step_number T must be >= 1")
    if horizon > n_actions:
        raise ValueError(f"T={horizon} exceeds n_actions={n_actions}")

    rng = np.random.default_rng(rng_seed)
    # Use the same posterior particle bank as pilot/eval (support split), not full train.
    from src.control.pilot import load_pilot_splits
    from src.contrastive.spce import log_prior_uniform_discrete

    splits = load_pilot_splits(run.exp_dir, run)
    table_support = TableThetaSupport(
        systems=splits["support_systems"],
        log_p0=log_prior_uniform_discrete(len(splits["support_systems"])),
    )
    U_support = extract_U_bank(splits["support_systems"])
    if U_support.size != len(table_support):
        raise RuntimeError("U-bank / particle bank size mismatch")
    pilot_cfg = dict(run.cfg.raw.get("pilot") or {})
    global_seed = int(pilot_cfg.get("global_seed", 1234))

    w0 = normalize_log_weights(np.asarray(table_support.log_p0, dtype=np.float64))
    prior_u = float(
        posterior_safe_u_ctrl(
            U_support,
            w0,
            control_spec.alpha,
            margin=control_spec.safety_margin,
            u_grid=control_spec.u_candidates,
        )
    )

    sim = build_simulator(run.cfg)
    sim.T_obs_sec = float(control_spec.T_obs_sec)
    sim.ode_dt = float(control_spec.ode_dt)
    sim.fs_hz = float(control_spec.fs_hz)
    control_engine = CudaControlEngine(sim, control_spec)

    test_systems = list(run.test_systems)
    real_rows: list[dict[str, Any]] = []
    no_update_ok = True

    print(
        f"  Objective observability: T={horizon}  rollouts={n_roll}  "
        f"seed={rng_seed}  prior_u_ctrl={prior_u:.6f}  particles={len(table_support)}"
    )

    for i in range(n_roll):
        tid = int(rng.integers(0, len(test_systems)))
        system = test_systems[tid]
        # Independent RNG stream per rollout for probe sampling.
        roll_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        real = run_diagnostic_rollout(
            system=system,
            table_support=table_support,
            U_support=U_support,
            horizon=horizon,
            n_actions=n_actions,
            sigma_y=float(run.cfg.sigma_y),
            alpha=float(control_spec.alpha),
            rng=roll_rng,
            update_posterior=True,
            theta_test_id=tid,
            margin=float(control_spec.safety_margin),
            u_grid=control_spec.u_candidates,
            global_seed=global_seed,
            rollout_id=i,
            use_keyed_noise=True,
        )
        # No-update check on the same forced actions/obs.
        no_up = run_diagnostic_rollout(
            system=system,
            table_support=table_support,
            U_support=U_support,
            horizon=horizon,
            n_actions=n_actions,
            sigma_y=float(run.cfg.sigma_y),
            alpha=float(control_spec.alpha),
            rng=roll_rng,
            update_posterior=False,
            forced_sequence=list(real["sequence"]),
            forced_observations=list(real["y_obs"]),
            theta_test_id=tid,
            margin=float(control_spec.safety_margin),
            u_grid=control_spec.u_candidates,
            global_seed=global_seed,
            rollout_id=i,
            use_keyed_noise=True,
        )
        if abs(no_up["u_ctrl_final"] - prior_u) > 1e-9:
            no_update_ok = False

        metrics = evaluate_true_safety(control_engine, system, real["u_ctrl_final"])
        excess = float(real["u_ctrl_final"] - real["u_req_true"])
        real_rows.append(
            {
                **{k: real[k] for k in (
                    "theta_test_id", "sequence", "y_obs", "u_ctrl_path", "ess_path",
                    "u_ctrl_prior", "u_ctrl_final", "u_req_true",
                    "posterior_mean_U", "changed", "reduced", "increased",
                )},
                "excess_control": excess,
                "max_rocof": metrics["rocof_max"],
                "frequency_nadir": metrics["delta_f_nadir"],
                "safe_total": bool(metrics["safe_total"] >= 0.5),
                "no_update_u_ctrl": no_up["u_ctrl_final"],
            }
        )

    # Shuffled-observation check: permute observation trajectories across rollouts.
    shuf_rng = np.random.default_rng(rng_seed + 17)
    perm = shuf_rng.permutation(n_roll)
    shuffled_rows: list[dict[str, Any]] = []
    for i in range(n_roll):
        base = real_rows[i]
        donor = real_rows[int(perm[i])]
        sh = run_diagnostic_rollout(
            system=test_systems[int(base["theta_test_id"])],
            table_support=table_support,
            U_support=U_support,
            horizon=horizon,
            n_actions=n_actions,
            sigma_y=float(run.cfg.sigma_y),
            alpha=float(control_spec.alpha),
            rng=np.random.default_rng(0),
            update_posterior=True,
            forced_sequence=list(base["sequence"]),
            forced_observations=list(donor["y_obs"]),
            theta_test_id=int(base["theta_test_id"]),
            margin=float(control_spec.safety_margin),
            u_grid=control_spec.u_candidates,
        )
        shuffled_rows.append(
            {
                "theta_test_id": base["theta_test_id"],
                "u_req_true": base["u_req_true"],
                "u_ctrl_final": sh["u_ctrl_final"],
                "posterior_mean_U": sh["posterior_mean_U"],
            }
        )

    finals = np.asarray([r["u_ctrl_final"] for r in real_rows], dtype=np.float64)
    ureq = np.asarray([r["u_req_true"] for r in real_rows], dtype=np.float64)
    post_mean = np.asarray([r["posterior_mean_U"] for r in real_rows], dtype=np.float64)
    shuf_mean = np.asarray([r["posterior_mean_U"] for r in shuffled_rows], dtype=np.float64)
    excess = np.asarray([r["excess_control"] for r in real_rows], dtype=np.float64)
    safe = np.asarray([1.0 if r["safe_total"] else 0.0 for r in real_rows], dtype=np.float64)

    real_spear = spearman_corr(post_mean, ureq)
    shuf_spear = spearman_corr(shuf_mean, ureq)

    # Stepwise variation of u_ctrl(h_t) across rollouts.
    stepwise: list[dict[str, Any]] = []
    for t in range(horizon + 1):
        vals = np.asarray([r["u_ctrl_path"][t] for r in real_rows], dtype=np.float64)
        if t == 0:
            frac_changed_prev = 0.0
        else:
            prev = np.asarray([r["u_ctrl_path"][t - 1] for r in real_rows], dtype=np.float64)
            frac_changed_prev = float(np.mean(np.abs(vals - prev) > 1e-12))
        ess_vals = np.asarray(
            [
                float(r["ess_path"][t])
                if "ess_path" in r and t < len(r["ess_path"])
                else float("nan")
                for r in real_rows
            ],
            dtype=np.float64,
        )
        stepwise.append(
            {
                "step": t,
                "u_ctrl_mean": float(np.mean(vals)),
                "u_ctrl_std": float(np.std(vals)),
                "u_ctrl_min": float(np.min(vals)),
                "u_ctrl_max": float(np.max(vals)),
                "n_unique": int(len({float(x) for x in vals.tolist()})),
                "fraction_changed_from_previous_step": frac_changed_prev,
                "posterior_ess_mean": float(np.nanmean(ess_vals)),
            }
        )

    unique_finals = sorted({float(x) for x in finals.tolist()})
    summary = {
        "exp_dir": str(run.exp_dir),
        "data_path": str(run.data_path),
        "T": horizon,
        "n_actions": n_actions,
        "n_rollouts": n_roll,
        "seed": rng_seed,
        "n_particles": len(table_support),
        "alpha": float(control_spec.alpha),
        "prior_u_ctrl": prior_u,
        "unique_final_u_ctrl_count": int(len(unique_finals)),
        "unique_final_u_ctrl": unique_finals,
        "final_u_ctrl_mean": float(np.mean(finals)),
        "final_u_ctrl_std": float(np.std(finals)),
        "final_u_ctrl_min": float(np.min(finals)),
        "final_u_ctrl_median": float(np.median(finals)),
        "final_u_ctrl_max": float(np.max(finals)),
        "final_u_ctrl_values": finals.tolist(),
        "fraction_changed_from_prior": float(np.mean([r["changed"] for r in real_rows])),
        "fraction_reduced_from_prior": float(np.mean([r["reduced"] for r in real_rows])),
        "fraction_increased_from_prior": float(np.mean([r["increased"] for r in real_rows])),
        "true_safety_rate": float(np.mean(safe)),
        "mean_excess_control": float(np.mean(excess)),
        "real_spearman": real_spear,
        "shuffled_spearman": shuf_spear,
        "no_update_check_passed": bool(no_update_ok),
        "methods_trained": [],
        "methods_evaluated": [],
        "gate_config": asdict(gate),
    }

    gate_result = evaluate_gate(summary, gate)
    if not no_update_ok:
        gate_result["passed"] = False
        gate_result["failed_checks"] = list(
            dict.fromkeys([*gate_result["failed_checks"], "no_update_check"])
        )
        gate_result["checks"]["no_update_check"] = {
            "passed": False,
            "value": False,
            "threshold": True,
        }
    else:
        gate_result["checks"]["no_update_check"] = {
            "passed": True,
            "value": True,
            "threshold": True,
        }

    summary["gate"] = gate_result

    out = run.exp_dir / "diagnostics" / "objective_observability"
    out.mkdir(parents=True, exist_ok=True)
    (out / "observability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_csv(
        out / "observability_summary.csv",
        [
            {
                "prior_u_ctrl": summary["prior_u_ctrl"],
                "unique_final_u_ctrl_count": summary["unique_final_u_ctrl_count"],
                "final_u_ctrl_mean": summary["final_u_ctrl_mean"],
                "final_u_ctrl_std": summary["final_u_ctrl_std"],
                "final_u_ctrl_min": summary["final_u_ctrl_min"],
                "final_u_ctrl_median": summary["final_u_ctrl_median"],
                "final_u_ctrl_max": summary["final_u_ctrl_max"],
                "fraction_changed_from_prior": summary["fraction_changed_from_prior"],
                "fraction_reduced_from_prior": summary["fraction_reduced_from_prior"],
                "fraction_increased_from_prior": summary["fraction_increased_from_prior"],
                "true_safety_rate": summary["true_safety_rate"],
                "mean_excess_control": summary["mean_excess_control"],
                "real_spearman": summary["real_spearman"],
                "shuffled_spearman": summary["shuffled_spearman"],
                "gate_passed": gate_result["passed"],
            }
        ],
        [
            "prior_u_ctrl",
            "unique_final_u_ctrl_count",
            "final_u_ctrl_mean",
            "final_u_ctrl_std",
            "final_u_ctrl_min",
            "final_u_ctrl_median",
            "final_u_ctrl_max",
            "fraction_changed_from_prior",
            "fraction_reduced_from_prior",
            "fraction_increased_from_prior",
            "true_safety_rate",
            "mean_excess_control",
            "real_spearman",
            "shuffled_spearman",
            "gate_passed",
        ],
    )
    _write_csv(
        out / "rollout_details.csv",
        [
            {
                "rollout_id": i,
                "theta_test_id": r["theta_test_id"],
                "sequence": " ".join(str(a) for a in r["sequence"]),
                "y_obs": " ".join(f"{y:.8g}" for y in r["y_obs"]),
                "u_ctrl_prior": r["u_ctrl_prior"],
                "u_ctrl_final": r["u_ctrl_final"],
                "u_req_true": r["u_req_true"],
                "posterior_mean_U": r["posterior_mean_U"],
                "excess_control": r["excess_control"],
                "safe_total": r["safe_total"],
                "changed": r["changed"],
                "reduced": r["reduced"],
                "increased": r["increased"],
            }
            for i, r in enumerate(real_rows)
        ],
        [
            "rollout_id",
            "theta_test_id",
            "sequence",
            "y_obs",
            "u_ctrl_prior",
            "u_ctrl_final",
            "u_req_true",
            "posterior_mean_U",
            "excess_control",
            "safe_total",
            "changed",
            "reduced",
            "increased",
        ],
    )
    _write_csv(
        out / "stepwise_observability.csv",
        stepwise,
        [
            "step",
            "u_ctrl_mean",
            "u_ctrl_std",
            "u_ctrl_min",
            "u_ctrl_max",
            "n_unique",
            "fraction_changed_from_previous_step",
            "posterior_ess_mean",
        ],
    )
    _write_csv(
        out / "shuffled_summary.csv",
        [
            {
                "rollout_id": i,
                "theta_test_id": r["theta_test_id"],
                "u_req_true": r["u_req_true"],
                "posterior_mean_U": r["posterior_mean_U"],
                "u_ctrl_final": r["u_ctrl_final"],
            }
            for i, r in enumerate(shuffled_rows)
        ],
        ["rollout_id", "theta_test_id", "u_req_true", "posterior_mean_U", "u_ctrl_final"],
    )
    _write_report_md(out / "objective_observability_report.md", summary, gate_result)
    _make_plots(out, summary, stepwise)

    print(
        f"  prior_u={prior_u:.4f}  unique_final={summary['unique_final_u_ctrl_count']}  "
        f"std={summary['final_u_ctrl_std']:.4f}  changed={summary['fraction_changed_from_prior']:.3f}  "
        f"safety={summary['true_safety_rate']:.3f}  "
        f"spear_real={real_spear:.3f}  spear_shuf={shuf_spear:.3f}"
    )
    if gate_result["passed"]:
        print(f"  Objective-observability gate PASS → {out}")
    else:
        print(f"  Objective-observability gate FAIL → {out}")
        print(f"  Failed checks: {gate_result['failed_checks']}")
        for name in gate_result["failed_checks"]:
            c = gate_result["checks"][name]
            print(f"    - {name}: value={c['value']}  threshold={c['threshold']}")

    return summary
