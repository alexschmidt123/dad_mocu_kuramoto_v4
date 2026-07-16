"""Calibrate posterior terminal-control rule for empirical true-system safety."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.control.banks import extract_U_bank
from src.control.cuda_control import CudaControlEngine
from src.control.observability import evaluate_true_safety
from src.control.posterior_ctrl import (
    TerminalControlRule,
    normalize_log_weights,
    posterior_ess,
    posterior_safe_u_ctrl,
    weighted_cdf_at,
)
from src.control.u_req import ControlSpec
from src.contrastive.spce import log_prior_uniform_discrete
from src.data import lookup_action_y
from src.rollout import update_log_weights
from src.run_context import ExperimentRun, load_experiment_run
from src.swing_equation_ode.design import build_catalog, build_simulator
from src.swing_equation_ode.simulator import system_mk
from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables


CALIBRATED_RULE_NAME = "calibrated_terminal_rule.json"


@dataclass(frozen=True)
class SafetyCalibrationConfig:
    enabled: bool = True
    mode: str = "calibrate"  # "calibrate" | "frozen"
    frozen_rule_path: str = ""
    expected_alpha: float = 0.05
    expected_margin: float | None = None
    num_rollouts: int = 2000
    seed: int = 2468
    support_fraction: float = 0.625  # of train → posterior particles
    calibration_fraction: float = 0.1875
    validation_fraction: float = 0.1875
    alpha_grid: tuple[float, ...] = (0.05, 0.02, 0.01, 0.005, 0.001, 0.0)
    margin_grid: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
    min_calibration_safety_rate: float = 1.0
    min_validation_safety_rate: float = 1.0
    under_control_tolerance: float = 1e-9

    @classmethod
    def from_cfg(cls, cfg: Any) -> SafetyCalibrationConfig:
        raw = dict(getattr(cfg, "raw", {}).get("control_safety_calibration") or {})
        em = raw.get("expected_margin", None)
        return cls(
            enabled=bool(raw.get("enabled", True)),
            mode=str(raw.get("mode", "calibrate")),
            frozen_rule_path=str(raw.get("frozen_rule_path", "") or ""),
            expected_alpha=float(raw.get("expected_alpha", 0.05)),
            expected_margin=None if em is None else float(em),
            num_rollouts=int(raw.get("num_rollouts", 2000)),
            seed=int(raw.get("seed", 2468)),
            support_fraction=float(raw.get("support_fraction", 0.625)),
            calibration_fraction=float(raw.get("calibration_fraction", 0.1875)),
            validation_fraction=float(raw.get("validation_fraction", 0.1875)),
            alpha_grid=tuple(float(x) for x in raw.get("alpha_grid", cls.alpha_grid)),
            margin_grid=tuple(float(x) for x in raw.get("margin_grid", cls.margin_grid)),
            min_calibration_safety_rate=float(raw.get("min_calibration_safety_rate", 1.0)),
            min_validation_safety_rate=float(raw.get("min_validation_safety_rate", 1.0)),
            under_control_tolerance=float(raw.get("under_control_tolerance", 1e-9)),
        )


def calibrated_rule_path(exp_dir: Path) -> Path:
    return (
        Path(exp_dir) / "diagnostics" / "control_safety_calibration" / CALIBRATED_RULE_NAME
    )


def resolve_terminal_rule(exp_dir: Path | None, control_spec: ControlSpec) -> TerminalControlRule:
    """Prefer policy-robust or experiment-calibrated rule; else YAML alpha/margin."""
    base = control_spec.terminal_rule()
    if exp_dir is None:
        return base
    exp_dir = Path(exp_dir)
    candidates = [
        exp_dir / "selected_policy_robust_rule.json",
        calibrated_rule_path(exp_dir),
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return base
    raw = json.loads(path.read_text(encoding="utf-8"))
    rule_raw = raw.get("rule") or raw
    rule = TerminalControlRule.from_dict(rule_raw)
    # Always snap on the configured candidate grid.
    if not rule.u_candidates:
        rule = TerminalControlRule(
            alpha=rule.alpha, margin=rule.margin, u_candidates=control_spec.u_candidates
        )
    return rule


def make_train_splits(
    n_train: int,
    cfg: SafetyCalibrationConfig,
    rng: np.random.Generator,
) -> dict[str, list[int]]:
    """Disjoint support / calibration / validation index sets from training only."""
    if n_train < 6:
        raise ValueError(f"need at least 6 train systems for splits, got {n_train}")
    perm = rng.permutation(n_train).tolist()
    n_sup = max(2, int(round(cfg.support_fraction * n_train)))
    n_cal = max(2, int(round(cfg.calibration_fraction * n_train)))
    n_val = max(2, n_train - n_sup - n_cal)
    # Adjust if rounding overflowed.
    while n_sup + n_cal + n_val > n_train:
        if n_sup > n_cal and n_sup > n_val:
            n_sup -= 1
        elif n_cal >= n_val:
            n_cal -= 1
        else:
            n_val -= 1
    while n_sup + n_cal + n_val < n_train:
        n_sup += 1
    support = sorted(perm[:n_sup])
    calibration = sorted(perm[n_sup : n_sup + n_cal])
    validation = sorted(perm[n_sup + n_cal : n_sup + n_cal + n_val])
    assert len(set(support) & set(calibration)) == 0
    assert len(set(support) & set(validation)) == 0
    assert len(set(calibration) & set(validation)) == 0
    return {
        "support_ids": support,
        "calibration_ids": calibration,
        "validation_ids": validation,
        "test_ids_untouched": True,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def diagnose_observability_unsafe_rollouts(
    run: ExperimentRun,
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """
    Recompute posteriors for existing observability rollouts and diagnose under-control.
    Does not use the final test set for rule selection — diagnosis only.
    """
    obs_csv = (
        run.exp_dir / "diagnostics" / "objective_observability" / "rollout_details.csv"
    )
    if not obs_csv.is_file():
        raise FileNotFoundError(
            f"missing {obs_csv}; run check-objective-observability first for diagnosis, "
            "or skip diagnosis if calibrating from scratch."
        )

    control_spec = ControlSpec.from_cfg(run.cfg)
    # Observability used the full train particle bank + baseline alpha (no margin).
    mc_seed = int(run.cfg.prior.get("mc_support_seed", run.meta.test_seed))
    table_support = TableThetaSupport.from_train(
        run.train_systems, run.cfg, np.random.default_rng(mc_seed),
    )
    U = extract_U_bank(table_support.systems)
    alpha = float(control_spec.alpha)
    q_level = 1.0 - alpha
    N = int(run.meta.n_buses)

    rows_in = list(csv.DictReader(obs_csv.open(encoding="utf-8")))
    out_rows: list[dict[str, Any]] = []
    invariant_failures: list[int] = []

    for r in rows_in:
        seq = [int(x) for x in str(r["sequence"]).split()]
        y_obs = [float(x) for x in str(r["y_obs"]).split()]
        tid = int(r["theta_test_id"])
        system = run.test_systems[tid]
        u_req = float(r["u_req_true"])
        u_ctrl = float(r["u_ctrl_final"])
        true_safe = str(r["safe_total"]).lower() in {"1", "true", "yes"}

        log_w = np.asarray(table_support.log_p0, dtype=np.float64).copy()
        for a, y in zip(seq, y_obs):
            centres = y_sim_last_step_from_tables(table_support, [a])
            log_w = update_log_weights(log_w, y, centres, float(run.cfg.sigma_y))
        w = normalize_log_weights(log_w)
        u_recomputed = posterior_safe_u_ctrl(U, w, alpha)
        under = float(u_req - u_ctrl)
        M, K = system_mk(system, N)
        mean_U = float(np.sum(w * U))
        std_U = float(np.sqrt(max(np.sum(w * (U - mean_U) ** 2), 0.0)))
        row = {
            "rollout_id": int(r["rollout_id"]),
            "theta_true_id": tid,
            "true_u_req": u_req,
            "selected_u_ctrl": u_ctrl,
            "recomputed_u_ctrl": u_recomputed,
            "under_control_amount": under,
            "true_safe": true_safe,
            "posterior_ess": posterior_ess(w),
            "max_posterior_weight": float(np.max(w)),
            "posterior_mean_U": mean_U,
            "posterior_std_U": std_U,
            "posterior_quantile_level": q_level,
            "posterior_cdf_at_true_u_req": weighted_cdf_at(U, w, u_req),
            "selected_actions": " ".join(str(a) for a in seq),
            "observations": " ".join(f"{y:.8g}" for y in y_obs),
            "true_M_values": " ".join(f"{x:.8g}" for x in M),
            "true_K_values": " ".join(f"{x:.8g}" for x in K),
        }
        out_rows.append(row)

        # Invariant: if selected control covers u_req, true system must be safe.
        if u_ctrl + tolerance >= u_req and not true_safe:
            invariant_failures.append(int(r["rollout_id"]))

    if invariant_failures:
        raise AssertionError(
            "Safety invariant failed: u_ctrl >= u_req but true_safe=False for "
            f"rollout_ids={invariant_failures[:20]}"
        )

    unsafe = [r for r in out_rows if not r["true_safe"]]
    under_pos = [r for r in out_rows if r["under_control_amount"] > tolerance]
    out_dir = run.exp_dir / "diagnostics" / "control_safety_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "rollout_id",
        "theta_true_id",
        "true_u_req",
        "selected_u_ctrl",
        "recomputed_u_ctrl",
        "under_control_amount",
        "true_safe",
        "posterior_ess",
        "max_posterior_weight",
        "posterior_mean_U",
        "posterior_std_U",
        "posterior_quantile_level",
        "posterior_cdf_at_true_u_req",
        "selected_actions",
        "observations",
        "true_M_values",
        "true_K_values",
    ]
    _write_csv(out_dir / "unsafe_rollout_diagnosis.csv", out_rows, fields)
    # Also a filtered unsafe-only view for convenience.
    _write_csv(
        out_dir / "unsafe_rollouts_only.csv",
        unsafe,
        fields,
    )
    summary = {
        "n_rollouts": len(out_rows),
        "n_unsafe": len(unsafe),
        "n_under_control": len(under_pos),
        "mean_under_control_among_unsafe": float(
            np.mean([r["under_control_amount"] for r in unsafe])
        )
        if unsafe
        else 0.0,
        "mean_cdf_at_u_req_among_unsafe": float(
            np.mean([r["posterior_cdf_at_true_u_req"] for r in unsafe])
        )
        if unsafe
        else float("nan"),
        "mean_ess_among_unsafe": float(np.mean([r["posterior_ess"] for r in unsafe]))
        if unsafe
        else float("nan"),
        "invariant_checked": True,
        "invariant_failures": invariant_failures,
    }
    (out_dir / "unsafe_diagnosis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"  Diagnosis: n={summary['n_rollouts']}  unsafe={summary['n_unsafe']}  "
        f"under_control={summary['n_under_control']}  "
        f"mean_under|unsafe={summary['mean_under_control_among_unsafe']:.4f}"
    )
    return summary


def _evaluate_rule_on_pool(
    *,
    rule: TerminalControlRule,
    true_systems: list[dict[str, Any]],
    true_ids: list[int],
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    horizon: int,
    n_actions: int,
    sigma_y: float,
    control_engine: CudaControlEngine | None,
    n_rollouts: int,
    rng: np.random.Generator,
    verify_gpu: bool = False,
    under_tol: float = 1e-9,
) -> dict[str, Any]:
    """
    Estimate safety / excess for a terminal rule.

    When ``verify_gpu`` is False, safety is the U-bank proxy
    ``u_ctrl >= u_req`` (valid because safe(θ, u) for all u ≥ u_req(θ)).
    GPU true-system checks are used for the final validation rule.
    """
    safes = []
    excesses = []
    u_ctrls = []
    unders = []
    from src.rollout import RandomSelector

    for _ in range(n_rollouts):
        j = int(rng.integers(0, len(true_systems)))
        system = true_systems[j]
        roll_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        log_w = np.asarray(table_support.log_p0, dtype=np.float64).copy()
        used: set[int] = set()
        sel = RandomSelector(n_actions=n_actions)
        for _t in range(horizon):
            a = int(sel.select(used=used, rng=roll_rng))
            y = float(lookup_action_y(system, a))
            centres = y_sim_last_step_from_tables(table_support, [a])
            log_w = update_log_weights(log_w, y, centres, sigma_y)
            used.add(a)
        w = normalize_log_weights(log_w)
        u_ctrl = float(rule.apply(U_support, w))
        u_req = float(system["u_req"])
        if verify_gpu:
            assert control_engine is not None
            metrics = evaluate_true_safety(control_engine, system, u_ctrl)
            safe = bool(metrics["safe_total"] >= 0.5)
        else:
            safe = bool(u_ctrl + under_tol >= u_req)
        safes.append(1.0 if safe else 0.0)
        excesses.append(u_ctrl - u_req)
        u_ctrls.append(u_ctrl)
        unders.append(max(u_req - u_ctrl, 0.0))
    return {
        "safety_rate": float(np.mean(safes)),
        "mean_excess": float(np.mean(excesses)),
        "mean_u_ctrl": float(np.mean(u_ctrls)),
        "mean_under_control": float(np.mean(unders)),
        "std_u_ctrl": float(np.std(u_ctrls)),
        "n_rollouts": n_rollouts,
        "n_true_pool": len(true_systems),
        "verify_gpu": bool(verify_gpu),
    }


def install_frozen_terminal_rule(
    exp_dir: Path,
    frozen_rule_path: Path,
    *,
    expected_alpha: float = 0.05,
    expected_margin: float | None = None,
    split_source: Path | None = None,
) -> dict[str, Any]:
    """Copy a certified rule into the experiment; verify α/margin; do not recalibrate."""
    import shutil

    from src.control.terminal_rule import FrozenTerminalRule, load_frozen_terminal_rule

    src = Path(frozen_rule_path)
    if not src.is_file():
        raise FileNotFoundError(f"frozen_rule_path missing: {src}")
    out_dir = Path(exp_dir) / "diagnostics" / "control_safety_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = json.loads(src.read_text(encoding="utf-8"))
    rule = raw.get("rule") if isinstance(raw.get("rule"), dict) else raw
    # Write both loader locations.
    payload = {"rule": rule, "source": str(src.resolve()), "mode": "frozen"}
    (out_dir / CALIBRATED_RULE_NAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (Path(exp_dir) / "selected_policy_robust_rule.json").write_text(
        json.dumps({"rule": rule, **{k: v for k, v in rule.items() if k != "rule"}}, indent=2),
        encoding="utf-8",
    )
    if split_source is not None and Path(split_source).is_file():
        shutil.copy2(split_source, out_dir / "split_metadata.json")
    frozen = load_frozen_terminal_rule(exp_dir, expected_margin=expected_margin)
    if abs(frozen.alpha - float(expected_alpha)) > 1e-12:
        raise RuntimeError(
            f"Frozen rule α={frozen.alpha} != expected {expected_alpha}"
        )
    if expected_margin is not None and abs(frozen.margin - float(expected_margin)) > 1e-12:
        raise RuntimeError(
            f"Frozen rule margin={frozen.margin} != expected {expected_margin}"
        )
    meta = {
        "mode": "frozen",
        "passed": True,
        "reused_from": str(src.resolve()),
        "terminal_rule": frozen.metadata(),
        "message": "Verified frozen terminal rule; did not recalibrate.",
    }
    (out_dir / "rule_reuse.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"  Frozen terminal rule: α={frozen.alpha} margin={frozen.margin} "
        f"hash={frozen.terminal_rule_hash} ← {src}"
    )
    return meta


def calibrate_control_safety(
    exp_dir: Path | str,
    *,
    project_root: Path | None = None,
    num_rollouts: int | None = None,
    seed: int | None = None,
    skip_diagnosis: bool = False,
) -> dict[str, Any]:
    """
    Choose (alpha, margin) on calibration systems; verify on validation.
    Never uses the final test split for selection.

    If ``control_safety_calibration.mode: frozen``, install/verify the frozen
    rule and return without changing α or margin.
    """
    from src.config import repo_root

    root = project_root or repo_root()
    run = load_experiment_run(Path(exp_dir), root)
    cal_cfg = SafetyCalibrationConfig.from_cfg(run.cfg)
    n_roll = int(num_rollouts if num_rollouts is not None else cal_cfg.num_rollouts)
    rng_seed = int(seed if seed is not None else cal_cfg.seed)
    out_dir = run.exp_dir / "diagnostics" / "control_safety_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(cal_cfg.mode).lower() == "frozen":
        if not cal_cfg.frozen_rule_path:
            raise RuntimeError(
                "control_safety_calibration.mode=frozen requires frozen_rule_path"
            )
        src = Path(cal_cfg.frozen_rule_path)
        if not src.is_absolute():
            src = root / src
        split_src = (
            src.parent / "diagnostics" / "control_safety_calibration" / "split_metadata.json"
            if (src.parent / "diagnostics" / "control_safety_calibration" / "split_metadata.json").is_file()
            else None
        )
        # Prefer split metadata from the same experiment tree when available.
        alt_split = (
            root
            / "experiments"
            / "ieee5_policy_robust_calibration_T2"
            / "diagnostics"
            / "control_safety_calibration"
            / "split_metadata.json"
        )
        if split_src is None and alt_split.is_file():
            split_src = alt_split
        horizon_split = (
            root
            / "experiments"
            / "ieee5_horizon_sweep"
            / "T2"
            / "diagnostics"
            / "control_safety_calibration"
            / "split_metadata.json"
        )
        if split_src is None and horizon_split.is_file():
            split_src = horizon_split
        return install_frozen_terminal_rule(
            run.exp_dir,
            src,
            expected_alpha=cal_cfg.expected_alpha,
            expected_margin=cal_cfg.expected_margin,
            split_source=split_src,
        )

    if not skip_diagnosis:
        try:
            diagnose_observability_unsafe_rollouts(
                run, tolerance=cal_cfg.under_control_tolerance
            )
        except FileNotFoundError as exc:
            print(f"  Skipping observability diagnosis ({exc})")

    control_spec = ControlSpec.from_cfg(run.cfg)
    catalog = build_catalog(run.cfg)
    n_actions = len(catalog)
    horizon = int(run.cfg.step_number)
    rng = np.random.default_rng(rng_seed)

    splits = make_train_splits(len(run.train_systems), cal_cfg, rng)
    (out_dir / "split_metadata.json").write_text(
        json.dumps(
            {
                "seed": rng_seed,
                "n_train": len(run.train_systems),
                "n_test_untouched": len(run.test_systems),
                **splits,
                "fractions": {
                    "support": cal_cfg.support_fraction,
                    "calibration": cal_cfg.calibration_fraction,
                    "validation": cal_cfg.validation_fraction,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    support_systems = [run.train_systems[i] for i in splits["support_ids"]]
    cal_systems = [run.train_systems[i] for i in splits["calibration_ids"]]
    val_systems = [run.train_systems[i] for i in splits["validation_ids"]]
    table_support = TableThetaSupport(
        systems=support_systems,
        log_p0=log_prior_uniform_discrete(len(support_systems)),
    )
    U_support = extract_U_bank(support_systems)

    sim = build_simulator(run.cfg)
    sim.T_obs_sec = float(control_spec.T_obs_sec)
    sim.ode_dt = float(control_spec.ode_dt)
    sim.fs_hz = float(control_spec.fs_hz)
    engine = CudaControlEngine(sim, control_spec)

    # Split rollouts: modest budget for grid search (CPU proxy), larger for GPU validation.
    n_cal_roll = max(80, min(200, int(round(0.4 * n_roll))))
    n_val_roll = max(100, min(800, int(round(0.6 * n_roll))))

    print(
        f"  Control-safety calibration: T={horizon}  seed={rng_seed}  "
        f"support={len(support_systems)}  cal={len(cal_systems)}  val={len(val_systems)}  "
        f"rollouts cal/val={n_cal_roll}/{n_val_roll}"
    )
    print(f"  Searching alpha={list(cal_cfg.alpha_grid)}  margin={list(cal_cfg.margin_grid)}")

    grid_rows: list[dict[str, Any]] = []
    for alpha in cal_cfg.alpha_grid:
        for margin in cal_cfg.margin_grid:
            rule = TerminalControlRule(
                alpha=float(alpha),
                margin=float(margin),
                u_candidates=control_spec.u_candidates,
            )
            # Independent RNG per candidate for reproducibility of relative ranking.
            sub_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
            cal_stats = _evaluate_rule_on_pool(
                rule=rule,
                true_systems=cal_systems,
                true_ids=splits["calibration_ids"],
                table_support=table_support,
                U_support=U_support,
                horizon=horizon,
                n_actions=n_actions,
                sigma_y=float(run.cfg.sigma_y),
                control_engine=None,
                n_rollouts=n_cal_roll,
                rng=sub_rng,
                verify_gpu=False,
                under_tol=cal_cfg.under_control_tolerance,
            )
            row = {
                "alpha": float(alpha),
                "margin": float(margin),
                "quantile_level": 1.0 - float(alpha),
                **{f"cal_{k}": v for k, v in cal_stats.items()},
                "cal_passes": bool(
                    cal_stats["safety_rate"] >= cal_cfg.min_calibration_safety_rate - 1e-12
                ),
            }
            grid_rows.append(row)
            print(
                f"    α={alpha:.4f} m={margin:.2f}  "
                f"cal_safety={cal_stats['safety_rate']:.3f}  "
                f"cal_mean_u={cal_stats['mean_u_ctrl']:.4f}  "
                f"cal_excess={cal_stats['mean_excess']:.4f}"
            )

    _write_csv(
        out_dir / "calibration_grid.csv",
        grid_rows,
        [
            "alpha",
            "margin",
            "quantile_level",
            "cal_safety_rate",
            "cal_mean_excess",
            "cal_mean_u_ctrl",
            "cal_mean_under_control",
            "cal_std_u_ctrl",
            "cal_n_rollouts",
            "cal_passes",
        ],
    )

    passing = [r for r in grid_rows if r["cal_passes"]]
    if not passing:
        report = {
            "passed": False,
            "reason": "no (alpha, margin) achieved calibration safety rate",
            "grid": grid_rows,
            "splits": splits,
        }
        (out_dir / "calibration_summary.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print("  FAIL: no calibrated rule passed calibration safety.")
        return report

    # Minimize conservatism among cal-safe rules; escalate if validation fails.
    passing.sort(
        key=lambda r: (r["cal_mean_u_ctrl"], r["cal_mean_excess"], r["margin"], -r["alpha"])
    )
    chosen = None
    val_stats = None
    attempts: list[dict[str, Any]] = []
    for cand in passing:
        rule = TerminalControlRule(
            alpha=float(cand["alpha"]),
            margin=float(cand["margin"]),
            u_candidates=control_spec.u_candidates,
        )
        val_rng = np.random.default_rng(rng_seed + 99 + int(1000 * cand["margin"]))
        stats = _evaluate_rule_on_pool(
            rule=rule,
            true_systems=val_systems,
            true_ids=splits["validation_ids"],
            table_support=table_support,
            U_support=U_support,
            horizon=horizon,
            n_actions=n_actions,
            sigma_y=float(run.cfg.sigma_y),
            control_engine=engine,
            n_rollouts=n_val_roll,
            rng=val_rng,
            verify_gpu=True,
            under_tol=cal_cfg.under_control_tolerance,
        )
        attempts.append(
            {
                "alpha": cand["alpha"],
                "margin": cand["margin"],
                "cal_mean_u_ctrl": cand["cal_mean_u_ctrl"],
                **{f"val_{k}": v for k, v in stats.items()},
            }
        )
        print(
            f"  Validate α={cand['alpha']:.4f} m={cand['margin']:.2f}: "
            f"val_safety={stats['safety_rate']:.3f}  mean_u={stats['mean_u_ctrl']:.4f}"
        )
        if stats["safety_rate"] >= cal_cfg.min_validation_safety_rate - 1e-12:
            chosen = cand
            val_stats = stats
            best_rule = rule
            break

    if chosen is None or val_stats is None:
        report = {
            "passed": False,
            "reason": "no cal-safe rule passed GPU validation safety rate",
            "validation_attempts": attempts,
            "grid": grid_rows,
            "splits": splits,
        }
        (out_dir / "calibration_summary.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print("  FAIL: validation safety < 1.0 for all cal-safe rules tried.")
        return report

    best = chosen
    best_rule = TerminalControlRule(
        alpha=float(best["alpha"]),
        margin=float(best["margin"]),
        u_candidates=control_spec.u_candidates,
    )
    val_ok = True

    payload = {
        "rule": best_rule.to_dict(),
        "calibration": {k: best[k] for k in best if k.startswith("cal_")},
        "validation": val_stats,
        "validation_passed": val_ok,
        "validation_attempts": attempts,
        "splits": splits,
        "seed": rng_seed,
        "num_rollouts_requested": n_roll,
        "baseline_alpha": float(control_spec.alpha),
        "baseline_margin": float(control_spec.safety_margin),
        "n_passing_cal_rules": len(passing),
        "selection_criterion": (
            "min cal_mean_u_ctrl among cal_safety>=1, "
            "escalate until val GPU safety>=1"
        ),
    }
    (out_dir / CALIBRATED_RULE_NAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (out_dir / "calibration_summary.json").write_text(
        json.dumps({**payload, "passed": val_ok, "grid_n": len(grid_rows)}, indent=2),
        encoding="utf-8",
    )

    # Persist into run_config.yaml so train/eval load the same rule.
    cfg_path = run.exp_dir / "run_config.yaml"
    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    ctrl = dict(doc.get("control") or {})
    ctrl["alpha"] = float(best_rule.alpha)
    ctrl["safety_margin"] = float(best_rule.margin)
    doc["control"] = ctrl
    doc["control_safety_calibration"] = {
        **dict(getattr(run.cfg, "raw", {}).get("control_safety_calibration") or {}),
        "selected_alpha": float(best_rule.alpha),
        "selected_margin": float(best_rule.margin),
        "validation_safety_rate": float(val_stats["safety_rate"]),
        "calibrated": True,
    }
    cfg_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    print(
        f"  Selected rule: α={best_rule.alpha:.4f}  margin={best_rule.margin:.3f}  "
        f"cal_safety={best['cal_safety_rate']:.3f}  cal_mean_u={best['cal_mean_u_ctrl']:.4f}"
    )
    print(
        f"  Validation: safety={val_stats['safety_rate']:.3f}  "
        f"mean_u={val_stats['mean_u_ctrl']:.4f}  excess={val_stats['mean_excess']:.4f}  "
        f"{'PASS' if val_ok else 'FAIL'}"
    )
    payload["passed"] = val_ok
    return payload
