"""IEEE5 T=2 controlled four-method pilot (no full sweep)."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.control.banks import extract_U_bank
from src.control.cuda_control import CudaControlEngine
from src.control.eval_metrics import aggregate_control_metrics
from src.control.fixed_search import search_fixed_subset, save_fixed_search
from src.control.myopic import MyopicControlSelector
from src.control.posterior_ctrl import normalize_log_weights, posterior_ess
from src.control.terminal_rule import (
    FrozenTerminalRule,
    assert_shared_rule_metadata,
    load_frozen_terminal_rule,
    observe_with_keyed_noise,
    posterior_to_u_ctrl,
)
from src.control.u_req import ControlSpec
from src.contrastive.spce import log_prior_uniform_discrete
from src.rollout import FixedSelector, RandomSelector, update_log_weights
from src.run_context import ExperimentRun, load_experiment_run
from src.swing_equation_ode.design import build_catalog, build_simulator
from src.swing_equation_ode.simulator import system_mk
from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def load_pilot_splits(exp_dir: Path, run: ExperimentRun) -> dict[str, Any]:
    """Reuse calibration split metadata when present; never use test for selection."""
    split_path = (
        exp_dir / "diagnostics" / "control_safety_calibration" / "split_metadata.json"
    )
    if split_path.is_file():
        meta = json.loads(split_path.read_text(encoding="utf-8"))
        support_ids = list(meta["support_ids"])
        cal_ids = list(meta["calibration_ids"])
        val_ids = list(meta["validation_ids"])
    else:
        # Fallback deterministic split from train only.
        rng = np.random.default_rng(2468)
        n = len(run.train_systems)
        perm = rng.permutation(n).tolist()
        n_sup = max(2, int(0.625 * n))
        n_cal = max(2, int(0.1875 * n))
        support_ids = sorted(perm[:n_sup])
        cal_ids = sorted(perm[n_sup : n_sup + n_cal])
        val_ids = sorted(perm[n_sup + n_cal :])
        meta = {
            "support_ids": support_ids,
            "calibration_ids": cal_ids,
            "validation_ids": val_ids,
            "fallback": True,
        }
    support = [run.train_systems[i] for i in support_ids]
    cal = [run.train_systems[i] for i in cal_ids]
    val = [run.train_systems[i] for i in val_ids]
    # Fixed search / DAD train: support∪cal (no val for gradients? User said val for checkpoint).
    # Train on support+cal, validate on val, test = final test.
    train_for_dad = support + cal
    return {
        "meta": meta,
        "support_systems": support,
        "calibration_systems": cal,
        "validation_systems": val,
        "train_systems": train_for_dad,
        "test_systems": list(run.test_systems),
        "support_ids": support_ids,
        "calibration_ids": cal_ids,
        "validation_ids": val_ids,
        "test_ids": list(range(len(run.test_systems))),
    }


def rich_metrics(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    u = np.asarray([float(r["u_ctrl"]) for r in rollouts], dtype=np.float64)
    excess = np.asarray([float(r.get("excess_control", np.nan)) for r in rollouts], dtype=np.float64)
    ureq = np.asarray([float(r.get("u_req_true", np.nan)) for r in rollouts], dtype=np.float64)
    safe = np.asarray([1.0 if r.get("safe_total") else 0.0 for r in rollouts], dtype=np.float64)
    runtimes = np.asarray([float(r.get("runtime_s", np.nan)) for r in rollouts], dtype=np.float64)
    u_max = float(np.nanmax(u)) if u.size else float("nan")
    # Prefer configured grid max if present
    return {
        "n": int(u.size),
        "mean_u_ctrl": float(np.mean(u)) if u.size else float("nan"),
        "std_u_ctrl": float(np.std(u)) if u.size else float("nan"),
        "median_u_ctrl": float(np.median(u)) if u.size else float("nan"),
        "q10_u_ctrl": float(np.quantile(u, 0.10)) if u.size else float("nan"),
        "q90_u_ctrl": float(np.quantile(u, 0.90)) if u.size else float("nan"),
        "unique_u_ctrl_count": int(len({float(x) for x in u.tolist()})) if u.size else 0,
        "fraction_at_u_max": float(np.mean(np.isclose(u, np.max(u)))) if u.size else float("nan"),
        "true_safety_rate": float(np.mean(safe)) if safe.size else float("nan"),
        "mean_true_u_req": float(np.nanmean(ureq)) if ureq.size else float("nan"),
        "mean_excess_control": float(np.nanmean(excess)) if excess.size else float("nan"),
        "median_excess_control": float(np.nanmedian(excess)) if excess.size else float("nan"),
        "over_control_rate": float(np.mean(excess > 1e-12)) if excess.size else float("nan"),
        "under_control_rate": float(np.mean(excess < -1e-12)) if excess.size else float("nan"),
        "mean_runtime_per_rollout": float(np.nanmean(runtimes)) if runtimes.size else float("nan"),
    }


def paired_diff_stats(a: np.ndarray, b: np.ndarray, *, n_boot: int = 10000, seed: int = 0) -> dict[str, Any]:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if d.size == 0:
        return {}
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, d.size, size=d.size)
        boots.append(float(np.mean(d[idx])))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "mean_paired_diff": float(np.mean(d)),
        "median_paired_diff": float(np.median(d)),
        "std_error": float(np.std(d, ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "fraction_a_lower": float(np.mean(d < -1e-12)),
        "fraction_b_lower": float(np.mean(d > 1e-12)),
        "fraction_tied": float(np.mean(np.abs(d) <= 1e-12)),
        "n": int(d.size),
        "n_bootstrap": int(n_boot),
    }


def _run_one_rollout(
    *,
    system: dict[str, Any],
    theta_id: int,
    rollout_id: int,
    selector_factory: Callable,
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    frozen: FrozenTerminalRule,
    control_spec: ControlSpec,
    control_engine: CudaControlEngine,
    horizon: int,
    n_actions: int,
    sigma_y: float,
    global_seed: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    log_w = np.asarray(table_support.log_p0, dtype=np.float64).copy()
    w0 = normalize_log_weights(log_w)
    u_prior = float(posterior_to_u_ctrl(w0, U_support, frozen))
    used: set[int] = set()
    seq: list[int] = []
    y_list: list[float] = []
    selector = selector_factory()
    tie_exact = 0
    tie_near = 0
    tie_break = 0
    co_tied_counts: list[int] = []
    score_gaps: list[float] = []
    decision_runtimes: list[float] = []
    for step in range(horizon):
        weights = normalize_log_weights(log_w)
        t_dec = time.perf_counter()
        a = int(
            selector.select(
                step=step,
                history_actions=list(seq),
                history_obs=list(y_list),
                used=set(used),
                log_weights=log_w,
                weights=weights,
                rng=rng,
            )
        )
        decision_runtimes.append(float(time.perf_counter() - t_dec))
        if hasattr(selector, "last_tie_count"):
            tie_exact += int(getattr(selector, "last_tie_count", 0) or 0)
            # co-tied = selected + other exact ties
            co_tied_counts.append(1 + int(getattr(selector, "last_tie_count", 0) or 0))
        if hasattr(selector, "last_near_tie_count"):
            tie_near += int(getattr(selector, "last_near_tie_count", 0) or 0)
        if hasattr(selector, "last_tie_break_used") and getattr(
            selector, "last_tie_break_used", False
        ):
            tie_break += 1
        if hasattr(selector, "last_score_gap"):
            g = getattr(selector, "last_score_gap", float("nan"))
            if g is not None and np.isfinite(g):
                score_gaps.append(float(g))
        if a in used:
            raise RuntimeError(f"repeat action {a}")
        y = observe_with_keyed_noise(
            system,
            a,
            sigma_y=sigma_y,
            global_seed=global_seed,
            theta_id=theta_id,
            rollout_id=rollout_id,
            step=step,
        )
        centres = y_sim_last_step_from_tables(table_support, [a])
        log_w = update_log_weights(log_w, y, centres, sigma_y)
        seq.append(a)
        y_list.append(y)
        used.add(a)
    weights = normalize_log_weights(log_w)
    u_ctrl = float(posterior_to_u_ctrl(weights, U_support, frozen))
    u_req = float(system["u_req"])
    excess = u_ctrl - u_req
    M, K = system_mk(system, control_engine.N)
    metrics = control_engine.evaluate_one(M, K, u_ctrl)
    mean_U = float(np.sum(weights * U_support))
    std_U = float(np.sqrt(max(np.sum(weights * (U_support - mean_U) ** 2), 0.0)))
    runtime = float(time.perf_counter() - t0)
    return {
        "theta_test_id": theta_id,
        "rollout_id": rollout_id,
        "sequence": seq,
        "y_obs": y_list,
        "u_ctrl": u_ctrl,
        "u_req_true": u_req,
        "excess_control": excess,
        "u_ctrl_prior": u_prior,
        "changed_from_prior": abs(u_ctrl - u_prior) > 1e-12,
        "reduced_from_prior": u_ctrl < u_prior - 1e-12,
        "increased_from_prior": u_ctrl > u_prior + 1e-12,
        "safe_total": bool(metrics["safe_total"] >= 0.5),
        "max_rocof": metrics["rocof_max"],
        "frequency_nadir": metrics["delta_f_nadir"],
        "rocof_safe": bool(metrics["rocof_safe"] >= 0.5),
        "nadir_safe": bool(metrics["nadir_safe"] >= 0.5),
        "posterior_ess": posterior_ess(weights),
        "max_posterior_weight": float(np.max(weights)),
        "posterior_mean_U": mean_U,
        "posterior_std_U": std_U,
        "posterior_mean_U_error": mean_U - u_req,
        "runtime_s": runtime,
        "exact_tie_count": int(tie_exact),
        "near_tie_count": int(tie_near),
        "action_index_tie_break_count": int(tie_break),
        "median_co_tied_action_count": (
            float(np.median(co_tied_counts)) if co_tied_counts else float("nan")
        ),
        "mean_score_gap": float(np.mean(score_gaps)) if score_gaps else float("nan"),
        "mean_decision_runtime_s": (
            float(np.mean(decision_runtimes)) if decision_runtimes else float("nan")
        ),
        **{f"rule_{k}": v for k, v in frozen.metadata().items() if k != "u_candidates"},
    }


def evaluate_method_paired(
    *,
    method: str,
    selector_factory: Callable,
    test_systems: list[dict[str, Any]],
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    frozen: FrozenTerminalRule,
    control_spec: ControlSpec,
    control_engine: CudaControlEngine,
    horizon: int,
    n_actions: int,
    sigma_y: float,
    n_rollouts: int,
    global_seed: int,
    method_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(method_seed)
    rows: list[dict[str, Any]] = []
    for rid in range(n_rollouts):
        tid = int(rid % len(test_systems))
        system = test_systems[tid]
        row = _run_one_rollout(
            system=system,
            theta_id=tid,
            rollout_id=rid,
            selector_factory=selector_factory,
            table_support=table_support,
            U_support=U_support,
            frozen=frozen,
            control_spec=control_spec,
            control_engine=control_engine,
            horizon=horizon,
            n_actions=n_actions,
            sigma_y=sigma_y,
            global_seed=global_seed,
            rng=rng,
        )
        row["method"] = method
        rows.append(row)
    summary = rich_metrics(rows)
    summary["method"] = method
    summary["terminal_rule"] = frozen.metadata()
    if method == "myopic" and rows:
        summary["tie_diagnostics"] = {
            "exact_tie_rate": float(np.mean([r.get("exact_tie_count", 0) > 0 for r in rows])),
            "near_tie_rate": float(np.mean([r.get("near_tie_count", 0) > 0 for r in rows])),
            "mean_exact_ties_per_rollout": float(
                np.mean([r.get("exact_tie_count", 0) for r in rows])
            ),
            "mean_near_ties_per_rollout": float(
                np.mean([r.get("near_tie_count", 0) for r in rows])
            ),
            "median_co_tied_action_count": float(
                np.nanmedian([r.get("median_co_tied_action_count", np.nan) for r in rows])
            ),
            "mean_score_gap": float(
                np.nanmean([r.get("mean_score_gap", np.nan) for r in rows])
            ),
            "action_index_tie_break_count": int(
                sum(int(r.get("action_index_tie_break_count", 0)) for r in rows)
            ),
            "mean_runtime_per_decision": float(
                np.mean([r.get("mean_decision_runtime_s", np.nan) for r in rows])
            ),
        }
    return rows, summary


def action_frequencies(rows: list[dict[str, Any]], catalog) -> dict[str, Any]:
    acts = []
    amps = []
    buses = []
    pairs = []
    repeats = 0
    for r in rows:
        seq = list(r["sequence"])
        if len(seq) != len(set(seq)):
            repeats += 1
        acts.extend(seq)
        for a in seq:
            d = catalog[int(a)]
            amps.append(float(d.amplitude))
            buses.append(int(d.bus))
        if len(seq) >= 2:
            pairs.append(tuple(sorted(seq[:2])))
    return {
        "action_frequency": dict(Counter(acts)),
        "amplitude_frequency": {str(k): v for k, v in Counter(amps).items()},
        "bus_frequency": dict(Counter(buses)),
        "action_pair_frequency": {f"{a},{b}": c for (a, b), c in Counter(pairs).items()},
        "repeat_action_count": int(repeats),
    }


def dad_adaptation_table(rows: list[dict[str, Any]], n_bins: int = 5) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, int], Counter] = defaultdict(Counter)
    y_vals = [float(r["y_obs"][0]) for r in rows if len(r.get("y_obs", [])) >= 1]
    if not y_vals:
        return []
    edges = np.quantile(y_vals, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    for r in rows:
        seq = r["sequence"]
        y = r["y_obs"]
        if len(seq) < 2 or len(y) < 1:
            continue
        b = int(np.searchsorted(edges, float(y[0]), side="right") - 1)
        b = min(max(b, 0), n_bins - 1)
        buckets[(int(seq[0]), b)][int(seq[1])] += 1
    out = []
    for (a0, b), ctr in sorted(buckets.items()):
        total = sum(ctr.values())
        most = ctr.most_common(1)[0][0]
        out.append(
            {
                "first_action": a0,
                "first_observation_bin": b,
                "most_common_second_action": most,
                "second_action_distribution": {str(k): v / total for k, v in ctr.items()},
                "n": total,
            }
        )
    return out


def _make_plots(out_dir: Path, summaries: dict[str, dict], train_curves: dict | None, dad_rows: list) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    methods = list(summaries.keys())

    def bar(metric: str, fname: str, ylabel: str):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        vals = [summaries[m].get(metric, np.nan) for m in methods]
        ax.bar(methods, vals, color="#2c5f7c")
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        fig.tight_layout()
        fig.savefig(plots / fname, dpi=120)
        plt.close(fig)

    bar("mean_u_ctrl", "mean_u_ctrl_comparison.png", "mean u_ctrl")
    bar("true_safety_rate", "safety_rate_comparison.png", "safety rate")
    bar("mean_runtime_per_rollout", "runtime_comparison.png", "seconds / rollout")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for m in methods:
        # approximate from summary quantiles only — skip if no raw
        pass
    plt.close(fig)

    if train_curves:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for seed, curve in train_curves.items():
            ax.plot(curve.get("epoch_val_u_ctrl", []), label=f"seed {seed}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation mean u_ctrl")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "dad_training_curve.png", dpi=120)
        plt.close(fig)


def run_pilot(
    exp_dir: Path | str,
    *,
    project_root: Path | None = None,
    debug_one_seed: bool = False,
    n_eval_rollouts: int | None = None,
) -> dict[str, Any]:
    from src.config import repo_root
    from src.neural.train import train_dad_policy as train_core

    root = project_root or repo_root()
    run = load_experiment_run(Path(exp_dir), root)
    exp_dir = run.exp_dir
    frozen = load_frozen_terminal_rule(exp_dir)
    print(f"  Frozen terminal rule hash={frozen.terminal_rule_hash}  α={frozen.alpha}  margin={frozen.margin}")

    base_spec = ControlSpec.from_cfg(run.cfg)
    control_spec = frozen.to_control_spec(base_spec)
    # Production Myopic sample count: prefer frozen myopic.n_hypothetical.
    from dataclasses import replace

    myopic_block = dict(run.cfg.raw.get("myopic") or {})
    if myopic_block.get("n_hypothetical") is not None:
        control_spec = replace(
            control_spec, myopic_hypothetical=int(myopic_block["n_hypothetical"])
        )
    run.cfg.raw.setdefault("control", {})
    run.cfg.raw["control"]["alpha"] = frozen.alpha
    run.cfg.raw["control"]["safety_margin"] = frozen.margin
    run.cfg.raw["control"]["u_candidates"] = list(frozen.u_candidates)
    run.cfg.raw["control"]["myopic_hypothetical"] = int(control_spec.myopic_hypothetical)
    print(
        f"  Myopic n_hypothetical={control_spec.myopic_hypothetical} "
        f"(source={myopic_block.get('selection_source', 'control.myopic_hypothetical')})"
    )

    pilot_cfg = dict(run.cfg.raw.get("pilot") or {})
    epochs = int(pilot_cfg.get("epochs", run.cfg.raw.get("training", {}).get("epochs", 50)))
    run.cfg.raw.setdefault("training", {})["epochs"] = epochs
    seeds = list(pilot_cfg.get("training_seeds", [101, 202, 303]))
    if debug_one_seed:
        seeds = seeds[:1]
        run.cfg.raw["training"]["epochs"] = min(epochs, 10)
    eval_rollouts = int(n_eval_rollouts or pilot_cfg.get("evaluation_rollouts", 1000))
    global_seed = int(pilot_cfg.get("global_seed", 1234))
    method_seeds = list(pilot_cfg.get("method_seeds", [101, 202, 303]))

    splits = load_pilot_splits(exp_dir, run)
    catalog = build_catalog(run.cfg)
    n_actions = len(catalog)
    horizon = int(run.cfg.step_number)
    table_support = TableThetaSupport(
        systems=splits["support_systems"],
        log_p0=log_prior_uniform_discrete(len(splits["support_systems"])),
    )
    U_support = extract_U_bank(splits["support_systems"])

    sim = build_simulator(run.cfg)
    sim.T_obs_sec = control_spec.T_obs_sec
    sim.ode_dt = control_spec.ode_dt
    sim.fs_hz = control_spec.fs_hz
    engine = CudaControlEngine(sim, control_spec)

    train_root = exp_dir / "train" / "dad"
    train_root.mkdir(parents=True, exist_ok=True)
    train_curves: dict[str, Any] = {}
    policy_paths: dict[int, Path] = {}

    print(f"\n=== Pilot DAD training seeds={seeds} epochs={run.cfg.raw['training']['epochs']} ===")
    for seed in seeds:
        seed_dir = train_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        run.cfg.raw["data_generation"] = dict(run.cfg.raw.get("data_generation") or {})
        run.cfg.raw["data_generation"]["train_seed"] = int(seed)
        meta = {
            **run.policy_meta,
            "n_actions": n_actions,
            "step_number": horizon,
            "sigma_y": run.cfg.sigma_y,
            "experiment_dir": str(exp_dir.resolve()),
            "method": "dad",
            "terminal_rule": frozen.metadata(),
            "train_seed": int(seed),
        }
        path = train_core(
            run.cfg,
            splits["train_systems"],
            meta,
            seed_dir,
            data_dir=run.data_path,
            run_tag=f"dad_seed{seed}",
            validation_systems=splits["validation_systems"],
            support_systems=splits["support_systems"],
        )
        policy_paths[int(seed)] = path
        metrics_path = seed_dir / "dad_training_metrics.json"
        if metrics_path.is_file():
            train_curves[str(seed)] = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Fixed subset on train/validation only (no test).
    print("\n=== Fixed subset search (train/val only) ===")
    fixed_rng = np.random.default_rng(int(pilot_cfg.get("fixed_seed", 7)))
    fixed_cal = splits["train_systems"][: min(32, len(splits["train_systems"]))]
    t_fixed0 = time.perf_counter()
    fixed_result = search_fixed_subset(
        n_actions=n_actions,
        horizon=horizon,
        table_support=table_support,
        U_support=U_support,
        calibration_systems=fixed_cal,
        sigma_y=float(run.cfg.sigma_y),
        alpha=frozen.alpha,
        rng=fixed_rng,
        exhaustive_threshold=int(control_spec.fixed_exhaustive_threshold),
        noise_replicas=int(control_spec.fixed_noise_replicas),
        greedy_restarts=int(control_spec.fixed_greedy_restarts),
        seed=int(pilot_cfg.get("fixed_seed", 7)),
        margin=frozen.margin,
        u_grid=frozen.u_candidates,
    )
    fixed_runtime = float(time.perf_counter() - t_fixed0)
    fixed_subset = list(sorted(fixed_result.subset))
    eval_root = exp_dir / "eval"
    save_fixed_search(fixed_result, eval_root / "fixed" / "fixed_subset_search.json")
    fixed_meta = {
        "selected_action_ids": fixed_subset,
        "selected_amplitudes": [float(catalog[a].amplitude) for a in fixed_subset],
        "selected_buses": [int(catalog[a].bus) for a in fixed_subset],
        "estimated_mean_u_ctrl": float(fixed_result.objective),
        "number_of_subsets_evaluated": int(fixed_result.n_candidates_evaluated),
        "search_runtime": fixed_runtime,
        "search_seed": int(pilot_cfg.get("fixed_seed", 7)),
        "search_mode": fixed_result.search_mode,
        "used_test_systems": False,
        "terminal_rule": frozen.metadata(),
    }
    _write_json(eval_root / "fixed" / "subset_meta.json", fixed_meta)
    print(f"  Fixed subset={fixed_subset}  mode={fixed_result.search_mode}  obj={fixed_result.objective:.4f}")

    method_metas: dict[str, dict[str, Any]] = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}

    def eval_and_store(name: str, factory, seed: int):
        rows, summary = evaluate_method_paired(
            method=name,
            selector_factory=factory,
            test_systems=splits["test_systems"],
            table_support=table_support,
            U_support=U_support,
            frozen=frozen,
            control_spec=control_spec,
            control_engine=engine,
            horizon=horizon,
            n_actions=n_actions,
            sigma_y=float(run.cfg.sigma_y),
            n_rollouts=eval_rollouts,
            global_seed=global_seed,
            method_seed=seed,
        )
        freq = action_frequencies(rows, catalog)
        summary["probe_behavior"] = freq
        summary["decision_behavior"] = {
            "fraction_changed_from_prior": float(np.mean([r["changed_from_prior"] for r in rows])),
            "fraction_reduced_from_prior": float(np.mean([r["reduced_from_prior"] for r in rows])),
            "fraction_increased_from_prior": float(np.mean([r["increased_from_prior"] for r in rows])),
            "mean_prior_u_ctrl": float(np.mean([r["u_ctrl_prior"] for r in rows])),
        }
        summary["posterior_behavior"] = {
            "mean_ess": float(np.mean([r["posterior_ess"] for r in rows])),
            "mean_max_weight": float(np.mean([r["max_posterior_weight"] for r in rows])),
            "mean_posterior_mean_U": float(np.mean([r["posterior_mean_U"] for r in rows])),
            "mean_posterior_mean_U_error": float(np.mean([r["posterior_mean_U_error"] for r in rows])),
        }
        out = eval_root / name
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "summary.json", summary)
        _write_csv(
            out / "rollouts.csv",
            [
                {
                    **{k: r[k] for k in (
                        "theta_test_id", "rollout_id", "u_ctrl", "u_req_true",
                        "excess_control", "safe_total", "runtime_s",
                        "posterior_ess", "u_ctrl_prior",
                    )},
                    "sequence": " ".join(map(str, r["sequence"])),
                    "y_obs": " ".join(f"{y:.8g}" for y in r["y_obs"]),
                }
                for r in rows
            ],
            [
                "theta_test_id", "rollout_id", "sequence", "y_obs", "u_ctrl", "u_req_true",
                "excess_control", "safe_total", "runtime_s", "posterior_ess", "u_ctrl_prior",
            ],
        )
        _write_json(out / "probe_behavior.json", freq)
        meta = {
            **frozen.metadata(),
            "method": name,
            "test_theta_ids": splits["test_ids"],
            "T": horizon,
            "n_rollouts": eval_rollouts,
            "method_seed": seed,
            "sigma_y": float(run.cfg.sigma_y),
        }
        method_metas[name] = meta
        all_rows[name] = rows
        summaries[name] = summary
        print(
            f"  [{name}] mean_u={summary['mean_u_ctrl']:.4f}  "
            f"safety={summary['true_safety_rate']:.3f}  "
            f"excess={summary['mean_excess_control']:.4f}"
        )
        return rows, summary

    print(f"\n=== Paired evaluation n_rollouts={eval_rollouts} ===")
    # Use first method seed for myopic/fixed/random; DAD aggregated across seeds below.
    ms0 = int(method_seeds[0])
    eval_and_store("myopic", lambda: MyopicControlSelector(
        table_support=table_support,
        U_support=U_support,
        n_actions=n_actions,
        sigma_y=float(run.cfg.sigma_y),
        alpha=frozen.alpha,
        n_hypothetical=int(control_spec.myopic_hypothetical),
        safety_margin=frozen.margin,
        u_candidates=frozen.u_candidates,
    ), ms0)
    eval_and_store(
        "fixed",
        lambda: FixedSelector(sequence=list(fixed_subset)),
        ms0,
    )
    eval_and_store("random", lambda: RandomSelector(n_actions=n_actions), ms0)

    # DAD: evaluate each seed; primary "dad" = best validation checkpoint seed.
    dad_seed_summaries = {}
    dad_rows_by_seed: dict[str, list] = {}
    best_val_seed = None
    best_val_score = float("inf")
    for i, seed in enumerate(seeds):
        policy_path = policy_paths[int(seed)]
        meta = {
            "n_actions": n_actions,
            "step_number": horizon,
            "sigma_y": run.cfg.sigma_y,
            "experiment_dir": str(exp_dir.resolve()),
        }
        metrics_path = train_root / f"seed_{seed}" / "dad_training_metrics.json"
        if metrics_path.is_file():
            tm = json.loads(metrics_path.read_text(encoding="utf-8"))
            bv = tm.get("best_val_u_ctrl")
            vs = tm.get("best_val_safety")
            # Safety-first: reject seeds with validation safety < 1.0.
            if vs is not None and abs(float(vs) - 1.0) > 1e-12:
                continue
            if bv is not None and float(bv) < best_val_score:
                best_val_score = float(bv)
                best_val_seed = int(seed)

        def factory(pp=policy_path, m=meta):
            from src.neural.policy import DADPolicy
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(pp, map_location=device, weights_only=False)
            pol = DADPolicy(n_actions, max_steps=horizon).to(device)
            state = ckpt.get("state_dict") or ckpt.get("policy")
            pol.load_state_dict(state)
            pol.eval()

            class _DadSel:
                def __init__(self):
                    self.act_h: list[int] = []
                    self.obs_h: list[float] = []

                def select(self, *, used, rng, **_k):
                    import torch

                    if not self.act_h:
                        act_t = torch.zeros(1, 0, dtype=torch.long, device=device)
                        obs_t = torch.zeros(1, 0, device=device)
                        mask_t = torch.zeros(1, 0, device=device)
                    else:
                        act_t = torch.tensor([self.act_h], dtype=torch.long, device=device)
                        obs_t = torch.tensor([self.obs_h], dtype=torch.float32, device=device)
                        mask_t = torch.ones(1, len(self.act_h), device=device)
                    feas = torch.ones(1, n_actions, dtype=torch.bool, device=device)
                    for u in used:
                        feas[0, int(u)] = False
                    with torch.no_grad():
                        a, _, _ = pol.select_action(act_t, obs_t, mask_t, feas, deterministic=True)
                    return int(a.item())

            sel = _DadSel()

            class _Adapter:
                def select(self, *, step, history_actions, history_obs, used, rng, **_k):
                    sel.act_h = list(history_actions)
                    sel.obs_h = list(history_obs)
                    return sel.select(used=used, rng=rng)

            return _Adapter()

        rows, summary = eval_and_store(
            f"dad_seed_{seed}",
            factory,
            int(method_seeds[min(i, len(method_seeds) - 1)]),
        )
        dad_seed_summaries[str(seed)] = summary
        dad_rows_by_seed[str(seed)] = rows

    if best_val_seed is None:
        best_val_seed = int(seeds[0])
    # Prespecified primary: best validation mean u_ctrl (not test).
    # Record per-seed safeties for diagnostics; do not re-rank by test safety.
    primary_key = f"dad_seed_{best_val_seed}"
    all_rows["dad"] = dad_rows_by_seed[str(best_val_seed)]
    summaries["dad"] = {
        **dad_seed_summaries[str(best_val_seed)],
        "method": "dad",
        "seed": best_val_seed,
        "selection": "safety_first_then_best_validation_mean_u_ctrl",
        "best_val_u_ctrl": best_val_score,
    }
    method_metas["dad"] = {
        **method_metas[primary_key],
        "method": "dad",
        "selected_seed": best_val_seed,
    }
    dst = eval_root / "dad"
    dst.mkdir(parents=True, exist_ok=True)
    src = eval_root / primary_key
    for p in src.glob("*"):
        (dst / p.name).write_bytes(p.read_bytes())
    print(
        f"  Primary DAD seed={best_val_seed} "
        f"(best val u_ctrl={best_val_score:.4f})"
    )
    shared = assert_shared_rule_metadata(method_metas)
    _write_json(eval_root / "method_metadata.json", {"shared": shared, "methods": method_metas})

    # Paired differences on aligned rollout_ids (10k bootstrap)
    pairs = {}
    for a, b in [
        ("dad", "myopic"),
        ("dad", "fixed"),
        ("dad", "random"),
        ("myopic", "fixed"),
        ("myopic", "random"),
        ("fixed", "random"),
    ]:
        if a in all_rows and b in all_rows:
            ua = np.asarray([r["u_ctrl"] for r in all_rows[a]], dtype=np.float64)
            ub = np.asarray([r["u_ctrl"] for r in all_rows[b]], dtype=np.float64)
            n = min(ua.size, ub.size)
            pairs[f"{a}_minus_{b}"] = paired_diff_stats(
                ua[:n], ub[:n], n_boot=10000, seed=global_seed
            )

    # Random uniformity check
    rand_freq = summaries["random"]["probe_behavior"]["action_frequency"]
    counts = np.asarray(list(rand_freq.values()), dtype=np.float64)
    rand_uniform = {
        "repeat_action_count": summaries["random"]["probe_behavior"]["repeat_action_count"],
        "action_count_cv": float(np.std(counts) / np.mean(counts)) if counts.size and np.mean(counts) > 0 else float("nan"),
        "n_unique_actions_used": int(len(rand_freq)),
        "n_actions": n_actions,
    }

    adapt = dad_adaptation_table(all_rows.get("dad", []))
    _write_json(eval_root / "dad" / "adaptation.json", adapt)

    # Paired rollouts CSV
    paired_rows = []
    n = min(len(all_rows.get(m, [])) for m in ("dad", "myopic", "fixed", "random"))
    for i in range(n):
        paired_rows.append(
            {
                "rollout_id": i,
                "theta_test_id": all_rows["dad"][i]["theta_test_id"],
                "u_dad": all_rows["dad"][i]["u_ctrl"],
                "u_myopic": all_rows["myopic"][i]["u_ctrl"],
                "u_fixed": all_rows["fixed"][i]["u_ctrl"],
                "u_random": all_rows["random"][i]["u_ctrl"],
                "u_req": all_rows["dad"][i]["u_req_true"],
                "safe_dad": all_rows["dad"][i]["safe_total"],
                "safe_myopic": all_rows["myopic"][i]["safe_total"],
                "safe_fixed": all_rows["fixed"][i]["safe_total"],
                "safe_random": all_rows["random"][i]["safe_total"],
            }
        )
    _write_csv(
        eval_root / "paired_rollouts.csv",
        paired_rows,
        list(paired_rows[0].keys()) if paired_rows else ["rollout_id"],
    )

    summary_rows = []
    for m in ("dad", "myopic", "fixed", "random"):
        s = summaries[m]
        summary_rows.append(
            {
                "method": m,
                **{k: s[k] for k in rich_metrics([]).keys() if k in s},
            }
        )
    _write_csv(
        eval_root / "summary.csv",
        summary_rows,
        ["method"] + [k for k in rich_metrics([]).keys()],
    )
    report = {
        "pilot": "ieee5_T2_four_method",
        "terminal_rule": frozen.metadata(),
        "shared_rule_assert": shared,
        "myopic_n_hypothetical": int(control_spec.myopic_hypothetical),
        "summaries": {m: summaries[m] for m in ("dad", "myopic", "fixed", "random")},
        "dad_seed_summaries": dad_seed_summaries,
        "fixed_subset": fixed_meta,
        "paired_differences": pairs,
        "random_uniformity": rand_uniform,
        "dad_adaptation_n_bins": len(adapt),
        "splits": {
            "support": len(splits["support_systems"]),
            "calibration": len(splits["calibration_systems"]),
            "validation": len(splits["validation_systems"]),
            "test": len(splits["test_systems"]),
            "test_ids": splits["test_ids"],
        },
        "eval_rollouts": eval_rollouts,
        "training_seeds": seeds,
    }
    # Pilot pass criteria
    safeties = [summaries[m]["true_safety_rate"] for m in ("dad", "myopic", "fixed", "random")]
    report["pilot_passed"] = bool(
        all(abs(s - 1.0) < 1e-12 for s in safeties)
        and rand_uniform["repeat_action_count"] == 0
        and shared
    )
    _write_json(eval_root / "summary.json", report)

    # Markdown report
    lines = [
        f"# IEEE5 T={horizon} four-method experiment report",
        "",
        f"**Passed: {report['pilot_passed']}**",
        "",
        f"Terminal rule hash: `{frozen.terminal_rule_hash}`",
        f"α={frozen.alpha}, margin={frozen.margin}, quantile={frozen.quantile_level}",
        "",
        "## Per-method metrics",
        "",
        "| method | mean u_ctrl | safety | mean excess | runtime |",
        "|---|---|---|---|---|",
    ]
    for m in ("dad", "myopic", "fixed", "random"):
        s = summaries[m]
        lines.append(
            f"| {m} | {s['mean_u_ctrl']:.4f} | {s['true_safety_rate']:.3f} | "
            f"{s['mean_excess_control']:.4f} | {s['mean_runtime_per_rollout']:.4f}s |"
        )
    lines.extend(["", "## Paired differences (u_A - u_B)", ""])
    for k, v in pairs.items():
        lines.append(
            f"- `{k}`: mean={v['mean_paired_diff']:.4f}  "
            f"CI95=[{v['ci95_low']:.4f},{v['ci95_high']:.4f}]  "
            f"frac_A_lower={v['fraction_a_lower']:.3f}"
        )
    lines.extend(["", f"Fixed subset: `{fixed_subset}`", ""])
    (eval_root / "pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _make_plots(eval_root, {m: summaries[m] for m in ("dad", "myopic", "fixed", "random")}, train_curves, all_rows.get("dad", []))

    # Simple distribution plots
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots = eval_root / "plots"
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for m in ("dad", "myopic", "fixed", "random"):
            ax.hist([r["u_ctrl"] for r in all_rows[m]], bins=20, alpha=0.35, label=m)
        ax.legend()
        ax.set_xlabel("u_ctrl")
        fig.tight_layout()
        fig.savefig(plots / "u_ctrl_distribution.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3.8))
        for m in ("dad", "myopic", "fixed", "random"):
            ax.hist([r["excess_control"] for r in all_rows[m]], bins=20, alpha=0.35, label=m)
        ax.legend()
        ax.set_xlabel("excess control")
        fig.tight_layout()
        fig.savefig(plots / "excess_control_distribution.png", dpi=120)
        plt.close(fig)

        for m in ("dad", "myopic", "fixed", "random"):
            freq = summaries[m]["probe_behavior"]["action_frequency"]
            fig, ax = plt.subplots(figsize=(7, 3.5))
            xs = sorted(int(k) for k in freq)
            ax.bar([str(x) for x in xs], [freq[x] for x in xs], color="#2c5f7c")
            ax.set_title(f"action frequency — {m}")
            fig.tight_layout()
            fig.savefig(plots / f"action_frequency_{m}.png", dpi=120)
            plt.close(fig)

        # DAD adaptation heatmap: first_action x obs_bin → most common second action
        adapt = dad_adaptation_table(all_rows.get("dad", []))
        if adapt:
            firsts = sorted({r["first_action"] for r in adapt})
            bins = sorted({r["first_observation_bin"] for r in adapt})
            mat = np.full((len(firsts), len(bins)), np.nan)
            fi = {a: i for i, a in enumerate(firsts)}
            bi = {b: i for i, b in enumerate(bins)}
            for r in adapt:
                mat[fi[r["first_action"]], bi[r["first_observation_bin"]]] = r[
                    "most_common_second_action"
                ]
            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(bins)))
            ax.set_xticklabels([str(b) for b in bins])
            ax.set_yticks(range(len(firsts)))
            ax.set_yticklabels([str(a) for a in firsts])
            ax.set_xlabel("first observation bin")
            ax.set_ylabel("first action")
            ax.set_title("DAD adaptation (most common 2nd action)")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(plots / "dad_adaptation_heatmap.png", dpi=120)
            plt.close(fig)
    except Exception:
        pass

    print(f"\nPilot passed={report['pilot_passed']}  → {eval_root}")
    return report
