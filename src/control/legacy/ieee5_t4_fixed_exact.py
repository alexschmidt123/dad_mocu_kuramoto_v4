"""IEEE5 T=4 exact Fixed search to close the DAD-versus-Fixed fairness gap.

Does not retrain DAD or change Myopic/Random. Archives approximate Fixed.
"""

from __future__ import annotations

import csv
import json
import math
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from src.control.banks import extract_U_bank
from src.control.fixed_search import estimate_fixed_subset_objective
from src.control.legacy.ieee5_t4 import (
    EXPECTED_HASH,
    FROZEN_MARGIN,
    _write_csv,
    _write_json,
)
from src.control.pilot import (
    evaluate_method_paired,
    load_pilot_splits,
    paired_diff_stats,
)
from src.control.posterior_ctrl import snap_up_to_grid
from src.control.terminal_rule import keyed_noise, load_frozen_terminal_rule
from src.control.u_req import ControlSpec
from src.contrastive.spce import log_prior_uniform_discrete
from src.data import lookup_action_y_sim
from src.rollout import FixedSelector
from src.run_context import load_experiment_run
from src.swing_equation_ode.design import build_catalog
from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables

DAD_DOMINANT_SEQUENCE = [23, 29, 10, 19]
DAD_SUBSET = tuple(sorted(DAD_DOMINANT_SEQUENCE))  # {10, 19, 23, 29}
APPROX_FIXED_SUBSET = (0, 1, 4, 8)


def _log_gauss_const(sigma: float) -> float:
    return float(-0.5 * math.log(2.0 * math.pi * sigma * sigma))


def score_subset_detailed(
    subset: tuple[int, ...] | list[int],
    *,
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    systems: list[dict[str, Any]],
    sigma_y: float,
    alpha: float,
    noise_replicas: int,
    margin: float,
    u_grid: list[float] | np.ndarray,
    global_seed: int,
    centres_cache: dict[int, np.ndarray] | None = None,
) -> dict[str, float]:
    """
    Same Fixed objective as ``estimate_fixed_subset_objective``, with:

    - deterministic keyed noise (order-independent across subsets);
    - validation safety vs banked ``u_req``;
    - score standard error across MC draws.
    """
    subset = tuple(sorted(int(a) for a in subset))
    n_rep = max(1, int(noise_replicas))
    q = 1.0 - float(alpha)
    u_grid_arr = np.asarray(u_grid, dtype=np.float64)
    log_p0 = np.asarray(table_support.log_p0, dtype=np.float64)
    centres = centres_cache or {}
    for a in subset:
        if a not in centres:
            centres[a] = np.asarray(
                y_sim_last_step_from_tables(table_support, [a]), dtype=np.float64
            )

    vals: list[float] = []
    safes: list[float] = []
    c = _log_gauss_const(float(sigma_y))
    inv_2s2 = 1.0 / (2.0 * float(sigma_y) ** 2)

    for s_idx, sys in enumerate(systems):
        u_req = float(sys["u_req"])
        y_clean = np.asarray(
            [lookup_action_y_sim(sys, a) for a in subset], dtype=np.float64
        )
        for rep in range(n_rep):
            y_obs = np.empty(len(subset), dtype=np.float64)
            for t, a in enumerate(subset):
                z = keyed_noise(
                    global_seed=int(global_seed),
                    theta_id=int(s_idx),
                    rollout_id=int(rep),
                    step=int(t),
                    action_id=int(a),
                )
                y_obs[t] = y_clean[t] + float(sigma_y) * z
            log_w = log_p0.copy()
            for t, a in enumerate(subset):
                diff = y_obs[t] - centres[a]
                log_w = log_w + (c - inv_2s2 * diff * diff)
            # softmax
            m = float(np.max(log_w))
            w = np.exp(log_w - m)
            w = w / float(np.sum(w))
            # weighted quantile of U
            order = np.argsort(U_support, kind="mergesort")
            cdf = np.cumsum(w[order])
            idx = int(np.searchsorted(cdf, q, side="left"))
            idx = min(max(idx, 0), U_support.size - 1)
            u0 = float(U_support[order[idx]])
            u_ctrl = snap_up_to_grid(u0 + float(margin), u_grid_arr)
            vals.append(u_ctrl)
            safes.append(1.0 if u_ctrl + 1e-12 >= u_req else 0.0)

    arr = np.asarray(vals, dtype=np.float64)
    safe_arr = np.asarray(safes, dtype=np.float64)
    n = int(arr.size)
    se = float(np.std(arr, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {
        "validation_mean_u_ctrl": float(np.mean(arr)) if n else float("inf"),
        "validation_safety_rate": float(np.mean(safe_arr)) if n else float("nan"),
        "score_standard_error": se,
        "n_mc": float(n),
    }


def exhaustive_score_all_subsets(
    *,
    n_actions: int,
    horizon: int,
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    systems: list[dict[str, Any]],
    sigma_y: float,
    alpha: float,
    noise_replicas: int,
    margin: float,
    u_grid: list[float] | np.ndarray,
    global_seed: int,
    progress_every: int = 2000,
) -> list[dict[str, Any]]:
    """Score every unordered size-T subset with keyed-noise MC (order-independent)."""
    centres: dict[int, np.ndarray] = {
        a: np.asarray(y_sim_last_step_from_tables(table_support, [a]), dtype=np.float64)
        for a in range(n_actions)
    }
    n_total = math.comb(n_actions, horizon)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, subset in enumerate(combinations(range(n_actions), horizon)):
        m = score_subset_detailed(
            subset,
            table_support=table_support,
            U_support=U_support,
            systems=systems,
            sigma_y=sigma_y,
            alpha=alpha,
            noise_replicas=noise_replicas,
            margin=margin,
            u_grid=u_grid,
            global_seed=global_seed,
            centres_cache=centres,
        )
        rows.append(
            {
                "subset": " ".join(map(str, subset)),
                "subset_tuple": subset,
                "validation_mean_u_ctrl": m["validation_mean_u_ctrl"],
                "validation_safety_rate": m["validation_safety_rate"],
                "score_standard_error": m["score_standard_error"],
            }
        )
        if progress_every and ((i + 1) % progress_every == 0 or (i + 1) == n_total):
            print(
                f"  scored {i+1}/{n_total} ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
    return rows


def select_exact_fixed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Lexicographic: safety=1, then min mean u_ctrl, then lex subset."""
    ranked = sorted(
        rows,
        key=lambda r: (
            0 if abs(float(r["validation_safety_rate"]) - 1.0) < 1e-12 else 1,
            float(r["validation_mean_u_ctrl"]),
            tuple(int(x) for x in str(r["subset"]).split()),
        ),
    )
    for rank, r in enumerate(ranked, start=1):
        r["rank"] = rank
    best = ranked[0]
    dad_row = next(
        r
        for r in ranked
        if tuple(int(x) for x in str(r["subset"]).split()) == DAD_SUBSET
    )
    return {
        "exact_fixed_subset": [int(x) for x in str(best["subset"]).split()],
        "exact_fixed_validation_mean": float(best["validation_mean_u_ctrl"]),
        "exact_fixed_validation_safety": float(best["validation_safety_rate"]),
        "DAD_subset": list(DAD_SUBSET),
        "DAD_subset_rank": int(dad_row["rank"]),
        "DAD_subset_validation_mean": float(dad_row["validation_mean_u_ctrl"]),
        "DAD_subset_validation_safety": float(dad_row["validation_safety_rate"]),
        "total_subsets_evaluated": len(ranked),
        "ranked_rows": ranked,
    }


def _load_rollout_u(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return np.asarray([float(r["u_ctrl"]) for r in rows], dtype=np.float64)


def run_ieee5_t4_fixed_exact(
    *,
    project_root: Path | None = None,
    exp_dir: Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    exp_dir = Path(exp_dir or (root / "experiments" / "ieee5_T4")).resolve()
    out = exp_dir / "eval" / "fixed_exact"
    out.mkdir(parents=True, exist_ok=True)

    print("=== IEEE5 T=4 exact Fixed fairness correction ===", flush=True)
    run = load_experiment_run(exp_dir, root)
    splits = load_pilot_splits(exp_dir, run)
    frozen = load_frozen_terminal_rule(exp_dir, expected_margin=FROZEN_MARGIN)
    assert frozen.terminal_rule_hash == EXPECTED_HASH
    control_spec = frozen.to_control_spec(ControlSpec.from_cfg(run.cfg))
    pilot_cfg = dict(run.cfg.raw.get("pilot") or {})
    fixed_seed = int(pilot_cfg.get("fixed_seed", 7))
    global_seed = int(pilot_cfg.get("global_seed", 1234))
    method_seed = int(list(pilot_cfg.get("method_seeds", [101]))[0])
    n_eval = int(pilot_cfg.get("evaluation_rollouts", 1000))
    noise_replicas = int(control_spec.fixed_noise_replicas)

    # Design-selection systems: validation only (never test).
    val_systems = list(splits["validation_systems"])
    # Original Fixed scoring set (archived comparison).
    fixed_cal_original = list(splits["train_systems"][: min(32, len(splits["train_systems"]))])

    table_support = TableThetaSupport(
        systems=splits["support_systems"],
        log_p0=log_prior_uniform_discrete(len(splits["support_systems"])),
    )
    U_support = extract_U_bank(splits["support_systems"])
    catalog = build_catalog(run.cfg)
    n_actions = len(catalog)
    assert math.comb(n_actions, 4) == 27405

    # --- §2: score DAD subset with Fixed search code ---
    print("=== Score DAD subset with Fixed search scorer ===", flush=True)
    rng_dad = np.random.default_rng(fixed_seed)
    dad_obj_legacy = estimate_fixed_subset_objective(
        DAD_SUBSET,
        table_support=table_support,
        U_support=U_support,
        calibration_systems=fixed_cal_original,
        sigma_y=float(run.cfg.sigma_y),
        alpha=frozen.alpha,
        noise_replicas=noise_replicas,
        rng=rng_dad,
        margin=frozen.margin,
        u_grid=frozen.u_candidates,
    )
    rng_fix = np.random.default_rng(fixed_seed)
    approx_obj_legacy = estimate_fixed_subset_objective(
        APPROX_FIXED_SUBSET,
        table_support=table_support,
        U_support=U_support,
        calibration_systems=fixed_cal_original,
        sigma_y=float(run.cfg.sigma_y),
        alpha=frozen.alpha,
        noise_replicas=noise_replicas,
        rng=rng_fix,
        margin=frozen.margin,
        u_grid=frozen.u_candidates,
    )
    dad_val = score_subset_detailed(
        DAD_SUBSET,
        table_support=table_support,
        U_support=U_support,
        systems=val_systems,
        sigma_y=float(run.cfg.sigma_y),
        alpha=frozen.alpha,
        noise_replicas=noise_replicas,
        margin=frozen.margin,
        u_grid=frozen.u_candidates,
        global_seed=fixed_seed,
    )
    approx_val = score_subset_detailed(
        APPROX_FIXED_SUBSET,
        table_support=table_support,
        U_support=U_support,
        systems=val_systems,
        sigma_y=float(run.cfg.sigma_y),
        alpha=frozen.alpha,
        noise_replicas=noise_replicas,
        margin=frozen.margin,
        u_grid=frozen.u_candidates,
        global_seed=fixed_seed,
    )
    print(
        f"  Fixed-code score (train[:32] MC): DAD_subset={dad_obj_legacy:.6f} "
        f"approx_Fixed={approx_obj_legacy:.6f}",
        flush=True,
    )
    print(
        f"  Validation keyed score: DAD mean={dad_val['validation_mean_u_ctrl']:.6f} "
        f"safety={dad_val['validation_safety_rate']:.3f} | "
        f"approx_Fixed mean={approx_val['validation_mean_u_ctrl']:.6f} "
        f"safety={approx_val['validation_safety_rate']:.3f}",
        flush=True,
    )

    # --- §3–4: exhaustive search ---
    print("=== Exhaustive Fixed search C(30,4)=27405 (validation systems) ===", flush=True)
    t_search0 = time.perf_counter()
    raw_rows = exhaustive_score_all_subsets(
        n_actions=n_actions,
        horizon=4,
        table_support=table_support,
        U_support=U_support,
        systems=val_systems,
        sigma_y=float(run.cfg.sigma_y),
        alpha=frozen.alpha,
        noise_replicas=noise_replicas,
        margin=frozen.margin,
        u_grid=frozen.u_candidates,
        global_seed=fixed_seed,
    )
    search_runtime = float(time.perf_counter() - t_search0)
    selection = select_exact_fixed(raw_rows)
    selection["search_runtime"] = search_runtime
    selection["noise_replicas"] = noise_replicas
    selection["n_validation_systems"] = len(val_systems)
    selection["scoring"] = {
        "systems": "validation_ids_only",
        "validation_ids": list(splits["validation_ids"]),
        "noise": "keyed_gaussian",
        "fixed_seed": fixed_seed,
        "terminal_rule_hash": frozen.terminal_rule_hash,
        "used_test_systems": False,
    }

    # Archive approximate Fixed label
    approx_meta_path = exp_dir / "eval" / "fixed" / "subset_meta.json"
    if approx_meta_path.is_file():
        approx_meta = json.loads(approx_meta_path.read_text(encoding="utf-8"))
        approx_meta["label"] = "approximately optimized Fixed baseline (archived)"
        approx_meta["superseded_by"] = "eval/fixed_exact"
        _write_json(approx_meta_path, approx_meta)

    exact_subset = list(selection["exact_fixed_subset"])
    print(
        f"  exact Fixed={exact_subset}  val_mean={selection['exact_fixed_validation_mean']:.6f} "
        f"DAD_rank={selection['DAD_subset_rank']}  runtime={search_runtime:.1f}s",
        flush=True,
    )

    # Inconsistency stop: exact Fixed cannot be worse than DAD subset on same scores
    if (
        selection["exact_fixed_validation_mean"]
        > selection["DAD_subset_validation_mean"] + 1e-12
    ):
        msg = (
            "IMPLEMENTATION INCONSISTENCY: exact Fixed validation mean "
            f"{selection['exact_fixed_validation_mean']} > DAD subset "
            f"{selection['DAD_subset_validation_mean']}"
        )
        _write_json(out / "selected_fixed_subset.json", {**selection, "error": msg})
        raise RuntimeError(msg)

    # Persist all scores
    score_rows = [
        {
            "subset": r["subset"],
            "validation_mean_u_ctrl": r["validation_mean_u_ctrl"],
            "validation_safety_rate": r["validation_safety_rate"],
            "score_standard_error": r["score_standard_error"],
            "rank": r["rank"],
        }
        for r in selection["ranked_rows"]
    ]
    _write_csv(out / "all_subset_scores.csv", score_rows)

    selected_payload = {
        k: selection[k]
        for k in (
            "exact_fixed_subset",
            "exact_fixed_validation_mean",
            "exact_fixed_validation_safety",
            "DAD_subset",
            "DAD_subset_rank",
            "DAD_subset_validation_mean",
            "DAD_subset_validation_safety",
            "total_subsets_evaluated",
            "search_runtime",
            "noise_replicas",
            "n_validation_systems",
            "scoring",
        )
    }
    selected_payload["archived_approximate_fixed"] = {
        "subset": list(APPROX_FIXED_SUBSET),
        "label": "approximately optimized Fixed baseline",
        "validation_keyed_mean_u_ctrl": approx_val["validation_mean_u_ctrl"],
        "validation_keyed_safety": approx_val["validation_safety_rate"],
        "legacy_train32_objective_dad_subset": dad_obj_legacy,
        "legacy_train32_objective_approx_fixed": approx_obj_legacy,
    }
    selected_payload["dad_sequence"] = list(DAD_DOMINANT_SEQUENCE)
    selected_payload["terminal_rule_hash"] = frozen.terminal_rule_hash
    _write_json(out / "selected_fixed_subset.json", selected_payload)

    dad_cmp_rows = [
        {
            "label": "DAD_subset",
            "subset": " ".join(map(str, DAD_SUBSET)),
            "validation_mean_u_ctrl": selection["DAD_subset_validation_mean"],
            "validation_safety_rate": selection["DAD_subset_validation_safety"],
            "rank": selection["DAD_subset_rank"],
            "legacy_train32_objective": dad_obj_legacy,
        },
        {
            "label": "exact_Fixed",
            "subset": " ".join(map(str, exact_subset)),
            "validation_mean_u_ctrl": selection["exact_fixed_validation_mean"],
            "validation_safety_rate": selection["exact_fixed_validation_safety"],
            "rank": 1,
            "legacy_train32_objective": "",
        },
        {
            "label": "archived_approximate_Fixed",
            "subset": " ".join(map(str, APPROX_FIXED_SUBSET)),
            "validation_mean_u_ctrl": approx_val["validation_mean_u_ctrl"],
            "validation_safety_rate": approx_val["validation_safety_rate"],
            "rank": next(
                r["rank"]
                for r in selection["ranked_rows"]
                if tuple(int(x) for x in str(r["subset"]).split()) == APPROX_FIXED_SUBSET
            ),
            "legacy_train32_objective": approx_obj_legacy,
        },
    ]
    _write_csv(out / "dad_subset_comparison.csv", dad_cmp_rows)

    # --- §5: reevaluate exact Fixed on paired test rollouts ---
    print("=== Paired test evaluation of exact Fixed ===", flush=True)
    from src.swing_equation_ode.design import build_simulator
    from src.control.cuda_control import CudaControlEngine

    sim = build_simulator(run.cfg)
    sim.T_obs_sec = control_spec.T_obs_sec
    sim.ode_dt = control_spec.ode_dt
    sim.fs_hz = control_spec.fs_hz
    engine = CudaControlEngine(sim, control_spec)

    fixed_rows, fixed_summary = evaluate_method_paired(
        method="fixed_exact",
        selector_factory=lambda: FixedSelector(sequence=list(exact_subset)),
        test_systems=splits["test_systems"],
        table_support=table_support,
        U_support=U_support,
        frozen=frozen,
        control_spec=control_spec,
        control_engine=engine,
        horizon=4,
        n_actions=n_actions,
        sigma_y=float(run.cfg.sigma_y),
        n_rollouts=n_eval,
        global_seed=global_seed,
        method_seed=method_seed,
    )
    _write_json(out / "fixed_exact_test_summary.json", fixed_summary)
    u_fixed = np.asarray([r["u_ctrl"] for r in fixed_rows], dtype=np.float64)
    safe_fixed = float(np.mean([1.0 if r["safe_total"] else 0.0 for r in fixed_rows]))

    # Load DAD seed / primary rollouts (unchanged; do not retrain DAD)
    seed_paths = {
        101: exp_dir / "eval" / "dad_seed_101" / "rollouts.csv",
        202: exp_dir / "eval" / "dad_seed_202" / "rollouts.csv",
        303: exp_dir / "eval" / "dad_seed_303" / "rollouts.csv",
    }
    u_by_seed = {s: _load_rollout_u(p) for s, p in seed_paths.items()}
    u_dad_primary = _load_rollout_u(exp_dir / "eval" / "dad" / "rollouts.csv")
    # Dominant-sequence diagnostic on the same paired test keys
    dom_rows, dom_summary = evaluate_method_paired(
        method="dad_dominant_sequence_diagnostic",
        selector_factory=lambda: FixedSelector(sequence=list(DAD_DOMINANT_SEQUENCE)),
        test_systems=splits["test_systems"],
        table_support=table_support,
        U_support=U_support,
        frozen=frozen,
        control_spec=control_spec,
        control_engine=engine,
        horizon=4,
        n_actions=n_actions,
        sigma_y=float(run.cfg.sigma_y),
        n_rollouts=n_eval,
        global_seed=global_seed,
        method_seed=method_seed,
    )
    u_dom = np.asarray([r["u_ctrl"] for r in dom_rows], dtype=np.float64)

    paired_rows = []
    paired_stats: dict[str, Any] = {}

    def _add_pair(name: str, ua: np.ndarray) -> None:
        n = min(len(ua), len(u_fixed))
        stats = paired_diff_stats(ua[:n], u_fixed[:n], n_boot=10000, seed=global_seed)
        paired_stats[name] = stats
        for i in range(n):
            paired_rows.append(
                {
                    "contrast": name,
                    "rollout_id": i,
                    "u_first": float(ua[i]),
                    "u_exact_fixed": float(u_fixed[i]),
                    "diff": float(ua[i] - u_fixed[i]),
                }
            )

    for seed, ua in u_by_seed.items():
        _add_pair(f"dad_seed_{seed}_minus_exact_fixed", ua)
    _add_pair("dad_primary_minus_exact_fixed", u_dad_primary)
    # Aggregate across seeds: mean u per rollout, then pair
    u_agg = np.mean(np.stack([u_by_seed[s] for s in (101, 202, 303)], axis=0), axis=0)
    _add_pair("dad_seed_mean_minus_exact_fixed", u_agg)
    _add_pair("dad_dominant_sequence_minus_exact_fixed", u_dom)

    _write_csv(out / "paired_dad_fixed.csv", paired_rows)
    _write_json(out / "paired_stats.json", paired_stats)

    seed_means = {s: float(np.mean(u_by_seed[s])) for s in (101, 202, 303)}
    seed_arr = np.asarray([seed_means[s] for s in (101, 202, 303)], dtype=np.float64)
    across = {
        "seed_means": seed_means,
        "mean_across_seeds": float(np.mean(seed_arr)),
        "std_across_seeds": float(np.std(seed_arr, ddof=1)),
        "primary_seed": 101,
        "primary_mean": float(np.mean(u_dad_primary)),
    }

    # Interpretation (prescribed rules; no adaptive claim without action variation)
    dad_eq_exact = list(exact_subset) == list(DAD_SUBSET)
    primary_stats = paired_stats["dad_primary_minus_exact_fixed"]
    agg_stats = paired_stats["dad_seed_mean_minus_exact_fixed"]
    ci_primary = (primary_stats["ci95_low"], primary_stats["ci95_high"])
    ci_agg = (agg_stats["ci95_low"], agg_stats["ci95_high"])
    primary_beats = ci_primary[1] < 0.0
    agg_tied_or_worse = ci_agg[0] <= 0.0  # CI includes 0 or Fixed better

    if dad_eq_exact:
        interpretation = (
            "DAD learned the optimal or tied-best fixed design, but did not use "
            "observation-dependent adaptation."
        )
    elif agg_tied_or_worse:
        interpretation = (
            "DAD did not demonstrate an adaptive advantage at IEEE5 T=4. "
            "Its previous advantage resulted from a stronger learned fixed sequence "
            "compared with an underoptimized approximate Fixed baseline. "
            "Across training seeds, DAD is statistically tied with exact Fixed; "
            "DAD remains effectively nonadaptive (one sequence on every rollout)."
        )
    elif primary_beats:
        interpretation = (
            "DAD did not demonstrate an adaptive advantage at IEEE5 T=4. "
            "The validation-selected seed beats exact Fixed on the paired test, "
            "but the policy is effectively nonadaptive (frozen sequence "
            f"{DAD_DOMINANT_SEQUENCE}); treat any gain as a learned fixed design, "
            "not observation-dependent adaptation."
        )
    else:
        interpretation = (
            "DAD did not demonstrate an adaptive advantage at IEEE5 T=4. "
            "Its previous advantage resulted from a stronger learned fixed sequence "
            "compared with an underoptimized approximate Fixed baseline."
        )

    report = {
        "exact_fixed_subset": exact_subset,
        "exact_fixed_validation_mean": selection["exact_fixed_validation_mean"],
        "DAD_subset_rank": selection["DAD_subset_rank"],
        "DAD_subset_validation_mean": selection["DAD_subset_validation_mean"],
        "exact_fixed_test_mean_u_ctrl": float(fixed_summary["mean_u_ctrl"]),
        "exact_fixed_test_safety": safe_fixed,
        "across_seeds": across,
        "paired_stats": paired_stats,
        "dad_equals_exact_fixed": dad_eq_exact,
        "interpretation": interpretation,
        "adaptive_benefit_demonstrated": False,
        "can_freeze_ieee5": True,
        "terminal_rule_hash": frozen.terminal_rule_hash,
        "search_runtime": search_runtime,
        "total_subsets_evaluated": selection["total_subsets_evaluated"],
        "dom_summary_mean_u": float(dom_summary["mean_u_ctrl"]),
    }
    _write_json(out / "summary.json", report)
    _write_fixed_exact_report(out, report, dad_cmp_rows, selected_payload)
    update_horizon_corrected(root, report)

    print(f"  exact Fixed test mean_u={report['exact_fixed_test_mean_u_ctrl']:.4f} "
          f"safety={safe_fixed:.3f}", flush=True)
    print(f"  DAD subset rank={report['DAD_subset_rank']}", flush=True)
    print(f"  interpretation: {interpretation}", flush=True)
    print(f"Outputs → {out}", flush=True)
    return report


def _write_fixed_exact_report(
    out: Path,
    report: dict[str, Any],
    dad_cmp_rows: list[dict[str, Any]],
    selected: dict[str, Any],
) -> None:
    ps = report["paired_stats"]
    across = report["across_seeds"]
    lines = [
        "# IEEE5 T=4 exact Fixed fairness report",
        "",
        "Approximate Fixed (`greedy_multistart`, 340 subsets) is archived.",
        "This report uses exhaustive search over all 27,405 size-4 subsets.",
        "",
        "## Frozen rule (unchanged)",
        "",
        f"- terminal_rule_hash = `{report['terminal_rule_hash']}`",
        "- alpha = 0.05, additive_margin = 0.55",
        "",
        "## Exact Fixed selection",
        "",
        f"- exact_fixed_subset: `{report['exact_fixed_subset']}`",
        f"- exact_fixed_validation_mean: `{report['exact_fixed_validation_mean']}`",
        f"- DAD_subset: `{selected['DAD_subset']}`",
        f"- DAD_subset_rank: `{report['DAD_subset_rank']}`",
        f"- DAD_subset_validation_mean: `{report['DAD_subset_validation_mean']}`",
        f"- total_subsets_evaluated: `{report['total_subsets_evaluated']}`",
        f"- search_runtime: `{report['search_runtime']:.2f}s`",
        "",
        "## Validation subset comparison",
        "",
    ]
    for r in dad_cmp_rows:
        lines.append(
            f"- {r['label']}: subset=`{r['subset']}` mean=`{r['validation_mean_u_ctrl']}` "
            f"safety=`{r['validation_safety_rate']}` rank=`{r['rank']}`"
        )
    lines.extend(
        [
            "",
            "## Test evaluation (paired rollouts)",
            "",
            f"- exact Fixed mean u_ctrl: `{report['exact_fixed_test_mean_u_ctrl']:.6f}`",
            f"- exact Fixed safety: `{report['exact_fixed_test_safety']:.6f}`",
            "",
            "## DAD across seeds (test)",
            "",
            f"- seed means: `{across['seed_means']}`",
            f"- mean across seeds: `{across['mean_across_seeds']:.6f}`",
            f"- std across seeds: `{across['std_across_seeds']:.6f}`",
            f"- selected-model (seed {across['primary_seed']}): `{across['primary_mean']:.6f}`",
            "",
            "## Paired differences (DAD − exact Fixed)",
            "",
        ]
    )
    for name, st in ps.items():
        lines.append(
            f"- `{name}`: mean={st['mean_paired_diff']:.6f} "
            f"CI95=[{st['ci95_low']:.6f}, {st['ci95_high']:.6f}] "
            f"tied={st['fraction_tied']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            f"- adaptive_benefit_demonstrated: `{report['adaptive_benefit_demonstrated']}`",
            f"- dad_equals_exact_fixed: `{report['dad_equals_exact_fixed']}`",
            f"- can_freeze_ieee5: `{report['can_freeze_ieee5']}`",
            "",
        ]
    )
    (out / "fixed_exact_report.md").write_text("\n".join(lines), encoding="utf-8")


def update_horizon_corrected(root: Path, t4_report: dict[str, Any]) -> None:
    """Write corrected horizon summary with exact Fixed at T=4."""
    out = root / "experiments" / "ieee5_horizon_summary"
    out.mkdir(parents=True, exist_ok=True)

    # Load prior T=2/T=3 from existing summary if present
    old = out / "ieee5_T2_T3_T4_summary.csv"
    rows: list[dict[str, Any]] = []
    if old.is_file():
        with old.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if int(r["T"]) in (2, 3):
                    rows.append(dict(r))

    # T=4 corrected rows
    t4_eval = root / "experiments" / "ieee5_T4" / "eval" / "summary.json"
    rep = json.loads(t4_eval.read_text(encoding="utf-8")) if t4_eval.is_file() else {}
    summaries = rep.get("summaries") or {}
    across = t4_report["across_seeds"]
    exact_u = t4_report["exact_fixed_test_mean_u_ctrl"]
    exact_safe = t4_report["exact_fixed_test_safety"]
    ps = t4_report["paired_stats"]

    def _row(method: str, mean_u: float, safety: float, excess: Any, runtime: Any, **paired):
        return {
            "T": 4,
            "method": method,
            "mean_u_ctrl": mean_u,
            "true_safety_rate": safety,
            "mean_excess_control": excess,
            "runtime": runtime,
            "paired_difference_vs_exact_fixed": paired.get("vs_fixed", ""),
            "note": paired.get("note", ""),
            "fixed_search": paired.get("fixed_search", "exact_exhaustive"),
        }

    rows.append(
        _row(
            "dad_primary_seed101",
            across["primary_mean"],
            1.0,
            (summaries.get("dad") or {}).get("mean_excess_control", ""),
            (summaries.get("dad") or {}).get("mean_runtime_per_rollout", ""),
            vs_fixed=ps["dad_primary_minus_exact_fixed"]["mean_paired_diff"],
            note="selected-model; effectively_nonadaptive",
        )
    )
    rows.append(
        _row(
            "dad_across_seeds_mean",
            across["mean_across_seeds"],
            1.0,
            "",
            "",
            vs_fixed=ps["dad_seed_mean_minus_exact_fixed"]["mean_paired_diff"],
            note=f"std_across_seeds={across['std_across_seeds']:.6f}; nonadaptive",
        )
    )
    seed_means = across["seed_means"]
    for seed in (101, 202, 303):
        mean_u = seed_means.get(seed, seed_means.get(str(seed)))
        rows.append(
            _row(
                f"dad_seed_{seed}",
                mean_u,
                1.0,
                "",
                "",
                vs_fixed=ps[f"dad_seed_{seed}_minus_exact_fixed"]["mean_paired_diff"],
                note="effectively_nonadaptive",
            )
        )
    rows.append(
        _row(
            "fixed_exact",
            exact_u,
            exact_safe,
            "",
            "",
            vs_fixed=0.0,
            note="exhaustive C(30,4)=27405",
            fixed_search="exact_exhaustive",
        )
    )
    rows.append(
        _row(
            "fixed_approximate_archived",
            (summaries.get("fixed") or {}).get("mean_u_ctrl", 0.8542),
            1.0,
            (summaries.get("fixed") or {}).get("mean_excess_control", ""),
            (summaries.get("fixed") or {}).get("mean_runtime_per_rollout", ""),
            vs_fixed="",
            note="greedy_multistart 340 subsets; archived",
            fixed_search="approximate_greedy_multistart",
        )
    )
    for m in ("myopic", "random"):
        s = summaries.get(m) or {}
        rows.append(
            _row(
                m,
                s.get("mean_u_ctrl", ""),
                s.get("true_safety_rate", ""),
                s.get("mean_excess_control", ""),
                s.get("mean_runtime_per_rollout", ""),
                note="unchanged from T4 pilot",
                fixed_search="n/a",
            )
        )

    _write_csv(out / "ieee5_T2_T3_T4_summary_corrected.csv", rows)

    lines = [
        "# IEEE5 horizon report (corrected: exact Fixed at T=4)",
        "",
        "Frozen terminal rule: α=0.05, margin=0.55, hash=`c2e2af33cb68a5ea`.",
        "",
        "## Adaptivity",
        "",
        "- T=3 DAD: effectively nonadaptive (one sequence, fraction 1.0).",
        "- T=4 DAD: effectively nonadaptive (sequence `[23,29,10,19]`, fraction 1.0).",
        "- No observation-dependent adaptation benefit is claimed.",
        "",
        "## T=4 Fixed correction",
        "",
        f"- Exact Fixed subset: `{t4_report['exact_fixed_subset']}`",
        f"- Exact Fixed validation mean: `{t4_report['exact_fixed_validation_mean']}`",
        f"- DAD subset rank: `{t4_report['DAD_subset_rank']}` / 27405",
        f"- Exact Fixed test mean u_ctrl: `{t4_report['exact_fixed_test_mean_u_ctrl']:.6f}`",
        f"- Exact Fixed safety: `{t4_report['exact_fixed_test_safety']:.6f}`",
        "- Previous approximate Fixed (`[0,1,4,8]`, 340 subsets) remains archived.",
        "",
        "## T=4 DAD results",
        "",
        f"- Selected-model (seed 101): `{across['primary_mean']:.6f}`",
        f"- Across-seed mean ± std: `{across['mean_across_seeds']:.6f} ± {across['std_across_seeds']:.6f}`",
        f"- Seed means: `{across['seed_means']}`",
        "",
        "## Paired DAD − exact Fixed",
        "",
    ]
    for name, st in t4_report["paired_stats"].items():
        lines.append(
            f"- `{name}`: mean={st['mean_paired_diff']:.6f} "
            f"CI95=[{st['ci95_low']:.6f}, {st['ci95_high']:.6f}]"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            t4_report["interpretation"],
            "",
            f"- Adaptive benefit demonstrated: `{t4_report['adaptive_benefit_demonstrated']}`",
            f"- IEEE5 scientifically freezable: `{t4_report['can_freeze_ieee5']}`",
            "",
            "## Mean u_ctrl by T (reference; T=4 Fixed = exact)",
            "",
            "| T | dad (primary) | fixed | myopic | random |",
            "|---|---:|---:|---:|---:|",
            "| 2 | 0.8246 | 0.8501 | 0.8688 | 0.8916 |",
            "| 3 | 0.8433 | 0.8420 | 0.8604 | 0.8704 |",
            f"| 4 | {across['primary_mean']:.4f} | "
            f"{t4_report['exact_fixed_test_mean_u_ctrl']:.4f} | "
            f"{(summaries.get('myopic') or {}).get('mean_u_ctrl', float('nan')):.4f} | "
            f"{(summaries.get('random') or {}).get('mean_u_ctrl', float('nan')):.4f} |",
            "",
        ]
    )
    (out / "ieee5_horizon_report_corrected.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    run_ieee5_t4_fixed_exact()
