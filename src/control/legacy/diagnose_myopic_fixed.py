"""Diagnose Myopic vs Fixed: paired CI, MC error, ties, complementarity, consistency."""

from __future__ import annotations

import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from src.control.banks import extract_U_bank
from src.control.cuda_control import CudaControlEngine
from src.control.fixed_search import estimate_fixed_subset_objective
from src.control.myopic import MyopicControlSelector, score_all_actions_myopic
from src.control.pilot import (
    evaluate_method_paired,
    load_pilot_splits,
    paired_diff_stats,
    rich_metrics,
)
from src.control.posterior_ctrl import normalize_log_weights, posterior_safe_u_ctrl
from src.control.terminal_rule import load_frozen_terminal_rule, posterior_to_u_ctrl
from src.control.u_req import ControlSpec
from src.contrastive.spce import log_prior_uniform_discrete
from src.rollout import FixedSelector, update_log_weights
from src.run_context import load_experiment_run
from src.swing_equation_ode.design import build_catalog, build_simulator
from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_fixed_subset(exp_dir: Path) -> list[int]:
    meta_path = exp_dir / "eval" / "fixed" / "subset_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Fixed subset meta missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return [int(a) for a in meta["selected_action_ids"]]


def _mc_verdict(by_nh: dict[str, Any], configured: int) -> dict[str, Any]:
    if str(configured) in by_nh:
        key = str(configured)
    else:
        key = sorted(by_nh, key=lambda k: abs(int(k) - configured))[0]
    row = by_nh[key]
    excessive = bool(
        row["top1_flip_rate"] > 0.25
        or row["fraction_gap_lt_2se"] > 0.5
        or row["mean_gap_best_second"]
        < 1.5 * max(row["empirical_se_ref_best"], 1e-15)
    )
    return {
        "configured_n_hypothetical": configured,
        "row_used": key,
        "excessive_mc_error": excessive,
        "reason": (
            "Top-1 unstable and/or best–second gap comparable to score SE"
            if excessive
            else "Top-1 stable relative to score SE at configured n_hypothetical"
        ),
    }


def diagnose_myopic_mc_accuracy(
    *,
    selector_template: dict[str, Any],
    n_actions: int,
    n_reps: int,
    seed: int,
    n_h_list: list[int] | None = None,
) -> dict[str, Any]:
    """Prior-step myopic score stability vs hypothetical sample count (no test)."""
    n_h_list = n_h_list or [8, 16, 32, 64, 128, 256]
    log_w = np.asarray(selector_template["table_support"].log_p0, dtype=np.float64).copy()
    weights = normalize_log_weights(log_w)

    ref_sel = MyopicControlSelector(**{**selector_template, "n_hypothetical": max(n_h_list)})
    ref_scores = score_all_actions_myopic(
        ref_sel,
        used=set(),
        log_weights=log_w,
        weights=weights,
        rng=np.random.default_rng(seed + 1),
    )
    ref_order = sorted(ref_scores, key=lambda a: (ref_scores[a], a))
    ref_best = ref_order[0]

    by_nh: dict[str, Any] = {}
    for n_h in n_h_list:
        tops: list[int] = []
        gaps: list[float] = []
        tie_fracs: list[float] = []
        score_by_a = np.zeros((n_reps, n_actions), dtype=np.float64)
        for r in range(n_reps):
            scores = score_all_actions_myopic(
                MyopicControlSelector(**{**selector_template, "n_hypothetical": n_h}),
                used=set(),
                log_weights=log_w,
                weights=weights,
                rng=np.random.default_rng(seed + 1000 * n_h + r),
            )
            for a, j in scores.items():
                score_by_a[r, a] = j
            ordered = sorted(scores, key=lambda a: (scores[a], a))
            tops.append(ordered[0])
            gaps.append(float(scores[ordered[1]] - scores[ordered[0]]))
            best = scores[ordered[0]]
            near = sum(1 for j in scores.values() if abs(j - best) <= 1e-12)
            tie_fracs.append((near - 1) / max(n_actions - 1, 1))

        emp_se = float(np.mean(np.std(score_by_a, axis=0, ddof=1)))
        se_best = float(np.std(score_by_a[:, ref_best], ddof=1)) if n_reps > 1 else float("nan")
        top_counts = Counter(tops)
        flip_rate = 1.0 - (top_counts.most_common(1)[0][1] / n_reps)
        by_nh[str(n_h)] = {
            "n_hypothetical": n_h,
            "n_reps": n_reps,
            "top1_mode": int(top_counts.most_common(1)[0][0]),
            "top1_mode_frequency": float(top_counts.most_common(1)[0][1] / n_reps),
            "top1_flip_rate": float(flip_rate),
            "agree_with_large_nh_ref": float(np.mean([t == ref_best for t in tops])),
            "mean_gap_best_second": float(np.mean(gaps)),
            "median_gap_best_second": float(np.median(gaps)),
            "mean_empirical_score_se": emp_se,
            "empirical_se_ref_best": se_best,
            "fraction_gap_lt_2se": float(
                np.mean(np.asarray(gaps) < 2.0 * max(se_best, 1e-15))
            ),
            "mean_exact_tie_fraction_among_others": float(np.mean(tie_fracs)),
            "ref_best_action": int(ref_best),
            "ref_best_score": float(ref_scores[ref_best]),
        }

    return {
        "setting": "prior_step_uniform_weights",
        "ref_n_hypothetical": max(n_h_list),
        "ref_best_action": int(ref_best),
        "ref_top5": [
            {"action": int(a), "score": float(ref_scores[a])} for a in ref_order[:5]
        ],
        "by_n_hypothetical": by_nh,
        "configured_n_hypothetical": int(selector_template.get("n_hypothetical", 16)),
        "verdict_hints": _mc_verdict(by_nh, int(selector_template.get("n_hypothetical", 16))),
    }


def diagnose_quantized_ties(
    *,
    selector: MyopicControlSelector,
    n_contexts: int,
    seed: int,
    systems: list[dict[str, Any]],
    sigma_y: float,
    global_seed: int,
) -> dict[str, Any]:
    """Exact/near ties in myopic scores on train/val contexts (not test)."""
    from src.control.terminal_rule import observe_with_keyed_noise

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    near_eps = 1e-3
    for c in range(n_contexts):
        sys = systems[int(rng.integers(len(systems)))]
        log_w = np.asarray(selector.table_support.log_p0, dtype=np.float64).copy()
        weights = normalize_log_weights(log_w)
        a0 = selector.select(used=set(), log_weights=log_w, weights=weights, rng=rng)
        scores0 = dict(selector.last_scores)
        ordered0 = sorted(scores0, key=lambda a: (scores0[a], a))
        best0 = scores0[ordered0[0]]
        exact_ties0 = sum(1 for j in scores0.values() if abs(j - best0) <= 1e-12) - 1
        near_ties0 = sum(1 for j in scores0.values() if abs(j - best0) <= near_eps) - 1
        gap0 = float(scores0[ordered0[1]] - best0) if len(ordered0) > 1 else float("nan")

        y0 = observe_with_keyed_noise(
            sys,
            a0,
            sigma_y=sigma_y,
            global_seed=global_seed,
            theta_id=c,
            rollout_id=c,
            step=0,
        )
        centres = y_sim_last_step_from_tables(selector.table_support, [a0])
        log_w = update_log_weights(log_w, y0, centres, sigma_y)
        weights = normalize_log_weights(log_w)
        a1 = selector.select(used={a0}, log_weights=log_w, weights=weights, rng=rng)
        scores1 = dict(selector.last_scores)
        ordered1 = sorted(scores1, key=lambda a: (scores1[a], a))
        best1 = scores1[ordered1[0]]
        exact_ties1 = sum(1 for j in scores1.values() if abs(j - best1) <= 1e-12) - 1
        near_ties1 = sum(1 for j in scores1.values() if abs(j - best1) <= near_eps) - 1
        gap1 = float(scores1[ordered1[1]] - best1) if len(ordered1) > 1 else float("nan")
        records.append(
            {
                "exact_ties_step0": exact_ties0,
                "near_ties_step0": near_ties0,
                "gap_step0": gap0,
                "exact_ties_step1": exact_ties1,
                "near_ties_step1": near_ties1,
                "gap_step1": gap1,
                "a0": a0,
                "a1": a1,
            }
        )

    def _agg(key: str) -> dict[str, float]:
        xs = np.asarray([r[key] for r in records], dtype=np.float64)
        return {
            "mean": float(np.mean(xs)),
            "median": float(np.median(xs)),
            "frac_positive": float(np.mean(xs > 0)),
            "frac_ge_5": float(np.mean(xs >= 5)),
        }

    many = bool(
        _agg("exact_ties_step0")["frac_positive"] > 0.3
        or _agg("near_ties_step0")["mean"] >= 3.0
        or float(np.mean([r["gap_step0"] < 1e-3 for r in records])) > 0.5
    )
    return {
        "n_contexts": n_contexts,
        "step0_exact_ties": _agg("exact_ties_step0"),
        "step0_near_ties_eps_1e-3": _agg("near_ties_step0"),
        "step0_gap": {
            "mean": float(np.mean([r["gap_step0"] for r in records])),
            "median": float(np.median([r["gap_step0"] for r in records])),
            "frac_gap_lt_1e-3": float(np.mean([r["gap_step0"] < 1e-3 for r in records])),
            "frac_gap_lt_1e-2": float(np.mean([r["gap_step0"] < 1e-2 for r in records])),
        },
        "step1_exact_ties": _agg("exact_ties_step1"),
        "step1_near_ties_eps_1e-3": _agg("near_ties_step1"),
        "step1_gap": {
            "mean": float(np.mean([r["gap_step1"] for r in records])),
            "median": float(np.median([r["gap_step1"] for r in records])),
            "frac_gap_lt_1e-3": float(np.mean([r["gap_step1"] < 1e-3 for r in records])),
        },
        "many_quantized_ties": many,
    }


def diagnose_complementarity(
    *,
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    systems: list[dict[str, Any]],
    sigma_y: float,
    frozen,
    n_actions: int,
    fixed_subset: list[int],
    noise_replicas: int,
    seed: int,
    top_k_singles: int = 10,
) -> dict[str, Any]:
    """Joint subset vs singles on train/val systems only."""
    rng = np.random.default_rng(seed)
    alpha, margin, grid = frozen.alpha, frozen.margin, frozen.u_candidates

    single_scores: dict[int, float] = {}
    for a in range(n_actions):
        single_scores[a] = estimate_fixed_subset_objective(
            [a],
            table_support=table_support,
            U_support=U_support,
            calibration_systems=systems,
            sigma_y=sigma_y,
            alpha=alpha,
            noise_replicas=noise_replicas,
            rng=np.random.default_rng(seed + a),
            margin=margin,
            u_grid=grid,
        )
    ranked_singles = sorted(single_scores, key=lambda a: (single_scores[a], a))
    top_singles = ranked_singles[:top_k_singles]
    best_single = ranked_singles[0]
    second_single = ranked_singles[1]
    naive_pair = tuple(sorted((best_single, second_single)))
    fixed_t = tuple(sorted(fixed_subset))

    pairs_of_interest = {fixed_t, naive_pair}
    for i in range(len(top_singles)):
        for j in range(i + 1, len(top_singles)):
            pairs_of_interest.add(tuple(sorted((top_singles[i], top_singles[j]))))
    all_pairs = list(combinations(range(n_actions), 2))
    extra = rng.choice(len(all_pairs), size=min(40, len(all_pairs)), replace=False)
    for idx in extra:
        pairs_of_interest.add(all_pairs[int(idx)])

    pair_scores: dict[str, float] = {}
    for pair in pairs_of_interest:
        pair_scores[",".join(map(str, pair))] = estimate_fixed_subset_objective(
            list(pair),
            table_support=table_support,
            U_support=U_support,
            calibration_systems=systems,
            sigma_y=sigma_y,
            alpha=alpha,
            noise_replicas=noise_replicas,
            rng=np.random.default_rng(seed + 10_000 + (hash(pair) % 10_000)),
            margin=margin,
            u_grid=grid,
        )

    fixed_key = ",".join(map(str, fixed_t))
    naive_key = ",".join(map(str, naive_pair))
    fixed_obj = pair_scores[fixed_key]
    naive_obj = pair_scores[naive_key]
    a, b = fixed_t
    better_single_in_fixed = min(single_scores[a], single_scores[b])
    joint_gain = better_single_in_fixed - fixed_obj

    seq_first = best_single
    cond_second_scores = {}
    for a2 in range(n_actions):
        if a2 == seq_first:
            continue
        cond_second_scores[a2] = estimate_fixed_subset_objective(
            [seq_first, a2],
            table_support=table_support,
            U_support=U_support,
            calibration_systems=systems,
            sigma_y=sigma_y,
            alpha=alpha,
            noise_replicas=noise_replicas,
            rng=np.random.default_rng(seed + 50_000 + a2),
            margin=margin,
            u_grid=grid,
        )
    seq_second = min(cond_second_scores, key=lambda a2: (cond_second_scores[a2], a2))
    seq_pair = tuple(sorted((seq_first, seq_second)))
    seq_key = ",".join(map(str, seq_pair))
    if seq_key not in pair_scores:
        pair_scores[seq_key] = cond_second_scores[seq_second]
    seq_obj = pair_scores[seq_key]

    sample_vals = np.asarray(list(pair_scores.values()), dtype=np.float64)
    return {
        "n_systems_scored": len(systems),
        "noise_replicas": noise_replicas,
        "best_single": {
            "action": int(best_single),
            "mean_u_ctrl": float(single_scores[best_single]),
        },
        "top_singles": [
            {"action": int(x), "mean_u_ctrl": float(single_scores[x])} for x in top_singles
        ],
        "fixed_subset": {
            "actions": list(fixed_t),
            "mean_u_ctrl": float(fixed_obj),
            "member_single_scores": {str(x): float(single_scores[x]) for x in fixed_t},
            "better_member_single": float(better_single_in_fixed),
            "joint_gain_vs_better_single": float(joint_gain),
        },
        "naive_top2_singles_pair": {
            "actions": list(naive_pair),
            "mean_u_ctrl": float(naive_obj),
        },
        "greedy_sequential_pair": {
            "actions": list(seq_pair),
            "mean_u_ctrl": float(seq_obj),
        },
        "fixed_vs_naive_top2": float(naive_obj - fixed_obj),
        "fixed_vs_greedy_sequential": float(seq_obj - fixed_obj),
        "pair_score_sample": {
            "n_pairs": len(pair_scores),
            "mean": float(np.mean(sample_vals)),
            "std": float(np.std(sample_vals)),
            "fixed_percentile": float(100.0 * np.mean(sample_vals >= fixed_obj)),
        },
        "complementarity_present": bool(joint_gain > 1e-3 or (naive_obj - fixed_obj) > 1e-3),
    }


def diagnose_implementation_consistency(
    *,
    frozen,
    fixed_meta: dict[str, Any],
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    systems: list[dict[str, Any]],
    sigma_y: float,
    fixed_subset: list[int],
    seed: int,
) -> dict[str, Any]:
    issues: list[str] = []
    rule = frozen.metadata()
    nested = fixed_meta.get("terminal_rule") or fixed_meta
    for k in ("terminal_rule_hash", "quantile_level", "additive_margin", "control_grid_hash"):
        if nested.get(k) != rule[k]:
            issues.append(f"fixed meta mismatch on {k}: {nested.get(k)} vs {rule[k]}")

    w0 = normalize_log_weights(table_support.log_p0)
    u_shared = posterior_to_u_ctrl(w0, U_support, frozen)
    u_direct = posterior_safe_u_ctrl(
        U_support, w0, frozen.alpha, margin=frozen.margin, u_grid=frozen.u_candidates
    )
    if abs(u_shared - u_direct) > 1e-12:
        issues.append(
            f"posterior_to_u_ctrl vs posterior_safe_u_ctrl disagree: {u_shared} vs {u_direct}"
        )

    obj = estimate_fixed_subset_objective(
        fixed_subset,
        table_support=table_support,
        U_support=U_support,
        calibration_systems=systems,
        sigma_y=sigma_y,
        alpha=frozen.alpha,
        noise_replicas=4,
        rng=np.random.default_rng(seed),
        margin=frozen.margin,
        u_grid=frozen.u_candidates,
    )
    return {
        "shared_rule": rule,
        "prior_u_ctrl": float(u_shared),
        "fixed_subset_recomputed_mean_u": float(obj),
        "issues": issues,
        "consistent": len(issues) == 0,
        "expected_algorithmic_difference": (
            "Myopic optimizes one-step E[u_ctrl]; Fixed optimizes unordered size-T subset. "
            "Disagreement is expected when probes are complementary; not an implementation bug."
        ),
        "n_support_particles": int(len(table_support.systems)),
        "U_bank_size": int(len(U_support)),
        "sigma_y": float(sigma_y),
    }


def _recommendation(verdicts: dict[str, Any], paired: dict[str, Any]) -> str:
    del paired
    if verdicts["5_implementation_inconsistency"]["answer"]:
        return (
            "Fix implementation inconsistencies before the full sweep; "
            "do not interpret Myopic vs Fixed scientifically yet."
        )
    parts: list[str] = []
    if verdicts["1_statistically_tied"]["answer"]:
        parts.append(
            "Treat Myopic and Fixed as statistically tied at T=2 on this pilot "
            "(95% CI for paired difference contains 0)."
        )
    else:
        parts.append(
            "Paired CI excludes 0; difference is statistically detectable but still "
            "small — interpret mechanism before claiming scientific superiority."
        )
    if verdicts["2_excessive_myopic_mc_error"]["answer"]:
        parts.append(
            "Increase myopic_hypothetical before the sweep; current MC error may "
            "blur Myopic decisions."
        )
    if verdicts["3_many_quantized_ties"]["answer"]:
        parts.append(
            "Quantized u_ctrl ties are common; Myopic often breaks ties by action index, "
            "which can make it behave like a near-fixed rule."
        )
    if verdicts["4_fixed_complementarity"]["answer"]:
        parts.append(
            "Fixed benefits from joint probe complementarity that one-step Myopic "
            "cannot plan; expect Fixed ≥ Myopic at small T when pairs matter."
        )
    else:
        parts.append(
            "Little joint complementarity detected on train scoring; Myopic≈Fixed "
            "is consistent with weak pair interactions at T=2."
        )
    parts.append("Do not start the full IEEE5/9/14 sweep until this diagnosis is accepted.")
    return " ".join(parts)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    v = report["verdicts"]
    p = report["paired_comparison"]
    diff = p["myopic_minus_fixed"]
    lines = [
        "# Myopic vs Fixed diagnosis (IEEE5 T=2)",
        "",
        f"Seed `{report['seed']}`, paired rollouts `{report['evaluation_rollouts']}`.",
        f"Frozen rule hash `{report['frozen_terminal_rule']['terminal_rule_hash']}`.",
        f"Fixed subset `{report['fixed_subset']}`.",
        f"Myopic `n_hypothetical={report['myopic_n_hypothetical']}`.",
        "",
        "## Verdicts",
        "",
        f"1. Statistically tied? **{v['1_statistically_tied']['answer']}** — {v['1_statistically_tied']['detail']}",
        f"2. Excessive Myopic MC error? **{v['2_excessive_myopic_mc_error']['answer']}** — {v['2_excessive_myopic_mc_error']['detail']}",
        f"3. Many quantized ties? **{v['3_many_quantized_ties']['answer']}** — {v['3_many_quantized_ties']['detail']}",
        f"4. Fixed complementarity? **{v['4_fixed_complementarity']['answer']}** — {v['4_fixed_complementarity']['detail']}",
        f"5. Implementation inconsistency? **{v['5_implementation_inconsistency']['answer']}** — {v['5_implementation_inconsistency']['detail']}",
        "",
        "## Paired comparison (myopic − fixed)",
        "",
        f"- mean myopic u_ctrl: `{p['myopic']['mean_u_ctrl']:.4f}`",
        f"- mean fixed u_ctrl: `{p['fixed']['mean_u_ctrl']:.4f}`",
        f"- mean paired diff: `{diff['mean_paired_diff']:.4f}`",
        f"- 95% bootstrap CI: `[{diff['ci95_low']:.4f}, {diff['ci95_high']:.4f}]`",
        f"- fraction tied: `{diff['fraction_tied']:.3f}`",
        f"- fraction identical sequences: `{p['fraction_identical_sequences']:.3f}`",
        "",
        "## Recommendation",
        "",
        report["recommendation"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_diagnose_myopic_fixed(
    exp_dir: Path | str,
    *,
    project_root: Path | None = None,
    evaluation_rollouts: int = 2000,
    seed: int = 3579,
) -> dict[str, Any]:
    from src.config import repo_root

    root = project_root or repo_root()
    run = load_experiment_run(Path(exp_dir), root)
    exp_dir = run.exp_dir
    out_dir = exp_dir / "diagnostics" / "myopic_vs_fixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen = load_frozen_terminal_rule(exp_dir)
    base_spec = ControlSpec.from_cfg(run.cfg)
    control_spec = frozen.to_control_spec(base_spec)
    splits = load_pilot_splits(exp_dir, run)
    catalog = build_catalog(run.cfg)
    n_actions = len(catalog)
    horizon = int(run.cfg.step_number)
    sigma_y = float(run.cfg.sigma_y)
    fixed_subset = _load_fixed_subset(exp_dir)

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

    selector_kwargs = dict(
        table_support=table_support,
        U_support=U_support,
        n_actions=n_actions,
        sigma_y=sigma_y,
        alpha=frozen.alpha,
        n_hypothetical=int(control_spec.myopic_hypothetical),
        safety_margin=frozen.margin,
        u_candidates=frozen.u_candidates,
    )

    print(f"=== diagnose-myopic-fixed  T={horizon}  n_actions={n_actions} ===")
    print(f"  frozen rule hash={frozen.terminal_rule_hash}  fixed_subset={fixed_subset}")
    print(f"  evaluation_rollouts={evaluation_rollouts}  seed={seed}")

    print("\n[1/5] Paired Myopic vs Fixed evaluation...")
    rows_m, sum_m = evaluate_method_paired(
        method="myopic",
        selector_factory=lambda: MyopicControlSelector(**selector_kwargs),
        test_systems=splits["test_systems"],
        table_support=table_support,
        U_support=U_support,
        frozen=frozen,
        control_spec=control_spec,
        control_engine=engine,
        horizon=horizon,
        n_actions=n_actions,
        sigma_y=sigma_y,
        n_rollouts=evaluation_rollouts,
        global_seed=seed,
        method_seed=seed,
    )
    rows_f, sum_f = evaluate_method_paired(
        method="fixed",
        selector_factory=lambda: FixedSelector(sequence=list(sorted(fixed_subset))),
        test_systems=splits["test_systems"],
        table_support=table_support,
        U_support=U_support,
        frozen=frozen,
        control_spec=control_spec,
        control_engine=engine,
        horizon=horizon,
        n_actions=n_actions,
        sigma_y=sigma_y,
        n_rollouts=evaluation_rollouts,
        global_seed=seed,
        method_seed=seed,
    )
    u_m = np.asarray([r["u_ctrl"] for r in rows_m], dtype=np.float64)
    u_f = np.asarray([r["u_ctrl"] for r in rows_f], dtype=np.float64)
    paired = paired_diff_stats(u_m, u_f, n_boot=4000, seed=seed)
    statistically_tied = bool(paired["ci95_low"] <= 0.0 <= paired["ci95_high"])
    myopic_pairs = Counter(
        tuple(sorted(r["sequence"])) for r in rows_m if len(r["sequence"]) >= 2
    )
    same_seq = float(
        np.mean(
            [
                tuple(rows_m[i]["sequence"]) == tuple(rows_f[i]["sequence"])
                for i in range(evaluation_rollouts)
            ]
        )
    )
    paired_block = {
        "n": evaluation_rollouts,
        "myopic": {k: sum_m[k] for k in rich_metrics([]).keys() if k in sum_m},
        "fixed": {k: sum_f[k] for k in rich_metrics([]).keys() if k in sum_f},
        "myopic_minus_fixed": paired,
        "statistically_tied_at_95": statistically_tied,
        "pilot_diff_reference": 0.004,
        "fraction_identical_sequences": same_seq,
        "myopic_top_unordered_pairs": [
            {"pair": list(p), "count": c} for p, c in myopic_pairs.most_common(8)
        ],
        "fixed_pair_frequency_under_myopic": int(
            myopic_pairs.get(tuple(sorted(fixed_subset)), 0)
        ),
    }
    print(
        f"  myopic={sum_m['mean_u_ctrl']:.4f}  fixed={sum_f['mean_u_ctrl']:.4f}  "
        f"diff={paired['mean_paired_diff']:.4f}  "
        f"CI95=[{paired['ci95_low']:.4f},{paired['ci95_high']:.4f}]  "
        f"tied={statistically_tied}"
    )

    print("\n[2/5] Myopic Monte Carlo accuracy...")
    mc = diagnose_myopic_mc_accuracy(
        selector_template=selector_kwargs,
        n_actions=n_actions,
        n_reps=48,
        seed=seed + 11,
        n_h_list=[8, 16, 32, 64, 128, 256],
    )
    print(
        f"  configured n_h={mc['configured_n_hypothetical']}  "
        f"excessive_mc={mc['verdict_hints']['excessive_mc_error']}"
    )

    print("\n[3/5] Quantized-objective ties...")
    ties = diagnose_quantized_ties(
        selector=MyopicControlSelector(**selector_kwargs),
        n_contexts=200,
        seed=seed + 22,
        systems=splits["validation_systems"] or splits["train_systems"],
        sigma_y=sigma_y,
        global_seed=seed,
    )
    print(f"  many_quantized_ties={ties['many_quantized_ties']}")

    print("\n[4/5] Probe complementarity (train scoring only)...")
    comp_systems = splits["train_systems"][: min(32, len(splits["train_systems"]))]
    comp = diagnose_complementarity(
        table_support=table_support,
        U_support=U_support,
        systems=comp_systems,
        sigma_y=sigma_y,
        frozen=frozen,
        n_actions=n_actions,
        fixed_subset=fixed_subset,
        noise_replicas=int(control_spec.fixed_noise_replicas),
        seed=seed + 33,
    )
    print(
        f"  complementarity={comp['complementarity_present']}  "
        f"joint_gain={comp['fixed_subset']['joint_gain_vs_better_single']:.4f}"
    )

    print("\n[5/5] Implementation consistency...")
    consistency = diagnose_implementation_consistency(
        frozen=frozen,
        fixed_meta=json.loads(
            (exp_dir / "eval" / "fixed" / "subset_meta.json").read_text(encoding="utf-8")
        ),
        table_support=table_support,
        U_support=U_support,
        systems=comp_systems,
        sigma_y=sigma_y,
        fixed_subset=fixed_subset,
        seed=seed + 44,
    )
    print(f"  consistent={consistency['consistent']}")

    verdicts = {
        "1_statistically_tied": {
            "answer": bool(statistically_tied),
            "detail": (
                f"myopic−fixed mean={paired['mean_paired_diff']:.4f}, "
                f"95% CI [{paired['ci95_low']:.4f}, {paired['ci95_high']:.4f}]"
            ),
        },
        "2_excessive_myopic_mc_error": {
            "answer": bool(mc["verdict_hints"]["excessive_mc_error"]),
            "detail": mc["verdict_hints"]["reason"],
        },
        "3_many_quantized_ties": {
            "answer": bool(ties["many_quantized_ties"]),
            "detail": (
                f"step0 exact-tie rate={ties['step0_exact_ties']['frac_positive']:.3f}, "
                f"frac gap<1e-3={ties['step0_gap']['frac_gap_lt_1e-3']:.3f}"
            ),
        },
        "4_fixed_complementarity": {
            "answer": bool(comp["complementarity_present"]),
            "detail": (
                f"joint_gain_vs_better_single="
                f"{comp['fixed_subset']['joint_gain_vs_better_single']:.4f}, "
                f"fixed_vs_naive_top2={comp['fixed_vs_naive_top2']:.4f}"
            ),
        },
        "5_implementation_inconsistency": {
            "answer": not bool(consistency["consistent"]),
            "detail": (
                "none" if consistency["consistent"] else "; ".join(consistency["issues"])
            ),
        },
    }

    report = {
        "experiment": str(exp_dir),
        "seed": seed,
        "evaluation_rollouts": evaluation_rollouts,
        "frozen_terminal_rule": frozen.metadata(),
        "fixed_subset": fixed_subset,
        "myopic_n_hypothetical": int(control_spec.myopic_hypothetical),
        "paired_comparison": paired_block,
        "myopic_mc_accuracy": mc,
        "quantized_ties": ties,
        "complementarity": comp,
        "implementation_consistency": consistency,
        "verdicts": verdicts,
        "recommendation": _recommendation(verdicts, paired_block),
    }
    _write_json(out_dir / "diagnosis.json", report)
    _write_markdown(out_dir / "diagnosis_report.md", report)

    with (out_dir / "paired_rollouts.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rollout_id",
                "theta_test_id",
                "u_myopic",
                "u_fixed",
                "diff_m_minus_f",
                "seq_myopic",
                "seq_fixed",
                "safe_myopic",
                "safe_fixed",
            ],
        )
        w.writeheader()
        for i in range(evaluation_rollouts):
            w.writerow(
                {
                    "rollout_id": i,
                    "theta_test_id": rows_m[i]["theta_test_id"],
                    "u_myopic": rows_m[i]["u_ctrl"],
                    "u_fixed": rows_f[i]["u_ctrl"],
                    "diff_m_minus_f": rows_m[i]["u_ctrl"] - rows_f[i]["u_ctrl"],
                    "seq_myopic": " ".join(map(str, rows_m[i]["sequence"])),
                    "seq_fixed": " ".join(map(str, rows_f[i]["sequence"])),
                    "safe_myopic": rows_m[i]["safe_total"],
                    "safe_fixed": rows_f[i]["safe_total"],
                }
            )

    print(f"\nDiagnosis written → {out_dir}")
    for k, vv in verdicts.items():
        print(f"  {k}: {vv['answer']}  ({vv['detail']})")
    print(f"Recommendation: {report['recommendation']}")
    return report
