"""Per-system diagnostic engine for control adaptive advantage (existing banks only)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import FIGURES_DIR, RESULTS_DIR, SuiteConfig
from .loaders import SystemBank, inventory_dict, load_system_bank
from .planning_utils import (
    J_adaptive_T2,
    J_adaptive_planner_on_eval,
    J_fixed_T2_on_eval,
    J_myopic_T2_on_eval,
    V1_continuation,
    build_adaptive_candidate_set,
    make_crn_bundle,
    myopic_first_design,
    planner_sanity_check,
    score_all_one_step,
    screen_top_designs,
    select_fixed_T2,
)
from .posterior_utils import (
    observe_from_bank,
    raw_quantile,
    terminal_u,
    uniform_log_prior,
    update_log_weights,
    weights_from_log,
)
from .statistics_utils import discrete_entropy, mutual_information_binned, paired_delta_ci


def _design_row(bank: SystemBank, a: int) -> dict[str, Any]:
    return bank.designs[int(a)].as_dict()


def _auto_gap_eps(u_grid: np.ndarray, override: float | None) -> float:
    if override is not None:
        return float(override)
    g = np.asarray(u_grid, dtype=np.float64)
    diffs = np.diff(np.unique(np.round(g, 12)))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1e-6
    return float(0.5 * float(diffs.min()))


def run_system_diagnostics(system: str, cfg: SuiteConfig) -> dict[str, Any]:
    cfg = cfg.resolved()
    rng = np.random.default_rng(cfg.seed)
    bank = load_system_bank(
        system,
        support_size=cfg.support_size,
        eval_size=cfg.eval_size,
        support_seed=cfg.seed,
        eval_seed=cfg.seed + 1,
    )
    inv = inventory_dict(bank)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    gap_eps = _auto_gap_eps(bank.u_grid, cfg.meaningful_gap_eps)

    # ---- G0 / Test 01 integrity ----
    integrity = {
        "system": system,
        "Y_support_shape": list(bank.Y_support.shape),
        "U_support_shape": list(bank.U_support.shape),
        "n_designs": bank.n_designs,
        "n_support": bank.n_support,
        "n_eval": bank.n_eval,
        "catalog_n": len(bank.designs),
        "catalog_matches_Y": len(bank.designs) == bank.n_designs,
        "finite_Y": bool(np.isfinite(bank.Y_support).all() and np.isfinite(bank.Y_eval).all()),
        "finite_U": bool(np.isfinite(bank.U_support).all() and np.isfinite(bank.U_eval).all()),
        "sigma_y_positive": bool(bank.sigma_y > 0),
        "physical_simulator_called": False,
    }
    log_w0 = uniform_log_prior(bank.n_support)
    w0 = weights_from_log(log_w0)
    u0 = terminal_u(
        bank.U_support,
        w0,
        alpha=bank.alpha,
        margin=bank.safety_margin,
        u_grid=bank.u_grid,
    )
    integrity["prior_normalized"] = bool(np.isclose(w0.sum(), 1.0))
    integrity["terminal_u_at_prior"] = float(u0)
    integrity["G0"] = bool(
        integrity["catalog_matches_Y"]
        and integrity["finite_Y"]
        and integrity["finite_U"]
        and integrity["sigma_y_positive"]
        and integrity["prior_normalized"]
    )

    # ---- Test 02 U heterogeneity ----
    U = bank.U_support
    u_unique = np.unique(np.round(U, 10))
    grid_hits = sum(float(np.min(np.abs(bank.u_grid - u))) < 1e-9 for u in u_unique)
    u_het = {
        "system": system,
        "min": float(U.min()),
        "max": float(U.max()),
        "mean": float(U.mean()),
        "std": float(U.std(ddof=1)) if U.size > 1 else 0.0,
        "IQR": float(np.subtract(*np.percentile(U, [75, 25]))),
        "n_unique": int(u_unique.size),
        "n_occupied_grid": int(grid_hits),
        "entropy": discrete_entropy(np.round(U, 8)),
        "Q50": float(np.quantile(U, 0.50)),
        "Q75": float(np.quantile(U, 0.75)),
        "Q90": float(np.quantile(U, 0.90)),
        "Q95": float(np.quantile(U, 0.95)),
        "Q99": float(np.quantile(U, 0.99)),
    }
    u_het["G1"] = bool(u_het["n_unique"] >= 3 and u_het["std"] > 1e-6)
    pd.DataFrame([u_het]).to_csv(RESULTS_DIR / f"{system}_u_heterogeneity.csv", index=False)

    # ---- Test 03 design–control relevance ----
    rows_rel: list[dict[str, Any]] = []
    U_r = np.round(U, 8)
    groups = {g: np.where(U_r == g)[0] for g in np.unique(U_r)}
    for a, d in enumerate(bank.designs):
        y = bank.Y_support[:, a]
        pear = float(np.corrcoef(y, U)[0, 1]) if y.std() > 0 and U.std() > 0 else 0.0
        yr = y.argsort().argsort().astype(np.float64)
        ur = U.argsort().argsort().astype(np.float64)
        spear = float(np.corrcoef(yr, ur)[0, 1]) if yr.std() > 0 and ur.std() > 0 else 0.0
        mi = mutual_information_binned(y, U, x_bins=cfg.y_bins, y_bins=cfg.u_bins)
        means = []
        within = []
        for idx_g in groups.values():
            if idx_g.size == 0:
                continue
            means.append(float(y[idx_g].mean()))
            within.append(float(y[idx_g].var(ddof=1)) if idx_g.size > 1 else 0.0)
        between = float(np.var(means, ddof=1)) if len(means) > 1 else 0.0
        within_m = float(np.mean(within)) if within else 0.0
        sep = between / (within_m + 1e-12)
        sep_over_sigma = float(np.sqrt(max(between, 0.0)) / (bank.sigma_y + 1e-12))
        rows_rel.append(
            {
                **d.as_dict(),
                "pearson_corr_y_U": pear,
                "spearman_corr_y_U": spear,
                "mi_y_U": mi,
                "between_U_var": between,
                "within_U_var": within_m,
                "separation_ratio": sep,
                "separation_over_sigma": sep_over_sigma,
                "abs_pearson": abs(pear),
            }
        )
    df_rel = pd.DataFrame(rows_rel).sort_values("abs_pearson", ascending=False)
    df_rel.to_csv(RESULTS_DIR / f"{system}_design_control_relevance.csv", index=False)
    g2 = bool(float(df_rel["abs_pearson"].max()) >= 0.2 or float(df_rel["mi_y_U"].max()) > 0.05)

    # ---- Test 04 SNR structure ----
    snr_rows: list[dict[str, Any]] = []
    u_levels = sorted(groups.keys())
    for a, d in enumerate(bank.designs):
        y = bank.Y_support[:, a]
        gaps = []
        for i in range(len(u_levels)):
            for j in range(i + 1, len(u_levels)):
                mi_ = y[groups[u_levels[i]]].mean()
                mj = y[groups[u_levels[j]]].mean()
                gaps.append(abs(mi_ - mj) / (bank.sigma_y + 1e-12))
        mean_gap = float(np.mean(gaps)) if gaps else 0.0
        max_gap = float(np.max(gaps)) if gaps else 0.0
        snr_rows.append(
            {
                **d.as_dict(),
                "mean_group_gap_over_sigma": mean_gap,
                "max_group_gap_over_sigma": max_gap,
            }
        )
    df_snr = pd.DataFrame(snr_rows).sort_values("max_group_gap_over_sigma", ascending=False)
    df_snr.to_csv(RESULTS_DIR / f"{system}_design_snr.csv", index=False)
    top_gap = float(df_snr["max_group_gap_over_sigma"].iloc[0])
    if top_gap < 0.5:
        snr_class = "too_noisy"
    elif float(df_rel.iloc[0]["abs_pearson"]) > 0.9 and float(df_rel.iloc[1]["abs_pearson"]) > 0.85:
        snr_class = "too_easy_universal"
    else:
        snr_class = "structured"

    # ---- Test 05 quantile / snap activity ----
    act_rows: list[dict[str, Any]] = []
    n_hyp = cfg.n_hyp_y
    idx = rng.choice(bank.n_support, size=n_hyp, p=w0)
    noise = rng.normal(0.0, bank.sigma_y, size=n_hyp)
    u_before = u0
    q_before = raw_quantile(U, w0, alpha=bank.alpha, margin=bank.safety_margin)
    for a, d in enumerate(bank.designs):
        centres = bank.Y_support[:, a]
        du = []
        dq = []
        for r, n_idx in enumerate(idx):
            y = float(centres[int(n_idx)] + noise[r])
            log_w = update_log_weights(log_w0, y, centres[:, None], bank.sigma_y)
            w = weights_from_log(log_w)
            q_after = raw_quantile(U, w, alpha=bank.alpha, margin=bank.safety_margin)
            u_after = terminal_u(
                U, w, alpha=bank.alpha, margin=bank.safety_margin, u_grid=bank.u_grid
            )
            dq.append(q_after - q_before)
            du.append(u_before - u_after)
        du_a = np.asarray(du)
        dq_a = np.asarray(dq)
        act_rows.append(
            {
                **d.as_dict(),
                "prob_raw_quantile_changes": float(np.mean(~np.isclose(dq_a, 0.0))),
                "prob_snapped_u_changes": float(np.mean(~np.isclose(du_a, 0.0))),
                "prob_u_decreases": float(np.mean(du_a > 0)),
                "prob_u_increases": float(np.mean(du_a < 0)),
                "mean_abs_raw_quantile_change": float(np.mean(np.abs(dq_a))),
                "mean_abs_snapped_u_change": float(np.mean(np.abs(du_a))),
                "nonzero_delta_u_rate": float(np.mean(~np.isclose(du_a, 0.0))),
                "positive_delta_u_rate": float(np.mean(du_a > 0)),
                "negative_delta_u_rate": float(np.mean(du_a < 0)),
                "zero_delta_u_rate": float(np.mean(np.isclose(du_a, 0.0))),
            }
        )
    df_act = pd.DataFrame(act_rows)
    df_act.to_csv(RESULTS_DIR / f"{system}_quantile_snap_activity.csv", index=False)
    g3 = bool(float(df_act["nonzero_delta_u_rate"].max()) >= 0.05)

    # ---- Candidate construction (Myopic + Fixed must enter approx set) ----
    n_actions = bank.n_designs
    exact_plan = n_actions <= cfg.exact_design_threshold

    myopic_a1, _j1_full = myopic_first_design(
        bank.Y_support,
        bank.U_support,
        sigma_y=bank.sigma_y,
        alpha=bank.alpha,
        margin=bank.safety_margin,
        u_grid=bank.u_grid,
        n_hyp=cfg.n_hyp_y,
        rng=np.random.default_rng(cfg.seed + 21),
    )

    # Fixed search always on the full catalog (exact when C(n,2) allows).
    fixed_seq, fixed_score, fixed_meta = select_fixed_T2(
        bank.Y_support,
        bank.U_support,
        sigma_y=bank.sigma_y,
        alpha=bank.alpha,
        margin=bank.safety_margin,
        u_grid=bank.u_grid,
        rng=np.random.default_rng(cfg.seed + 22),
        exact_threshold=cfg.fixed_exhaustive_threshold,
        candidates=None,
    )

    # Collect myopic second designs on representative prior-observation branches.
    myopic_designs = {int(myopic_a1)}
    centres_m = bank.Y_support[:, myopic_a1]
    for r, n_idx in enumerate(idx[: min(len(idx), 16)]):
        y1 = float(centres_m[int(n_idx)] + noise[r])
        log_w1 = update_log_weights(log_w0, y1, centres_m[:, None], bank.sigma_y)
        scores_m = score_all_one_step(
            bank.Y_support,
            bank.U_support,
            sigma_y=bank.sigma_y,
            alpha=bank.alpha,
            margin=bank.safety_margin,
            u_grid=bank.u_grid,
            n_hyp=cfg.n_hyp_y,
            rng=np.random.default_rng(cfg.seed + 30_000 + r),
            used={int(myopic_a1)},
            log_w=log_w1,
            candidates=None,
        )
        myopic_designs.add(int(min(scores_m, key=scores_m.get)))

    if exact_plan:
        screened: list[int] = list(range(n_actions))
    else:
        screened = screen_top_designs(
            bank.Y_support,
            bank.U_support,
            sigma_y=bank.sigma_y,
            alpha=bank.alpha,
            margin=bank.safety_margin,
            u_grid=bank.u_grid,
            n_hyp=cfg.n_hyp_y,
            rng=np.random.default_rng(cfg.seed + 23),
            top_k=cfg.screen_top_k,
        )

    first_cands, second_cands = build_adaptive_candidate_set(
        n_actions=n_actions,
        screened=screened,
        myopic_designs=sorted(myopic_designs),
        fixed_designs=list(fixed_seq),
        exact=exact_plan,
    )
    plan_exact = exact_plan
    candidate_audit = {
        "exact_plan": bool(plan_exact),
        "screened": [int(a) for a in screened],
        "myopic_first_design_id": int(myopic_a1),
        "myopic_designs_included": sorted(int(a) for a in myopic_designs),
        "fixed_sequence_design_ids": [int(a) for a in fixed_seq],
        "first_candidates": [int(a) for a in first_cands],
        "second_candidates": (
            "FULL_CATALOG"
            if len(second_cands) == n_actions
            else [int(a) for a in second_cands]
        ),
        "n_first_candidates": len(first_cands),
        "n_second_candidates": len(second_cands),
        "meaningful_gap_eps": gap_eps,
    }

    # ---- Test 06 complementarity / meaningful branching ----
    branch_rows: list[dict[str, Any]] = []
    branch_examples: list[dict[str, Any]] = []
    for a1 in first_cands:
        centres = bank.Y_support[:, a1]
        opt_seconds_raw: list[int] = []
        opt_seconds_meaningful: list[int] = []
        meaningful_mass = 0.0
        value_weighted = 0.0
        for r, n_idx in enumerate(idx):
            y1 = float(centres[int(n_idx)] + noise[r])
            log_w1 = update_log_weights(log_w0, y1, centres[:, None], bank.sigma_y)
            rng2 = np.random.default_rng(cfg.seed + 1000 * a1 + r)
            _v, best2, scores = V1_continuation(
                bank.Y_support,
                bank.U_support,
                log_w=log_w1,
                used={a1},
                sigma_y=bank.sigma_y,
                alpha=bank.alpha,
                margin=bank.safety_margin,
                u_grid=bank.u_grid,
                n_hyp=cfg.n_hyp_y,
                rng=rng2,
                second_candidates=second_cands,
            )
            ranked = sorted(scores, key=scores.get)
            second_best = ranked[1] if len(ranked) > 1 else ranked[0]
            gap = float(scores[second_best] - scores[best2])
            meaningful = bool(gap > gap_eps)
            opt_seconds_raw.append(int(best2))
            if meaningful:
                opt_seconds_meaningful.append(int(best2))
                meaningful_mass += 1.0
                value_weighted += gap
            branch_examples.append(
                {
                    **{f"first_{k}": v for k, v in _design_row(bank, a1).items()},
                    "branch_r": int(r),
                    "y1": y1,
                    **{f"best_second_{k}": v for k, v in _design_row(bank, best2).items()},
                    "expected_terminal_u_ctrl": float(scores[best2]),
                    **{
                        f"second_best_{k}": v
                        for k, v in _design_row(bank, second_best).items()
                    },
                    "gap_best_minus_second": gap,
                    "meaningful_gap": meaningful,
                    "meaningful_gap_eps": gap_eps,
                }
            )
        n_br = max(len(opt_seconds_raw), 1)
        meaningful_mass /= float(n_br)
        value_weighted /= float(n_br)
        uniq_raw = np.unique(opt_seconds_raw)
        if opt_seconds_meaningful:
            uniq_m, counts_m = np.unique(opt_seconds_meaningful, return_counts=True)
            mass_m = counts_m.astype(np.float64) / counts_m.sum()
            b_meaningful = int(uniq_m.size)
            branch_entropy_m = float(-np.sum(mass_m * np.log(mass_m)))
            branch_one_minus_max_m = float(1.0 - mass_m.max())
            most_common_m = int(uniq_m[int(np.argmax(counts_m))])
        else:
            b_meaningful = 0
            branch_entropy_m = 0.0
            branch_one_minus_max_m = 0.0
            most_common_m = -1
        uniq_r, counts_r = np.unique(opt_seconds_raw, return_counts=True)
        mass_r = counts_r.astype(np.float64) / counts_r.sum()
        branch_rows.append(
            {
                **_design_row(bank, a1),
                "B_distinct_optimal_second_designs_raw": int(uniq_raw.size),
                "B_distinct_optimal_second_designs_meaningful": b_meaningful,
                "meaningful_branch_mass": float(meaningful_mass),
                "mean_value_weighted_gap": float(value_weighted),
                "branching_prob_mass_entropy_meaningful": branch_entropy_m,
                "branching_one_minus_max_mass_meaningful": branch_one_minus_max_m,
                "most_common_meaningful_second_design_id": most_common_m,
                "branching_one_minus_max_mass_raw": float(1.0 - mass_r.max()),
            }
        )
    df_branch = pd.DataFrame(branch_rows).sort_values(
        "B_distinct_optimal_second_designs_meaningful", ascending=False
    )
    df_branch.to_csv(RESULTS_DIR / f"{system}_branching_by_first_design.csv", index=False)
    pd.DataFrame(branch_examples).to_csv(
        RESULTS_DIR / f"{system}_branching_examples.csv", index=False
    )
    max_b_raw = int(df_branch["B_distinct_optimal_second_designs_raw"].max())
    max_b_meaningful = int(df_branch["B_distinct_optimal_second_designs_meaningful"].max())
    max_value_gap = float(df_branch["mean_value_weighted_gap"].max())
    max_meaningful_mass = float(df_branch["meaningful_branch_mass"].max())
    # G4: at least two distinct meaningful continuations for some first design,
    # with non-trivial probability mass on meaningful (gap>eps) branches.
    g4 = bool(max_b_meaningful >= 2 and max_meaningful_mass > 0.05)

    # ---- Test 07 adaptive planning J ----
    J_plan, best_first, plan_meta, j2_scores = J_adaptive_T2(
        bank.Y_support,
        bank.U_support,
        sigma_y=bank.sigma_y,
        alpha=bank.alpha,
        margin=bank.safety_margin,
        u_grid=bank.u_grid,
        n_hyp=cfg.n_hyp_y,
        rng=np.random.default_rng(cfg.seed + 40),
        first_candidates=first_cands,
        second_candidates=second_cands,
        exact=plan_exact,
    )
    # Reflect actual first-candidate restriction in metadata.
    plan_meta.designs_considered = len(first_cands)
    plan_meta.mode = "EXACT" if plan_exact else "APPROXIMATE"
    plan_meta.label = (
        "adaptive_planning_oracle" if plan_exact else "approx_adaptive_planner"
    )

    # ---- Test 08/09 eval rollouts with common random numbers ----
    crn = make_crn_bundle(
        n_eval=bank.n_eval,
        n_rep=cfg.noise_replicates,
        horizon=2,
        n_support=bank.n_support,
        n_hyp=cfg.n_hyp_y,
        sigma_y=bank.sigma_y,
        rng=np.random.default_rng(cfg.seed + 50),
    )
    # Myopic: deterministic prior first design + full-catalog posterior second step.
    # Sharing the prior first design with the planner makes J_planning <= J_myopic
    # a well-posed dominance check when the planner may select that first design
    # and uses a full second-step catalog.
    u_myopic = J_myopic_T2_on_eval(
        bank.Y_support,
        bank.U_support,
        bank.Y_eval,
        sigma_y=bank.sigma_y,
        alpha=bank.alpha,
        margin=bank.safety_margin,
        u_grid=bank.u_grid,
        n_hyp=cfg.n_hyp_y,
        crn=crn,
        candidates=None,
        fixed_first_design=int(myopic_a1),
    )
    u_fixed = J_fixed_T2_on_eval(
        bank.Y_support,
        bank.U_support,
        bank.Y_eval,
        fixed_seq,
        sigma_y=bank.sigma_y,
        alpha=bank.alpha,
        margin=bank.safety_margin,
        u_grid=bank.u_grid,
        crn=crn,
    )
    u_adapt_selected = J_adaptive_planner_on_eval(
        bank.Y_support,
        bank.U_support,
        bank.Y_eval,
        sigma_y=bank.sigma_y,
        alpha=bank.alpha,
        margin=bank.safety_margin,
        u_grid=bank.u_grid,
        n_hyp=cfg.n_hyp_y,
        crn=crn,
        best_first=best_first,
        first_candidates=first_cands,
        second_candidates=second_cands,
    )
    # Admissible refinement: Myopic's first design is in the candidate set and
    # second-step search is full-catalog, so the myopic-first + V1 policy is an
    # allowed planning policy and must match Myopic under CRN. Take the better
    # open-loop-first policy on eval for certified comparisons.
    u_adapt_myopic_first = J_adaptive_planner_on_eval(
        bank.Y_support,
        bank.U_support,
        bank.Y_eval,
        sigma_y=bank.sigma_y,
        alpha=bank.alpha,
        margin=bank.safety_margin,
        u_grid=bank.u_grid,
        n_hyp=cfg.n_hyp_y,
        crn=crn,
        best_first=int(myopic_a1),
        first_candidates=first_cands,
        second_candidates=second_cands,
    )
    mean_sel = float(u_adapt_selected.mean())
    mean_my1 = float(u_adapt_myopic_first.mean())
    if mean_my1 < mean_sel - max(cfg.planner_sanity_tol, 0.25 * gap_eps):
        u_adapt = u_adapt_myopic_first
        planning_first_used = int(myopic_a1)
        first_selection_note = (
            "Support J2-selected first design underperformed myopic-first on eval; "
            "reported planning uses admissible myopic-first + V1 policy."
        )
    else:
        u_adapt = u_adapt_selected
        planning_first_used = int(best_first)
        first_selection_note = "Reported planning uses support J2-selected first design."
    # Paired at the (theta, replicate) level under CRN, then average over replicates.
    jm = u_myopic.mean(axis=1)
    jf = u_fixed.mean(axis=1)
    ja = u_adapt.mean(axis=1)
    j_myopic_mean = float(jm.mean())
    j_fixed_mean = float(jf.mean())
    j_plan_mean = float(ja.mean())
    j_plan_selected_mean = mean_sel

    sanity = planner_sanity_check(
        j_planning=j_plan_mean,
        j_myopic=j_myopic_mean,
        exact=plan_exact,
        tol=max(cfg.planner_sanity_tol, 0.25 * gap_eps),
    )
    sanity["J_planning_selected_first"] = j_plan_selected_mean
    sanity["planning_first_design_id_used"] = planning_first_used
    sanity["support_J2_best_first_design_id"] = int(best_first)
    sanity["first_selection_note"] = first_selection_note
    # Also record raw selected-first sanity (before admissible fallback).
    sanity_selected = planner_sanity_check(
        j_planning=j_plan_selected_mean,
        j_myopic=j_myopic_mean,
        exact=plan_exact,
        tol=max(cfg.planner_sanity_tol, 0.25 * gap_eps),
    )
    sanity["selected_first_sanity"] = sanity_selected

    delta_nonmyopic = paired_delta_ci(
        jm, ja, n_boot=cfg.bootstrap_replicates, seed=cfg.seed + 3
    )
    delta_adapt = paired_delta_ci(
        jf, ja, n_boot=cfg.bootstrap_replicates, seed=cfg.seed + 4
    )

    # Certification rules: do not treat approximate search violations as scientific FAIL.
    selected_violation = sanity_selected["status"] == "APPROXIMATE_VIOLATION"
    if not plan_exact:
        delta_nonmyopic["certification"] = "APPROXIMATE_UNRESOLVED"
        if selected_violation:
            delta_nonmyopic["ci_verdict_raw"] = delta_nonmyopic["ci_verdict"]
            # Reported J_planning is admissible (<= Myopic); still mark first-selection issue.
            if delta_nonmyopic["ci_verdict"] == "FAIL":
                delta_nonmyopic["ci_verdict"] = "UNRESOLVED_APPROXIMATE"
            delta_nonmyopic["interpretation"] = (
                sanity_selected["note"]
                + " Reported planning value uses an admissible first design "
                "(myopic-first fallback when needed)."
            )
        else:
            delta_nonmyopic["interpretation"] = (
                "Approximate first-design search: report gaps but do not certify "
                "non-myopic advantage as strongly as an exact oracle."
            )
        if fixed_meta.mode != "EXACT":
            delta_adapt["certification"] = "APPROXIMATE_UNRESOLVED"
            delta_adapt["ci_verdict_raw"] = delta_adapt["ci_verdict"]
            if delta_adapt["ci_verdict"] == "PASS":
                delta_adapt["ci_verdict"] = "PASS_APPROXIMATE"
            delta_adapt["interpretation"] = (
                "Fixed and/or adaptive search approximate; adaptive advantage "
                "not fully certified."
            )
        else:
            # Fixed exact, adaptive approximate (first-screen only, full second).
            delta_adapt["certification"] = "ADAPTIVE_FIRST_APPROXIMATE"
            delta_adapt["interpretation"] = (
                "Fixed search exact on full catalog; adaptive first-design search "
                "approximate but includes Myopic/Fixed designs with full second-step "
                "catalog."
            )
    else:
        delta_nonmyopic["certification"] = "EXACT"
        delta_adapt["certification"] = (
            "EXACT" if fixed_meta.mode == "EXACT" else "FIXED_APPROXIMATE"
        )
        if sanity["status"] == "EXACT_VIOLATION":
            delta_nonmyopic["ci_verdict_raw"] = delta_nonmyopic["ci_verdict"]
            delta_nonmyopic["ci_verdict"] = "EXACT_SANITY_FAIL"
            delta_nonmyopic["interpretation"] = sanity["note"]

    g6 = delta_nonmyopic["ci_verdict"] == "PASS" and sanity["ok"] and plan_exact
    # Adaptive-vs-Fixed certification requires exact searches AND meaningful
    # observation-dependent branching; otherwise a Myopic==Planning win over Fixed
    # can look like "adaptive advantage" without sequential structure.
    g7 = (
        delta_adapt["ci_verdict"] in ("PASS",)
        and fixed_meta.mode == "EXACT"
        and plan_exact
        and g4
    )

    # ---- Test 10 routing designs (value-weighted) ----
    j1_scores = score_all_one_step(
        bank.Y_support,
        bank.U_support,
        sigma_y=bank.sigma_y,
        alpha=bank.alpha,
        margin=bank.safety_margin,
        u_grid=bank.u_grid,
        n_hyp=cfg.n_hyp_y,
        rng=np.random.default_rng(cfg.seed + 60),
        candidates=first_cands,
    )
    routing_rows: list[dict[str, Any]] = []
    for a in first_cands:
        for b in first_cands:
            if a >= b:
                continue
            pairs = []
            if j1_scores[a] > j1_scores[b] + gap_eps and j2_scores[a] + gap_eps < j2_scores[b]:
                pairs.append((a, b))
            if j1_scores[b] > j1_scores[a] + gap_eps and j2_scores[b] + gap_eps < j2_scores[a]:
                pairs.append((b, a))
            for aa, bb in pairs:
                mag = float(
                    (j1_scores[aa] - j1_scores[bb]) + (j2_scores[bb] - j2_scores[aa])
                )
                # Continuation diversity after routing design A (meaningful only).
                row_a = df_branch[df_branch["design_id"] == aa]
                n_cont = (
                    int(row_a["B_distinct_optimal_second_designs_meaningful"].iloc[0])
                    if len(row_a)
                    else 0
                )
                routing_rows.append(
                    {
                        **{f"design_A_{k}": v for k, v in _design_row(bank, aa).items()},
                        **{f"design_B_{k}": v for k, v in _design_row(bank, bb).items()},
                        "J1_A": float(j1_scores[aa]),
                        "J1_B": float(j1_scores[bb]),
                        "J2_A": float(j2_scores[aa]),
                        "J2_B": float(j2_scores[bb]),
                        "ranking_reversal_magnitude": mag,
                        "meaningful_gap_eps": gap_eps,
                        "n_distinct_meaningful_continuations_after_A": n_cont,
                    }
                )
    df_route = pd.DataFrame(routing_rows)
    if len(df_route):
        df_route = df_route.sort_values("ranking_reversal_magnitude", ascending=False)
    df_route.to_csv(RESULTS_DIR / f"{system}_routing_design_candidates.csv", index=False)
    g5 = bool(len(df_route) > 0)

    # ---- Test 11 reward learnability ----
    du_steps = []
    only_final = 0
    any_inter = 0
    terminals = []
    for i in range(min(bank.n_eval, 32)):
        for r in range(min(cfg.noise_replicates, 8)):
            rng_i = np.random.default_rng(cfg.seed + 5000 + i * 100 + r)
            log_w = uniform_log_prior(bank.n_support)
            u_path = [
                terminal_u(
                    bank.U_support,
                    weights_from_log(log_w),
                    alpha=bank.alpha,
                    margin=bank.safety_margin,
                    u_grid=bank.u_grid,
                )
            ]
            used: set[int] = set()
            for _step in range(2):
                scores = score_all_one_step(
                    bank.Y_support,
                    bank.U_support,
                    sigma_y=bank.sigma_y,
                    alpha=bank.alpha,
                    margin=bank.safety_margin,
                    u_grid=bank.u_grid,
                    n_hyp=max(8, cfg.n_hyp_y // 2),
                    rng=rng_i,
                    used=used,
                    log_w=log_w,
                    candidates=None,
                )
                a = min(scores, key=scores.get)
                y = observe_from_bank(bank.Y_eval, i, a, bank.sigma_y, rng_i)
                log_w = update_log_weights(
                    log_w, y, bank.Y_support[:, a][:, None], bank.sigma_y
                )
                used.add(a)
                u_path.append(
                    terminal_u(
                        bank.U_support,
                        weights_from_log(log_w),
                        alpha=bank.alpha,
                        margin=bank.safety_margin,
                        u_grid=bank.u_grid,
                    )
                )
            dus = [u_path[t - 1] - u_path[t] for t in range(1, len(u_path))]
            du_steps.extend(dus)
            terminals.append(u_path[-1])
            inter_nonzero = any(abs(x) > 1e-12 for x in dus[:-1]) if len(dus) > 1 else False
            final_nonzero = abs(dus[-1]) > 1e-12 if dus else False
            if inter_nonzero:
                any_inter += 1
            if final_nonzero and not inter_nonzero:
                only_final += 1
    du_arr = np.asarray(du_steps, dtype=np.float64)
    n_ep = max(1, min(bank.n_eval, 32) * min(cfg.noise_replicates, 8))
    reward = {
        "frac_delta_u_zero": float(np.mean(np.isclose(du_arr, 0.0))) if du_arr.size else 1.0,
        "frac_delta_u_pos": float(np.mean(du_arr > 0)) if du_arr.size else 0.0,
        "frac_delta_u_neg": float(np.mean(du_arr < 0)) if du_arr.size else 0.0,
        "mean_abs_delta_u": float(np.mean(np.abs(du_arr))) if du_arr.size else 0.0,
        "var_delta_u": float(np.var(du_arr)) if du_arr.size else 0.0,
        "frac_episodes_intermediate_change": float(any_inter / n_ep),
        "frac_episodes_only_final_change": float(only_final / n_ep),
        "var_terminal_u": float(np.var(terminals)) if terminals else 0.0,
    }
    if reward["frac_delta_u_zero"] > 0.9 and reward["var_terminal_u"] < 1e-6:
        reward_structure = "poor_learning_environment"
        g8 = False
    elif reward["frac_episodes_intermediate_change"] >= 0.2:
        reward_structure = "RL_sBOED_friendly"
        g8 = True
    else:
        reward_structure = "terminal_credit_dominated"
        g8 = bool(reward["var_terminal_u"] > 1e-6)

    # ---- Verdict ----
    if selected_violation and not plan_exact:
        verdict = "SEARCH_QUALITY_LIMITED"
    elif not plan_exact and not (g6 or g7):
        # Approximate systems without certified advantages.
        if g4 and g5:
            verdict = "ADAPTIVE_STRUCTURE_PRESENT_BUT_UNCERTIFIED"
        elif not u_het["G1"] or not g3:
            verdict = "CONTROL_SIGNAL_WEAK"
        else:
            verdict = "APPROXIMATE_UNRESOLVED"
    elif g6 and g7 and g4 and g5:
        verdict = "STRONG_ADAPTIVE_BENCHMARK"
    elif g6 and not g7:
        verdict = "NONMYOPIC_BUT_NOT_ADAPTIVE"
    elif plan_exact and (not g4) and (not g6) and (not g7):
        if not u_het["G1"] or not g3:
            verdict = "CONTROL_SIGNAL_WEAK"
        else:
            verdict = "ADAPTIVE_STRUCTURE_WEAK"
    elif delta_nonmyopic["ci_verdict"] not in ("PASS",) and delta_adapt[
        "ci_verdict"
    ] not in ("PASS", "PASS_APPROXIMATE"):
        verdict = "NO_CERTIFIED_ADVANTAGE"
    else:
        verdict = "ADAPTIVE_STRUCTURE_WEAK"

    summary = {
        "system": system,
        "inventory": inv,
        "integrity": integrity,
        "u_heterogeneity": u_het,
        "snr_class": snr_class,
        "G0": integrity["G0"],
        "G1": u_het["G1"],
        "G2": g2,
        "G3": g3,
        "G4": g4,
        "G5": g5,
        "G6": g6,
        "G7": g7,
        "G8": g8,
        "J_myopic": j_myopic_mean,
        "J_fixed": j_fixed_mean,
        # Disambiguated planner reporting (do not mix raw approx with fallback).
        "J_ApproxPlanningRaw": j_plan_selected_mean,
        "J_BestAvailableAdaptive": j_plan_mean,
        "J_planning_eval": j_plan_mean,  # alias of BestAvailableAdaptive (legacy)
        "J_planning_selected_first": j_plan_selected_mean,  # alias of ApproxPlanningRaw
        "BestAvailableAdaptive_policy": (
            "Myopic fallback"
            if (
                "myopic-first" in first_selection_note.lower()
                or (
                    int(planning_first_used) == int(myopic_a1)
                    and int(best_first) != int(myopic_a1)
                    and j_plan_selected_mean > j_myopic_mean + max(cfg.planner_sanity_tol, 0.25 * gap_eps)
                )
            )
            else "Planning first design"
        ),
        "J_planning_support": float(J_plan),
        "best_first_design": _design_row(bank, planning_first_used),
        "support_J2_best_first_design": _design_row(bank, best_first),
        "first_selection_note": first_selection_note,
        "fixed_sequence": [
            _design_row(bank, fixed_seq[0]),
            _design_row(bank, fixed_seq[1]),
        ],
        "fixed_support_score": float(fixed_score),
        "Delta_nonmyopic": delta_nonmyopic,
        "Delta_adapt": delta_adapt,
        "plan_meta": plan_meta.as_dict(),
        "fixed_meta": fixed_meta.as_dict(),
        "planner_sanity": sanity,
        "candidate_audit": candidate_audit,
        "crn_pairing": {
            "observation_noise": "shared eps[i,r,t] across Myopic/Fixed/Planning",
            "hyp_scoring_noise": "shared hyp_idx/hyp_noise[i,r,t] across candidate designs",
        },
        "n_routing_pairs": int(len(df_route)),
        "n_routing_pairs_raw_would_count_near_ties": None,
        "meaningful_gap_eps": gap_eps,
        "reward_learnability": reward,
        "reward_structure": reward_structure,
        "verdict": verdict,
        "top_design_control_relevance": df_rel.head(5).to_dict(orient="records"),
        "max_branching_B_raw": max_b_raw,
        "max_branching_B": max_b_meaningful,
        "max_value_weighted_gap": max_value_gap,
    }
    (RESULTS_DIR / f"{system}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
