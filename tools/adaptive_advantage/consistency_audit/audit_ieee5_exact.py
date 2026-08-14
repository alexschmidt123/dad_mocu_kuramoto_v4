"""Exact T=2 Fixed and Adaptive enumeration for IEEE-5 under shared CRN."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tools.adaptive_advantage.config import SuiteConfig
from tools.adaptive_advantage.loaders import SystemBank
from tools.adaptive_advantage.planning_utils import (
    J_myopic_T2_on_eval,
    myopic_first_design,
)
from tools.adaptive_advantage.posterior_utils import (
    terminal_u,
    uniform_log_prior,
    update_log_weights,
    weights_from_log,
)
from src.observations.likelihood import vector_gaussian_loglik

from .common import (
    AUDIT_RESULTS,
    centres,
    design_dict,
    gap_eps,
    make_audit_crn,
    observe,
    paired_ci,
    terminal_after_sequence,
)


def _score_all_seconds_fast(
    bank: SystemBank,
    log_w: np.ndarray,
    used_a1: int,
    *,
    hyp_idx_raw: np.ndarray,
    hyp_noise: np.ndarray,
) -> dict[int, float]:
    """Vectorized-ish E_y2[u] for all a2 != a1 under shared hyp CRN."""
    w = weights_from_log(log_w)
    cdf = np.cumsum(w)
    n = bank.n_support
    n_hyp = len(hyp_noise)
    # Inverse-CDF particle indices from shared raw stream.
    parts = []
    for r in range(n_hyp):
        u = (float(int(hyp_idx_raw[r]) % max(n, 1)) + 0.5) / float(max(n, 1))
        n_idx = int(np.searchsorted(cdf, min(u, 1.0 - 1e-12), side="left"))
        parts.append(min(max(n_idx, 0), n - 1))
    parts = np.asarray(parts, dtype=np.int64)

    scores: dict[int, float] = {}
    log_w0 = np.asarray(log_w, dtype=np.float64)
    for a in range(bank.n_designs):
        if a == used_a1:
            continue
        centres_a = bank.Y_support[:, a][:, None]
        vals = []
        for r in range(n_hyp):
            y = float(bank.Y_support[parts[r], a] + hyp_noise[r])
            log_w2 = log_w0 + vector_gaussian_loglik(
                np.asarray([y]), centres_a, bank.sigma_y
            )
            ww = weights_from_log(log_w2)
            vals.append(
                terminal_u(
                    bank.U_support,
                    ww,
                    alpha=bank.alpha,
                    margin=bank.safety_margin,
                    u_grid=bank.u_grid,
                )
            )
        scores[a] = float(np.mean(vals))
    return scores


def enumerate_fixed_ordered_pairs(
    bank: SystemBank,
    cfg: SuiteConfig,
    crn,
) -> dict[str, Any]:
    """Exact min over all ordered pairs (a1!=a2) on held-out eval + CRN."""
    print("[audit] enumerating all ordered Fixed pairs for ieee5...", flush=True)
    n_a = bank.n_designs
    best_mean = float("inf")
    best_pair: tuple[int, int] | None = None
    best_u: np.ndarray | None = None
    rows = []

    for a1 in range(n_a):
        for a2 in range(n_a):
            if a2 == a1:
                continue
            u = np.empty((bank.n_eval, cfg.noise_replicates), dtype=np.float64)
            for i in range(bank.n_eval):
                for r in range(cfg.noise_replicates):
                    u[i, r] = terminal_after_sequence(
                        bank, i, (a1, a2), crn.eps_obs[i, r]
                    )
            m = float(u.mean())
            rows.append(
                {
                    "a1": a1,
                    "a2": a2,
                    "J_mean": m,
                    **{f"first_{k}": v for k, v in design_dict(bank, a1).items()},
                    **{f"second_{k}": v for k, v in design_dict(bank, a2).items()},
                }
            )
            if m < best_mean:
                best_mean = m
                best_pair = (a1, a2)
                best_u = u
        if (a1 + 1) % 5 == 0:
            print(f"  fixed pairs: finished first-design {a1+1}/{n_a}", flush=True)

    assert best_pair is not None and best_u is not None
    df = pd.DataFrame(rows).sort_values("J_mean")
    df.to_csv(AUDIT_RESULTS / "ieee5_fixed_ordered_pairs.csv", index=False)

    # Best unordered via min of both orders
    best_un_mean = float("inf")
    best_un_order = best_pair
    best_un_key = tuple(sorted(best_pair))
    by_un: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {}
    for row in rows:
        key = tuple(sorted((int(row["a1"]), int(row["a2"]))))
        by_un.setdefault(key, []).append(((int(row["a1"]), int(row["a2"])), float(row["J_mean"])))
    for key, lst in by_un.items():
        order, m = min(lst, key=lambda t: t[1])
        if m < best_un_mean:
            best_un_mean = m
            best_un_order = order
            best_un_key = key

    return {
        "J_fixed_best_exact_ordered": best_mean,
        "best_ordered_sequence": {
            "first": design_dict(bank, best_pair[0]),
            "second": design_dict(bank, best_pair[1]),
            "pair": list(best_pair),
        },
        "J_fixed_best_exact_unordered_via_best_order": float(best_un_mean),
        "best_unordered_via_best_order": {
            "unordered_ids": list(best_un_key),
            "best_evaluation_order": list(best_un_order),
            "first": design_dict(bank, best_un_order[0]),
            "second": design_dict(bank, best_un_order[1]),
        },
        "top5_ordered": df.head(5).to_dict(orient="records"),
        "u_best_ordered_full": best_u,
    }


def enumerate_adaptive_exact(
    bank: SystemBank,
    cfg: SuiteConfig,
    crn,
) -> dict[str, Any]:
    """Exact adaptive T=2 over all first designs with CRN-matched evaluation."""
    print("[audit] enumerating exact Adaptive tree for ieee5...", flush=True)
    n_a = bank.n_designs
    eps = gap_eps(bank, cfg.meaningful_gap_eps)
    per_first: dict[int, dict[str, Any]] = {}
    best_first = None
    best_mean = float("inf")
    best_u = None
    branch_examples = []

    for a1 in range(n_a):
        u = np.empty((bank.n_eval, cfg.noise_replicates), dtype=np.float64)
        fixed_cont_sums = np.zeros(n_a, dtype=np.float64)
        fixed_cont_counts = 0
        v_best_sum = 0.0
        raw_seconds: list[int] = []
        meaningful_seconds: list[int] = []
        gaps: list[float] = []
        score_snapshots: list[dict[int, float]] = []

        for i in range(bank.n_eval):
            for r in range(cfg.noise_replicates):
                log_w = uniform_log_prior(bank.n_support)
                y1 = observe(bank.Y_eval, i, a1, crn.eps_obs[i, r, 0])
                log_w = update_log_weights(
                    log_w, y1, centres(bank.Y_support, a1), bank.sigma_y
                )
                scores = _score_all_seconds_fast(
                    bank,
                    log_w,
                    a1,
                    hyp_idx_raw=crn.hyp_idx[i, r, 1],
                    hyp_noise=crn.hyp_noise[i, r, 1],
                )
                ranked = sorted(scores, key=scores.get)
                a2_star = ranked[0]
                a2_second = ranked[1]
                gap = float(scores[a2_second] - scores[a2_star])
                gaps.append(gap)
                v_best_sum += float(scores[a2_star])
                raw_seconds.append(int(a2_star))
                if gap > eps:
                    meaningful_seconds.append(int(a2_star))
                for a2, val in scores.items():
                    fixed_cont_sums[a2] += val
                fixed_cont_counts += 1

                if len(score_snapshots) < 8:
                    if not score_snapshots or a2_star != min(
                        score_snapshots[-1], key=score_snapshots[-1].get
                    ):
                        score_snapshots.append(scores)

                y2 = observe(bank.Y_eval, i, a2_star, crn.eps_obs[i, r, 1])
                log_w2 = update_log_weights(
                    log_w, y2, centres(bank.Y_support, a2_star), bank.sigma_y
                )
                u[i, r] = terminal_u(
                    bank.U_support,
                    weights_from_log(log_w2),
                    alpha=bank.alpha,
                    margin=bank.safety_margin,
                    u_grid=bank.u_grid,
                )

                if i < 2 and r < 3:
                    branch_examples.append(
                        {
                            "first_design_id": a1,
                            "theta_i": i,
                            "rep_r": r,
                            "y1": y1,
                            "best_second_design_id": int(a2_star),
                            "second_best_design_id": int(a2_second),
                            "V_best": float(scores[a2_star]),
                            "V_second": float(scores[a2_second]),
                            "continuation_gap": gap,
                            "meaningful": bool(gap > eps),
                            "realized_u": float(u[i, r]),
                        }
                    )

        switch_penalties = []
        for ii in range(len(score_snapshots)):
            for jj in range(ii + 1, len(score_snapshots)):
                sb = score_snapshots[ii]
                sc = score_snapshots[jj]
                b = min(sb, key=sb.get)
                c = min(sc, key=sc.get)
                if b == c:
                    continue
                switch_penalties.append(
                    {
                        "xi_B": int(b),
                        "xi_C": int(c),
                        "penalty_branchB_using_C": float(sb[c] - sb[b]),
                        "penalty_branchC_using_B": float(sc[b] - sc[c]),
                    }
                )

        mean_u = float(u.mean())
        fixed_cont_means = {
            int(a2): float(fixed_cont_sums[a2] / max(fixed_cont_counts, 1))
            for a2 in range(n_a)
            if a2 != a1
        }
        best_fixed_cont = min(fixed_cont_means, key=fixed_cont_means.get)
        v_best_mean = float(v_best_sum / max(fixed_cont_counts, 1))
        v_fixed_mean = float(fixed_cont_means[best_fixed_cont])
        # V-space branching advantage must be >= 0 by construction of argmin.
        delta_branch_V = float(v_fixed_mean - v_best_mean)

        u_fixed_cont = np.empty_like(u)
        for i in range(bank.n_eval):
            for r in range(cfg.noise_replicates):
                u_fixed_cont[i, r] = terminal_after_sequence(
                    bank, i, (a1, best_fixed_cont), crn.eps_obs[i, r]
                )
        # Realized-u branching gap (can be noisy vs V1 selection metric).
        delta_branch_u = float(u_fixed_cont.mean() - mean_u)

        per_first[a1] = {
            "first": design_dict(bank, a1),
            "J_adaptive_exact": mean_u,
            "B_raw": int(len(set(raw_seconds))),
            "B_meaningful": int(len(set(meaningful_seconds))),
            "mean_continuation_gap": float(np.mean(gaps)) if gaps else 0.0,
            "max_continuation_gap": float(np.max(gaps)) if gaps else 0.0,
            "frac_meaningful_gap": float(np.mean(np.asarray(gaps) > eps)) if gaps else 0.0,
            "best_fixed_continuation_design_id": int(best_fixed_cont),
            "best_fixed_continuation_design": design_dict(bank, best_fixed_cont),
            "J_best_fixed_continuation_given_first": float(u_fixed_cont.mean()),
            "J_adaptive_continuation_V": v_best_mean,
            "J_best_fixed_continuation_V": v_fixed_mean,
            "Delta_branching_given_first_V": delta_branch_V,
            "Delta_branching_given_first": delta_branch_u,
            "Delta_branching_given_first_realized_u": delta_branch_u,
            "switch_penalties_sample": switch_penalties[:8],
            "u_adaptive_full": u,
        }
        if mean_u < best_mean:
            best_mean = mean_u
            best_first = a1
            best_u = u
        print(
            f"  adaptive first={a1}: J={mean_u:.4f} "
            f"B_raw={per_first[a1]['B_raw']} B_m={per_first[a1]['B_meaningful']} "
            f"Δ_branch_V={delta_branch_V:.4f} Δ_branch_u={delta_branch_u:.4f}",
            flush=True,
        )

    assert best_first is not None and best_u is not None
    summary_rows = [
        {
            "first_design_id": a1,
            "J_adaptive_exact": d["J_adaptive_exact"],
            "J_best_fixed_continuation_given_first": d[
                "J_best_fixed_continuation_given_first"
            ],
            "Delta_branching_given_first": d["Delta_branching_given_first"],
            "B_raw": d["B_raw"],
            "B_meaningful": d["B_meaningful"],
            "mean_continuation_gap": d["mean_continuation_gap"],
            "max_continuation_gap": d["max_continuation_gap"],
            "frac_meaningful_gap": d["frac_meaningful_gap"],
            "best_fixed_continuation_design_id": d["best_fixed_continuation_design_id"],
        }
        for a1, d in per_first.items()
    ]
    pd.DataFrame(summary_rows).sort_values("J_adaptive_exact").to_csv(
        AUDIT_RESULTS / "ieee5_adaptive_by_first_design.csv", index=False
    )
    pd.DataFrame(branch_examples).to_csv(
        AUDIT_RESULTS / "ieee5_branch_examples.csv", index=False
    )

    return {
        "J_adaptive_best_exact": best_mean,
        "best_first_design": design_dict(bank, best_first),
        "best_first_id": int(best_first),
        "per_first": per_first,
        "meaningful_gap_eps": eps,
        "branch_examples_head": branch_examples[:20],
        "u_best_adaptive_full": best_u,
    }


def run_ieee5_exact_audit(bank: SystemBank, cfg: SuiteConfig) -> dict[str, Any]:
    assert bank.system == "ieee5"
    crn = make_audit_crn(bank, cfg)

    crn_audit = {
        "eps_obs_shape": list(crn.eps_obs.shape),
        "hyp_idx_shape": list(crn.hyp_idx.shape),
        "shared_across_policies": True,
        "eps_obs_checksum": float(np.sum(crn.eps_obs) + np.sum(crn.hyp_noise)),
        "note": (
            "All Fixed ordered pairs, Adaptive policies, and Myopic use this same "
            "CRNBundle instance (identical eps[i,r,t] and hyp draws)."
        ),
    }

    fixed = enumerate_fixed_ordered_pairs(bank, cfg, crn)
    adapt = enumerate_adaptive_exact(bank, cfg, crn)

    myopic_a1, _ = myopic_first_design(
        bank.Y_support,
        bank.U_support,
        sigma_y=bank.sigma_y,
        alpha=bank.alpha,
        margin=bank.safety_margin,
        u_grid=bank.u_grid,
        n_hyp=cfg.n_hyp_y,
        rng=np.random.default_rng(cfg.seed + 21),
    )
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

    ja = adapt["u_best_adaptive_full"].mean(axis=1)
    jf = fixed["u_best_ordered_full"].mean(axis=1)
    jm = u_myopic.mean(axis=1)

    delta_adapt = paired_ci(jf, ja, n_boot=cfg.bootstrap_replicates, seed=cfg.seed + 4)
    delta_nonmyopic = paired_ci(jm, ja, n_boot=cfg.bootstrap_replicates, seed=cfg.seed + 3)

    a_adapt = int(adapt["best_first_id"])
    a_fixed = int(fixed["best_ordered_sequence"]["pair"][0])
    first_ids = sorted({a_adapt, a_fixed, int(myopic_a1)})
    first_table = []
    for a1 in first_ids:
        d = adapt["per_first"][a1]
        roles = []
        if a1 == a_adapt:
            roles.append("adaptive_best_first")
        if a1 == a_fixed:
            roles.append("fixed_best_first")
        if a1 == myopic_a1:
            roles.append("myopic_first")
        first_table.append(
            {
                "first_design": design_dict(bank, a1),
                "role": "|".join(roles),
                "Adaptive_continuation_realized_u": d["J_adaptive_exact"],
                "Best_fixed_continuation_realized_u": d[
                    "J_best_fixed_continuation_given_first"
                ],
                "Adaptive_continuation_V": d["J_adaptive_continuation_V"],
                "Best_fixed_continuation_V": d["J_best_fixed_continuation_V"],
                "Branching_advantage_V": d["Delta_branching_given_first_V"],
                "Branching_advantage_realized_u": d[
                    "Delta_branching_given_first_realized_u"
                ],
                "B_raw": d["B_raw"],
                "B_meaningful": d["B_meaningful"],
                "max_continuation_gap": d["max_continuation_gap"],
            }
        )

    tol = max(cfg.planner_sanity_tol, 0.25 * adapt["meaningful_gap_eps"])
    d_star = adapt["per_first"][a_adapt]
    check_a = float(adapt["J_adaptive_best_exact"]) <= float(u_myopic.mean()) + tol
    # Check B in V-space (selection metric): must hold by construction of argmin.
    check_b_violations = [
        a1
        for a1, d in adapt["per_first"].items()
        if d["Delta_branching_given_first_V"] < -tol
    ]
    check_b = len(check_b_violations) == 0
    # Check C on realized CRN: Adaptive_best should not be much worse than Fixed_best.
    # Small violations can arise from V1 Monte Carlo noise in second-design selection.
    gap_c = float(adapt["J_adaptive_best_exact"]) - float(
        fixed["J_fixed_best_exact_ordered"]
    )
    check_c = gap_c <= tol
    all_gaps_tiny = d_star["max_continuation_gap"] <= adapt["meaningful_gap_eps"]
    branching_adv_V = float(d_star["Delta_branching_given_first_V"])
    branching_adv_u = float(d_star["Delta_branching_given_first"])
    # Check D (soft): best-vs-second gaps all tiny does NOT imply zero advantage vs a
    # single fixed second design (near-tie switching can still help). Warn only.
    check_d = True
    check_d_note = (
        "Best-vs-second continuation gaps are all <= eps, but "
        f"Delta_branching_V={branching_adv_V:.4f} vs best single fixed second. "
        "This can occur with many near-tied top designs; treat as small/noisy "
        "unless Delta_adapt is also statistically positive."
        if all_gaps_tiny and branching_adv_V > adapt["meaningful_gap_eps"]
        else (
            "Consistent: tiny pairwise gaps and small/no V-branching advantage."
            if all_gaps_tiny
            else "Pairwise continuation gaps exceed eps on some branches."
        )
    )

    checks = {
        "A_adaptive_le_myopic": {
            "pass": bool(check_a),
            "J_adaptive": float(adapt["J_adaptive_best_exact"]),
            "J_myopic": float(u_myopic.mean()),
            "tol": tol,
        },
        "B_adaptive_cont_le_fixed_cont_same_first_Vspace": {
            "pass": bool(check_b),
            "n_violations": len(check_b_violations),
            "violation_first_ids": check_b_violations[:10],
            "note": "Compared in V1 selection metric space (must hold by argmin).",
        },
        "C_adaptive_best_le_fixed_best_realized": {
            "pass": bool(check_c),
            "J_adaptive": float(adapt["J_adaptive_best_exact"]),
            "J_fixed_best_ordered": float(fixed["J_fixed_best_exact_ordered"]),
            "gap_adaptive_minus_fixed": gap_c,
            "note": (
                "Realized-u comparison; failures with tiny V-branching usually indicate "
                "V1 MC noise or Fixed open-loop luck, not true adaptive value."
            ),
        },
        "D_tiny_pairwise_gaps_vs_branching_advantage": {
            "pass": bool(check_d),
            "warning": bool(
                all_gaps_tiny and branching_adv_V > adapt["meaningful_gap_eps"]
            ),
            "all_gaps_tiny_for_best_first": bool(all_gaps_tiny),
            "Delta_branching_given_first_V": branching_adv_V,
            "Delta_branching_given_first_realized_u": branching_adv_u,
            "note": check_d_note,
        },
    }
    checks["any_fail"] = (not check_a) or (not check_b)
    checks["C_realized_warning"] = not check_c

    delta_mean = float(delta_adapt["mean"])
    reported_fixed = (23, 1)
    u_reported = np.empty((bank.n_eval, cfg.noise_replicates), dtype=np.float64)
    for i in range(bank.n_eval):
        for r in range(cfg.noise_replicates):
            u_reported[i, r] = terminal_after_sequence(
                bank, i, reported_fixed, crn.eps_obs[i, r]
            )

    # Classify primarily by V-space branching + comparison of reported Fixed vs exact.
    reported_gap_vs_exact = float(
        u_reported.mean() - fixed["J_fixed_best_exact_ordered"]
    )
    eps = float(adapt["meaningful_gap_eps"])
    delta_ci_crosses_zero = (
        float(delta_adapt["ci_low"]) <= 0.0 <= float(delta_adapt["ci_high"])
    )
    reported_is_suboptimal = (
        list(reported_fixed) != list(fixed["best_ordered_sequence"]["pair"])
        and reported_gap_vs_exact > 1e-9
    )
    if checks["any_fail"]:
        verdict = "UNRESOLVED_IMPLEMENTATION_INCONSISTENCY"
    elif (
        branching_adv_V > eps
        and delta_mean > tol
        and not delta_ci_crosses_zero
        and d_star["B_meaningful"] >= 1
    ):
        verdict = "GENUINE_ADAPTIVE_ADVANTAGE"
    elif reported_is_suboptimal and delta_ci_crosses_zero and d_star["B_meaningful"] == 0:
        # Explains prior Delta_adapt>0 with B_meaningful=0: suboptimal Fixed baseline.
        verdict = "FIXED_SEARCH_BUG_OR_MISMATCH"
    elif d_star["B_meaningful"] == 0 and (
        delta_ci_crosses_zero or abs(delta_mean) <= tol
    ):
        verdict = "NO_MEANINGFUL_ADAPTIVE_ADVANTAGE"
    else:
        verdict = "NO_MEANINGFUL_ADAPTIVE_ADVANTAGE"

    return {
        "system": "ieee5",
        "crn_audit": crn_audit,
        "splits": {
            "posterior_support_split": bank.split_support,
            "evaluation_truth_split": bank.split_eval,
            "observation_centre_source": "Y_support = train max_rocof.npy (subsample)",
            "Y_eval_source": "Y_eval = test max_rocof.npy (subsample)",
            "U_support_source": "U_support = train U.npy (subsample)",
        },
        "J_myopic_exact": float(u_myopic.mean()),
        "myopic_first_design": design_dict(bank, myopic_a1),
        "fixed": {
            k: v
            for k, v in fixed.items()
            if k != "u_best_ordered_full"
        },
        "adaptive": {
            "J_adaptive_best_exact": adapt["J_adaptive_best_exact"],
            "best_first_design": adapt["best_first_design"],
            "best_first_id": adapt["best_first_id"],
            "meaningful_gap_eps": adapt["meaningful_gap_eps"],
            "branch_examples_head": adapt["branch_examples_head"],
            "best_first_branching": {
                "B_raw": d_star["B_raw"],
                "B_meaningful": d_star["B_meaningful"],
                "mean_continuation_gap": d_star["mean_continuation_gap"],
                "max_continuation_gap": d_star["max_continuation_gap"],
                "frac_meaningful_gap": d_star["frac_meaningful_gap"],
                "Delta_branching_given_first_V": d_star["Delta_branching_given_first_V"],
                "Delta_branching_given_first": d_star[
                    "Delta_branching_given_first_V"
                ],  # primary = V-space
                "Delta_branching_given_first_realized_u": d_star[
                    "Delta_branching_given_first_realized_u"
                ],
                "J_adaptive_continuation_V": d_star["J_adaptive_continuation_V"],
                "J_best_fixed_continuation_V": d_star["J_best_fixed_continuation_V"],
                "J_best_fixed_continuation_given_first": d_star[
                    "J_best_fixed_continuation_given_first"
                ],
                "best_fixed_continuation_design": d_star[
                    "best_fixed_continuation_design"
                ],
                "switch_penalties_sample": d_star["switch_penalties_sample"],
            },
        },
        "Delta_adapt_exact": delta_adapt,
        "Delta_nonmyopic_exact": delta_nonmyopic,
        "first_design_effect_table": first_table,
        "reported_fixed_sequence_23_1": {
            "pair": list(reported_fixed),
            "J_on_same_CRN": float(u_reported.mean()),
            "gap_vs_exact_best_ordered": float(
                u_reported.mean() - fixed["J_fixed_best_exact_ordered"]
            ),
            "gap_vs_adaptive_best": float(
                u_reported.mean() - adapt["J_adaptive_best_exact"]
            ),
        },
        "consistency_checks": checks,
        "ieee5_audit_verdict": verdict,
        "previous_report_Delta_adapt_0_0089_interpretation": (
            "The previously reported Fixed sequence (23,1) is compared here on the "
            "same CRN. If its J exceeds the exact best ordered Fixed and/or the "
            "same-first branching advantage is ~0, the old Delta_adapt=0.0089 is an "
            "artifact of Fixed search/order/support-selection mismatch, not genuine "
            "observation-dependent adaptive value."
        ),
    }
