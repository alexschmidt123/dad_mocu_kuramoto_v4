#!/usr/bin/env python3
"""Run consistency audit for control adaptive-advantage diagnostics.

Example:
  python tools/diagnostics/run_consistency_audit.py
  python tools/diagnostics/run_consistency_audit.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.adaptive_advantage.config import SuiteConfig
from tools.adaptive_advantage.consistency_audit.audit_branching_values import (
    summarize_branching_levels,
)
from tools.adaptive_advantage.consistency_audit.audit_fixed_vs_adaptive import (
    audit_policy_definitions,
)
from tools.adaptive_advantage.consistency_audit.audit_ieee5_exact import (
    run_ieee5_exact_audit,
)
from tools.adaptive_advantage.consistency_audit.audit_noise_pairing import (
    audit_noise_pairing,
)
from tools.adaptive_advantage.consistency_audit.audit_norepeat import (
    audit_norepeat,
)
from tools.adaptive_advantage.consistency_audit.audit_planner_reporting import (
    audit_ieee9_ieee14_reporting,
)
from tools.adaptive_advantage.consistency_audit.audit_sequence_semantics import (
    audit_fixed_scoring_sorts_subset,
    audit_order_on_eval,
)
from tools.adaptive_advantage.consistency_audit.common import (
    AUDIT_REPORTS,
    AUDIT_RESULTS,
    ensure_audit_dirs,
    load_audit_bank,
    save_json,
)


def _audit_cfg(args: argparse.Namespace) -> SuiteConfig:
    if args.quick:
        return SuiteConfig(
            systems=("ieee5",),
            noise_replicates=12,
            bootstrap_replicates=200,
            seed=int(args.seed),
            quick=True,
            support_size=64,
            eval_size=20,
            n_hyp_y=10,
        ).resolved()
    if args.match_diagnostic:
        return SuiteConfig(
            systems=("ieee5",),
            noise_replicates=100,
            bootstrap_replicates=2000,
            seed=int(args.seed),
            support_size=128,
            eval_size=64,
            n_hyp_y=32,
        )
    # Default audit: exact over design space, moderate MC budget.
    return SuiteConfig(
        systems=("ieee5",),
        noise_replicates=40,
        bootstrap_replicates=2000,
        seed=int(args.seed),
        support_size=128,
        eval_size=48,
        n_hyp_y=16,
    )


def _write_report(payload: dict[str, Any]) -> tuple[Path, Path]:
    ensure_audit_dirs()
    ieee5 = payload["ieee5_exact"]
    defs = payload["policy_definitions"]
    seq = payload["sequence_semantics"]
    order = payload["order_on_eval"]
    branch = payload["branching_levels"]
    noise = payload["noise_pairing"]
    norep = payload["norepeat"]
    rep914 = payload["planner_reporting"]
    checks = ieee5["consistency_checks"]

    lines: list[str] = []
    lines.append("# Consistency Audit Report\n")
    lines.append(
        "Purpose: verify mathematical/implementation consistency of the control "
        "adaptive-advantage diagnostic suite (existing banks only).\n"
    )

    lines.append("## 1. Executive Summary\n")
    lines.append(
        f"- **IEEE-5 audit verdict:** `{ieee5['ieee5_audit_verdict']}`\n"
        f"- Exact J_Myopic = `{ieee5['J_myopic_exact']:.6f}`\n"
        f"- Exact J_Fixed(best ordered) = "
        f"`{ieee5['fixed']['J_fixed_best_exact_ordered']:.6f}`\n"
        f"- Exact J_Adaptive = `{ieee5['adaptive']['J_adaptive_best_exact']:.6f}`\n"
        f"- Exact Δ_adapt = `{ieee5['Delta_adapt_exact']['mean']:.6f}` "
        f"CI=[{ieee5['Delta_adapt_exact']['ci_low']:.6f}, "
        f"{ieee5['Delta_adapt_exact']['ci_high']:.6f}] "
        f"({ieee5['Delta_adapt_exact']['ci_verdict']})\n"
        f"- Branching advantage given best adaptive first design = "
        f"`{ieee5['adaptive']['best_first_branching']['Delta_branching_given_first']:.6f}`\n"
        f"- IEEE-14 J_ApproxPlanningRaw = "
        f"`{rep914['ieee14'].get('J_ApproxPlanningRaw')}`\n"
        f"- IEEE-14 J_Myopic = `{rep914['ieee14'].get('J_Myopic')}`\n"
        f"- IEEE-14 J_BestAvailableAdaptive = "
        f"`{rep914['ieee14'].get('J_BestAvailableAdaptive')}` "
        f"(policy=`{rep914['ieee14'].get('BestAvailableAdaptive_policy')}`)\n"
    )

    lines.append("## 2. IEEE-5 contradiction being tested\n")
    lines.append(
        "Previous report: J_Myopic≈J_Planning=0.5631 < J_Fixed*=0.5720 "
        "(Δ_adapt≈0.0089 PASS, both EXACT), yet meaningful branching Bmax=0 and "
        "routing pairs=0. This audit independently recomputes exact Fixed and "
        "Adaptive under identical CRN to determine whether that gap is genuine "
        "adaptive value or an artifact.\n"
    )

    lines.append("## 3. Fixed policy definition\n")
    lines.append("```json\n" + json.dumps(defs["Fixed"], indent=2) + "\n```\n")

    lines.append("## 4. Adaptive policy definition\n")
    lines.append(
        "```json\n" + json.dumps(defs["AdaptivePlanning_T2"], indent=2) + "\n```\n"
    )
    lines.append("Myopic:\n")
    lines.append("```json\n" + json.dumps(defs["MyopicControl"], indent=2) + "\n```\n")

    lines.append("## 5. Ordered versus unordered design sequence audit\n")
    lines.append(
        "```json\n" + json.dumps(seq, indent=2) + "\n```\n"
    )
    lines.append(
        f"- Max |mean| order difference on tested pairs: "
        f"`{order['max_abs_mean_order_difference']:.6f}`\n"
        f"- Max scenario |order difference|: "
        f"`{order['max_abs_scenario_order_difference']:.6f}`\n"
    )
    if seq.get("production_score_fixed_subset_sorts_actions"):
        lines.append(
            "**Flag:** production `_score_fixed_subset` sorts actions, so Fixed "
            "*search* is unordered-subset scoring while evaluation uses a stored "
            "ordered sequence. This is a search/evaluation semantic mismatch.\n"
        )

    lines.append("## 6. Exact IEEE-5 Fixed enumeration\n")
    bf = ieee5["fixed"]["best_ordered_sequence"]
    lines.append(
        f"- J_fixed_best_exact_ordered = `{ieee5['fixed']['J_fixed_best_exact_ordered']:.6f}`\n"
        f"- Best ordered sequence: first=`{bf['first']}`, second=`{bf['second']}`\n"
        f"- Best unordered (via best evaluation order): "
        f"`{ieee5['fixed']['best_unordered_via_best_order']}`\n"
        f"- Previously reported Fixed (23,1) on same CRN: "
        f"J=`{ieee5['reported_fixed_sequence_23_1']['J_on_same_CRN']:.6f}` "
        f"(gap vs exact best ordered = "
        f"`{ieee5['reported_fixed_sequence_23_1']['gap_vs_exact_best_ordered']:.6f}`)\n"
    )

    lines.append("## 7. Exact IEEE-5 Adaptive enumeration\n")
    lines.append(
        f"- J_adaptive_best_exact = `{ieee5['adaptive']['J_adaptive_best_exact']:.6f}`\n"
        f"- Best first design = `{ieee5['adaptive']['best_first_design']}`\n"
        f"- Meaningful gap eps = `{ieee5['adaptive']['meaningful_gap_eps']}`\n"
    )

    lines.append("## 8. Exact adaptive gap\n")
    dn = ieee5["Delta_adapt_exact"]
    lines.append(
        f"- Δ_adapt_exact = J_fixed_best_ordered − J_adaptive_best = `{dn['mean']:.6f}`\n"
        f"- 95% CI = [{dn['ci_low']:.6f}, {dn['ci_high']:.6f}] ({dn['ci_verdict']})\n"
        f"- frac Fixed higher / equal / lower: "
        f"{dn['frac_a_higher']:.3f} / {dn['frac_equal']:.3f} / {dn['frac_a_lower']:.3f}\n"
    )

    lines.append("## 9. Branching-value audit\n")
    lines.append("```json\n" + json.dumps(branch, indent=2) + "\n```\n")

    lines.append(
        "## 10. Adaptive versus best fixed continuation under same first design\n"
    )
    bb = ieee5["adaptive"]["best_first_branching"]
    lines.append(
        f"- For adaptive-optimal first design `{ieee5['adaptive']['best_first_design']}`:\n"
        f"  - J_adaptive_continuation = `{ieee5['adaptive']['J_adaptive_best_exact']:.6f}`\n"
        f"  - J_best_fixed_continuation = "
        f"`{bb['J_best_fixed_continuation_given_first']:.6f}` "
        f"(second=`{bb['best_fixed_continuation_design']}`)\n"
        f"  - Δ_branching_given_first = `{bb['Delta_branching_given_first']:.6f}`\n"
        f"  - B_raw / B_meaningful = `{bb['B_raw']}` / `{bb['B_meaningful']}`\n"
        f"  - max continuation gap = `{bb['max_continuation_gap']:.6f}`\n"
    )

    lines.append("## 11. First-design effect\n")
    lines.append(
        "| First design | Role | Adapt V | Fixed-cont V | Branch adv V | "
        "Adapt u | Fixed-cont u | Branch adv u |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|"
    )
    for row in ieee5["first_design_effect_table"]:
        fd = row["first_design"]
        lines.append(
            f"| id={fd['design_id']} Amp={fd['Amp']} bus={fd['bus_location']} | "
            f"{row['role']} | {row['Adaptive_continuation_V']:.6f} | "
            f"{row['Best_fixed_continuation_V']:.6f} | {row['Branching_advantage_V']:.6f} | "
            f"{row['Adaptive_continuation_realized_u']:.6f} | "
            f"{row['Best_fixed_continuation_realized_u']:.6f} | "
            f"{row['Branching_advantage_realized_u']:.6f} |"
        )

    lines.append("\n## 12. Noise-pairing audit\n")
    lines.append("```json\n" + json.dumps(noise, indent=2) + "\n```\n")
    lines.append("CRN used in exact IEEE-5 audit:\n")
    lines.append("```json\n" + json.dumps(ieee5["crn_audit"], indent=2) + "\n```\n")

    lines.append("## 13. Support/evaluation split audit\n")
    lines.append("```json\n" + json.dumps(ieee5["splits"], indent=2) + "\n```\n")

    lines.append("## 14. No-repeat constraint audit\n")
    lines.append("```json\n" + json.dumps(norep, indent=2) + "\n```\n")

    lines.append("## 15. IEEE-9 reporting interpretation\n")
    lines.append("```json\n" + json.dumps(rep914["ieee9"], indent=2) + "\n```\n")
    lines.append(
        "Conservative interpretation retained: first-design adaptive search is "
        "APPROXIMATE; do not certify absence or presence of adaptive advantage.\n"
    )

    lines.append("## 16. IEEE-14 raw planner versus Myopic fallback\n")
    lines.append("```json\n" + json.dumps(rep914["ieee14"], indent=2) + "\n```\n")

    lines.append("## 17. Consistency checks A–D\n")
    lines.append("```json\n" + json.dumps(checks, indent=2) + "\n```\n")
    if checks.get("any_fail"):
        lines.append("**IMPLEMENTATION_OR_EVALUATION_INCONSISTENCY flagged.**\n")

    lines.append("## 18. Corrected interpretation\n")
    lines.append(ieee5["previous_report_Delta_adapt_0_0089_interpretation"] + "\n")
    lines.append(
        f"- Same-first branching advantage ≈ "
        f"{bb['Delta_branching_given_first']:.6f} with B_meaningful="
        f"{bb['B_meaningful']} implies observation-dependent continuation does "
        f"{'NOT ' if abs(bb['Delta_branching_given_first']) < 1e-6 else ''}"
        f"materially lower expected terminal u_ctrl.\n"
        f"- Previously reported Fixed (23,1) J on audit CRN = "
        f"{ieee5['reported_fixed_sequence_23_1']['J_on_same_CRN']:.6f}; "
        f"exact best ordered Fixed = "
        f"{ieee5['fixed']['J_fixed_best_exact_ordered']:.6f}.\n"
    )

    lines.append("## 19. Final verdict\n")
    lines.append(f"- **IEEE-5:** `{ieee5['ieee5_audit_verdict']}`\n")
    lines.append(
        f"- **IEEE-9:** reporting disambiguated; search "
        f"`{rep914['ieee9'].get('adaptive_search')}`; "
        f"J_ApproxPlanningRaw=`{rep914['ieee9'].get('J_ApproxPlanningRaw')}`, "
        f"J_Myopic=`{rep914['ieee9'].get('J_Myopic')}`, "
        f"J_BestAvailableAdaptive=`{rep914['ieee9'].get('J_BestAvailableAdaptive')}`.\n"
    )
    lines.append(
        f"- **IEEE-14:** "
        f"J_ApproxPlanningRaw=`{rep914['ieee14'].get('J_ApproxPlanningRaw')}`, "
        f"J_Myopic=`{rep914['ieee14'].get('J_Myopic')}`, "
        f"J_BestAvailableAdaptive=`{rep914['ieee14'].get('J_BestAvailableAdaptive')}` "
        f"(fallback_used=`{rep914['ieee14'].get('myopic_fallback_used')}`).\n"
    )

    md_path = AUDIT_REPORTS / "consistency_audit_report.md"
    json_path = AUDIT_REPORTS / "consistency_audit_report.json"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Drop huge arrays if any slipped in
    save_json(json_path, payload)
    return md_path, json_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--match-diagnostic",
        action="store_true",
        help="Use diagnostic MC sizes (64 eval × 100 noise × 32 hyp); slower.",
    )
    args = p.parse_args(argv)

    cfg = _audit_cfg(args)
    ensure_audit_dirs()
    t0 = time.time()

    print("=== Consistency audit: loading ieee5 bank ===", flush=True)
    bank = load_audit_bank("ieee5", cfg)
    print(
        f"n_designs={bank.n_designs} support={bank.n_support} eval={bank.n_eval} "
        f"noise_rep={cfg.noise_replicates} n_hyp={cfg.n_hyp_y}",
        flush=True,
    )

    payload: dict[str, Any] = {
        "config": {
            "seed": cfg.seed,
            "support_size": bank.n_support,
            "eval_size": bank.n_eval,
            "noise_replicates": cfg.noise_replicates,
            "n_hyp_y": cfg.n_hyp_y,
            "bootstrap_replicates": cfg.bootstrap_replicates,
            "quick": bool(args.quick),
            "match_diagnostic": bool(args.match_diagnostic),
        },
        "policy_definitions": audit_policy_definitions(),
        "sequence_semantics": audit_fixed_scoring_sorts_subset(),
    }

    print("=== Order sensitivity on eval CRN ===", flush=True)
    payload["order_on_eval"] = audit_order_on_eval(
        bank, cfg, pairs=[(23, 1), (1, 23), (0, 1), (1, 0), (5, 10), (10, 5)]
    )

    print("=== Noise pairing ===", flush=True)
    payload["noise_pairing"] = audit_noise_pairing(bank, cfg)
    payload["norepeat"] = audit_norepeat(bank)

    print("=== Exact IEEE-5 Fixed + Adaptive ===", flush=True)
    ieee5 = run_ieee5_exact_audit(bank, cfg)
    # Strip non-JSON arrays from nested per_first before saving
    for a1, d in list(ieee5.get("adaptive", {}).items()):
        pass
    # per_first not in exported adaptive summary (already stripped in return)
    payload["ieee5_exact"] = ieee5
    payload["branching_levels"] = summarize_branching_levels(ieee5)

    print("=== IEEE-9 / IEEE-14 planner reporting ===", flush=True)
    payload["planner_reporting"] = audit_ieee9_ieee14_reporting()

    save_json(AUDIT_RESULTS / "consistency_audit_payload.json", payload)
    md_path, json_path = _write_report(payload)

    e5 = payload["ieee5_exact"]
    r14 = payload["planner_reporting"]["ieee14"]
    print("\n========== CONSISTENCY AUDIT SUMMARY ==========")
    print(f"IEEE-5 J_Myopic              = {e5['J_myopic_exact']:.6f}")
    print(f"IEEE-5 J_Fixed(best ordered) = {e5['fixed']['J_fixed_best_exact_ordered']:.6f}")
    print(f"IEEE-5 J_Adaptive            = {e5['adaptive']['J_adaptive_best_exact']:.6f}")
    print(
        f"IEEE-5 Delta_adapt           = {e5['Delta_adapt_exact']['mean']:.6f} "
        f"[{e5['Delta_adapt_exact']['ci_low']:.6f}, {e5['Delta_adapt_exact']['ci_high']:.6f}] "
        f"({e5['Delta_adapt_exact']['ci_verdict']})"
    )
    print(
        "IEEE-5 branching advantage (same first) = "
        f"{e5['adaptive']['best_first_branching']['Delta_branching_given_first']:.6f}"
    )
    print(f"IEEE-5 final audit verdict   = {e5['ieee5_audit_verdict']}")
    print(f"IEEE-14 J_ApproxPlanningRaw  = {r14.get('J_ApproxPlanningRaw')}")
    print(f"IEEE-14 J_Myopic             = {r14.get('J_Myopic')}")
    print(f"IEEE-14 J_BestAvailableAdaptive = {r14.get('J_BestAvailableAdaptive')}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print(f"Report: {md_path}")
    print(f"JSON:   {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
