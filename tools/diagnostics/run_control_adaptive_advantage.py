#!/usr/bin/env python3
"""Run control adaptive-advantage diagnostics on existing banks only.

Example:
  python tools/diagnostics/run_control_adaptive_advantage.py
  python tools/diagnostics/run_control_adaptive_advantage.py --quick
  python tools/diagnostics/run_control_adaptive_advantage.py --systems ieee5 ieee9 ieee14
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure repo root on sys.path when executed as a script.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.adaptive_advantage.config import (  # noqa: E402
    REPORTS_DIR,
    RESULTS_DIR,
    SuiteConfig,
    ensure_output_dirs,
)
from tools.adaptive_advantage.diagnostics_core import (  # noqa: E402
    run_system_diagnostics,
)


def _write_reports(summaries: list[dict]) -> tuple[Path, Path]:
    ensure_output_dirs()
    inv = {}
    for s in summaries:
        inv[s["system"]] = s["inventory"]
    (RESULTS_DIR.parent / "data_inventory.json").write_text(
        json.dumps(inv, indent=2), encoding="utf-8"
    )

    md_path = REPORTS_DIR / "control_adaptive_advantage_report.md"
    json_path = REPORTS_DIR / "control_adaptive_advantage_report.json"

    lines: list[str] = []
    lines.append("# Control Adaptive-Advantage Diagnostic Report\n")
    lines.append(
        "Objective: `J(π)=E[u_ctrl(H_T)]` using **existing banks only** "
        "(no ODE / CUDA regeneration).\n"
    )
    lines.append(
        "Terminology: a **design** is `ξ={Amp, bus location, duration}`; "
        "`probe` means bus location only.\n"
    )
    lines.append(
        "Validation pass notes: (1) planner sanity "
        "`J_planning ≤ J_myopic` (exact enforced / approximate flagged); "
        "(2) approximate adaptive candidates always include Myopic + Fixed designs; "
        "(3) paired comparisons use common random numbers; "
        "(4) branching/routing use value-weighted meaningful gaps.\n"
    )

    lines.append("## 1. Executive Summary\n")
    lines.append(
        "| System | U heterogeneity | Meaningful branching | Routing | "
        "J_Myopic | J_Fixed* | J_Planning | Sanity | Δ_nonmyopic | 95% CI | "
        "Δ_adapt | 95% CI | Verdict |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---|---:|---|---:|---|---|")
    for s in summaries:
        dn = s["Delta_nonmyopic"]
        da = s["Delta_adapt"]
        san = s.get("planner_sanity", {})
        lines.append(
            f"| {s['system']} | {s['u_heterogeneity']['n_unique']} uniq / "
            f"std={s['u_heterogeneity']['std']:.3f} | "
            f"Bmax={s['max_branching_B']} (raw={s.get('max_branching_B_raw', '?')}) | "
            f"n_pairs={s['n_routing_pairs']} | "
            f"{s['J_myopic']:.4f} | {s['J_fixed']:.4f} | {s['J_planning_eval']:.4f} | "
            f"{san.get('status', '?')} | "
            f"{dn['mean']:.4f} | [{dn['ci_low']:.4f},{dn['ci_high']:.4f}] ({dn['ci_verdict']}) | "
            f"{da['mean']:.4f} | [{da['ci_low']:.4f},{da['ci_high']:.4f}] ({da['ci_verdict']}) | "
            f"**{s['verdict']}** |"
        )

    lines.append("\n### Search labels and candidate audit\n")
    for s in summaries:
        audit = s.get("candidate_audit", {})
        lines.append(
            f"- **{s['system']}**: Adaptive search: `{s['plan_meta']['Adaptive_search']}` "
            f"({s['plan_meta']['planner_label']}); "
            f"{s['plan_meta']['Candidate_designs_used']}. "
            f"Fixed search: `{s['fixed_meta']['Adaptive_search']}`; "
            f"{s['fixed_meta']['Candidate_designs_used']}. "
            f"First candidates include myopic designs "
            f"`{audit.get('myopic_designs_included')}` and fixed "
            f"`{audit.get('fixed_sequence_design_ids')}`; "
            f"second candidates=`{audit.get('second_candidates')}`. "
            f"meaningful_gap_eps=`{s.get('meaningful_gap_eps')}`."
        )

    lines.append("\n### Planner sanity\n")
    for s in summaries:
        san = s.get("planner_sanity", {})
        gap = san.get("gap_planning_minus_myopic", float("nan"))
        lines.append(
            f"- **{s['system']}**: `{san.get('status')}` — "
            f"J_plan−J_myopic=`{gap:.6f}` "
            f"(tol=`{san.get('tolerance')}`). {san.get('note', '')}"
        )

    lines.append("\n## 2. Data and split inventory\n")
    lines.append("```json\n" + json.dumps(inv, indent=2) + "\n```\n")

    lines.append("## 3–6. Design space\n")
    for s in summaries:
        inv_s = s["inventory"]
        lines.append(f"### {s['system']}\n")
        lines.append(f"- Amp values: `{inv_s['Amp_values']}`")
        lines.append(f"- Bus locations: `{inv_s['bus_locations']}`")
        lines.append(f"- Fixed duration: `{inv_s['fixed_duration']}`")
        lines.append(f"- n_designs: `{inv_s['n_designs']}`, σ_y=`{inv_s['sigma_y']}`")
        lines.append(f"- Control grid: `{inv_s['control_grid']}`\n")

    lines.append("## 7. U-bank heterogeneity\n")
    for s in summaries:
        u = s["u_heterogeneity"]
        lines.append(
            f"- **{s['system']}**: min={u['min']:.4f}, max={u['max']:.4f}, "
            f"mean={u['mean']:.4f}, std={u['std']:.4f}, n_unique={u['n_unique']}, "
            f"entropy={u['entropy']:.3f}, Q95={u['Q95']:.4f} (G1={s['G1']})"
        )

    lines.append("\n## 8. Design–control relevance (top 5 by |Pearson|)\n")
    for s in summaries:
        lines.append(f"### {s['system']}\n")
        for row in s["top_design_control_relevance"]:
            lines.append(
                f"- design_id={row['design_id']}, Amp={row['Amp']}, bus={row['bus_location']}, "
                f"dur={row['duration']}, |corr|={row['abs_pearson']:.3f}, "
                f"Spearman={row['spearman_corr_y_U']:.3f}, MI={row['mi_y_U']:.3f}"
            )

    lines.append("\n## 9. Design SNR structure\n")
    for s in summaries:
        lines.append(f"- **{s['system']}**: class=`{s['snr_class']}`")

    lines.append(
        "\n## 10–13. Quantile/snap, complementarity, meaningful branching, routing\n"
    )
    for s in summaries:
        lines.append(
            f"- **{s['system']}**: G3(snap activity)={s['G3']}, "
            f"G4(meaningful branching)={s['G4']} "
            f"(Bmax_meaningful={s['max_branching_B']}, "
            f"Bmax_raw={s.get('max_branching_B_raw')}, "
            f"max_value_gap={s.get('max_value_weighted_gap', float('nan')):.4f}, "
            f"eps={s.get('meaningful_gap_eps')}), "
            f"G5(meaningful routing)={s['G5']} (n_pairs={s['n_routing_pairs']})"
        )

    lines.append("\n## 14–15. Myopic / Fixed vs adaptive planning\n")
    for s in summaries:
        dn = s["Delta_nonmyopic"]
        da = s["Delta_adapt"]
        j_raw = s.get("J_ApproxPlanningRaw", s.get("J_planning_selected_first"))
        j_best = s.get("J_BestAvailableAdaptive", s.get("J_planning_eval"))
        lines.append(
            f"- **{s['system']}**: "
            f"J_Myopic=`{s['J_myopic']:.4f}`, "
            f"J_ApproxPlanningRaw=`{j_raw:.4f}`, "
            f"J_BestAvailableAdaptive=`{j_best:.4f}` "
            f"(policy=`{s.get('BestAvailableAdaptive_policy', '?')}`), "
            f"J_Fixed*=`{s['J_fixed']:.4f}`."
        )
        lines.append(
            f"  - Δ_nonmyopic=J_myopic−J_BestAvailableAdaptive="
            f"{dn['mean']:.4f} CI=[{dn['ci_low']:.4f},{dn['ci_high']:.4f}] "
            f"→ {dn['ci_verdict']} (cert={dn.get('certification', '?')}); "
            f"Δ_adapt=J_fixed−J_BestAvailableAdaptive={da['mean']:.4f} "
            f"CI=[{da['ci_low']:.4f},{da['ci_high']:.4f}] → {da['ci_verdict']} "
            f"(cert={da.get('certification', '?')})"
        )
        if dn.get("interpretation"):
            lines.append(f"  - Non-myopic note: {dn['interpretation']}")
        if da.get("interpretation"):
            lines.append(f"  - Adaptive note: {da['interpretation']}")
        if s.get("BestAvailableAdaptive_policy") == "Myopic fallback":
            lines.append(
                "  - **Reporting note:** J_BestAvailableAdaptive equals Myopic via "
                "fallback; this is NOT evidence that the approximate non-myopic "
                "planner matched Myopic. Cite J_ApproxPlanningRaw separately."
            )

    lines.append("\n## 16. DAD/RL reward learnability\n")
    for s in summaries:
        r = s["reward_learnability"]
        lines.append(
            f"- **{s['system']}**: structure=`{s['reward_structure']}`; "
            f"frac Δu=0: {r['frac_delta_u_zero']:.3f}; "
            f"intermediate-change episodes: {r['frac_episodes_intermediate_change']:.3f}; "
            f"var(u_T)={r['var_terminal_u']:.4f}"
        )

    lines.append("\n## 17–19. Per-system verdicts\n")
    for s in summaries:
        lines.append(f"### {s['system']}: **{s['verdict']}**\n")
        lines.append(
            f"Gates G0–G8: {[s[f'G{i}'] for i in range(9)]}\n"
        )

    lines.append("## 20. Limitations\n")
    lines.append(
        "- T=2 only in this run (primary branching diagnostic).\n"
        "- Large catalogs (ieee9/14) may use **APPROXIMATE** first-design screens; "
        "second-step search uses the full catalog; Myopic/Fixed designs are forced "
        "into the first-design candidate set. Approximate results are never labeled "
        "as exact oracles.\n"
        "- Eval rollouts use support-bank likelihood centres (production style) and "
        "held-out test θ for clean Y. Policy comparisons use **common random numbers**: "
        "shared observation noise `eps[i,r,t]` and shared hypothetical scoring draws "
        "across methods / candidate designs.\n"
        "- Branching/routing counts require a minimum continuation-value gap "
        "`meaningful_gap_eps` (default: half minimum positive control-grid spacing) "
        "so near-ties are not treated as adaptive structure.\n"
        "- If an approximate planner violates `J_planning ≤ J_myopic`, the report "
        "flags a **search-quality limitation**, not evidence that non-myopic planning "
        "is intrinsically harmful.\n"
        "- Does not train DAD/RL; certifies data structure for adaptive advantage only.\n"
    )

    lines.append("## 21. Final conclusion\n")
    ieee5 = next((s for s in summaries if s["system"] == "ieee5"), None)
    ieee9 = next((s for s in summaries if s["system"] == "ieee9"), None)
    ieee14 = next((s for s in summaries if s["system"] == "ieee14"), None)
    if ieee5 and ieee5["plan_meta"]["Adaptive_search"] == "EXACT":
        lines.append(
            "- **IEEE-5**: exact search is the strongest currently reliable conclusion. "
            f"Verdict `{ieee5['verdict']}` with "
            f"Δ_nonmyopic={ieee5['Delta_nonmyopic']['ci_verdict']}, "
            f"Δ_adapt={ieee5['Delta_adapt']['ci_verdict']} "
            f"(G7 certified adaptive advantage={ieee5['G7']}; "
            f"meaningful branching G4={ieee5['G4']}). "
            "Raw continuation argmin changes can be large while meaningful "
            "value-weighted branching is zero — near-ties, not exploitable sequential "
            "structure. A small Planning/Myopic edge over Fixed without meaningful "
            "branching is not sequential adaptive value.\n"
        )
    if ieee9:
        lines.append(
            f"- **IEEE-9**: `{ieee9['verdict']}` "
            f"(adaptive search `{ieee9['plan_meta']['Adaptive_search']}`, "
            f"sanity `{ieee9.get('planner_sanity', {}).get('status')}`). "
            "Do not conclude adaptive structure is weak solely from approximate search.\n"
        )
    if ieee14:
        lines.append(
            f"- **IEEE-14**: `{ieee14['verdict']}` "
            f"(adaptive search `{ieee14['plan_meta']['Adaptive_search']}`, "
            f"sanity `{ieee14.get('planner_sanity', {}).get('status')}`). "
            "A planner worse than Myopic under approximate search indicates a "
            "search-quality limitation, not that non-myopic planning is intrinsically "
            "worse than Myopic Control.\n"
        )
    lines.append(
        "- Overall: do **not** conclude DAD/RL should beat Myopic/Fixed from "
        "parameter uncertainty alone; require meaningful branching + routing + "
        "certified positive Δ_nonmyopic/Δ_adapt under exact (or sanity-OK) search.\n"
    )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps({"systems": summaries}, indent=2), encoding="utf-8")
    return md_path, json_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--systems", nargs="+", default=["ieee5", "ieee9", "ieee14"])
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--noise-replicates", type=int, default=64)
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    if int(args.horizon) != 2:
        print("WARNING: suite currently implements T=2 as the primary diagnostic.")

    cfg = SuiteConfig(
        systems=tuple(args.systems),
        horizon=int(args.horizon),
        noise_replicates=int(args.noise_replicates),
        bootstrap_replicates=int(args.bootstrap_replicates),
        seed=int(args.seed),
        quick=bool(args.quick),
    ).resolved()

    ensure_output_dirs()
    summaries = []
    t0 = time.time()
    for sys_name in cfg.systems:
        print(f"\n=== Running diagnostics for {sys_name} ===", flush=True)
        t_sys = time.time()
        summary = run_system_diagnostics(sys_name, cfg)
        summaries.append(summary)
        inv = summary.get("inventory") or {}
        san = summary.get("planner_sanity", {})
        print(
            "inventory:",
            {
                "n_designs": inv.get("n_designs"),
                "sigma_y": inv.get("sigma_y"),
                "Y_support_shape": inv.get("Y_support_shape"),
                "physical_simulator_called": inv.get("physical_simulator_called", False),
            },
            flush=True,
        )
        print(
            f"[{sys_name}] verdict={summary['verdict']} "
            f"sanity={san.get('status')} "
            f"J_m={summary['J_myopic']:.4f} J_p={summary['J_planning_eval']:.4f} "
            f"Δ_nonmyopic={summary['Delta_nonmyopic']['mean']:.4f} "
            f"({summary['Delta_nonmyopic']['ci_verdict']}) "
            f"Δ_adapt={summary['Delta_adapt']['mean']:.4f} "
            f"({summary['Delta_adapt']['ci_verdict']}) "
            f"B_meaningful={summary['max_branching_B']} "
            f"(raw={summary.get('max_branching_B_raw')}) "
            f"elapsed={time.time()-t_sys:.1f}s",
            flush=True,
        )

    md_path, json_path = _write_reports(summaries)
    print("\n========== CONSOLE SUMMARY ==========")
    for s in summaries:
        san = s.get("planner_sanity", {})
        print(
            f"{s['system']}: {s['verdict']} | sanity={san.get('status')} | "
            f"G4_branch={s['G4']} G5_route={s['G5']} "
            f"G6_nonmyopic={s['G6']} G7_adapt={s['G7']}"
        )
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print(f"Report: {md_path}")
    print(f"JSON:   {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
