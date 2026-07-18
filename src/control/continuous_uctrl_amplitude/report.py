"""Final reports for the continuous u_ctrl amplitude study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.control.continuous_uctrl_amplitude import OUT
from src.control.continuous_uctrl_amplitude.audit import U_BANK_AUDIT


def write_final_reports(summaries: list[dict[str, Any]]) -> Path:
    summary_dir = OUT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    by = {s["system"]: s for s in summaries}

    for system, s in by.items():
        path = summary_dir / f"{system}_amplitude_report.md"
        path.write_text(_system_report(s), encoding="utf-8")
        # also under system tree
        sys_sum = OUT / f"{system}_T3" / "summary"
        sys_sum.mkdir(parents=True, exist_ok=True)
        (sys_sum / "amplitude_report.md").write_text(_system_report(s), encoding="utf-8")

    final = summary_dir / "final_report.md"
    final.write_text(_final_report(by), encoding="utf-8")
    return final


def _system_report(s: dict[str, Any]) -> str:
    lines = [
        f"# {s['system']} amplitude adaptive-value report",
        "",
        f"- Case: **{s['case']}** ({s.get('case_note', '')})",
        f"- Histories (h1): {s['number_of_histories']}",
        f"- Design space: {s['number_of_amplitudes']} amplitudes × "
        f"{s['number_of_valid_buses']} buses = {s['n_designs']} "
        f"(duration={s['probe_duration_sec']} s from config)",
        f"- Dominant amplitude: {s['dominant_amplitude']} "
        f"(fraction {s['dominant_amplitude_fraction']:.3f})",
        f"- Unique optimal amplitudes (continuous): "
        f"{s['number_of_unique_optimal_amplitudes']}",
        f"- Unique optimal amplitudes (snapped): "
        f"{s['unique_optimal_amplitudes_snapped']}",
        f"- Mean best−second gap continuous: {s['mean_best_second_gap']:.6g}",
        f"- Mean best−second gap snapped: {s['mean_best_second_gap_snapped']:.6g}",
        f"- Mean wrong-amplitude regret: {s['wrong_amplitude_regret']['mean']:.6g}",
        f"- Mean cross-history amplitude regret: "
        f"{s['cross_history_amplitude_regret']['mean']:.6g}",
        f"- Continuous terminal-rule hash: `{s['continuous_terminal_rule_hash']}`",
        "",
    ]
    return "\n".join(lines)


def _final_report(by: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Continuous u_ctrl + amplitude adaptive-value — final report",
        "",
        "## Validity of continuous u_ctrl",
        "",
        f"- U_n nature: `{U_BANK_AUDIT['U_n_nature']}`",
        f"- Continuous u_ctrl status: **{U_BANK_AUDIT['continuous_u_ctrl_status']}**",
        f"- Physically validated continuous intermediates: "
        f"**{U_BANK_AUDIT['physically_validated']}**",
        "",
        U_BANK_AUDIT["validation_note"],
        "",
        "Primary definition for this study:",
        "",
        "    u_ctrl = Q_{1-α}(U|w) + margin   (no snap_up)",
        "",
        "Historical snapped control retained only as `u_ctrl_snapped` diagnostic.",
        "Safety constraints (ROCOF, nadir) and calibrated margin are unchanged.",
        "",
        "## Design space (from repository configs)",
        "",
        "Amplitudes remain the existing six options; duration is the configured",
        "probe duration (0.2 s), not expanded. Buses = all system buses.",
        "",
    ]
    for system in ("ieee5", "ieee9"):
        if system not in by:
            continue
        s = by[system]
        lines += [
            f"### {system}",
            "",
            f"- N_design = {s['n_designs']} "
            f"({s['number_of_amplitudes']} × {s['number_of_valid_buses']})",
            f"- Case **{s['case']}**: {s.get('case_note', '')}",
            f"- Dominant A* fraction: {s['dominant_amplitude_fraction']:.3f}",
            f"- Unique A* (continuous / snapped): "
            f"{s['number_of_unique_optimal_amplitudes']} / "
            f"{s['unique_optimal_amplitudes_snapped']}",
            f"- Mean wrong-amplitude regret: {s['wrong_amplitude_regret']['mean']:.6g}",
            f"- Continuous vs snapped J std: "
            f"{s['continuous_J_std']:.6g} / {s['snapped_J_std']:.6g}",
            "",
        ]

    lines += [
        "## Answers to Part XVII",
        "",
    ]
    ieee5 = by.get("ieee5")
    ieee9 = by.get("ieee9")

    def _q_var(s: dict[str, Any] | None) -> str:
        if not s:
            return "n/a"
        return (
            f"continuous std={s['continuous_J_std']:.6g}, "
            f"snapped std={s['snapped_J_std']:.6g}, "
            f"unique J continuous/snapped="
            f"{s['continuous_unique_J']}/{s['snapped_unique_J']}"
        )

    lines.append(
        "1. **After removing snap_up, how much more variable is terminal u_ctrl?**  "
    )
    lines.append(f"   IEEE5: {_q_var(ieee5)}. IEEE9: {_q_var(ieee9)}.")
    lines.append("")
    lines.append(
        "2. **Does continuous u_ctrl produce larger objective gaps among designs?**  "
    )
    if ieee5 and ieee9:
        lines.append(
            f"   IEEE5 mean best−second continuous={ieee5['mean_best_second_gap']:.6g} "
            f"vs snapped={ieee5['mean_best_second_gap_snapped']:.6g}. "
            f"IEEE9 continuous={ieee9['mean_best_second_gap']:.6g} "
            f"vs snapped={ieee9['mean_best_second_gap_snapped']:.6g}."
        )
    lines.append("")
    lines.append(
        "3. **Do different histories select different optimal amplitudes?**  "
    )
    for name, s in (("IEEE5", ieee5), ("IEEE9", ieee9)):
        if s:
            lines.append(
                f"   {name}: unique A*={s['number_of_unique_optimal_amplitudes']}, "
                f"non-dominant fraction="
                f"{s['fraction_histories_with_non_dominant_amplitude']:.3f}."
            )
    lines.append("")
    lines.append("4. **Systematic or mostly random/tied?**  ")
    lines.append(
        "   Interpret via case labels and near-zero regrets: if Case A/B, "
        "changes are nominal/tied rather than practically meaningful."
    )
    lines.append("")
    lines.append("5. **Does preferred amplitude change with bus held fixed?**  ")
    lines.append(
        "   See `results/bus_conditional_amplitude.csv` "
        "(per-bus A*(h,b) diversity)."
    )
    lines.append("")
    lines.append("6. **Regret of wrong amplitude?**  ")
    for name, s in (("IEEE5", ieee5), ("IEEE9", ieee9)):
        if s:
            r = s["wrong_amplitude_regret"]
            lines.append(
                f"   {name}: mean={r['mean']:.6g}, p95={r['p95']:.6g}, max={r['max']:.6g}."
            )
    lines.append("")
    lines.append("7. **Does the globally dominant amplitude perform nearly as well?**  ")
    for name, s in (("IEEE5", ieee5), ("IEEE9", ieee9)):
        if s:
            r = s["dominant_amplitude_regret"]
            lines.append(f"   {name}: mean dominant-amp regret={r['mean']:.6g}.")
    lines.append("")
    lines.append("8. **Does Fixed plan amplitude perform nearly as well?**  ")
    for name, s in (("IEEE5", ieee5), ("IEEE9", ieee9)):
        if s:
            r = s["fixed_amplitude_regret"]
            lines.append(
                f"   {name}: Fixed amp={s.get('fixed_plan_amplitude')}, "
                f"mean regret={r['mean']:.6g}."
            )
    lines.append("")
    lines.append(
        "9. **Does continuous u_ctrl reveal adaptive amplitude value hidden by snap_up?**  "
    )
    lines.append(
        "   Compare unique A* and gaps continuous vs snapped in the system tables above; "
        "Case D only if continuous shows meaningful regret while snapped does not."
    )
    lines.append("")
    lines.append("10. **IEEE5 amplitude adaptivity:**  ")
    if ieee5:
        label = {
            "A": "low",
            "B": "nominal only",
            "C": "practically meaningful",
            "D": "practically meaningful (revealed by continuous)",
            "E": "low",
        }.get(ieee5["case"], ieee5["case"])
        lines.append(f"   **{label}** (Case {ieee5['case']}).")
    lines.append("")
    lines.append("11. **IEEE9 amplitude adaptivity:**  ")
    if ieee9:
        label = {
            "A": "low",
            "B": "nominal only",
            "C": "practically meaningful",
            "D": "practically meaningful (revealed by continuous)",
            "E": "low",
        }.get(ieee9["case"], ieee9["case"])
        lines.append(f"   **{label}** (Case {ieee9['case']}).")
    lines.append("")
    lines.append("12. **Is a finer 0.01 amplitude grid justified next?**  ")
    recommend = False
    for s in by.values():
        if s["case"] in ("C", "D") and s["wrong_amplitude_regret"]["mean"] > 5e-4:
            recommend = True
    if recommend:
        lines.append(
            "   **Yes — candidate for a follow-up study.** Existing six amplitudes "
            "already show meaningful history-dependent specialization under continuous "
            "u_ctrl. Do **not** generate the 0.01 grid in this experiment."
        )
    else:
        lines.append(
            "   **Not yet.** Preference barely changes and/or wrong-amplitude regret "
            "is approximately zero under the existing six amplitudes. Increasing "
            "amplitude resolution alone may not materially raise intrinsic adaptive "
            "value. Do **not** generate the 0.01 grid based on this study."
        )
    lines.append("")
    lines.append("## Decision rule outcome")
    lines.append("")
    lines.append(
        "This study is diagnostic only. No DAD/RL-sBOED retraining and no amplitude "
        "grid expansion were performed."
    )
    lines.append("")
    return "\n".join(lines) + "\n"
