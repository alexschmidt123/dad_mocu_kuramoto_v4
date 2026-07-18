"""Reports for bus + joint adaptive-value diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.control.bus_joint_adaptive_value import OUT
from src.control.continuous_uctrl_amplitude.audit import U_BANK_AUDIT


def write_final_reports(summaries: list[dict[str, Any]]) -> Path:
    summary_dir = OUT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    by = {s["system"]: s for s in summaries}
    for system, s in by.items():
        text = _system_report(s)
        (summary_dir / f"{system}_bus_report.md").write_text(text, encoding="utf-8")
        sys_dir = OUT / f"{system}_T3" / "summary"
        sys_dir.mkdir(parents=True, exist_ok=True)
        (sys_dir / "bus_report.md").write_text(text, encoding="utf-8")
    path = summary_dir / "final_report.md"
    path.write_text(_final_report(by), encoding="utf-8")
    return path


def _system_report(s: dict[str, Any]) -> str:
    d = s.get("decomposition") or {}
    lines = [
        f"# {s['system']} bus + joint adaptive-value report",
        "",
        f"- Case: **{s['case']}** — {s.get('case_note', '')}",
        f"- Histories (h1): {s['number_of_histories']}",
        f"- Design: {s['number_of_amplitudes']} amps × {s['number_of_valid_buses']} buses "
        f"= {s['n_designs']} (duration={s['probe_duration_sec']} s)",
        f"- Dominant bus: {s['dominant_bus']} (fraction {s['dominant_bus_fraction']:.3f})",
        f"- Unique optimal buses: {s['number_of_unique_optimal_buses']}",
        f"- Mean wrong-bus regret (cont): {s['mean_wrong_bus_regret']:.6g}",
        f"- Mean wrong-bus regret (snap): {s['wrong_bus_regret_snap']['mean']:.6g}",
        f"- Prior wrong-amplitude regret: {s['prior_wrong_amplitude_regret_mean']:.6g}",
        f"- Mean Fixed-bus regret: {s['mean_fixed_bus_regret']:.6g}",
        "",
        "## Four-way decomposition (continuous terminal)",
        "",
    ]
    for row in d.get("rows") or []:
        lines.append(
            f"- {row['structure']}: mean u_cont={row['mean_u_cont']:.6g}, "
            f"mean u_snap={row['mean_u_snap']:.6g}"
        )
    lines.append("")
    for p in d.get("paired") or []:
        lines.append(
            f"- {p['comparison']}: {p['mean_diff']:+.6g} "
            f"CI95=[{p['ci95_low']:+.6g}, {p['ci95_high']:+.6g}]"
        )
    lines.append("")
    return "\n".join(lines)


def _final_report(by: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Bus-location and joint bus–amplitude adaptive-value — final report",
        "",
        "## Context",
        "",
        "Prior amplitude study: IEEE5/IEEE9 = **Case B** (nominal amplitude branching,",
        "near-zero practical amplitude regret). No amplitude-grid expansion.",
        "No DAD/RL-sBOED retraining in this diagnostic.",
        "",
        f"Continuous `u_cont` is a **diagnostic** objective "
        f"({U_BANK_AUDIT['continuous_u_ctrl_status']}); "
        "snapped `u_ctrl` retained for comparison. "
        f"Physically validated continuous intermediates: "
        f"**{U_BANK_AUDIT['physically_validated']}**.",
        "",
        "## Per-system bus results",
        "",
    ]
    for system in ("ieee5", "ieee9"):
        if system not in by:
            continue
        s = by[system]
        lines += [
            f"### {system}",
            "",
            f"- **{s['case']}**: {s.get('case_note', '')}",
            f"- Unique b*: {s['number_of_unique_optimal_buses']}, "
            f"dominant fraction={s['dominant_bus_fraction']:.3f}",
            f"- Mean wrong-bus regret (cont/snap): "
            f"{s['mean_wrong_bus_regret']:.6g} / "
            f"{s['wrong_bus_regret_snap']['mean']:.6g}",
            f"- Mean best−second bus gap (cont/snap): "
            f"{s['mean_best_second_bus_gap']:.6g} / "
            f"{s['mean_best_second_bus_gap_snapped']:.6g}",
            f"- Prior wrong-amp regret: {s['prior_wrong_amplitude_regret_mean']:.6g}",
            "",
        ]
        for row in (s.get("decomposition") or {}).get("rows") or []:
            lines.append(
                f"  - {row['structure']}: u_cont={row['mean_u_cont']:.6g}"
            )
        lines.append("")

    ieee5, ieee9 = by.get("ieee5"), by.get("ieee9")

    def _label(case: str) -> str:
        return {
            "BUS-A": "low",
            "BUS-B": "nominal only",
            "BUS-C": "practically meaningful",
            "BUS-D": "meaningful (revealed by continuous)",
            "BUS-E": "low",
        }.get(case, case)

    lines += [
        "## Answers to Part XIX",
        "",
        "1. **Different buses preferred on IEEE5?**  ",
        (
            f"   Yes nominally — unique b*={ieee5['number_of_unique_optimal_buses']}, "
            f"non-dominant fraction={ieee5['fraction_non_dominant_bus']:.3f}."
            if ieee5
            else "   n/a"
        ),
        "",
        "2. **Different buses preferred on IEEE9?**  ",
        (
            f"   Unique b*={ieee9['number_of_unique_optimal_buses']}, "
            f"non-dominant fraction={ieee9['fraction_non_dominant_bus']:.3f}."
            if ieee9
            else "   n/a"
        ),
        "",
        "3. **Systematic or near-tied?**  ",
        "   Interpret via case labels and median/mean wrong-bus regret "
        "(median≈0 with tiny mean ⇒ near-tied / Case BUS-B).",
        "",
        "4. **Wrong-bus regret?**  ",
    ]
    for name, s in (("IEEE5", ieee5), ("IEEE9", ieee9)):
        if s:
            r = s["wrong_bus_regret_cont"]
            lines.append(
                f"   {name}: mean={r['mean']:.6g}, median={r['median']:.6g}, "
                f"p95={r['p95']:.6g}, max={r['max']:.6g}."
            )
    lines.append("")
    lines.append("5. **Wrong-bus vs prior wrong-amplitude regret?**  ")
    for name, s in (("IEEE5", ieee5), ("IEEE9", ieee9)):
        if s:
            lines.append(
                f"   {name}: bus={s['mean_wrong_bus_regret']:.6g} vs "
                f"amp={s['prior_wrong_amplitude_regret_mean']:.6g}."
            )
    lines.append("")
    lines.append("6. **Does bus contain more adaptive value than amplitude?**  ")
    lines.append(
        "   Compare the regrets above and four-way decomposition "
        "(Adaptive Bus + Fixed Amp vs Fixed Bus + Adaptive Amp)."
    )
    lines.append("")
    lines.append("7. **Did continuous resolution reveal bus value hidden by snap_up?**  ")
    lines.append(
        "   Only if Case BUS-D / partial-D note; otherwise continuous still low ⇒ BUS-E."
    )
    lines.append("")
    lines.append("8–10. **Four-way comparisons** — see per-system paired bootstrap in "
                 "`results/joint_decomposition.csv` and system reports.")
    lines.append("")
    lines.append("11. **Cause of low adaptive value?**  ")
    cases = [s["case"] for s in (ieee5, ieee9) if s]
    if all(c in ("BUS-A", "BUS-B") for c in cases):
        lines.append(
            "   **Both dimensions + overall experiment structure** under the current "
            "6×bus design: prior amplitude Case B; bus "
            f"{ieee5['case'] if ieee5 else '?'}/{ieee9['case'] if ieee9 else '?'} with "
            "four-way terminal structures ≈ Fully Fixed."
        )
    else:
        lines.append(
            "   Bus shows stronger one-step structure than amplitude, but check "
            "decomposition before attributing terminal adaptive value."
        )
    lines.append("")
    lines.append("12. **Meaningful adaptive value for DAD/RL-sBOED?**  ")
    meaningful = any(s and s["case"] in ("BUS-C", "BUS-D") for s in (ieee5, ieee9))
    if meaningful:
        lines.append(
            "   **Yes (bus-side, terminal)** — adaptive-bus structures beat Fixed; "
            "focus next training on history-dependent bus selection."
        )
    else:
        lines.append(
            "   **Not yet for policy training.** One-step bus gaps may exist, but "
            "Fully Fixed ≈ adaptive references on terminal u_ctrl. "
            "**Do not continue generic RL tuning.** Next: modify experimental design "
            "(probes / horizon / systems), not retrain DAD/RL-sBOED yet."
        )
    lines.append("")
    lines.append("## Decision rule outcome")
    lines.append("")
    if meaningful:
        lines.append(
            "Bus adaptivity is Case C/D on at least one system → focus future "
            "DAD/RL-sBOED on history-dependent bus selection."
        )
    else:
        lines.append(
            "Bus Case A/B and amplitude Case B, with four-way ≈ Fixed → current probe "
            "design space has **low intrinsic terminal adaptive value**. "
            "**Do not retrain DAD/RL-sBOED yet.**"
        )
    lines.append("")
    for name, s in (("IEEE5", ieee5), ("IEEE9", ieee9)):
        if s:
            lines.append(f"- {name} bus adaptivity: **{_label(s['case'])}** ({s['case']})")
    lines.append("")
    return "\n".join(lines)
