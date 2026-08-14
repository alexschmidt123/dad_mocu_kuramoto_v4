"""Separate raw approximate planner vs Myopic fallback reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.adaptive_advantage.config import RESULTS_DIR


def _load_summary(system: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / f"{system}_summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def audit_planner_reporting(system: str) -> dict[str, Any]:
    """Read existing diagnostic summary and disambiguate planner reporting."""
    s = _load_summary(system)
    if s is None:
        return {
            "system": system,
            "status": "MISSING_SUMMARY",
            "note": f"No {system}_summary.json under diagnostics results/",
        }

    san = s.get("planner_sanity", {})
    j_myopic = float(s["J_myopic"])
    j_reported_planning = float(s["J_planning_eval"])
    j_raw = san.get("J_planning_selected_first", j_reported_planning)
    if j_raw is None:
        j_raw = j_reported_planning
    j_raw = float(j_raw)

    used_id = san.get("planning_first_design_id_used")
    support_id = san.get("support_J2_best_first_design_id")
    note = san.get("first_selection_note", "")
    fallback_used = bool(
        "myopic-first" in str(note).lower()
        or (
            used_id is not None
            and support_id is not None
            and int(used_id) != int(support_id)
            and abs(j_reported_planning - j_myopic) <= 1e-9
            and j_raw > j_myopic + 1e-9
        )
    )

    j_best_available = float(min(j_raw, j_myopic))

    return {
        "system": system,
        "adaptive_search": s.get("plan_meta", {}).get("Adaptive_search"),
        "fixed_search": s.get("fixed_meta", {}).get("Adaptive_search"),
        "J_ApproxPlanningRaw": j_raw,
        "J_Myopic": j_myopic,
        "J_BestAvailableAdaptive": j_best_available,
        "J_FixedExact_or_reported": float(s["J_fixed"]),
        "J_Planning_as_previously_reported": j_reported_planning,
        "myopic_fallback_used": fallback_used,
        "BestAvailableAdaptive_policy": (
            "Myopic fallback" if fallback_used else "Approx/exact planning first design"
        ),
        "support_J2_best_first_design_id": support_id,
        "planning_first_design_id_used": used_id,
        "first_selection_note": note,
        "interpretation": (
            "Do NOT interpret J_BestAvailableAdaptive == J_Myopic as evidence that "
            "the non-myopic approximate planner matched Myopic. When fallback is used, "
            "it only means the reporting baseline was protected by an admissible Myopic "
            "policy. Always cite J_ApproxPlanningRaw separately."
            if fallback_used
            else "No Myopic fallback detected in reported planning value."
        ),
        "candidate_audit": s.get("candidate_audit"),
        "verdict_from_diagnostic": s.get("verdict"),
    }


def audit_ieee9_ieee14_reporting() -> dict[str, Any]:
    return {
        "ieee9": audit_planner_reporting("ieee9"),
        "ieee14": audit_planner_reporting("ieee14"),
    }
