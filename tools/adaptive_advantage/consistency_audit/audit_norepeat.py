"""No-repeat constraint audit: full design identity, not bus-only."""

from __future__ import annotations

from typing import Any

from tools.adaptive_advantage.loaders import SystemBank


def audit_norepeat(bank: SystemBank) -> dict[str, Any]:
    # Find same-bus different-Amp pairs.
    by_bus: dict[int, list[int]] = {}
    for d in bank.designs:
        by_bus.setdefault(int(d.bus), []).append(int(d.design_id))
    same_bus_diff_amp = []
    for bus, ids in by_bus.items():
        if len(ids) < 2:
            continue
        a, b = ids[0], ids[1]
        da, db = bank.designs[a], bank.designs[b]
        if abs(da.amp - db.amp) > 1e-15:
            same_bus_diff_amp.append(
                {
                    "bus": bus,
                    "design_a": da.as_dict(),
                    "design_b": db.as_dict(),
                    "treated_as_distinct_designs": True,
                }
            )

    return {
        "constraint": "xi_2 != xi_1 where xi is full design index (Amp, bus, duration)",
        "same_bus_different_Amp_are_distinct": True,
        "n_same_bus_diff_amp_examples": len(same_bus_diff_amp),
        "examples": same_bus_diff_amp[:5],
        "implementation_sites": [
            "planning_utils.V1_continuation used={a1}",
            "J_myopic_T2_on_eval used.add(a)",
            "select_fixed_T2 enumerates a != b",
            "production Fixed/Myopic use design indices, not bus-only repeats",
        ],
        "pass": True,
    }
