"""Branching-value helpers (LEVEL 1–3)."""

from __future__ import annotations

from typing import Any


def summarize_branching_levels(ieee5_exact: dict[str, Any]) -> dict[str, Any]:
    b = ieee5_exact["adaptive"]["best_first_branching"]
    delta = ieee5_exact["Delta_adapt_exact"]
    return {
        "LEVEL_1_raw_branching": {
            "B_raw": b["B_raw"],
            "description": "Number of distinct argmin second design IDs across branches",
        },
        "LEVEL_2_value_meaningful_branching": {
            "B_meaningful": b["B_meaningful"],
            "max_continuation_gap": b["max_continuation_gap"],
            "mean_continuation_gap": b["mean_continuation_gap"],
            "frac_meaningful_gap": b["frac_meaningful_gap"],
            "meaningful_gap_eps": ieee5_exact["adaptive"]["meaningful_gap_eps"],
            "switch_penalties_sample": b["switch_penalties_sample"],
            "description": (
                "continuation_gap = V_second - V_best; meaningful if > eps. "
                "Switch penalties measure cost of forcing the wrong continuation."
            ),
        },
        "LEVEL_3_realized_adaptive_advantage": {
            "Delta_adapt_exact_mean": delta["mean"],
            "Delta_adapt_ci": [delta["ci_low"], delta["ci_high"]],
            "Delta_branching_given_first_V": b.get("Delta_branching_given_first_V"),
            "Delta_branching_given_first_realized_u": b.get(
                "Delta_branching_given_first_realized_u"
            ),
            "description": (
                "Overall Fixed_best - Adaptive_best on realized CRN u; same-first "
                "branching advantage reported in V1 selection space (primary) and "
                "realized-u space (secondary; noisier)."
            ),
        },
    }
