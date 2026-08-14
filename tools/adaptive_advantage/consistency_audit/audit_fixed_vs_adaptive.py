"""Document Fixed / Adaptive / Myopic policy definitions from code."""

from __future__ import annotations

from typing import Any


POLICY_DEFINITIONS: dict[str, Any] = {
    "objective": "J(pi)=E[u_ctrl(H_T)] with posterior-safe snapped control",
    "design": "xi={Amp, bus_location, duration}; duration fixed; actions are design indices",
    "MyopicControl": {
        "definition": (
            "At each step t, choose xi_t = argmin_a E_y[u_ctrl after observing y under a] "
            "under the current particle posterior, subject to no-repeat of design indices. "
            "In the diagnostic suite, step-0 may be fixed to the prior Myopic design for "
            "dominance sanity; step-1 is full-catalog myopic."
        ),
        "adaptive": True,
        "depends_on_y1_for_xi2": True,
    },
    "Fixed": {
        "definition": (
            "Choose an ordered sequence (xi_1, xi_2) with xi_1 != xi_2 before any "
            "observations. Execute xi_1 -> y_1, then xi_2 -> y_2 without changing xi_2 "
            "based on y_1."
        ),
        "adaptive": False,
        "depends_on_y1_for_xi2": False,
        "search_implementation_note": (
            "Production _score_fixed_subset sorts actions before scoring, so search "
            "scores are unordered-subset scores. The diagnostic select_fixed_T2 still "
            "stores an ordered pair for evaluation. This is a search/eval semantic gap."
        ),
    },
    "AdaptivePlanning_T2": {
        "definition": (
            "Choose xi_1 by minimizing E_y1[V1(xi_1,y1)] where "
            "V1=min_{xi2!=xi1} E_y2[u_ctrl]. After observing y1, choose "
            "xi_2*(y1)=argmin E_y2[u_ctrl]."
        ),
        "adaptive": True,
        "depends_on_y1_for_xi2": True,
    },
    "terminal_control": {
        "definition": (
            "u_raw = Q_(1-alpha)(U|posterior) + safety_margin; snap upward to u_grid."
        ),
        "implementation": "src.control.posterior_ctrl.posterior_safe_u_ctrl",
    },
    "no_repeat": {
        "rule": "xi_2 != xi_1 as full design indices (Amp+bus+duration)",
        "same_bus_different_amp": "different designs; not treated as repeats",
    },
}


def audit_policy_definitions() -> dict[str, Any]:
    return dict(POLICY_DEFINITIONS)
