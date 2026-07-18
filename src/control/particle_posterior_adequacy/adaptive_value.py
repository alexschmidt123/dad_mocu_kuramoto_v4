"""Adaptive-value and case classification vs particle count (no DAD/RL training)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from src.control.legacy.objective_adaptive_value import classify_case
from src.control.particle_posterior_adequacy.supports import GLOBAL_HISTORY_SEED


NEAR_TIE_TOL = 1e-4
N_BOOT = 4000


def adaptive_gain_from_scores(
    scored_histories: list[dict[str, Any]],
    *,
    xi1: int | None = None,
) -> dict[str, Any]:
    """
    J_adaptive / J_common / Delta_adaptive from per-history snapped design scores.

    Each scored history must contain:
      history_id, xi1, optimal_design, J_star_snapped, scores_snapped (dict)
    """
    if not scored_histories:
        return {
            "xi1": xi1,
            "n_histories": 0,
            "J_adaptive": float("nan"),
            "J_common": float("nan"),
            "Delta_adaptive": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "n_unique_xi2_star": 0,
            "dominant_xi2_star": None,
            "dominant_fraction": float("nan"),
        }
    j_adaptive_values = np.asarray(
        [float(h["J_star_snapped"]) for h in scored_histories], dtype=np.float64
    )
    j_adaptive = float(j_adaptive_values.mean())
    actions = sorted(
        {
            int(a)
            for h in scored_histories
            for a in h["scores_snapped"].keys()
        }
    )
    common_means = {
        a: float(np.mean([float(h["scores_snapped"][a]) for h in scored_histories if a in h["scores_snapped"]]))
        for a in actions
    }
    # Only actions present in all histories
    common_means = {
        a: v
        for a, v in common_means.items()
        if all(a in h["scores_snapped"] for h in scored_histories)
    }
    if not common_means:
        common_action = int(scored_histories[0]["optimal_design"])
        j_common_values = j_adaptive_values.copy()
    else:
        common_action = min(common_means, key=lambda a: (common_means[a], a))
        j_common_values = np.asarray(
            [float(h["scores_snapped"][common_action]) for h in scored_histories],
            dtype=np.float64,
        )
    j_common = float(j_common_values.mean())
    paired = j_common_values - j_adaptive_values
    delta = float(paired.mean())
    rng = np.random.default_rng(GLOBAL_HISTORY_SEED + 91 + int(xi1 or 0))
    n = len(paired)
    boots = np.empty(min(N_BOOT, max(200, 20 * n)), dtype=np.float64)
    for i in range(boots.size):
        sample = rng.integers(0, n, size=n)
        boots[i] = float(paired[sample].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    stars = [int(h["optimal_design"]) for h in scored_histories]
    counts = Counter(stars)
    dominant, dominant_count = counts.most_common(1)[0]
    return {
        "xi1": xi1 if xi1 is not None else -1,
        "n_histories": len(scored_histories),
        "J_adaptive": j_adaptive,
        "J_common": j_common,
        "Delta_adaptive": delta,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "common_xi2": int(common_action),
        "n_unique_xi2_star": len(counts),
        "dominant_xi2_star": int(dominant),
        "dominant_fraction": dominant_count / len(scored_histories),
        "near_tie_rate": float(
            np.mean(
                [
                    sum(
                        abs(float(v) - float(h["J_star_snapped"])) <= NEAR_TIE_TOL
                        for v in h["scores_snapped"].values()
                    )
                    > 1
                    for h in scored_histories
                ]
            )
        ),
    }


def classify_bus_case_from_designs(design_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Lightweight BUS-A..E label from one-step design stability (no four-way decomp)."""
    rows = [r for r in design_rows if r.get("optimal_bus") is not None]
    if not rows:
        return {"bus_case_classification": "BUS-A", "case_note": "no_designs"}
    buses = [int(r["optimal_bus"]) for r in rows]
    amps = [float(r["optimal_amplitude"]) for r in rows]
    regrets = [
        float(r["reference_regret"])
        for r in rows
        if r.get("reference_regret") is not None
        and r["reference_regret"] == r["reference_regret"]
    ]
    counts = Counter(buses)
    dominant, dom_n = counts.most_common(1)[0]
    frac_non_dom = 1.0 - dom_n / max(len(buses), 1)
    bus_changes = len(counts) >= 2 and frac_non_dom > 0.05
    mean_regret = float(np.mean(regrets)) if regrets else 0.0
    median_regret = float(np.median(regrets)) if regrets else 0.0
    p95_regret = float(np.quantile(regrets, 0.95)) if regrets else 0.0
    meaningful = (median_regret > 1e-3) or (p95_regret > 2e-2 and mean_regret > 5e-3)
    if bus_changes and meaningful:
        # Without four-way terminal decomp CI, do not promote to BUS-C.
        case = "BUS-B"
        note = (
            "bus branches with non-trivial particle-reference regret, but four-way "
            "terminal decomp is not re-run here; label BUS-B (+ BUS-E for policy). "
            "Matches prior study rule that terminal adaptive-bus value must beat Fixed."
        )
    elif bus_changes:
        case = "BUS-B"
        note = "nominal bus branching with near-zero reference regret; also BUS-E"
    else:
        case = "BUS-A"
        note = "same bus preferred almost always; also BUS-E"
    amp_counts = Counter(amps)
    return {
        "bus_case_classification": case,
        "case_note": note,
        "dominant_bus": int(dominant),
        "dominant_bus_fraction": dom_n / max(len(buses), 1),
        "n_unique_buses": len(counts),
        "n_unique_amplitudes": len(amp_counts),
        "mean_reference_regret": mean_regret,
        "median_reference_regret": median_regret,
        "p95_reference_regret": p95_regret,
    }


def summarize_adaptive_for_grid(
    *,
    system: str,
    particle_count: int,
    support_seed: int,
    history_scores: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    fixed_objective: float | None,
) -> dict[str, Any]:
    """Pool histories (h1) for overall Delta; also per-xi1 for Case A–D."""
    h1 = [h for h in history_scores if int(h.get("history_step", 1)) == 1]
    overall = adaptive_gain_from_scores(h1, xi1=None)
    by_xi1: dict[int, list[dict[str, Any]]] = {}
    for h in h1:
        if h.get("xi1") is None:
            continue
        by_xi1.setdefault(int(h["xi1"]), []).append(h)
    gain_rows = [
        adaptive_gain_from_scores(rows, xi1=xi1)
        for xi1, rows in sorted(by_xi1.items())
        if len(rows) >= 2
    ]
    if gain_rows:
        case = classify_case(gain_rows)
    else:
        changing = int(overall["n_unique_xi2_star"]) > 1
        significant = (
            float(overall["ci95_low"]) > 0
            and float(overall["Delta_adaptive"]) > 1e-4
        )
        if not significant and not changing:
            case = "A"
        elif changing and not significant:
            case = "B"
        elif significant:
            case = "C"
        else:
            case = "B"
    bus = classify_bus_case_from_designs(
        [r for r in design_rows if int(r.get("history_step", 1)) == 1]
    )
    return {
        "system": system,
        "particle_count": particle_count,
        "support_seed": support_seed,
        "J_adaptive": overall["J_adaptive"],
        "J_common": overall["J_common"],
        "Delta_adaptive": overall["Delta_adaptive"],
        "ci95_low": overall["ci95_low"],
        "ci95_high": overall["ci95_high"],
        "Fixed_objective": fixed_objective,
        "adaptive_reference_objective": overall["J_adaptive"],
        "case_classification": case,
        "bus_case_classification": bus["bus_case_classification"],
        "bus_case_note": bus["case_note"],
        "n_histories": overall["n_histories"],
        "n_xi1_groups": len(gain_rows),
        "n_unique_optimal_design": overall["n_unique_xi2_star"],
    }
