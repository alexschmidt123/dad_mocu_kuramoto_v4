"""Audit ordered vs unordered Fixed sequence semantics."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from src.objectives.mocu.context import _score_fixed_subset

from tools.adaptive_advantage.config import SuiteConfig
from tools.adaptive_advantage.loaders import SystemBank

from .common import design_dict, make_audit_crn, terminal_after_sequence


def audit_fixed_scoring_sorts_subset() -> dict[str, Any]:
    src = inspect.getsource(_score_fixed_subset)
    sorts = "tuple(sorted(" in src or "sorted(int(a)" in src
    return {
        "production_score_fixed_subset_sorts_actions": bool(sorts),
        "docstring_claim": "unordered probe subset (order sorted)",
        "implication": (
            "Fixed search scores are order-invariant by construction. "
            "Selecting (a,b) vs (b,a) during search does not change the "
            "support score; only the stored evaluation order matters."
        ),
        "source_snippet_contains_sorted": bool(sorts),
    }


def audit_order_on_eval(
    bank: SystemBank,
    cfg: SuiteConfig,
    *,
    pairs: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Compare J_fixed(a,b) vs J_fixed(b,a) under identical CRN."""
    crn = make_audit_crn(bank, cfg)
    if pairs is None:
        # Sample diverse pairs + the reported Fixed sequence if present.
        rng = np.random.default_rng(cfg.seed + 77)
        pairs = []
        for _ in range(20):
            a, b = rng.choice(bank.n_designs, size=2, replace=False)
            pairs.append((int(a), int(b)))
        # Include a few systematic neighbors.
        for a in range(min(5, bank.n_designs)):
            for b in range(a + 1, min(a + 4, bank.n_designs)):
                pairs.append((a, b))
        # Dedup
        pairs = sorted(set(pairs))

    rows = []
    max_abs = 0.0
    for a, b in pairs:
        u_ab = np.empty((bank.n_eval, cfg.noise_replicates), dtype=np.float64)
        u_ba = np.empty_like(u_ab)
        for i in range(bank.n_eval):
            for r in range(cfg.noise_replicates):
                eps = crn.eps_obs[i, r]
                u_ab[i, r] = terminal_after_sequence(bank, i, (a, b), eps)
                u_ba[i, r] = terminal_after_sequence(bank, i, (b, a), eps)
        dab = float(np.mean(u_ab - u_ba))
        max_abs = max(max_abs, abs(dab), float(np.max(np.abs(u_ab - u_ba))))
        rows.append(
            {
                "a": a,
                "b": b,
                "design_a": design_dict(bank, a),
                "design_b": design_dict(bank, b),
                "mean_J_ab": float(u_ab.mean()),
                "mean_J_ba": float(u_ba.mean()),
                "mean_J_ab_minus_ba": dab,
                "max_abs_scenario_diff": float(np.max(np.abs(u_ab - u_ba))),
                "frac_equal": float(np.mean(np.isclose(u_ab, u_ba))),
            }
        )
    rows = sorted(rows, key=lambda x: -abs(x["mean_J_ab_minus_ba"]))
    return {
        "n_pairs_tested": len(rows),
        "max_abs_mean_order_difference": float(
            max(abs(r["mean_J_ab_minus_ba"]) for r in rows)
        ),
        "max_abs_scenario_order_difference": float(max_abs),
        "top_order_sensitive_pairs": rows[:10],
        "conclusion": (
            "Ordered sequences can differ under CRN evaluation even though "
            "production Fixed *search scoring* sorts subsets."
            if max_abs > 1e-12
            else "No order sensitivity detected on tested pairs."
        ),
    }
