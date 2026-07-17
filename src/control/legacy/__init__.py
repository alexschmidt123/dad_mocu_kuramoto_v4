"""Frozen / one-shot experiment runners (not the production control API).

Production ``src/control/`` keeps shared inference, banks, safety, pilot, and
method helpers. IEEE5-specific historical experiment orchestrators live here so
they do not look like active scientific methods.
"""

from __future__ import annotations

__all__ = [
    "ieee5_t3",
    "ieee5_t4",
    "ieee5_t4_fixed_exact",
    "ieee5_horizon_sweep",
    "policy_robust_calibration",
    "diagnose_myopic_fixed",
    "myopic_convergence",
    "adaptive_value_diagnosis",
    "objective_adaptive_value",
]
