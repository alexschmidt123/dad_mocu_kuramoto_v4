"""Terminal control objective: U-bank, posterior-safe u_ctrl, control simulation.

Production modules live in this package. One-shot IEEE5 historical experiment
runners are under ``src.control.legacy`` and are not manuscript methods.
"""

from src.control.banks import extract_U_bank, validate_control_invariants
from src.control.posterior_ctrl import (
    posterior_control_decision,
    posterior_safe_u_ctrl,
    posterior_u_raw,
    weighted_quantile,
)
from src.control.u_req import ControlSpec, is_control_safe

__all__ = [
    "ControlSpec",
    "extract_U_bank",
    "is_control_safe",
    "posterior_control_decision",
    "posterior_safe_u_ctrl",
    "posterior_u_raw",
    "validate_control_invariants",
    "weighted_quantile",
]
