"""Thin CLI shim — implementation lives in ``src.banks.diagnose_control``."""

from __future__ import annotations

from src.banks.diagnose_control import (  # noqa: F401
    control_bank_nondegenerate,
    diagnose_control_objective,
    diagnose_split,
)

__all__ = [
    "control_bank_nondegenerate",
    "diagnose_control_objective",
    "diagnose_split",
]
