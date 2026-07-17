"""Objective-based DAD vs RL-sBOED study (IEEE5/IEEE9 T=3).

Scientific methods: DAD, RL-sBOED, Myopic, Fixed, Random.
"""

from __future__ import annotations

__all__ = ["OUT", "ROOT"]

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "objective_rl_sboed"
