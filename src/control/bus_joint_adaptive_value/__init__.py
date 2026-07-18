"""Bus-location and joint bus-amplitude adaptive-value diagnostic.

Does NOT expand the amplitude grid and does NOT train DAD/RL-sBOED.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "bus_joint_adaptive_value"

__all__ = ["OUT", "ROOT"]
