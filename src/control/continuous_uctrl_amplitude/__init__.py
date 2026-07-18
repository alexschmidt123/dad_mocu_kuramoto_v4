"""Continuous u_ctrl + history-dependent amplitude adaptive-value diagnostic.

Scientific methods are not trained here. This study:
  1. Documents U-bank generation and freezes a continuous terminal rule
     (u_ctrl = Q_{1-α}(U|w) + margin, no snap_up).
  2. Tests whether existing 6 amplitudes show meaningful history-dependent
     specialization on IEEE5/IEEE9 T=3.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "continuous_uctrl_amplitude_adaptive_value"

__all__ = ["OUT", "ROOT"]
