"""Terminal control: posterior → ψ*, CUDA sim, myopic/fixed selectors.

  posterior_ctrl.py — discrete-θ belief + Yoon IBR ψ*(h_T)
  terminal_rule.py  — frozen calibrated rule shared by all methods
  u_req.py          — control spec (limits, contingency, u grid)
  cuda_control.py   — PyCUDA terminal-control simulation
  oracle_u_ctrl.py  — true-θ oracle u_ctrl cache
  myopic.py / fixed_search.py — baseline selectors

U-bank generation lives in ``src.banks.control_u`` / ``src.banks.generate_control``.
Import submodules directly to avoid circular imports.
"""

from __future__ import annotations

__all__: list[str] = []
