"""Terminal-control / MOCU math (u_req, posterior, rewards, myopic/fixed).

U-bank I/O lives in ``src.banks.control_u``. On-disk banks are under ``data/``.
Experiment CLI: ``python -m src.experiment``.

Import submodules directly (``src.control.u_req``, …) to avoid circular
imports with ``src.banks.control_u``.
"""

from __future__ import annotations

__all__: list[str] = []
