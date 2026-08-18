"""On-disk banks under top-level ``data/`` — generate, load, and audit.

  paths.py            — ``data/<system>/`` location
  power_grid.py       — physical Δf / duration banks (IEEE, etc.)
  tables.py           — table (Foster) probe banks
  control_u.py        — ψ* / U-bank I/O and per-split generation
  generate_control.py — certify / write the control bank for a dataset
  quality.py          — bank completeness / quality gates
  audit.py            — design-redundancy / adaptive-room audit
  diagnose_control.py / diagnose_duration.py — CLI diagnostics
"""

from __future__ import annotations

from src.banks.paths import DATA_ROOT, resolve_shared_data_dir, system_name_for_data

__all__ = ["DATA_ROOT", "resolve_shared_data_dir", "system_name_for_data"]
