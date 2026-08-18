"""Summary markdown, comparison tables, and T-sweep plots.

Experiment folder names / run_config live in ``src.layout``.
This package only writes the human-readable artifacts:

  summary.py — one-run ``summary.md``
  tables.py  — per-rollout and method-comparison CSVs
  plots.py   — T-sweep metric/time bundle (``python -m src.results.plots``)
"""

from __future__ import annotations

__all__: list[str] = []
