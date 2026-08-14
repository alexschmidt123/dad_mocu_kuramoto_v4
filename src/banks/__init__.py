"""Bank I/O, generation helpers, and core audits.

On-disk artifacts live under the top-level ``data/`` folder; this package is
the code that builds, loads, and audits those banks.
"""

from __future__ import annotations

from src.banks.paths import DATA_ROOT, resolve_shared_data_dir, system_name_for_data

__all__ = ["DATA_ROOT", "resolve_shared_data_dir", "system_name_for_data"]
