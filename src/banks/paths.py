"""Shared dataset path helpers (no experiment-type suffix)."""

from __future__ import annotations

from pathlib import Path

from src.config import SBOEDConfig, repo_root

DATA_ROOT = "data"


def system_name_for_data(cfg: SBOEDConfig) -> str:
    """Canonical system key for shared data folders (e.g. ``ieee5``)."""
    sys_sec = cfg.raw.get("system") or {}
    if isinstance(sys_sec, dict) and sys_sec.get("name"):
        return str(sys_sec["name"])
    return str(cfg.run_slug)


def resolve_shared_data_dir(
    project_root: Path | None,
    cfg: SBOEDConfig,
) -> Path:
    """
    Single data directory for a system — shared across experiment types / methods.

    Priority:
      1. ``data.dataset_dir`` in the YAML config
      2. ``data/<system.name>``
    """
    root = Path(project_root) if project_root is not None else repo_root()
    data_sec = dict(cfg.raw.get("data") or {})
    raw = data_sec.get("dataset_dir")
    if raw:
        p = Path(str(raw))
        return (p if p.is_absolute() else (root / p)).resolve()
    return (root / DATA_ROOT / system_name_for_data(cfg)).resolve()
