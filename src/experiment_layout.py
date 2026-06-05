"""Paths and writers for the standard experiment folder layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.config import SBOEDConfig

RUN_CONFIG_FILENAME = "run_config.yaml"
LEGACY_DATA_DIR_POINTER = "data_dir.txt"
MODEL_SUBDIR = "model"
EVAL_SUBDIR = "eval"
EVAL_SUMMARY_FILENAME = "summary.json"


def model_dir(exp_dir: Path) -> Path:
    return exp_dir / MODEL_SUBDIR


def eval_dir(exp_dir: Path) -> Path:
    return exp_dir / EVAL_SUBDIR


def run_config_path(exp_dir: Path) -> Path:
    return exp_dir / RUN_CONFIG_FILENAME


def eval_summary_path(exp_dir: Path) -> Path:
    return eval_dir(exp_dir) / EVAL_SUMMARY_FILENAME


def eval_method_path(exp_dir: Path, method: str) -> Path:
    return eval_dir(exp_dir) / f"{method}.json"


def load_run_config_doc(exp_dir: Path) -> dict[str, Any]:
    path = run_config_path(exp_dir)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_linked_data_dir(exp_dir: Path) -> Path:
    """
    Shared table path from ``run_config.yaml`` field ``data_dir``.

    Falls back to legacy ``data_dir.txt`` for older experiment folders.
    """
    exp_dir = exp_dir.resolve()
    doc = load_run_config_doc(exp_dir)
    if doc.get("data_dir"):
        return Path(str(doc["data_dir"])).resolve()

    legacy = exp_dir / LEGACY_DATA_DIR_POINTER
    if legacy.is_file():
        return Path(legacy.read_text(encoding="utf-8").strip()).resolve()

    raise FileNotFoundError(
        f"No data_dir in {run_config_path(exp_dir)} (and no legacy {LEGACY_DATA_DIR_POINTER})"
    )


def write_run_config(exp_dir: Path, cfg: SBOEDConfig, data_path: Path) -> Path:
    """
    Single experiment record: effective YAML, ``step_number``, and ``data_dir``.
    """
    path = run_config_path(exp_dir)
    exp = dict(cfg.raw.get("experiment") or {})
    exp["step_number"] = int(cfg.step_number)
    body = dict(cfg.raw)
    body["experiment"] = exp
    doc: dict[str, Any] = {
        "step_number": int(cfg.step_number),
        "source_config": str(cfg.config_path.resolve()),
        "data_dir": str(data_path.resolve()),
        **body,
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)

    legacy_ptr = exp_dir / LEGACY_DATA_DIR_POINTER
    if legacy_ptr.is_file():
        legacy_ptr.unlink()

    return path


def resolve_experiment_config_path(exp_dir: Path) -> Path:
    """``run_config.yaml`` for this run, else legacy experiment YAML names."""
    exp_dir = exp_dir.resolve()
    run_cfg = run_config_path(exp_dir)
    if run_cfg.is_file():
        return run_cfg
    legacy = exp_dir / "config.yaml"
    if legacy.is_file():
        return legacy
    pointer = exp_dir / "config_source.txt"
    if pointer.is_file():
        name = pointer.read_text(encoding="utf-8").strip()
        candidate = exp_dir / name
        if candidate.is_file():
            return candidate.resolve()
    yamls = sorted(p for p in exp_dir.glob("*.yaml") if p.name != "manifest.yaml")
    if len(yamls) == 1:
        return yamls[0].resolve()
    if not yamls:
        raise FileNotFoundError(f"No run_config.yaml in {exp_dir}")
    raise FileNotFoundError(f"Ambiguous YAML in {exp_dir}: {[p.name for p in yamls]}")


def load_eval_summary(exp_dir: Path) -> dict[str, Any]:
    """Read ``eval/summary.json``, with legacy ``results.json`` fallback."""
    import json

    exp_dir = exp_dir.resolve()
    primary = eval_summary_path(exp_dir)
    if primary.is_file():
        with primary.open(encoding="utf-8") as f:
            return json.load(f)
    legacy = exp_dir / "results.json"
    if legacy.is_file():
        with legacy.open(encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"No eval/summary.json in {exp_dir}")
