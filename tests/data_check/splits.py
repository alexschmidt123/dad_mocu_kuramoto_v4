"""Data splits for information_redundancy checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import SBOEDConfig
from src.data import (
    _hydrate_systems,
    generate_split,
    get_systems,
    load_tables,
    save_json,
)

# Roles (see documents/DATA_GENERATION_AND_VALIDATION.txt):
#   train          — particle-support pool (TableThetaSupport subsample)
#   calibration    — scenario search (redundancy, Q2 planning, best-fixed selection)
#   certification  — independent hidden-truth evaluation (virtual noise replicas)
#   validation     — reserved for DAD pipeline
#   test           — reserved for DAD evaluation
REDUNDANCY_SPLITS = ("calibration", "certification", "train", "validation", "test")

SUPPORT_SPLIT = "train"
SEARCH_SPLIT = "calibration"
CERTIFICATION_SPLIT = "certification"

_DEFAULT_SPLIT_SIZES = {
    "calibration": 100,
    "certification": 100,
    "train": 400,
    "validation": 100,
    "test": 200,
}

_DEFAULT_SPLIT_SEEDS = {
    "calibration": 10,
    "certification": 30,
    "train": 0,
    "validation": 20,
    "test": 1,
}


def redundancy_split_config(cfg: SBOEDConfig) -> dict[str, Any]:
    g = dict(cfg.raw.get("information_redundancy_splits") or cfg.raw.get("gate_splits") or {})
    sizes = dict(_DEFAULT_SPLIT_SIZES)
    sizes.update(g.get("sizes") or {})
    seeds = dict(_DEFAULT_SPLIT_SEEDS)
    seeds.update(g.get("seeds") or {})
    roles = {
        "support": g.get("support_split", SUPPORT_SPLIT),
        "search": g.get("search_split", SEARCH_SPLIT),
        "certification": g.get("certification_split", CERTIFICATION_SPLIT),
    }
    return {"sizes": sizes, "seeds": seeds, "roles": roles}


def split_json_path(data_path: Path, split: str) -> Path:
    if split not in REDUNDANCY_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {REDUNDANCY_SPLITS}")
    return data_path / f"{split}.json"


def load_redundancy_split_systems(data_path: Path, split: str) -> list[dict]:
    path = split_json_path(data_path, split)
    if not path.is_file():
        fallback_map = {
            "calibration": "train",
            "certification": "test",
            "validation": "train",
        }
        fallback = fallback_map.get(split)
        if fallback and split_json_path(data_path, fallback).is_file():
            import warnings
            warnings.warn(
                f"Redundancy split {split!r} missing; falling back to {fallback!r} at {data_path}",
                stacklevel=2,
            )
            path = split_json_path(data_path, fallback)
        else:
            raise FileNotFoundError(
                f"Missing {split} split at {path}; run ensure_redundancy_splits first"
            )
    payload = load_tables(path)
    return _hydrate_systems(get_systems(payload), data_path.resolve())


def ensure_redundancy_splits(
    project_root: Path,
    cfg: SBOEDConfig,
    *,
    splits: tuple[str, ...] = REDUNDANCY_SPLITS,
    force: bool = False,
) -> Path:
    """Generate missing split JSON banks under data/<run_slug>/."""
    from src.data import is_present, resolve_data_path

    data_path = resolve_data_path(project_root, cfg)
    data_path.mkdir(parents=True, exist_ok=True)
    sc = redundancy_split_config(cfg)

    for split in splits:
        out = split_json_path(data_path, split)
        if out.is_file() and not force:
            continue
        size = int(sc["sizes"].get(split, _DEFAULT_SPLIT_SIZES.get(split, 100)))
        seed = int(sc["seeds"].get(split, _DEFAULT_SPLIT_SEEDS.get(split, 0)))
        print(f"  generating redundancy split {split}: n={size} seed={seed}")
        payload = generate_split(cfg, split, seed, theta_sample_size=size, data_dir=data_path)
        save_json(payload, out)

    if not is_present(data_path):
        train_out = data_path / "train.json"
        test_out = data_path / "test.json"
        if not train_out.is_file() and split_json_path(data_path, "train").is_file():
            import shutil
            shutil.copy2(split_json_path(data_path, "train"), train_out)
        if not test_out.is_file() and split_json_path(data_path, "test").is_file():
            import shutil
            shutil.copy2(split_json_path(data_path, "test"), test_out)

    return data_path
