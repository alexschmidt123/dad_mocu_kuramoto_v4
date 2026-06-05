"""Train/eval context resolved from shared data tables (not experiment YAML alone)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import SBOEDConfig, config_from_data_meta, repo_root
from src.data import DataRunMeta, load_data_run_meta, load_split_systems, resolve_data_dir


@dataclass
class ExperimentRun:
    """Everything needed for train/eval, anchored on ``run_config.yaml`` + ``train.json`` meta."""

    exp_dir: Path
    data_path: Path
    meta: DataRunMeta
    cfg: SBOEDConfig
    train_systems: list[dict[str, Any]]
    test_systems: list[dict[str, Any]]

    @property
    def step_number(self) -> int:
        return self.meta.step_number

    @property
    def policy_meta(self) -> dict[str, Any]:
        return self.meta.policy_meta()


def load_experiment_run(
    exp_dir: Path,
    project_root: Path | None = None,
) -> ExperimentRun:
    """
    Resolve ``run_config.yaml`` ``data_dir`` → load table metadata → build cfg synced to data.

    Horizon ``T``, ``sigma_y``, action count, and seeds come from the data bundle;
    YAML (via ``manifest.yaml``) supplies swing/prior/training hyperparameters only.
    """
    root = project_root or repo_root()
    exp_dir = Path(exp_dir).resolve()
    data_path = resolve_data_dir(exp_dir, root)
    meta = load_data_run_meta(data_path)
    cfg = config_from_data_meta(meta)
    meta.validate_against_config(cfg)
    train_systems, test_systems = load_split_systems(data_path)
    return ExperimentRun(
        exp_dir=exp_dir,
        data_path=data_path,
        meta=meta,
        cfg=cfg,
        train_systems=train_systems,
        test_systems=test_systems,
    )
