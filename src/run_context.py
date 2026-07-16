"""Train/eval context resolved from shared data tables (not experiment YAML alone)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import SBOEDConfig, apply_data_meta_to_cfg, load_config, repo_root, with_step_number
from src.data import DataRunMeta, load_data_run_meta, load_split_systems, resolve_data_dir
from src.experiment_layout import load_run_config_doc, resolve_experiment_config_path


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
        """Experiment BOED horizon T (from run_config), not the one-step data bank."""
        return self.cfg.step_number

    @property
    def policy_meta(self) -> dict[str, Any]:
        base = self.meta.policy_meta()
        horizon = int(self.cfg.step_number)
        return {
            **base,
            "step_number": horizon,
            "data_bank_step_number": int(base["step_number"]),
            "experiment_dir": str(self.exp_dir.resolve()),
            "experiment_step_number": horizon,
            "training_horizon": horizon,
        }


def load_experiment_run(
    exp_dir: Path,
    project_root: Path | None = None,
) -> ExperimentRun:
    """
    Resolve ``run_config.yaml`` → experiment horizon T; ``data_dir`` → shared one-step tables.

    Physics/catalog fields come from the data bundle; BOED horizon T comes from the experiment folder.
    """
    root = project_root or repo_root()
    exp_dir = Path(exp_dir).resolve()

    cfg_path = resolve_experiment_config_path(exp_dir)
    cfg = load_config(cfg_path)
    doc = load_run_config_doc(exp_dir)
    if doc.get("step_number") is not None:
        cfg = with_step_number(cfg, int(doc["step_number"]))

    data_path = resolve_data_dir(exp_dir, root)
    meta = load_data_run_meta(data_path)
    cfg = apply_data_meta_to_cfg(cfg, meta)
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
