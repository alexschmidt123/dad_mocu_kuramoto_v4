"""Standard layout helpers for the objective RL-sBOED study."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.control.objective_rl_sboed import OUT, ROOT
from src.experiment_layout import (
    RunMetadata,
    ensure_standard_layout,
    eval_method_dir,
    git_commit_hash,
    link_or_copy_checkpoint,
    train_seed_dir,
    utc_now_stamp,
    write_run_metadata,
    write_study_run_config,
)

SCIENTIFIC_METHODS = ["DAD", "RL-sBOED", "Myopic", "Fixed", "Random"]


def system_exp_dir(system: str, horizon: int = 3) -> Path:
    return OUT / f"{system}_T{horizon}"


def method_key(method: str, init_mode: str) -> str:
    """Filesystem key; not a manuscript method name."""
    prefix = "dad" if method == "DAD" else "rl_sboed"
    return f"{prefix}_{init_mode}_init"


def prepare_system_experiment(
    system: str,
    *,
    horizon: int = 3,
    terminal_rule_hash: str | None = None,
    data_dir: Path | None = None,
    source_config: Path | None = None,
    entry_point: str = "run.sh",
) -> Path:
    exp_dir = system_exp_dir(system, horizon)
    ensure_standard_layout(exp_dir)
    bank = data_dir or (ROOT / "data" / system.replace("ieee", "ieee"))
    # data dirs are ieee5 / ieee9
    bank = ROOT / "data" / system
    cfg = source_config or (ROOT / "config" / f"{system}_config.yaml")
    write_study_run_config(
        exp_dir,
        study_name="objective_rl_sboed",
        system=system,
        horizon=horizon,
        methods=SCIENTIFIC_METHODS,
        data_dir=bank if bank.exists() else None,
        source_config=cfg if cfg.exists() else None,
        terminal_rule_hash=terminal_rule_hash,
        extra={
            "parent_study": str(OUT),
            "reference_layout": "experiments/ieee5_T3 + pre-2026-07-13 stamped runs",
        },
    )
    write_run_metadata(
        exp_dir,
        RunMetadata(
            experiment_name=f"objective_rl_sboed/{system}_T{horizon}",
            entry_point=entry_point,
            timestamp_utc=utc_now_stamp(),
            system=system,
            horizon=horizon,
            method="multi",
            git_commit=git_commit_hash(ROOT),
            terminal_rule_hash=terminal_rule_hash,
            data_dir=str(bank) if bank.exists() else None,
            config_profile=str(cfg.name) if cfg.exists() else None,
        ),
    )
    # Study-level layout
    ensure_standard_layout(OUT)
    write_study_run_config(
        OUT,
        study_name="objective_rl_sboed",
        system="multi",
        horizon=horizon,
        methods=SCIENTIFIC_METHODS,
        data_dir=None,
        source_config=None,
        terminal_rule_hash=None,
        extra={"systems": ["ieee5", "ieee9"], "primary_horizon": horizon},
    )
    return exp_dir


def training_output_dir(
    system: str,
    method: str,
    init_mode: str,
    seed: int,
    *,
    horizon: int = 3,
) -> Path:
    exp_dir = prepare_system_experiment(system, horizon=horizon, entry_point="run.sh")
    return train_seed_dir(exp_dir, method_key(method, init_mode), seed)


def publish_checkpoint_to_model(
    exp_dir: Path,
    method: str,
    init_mode: str,
    seed: int,
    checkpoint_path: Path,
) -> Path:
    key = method_key(method, init_mode)
    dest = exp_dir / "model" / key / f"seed_{seed}_best.pt"
    link_or_copy_checkpoint(checkpoint_path, dest)
    return dest


def migrate_legacy_objective_rl_sboed_tree() -> dict[str, Any]:
    """Move ad-hoc seed folders into ``train/`` and logs into ``logs/``.

    Does not delete historical authoritative experiments outside this study.
    Incomplete seed dirs without ``result.json`` are moved to ``logs/scratch/``.
    """
    report: dict[str, Any] = {"moved": [], "archived_incomplete": [], "skipped": []}
    ensure_standard_layout(OUT)

    # Top-level logs
    for name in ("sensitivity_run.log", "ieee5_train.log", "ieee9_train.log"):
        src = OUT / name
        if src.is_file():
            dest = OUT / "logs" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            report["moved"].append({"from": str(src), "to": str(dest)})

    # sensitivity_audit -> diagnostics/sensitivity_audit
    sens = OUT / "sensitivity_audit"
    if sens.is_dir() and not (OUT / "diagnostics" / "sensitivity_audit").exists():
        dest = OUT / "diagnostics" / "sensitivity_audit"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(sens), str(dest))
        report["moved"].append({"from": str(sens), "to": str(dest)})

    for system_dir in sorted(OUT.glob("ieee*_T*")):
        if not system_dir.is_dir():
            continue
        ensure_standard_layout(system_dir)
        # Legacy: ieee5_T3/dad_random_init/seed_*
        for child in list(system_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name in {
                "train",
                "eval",
                "model",
                "logs",
                "diagnostics",
                "summary",
                "plots",
            }:
                continue
            if child.name.endswith("_init"):
                for seed_dir in list(child.glob("seed_*")):
                    dest = system_dir / "train" / child.name / seed_dir.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        report["skipped"].append(str(seed_dir))
                        continue
                    has_result = (seed_dir / "result.json").exists()
                    if not has_result:
                        scratch = system_dir / "logs" / "scratch" / child.name / seed_dir.name
                        scratch.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(seed_dir), str(scratch))
                        report["archived_incomplete"].append(
                            {"from": str(seed_dir), "to": str(scratch)}
                        )
                        continue
                    shutil.move(str(seed_dir), str(dest))
                    report["moved"].append({"from": str(seed_dir), "to": str(dest)})
                    ckpt = dest / "best_checkpoint.pt"
                    if ckpt.exists():
                        parts = child.name.split("_")
                        # dad_random_init / rl_sboed_fixed_init
                        if child.name.startswith("dad_"):
                            method = "DAD"
                            init_mode = child.name[len("dad_") : -len("_init")]
                        else:
                            method = "RL-sBOED"
                            init_mode = child.name[len("rl_sboed_") : -len("_init")]
                        publish_checkpoint_to_model(
                            system_dir, method, init_mode, int(seed_dir.name.split("_")[1]), ckpt
                        )
                # remove empty parent
                if child.exists() and not any(child.iterdir()):
                    child.rmdir()
            elif child.name.startswith("baseline_") or child.suffix == ".csv":
                # leave comparison csvs for now; move baselines into eval later
                pass

        # Move loose comparison CSVs into eval/
        for csv_name in (
            "comparison.csv",
            "paired_bootstrap.csv",
            "baseline_summary.csv",
            "adaptivity.csv",
            "action_regret.csv",
        ):
            src = system_dir / csv_name
            if src.is_file():
                dest = system_dir / "eval" / csv_name
                shutil.move(str(src), str(dest))
                report["moved"].append({"from": str(src), "to": str(dest)})
        for name in (
            "baseline_fixed_rollouts.csv",
            "baseline_random_rollouts.csv",
            "baseline_myopic_rollouts.csv",
        ):
            src = system_dir / name
            if src.is_file():
                method = name.replace("baseline_", "").replace("_rollouts.csv", "")
                dest_dir = eval_method_dir(system_dir, method)
                dest = dest_dir / "rollouts.csv"
                shutil.move(str(src), str(dest))
                report["moved"].append({"from": str(src), "to": str(dest)})

    (OUT / "logs" / "layout_migration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
