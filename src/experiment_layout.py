"""Paths and writers for the standard experiment folder layout.

Covers stamped pre-2026-07-13 runs (``run_config.yaml``, ``model/``, ``eval/``)
and control-objective / study runs (``train/``, ``diagnostics/``, ``logs/``,
``summary/``, ``run_metadata.json``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.config import SBOEDConfig

RUN_CONFIG_FILENAME = "run_config.yaml"
LEGACY_DATA_DIR_POINTER = "data_dir.txt"
MODEL_SUBDIR = "model"
EVAL_SUBDIR = "eval"
EVAL_SUMMARY_FILENAME = "summary.csv"
TRAIN_SUBDIR = "train"
LOGS_SUBDIR = "logs"
DIAGNOSTICS_SUBDIR = "diagnostics"
SUMMARY_SUBDIR = "summary"
PLOTS_SUBDIR = "plots"
SCRATCH_SUBDIR = "scratch"
RUN_METADATA_FILENAME = "run_metadata.json"


def make_experiment_dir_name(
    run_name: str,
    step_number: int,
    *,
    stamp: str | None = None,
) -> str:
    """e.g. ``06282026_005316_ieee5_T1`` (timestamp first for chronological sorting)."""
    if stamp is None:
        stamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    return f"{stamp}_{run_name}_T{int(step_number)}"


def model_dir(exp_dir: Path) -> Path:
    return exp_dir / MODEL_SUBDIR


def reset_model_dir(exp_dir: Path) -> Path:
    """Remove any stale policies so a new experiment run always trains fresh."""
    mdir = model_dir(exp_dir)
    if mdir.exists():
        shutil.rmtree(mdir)
    mdir.mkdir(parents=True, exist_ok=True)
    return mdir


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
        **cfg.run_labels(),
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


@dataclass
class RunMetadata:
    """Required audit fields for an authoritative complete run."""

    experiment_name: str
    entry_point: str  # "run.sh" | "sweep_run.sh"
    timestamp_utc: str
    system: str
    horizon: int
    method: str
    seed: int | None = None
    git_commit: str | None = None
    terminal_rule_hash: str | None = None
    data_dir: str | None = None
    config_profile: str | None = None
    initialization: str | None = None
    scientific_methods: tuple[str, ...] = (
        "DAD",
        "RL-sBOED",
        "Myopic",
        "Fixed",
        "Random",
    )
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scientific_methods"] = list(self.scientific_methods)
        return payload


def git_commit_hash(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or None
    except OSError:
        return None
    return None


def utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class StandardExperimentPaths:
    """Resolved paths for one experiment directory."""

    root: Path

    @property
    def run_config(self) -> Path:
        return self.root / RUN_CONFIG_FILENAME

    @property
    def run_metadata(self) -> Path:
        return self.root / RUN_METADATA_FILENAME

    @property
    def model(self) -> Path:
        return self.root / MODEL_SUBDIR

    @property
    def train(self) -> Path:
        return self.root / TRAIN_SUBDIR

    @property
    def eval(self) -> Path:
        return self.root / EVAL_SUBDIR

    @property
    def logs(self) -> Path:
        return self.root / LOGS_SUBDIR

    @property
    def diagnostics(self) -> Path:
        return self.root / DIAGNOSTICS_SUBDIR

    @property
    def summary(self) -> Path:
        return self.root / SUMMARY_SUBDIR

    @property
    def plots(self) -> Path:
        return self.eval / PLOTS_SUBDIR

    @property
    def scratch(self) -> Path:
        return self.logs / SCRATCH_SUBDIR


def ensure_standard_layout(exp_dir: Path) -> StandardExperimentPaths:
    """Create the standard subdirectory tree under ``exp_dir``."""
    paths = StandardExperimentPaths(root=exp_dir.resolve())
    for path in (
        paths.model,
        paths.train,
        paths.eval,
        paths.logs,
        paths.diagnostics,
        paths.summary,
        paths.plots,
        paths.scratch,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_run_metadata(exp_dir: Path, metadata: RunMetadata) -> Path:
    paths = ensure_standard_layout(exp_dir)
    path = paths.run_metadata
    path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    return path


def write_study_run_config(
    exp_dir: Path,
    *,
    study_name: str,
    system: str,
    horizon: int,
    methods: list[str],
    data_dir: str | Path | None,
    source_config: str | Path | None,
    terminal_rule_hash: str | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a study-level ``run_config.yaml`` compatible with stamped runs."""
    ensure_standard_layout(exp_dir)
    doc: dict[str, Any] = {
        "study_name": study_name,
        "step_number": int(horizon),
        "system": system,
        "topology": system,
        "run_name": system,
        "methods": list(methods),
        "terminal_rule_hash": terminal_rule_hash,
        "data_dir": str(data_dir) if data_dir is not None else None,
        "source_config": str(source_config) if source_config is not None else None,
        "output_layout": {
            "model": MODEL_SUBDIR,
            "train": TRAIN_SUBDIR,
            "eval": EVAL_SUBDIR,
            "logs": LOGS_SUBDIR,
            "diagnostics": DIAGNOSTICS_SUBDIR,
            "summary": SUMMARY_SUBDIR,
        },
        "entry_points": {
            "single_run": "./run.sh",
            "sweep": "./sweep_run.sh",
        },
    }
    if extra:
        doc.update(extra)
    path = run_config_path(exp_dir)
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(doc, handle, default_flow_style=False, sort_keys=False)
    return path


def train_seed_dir(exp_dir: Path, method_key: str, seed: int) -> Path:
    """``train/<method_key>/seed_<seed>/``."""
    path = ensure_standard_layout(exp_dir).train / method_key / f"seed_{int(seed)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def eval_method_dir(exp_dir: Path, method_key: str) -> Path:
    path = ensure_standard_layout(exp_dir).eval / method_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def link_or_copy_checkpoint(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        dest.write_bytes(src.read_bytes())
