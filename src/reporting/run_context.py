"""Experiment folder layout + resolved train/eval run context.

Merged from former ``experiment_layout`` and ``run_context`` modules.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.config import (
    SBOEDConfig,
    apply_data_meta_to_cfg,
    load_config,
    repo_root,
    with_step_number,
)
from src.banks.tables import (
    DataRunMeta,
    load_data_run_meta,
    load_split_systems,
    resolve_data_dir,
)

RUN_CONFIG_FILENAME = "run_config.json"
LEGACY_RUN_CONFIG_YAML = "run_config.yaml"
LEGACY_CONFIG_COPY = "config.yaml"
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

# Result folders:
# MMDDYYYY_HHMMSS_config_Uctrl|EIG_Tnum_NobsN_sigma0p005
RESULT_DIR_RE = re.compile(
    r"^(?P<stamp>\d{8}_\d{6})_(?P<config>.+)_"
    r"(?P<label>Uctrl|EIG)_T(?P<T>\d+)_Nobs(?P<nobs>\d+)_"
    r"sigma(?P<sigma>\d+(?:p\d+)?)$"
)
EXPERIMENT_TYPES = ("objective_based", "eig_based")
EXPERIMENT_FOLDER_LABELS = {
    "objective_based": "Uctrl",
    "eig_based": "EIG",
}


def _folder_sigma(value: float) -> str:
    text = f"{float(value):.12f}".rstrip("0").rstrip(".")
    if not text or text == "0":
        raise ValueError(f"noise_sigma is too small for folder naming: {value}")
    return text.replace(".", "p")


def _cfg_observation_values(cfg: SBOEDConfig) -> tuple[int, float]:
    obs = dict(cfg.raw.get("observation") or {})
    return int(obs.get("N_obs", 0)), float(obs.get("noise_sigma", 0.005))


def make_experiment_dir_name(
    config_name: str,
    experiment_type: str,
    step_number: int,
    *,
    stamp: str | None = None,
    n_obs: int = 0,
    noise_sigma: float = 0.005,
) -> str:
    """Return the compact, observation-explicit experiment folder name."""
    et = str(experiment_type).strip().lower().replace("-", "_")
    if et not in EXPERIMENT_TYPES:
        raise ValueError(
            f"Invalid experiment_type {experiment_type!r} "
            f"(allowed: {', '.join(EXPERIMENT_TYPES)})"
        )
    name = str(config_name).strip()
    if not name or "/" in name or "\\" in name or " " in name:
        raise ValueError(f"Invalid config name for result folder: {config_name!r}")
    if stamp is None:
        stamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    if int(n_obs) < 0:
        raise ValueError(f"N_obs must be non-negative, got {n_obs}")
    label = EXPERIMENT_FOLDER_LABELS[et]
    return (
        f"{stamp}_{name}_{label}_T{int(step_number)}_"
        f"Nobs{int(n_obs)}_sigma{_folder_sigma(noise_sigma)}"
    )


def parse_result_dir_name(name: str) -> dict[str, Any] | None:
    """Parse a result folder basename; return None if it does not match the rule."""
    m = RESULT_DIR_RE.match(str(name))
    if not m:
        return None
    return {
        "stamp": m.group("stamp"),
        "config": m.group("config"),
        "experiment_type": (
            "objective_based" if m.group("label") == "Uctrl" else "eig_based"
        ),
        "step_number": int(m.group("T")),
        "N_obs": int(m.group("nobs")),
        "noise_sigma": float(m.group("sigma").replace("p", ".")),
    }


def validate_result_dir_name(name: str) -> dict[str, Any]:
    # Reject scratch / invalid folders such as experiments/_sir_smoke_ctx
    # or experiments/plan2_trap_objective_sweep_logs.
    basename = Path(str(name)).name
    if (
        basename.startswith("_")
        or "/_" in f"/{basename}"
        or "sweep_logs" in basename
        or basename.endswith("_logs")
    ):
        raise ValueError(
            f"Invalid result folder name {basename!r}: under experiments/ "
            f"only stamped result dirs are allowed "
            f"(MMDDYYYY_HHMMSS_config_Uctrl|EIG_T#_Nobs#_sigmaX). "
            f"Put smoke/scratch under /tmp. Run logs belong in "
            f"<experiments/...>/logs/ (never a top-level logs/ folder)."
        )
    parsed = parse_result_dir_name(basename)
    if parsed is None:
        raise ValueError(
            f"Result folder name must be "
            f"'date_time_config_Uctrl|EIG_Tnum_NobsN_sigmaX', got {basename!r}"
        )
    return parsed


def assert_experiments_result_dir(
    path: Path | str,
    *,
    project_root: Path | None = None,
) -> Path:
    """If ``path`` is under ``experiments/``, require a valid result basename."""
    root = (project_root or repo_root()).resolve()
    path = Path(path).resolve()
    experiments = (root / "experiments").resolve()
    try:
        path.relative_to(experiments)
    except ValueError:
        return path
    validate_result_dir_name(path.name)
    return path


def allocate_result_dir(
    cfg: SBOEDConfig,
    experiment_type: str,
    *,
    project_root: Path | None = None,
    stamp: str | None = None,
) -> Path:
    """Create a new stamped result folder under ``experiments/``."""
    root = project_root or repo_root()
    n_obs, noise_sigma = _cfg_observation_values(cfg)
    name = make_experiment_dir_name(
        cfg.run_slug,
        experiment_type,
        int(cfg.step_number),
        stamp=stamp,
        n_obs=n_obs,
        noise_sigma=noise_sigma,
    )
    path = (root / "experiments" / name).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_latest_result_dir(
    cfg: SBOEDConfig,
    experiment_type: str,
    *,
    project_root: Path | None = None,
) -> Path | None:
    """Latest folder matching ``*_configname_experimentType_Tnum``."""
    root = project_root or repo_root()
    experiments = root / "experiments"
    if not experiments.is_dir():
        return None
    et = str(experiment_type).strip().lower().replace("-", "_")
    want_config = cfg.run_slug
    want_T = int(cfg.step_number)
    want_n_obs, want_sigma = _cfg_observation_values(cfg)
    matches: list[Path] = []
    for path in experiments.iterdir():
        if not path.is_dir():
            continue
        # Never treat scratch dirs (e.g. _sir_smoke_ctx, _plots) as results.
        if path.name.startswith("_"):
            continue
        parsed = parse_result_dir_name(path.name)
        if parsed is None:
            continue
        if (
            parsed["config"] == want_config
            and parsed["experiment_type"] == et
            and parsed["step_number"] == want_T
            and parsed["N_obs"] == want_n_obs
            and abs(parsed["noise_sigma"] - want_sigma) <= 1e-15
        ):
            matches.append(path)
    if not matches:
        return None
    matches.sort(key=lambda p: (p.name, p.stat().st_mtime))
    return matches[-1].resolve()


def resolve_result_dir(
    cfg: SBOEDConfig,
    experiment_type: str,
    *,
    exp_dir: str | Path | None = None,
    project_root: Path | None = None,
    create_new: bool = False,
) -> Path:
    """
    Resolve the result folder for this config / experiment type / T.

    - ``exp_dir`` given → validate name and use it
    - ``create_new`` → allocate a fresh stamped folder
    - else → latest matching folder (error if none)
    """
    root = project_root or repo_root()
    if exp_dir is not None:
        path = Path(exp_dir)
        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()
        path = assert_experiments_result_dir(path, project_root=root)
        path.mkdir(parents=True, exist_ok=True)
        return path
    if create_new:
        return allocate_result_dir(cfg, experiment_type, project_root=root)
    found = find_latest_result_dir(cfg, experiment_type, project_root=root)
    if found is None:
        raise FileNotFoundError(
            "No result folder matching "
            f"'*_{cfg.run_slug}_{EXPERIMENT_FOLDER_LABELS[experiment_type]}_"
            f"T{int(cfg.step_number)}_Nobs{_cfg_observation_values(cfg)[0]}_*' "
            f"under {root / 'experiments'}. "
            "Run data_generation first, or pass --exp-dir."
        )
    return found


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
    """Load ``run_config.json`` (preferred) or legacy ``run_config.yaml``."""
    exp_dir = Path(exp_dir)
    json_path = exp_dir / RUN_CONFIG_FILENAME
    if json_path.is_file():
        with json_path.open(encoding="utf-8") as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else {}
    yaml_path = exp_dir / LEGACY_RUN_CONFIG_YAML
    if yaml_path.is_file():
        with yaml_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def read_linked_data_dir(exp_dir: Path) -> Path:
    """
    Shared table path from ``run_config.json`` field ``data_dir``.

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


def ensure_result_layout(exp_dir: Path) -> tuple[Path, Path]:
    """Ensure ``model/`` and ``eval/`` exist under the result folder."""
    exp_dir = assert_experiments_result_dir(Path(exp_dir).resolve())
    mdir = exp_dir / MODEL_SUBDIR
    edir = exp_dir / EVAL_SUBDIR
    mdir.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)
    return mdir, edir


def write_run_config(
    exp_dir: Path,
    cfg: SBOEDConfig,
    data_path: Path,
    *,
    experiment_type: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write this run's record as ``run_config.json``.

    Identity fields (type, T, N_obs, noise_sigma, seed, methods) are explicit.
    Training knobs are filtered to the selected methods (no MoE keys on a DAD-only
    run). YAML is the template; this file is what the experiment actually used.
    """
    exp_dir = Path(exp_dir)
    ensure_result_layout(exp_dir)
    path = run_config_path(exp_dir)
    extra_doc = dict(extra or {})
    extra_methods = extra_doc.pop("methods", None)
    extra_seed = extra_doc.pop("seed", None)
    prev = load_run_config_doc(exp_dir)
    extra_keys = set((extra or {}).keys())

    exp_type = (
        str(experiment_type).strip().lower().replace("-", "_")
        if experiment_type is not None
        else str(
            extra_doc.get("experiment_type")
            or prev.get("experiment_type")
            or (cfg.raw.get("experiment") or {}).get("experiment_type")
            or "objective_based"
        )
        .strip()
        .lower()
        .replace("-", "_")
    )
    stamped = _resolve_run_methods(
        extra_methods=extra_methods,
        previous=prev,
        yaml_methods=list(cfg.methods),
    )
    obs = dict(cfg.raw.get("observation") or {})
    n_obs = int(
        extra_doc.get("N_obs", obs.get("N_obs", prev.get("N_obs", 0)))
    )
    noise_sigma = float(
        extra_doc.get(
            "noise_sigma",
            obs.get("noise_sigma", prev.get("noise_sigma", 0.005)),
        )
    )
    seed = _resolve_run_seed(extra_seed, prev)
    horizon = int(cfg.step_number)

    exp = dict(cfg.raw.get("experiment") or {})
    exp["step_number"] = horizon
    exp["experiment_type"] = exp_type
    exp["methods"] = stamped
    exp.pop("output_dir", None)

    body = _filter_run_body(
        dict(cfg.raw),
        methods=stamped,
        experiment_type=exp_type,
    )
    body["experiment"] = exp
    obs_block = dict(body.get("observation") or {})
    obs_block["N_obs"] = n_obs
    obs_block["noise_sigma"] = noise_sigma
    body["observation"] = obs_block

    labels = dict(cfg.run_labels())
    labels.pop("step_number", None)
    doc: dict[str, Any] = {
        "source_config": str(cfg.config_path.resolve()),
        "config_name": cfg.run_slug,
        "data_dir": str(Path(data_path).resolve()),
        **labels,
        "experiment_type": exp_type,
        "T": horizon,
        "step_number": horizon,
        "N_obs": n_obs,
        "noise_sigma": noise_sigma,
        "seed": seed,
        "methods": stamped,
        **body,
    }
    extra_doc.pop("experiment_type", None)
    extra_doc.pop("N_obs", None)
    extra_doc.pop("noise_sigma", None)
    extra_doc.pop("exp_dir", None)
    if extra_doc:
        doc.update(extra_doc)
    if "training_results" in prev and "training_results" not in extra_keys:
        doc["training_results"] = prev["training_results"]
    if "data_generation" in prev and "data_generation" not in extra_keys:
        doc["data_generation"] = prev["data_generation"]
    # Re-apply identity after extra merge so YAML leftovers cannot overwrite them.
    doc["experiment_type"] = exp_type
    doc["T"] = horizon
    doc["step_number"] = horizon
    doc["N_obs"] = n_obs
    doc["noise_sigma"] = noise_sigma
    doc["seed"] = seed
    doc["methods"] = stamped
    exp_block = dict(doc.get("experiment") or {})
    exp_block["methods"] = stamped
    exp_block["experiment_type"] = exp_type
    exp_block["step_number"] = horizon
    doc["experiment"] = exp_block
    training = _filter_training_block(doc.get("training"), stamped)
    if training:
        doc["training"] = training
    else:
        doc.pop("training", None)

    for stale in (LEGACY_RUN_CONFIG_YAML, LEGACY_CONFIG_COPY):
        stale_path = exp_dir / stale
        if stale_path.is_file():
            stale_path.unlink()
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")

    legacy_ptr = exp_dir / LEGACY_DATA_DIR_POINTER
    if legacy_ptr.is_file():
        legacy_ptr.unlink()

    return path


def _resolve_run_seed(extra_seed: Any, previous: dict[str, Any]) -> int | None:
    if extra_seed is not None and str(extra_seed).strip() != "":
        return int(extra_seed)
    if previous.get("seed") is not None and str(previous.get("seed")).strip() != "":
        return int(previous["seed"])
    return None


def _filter_run_body(
    raw: dict[str, Any],
    *,
    methods: list[str],
    experiment_type: str,
) -> dict[str, Any]:
    """Keep physics/bank sections; drop objective-only blocks on EIG runs."""
    body = dict(raw)
    if experiment_type == "eig_based":
        for key in ("control", "control_safety_calibration", "oracle"):
            body.pop(key, None)
    selected = set(methods)
    trainable = {"dad", "rl_sboed", "moe_sboed", "matched_dense"}
    if not selected & trainable:
        body.pop("training", None)
    return body


def _filter_training_block(
    training: Any, methods: list[str]
) -> dict[str, Any] | None:
    """Drop method-specific knobs that this run did not use."""
    if not isinstance(training, dict) or not training:
        return None
    selected = set(methods)
    trainable = {"dad", "rl_sboed", "moe_sboed", "matched_dense"}
    if not selected & trainable:
        return None
    keep_moe = bool(selected & {"moe_sboed", "matched_dense"})
    keep_rl = "rl_sboed" in selected
    keep_dad = "dad" in selected
    keep_step = "step_dad" in selected
    out: dict[str, Any] = {}
    for key, value in training.items():
        lk = str(key).lower()
        if "moe" in lk or "matched_dense" in lk or "matcheddense" in lk:
            if not keep_moe:
                continue
        elif "eig_rl" in lk or lk.startswith("rl_"):
            if not keep_rl:
                continue
        elif "eig_dad" in lk or "dad_optimizer" in lk or lk.startswith("dad_"):
            if not keep_dad:
                continue
        elif "step_dad" in lk:
            if not keep_step:
                continue
        out[key] = value
    return out or None


def _resolve_run_methods(
    *,
    extra_methods: Any,
    previous: dict[str, Any],
    yaml_methods: list[str],
) -> list[str]:
    """Prefer this-step selection, else already-stamped run methods, else YAML."""
    if extra_methods is not None:
        stamped = _canonical_run_methods(extra_methods)
        if stamped:
            return stamped
    prev_block = dict(previous.get("experiment") or {})
    prev_methods = prev_block.get("methods", previous.get("methods"))
    stamped = _canonical_run_methods(prev_methods)
    if stamped:
        return stamped
    stamped = _canonical_run_methods(yaml_methods)
    return stamped or list(yaml_methods)


def _canonical_run_methods(raw: Any) -> list[str]:
    """Canonical keys (dad, random, …); accepts vector-EIG aliases."""
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        items = [str(item).strip() for item in list(raw) if str(item).strip()]
    if not items:
        return []
    from src.objectives.mocu.context import normalize_method_key

    aliases = {
        "dad_eig": "dad",
        "rl_sboed_eig": "rl_sboed",
        "myopic_delta_h": "myopic",
        "fixed_open_loop": "fixed",
    }
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = aliases.get(item, item)
        try:
            canon = normalize_method_key(key)
        except ValueError:
            continue
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def resolve_experiment_config_path(exp_dir: Path) -> Path:
    """``run_config.json`` for this run, else legacy YAML names."""
    exp_dir = exp_dir.resolve()
    run_cfg = run_config_path(exp_dir)
    if run_cfg.is_file():
        return run_cfg
    legacy_yaml = exp_dir / LEGACY_RUN_CONFIG_YAML
    if legacy_yaml.is_file():
        return legacy_yaml
    legacy = exp_dir / LEGACY_CONFIG_COPY
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
        raise FileNotFoundError(f"No {RUN_CONFIG_FILENAME} in {exp_dir}")
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
    """Create the standard subdirectory tree under ``exp_dir``.

    Required result layout: ``model/``, ``eval/``, and ``run_config.json``.
    """
    paths = StandardExperimentPaths(root=exp_dir.resolve())
    # Primary artifacts live under model/ and eval/.
    for path in (paths.model, paths.eval):
        path.mkdir(parents=True, exist_ok=True)
    # Optional extras used by some studies / diagnostics.
    for path in (
        paths.train,
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


# ---------------------------------------------------------------------------
# ExperimentRun: train/eval context from linked data tables
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRun:
    """Everything needed for train/eval, anchored on ``run_config.yaml`` + data meta."""

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
    Resolve ``run_config.yaml`` → experiment horizon T; ``data_dir`` → shared tables.

    Physics/catalog fields come from the data bundle; BOED horizon T from the experiment folder.
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
