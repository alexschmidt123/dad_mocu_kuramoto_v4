"""Load sBOED experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ALL_METHODS = ["dad", "rl_sboed", "myopic", "fixed", "random"]

# Human-readable labels for built-in MATPOWER-style feeders.
IEEE_SYSTEM_LABELS: dict[str, str] = {
    "ieee5": "IEEE-5",
    "ieee9": "IEEE-9",
    "ieee14": "IEEE-14",
    "sir_ode": "SIR-ODE",
    "sir": "SIR-ODE",
}

# Horizon when CLI does not pass ``-T`` (see ``run.sh`` / ``src.cli``).
DEFAULT_STEP_NUMBER = 5


@dataclass
class SBOEDConfig:
    raw: dict[str, Any]
    config_path: Path

    @property
    def name(self) -> str:
        return self.config_path.stem

    @property
    def N(self) -> int:
        return int(self.raw.get("N", 14))

    @property
    def step_number(self) -> int:
        if "step_number" in self.raw:
            return int(self.raw["step_number"])
        exp = self.raw.get("experiment") or {}
        if "step_number" in exp:
            return int(exp["step_number"])
        return int(exp.get("horizon", 3))  # legacy alias


    @property
    def methods(self) -> list[str]:
        m = self.raw.get("experiment", {}).get("methods", ALL_METHODS)
        return list(m)

    @property
    def probe_amplitudes(self) -> list[float]:
        return list(self.raw.get("swing_equation", {}).get("probe_amplitudes", [0.05, 0.1, 0.2]))

    @property
    def probe_buses(self) -> list[int] | None:
        """Optional probe-bus subset (0-based). ``None`` → all buses ``0..N-1``."""
        raw = self.raw.get("swing_equation", {}).get("probe_buses")
        if raw is None:
            return None
        return [int(b) for b in raw]

    @property
    def probe_duration(self) -> float:
        return float(self.raw.get("swing_equation", {}).get("probe_duration", 0.2))

    @property
    def probe_durations(self) -> list[float] | None:
        """Optional multi-duration catalog. ``None`` → single ``probe_duration``.

        Prefer an explicit ``probe_durations`` list. Otherwise, if
        ``duration_min`` / ``duration_max`` / ``duration_step`` are set, build
        ``np.arange(min, max+ε, step)`` (inclusive of max when it lands on-grid).
        """
        sw = self.raw.get("swing_equation") or {}
        raw = sw.get("probe_durations")
        if raw is not None:
            out = [float(d) for d in raw]
            if not out:
                raise ValueError(
                    "swing_equation.probe_durations must be non-empty when set"
                )
            return out
        if "duration_min" in sw and "duration_max" in sw and "duration_step" in sw:
            d0 = float(sw["duration_min"])
            d1 = float(sw["duration_max"])
            step = float(sw["duration_step"])
            if step <= 0:
                raise ValueError(f"duration_step must be > 0, got {step}")
            if d1 < d0:
                raise ValueError(f"duration_max < duration_min ({d1} < {d0})")
            # Inclusive upper bound when representable on the grid.
            n = int(round((d1 - d0) / step)) + 1
            out = [float(d0 + i * step) for i in range(max(n, 0))]
            out = [d for d in out if d <= d1 + 0.5 * step]
            if not out:
                raise ValueError("duration grid is empty")
            return out
        return None

    @property
    def reset_after_probe(self) -> bool:
        """Plan-2 default True; continuous-duration mode sets False."""
        sw = self.raw.get("swing_equation") or {}
        if "reset_after_probe" in sw:
            return bool(sw["reset_after_probe"])
        exp = self.raw.get("experiment") or {}
        mode = str(exp.get("mode") or exp.get("problem_setting") or "").lower()
        if mode in {"continuous_duration", "power_grid_continuous", "pg_continuous"}:
            return False
        return True

    @property
    def continuous_duration_mode(self) -> bool:
        """Duration-only chronological designs (fixed amp/bus)."""
        if not self.reset_after_probe:
            return True
        exp = self.raw.get("experiment") or {}
        mode = str(exp.get("mode") or exp.get("problem_setting") or "").lower()
        return mode in {"continuous_duration", "power_grid_continuous", "pg_continuous"}

    @property
    def sigma_y(self) -> float:
        obs = dict(self.raw.get("observation") or {})
        if "noise_sigma" in obs:
            return float(obs["noise_sigma"])
        if self.raw.get("sir_sde"):
            raise ValueError(
                "Config key 'sir_sde' is removed. Use 'sir_ode' "
                "(deterministic ODE + explicit Gaussian likelihood)."
            )
        sir = dict(self.raw.get("sir_ode") or self.raw.get("sir") or {})
        if "noise_sigma" in sir:
            return float(sir["noise_sigma"])
        if "likelihood_sigma" in sir:
            return float(sir["likelihood_sigma"])
        return float(self.raw.get("swing_equation", {}).get("sigma", 0.05))

    @property
    def T_obs_sec(self) -> float:
        return float(self.raw.get("swing_equation", {}).get("T_obs_sec", 10.0))

    @property
    def fs_hz(self) -> float:
        return float(self.raw.get("swing_equation", {}).get("fs_hz", 12.0))

    @property
    def swing(self) -> dict[str, Any]:
        sw = dict(self.raw.get("swing_equation") or {})
        sw["N"] = self.N
        return sw

    @property
    def prior(self) -> dict[str, Any]:
        return dict(self.raw.get("prior") or {})

    @property
    def spce(self) -> dict[str, Any]:
        return dict(self.raw.get("spce") or {})

    @property
    def data(self) -> dict[str, Any]:
        return dict(self.raw.get("data_generation") or {})

    def theta_sample_size(self, split: str) -> int:
        """Number of independent θ=(M,K) draws for ``train`` or ``test`` split."""
        if split == "train":
            return int(
                self.data.get(
                    "theta_sample_size_train",
                    self.data.get("n_systems_train", 10),
                )
            )
        return int(
            self.data.get(
                "theta_sample_size_test",
                self.data.get("n_systems_test", 10),
            )
        )

    @property
    def training(self) -> dict[str, Any]:
        return dict(self.raw.get("training") or {})

    @property
    def topology(self) -> str:
        sys_name = (self.raw.get("system") or {}).get("name")
        if sys_name:
            return str(sys_name)
        return str(self.swing.get("topology", "ieee14"))

    @property
    def system_label(self) -> str:
        """Display name for the domain, e.g. ``IEEE-14`` or ``SIR-ODE``."""
        sys_name = str((self.raw.get("system") or {}).get("name") or self.topology)
        label = IEEE_SYSTEM_LABELS.get(sys_name) or IEEE_SYSTEM_LABELS.get(self.topology)
        if label is not None:
            return label
        if self.N <= 1:
            return sys_name
        return f"{self.topology} ({self.N}-bus)"

    @property
    def run_slug(self) -> str:
        """Short run label for data/experiment dirs (no ``_config`` suffix)."""
        name = self.name
        if name.endswith("_config"):
            return name[: -len("_config")]
        return name

    @property
    def config_preset(self) -> str:
        """Single canonical preset label for current configs."""
        return "default"

    def run_labels(self) -> dict[str, Any]:
        """Metadata stamped on data manifests, run_config, and eval summaries."""
        return {
            "system_label": self.system_label,
            "topology": self.topology,
            "run_name": self.run_slug,
            "preset": self.config_preset,
            "n_buses": int(self.N),
            "step_number": int(self.step_number),
        }


def load_config(path: str | Path) -> SBOEDConfig:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return SBOEDConfig(raw=raw, config_path=path)


def effective_step_number(cli_T: int | None, *, default: int = DEFAULT_STEP_NUMBER) -> int:
    """Probe horizon: ``-T`` / ``--T`` on CLI, else ``default`` (5)."""
    return int(cli_T) if cli_T is not None else int(default)


def with_step_number(cfg: SBOEDConfig, step_number: int) -> SBOEDConfig:
    """Return the same config object with ``experiment.step_number`` set."""
    exp = dict(cfg.raw.get("experiment") or {})
    exp["step_number"] = int(step_number)
    cfg.raw["experiment"] = exp
    return cfg


def sync_cfg_from_data_meta(cfg: SBOEDConfig, meta) -> SBOEDConfig:
    """Backward-compatible alias; preserves experiment horizon T."""
    return apply_data_meta_to_cfg(cfg, meta)


def apply_data_meta_to_cfg(cfg: SBOEDConfig, meta) -> SBOEDConfig:
    """Sync physics/catalog fields from data tables; keep experiment ``step_number`` unchanged."""
    cfg.raw["N"] = int(meta.n_buses)
    sw = dict(cfg.raw.get("swing_equation") or {})
    sw["sigma"] = float(meta.sigma_y)
    sw["probe_amplitudes"] = list(meta.probe_amplitudes)
    sw["probe_duration"] = float(meta.probe_duration)
    if getattr(meta, "probe_buses", None) is not None:
        sw["probe_buses"] = list(meta.probe_buses)
    cfg.raw["swing_equation"] = sw
    dg = dict(cfg.raw.get("data_generation") or {})
    dg["train_seed"] = int(meta.train_seed)
    dg["test_seed"] = int(meta.test_seed)
    cfg.raw["data_generation"] = dg
    return cfg


def config_from_data_meta(meta: "DataRunMeta") -> SBOEDConfig:
    """Load generation YAML from manifest, then override from table metadata."""
    if meta.config_path is None or not meta.config_path.is_file():
        raise FileNotFoundError(
            f"No config path in {meta.data_path / 'manifest.yaml'}; re-run generate-data"
        )
    cfg = load_config(meta.config_path)
    return sync_cfg_from_data_meta(cfg, meta)


def load_config_for_experiment(exp_dir: Path, project_root: Path | None = None) -> SBOEDConfig:
    """Backward-compatible: cfg synced to the experiment's linked data tables."""
    from src.reporting.run_context import load_experiment_run

    return load_experiment_run(exp_dir, project_root).cfg


def load_config_for_run(
    name_or_path: str | Path,
    project_root: Path | None = None,
    *,
    step_number: int | None = None,
) -> SBOEDConfig:
    """Load YAML and apply CLI horizon (default ``DEFAULT_STEP_NUMBER`` when omitted)."""
    root = project_root or repo_root()
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml") and path.is_file():
        cfg = load_config(path.resolve())
    else:
        cfg = load_config(resolve_config_path(str(name_or_path), root))
    return with_step_number(cfg, effective_step_number(step_number))


def repo_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[1]


# Canonical study-system config stems under configs/ (no experiment package import).
SYSTEM_CONFIGS = {
    "ieee5": "ieee5",
    "ieee9": "ieee9",
    "ieee14": "ieee14",
}
DEFAULT_N_OBS = 5


def resolve_exp_dir(project_root: Path, exp_dir_arg: str | None) -> Path | None:
    if not exp_dir_arg:
        return None
    exp_dir = Path(exp_dir_arg)
    if not exp_dir.is_absolute():
        candidate = project_root / "experiments" / exp_dir_arg
        exp_dir = candidate if candidate.exists() else project_root / exp_dir_arg
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")
    return exp_dir.resolve()


def resolve_config_path(name_or_path: str, project_root: Path | None = None) -> Path:
    """Resolve config name/path under ``configs/``."""
    root = project_root or repo_root()
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p.resolve()
    for base in (Path.cwd(), root):
        cand = (base / name_or_path).resolve()
        if cand.is_file():
            return cand
    if not p.suffix:
        for stem in (name_or_path, f"{name_or_path}_config"):
            candidate = root / "configs" / f"{stem}.yaml"
            if candidate.exists():
                return candidate.resolve()
    candidate = root / "configs" / name_or_path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Config not found: {name_or_path} (looked under configs/)")
