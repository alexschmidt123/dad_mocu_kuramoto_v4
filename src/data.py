"""
Shared trajectory tables under ``data/<config>_T<T>/`` plus table lookup at train/eval time.
"""

from __future__ import annotations

import json
import zlib
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.config import SBOEDConfig, load_config
from src.swing_equation_ode.design import build_catalog, build_simulator, enumerate_no_repeat_sequences
from src.swing_equation_ode.simulator import mk_to_json, system_mk

DATA_ROOT = "data"


def save_json(data: Any, path: Path, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# --- paths -----------------------------------------------------------------

def data_slug(cfg: SBOEDConfig) -> str:
    return f"{cfg.name}_T{cfg.step_number}"


def data_dir(project_root: Path, cfg: SBOEDConfig) -> Path:
    return project_root / DATA_ROOT / data_slug(cfg)


def is_present(path: Path) -> bool:
    return (path / "train.json").is_file() and (path / "test.json").is_file()


def resolve_exp_config_path(exp_dir: Path) -> Path:
    from src.experiment_layout import resolve_experiment_config_path

    return resolve_experiment_config_path(exp_dir)


def resolve_data_dir(exp_dir: Path, project_root: Path) -> Path:
    from src.experiment_layout import read_linked_data_dir

    try:
        d = read_linked_data_dir(exp_dir)
        if is_present(d):
            return d.resolve()
        raise FileNotFoundError(
            f"Data not found at {d} (stale data_dir in run_config? re-run generate-data)"
        )
    except FileNotFoundError:
        pass

    try:
        from src.config import load_config, with_step_number

        cfg = load_config(resolve_exp_config_path(exp_dir))
        step_file = exp_dir / "step_number.txt"
        if step_file.is_file() and not (exp_dir / "run_config.yaml").is_file():
            cfg = with_step_number(cfg, int(step_file.read_text(encoding="utf-8").strip()))
        d = data_dir(project_root, cfg)
        if is_present(d):
            return d.resolve()
        raise FileNotFoundError(
            f"Data not found at {d} (need train.json + test.json; run: python -m src.cli generate-data)"
        )
    except FileNotFoundError:
        pass

    legacy = exp_dir / "data"
    if (legacy / "train.json").is_file():
        return legacy.resolve()

    raise FileNotFoundError(f"No data_dir in run_config.yaml (or legacy pointer) in {exp_dir}")


# --- JSON tables -----------------------------------------------------------

def build_system_record(
    M: np.ndarray,
    K: np.ndarray,
    trajectories: list[dict[str, Any]],
    *,
    trajectory_mode: str = "full_bank",
) -> dict[str, Any]:
    M_list, K_list = mk_to_json(M, K)
    return {
        "M": M_list,
        "K": K_list,
        "trajectories": trajectories,
        "trajectory_mode": trajectory_mode,
    }


def trajectory_storage_mode(cfg: SBOEDConfig) -> str:
    """
    ``full_bank``: pre-simulate all no-repeat sequences (feasible for small T).
    ``on_demand``: store θ only; PyCUDA sim at train/eval lookup time (T ≥ 5).
    ``auto``: full bank when T ≤ ``full_bank_max_T``, else on_demand.
    """
    mode = str(cfg.data.get("trajectory_mode", "auto")).lower()
    threshold = int(cfg.data.get("full_bank_max_T", 4))
    if mode == "auto":
        return "on_demand" if cfg.step_number > threshold else "full_bank"
    if mode not in {"full_bank", "on_demand"}:
        raise ValueError(
            f"data_generation.trajectory_mode must be full_bank, on_demand, or auto; got {mode!r}"
        )
    return mode


def system_uses_on_demand(system: dict[str, Any]) -> bool:
    if system.get("trajectory_mode") == "on_demand":
        return True
    return "trajectories" in system and len(system.get("trajectories") or []) == 0


@dataclass
class TrajectorySimContext:
    """Thread-local CUDA lookup context (stable noisy ``y`` per θ + prefix)."""

    cfg: SBOEDConfig
    split_seed: int
    _cache: dict[tuple[int, ...], dict[str, list[float]]] = field(default_factory=dict)
    _engine: Any | None = field(default=None, repr=False)

    def engine(self):
        if self._engine is None:
            from src.swing_equation_ode.cuda import CudaTrajectoryEngine

            self._engine = CudaTrajectoryEngine(build_simulator(self.cfg), build_catalog(self.cfg))
        return self._engine


_trajectory_sim_ctx: ContextVar[TrajectorySimContext | None] = ContextVar(
    "trajectory_sim_ctx", default=None,
)


def set_trajectory_sim_context(cfg: SBOEDConfig, split_seed: int) -> TrajectorySimContext:
    ctx = TrajectorySimContext(cfg=cfg, split_seed=int(split_seed))
    _trajectory_sim_ctx.set(ctx)
    return ctx


def clear_trajectory_sim_context() -> None:
    _trajectory_sim_ctx.set(None)


def get_trajectory_sim_context() -> TrajectorySimContext | None:
    return _trajectory_sim_ctx.get()


def _stable_observation_seed(system: dict[str, Any], prefix: tuple[int, ...], split_seed: int) -> int:
    M, K = system_mk(system, len(system["M"]))
    payload = np.concatenate([M, K, np.asarray(prefix, dtype=np.int64)])
    return int((int(split_seed) + zlib.adler32(payload.tobytes())) % (2**32 - 1))


def simulate_sequence_cuda(
    system: dict[str, Any],
    sequence: list[int],
    cfg: SBOEDConfig,
    *,
    split_seed: int,
    ctx: TrajectorySimContext | None = None,
) -> dict[str, list[float]]:
    """One θ, one probe sequence: noiseless ``y_sim`` + reproducible noisy ``y``."""
    seq = tuple(int(a) for a in sequence)
    if not seq:
        return {"sequence": [], "y_sim": [], "y": []}

    if ctx is None:
        ctx = get_trajectory_sim_context()
    if ctx is None:
        ctx = TrajectorySimContext(cfg=cfg, split_seed=int(split_seed))

    cached = ctx._cache.get(seq)
    if cached is not None:
        return cached

    M, K = system_mk(system, cfg.N)
    rows = ctx.engine().simulate_all_sequences(
        M, K, [seq], 0.0, np.random.default_rng(0),
        batch_size=_cuda_batch_size(cfg),
    )
    row = rows[0]
    y_sim = np.asarray(row["y_sim"], dtype=np.float64)
    noise_seed = _stable_observation_seed(system, seq, split_seed)
    noise = np.random.default_rng(noise_seed).normal(0.0, cfg.sigma_y, size=len(y_sim))
    out = {
        "sequence": list(seq),
        "y_sim": y_sim.tolist(),
        "y": (y_sim + noise).tolist(),
    }
    ctx._cache[seq] = out
    return out


def save_tables(payload: dict[str, Any], path: Path) -> None:
    save_json(payload, path)


def load_tables(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# --- data-run metadata (source of truth for train / eval) ------------------

@dataclass(frozen=True)
class DataRunMeta:
    """Fields fixed at ``generate-data`` time; stored in ``train.json`` + ``manifest.yaml``."""

    data_path: Path
    data_slug: str
    step_number: int
    n_actions: int
    n_buses: int
    theta_dim: int
    sigma_y: float
    probe_amplitudes: list[float]
    probe_duration: float
    catalog: list[tuple[float, int, float]]
    train_seed: int
    test_seed: int
    config_path: Path | None

    def policy_meta(self) -> dict[str, Any]:
        return {
            "n_actions": self.n_actions,
            "step_number": self.step_number,
            "sigma_y": self.sigma_y,
            "data_slug": self.data_slug,
            "data_path": str(self.data_path.resolve()),
        }

    def validate_against_config(self, cfg: SBOEDConfig) -> None:
        from src.swing_equation_ode.design import build_catalog

        if cfg.N != self.n_buses:
            raise ValueError(f"cfg.N={cfg.N} != data n_buses={self.n_buses}")
        if cfg.step_number != self.step_number:
            raise ValueError(f"cfg.step_number={cfg.step_number} != data T={self.step_number}")
        if abs(cfg.sigma_y - self.sigma_y) > 1e-12:
            raise ValueError(f"cfg.sigma_y={cfg.sigma_y} != data sigma_y={self.sigma_y}")
        if len(build_catalog(cfg)) != self.n_actions:
            raise ValueError(
                f"catalog size {len(build_catalog(cfg))} != data n_actions={self.n_actions}"
            )


def _meta_from_payload(data_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if "meta" not in payload:
        raise KeyError(f"{data_path}: JSON missing 'meta' block; re-run generate-data")
    return dict(payload["meta"])


def load_data_run_meta(data_path: Path) -> DataRunMeta:
    """Read run metadata from ``train.json`` (primary) and ``manifest.yaml`` (config path)."""
    data_path = data_path.resolve()
    train_payload = load_tables(data_path / "train.json")
    test_payload = load_tables(data_path / "test.json")
    tm = _meta_from_payload(data_path / "train.json", train_payload)
    test_m = _meta_from_payload(data_path / "test.json", test_payload)

    manifest: dict[str, Any] = {}
    manifest_path = data_path / "manifest.yaml"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}

    config_path = manifest.get("config")
    if config_path:
        config_path = Path(config_path).resolve()
    elif tm.get("config"):
        config_path = Path(tm["config"]).resolve()

    slug = str(manifest.get("data_slug") or data_path.name)
    catalog_raw = tm.get("catalog") or []
    catalog = [tuple(x) for x in catalog_raw]

    return DataRunMeta(
        data_path=data_path,
        data_slug=slug,
        step_number=int(tm["step_number"]),
        n_actions=int(tm["n_actions"]),
        n_buses=int(tm["n_buses"]),
        theta_dim=int(tm.get("theta_dim", 2 * int(tm["n_buses"]))),
        sigma_y=float(tm["sigma_y"]),
        probe_amplitudes=[float(x) for x in tm["probe_amplitudes"]],
        probe_duration=float(tm["probe_duration"]),
        catalog=catalog,
        train_seed=int(tm.get("seed", manifest.get("train_seed", 0))),
        test_seed=int(test_m.get("seed", manifest.get("test_seed", 1))),
        config_path=config_path,
    )


def get_systems(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "systems" in payload:
        return payload["systems"]
    return payload.get("records", [])


def load_split_systems(data_path: Path) -> tuple[list[dict], list[dict]]:
    train_payload = load_tables(data_path / "train.json")
    test_payload = load_tables(data_path / "test.json")
    return get_systems(train_payload), get_systems(test_payload)


# --- GPU generation --------------------------------------------------------

def _swing_bounds(cfg: SBOEDConfig) -> tuple[float, float, float, float]:
    sw = cfg.swing
    return (
        float(sw.get("M_lower", 0.01)),
        float(sw.get("M_upper", 0.06)),
        float(sw.get("K_lower", 0.05)),
        float(sw.get("K_upper", 0.50)),
    )


def _cuda_batch_size(cfg: SBOEDConfig) -> int:
    return int(cfg.data.get("cuda_batch_size", 512))


def simulate_all_trajectories_cuda(
    sim,
    M: np.ndarray,
    K: np.ndarray,
    catalog,
    sequences: list[tuple[int, ...]],
    sigma_y: float,
    rng: np.random.Generator,
    cfg: SBOEDConfig,
    *,
    progress_label: str = "",
) -> list[dict[str, Any]]:
    from src.swing_equation_ode.cuda import CudaTrajectoryEngine

    engine = CudaTrajectoryEngine(sim, catalog)
    return engine.simulate_all_sequences(
        M, K, sequences, sigma_y, rng,
        batch_size=_cuda_batch_size(cfg),
        progress_label=progress_label,
    )


def generate_split(
    cfg: SBOEDConfig,
    split: str,
    seed: int,
    theta_sample_size: int | None = None,
) -> dict[str, Any]:
    if theta_sample_size is None:
        theta_sample_size = cfg.theta_sample_size(split)

    from src.contrastive.spce import sample_mk_prior

    rng = np.random.default_rng(seed)
    catalog = build_catalog(cfg)
    sim = build_simulator(cfg)
    step_number = cfg.step_number
    n_buses = cfg.N
    n_actions = len(catalog)
    storage_mode = trajectory_storage_mode(cfg)
    sequences = (
        enumerate_no_repeat_sequences(catalog, step_number)
        if storage_mode == "full_bank"
        else []
    )
    n_seq = len(sequences)

    M_lo, M_hi, K_lo, K_hi = _swing_bounds(cfg)
    M_s, K_s = sample_mk_prior(
        M_lo, M_hi, K_lo, K_hi, theta_sample_size, rng, n_buses=n_buses,
    )

    systems: list[dict[str, Any]] = []
    for i in range(theta_sample_size):
        M_vec = M_s[i]
        K_vec = K_s[i]
        if storage_mode == "full_bank":
            print(
                f"  [{split}] θ sample {i + 1}/{theta_sample_size}  "
                f"M_bus[1:{n_buses}]∈[{M_vec.min():.4f},{M_vec.max():.4f}]  "
                f"K_bus[1:{n_buses}]∈[{K_vec.min():.4f},{K_vec.max():.4f}]  "
                f"CUDA bank {n_seq} trajectories (T={step_number})"
            )
            trajectories = simulate_all_trajectories_cuda(
                sim, M_vec, K_vec, catalog, sequences, cfg.sigma_y, rng, cfg,
                progress_label=f"[{split}] θ {i + 1}/{theta_sample_size}",
            )
        else:
            print(
                f"  [{split}] θ sample {i + 1}/{theta_sample_size}  "
                f"M_bus[1:{n_buses}]∈[{M_vec.min():.4f},{M_vec.max():.4f}]  "
                f"K_bus[1:{n_buses}]∈[{K_vec.min():.4f},{K_vec.max():.4f}]  "
                f"on-demand PyCUDA (T={step_number}, no full sequence bank)"
            )
            trajectories = []
        systems.append(
            build_system_record(
                M_vec, K_vec, trajectories, trajectory_mode=storage_mode,
            )
        )

    return {
        "meta": {
            "split": split,
            "seed": seed,
            "theta_sample_size": theta_sample_size,
            "n_buses": n_buses,
            "theta_dim": 2 * n_buses,
            "step_number": step_number,
            "n_actions": n_actions,
            "n_sequences_per_system": n_seq,
            "trajectory_mode": storage_mode,
            "history_dependent": True,
            "backend": "cuda",
            "probe_amplitudes": list(cfg.probe_amplitudes),
            "probe_duration": cfg.probe_duration,
            "sigma_y": cfg.sigma_y,
            "catalog": [d.as_tuple() for d in catalog],
        },
        "systems": systems,
    }


def ensure_data(project_root: Path, cfg: SBOEDConfig) -> Path:
    d = data_dir(project_root, cfg)
    train_path = d / "train.json"
    test_path = d / "test.json"

    if is_present(d):
        print(f"Using existing data → {d}")
        return d

    d.mkdir(parents=True, exist_ok=True)
    mode = trajectory_storage_mode(cfg)
    print(
        f"Generating data → {d}\n"
        f"  config={cfg.name}  T={cfg.step_number}  amplitudes={cfg.probe_amplitudes}\n"
        f"  trajectory_mode={mode}"
    )

    train_payload = generate_split(cfg, "train", int(cfg.data.get("train_seed", 0)))
    save_tables(train_payload, train_path)

    test_payload = generate_split(cfg, "test", int(cfg.data.get("test_seed", 1)))
    save_tables(test_payload, test_path)

    tm = train_payload["meta"]
    manifest: dict[str, Any] = {
        "data_slug": data_slug(cfg),
        "config": str(cfg.config_path.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "step_number": int(tm["step_number"]),
        "n_actions": int(tm["n_actions"]),
        "n_buses": int(tm["n_buses"]),
        "sigma_y": float(tm["sigma_y"]),
        "probe_amplitudes": list(tm["probe_amplitudes"]),
        "train_seed": int(tm["seed"]),
        "test_seed": int(test_payload["meta"]["seed"]),
        "train_theta_sample_size": len(get_systems(train_payload)),
        "test_theta_sample_size": len(get_systems(test_payload)),
        "trajectory_mode": str(tm.get("trajectory_mode", "full_bank")),
    }
    with (d / "manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(manifest, f)

    print(f"Data saved → {d}")
    return d


def grid_f_path(project_root: Path, cfg: SBOEDConfig) -> Path:
    """Scalar (M,K) grid cache of one-step F(θ_n, ξ_a); optional legacy artifact."""
    return data_dir(project_root, cfg) / "grid_f.npz"


def _load_grid_f_array(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "grid_f" in data:
            return np.asarray(data["grid_f"], dtype=np.float64)
        if "grid_mu" in data:
            return np.asarray(data["grid_mu"], dtype=np.float64)
    raise KeyError(f"{path} must contain 'grid_f'")


def build_and_save_grid_f(project_root: Path, cfg: SBOEDConfig) -> Path:
    """One-step F on scalar (M,K) grid (legacy helper; eval uses MC support)."""
    path = grid_f_path(project_root, cfg)
    legacy = path.parent / "grid_mu.npz"
    if path.is_file():
        return path
    if legacy.is_file():
        return legacy

    from src.contrastive.spce import build_mk_grid

    catalog = build_catalog(cfg)
    sim = build_simulator(cfg)
    grid_side = int(cfg.prior.get("grid_side", 6))
    sw = cfg.swing
    _, M_grid, K_grid = build_mk_grid(
        float(sw["M_lower"]), float(sw["M_upper"]),
        float(sw["K_lower"]), float(sw["K_upper"]),
        grid_side,
    )
    n_actions = len(catalog)
    n_g = len(M_grid)
    grid_f = np.zeros((n_actions, n_g), dtype=np.float64)
    print(f"  Building grid_f cache ({n_actions} actions × {n_g} nodes) → {path}")
    for a in range(n_actions):
        grid_f[a] = sim.map_batch(M_grid, K_grid, catalog[a])

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        grid_f=grid_f,
        M_grid=M_grid,
        K_grid=K_grid,
        grid_side=grid_side,
    )
    return path


def load_grid_f(project_root: Path, cfg: SBOEDConfig) -> np.ndarray:
    path = grid_f_path(project_root, cfg)
    legacy = path.parent / "grid_mu.npz"
    if not path.is_file() and not legacy.is_file():
        build_and_save_grid_f(project_root, cfg)
    return _load_grid_f_array(path if path.is_file() else legacy)


# --- lookup at train / eval time -------------------------------------------

def _trajectory_y_sim(traj: dict[str, Any]) -> list[float]:
    """ODE max-ROCOF before noise (sPCE likelihood centre; not a policy input)."""
    if "y_sim" not in traj:
        raise KeyError("trajectory missing 'y_sim' (regenerate data)")
    return list(traj["y_sim"])


def validate_trajectory_y_sim(
    systems: list[dict[str, Any]],
    *,
    split: str,
) -> None:
    if not systems:
        raise ValueError(f"{split}: empty system list")
    if system_uses_on_demand(systems[0]):
        return
    missing = 0
    for sys in systems:
        for traj in sys.get("trajectories", []):
            if "y_sim" not in traj:
                missing += 1
    if missing:
        raise ValueError(
            f"{split}.json: {missing} rows lack 'y_sim'. "
            "Regenerate data (CUDA writes sequence, y_sim, y per row)."
        )


def _lookup_trajectory_row(system: dict[str, Any], sequence: list[int]) -> dict[str, list[float]]:
    key = tuple(int(a) for a in sequence)
    if not system_uses_on_demand(system):
        for traj in system.get("trajectories", []):
            seq = tuple(int(a) for a in traj["sequence"])
            if seq == key:
                return traj
        for traj in system.get("trajectories", []):
            seq = tuple(int(a) for a in traj["sequence"])
            if len(seq) >= len(key) and seq[: len(key)] == key:
                return {
                    "sequence": list(key),
                    "y_sim": _trajectory_y_sim(traj)[: len(key)],
                    "y": list(traj["y"][: len(key)]),
                }
        raise KeyError(f"No trajectory with prefix {list(key)} in trajectory table")
    ctx = get_trajectory_sim_context()
    if ctx is None:
        raise RuntimeError(
            "on-demand trajectory lookup requires TrajectorySimContext "
            "(call set_trajectory_sim_context before train/eval)."
        )
    return simulate_sequence_cuda(system, list(key), ctx.cfg, split_seed=ctx.split_seed, ctx=ctx)


def lookup_sequence_y_sim(system: dict[str, Any], sequence: list[int]) -> list[float]:
    return _trajectory_y_sim(_lookup_trajectory_row(system, sequence))


def lookup_prefix_y_sim(system: dict[str, Any], prefix: list[int]) -> list[float]:
    if not prefix:
        return []
    row = _lookup_trajectory_row(system, prefix)
    return _trajectory_y_sim(row)[: len(prefix)]


def lookup_sequence_y(system: dict[str, Any], sequence: list[int]) -> list[float]:
    """Full-length sequence; offline bank or on-demand PyCUDA."""
    return list(_lookup_trajectory_row(system, sequence)["y"])


def lookup_prefix_y(system: dict[str, Any], prefix: list[int]) -> list[float]:
    """
    Observations for the first len(prefix) probes.

    For fixed θ, y_t depends only on (ξ_1,…,ξ_t), not later designs, so any table
    row matching the prefix gives the same y_{1:t} (history-consistent).
    """
    if not prefix:
        return []
    row = _lookup_trajectory_row(system, prefix)
    return list(row["y"][: len(prefix)])


def simulate_rollout(
    cfg: SBOEDConfig,
    system: dict[str, Any],
    sequence: list[int],
    rng: np.random.Generator,
) -> list[float]:
    M, K = system_mk(system, cfg.N)
    sim = build_simulator(cfg)
    catalog = build_catalog(cfg)
    designs = [catalog[int(a)] for a in sequence]
    return sim.simulate_sequence(M, K, designs, add_noise=cfg.sigma_y, rng=rng)


def sample_trajectory_rollout(
    systems: list[dict[str, Any]],
    rng: np.random.Generator,
    *,
    curriculum_weights: list[np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[int], list[float]]:
    """Uniform system + trajectory, or curriculum weights per system (sum to 1)."""
    i_sys = int(rng.integers(len(systems)))
    sys = systems[i_sys]
    trajs = sys["trajectories"]
    if curriculum_weights is not None:
        w = np.asarray(curriculum_weights[i_sys], dtype=np.float64)
        w = w / w.sum()
        traj = trajs[int(rng.choice(len(trajs), p=w))]
    else:
        traj = trajs[int(rng.integers(len(trajs)))]
    return sys, list(traj["sequence"]), list(traj["y"])
