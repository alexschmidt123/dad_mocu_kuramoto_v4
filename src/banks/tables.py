"""
Shared system data under ``data/<system>/`` (or config ``data.dataset_dir``).

One directory per grid system is reused by all experiment types
(``objective_based`` / ``eig_based``), methods, and horizons T.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.config import SBOEDConfig, load_config
from src.domains.swing.design import (
    build_catalog,
    build_simulator,
    unrank_sequence_chunk,
)
from src.domains.swing.simulator import mk_to_json, system_mk

try:  # pragma: no cover - cosmetic progress helper
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)

from src.banks.paths import DATA_ROOT, resolve_shared_data_dir, system_name_for_data

TRAJECTORY_MODE = "one_step_bank"


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
    """Shared data key per system (T- and experiment-type-independent)."""
    return system_name_for_data(cfg)


def legacy_data_slug(cfg: SBOEDConfig) -> str:
    """Pre-renaming slug (e.g. ``ieee14_config``)."""
    return cfg.name


def legacy_horizon_data_slug(cfg: SBOEDConfig) -> str:
    """Older per-horizon folders (e.g. ``ieee14_T3``)."""
    return f"{cfg.run_slug}_T{cfg.step_number}"


def data_dir(project_root: Path, cfg: SBOEDConfig) -> Path:
    """Canonical shared data dir (same for all experiment types on this system)."""
    return resolve_shared_data_dir(project_root, cfg)


def data_dir_candidates(project_root: Path, cfg: SBOEDConfig) -> list[Path]:
    """Lookup order when resolving an existing bank (canonical first, then legacy)."""
    root = project_root / DATA_ROOT
    seen: set[str] = set()
    out: list[Path] = []

    def add_path(p: Path) -> None:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)

    add_path(resolve_shared_data_dir(project_root, cfg))
    for slug in (
        data_slug(cfg),
        f"{data_slug(cfg)}_full_delta_f",
        legacy_data_slug(cfg),
        legacy_horizon_data_slug(cfg),
        f"{cfg.name}_T{cfg.step_number}",
        cfg.run_slug,
    ):
        add_path(root / slug)

    if root.is_dir():
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            name = p.name
            if name.startswith(f"{cfg.run_slug}_T") or name.startswith(f"{cfg.name}_T"):
                add_path(p)

    return out


def resolve_data_path(project_root: Path, cfg: SBOEDConfig) -> Path:
    """Prefer canonical shared dir if present; else first non-empty legacy candidate."""
    canonical = resolve_shared_data_dir(project_root, cfg)
    if canonical.is_dir() and any(canonical.iterdir()):
        return canonical
    for candidate in data_dir_candidates(project_root, cfg):
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return canonical


def is_present(path: Path) -> bool:
    return (path / "train.json").is_file() and (path / "test.json").is_file()


def resolve_exp_config_path(exp_dir: Path) -> Path:
    from src.reporting.run_context import resolve_experiment_config_path

    return resolve_experiment_config_path(exp_dir)


def resolve_data_dir(exp_dir: Path, project_root: Path) -> Path:
    from src.reporting.run_context import read_linked_data_dir

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
        for candidate in data_dir_candidates(project_root, cfg):
            if is_present(candidate):
                return candidate.resolve()
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
    trajectories: list[dict[str, Any]] | None = None,
    *,
    trajectory_bank: str | None = None,
    n_sequences: int = 0,
) -> dict[str, Any]:
    M_list, K_list = mk_to_json(M, K)
    out: dict[str, Any] = {
        "M": M_list,
        "K": K_list,
        "trajectory_mode": TRAJECTORY_MODE,
    }
    if trajectory_bank is not None:
        out["trajectory_bank"] = trajectory_bank
        out["n_sequences"] = int(n_sequences)
    else:
        out["trajectories"] = trajectories or []
    return out


def _bank_root(data_path: Path, split: str, system_index: int) -> Path:
    return data_path / f"{split}_banks" / f"{system_index:04d}"


def _resolve_bank_dir(system: dict[str, Any], data_path: Path | None = None) -> Path | None:
    rel = system.get("trajectory_bank")
    if not rel:
        return None
    bank = Path(str(rel))
    if bank.is_absolute():
        return bank.resolve()
    if data_path is not None:
        return (data_path / bank).resolve()
    return bank.resolve()


@dataclass
class TrajectoryBank:
    root: Path
    step_number: int
    n_sequences: int
    _sequences: np.memmap | None = field(default=None, repr=False)
    _y_sim: np.memmap | None = field(default=None, repr=False)
    _y: np.memmap | None = field(default=None, repr=False)

    def _open(self) -> None:
        if self._sequences is not None:
            return
        T = self.step_number
        n = self.n_sequences
        self._sequences = np.memmap(self.root / "sequences.npy", dtype=np.int16, mode="r", shape=(n, T))
        self._y_sim = np.memmap(self.root / "y_sim.npy", dtype=np.float32, mode="r", shape=(n, T))
        self._y = np.memmap(self.root / "y.npy", dtype=np.float32, mode="r", shape=(n, T))

    def find_exact_row(self, key: tuple[int, ...]) -> int | None:
        self._open()
        assert self._sequences is not None
        key_arr = np.asarray(key, dtype=np.int16)
        lo, hi = 0, self.n_sequences
        while lo < hi:
            mid = (lo + hi) // 2
            row = np.asarray(self._sequences[mid], dtype=np.int16)
            if np.array_equal(row, key_arr):
                return mid
            if tuple(row.tolist()) < key:
                lo = mid + 1
            else:
                hi = mid
        return None

    def row_payload(self, row: int, *, length: int | None = None) -> dict[str, list[float]]:
        self._open()
        assert self._sequences is not None and self._y_sim is not None and self._y is not None
        n = length if length is not None else self.step_number
        return {
            "sequence": [int(x) for x in self._sequences[row, :n]],
            "y_sim": [float(x) for x in self._y_sim[row, :n]],
            "y": [float(x) for x in self._y[row, :n]],
        }


def open_trajectory_bank(system: dict[str, Any], data_path: Path) -> TrajectoryBank | None:
    bank_dir = _resolve_bank_dir(system, data_path)
    if bank_dir is None:
        return None
    meta_path = bank_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Trajectory bank missing meta.json: {bank_dir}")
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    return TrajectoryBank(
        root=bank_dir,
        step_number=int(meta["step_number"]),
        n_sequences=int(meta["n_sequences"]),
    )


def write_trajectory_bank_mmap(
    bank_dir: Path,
    *,
    n_actions: int,
    step_number: int,
    n_sequences: int,
    sim,
    catalog,
    M: np.ndarray,
    K: np.ndarray,
    sigma_y: float,
    rng: np.random.Generator,
    cfg: SBOEDConfig,
    progress_label: str = "",
) -> None:
    """Simulate all P(n_actions,T) sequences in chunks; store as mmap arrays."""
    bank_dir.mkdir(parents=True, exist_ok=True)
    T = step_number
    seq_path = bank_dir / "sequences.npy"
    y_sim_path = bank_dir / "y_sim.npy"
    y_path = bank_dir / "y.npy"
    seq_mm = np.memmap(seq_path, dtype=np.int16, mode="w+", shape=(n_sequences, T))
    y_sim_mm = np.memmap(y_sim_path, dtype=np.float32, mode="w+", shape=(n_sequences, T))
    y_mm = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(n_sequences, T))

    chunk_size = int(cfg.data.get("full_bank_chunk_size", 10_000))
    for start in range(0, n_sequences, chunk_size):
        end = min(n_sequences, start + chunk_size)
        chunk_seqs = unrank_sequence_chunk(n_actions, T, start, end - start)
        if progress_label:
            print(f"    {progress_label} GPU trajectories {end}/{n_sequences}")
        chunk_trajs = simulate_all_trajectories_cuda(
            sim, M, K, catalog, chunk_seqs, sigma_y, rng, cfg,
        )
        for j, traj in enumerate(chunk_trajs):
            row = start + j
            seq_mm[row, :] = np.asarray(traj["sequence"], dtype=np.int16)
            y_sim_mm[row, :] = np.asarray(traj["y_sim"], dtype=np.float32)
            y_mm[row, :] = np.asarray(traj["y"], dtype=np.float32)

    seq_mm.flush()
    y_sim_mm.flush()
    y_mm.flush()
    del seq_mm, y_sim_mm, y_mm

    save_json({
        "step_number": T,
        "n_sequences": n_sequences,
        "n_actions": n_actions,
        "dtype": {"sequences": "int16", "y_sim": "float32", "y": "float32"},
    }, bank_dir / "meta.json")


def trajectory_bank_complete(bank_dir: Path, n_sequences: int, step_number: int) -> bool:
    """True when mmap bank files match expected shape."""
    meta_path = bank_dir / "meta.json"
    if not meta_path.is_file():
        return False
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    if int(meta.get("n_sequences", -1)) != int(n_sequences):
        return False
    if int(meta.get("step_number", -1)) != int(step_number):
        return False
    T = int(step_number)
    n = int(n_sequences)
    expected_bytes = {
        "sequences.npy": n * T * np.dtype(np.int16).itemsize,
        "y_sim.npy": n * T * np.dtype(np.float32).itemsize,
        "y.npy": n * T * np.dtype(np.float32).itemsize,
    }
    for name, nbytes in expected_bytes.items():
        path = bank_dir / name
        if not path.is_file() or path.stat().st_size < nbytes:
            return False
    return True


def _system_bank_complete(
    system: dict[str, Any],
    *,
    data_dir: Path,
    split: str,
    index: int,
    n_sequences: int,
    step_number: int,
) -> bool:
    if system.get("trajectory_mode") not in (None, TRAJECTORY_MODE):
        return False
    if system.get("trajectory_bank"):
        bank_dir = _bank_root(data_dir, split, index)
        return trajectory_bank_complete(bank_dir, n_sequences, step_number)
    trajs = system.get("trajectories") or []
    return len(trajs) == n_sequences and n_sequences > 0


def _hydrate_systems_from_disk_banks(
    systems: list[dict[str, Any] | None],
    *,
    data_dir: Path,
    split: str,
    M_s: np.ndarray,
    K_s: np.ndarray,
    n_seq: int,
    step_number: int,
    use_mmap_bank: bool,
) -> int:
    """Resume mmap banks on disk when ``train.json`` / ``test.json`` are not written yet."""
    if not use_mmap_bank or n_seq <= 0:
        return 0
    n_hydrated = 0
    for i, slot in enumerate(systems):
        if slot is not None:
            continue
        bank_dir = _bank_root(data_dir, split, i)
        if not trajectory_bank_complete(bank_dir, n_seq, step_number):
            continue
        bank_rel = str(bank_dir.relative_to(data_dir))
        systems[i] = build_system_record(
            M_s[i],
            K_s[i],
            trajectory_bank=bank_rel,
            n_sequences=n_seq,
        )
        n_hydrated += 1
    return n_hydrated


def _split_payload_path(data_dir: Path, split: str) -> Path:
    return data_dir / ("train.json" if split == "train" else "test.json")


def _load_split_payload_if_any(data_dir: Path, split: str) -> dict[str, Any] | None:
    path = _split_payload_path(data_dir, split)
    if not path.is_file():
        return None
    return load_tables(path)


def data_bundle_complete(cfg: SBOEDConfig, data_path: Path) -> bool:
    """All θ banks / inline tables present for this config + T."""
    data_path = data_path.resolve()
    if not is_present(data_path):
        return False
    manifest = load_manifest(data_path)
    if manifest and manifest.get("trajectory_mode") != TRAJECTORY_MODE:
        return False
    if manifest and bool(manifest.get("history_dependent", False)):
        return False
    try:
        validate_data_bundle(cfg, data_path)
    except (FileNotFoundError, ValueError):
        return False
    for split in ("train", "test"):
        payload = load_tables(_split_payload_path(data_path, split))
        meta = payload["meta"]
        n_seq = int(meta.get("n_sequences_per_system", 0))
        T = int(meta["step_number"])
        expected_n = int(meta.get("theta_sample_size", 0))
        raw = get_system_slots(payload)
        if len(raw) < expected_n:
            return False
        for i in range(expected_n):
            sys = raw[i]
            if not isinstance(sys, dict):
                return False
            if not _system_bank_complete(
                sys, data_dir=data_path, split=split, index=i,
                n_sequences=n_seq, step_number=T,
            ):
                return False
    return True


def _can_resume_existing_payloads(data_path: Path) -> bool:
    """Avoid loading obsolete full-bank JSON as resumable partial data."""
    if not is_present(data_path):
        return True
    manifest = load_manifest(data_path)
    if not manifest:
        return False
    return (
        manifest.get("trajectory_mode") == TRAJECTORY_MODE
        and not bool(manifest.get("history_dependent", False))
    )


def _reject_legacy_trajectory_mode(cfg: SBOEDConfig) -> None:
    """Only reset-based one-step banks are supported."""
    mode = cfg.data.get("trajectory_mode")
    if mode is None:
        return
    if str(mode).lower() != TRAJECTORY_MODE:
        raise ValueError(
            f"data_generation.trajectory_mode={mode!r} is no longer supported; "
            f"remove it from config (data always uses {TRAJECTORY_MODE!r})."
        )


def _validate_one_step_bank_systems(
    cfg: SBOEDConfig,
    data_path: Path,
    *,
    train_payload: dict[str, Any],
    test_payload: dict[str, Any],
) -> None:
    """Reject bundles without a complete reset-based per-action bank."""
    n_actions = len(build_catalog(cfg))
    errors: list[str] = []
    for split, payload in (("train", train_payload), ("test", test_payload)):
        for i, sys in enumerate(get_systems(payload)):
            mode = sys.get("trajectory_mode", TRAJECTORY_MODE)
            if mode != TRAJECTORY_MODE:
                errors.append(f"{split} θ {i}: trajectory_mode={mode!r}")
                continue
            if not _system_bank_complete(
                sys,
                data_dir=data_path,
                split=split,
                index=i,
                n_sequences=n_actions,
                step_number=1,
            ):
                errors.append(
                    f"{split} θ {i}: expected {n_actions} one-step action rows"
                )
                continue
            trajs = sys.get("trajectories") or []
            if trajs:
                seen = sorted(int((traj.get("sequence") or [-1])[0]) for traj in trajs)
                if seen != list(range(n_actions)):
                    errors.append(f"{split} θ {i}: action rows are not 0..{n_actions - 1}")
    if errors:
        preview = "\n  ".join(errors[:8])
        suffix = "\n  ..." if len(errors) > 8 else ""
        raise ValueError(
            f"Data at {data_path} is incomplete or invalid (one_step_bank required).\n  "
            f"{preview}{suffix}"
        )


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
        from src.domains.swing.design import build_catalog

        if cfg.N != self.n_buses:
            raise ValueError(f"cfg.N={cfg.N} != data n_buses={self.n_buses}")
        if self.step_number != 1:
            raise ValueError(
                f"data bank step_number={self.step_number} != 1 "
                "(expected reset one-step bank shared across experiment horizons)"
            )
        if abs(cfg.sigma_y - self.sigma_y) > 1e-12:
            raise ValueError(f"cfg.sigma_y={cfg.sigma_y} != data sigma_y={self.sigma_y}")
        if len(build_catalog(cfg)) != self.n_actions:
            raise ValueError(
                f"catalog size {len(build_catalog(cfg))} != data n_actions={self.n_actions}"
            )


def load_manifest(data_path: Path) -> dict[str, Any]:
    """Read ``manifest.yaml`` from a data bundle directory (empty dict if missing)."""
    path = data_path.resolve() / "manifest.yaml"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def format_generation_duration(seconds: float) -> str:
    """Human-readable wall time for manifest display."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, sec = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m {sec}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins}m {sec}s"


def write_data_manifest(
    data_path: Path,
    cfg: SBOEDConfig,
    meta: dict[str, Any],
    *,
    generation_elapsed_sec: float = 0.0,
    data_complete: bool | None = None,
) -> Path:
    """
    Write ``manifest.yaml`` for a data bundle, including cumulative generation wall time.

    ``generation_elapsed_sec`` is added to any prior ``generation_seconds`` (resume-safe).
    """
    data_path = data_path.resolve()
    prev = load_manifest(data_path)
    total_sec = float(prev.get("generation_seconds", 0.0) or 0.0) + float(generation_elapsed_sec)

    manifest: dict[str, Any] = {
        "data_slug": data_slug(cfg),
        "config": str(cfg.config_path.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "step_number": int(meta["step_number"]),
        "n_actions": int(meta["n_actions"]),
        "n_buses": int(meta["n_buses"]),
        "sigma_y": float(meta["sigma_y"]),
        "probe_amplitudes": list(meta["probe_amplitudes"]),
        "train_seed": int(cfg.data.get("train_seed", 0)),
        "test_seed": int(cfg.data.get("test_seed", 1)),
        "train_theta_sample_size": cfg.theta_sample_size("train"),
        "test_theta_sample_size": cfg.theta_sample_size("test"),
        "trajectory_mode": TRAJECTORY_MODE,
        "generation_seconds": round(total_sec, 3),
        "generation_duration": format_generation_duration(total_sec),
        **cfg.run_labels(),
    }
    if data_complete is not None:
        manifest["data_complete"] = bool(data_complete)
    elif "data_complete" in prev:
        manifest["data_complete"] = prev["data_complete"]

    out = data_path / "manifest.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    return out


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

    manifest = load_manifest(data_path)

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


def get_system_slots(payload: dict[str, Any]) -> list[Any]:
    if "systems" in payload:
        return list(payload["systems"])
    return list(payload.get("records", []))


def get_systems(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in get_system_slots(payload) if isinstance(s, dict)]


def _hydrate_systems(systems: list[dict[str, Any]], data_path: Path) -> list[dict[str, Any]]:
    root = str(data_path.resolve())
    out: list[dict[str, Any]] = []
    for sys in systems:
        if not isinstance(sys, dict):
            continue
        if sys.get("trajectory_bank"):
            sys["_data_root"] = root
        out.append(sys)
    return out


def load_split_systems(data_path: Path) -> tuple[list[dict], list[dict]]:
    data_path = data_path.resolve()
    train_payload = load_tables(data_path / "train.json")
    test_payload = load_tables(data_path / "test.json")
    return (
        _hydrate_systems(get_systems(train_payload), data_path),
        _hydrate_systems(get_systems(test_payload), data_path),
    )


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
    from src.domains.swing.cuda import CudaTrajectoryEngine

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
    *,
    data_dir: Path | None = None,
    theta_start: int = 0,
    theta_end: int | None = None,
    existing_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if theta_sample_size is None:
        theta_sample_size = cfg.theta_sample_size(split)
    if theta_end is None:
        theta_end = theta_sample_size
    theta_start = max(0, int(theta_start))
    theta_end = min(int(theta_end), theta_sample_size)
    if theta_start >= theta_end:
        if existing_payload is not None:
            return existing_payload
        raise ValueError(f"empty θ range [{theta_start}, {theta_end}) for split={split}")

    from src.inference.spce import sample_mk_prior

    _reject_legacy_trajectory_mode(cfg)

    rng = np.random.default_rng(seed)
    catalog = build_catalog(cfg)
    bank_step_number = 1
    n_buses = cfg.N
    n_actions = len(catalog)
    n_seq = n_actions
    use_mmap_bank = False
    sim = build_simulator(cfg)

    M_lo, M_hi, K_lo, K_hi = _swing_bounds(cfg)
    M_s, K_s = sample_mk_prior(
        M_lo, M_hi, K_lo, K_hi, theta_sample_size, rng, n_buses=n_buses,
    )

    existing_slots = get_system_slots(existing_payload) if existing_payload is not None else []
    systems: list[dict[str, Any] | None] = [
        existing_slots[i] if i < len(existing_slots) and isinstance(existing_slots[i], dict) else None
        for i in range(theta_sample_size)
    ]
    if data_dir is not None:
        n_hydrated = _hydrate_systems_from_disk_banks(
            systems,
            data_dir=data_dir,
            split=split,
            M_s=M_s,
            K_s=K_s,
            n_seq=n_seq,
            step_number=bank_step_number,
            use_mmap_bank=use_mmap_bank,
        )
        if n_hydrated:
            print(f"  [{split}] hydrated {n_hydrated} θ from on-disk one-step banks (no JSON yet)")

    theta_iter = tqdm(
        range(theta_start, theta_end),
        desc=f"{split} θ",
        unit="θ",
        leave=False,
    )
    for i in theta_iter:
        if (
            data_dir is not None
            and systems[i] is not None
            and _system_bank_complete(
                systems[i],
                data_dir=data_dir,
                split=split,
                index=i,
                n_sequences=n_seq,
                step_number=bank_step_number,
            )
        ):
            theta_iter.set_postfix_str(f"{i + 1}/{theta_sample_size} skip")
            continue

        M_vec = M_s[i]
        K_vec = K_s[i]
        theta_iter.set_postfix_str(
            f"{i + 1}/{theta_sample_size} actions={n_actions}"
        )
        if use_mmap_bank:
            assert data_dir is not None
            bank_dir = _bank_root(data_dir, split, i)
            write_trajectory_bank_mmap(
                bank_dir,
                n_actions=n_actions,
                step_number=bank_step_number,
                n_sequences=n_seq,
                sim=sim,
                catalog=catalog,
                M=M_vec,
                K=K_vec,
                sigma_y=cfg.sigma_y,
                rng=rng,
                cfg=cfg,
                progress_label="",
            )
            bank_rel = str(bank_dir.relative_to(data_dir))
            systems[i] = build_system_record(
                M_vec, K_vec,
                trajectory_bank=bank_rel, n_sequences=n_seq,
            )
            continue
        sequences = [(a,) for a in range(n_actions)]
        trajectories = simulate_all_trajectories_cuda(
            sim, M_vec, K_vec, catalog, sequences, cfg.sigma_y, rng, cfg,
            progress_label="",
        )
        systems[i] = build_system_record(M_vec, K_vec, trajectories)

    full_range = theta_start == 0 and theta_end == theta_sample_size
    if full_range and any(s is None for s in systems):
        missing = [i for i, s in enumerate(systems) if s is None]
        raise RuntimeError(
            f"[{split}] incomplete data bundle: missing θ indices {missing[:8]}"
            f"{'...' if len(missing) > 8 else ''}. "
            "Run batched generate-data for remaining ranges."
        )

    meta_block: dict[str, Any] = dict(existing_payload["meta"]) if existing_payload else {}
    meta_block.update({
        "split": split,
        "seed": seed,
        "theta_sample_size": theta_sample_size,
        "n_buses": n_buses,
        "theta_dim": 2 * n_buses,
        "step_number": bank_step_number,
        "n_actions": n_actions,
        "n_sequences_per_system": n_seq,
        "trajectory_mode": TRAJECTORY_MODE,
        "history_dependent": False,
        "observation_model": "reset_one_step",
        "backend": "cuda",
        "probe_amplitudes": list(cfg.probe_amplitudes),
        "probe_duration": cfg.probe_duration,
        "sigma_y": cfg.sigma_y,
        "catalog": [d.as_tuple() for d in catalog],
        **{**cfg.run_labels(), "step_number": bank_step_number},
    })

    return {
        "meta": meta_block,
        "systems": systems,
    }


def _data_slug_matches(cfg: SBOEDConfig, folder_name: str) -> bool:
    """Accept canonical slug or legacy per-horizon folders."""
    if folder_name in {data_slug(cfg), legacy_data_slug(cfg)}:
        return True
    if folder_name.startswith(f"{cfg.run_slug}_T"):
        return True
    if folder_name.startswith(f"{cfg.name}_T"):
        return True
    return False


def validate_data_bundle(cfg: SBOEDConfig, data_path: Path) -> None:
    """Ensure on-disk tables match this config (shared one-step bank, any experiment T)."""
    data_path = data_path.resolve()
    if not is_present(data_path):
        raise FileNotFoundError(f"Missing train.json/test.json in {data_path}")

    _reject_legacy_trajectory_mode(cfg)

    expected_slug = data_slug(cfg)
    if not _data_slug_matches(cfg, data_path.name):
        raise ValueError(
            f"Data directory {data_path.name!r} does not match run slug "
            f"{expected_slug!r} (or legacy {cfg.run_slug}_T* / {cfg.name}_T*)"
        )

    meta = load_data_run_meta(data_path)
    meta.validate_against_config(cfg)

    manifest = load_manifest(data_path)

    mismatches: list[str] = []

    def check(label: str, expected: Any, actual: Any) -> None:
        if expected != actual:
            mismatches.append(f"{label}: expected {expected!r}, found {actual!r}")

    train_payload = load_tables(data_path / "train.json")
    test_payload = load_tables(data_path / "test.json")
    train_m = train_payload["meta"]
    test_m = test_payload["meta"]
    train_n = len(get_systems(train_payload))
    test_n = len(get_systems(test_payload))

    manifest_slug = manifest.get("data_slug", meta.data_slug)
    if manifest_slug != expected_slug and not _data_slug_matches(cfg, str(manifest_slug)):
        check("data_slug", expected_slug, manifest_slug)
    check("bank_step_number", 1, meta.step_number)
    check(
        "train_theta_sample_size",
        cfg.theta_sample_size("train"),
        manifest.get("train_theta_sample_size", train_n),
    )
    check(
        "test_theta_sample_size",
        cfg.theta_sample_size("test"),
        manifest.get("test_theta_sample_size", test_n),
    )
    check(
        "train_seed",
        int(cfg.data.get("train_seed", 0)),
        manifest.get("train_seed", train_m.get("seed")),
    )
    check(
        "test_seed",
        int(cfg.data.get("test_seed", 1)),
        manifest.get("test_seed", test_m.get("seed")),
    )
    check(
        "trajectory_mode",
        TRAJECTORY_MODE,
        manifest.get("trajectory_mode", train_m.get("trajectory_mode")),
    )
    check(
        "history_dependent",
        False,
        bool(manifest.get("history_dependent", train_m.get("history_dependent", True))),
    )
    check("train.json θ count", cfg.theta_sample_size("train"), train_n)
    check("test.json θ count", cfg.theta_sample_size("test"), test_n)

    _validate_one_step_bank_systems(
        cfg, data_path, train_payload=train_payload, test_payload=test_payload,
    )

    if mismatches:
        raise ValueError(
            f"Existing data at {data_path} does not match config {cfg.name!r}. "
            "Delete that folder to regenerate, or use a different config.\n  "
            + "\n  ".join(mismatches)
        )


def ensure_data(
    project_root: Path,
    cfg: SBOEDConfig,
    *,
    splits: tuple[str, ...] = ("train", "test"),
    theta_ranges: dict[str, tuple[int, int | None]] | None = None,
) -> Path:
    d = resolve_data_path(project_root, cfg)
    train_path = d / "train.json"
    test_path = d / "test.json"

    if train_path.is_file() and test_path.is_file():
        print(
            f"Checking existing data at {d} "
            f"(~{train_path.stat().st_size + test_path.stat().st_size:,} bytes JSON; may take several minutes)..."
        )

    if data_bundle_complete(cfg, d):
        manifest = load_manifest(d)
        dur = manifest.get("generation_duration")
        if dur:
            print(f"Using existing data → {d}  (generation: {dur})")
        else:
            print(f"Using existing data → {d}")
        return d

    gen_t0 = time.perf_counter()
    d.mkdir(parents=True, exist_ok=True)
    _reject_legacy_trajectory_mode(cfg)
    n_actions = len(build_catalog(cfg))
    n_seq = n_actions
    n_theta = cfg.theta_sample_size("train") + cfg.theta_sample_size("test")
    print(
        f"Generating data → {d}\n"
        f"  system={cfg.system_label}  topology={cfg.topology}  preset={cfg.config_preset}\n"
        f"  run={cfg.run_slug}  experiment_T={cfg.step_number}  amplitudes={cfg.probe_amplitudes}\n"
        f"  trajectory_mode={TRAJECTORY_MODE}"
    )
    print(
        f"  reset one-step bank: {n_actions:,} actions/θ  "
        f"× {n_theta} θ → {n_seq * n_theta:,} total CUDA trajectories "
        f"(shared across all experiment horizons T)"
    )

    ranges = theta_ranges or {}
    can_resume_payloads = _can_resume_existing_payloads(d)
    if not can_resume_payloads:
        print(
            "  existing data uses an obsolete schema; overwriting train/test JSON "
            "with reset one-step banks"
        )
    train_payload: dict[str, Any] | None = (
        _load_split_payload_if_any(d, "train") if can_resume_payloads else None
    )
    test_payload: dict[str, Any] | None = (
        _load_split_payload_if_any(d, "test") if can_resume_payloads else None
    )

    if "train" in splits:
        t0, t1 = ranges.get("train", (0, None))
        train_payload = generate_split(
            cfg,
            "train",
            int(cfg.data.get("train_seed", 0)),
            data_dir=d,
            theta_start=t0,
            theta_end=t1,
            existing_payload=train_payload,
        )
        save_tables(train_payload, train_path)

    if "test" in splits:
        t0, t1 = ranges.get("test", (0, None))
        test_payload = generate_split(
            cfg,
            "test",
            int(cfg.data.get("test_seed", 1)),
            data_dir=d,
            theta_start=t0,
            theta_end=t1,
            existing_payload=test_payload,
        )
        save_tables(test_payload, test_path)

    if train_payload is None:
        train_payload = _load_split_payload_if_any(d, "train")
    if test_payload is None:
        test_payload = _load_split_payload_if_any(d, "test")

    ref = train_payload or test_payload
    if ref is None:
        raise RuntimeError("no split payload written")

    tm = ref["meta"]
    gen_elapsed = time.perf_counter() - gen_t0
    complete = data_bundle_complete(cfg, d)
    write_data_manifest(
        d, cfg, tm,
        generation_elapsed_sec=gen_elapsed,
        data_complete=complete,
    )
    manifest = load_manifest(d)
    dur = manifest.get("generation_duration", format_generation_duration(gen_elapsed))

    if complete:
        print(f"Data complete → {d}  (generation: {dur})")
    else:
        print(f"Data partial (resume with batched generate-data) → {d}  (generation so far: {dur})")
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

    from src.inference.spce import build_mk_grid

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
    for sys in systems:
        if sys.get("trajectory_mode", TRAJECTORY_MODE) != TRAJECTORY_MODE:
            raise ValueError(
                f"{split}: trajectory_mode={sys.get('trajectory_mode')!r}; "
                f"only {TRAJECTORY_MODE!r} data is valid"
            )
        if sys.get("trajectory_bank"):
            root = sys.get("_data_root")
            if root is None:
                raise ValueError(f"{split}: trajectory bank missing _data_root (reload data)")
            bank = open_trajectory_bank(sys, Path(root))
            assert bank is not None
            if bank.n_sequences <= 0:
                raise ValueError(f"{split}: empty trajectory bank at {bank.root}")
            probe = bank.row_payload(0)
            if "y_sim" not in probe:
                raise ValueError(f"{split}: trajectory bank missing y_sim at {bank.root}")
            continue
        trajs = sys.get("trajectories") or []
        if not trajs:
            raise ValueError(f"{split}: system missing one_step_bank trajectories")
        for traj in trajs:
            if "y_sim" not in traj:
                raise ValueError(
                    f"{split}.json: trajectory rows lack 'y_sim'. "
                    "Regenerate data (CUDA writes sequence, y_sim, y per row)."
                )


_ACTION_ROW_INDEX = "_action_row_index"


def _ensure_action_row_index(
    system: dict[str, Any],
) -> dict[int, dict[str, list[float]]]:
    """O(1) action lookup for reset-based one-step banks."""
    cached = system.get(_ACTION_ROW_INDEX)
    if cached is not None:
        return cached

    trajs = system.get("trajectories") or []
    index: dict[int, dict[str, list[float]]] = {}
    for traj in trajs:
        seq = tuple(int(a) for a in traj["sequence"])
        if len(seq) != 1:
            raise ValueError(
                "reset one-step data requires each trajectory row to have sequence length 1; "
                "delete legacy full-bank data and regenerate"
            )
        action = seq[0]
        y_sim = _trajectory_y_sim(traj)
        y = list(traj["y"])
        if len(y_sim) != 1 or len(y) != 1:
            raise ValueError("reset one-step data requires scalar y_sim/y per action row")
        index[action] = {
            "sequence": [action],
            "y_sim": y_sim[:1],
            "y": y[:1],
        }
    system[_ACTION_ROW_INDEX] = index
    return index


def _lookup_action_row(system: dict[str, Any], action: int) -> dict[str, list[float]]:
    key = int(action)
    if system.get("trajectory_bank"):
        root = system.get("_data_root")
        if root is None:
            raise ValueError("trajectory bank system missing _data_root; reload via load_split_systems")
        bank = open_trajectory_bank(system, Path(root))
        assert bank is not None
        row = bank.find_exact_row((key,))
        if row is not None:
            return bank.row_payload(row, length=1)
        raise KeyError(f"No action {key} in bank {bank.root}")
    trajs = system.get("trajectories") or []
    if not trajs and not system.get("trajectory_bank"):
        raise KeyError(
            f"system has no one_step_bank trajectories (invalid or incomplete data)"
        )
    index = _ensure_action_row_index(system)
    if key in index:
        return index[key]
    raise KeyError(f"No action {key} in trajectory table")


def lookup_action_y_sim(system: dict[str, Any], action: int) -> float:
    return float(_trajectory_y_sim(_lookup_action_row(system, int(action)))[0])


def lookup_action_y(system: dict[str, Any], action: int) -> float:
    return float(_lookup_action_row(system, int(action))["y"][0])


def lookup_sequence_y_sim(system: dict[str, Any], sequence: list[int]) -> list[float]:
    return [lookup_action_y_sim(system, int(a)) for a in sequence]


def lookup_prefix_y_sim(system: dict[str, Any], prefix: list[int]) -> list[float]:
    return lookup_sequence_y_sim(system, prefix)


def lookup_sequence_y(system: dict[str, Any], sequence: list[int]) -> list[float]:
    """Reset-based sequence observation assembled from per-action table rows."""
    return [lookup_action_y(system, int(a)) for a in sequence]


def lookup_prefix_y(system: dict[str, Any], prefix: list[int]) -> list[float]:
    """
    Observations for the first len(prefix) probes.

    Reset-based protocol: y_t depends only on (θ, ξ_t). Sequential decisions
    affect the posterior/history, not the swing-equation forward response.
    """
    return lookup_sequence_y(system, prefix)


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
    return [
        sim.simulate(M, K, design, add_noise=cfg.sigma_y, rng=rng)
        for design in designs
    ]


def sample_trajectory_rollout(
    systems: list[dict[str, Any]],
    rng: np.random.Generator,
    *,
    curriculum_weights: list[np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[int], list[float]]:
    """Uniform system + trajectory, or curriculum weights per system (sum to 1)."""
    i_sys = int(rng.integers(len(systems)))
    sys = systems[i_sys]
    if sys.get("trajectory_bank"):
        root = sys.get("_data_root")
        if root is None:
            raise ValueError("trajectory bank system missing _data_root")
        bank = open_trajectory_bank(sys, Path(root))
        assert bank is not None
        n_traj = bank.n_sequences
        if curriculum_weights is not None:
            w = np.asarray(curriculum_weights[i_sys], dtype=np.float64)
            w = w / w.sum()
            idx = int(rng.choice(n_traj, p=w))
        else:
            idx = int(rng.integers(n_traj))
        traj = bank.row_payload(idx)
    else:
        trajs = sys["trajectories"]
        if curriculum_weights is not None:
            w = np.asarray(curriculum_weights[i_sys], dtype=np.float64)
            w = w / w.sum()
            traj = trajs[int(rng.choice(len(trajs), p=w))]
        else:
            traj = trajs[int(rng.integers(len(trajs)))]
    return sys, list(traj["sequence"]), list(traj["y"])
