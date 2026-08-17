"""Generate / load physical observation banks (full Δf + max |ROCOF|; CUDA only).

Layout under ``data/<system>/``::

    meta/catalog.json
    meta/bank.yaml          # slim provenance (no time_vector)
    train/{delta_f,max_rocof,theta_M,theta_K,psi_star}.npy
    test/{...}.npy

``psi_star.npy`` = Yoon model-specific optimal operators ψ_θ* (min safe support).
Legacy ``U.npy`` is auto-migrated on load.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.config import SYSTEM_CONFIGS, SBOEDConfig, load_config_for_run, repo_root
from src.banks.paths import DATA_ROOT
from src.banks.quality import validate_physical_bank_quality
from src.domains.swing.design import build_catalog, build_simulator
from src.inference.spce import sample_mk_prior
from src.observations.compress import obs_indices_for_n_obs


# Canonical Yoon ψ_θ* bank; legacy filename U.npy is migrated on load.
PSI_STAR_NAME = "psi_star.npy"
LEGACY_U_NAME = "U.npy"
SPLIT_ARRAYS_CORE = ("delta_f.npy", "max_rocof.npy", "theta_M.npy", "theta_K.npy")
SPLIT_ARRAYS = (*SPLIT_ARRAYS_CORE, PSI_STAR_NAME)
META_FILES = ("catalog.json", "bank.yaml")

# Relative paths that must exist for a complete neat bank.
NEAT_BANK_RELPATHS = (
    "meta/catalog.json",
    "meta/bank.yaml",
    *(f"train/{n}" for n in SPLIT_ARRAYS_CORE),
    *(f"test/{n}" for n in SPLIT_ARRAYS_CORE),
    f"train/{PSI_STAR_NAME}",
    f"test/{PSI_STAR_NAME}",
)

# Flat-layout legacy names (pre-reorg).
LEGACY_FLAT_MAP: dict[str, str] = {
    "design_catalog.json": "meta/catalog.json",
    "full_delta_f_bank_train.npy": "train/delta_f.npy",
    "max_abs_rocof_train.npy": "train/max_rocof.npy",
    "theta_M_train.npy": "train/theta_M.npy",
    "theta_K_train.npy": "train/theta_K.npy",
    "U_bank_train.npy": f"train/{PSI_STAR_NAME}",
    "full_delta_f_bank_test.npy": "test/delta_f.npy",
    "max_abs_rocof_test.npy": "test/max_rocof.npy",
    "theta_M_test.npy": "test/theta_M.npy",
    "theta_K_test.npy": "test/theta_K.npy",
    "U_bank_test.npy": f"test/{PSI_STAR_NAME}",
}


def data_slug(system: str) -> str:
    """Canonical shared data folder name for a system (no experiment-type suffix)."""
    return str(system)


def data_path(system: str, project_root: Path | None = None) -> Path:
    root = project_root or repo_root()
    return root / DATA_ROOT / data_slug(system)


def resolve_dataset_dir(cfg: SBOEDConfig, project_root: Path | None = None) -> Path:
    """Shared system data dir (same path for objective_based and eig_based)."""
    from src.banks.paths import resolve_shared_data_dir

    return resolve_shared_data_dir(project_root, cfg)


def system_name_from_cfg(cfg: SBOEDConfig) -> str:
    sys_sec = cfg.raw.get("system") or {}
    if isinstance(sys_sec, dict) and sys_sec.get("name"):
        return str(sys_sec["name"])
    return str(cfg.topology)


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _swing_bounds(cfg) -> tuple[float, float, float, float]:
    sw = cfg.swing
    return (
        float(sw.get("M_lower", 0.01)),
        float(sw.get("M_upper", 0.06)),
        float(sw.get("K_lower", 0.05)),
        float(sw.get("K_upper", 0.50)),
    )


def _sample_power_grid_prior(
    cfg: SBOEDConfig,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample a machine-wise, legacy buswise, or aggregate M/K prior.

    ``machinewise_mk`` samples a distinct inertia and fast-response value for
    every retained dynamic machine.  Per-machine bounds are required so the
    six IEEE-9 latent coordinates retain their physical bus identities.

    ``aggregate_mk`` keeps the nominal inertia shape fixed and makes only its
    global scale and the aggregate fast-response magnitude uncertain.  The
    simulator still receives one M/K value per retained dynamic bus.
    """
    sw = cfg.swing
    model = str(sw.get("latent_model", "buswise_mk")).lower()
    if model == "machinewise_mk":
        m_lo = np.asarray(sw.get("M_lower_nodes"), dtype=np.float64)
        m_hi = np.asarray(sw.get("M_upper_nodes"), dtype=np.float64)
        k_lo = np.asarray(sw.get("K_lower_nodes"), dtype=np.float64)
        k_hi = np.asarray(sw.get("K_upper_nodes"), dtype=np.float64)
        for name, values in (
            ("M_lower_nodes", m_lo),
            ("M_upper_nodes", m_hi),
            ("K_lower_nodes", k_lo),
            ("K_upper_nodes", k_hi),
        ):
            if values.shape != (cfg.N,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain {cfg.N} finite values")
        if np.any(m_lo <= 0.0) or np.any(m_hi <= m_lo):
            raise ValueError("machine-wise inertia bounds require 0 < lower < upper")
        if np.any(k_lo < 0.0) or np.any(k_hi <= k_lo):
            raise ValueError("machine-wise K bounds require 0 <= lower < upper")

        M = rng.uniform(m_lo, m_hi, size=(n, cfg.N))
        K = rng.uniform(k_lo, k_hi, size=(n, cfg.N))
        dynamic_buses = list(sw.get("dynamic_machine_buses", range(1, cfg.N + 1)))
        if len(dynamic_buses) != cfg.N:
            raise ValueError(
                f"dynamic_machine_buses must contain {cfg.N} physical bus numbers"
            )
        return M, K, {
            "latent_model": "machinewise_mk",
            "latent_dimension": 2 * cfg.N,
            "latent_names": [
                *(f"M_{bus}" for bus in dynamic_buses),
                *(f"K_{bus}" for bus in dynamic_buses),
            ],
            "M_lower_nodes": m_lo.tolist(),
            "M_upper_nodes": m_hi.tolist(),
            "K_lower_nodes": k_lo.tolist(),
            "K_upper_nodes": k_hi.tolist(),
        }
    if model != "aggregate_mk":
        M_lo, M_hi, K_lo, K_hi = _swing_bounds(cfg)
        M, K = sample_mk_prior(M_lo, M_hi, K_lo, K_hi, n, rng, n_buses=cfg.N)
        return M, K, {"latent_model": "buswise_mk", "latent_dimension": 2 * cfg.N}

    nominal_M = np.asarray(sw.get("M_nominal_nodes"), dtype=np.float64)
    participation = np.asarray(sw.get("K_participation_nodes"), dtype=np.float64)
    if nominal_M.shape != (cfg.N,) or np.any(nominal_M <= 0.0):
        raise ValueError(f"M_nominal_nodes must contain {cfg.N} positive values")
    if participation.shape != (cfg.N,) or np.any(participation < 0.0):
        raise ValueError(f"K_participation_nodes must contain {cfg.N} nonnegative values")
    part_sum = float(participation.sum())
    if part_sum <= 0.0:
        raise ValueError("K_participation_nodes must have positive sum")
    participation = participation / part_sum

    m_lo = float(sw["M_scale_lower"])
    m_hi = float(sw["M_scale_upper"])
    k_lo = float(sw["K_total_lower"])
    k_hi = float(sw["K_total_upper"])
    if not (0.0 < m_lo < m_hi and 0.0 <= k_lo < k_hi):
        raise ValueError("invalid aggregate M/K prior bounds")
    m_scale = rng.uniform(m_lo, m_hi, size=n)
    k_total = rng.uniform(k_lo, k_hi, size=n)
    M = m_scale[:, None] * nominal_M[None, :]
    K = k_total[:, None] * participation[None, :]
    return M, K, {
        "latent_model": "aggregate_mk",
        "latent_dimension": 2,
        "latent_names": ["M_scale", "K_total"],
        "M_scale_bounds": [m_lo, m_hi],
        "K_total_bounds": [k_lo, k_hi],
        "M_nominal_nodes": nominal_M.tolist(),
        "K_participation_nodes": participation.tolist(),
    }


def bank_is_neat(path: Path) -> bool:
    path = Path(path)
    if not path.is_dir() or not all((path / rel).is_file() for rel in NEAT_BANK_RELPATHS):
        return False
    return _has_psi_star_split(path / "train") and _has_psi_star_split(path / "test")


def _has_psi_star_split(split_dir: Path) -> bool:
    return (split_dir / PSI_STAR_NAME).is_file() or (split_dir / LEGACY_U_NAME).is_file()


def migrate_legacy_u_to_psi_star(path: Path) -> None:
    """Rename split ``U.npy`` → ``psi_star.npy`` when needed."""
    path = Path(path)
    for split in ("train", "test"):
        legacy = path / split / LEGACY_U_NAME
        canon = path / split / PSI_STAR_NAME
        if legacy.is_file() and not canon.is_file():
            legacy.rename(canon)
            print(f"[bank] migrated {legacy.name} → {canon}")


def load_psi_star(split_dir: Path) -> np.ndarray:
    """Load Yoon ψ_θ* vector from ``psi_star.npy`` (or legacy ``U.npy``)."""
    split_dir = Path(split_dir)
    canon = split_dir / PSI_STAR_NAME
    legacy = split_dir / LEGACY_U_NAME
    if canon.is_file():
        return np.load(canon)
    if legacy.is_file():
        return np.load(legacy)
    raise FileNotFoundError(
        f"Missing operator-cost bank in {split_dir}: need {PSI_STAR_NAME} "
        f"(legacy {LEGACY_U_NAME} also accepted)"
    )


def bank_is_legacy_flat(path: Path) -> bool:
    """True if the old flat layout has the core arrays (max_rocof optional)."""
    path = Path(path)
    required = [
        "design_catalog.json",
        "full_delta_f_bank_train.npy",
        "full_delta_f_bank_test.npy",
        "theta_M_train.npy",
        "theta_K_train.npy",
        "theta_M_test.npy",
        "theta_K_test.npy",
        "U_bank_train.npy",
        "U_bank_test.npy",
    ]
    return path.is_dir() and all((path / n).is_file() for n in required)


def bank_is_complete(path: Path) -> bool:
    path = Path(path)
    if bank_is_neat(path):
        return True
    return bank_is_legacy_flat(path)


def bank_has_max_rocof(path: Path) -> bool:
    path = Path(path)
    if bank_is_neat(path):
        return (path / "train" / "max_rocof.npy").is_file() and (
            path / "test" / "max_rocof.npy"
        ).is_file()
    return (path / "max_abs_rocof_train.npy").is_file() and (
        path / "max_abs_rocof_test.npy"
    ).is_file()


def _slim_bank_meta(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky / redundant fields (especially time_vector)."""
    skip = {"time_vector", "legacy_max_rocof_untouched"}
    out = {k: v for k, v in raw.items() if k not in skip}
    out.pop("manifest", None)
    return out


def write_bank_yaml(path: Path, meta: dict[str, Any]) -> None:
    meta_dir = Path(path) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / "bank.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(_slim_bank_meta(meta), f, default_flow_style=False, sort_keys=False)


def migrate_flat_bank_to_neat(path: Path) -> bool:
    """
    Move legacy flat files into meta/train/test. Returns True if migration ran.

    Safe to call repeatedly: no-op when already neat.
    """
    path = Path(path)
    if bank_is_neat(path):
        return False
    if not bank_is_legacy_flat(path):
        return False

    print(f"[data] migrating flat bank → neat layout under {path}")
    (path / "meta").mkdir(parents=True, exist_ok=True)
    (path / "train").mkdir(parents=True, exist_ok=True)
    (path / "test").mkdir(parents=True, exist_ok=True)

    for src_name, dst_rel in LEGACY_FLAT_MAP.items():
        src = path / src_name
        dst = path / dst_rel
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file():
            src.unlink()
            continue
        shutil.move(str(src), str(dst))

    # Build slim bank.yaml from old metadata.yaml if present.
    old_meta_path = path / "metadata.yaml"
    meta: dict[str, Any] = {}
    if old_meta_path.is_file():
        with old_meta_path.open(encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
    old_manifest = path / "manifest.yaml"
    if old_manifest.is_file() and not meta:
        with old_manifest.open(encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}

    # Fill shapes from arrays when missing.
    df_train = path / "train" / "delta_f.npy"
    df_test = path / "test" / "delta_f.npy"
    if df_train.is_file():
        shape_tr = list(np.load(df_train, mmap_mode="r").shape)
        meta.setdefault("bank_shape_train", shape_tr)
        meta.setdefault("train_theta_count", shape_tr[0])
        meta.setdefault("n_actions", shape_tr[1])
        meta.setdefault("N_sim", shape_tr[2])
    if df_test.is_file():
        shape_te = list(np.load(df_test, mmap_mode="r").shape)
        meta.setdefault("bank_shape_test", shape_te)
        meta.setdefault("test_theta_count", shape_te[0])
    meta.setdefault("dataset_name", path.name)
    meta.setdefault("system", path.name)
    meta.setdefault("layout", "meta_train_test_v1")
    meta.setdefault("physical_observation", "full_delta_f_plus_max_abs_rocof")
    write_bank_yaml(path, meta)

    for stale in ("metadata.yaml", "manifest.yaml"):
        fp = path / stale
        if fp.is_file():
            fp.unlink()

    # Drop any other leftover flat names.
    for src_name in list(LEGACY_FLAT_MAP):
        fp = path / src_name
        if fp.is_file():
            fp.unlink()

    if not bank_is_neat(path):
        missing = [rel for rel in NEAT_BANK_RELPATHS if not (path / rel).is_file()]
        raise RuntimeError(f"Migration incomplete under {path}; missing {missing}")
    print(f"[data] migration complete → {path}")
    return True


def sanitize_dataset_dir(path: Path) -> list[str]:
    """Remove non-bank artifacts; keep only neat tree (+ ignore empty dirs)."""
    removed: list[str] = []
    path = Path(path)
    if not path.is_dir():
        return removed

    migrate_flat_bank_to_neat(path)
    migrate_legacy_u_to_psi_star(path)

    allowed_files = {Path(rel) for rel in NEAT_BANK_RELPATHS}
    # Keep legacy U.npy only until migrated; do not delete mid-migration.
    allowed_files |= {
        Path(f"train/{LEGACY_U_NAME}"),
        Path(f"test/{LEGACY_U_NAME}"),
    }
    allowed_top_dirs = {"diagnostics"}
    for fp in sorted(path.rglob("*")):
        if not fp.is_file():
            continue
        rel = fp.relative_to(path)
        if rel in allowed_files:
            continue
        if rel.parts and rel.parts[0] in allowed_top_dirs:
            continue
        fp.unlink()
        removed.append(str(rel))
    if removed:
        print(f"[data] removed non-bank files under {path}: {', '.join(removed)}")
    return removed


def clear_physical_bank(path: Path) -> list[str]:
    """Remove neat (and leftover flat) bank artifacts."""
    removed: list[str] = []
    path = Path(path)
    if not path.is_dir():
        return removed
    for rel in NEAT_BANK_RELPATHS:
        fp = path / rel
        if fp.is_file():
            fp.unlink()
            removed.append(rel)
    for name in LEGACY_FLAT_MAP:
        fp = path / name
        if fp.is_file():
            fp.unlink()
            removed.append(name)
    for name in ("metadata.yaml", "manifest.yaml"):
        fp = path / name
        if fp.is_file():
            fp.unlink()
            removed.append(name)
    for sub in ("meta", "train", "test"):
        d = path / sub
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    removed.extend(sanitize_dataset_dir(path))
    return removed


def generate_if_missing_flag(cfg: SBOEDConfig) -> bool:
    data_sec = dict(cfg.raw.get("data") or {})
    return bool(data_sec.get("generate_if_missing", False))


def generate_full_delta_f_bank(
    system: str,
    *,
    project_root: Path | None = None,
    smoke: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Backward-compatible entry using study system configs."""
    if system not in SYSTEM_CONFIGS:
        raise ValueError(f"unsupported system {system}")
    root = project_root or repo_root()
    cfg = load_config_for_run(SYSTEM_CONFIGS[system], root, step_number=3)
    return generate_physical_bank(
        cfg, project_root=root, smoke=smoke, force=force
    )


def generate_physical_bank(
    cfg: SBOEDConfig,
    *,
    project_root: Path | None = None,
    smoke: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Generate full Δf + max |ROCOF| + U-bank from a config (CUDA only)."""
    root = project_root or repo_root()
    if str(cfg.data.get("backend", "")).lower() != "cuda":
        raise RuntimeError("physical banks require data_generation.backend: cuda")

    path = resolve_dataset_dir(cfg, root)
    if force and path.is_dir():
        removed = clear_physical_bank(path)
        print(
            f"[generate-data] --force: cleared {len(removed)} bank files under {path}"
        )

    # Migrate legacy flat banks in place (no CUDA).
    migrate_flat_bank_to_neat(path)

    if bank_is_complete(path):
        _ensure_max_rocof(path, cfg, batch=int(cfg.data.get("cuda_batch_size", 128)))
        print(
            f"Using existing physical bank → {path}\n"
            f"  (skipping CUDA generation; pass --force to regenerate)"
        )
        sanitize_dataset_dir(path)
        quality = validate_physical_bank_quality(path, cfg, smoke=smoke)
        return {
            "data_dir": str(path),
            "reused": True,
            "smoke_requested": bool(smoke),
            "bank_quality": quality,
        }

    from src.banks.control_u import generate_control_bank_for_split
    from src.control.cuda_control import CudaControlEngine
    from src.control.u_req import ControlSpec
    from src.domains.swing.cuda import CudaTrajectoryEngine

    path.mkdir(parents=True, exist_ok=True)
    (path / "meta").mkdir(parents=True, exist_ok=True)
    (path / "train").mkdir(parents=True, exist_ok=True)
    (path / "test").mkdir(parents=True, exist_ok=True)

    system = system_name_from_cfg(cfg)
    catalog = build_catalog(cfg)
    n_actions = len(catalog)
    sim = build_simulator(cfg)
    engine = CudaTrajectoryEngine(sim, catalog)
    continuous = bool(getattr(cfg, "continuous_duration_mode", False))
    duration_only = (
        cfg.probe_durations is not None
        and len(cfg.probe_amplitudes) == 1
        and cfg.probe_buses is not None
        and len(cfg.probe_buses) == 1
    )
    # Store the full clean trajectory for every design, including the
    # duration-only setting.  This keeps N_obs an observation-layer choice:
    # N_obs=0 can reuse max_rocof.npy, while N_obs>0 samples delta_f.npy.
    n_sim = engine.n_sim_steps()
    n_train = 8 if smoke else int(cfg.theta_sample_size("train"))
    n_test = 4 if smoke else int(cfg.theta_sample_size("test"))
    train_seed = int(cfg.data.get("train_seed", 101))
    test_seed = int(cfg.data.get("test_seed", 202))
    batch = int(cfg.data.get("cuda_batch_size", 128))
    _, _, latent_meta = _sample_power_grid_prior(
        cfg, 1, np.random.default_rng(0)
    )
    t_gen0 = time.time()
    print(
        f"[{system}] generating physical banks: "
        f"train={n_train} test={n_test} actions={n_actions} N_sim={n_sim}"
    )

    def _gen_split(split: str, n: int, seed: int) -> dict[str, Any]:
        split_dir = path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        M, K, latent_meta = _sample_power_grid_prior(cfg, n, rng)
        bank = np.lib.format.open_memmap(
            split_dir / "delta_f.npy",
            mode="w+",
            dtype=np.float64,
            shape=(n, n_actions, n_sim),
        )
        rocof_bank = np.lib.format.open_memmap(
            split_dir / "max_rocof.npy",
            mode="w+",
            dtype=np.float64,
            shape=(n, n_actions),
        )
        print(
            f"[{system}] generating {split}: {n} θ × {n_actions} × N_sim={n_sim} "
            f"(full Δf + max |ROCOF|)"
        )
        for i in range(n):
            M_rows = np.tile(M[i][None, :], (n_actions, 1))
            K_rows = np.tile(K[i][None, :], (n_actions, 1))
            actions = np.arange(n_actions, dtype=np.int32)
            df = engine.simulate_delta_f_batch(
                M_rows, K_rows, actions, batch_size=batch
            )
            bank[i] = df
            rocof_bank[i] = engine.simulate_one_step_f_batch(
                M_rows, K_rows, actions, batch_size=batch
            )
            if (i + 1) % max(1, n // 10) == 0 or i + 1 == n:
                print(f"  {split} θ {i + 1}/{n}")
        bank.flush()
        rocof_bank.flush()
        np.save(split_dir / "theta_M.npy", M)
        np.save(split_dir / "theta_K.npy", K)

        systems = [{"M": M[i].tolist(), "K": K[i].tolist()} for i in range(n)]
        spec = ControlSpec.from_cfg(cfg)
        ctrl_sim = build_simulator(cfg)
        ctrl_sim.T_obs_sec = float(spec.T_obs_sec)
        ctrl_sim.ode_dt = float(spec.ode_dt)
        ctrl_sim.fs_hz = float(spec.fs_hz)
        ctrl_engine = CudaControlEngine(ctrl_sim, spec)
        print(f"[{system}] generating {split} ψ★-bank (psi_star) n={n}")
        generate_control_bank_for_split(
            systems, ctrl_engine, spec, batch_size=batch, progress=True
        )
        psi_star = np.asarray([float(s["u_req"]) for s in systems], dtype=np.float64)
        np.save(split_dir / PSI_STAR_NAME, psi_star)
        # Remove legacy name if regenerating.
        legacy = split_dir / LEGACY_U_NAME
        if legacy.is_file():
            legacy.unlink()
        return {"n": n, "seed": seed, "psi_star_mean": float(psi_star.mean())}

    train_rep = _gen_split("train", n_train, train_seed)
    test_rep = _gen_split("test", n_test, test_seed)

    designs = [list(d.as_tuple()) for d in catalog]
    durations = [float(d[2]) for d in designs]
    catalog_doc: dict[str, Any] = {
        "designs": designs,
        "ordering": (
            "duration_grid_fixed_amp_bus"
            if duration_only
            else "outer_amplitude_inner_bus"
        ),
        "n_actions": n_actions,
        "amplitudes": list(cfg.probe_amplitudes),
        "buses": (
            list(cfg.probe_buses)
            if cfg.probe_buses is not None
            else list(range(cfg.N))
        ),
        "duration_s": float(cfg.probe_duration),
    }
    if duration_only:
        catalog_doc["durations_s"] = durations
    (path / "meta" / "catalog.json").write_text(
        json.dumps(catalog_doc, indent=2),
        encoding="utf-8",
    )
    meta = {
        "dataset_name": path.name,
        "system": system,
        "layout": "meta_train_test_v1",
        "physical_observation": (
            "full_delta_f_plus_max_abs_rocof"
        ),
        "delta_f_definition": (
            f"fixed physical PMU bus {int(cfg.swing.get('observation_bus', 1))} "
            "omega/(2*pi) [Hz deviation] at every ODE step; physical probe "
            "injections mapped through Kron reduction"
            if cfg.swing.get("physical_bus_count") is not None
            else "probe_bus omega/(2*pi) [Hz deviation] at every ODE step"
        ),
        "max_rocof_definition": "max |ROCOF| from fs-downsampled probe simulation",
        "nominal_frequency_hz": 60.0,
        "n_buses": int(cfg.N),
        "physical_bus_count": int(cfg.swing.get("physical_bus_count", cfg.N)),
        "dynamic_machine_buses": list(cfg.swing.get("dynamic_machine_buses", [])),
        "observation_bus_physical": int(cfg.swing.get("observation_bus", 1)),
        "physical_probe_mapping": (
            "lossless_linear_kron_to_dynamic_buses"
            if cfg.swing.get("physical_bus_count") is not None
            else "direct"
        ),
        **latent_meta,
        "n_actions": n_actions,
        "N_sim": n_sim,
        "continuous_duration_mode": continuous,
        "design_mode": "duration_only_reset" if duration_only and not continuous else (
            "duration_only_carry_state" if duration_only else "catalog"
        ),
        "ode_dt": float(sim.ode_dt),
        "T_obs_sec": float(sim.T_obs_sec),
        "fs_hz": float(sim.fs_hz),
        "train_theta_count": n_train,
        "test_theta_count": n_test,
        "train_seed": train_seed,
        "test_seed": test_seed,
        "probe_amplitudes": list(cfg.probe_amplitudes),
        "probe_duration": float(cfg.probe_duration),
        "sigma_y": float(cfg.sigma_y),
        "observation_shape": [n_train, n_actions, n_sim],
        "simulator": "offline PyCUDA through C++/CUDA",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "config_path": str(cfg.config_path.resolve()),
        "train_report": train_rep,
        "test_report": test_rep,
        "bank_shape_train": [n_train, n_actions, n_sim],
        "bank_shape_test": [n_test, n_actions, n_sim],
        "elapsed_seconds": time.time() - t_gen0,
        "data_complete": True,
    }
    write_bank_yaml(path, meta)
    print(
        f"Physical bank ready → {path} "
        f"(train={n_train}×{n_actions}×{n_sim}, test={n_test}×{n_actions}×{n_sim}, "
        f"{meta['elapsed_seconds']:.1f}s, reused=false)"
    )
    sanitize_dataset_dir(path)
    quality = validate_physical_bank_quality(path, cfg, smoke=smoke)
    return {
        "data_dir": str(path.resolve()),
        "reused": False,
        "N_sim": n_sim,
        "bank_quality": quality,
        **meta,
    }


def _ensure_max_rocof(path: Path, cfg: SBOEDConfig, *, batch: int) -> None:
    path = Path(path)
    migrate_flat_bank_to_neat(path)
    if bank_has_max_rocof(path):
        return
    print(f"Backfilling max_rocof into {path} (one-time CUDA pass)")
    from src.domains.swing.cuda import CudaTrajectoryEngine

    catalog = build_catalog(cfg)
    sim = build_simulator(cfg)
    engine = CudaTrajectoryEngine(sim, catalog)
    n_actions = len(catalog)
    for split in ("train", "test"):
        M = np.load(path / split / "theta_M.npy")
        K = np.load(path / split / "theta_K.npy")
        n = M.shape[0]
        out = np.zeros((n, n_actions), dtype=np.float64)
        for i in range(n):
            M_rows = np.tile(M[i][None, :], (n_actions, 1))
            K_rows = np.tile(K[i][None, :], (n_actions, 1))
            actions = np.arange(n_actions, dtype=np.int32)
            out[i] = engine.simulate_one_step_f_batch(
                M_rows, K_rows, actions, batch_size=batch
            )
        np.save(path / split / "max_rocof.npy", out)


def load_bank(system: str, project_root: Path | None = None) -> dict[str, Any]:
    return load_bank_from_path(data_path(system, project_root), project_root=project_root)


def load_bank_from_path(
    path: Path,
    *,
    project_root: Path | None = None,
    cfg: SBOEDConfig | None = None,
    smoke: bool = False,
    skip_quality_check: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    migrate_flat_bank_to_neat(path)
    migrate_legacy_u_to_psi_star(path)
    if not bank_is_complete(path):
        raise FileNotFoundError(
            f"Physical bank missing or incomplete at {path}.\n"
            f"Generate with: ./scripts/data_generation.sh --config <config.yaml>"
        )
    if cfg is not None and not bank_has_max_rocof(path):
        raise FileNotFoundError(
            f"Physical databank at {path} is missing train/test max_rocof.npy. "
            "Experiment loading is databank-only and will not backfill it by "
            "running the simulator. Complete the bank offline first."
        )
    with (path / "meta" / "bank.yaml").open(encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    if cfg is not None:
        expected_model = str(cfg.swing.get("latent_model", "buswise_mk")).lower()
        bank_model = str(meta.get("latent_model", "buswise_mk")).lower()
        expected_dim = 2 if expected_model == "aggregate_mk" else 2 * int(cfg.N)
        bank_dim = int(meta.get("latent_dimension", expected_dim))
        if bank_model != expected_model or bank_dim != expected_dim:
            raise ValueError(
                f"Physical databank latent model is incompatible with {cfg.config_path}: "
                f"bank has {bank_model!r} dimension {bank_dim}, config requires "
                f"{expected_model!r} dimension {expected_dim}. Regenerate the offline "
                "databank before training or evaluation."
            )
    catalog = json.loads((path / "meta" / "catalog.json").read_text(encoding="utf-8"))
    if cfg is not None:
        expected_catalog = [list(d.as_tuple()) for d in build_catalog(cfg)]
        bank_catalog = list(catalog.get("designs") or [])
        if bank_catalog != expected_catalog:
            raise ValueError(
                f"Physical databank action catalog is incompatible with {cfg.config_path}: "
                f"bank has {len(bank_catalog)} actions but its amplitude/bus/duration "
                f"tuples do not match the config's {len(expected_catalog)} actions. "
                "Regenerate the offline databank before training or evaluation."
            )
    psi_train = load_psi_star(path / "train")
    psi_test = load_psi_star(path / "test")
    out: dict[str, Any] = {
        "path": path,
        "meta": meta,
        "catalog": catalog,
        "full_train": np.load(path / "train" / "delta_f.npy", mmap_mode="r"),
        "full_test": np.load(path / "test" / "delta_f.npy", mmap_mode="r"),
        "M_train": np.load(path / "train" / "theta_M.npy"),
        "K_train": np.load(path / "train" / "theta_K.npy"),
        "M_test": np.load(path / "test" / "theta_M.npy"),
        "K_test": np.load(path / "test" / "theta_K.npy"),
        # Yoon ψ_θ* bank (canonical keys).
        "psi_star_train": psi_train,
        "psi_star_test": psi_test,
        # Backward-compatible aliases (same arrays).
        "U_train": psi_train,
        "U_test": psi_test,
        "max_rocof_train": None,
        "max_rocof_test": None,
    }
    if bank_has_max_rocof(path):
        out["max_rocof_train"] = np.load(path / "train" / "max_rocof.npy", mmap_mode="r")
        out["max_rocof_test"] = np.load(path / "test" / "max_rocof.npy", mmap_mode="r")
    sanitize_dataset_dir(path)
    if not skip_quality_check:
        out["bank_quality"] = validate_physical_bank_quality(
            path, cfg, smoke=smoke, write_report=False
        )
    return out


def compressed_centres(
    full_bank: np.ndarray,
    obs_indices: np.ndarray,
) -> np.ndarray:
    from src.observations.likelihood import compress_delta_f

    obs = compress_delta_f(np.asarray(full_bank), obs_indices)
    return np.transpose(obs, (1, 0, 2))


def obs_indices_for_config(n_sim: int, n_obs: int) -> np.ndarray:
    return obs_indices_for_n_obs(n_sim, n_obs)
