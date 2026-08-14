"""Generate / load SIR ODE banks matching iDAD SIR settings.

Primary artifact: full I(t) table on ``linspace(0, 100, 10000)`` for each
sampled (β, γ), as in ``epidemic_simulate_data.py``. Designs look up the
infected **count** I(τ) from that table (iDAD observation; not I/N).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import SBOEDConfig, repo_root
from src.banks.power_grid import resolve_dataset_dir, system_name_from_cfg
from src.domains.sir.simulator import (
    integrate_sir_ode,
    sample_lognormal_theta,
    trajectory_is_good,
)

SIR_BANK_FILES = (
    "meta/meta.json",
    "meta/t.npy",
    "train/I.npy",
    "train/theta_beta.npy",
    "train/theta_gamma.npy",
    "test/I.npy",
    "test/theta_beta.npy",
    "test/theta_gamma.npy",
)


def sir_bank_is_complete(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() and all((path / rel).is_file() for rel in SIR_BANK_FILES)


def _sir_section(cfg: SBOEDConfig) -> dict[str, Any]:
    if cfg.raw.get("sir_sde"):
        raise ValueError(
            "Config key 'sir_sde' is removed. Use 'sir_ode' (deterministic ODE + "
            "explicit Gaussian likelihood). Implicit SIR SDE is out of scope."
        )
    return dict(cfg.raw.get("sir_ode") or cfg.raw.get("sir") or {})


def build_design_times_on_grid(
    t_grid: np.ndarray,
    *,
    n_actions: int,
    t_min: float | None = None,
    t_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced design times that land on the iDAD ODE grid.

    Returns ``(times, grid_indices)``. Excludes ``t=0`` by default (iDAD does
    not use the initial time as a measurement).
    """
    t_grid = np.asarray(t_grid, dtype=np.float64).reshape(-1)
    if t_grid.size < 3:
        raise ValueError("t_grid too short for designs")
    lo = float(t_grid[1] if t_min is None else t_min)
    hi = float(t_grid[-1] if t_max is None else t_max)
    mask = (t_grid >= lo - 1e-12) & (t_grid <= hi + 1e-12) & (t_grid > 0.0)
    candidates = np.flatnonzero(mask)
    if candidates.size < int(n_actions):
        raise ValueError(
            f"Need >= {n_actions} grid points in ({lo}, {hi}], found {candidates.size}"
        )
    pick = np.linspace(0, candidates.size - 1, int(n_actions))
    idx = candidates[np.round(pick).astype(int)]
    # Ensure unique increasing indices
    idx = np.unique(idx)
    if idx.size < int(n_actions):
        # fill gaps if rounding collided
        need = int(n_actions) - idx.size
        unused = [i for i in candidates.tolist() if i not in set(idx.tolist())]
        idx = np.sort(np.concatenate([idx, np.asarray(unused[:need], dtype=int)]))
    idx = idx[: int(n_actions)]
    return t_grid[idx].astype(np.float64), idx.astype(np.int64)


def _simulate_split(
    n_target: int,
    *,
    population: float,
    i0: float,
    t_end: float,
    n_grid: int,
    beta_mean: float,
    gamma_mean: float,
    log_std: float,
    min_mean_infected: float,
    seed: int,
    max_tries_factor: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw θ until ``n_target`` paths pass the iDAD mean-I filter.

    Returns ``(beta, gamma, t, I)`` with ``I`` shape ``(n_target, n_grid)``.
    """
    betas: list[float] = []
    gammas: list[float] = []
    trajectories: list[np.ndarray] = []
    t_grid: np.ndarray | None = None
    rng = np.random.default_rng(int(seed))
    tries = 0
    max_tries = int(n_target) * int(max_tries_factor)
    while len(betas) < int(n_target) and tries < max_tries:
        tries += 1
        b, g = sample_lognormal_theta(
            1,
            beta_mean=beta_mean,
            gamma_mean=gamma_mean,
            log_std=log_std,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        t, _S, I = integrate_sir_ode(
            float(b[0]),
            float(g[0]),
            population=population,
            i0=i0,
            t_end=t_end,
            n_grid=n_grid,
        )
        if not trajectory_is_good(I, min_mean_infected=min_mean_infected):
            continue
        if t_grid is None:
            t_grid = np.asarray(t, dtype=np.float64)
        betas.append(float(b[0]))
        gammas.append(float(g[0]))
        trajectories.append(np.asarray(I, dtype=np.float64))
    if t_grid is None or len(betas) < int(n_target):
        raise RuntimeError(
            f"SIR bank: only {len(betas)}/{n_target} good paths after {tries} tries "
            f"(min_mean_infected={min_mean_infected})"
        )
    return (
        np.asarray(betas, dtype=np.float64),
        np.asarray(gammas, dtype=np.float64),
        t_grid,
        np.stack(trajectories, axis=0).astype(np.float64),
    )


def centres_from_trajectories(
    t: np.ndarray,
    I: np.ndarray,
    measurement_times: np.ndarray,
    *,
    grid_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Look up infected **count** I(τ) at design times from full I(t) tables.

    Returns centres shape ``(n_theta, n_actions, 1)``.
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    I = np.asarray(I, dtype=np.float64)
    times = np.asarray(measurement_times, dtype=np.float64).reshape(-1)
    if I.ndim != 2:
        raise ValueError(f"I must be (n_theta, n_t), got {I.shape}")
    if grid_indices is not None:
        idx = np.asarray(grid_indices, dtype=int).reshape(-1)
        vals = I[:, idx]
    else:
        vals = np.vstack([np.interp(times, t, I[i]) for i in range(I.shape[0])])
    return vals.astype(np.float64)[..., None]


def generate_sir_bank(
    cfg: SBOEDConfig,
    *,
    project_root: Path | None = None,
    smoke: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Write ``data/sir_ode/`` with iDAD-matched full I(t) tables."""
    root = project_root or repo_root()
    data_dir = resolve_dataset_dir(cfg, root)
    if sir_bank_is_complete(data_dir) and not force:
        meta = json.loads((data_dir / "meta" / "meta.json").read_text())
        return {
            "reused": True,
            "path": str(data_dir),
            "system": system_name_from_cfg(cfg),
            "n_actions": int(meta.get("n_actions", 0)),
            "obs_dim": int(meta.get("obs_dim", 1)),
            "train_theta_count": int(meta.get("train_theta_count", 0)),
            "test_theta_count": int(meta.get("test_theta_count", 0)),
            "elapsed_seconds": 0.0,
            "reference": meta.get("reference"),
        }

    started = time.perf_counter()
    sec = _sir_section(cfg)
    dg = dict(cfg.raw.get("data_generation") or {})
    n_train = int(dg.get("theta_sample_size_train", 10000))
    n_test = int(dg.get("theta_sample_size_test", 3000))
    if smoke:
        n_train = min(n_train, 32)
        n_test = min(n_test, 12)

    population = float(sec.get("population", 500.0))
    i0 = float(sec.get("i0", 2.0))
    t_end = float(sec.get("t_end", 100.0))
    n_grid = int(sec.get("n_grid", 10000))  # iDAD GRID
    beta_mean = float(sec.get("beta_mean", 0.5))
    gamma_mean = float(sec.get("gamma_mean", 0.1))
    log_std = float(sec.get("log_std", 0.5))
    min_mean_infected = float(sec.get("min_mean_infected", 1.0))
    n_actions = int(sec.get("n_actions", 100 if not smoke else 20))
    t_min = sec.get("t_min", None)
    t_max = sec.get("t_max", None)
    train_seed = int(dg.get("train_seed", 101))
    test_seed = int(dg.get("test_seed", 202))

    print(
        f"[sir-bank] iDAD-matched SIR ODE: N={population} I0={i0} "
        f"n_grid={n_grid} t_end={t_end} (observe I count)"
    )
    beta_tr, gamma_tr, t_grid, I_tr = _simulate_split(
        n_train,
        population=population,
        i0=i0,
        t_end=t_end,
        n_grid=n_grid,
        beta_mean=beta_mean,
        gamma_mean=gamma_mean,
        log_std=log_std,
        min_mean_infected=min_mean_infected,
        seed=train_seed,
    )
    beta_te, gamma_te, t_grid_te, I_te = _simulate_split(
        n_test,
        population=population,
        i0=i0,
        t_end=t_end,
        n_grid=n_grid,
        beta_mean=beta_mean,
        gamma_mean=gamma_mean,
        log_std=log_std,
        min_mean_infected=min_mean_infected,
        seed=test_seed,
    )
    if not np.allclose(t_grid, t_grid_te):
        raise RuntimeError("train/test ODE time grids differ")

    design_times, design_idx = build_design_times_on_grid(
        t_grid,
        n_actions=n_actions,
        t_min=None if t_min is None else float(t_min),
        t_max=None if t_max is None else float(t_max),
    )

    for sub in ("meta", "train", "test"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)

    np.save(data_dir / "meta" / "t.npy", t_grid)
    np.save(data_dir / "meta" / "design_indices.npy", design_idx)
    np.save(data_dir / "train" / "I.npy", I_tr)
    np.save(data_dir / "train" / "theta_beta.npy", beta_tr)
    np.save(data_dir / "train" / "theta_gamma.npy", gamma_tr)
    np.save(data_dir / "test" / "I.npy", I_te)
    np.save(data_dir / "test" / "theta_beta.npy", beta_te)
    np.save(data_dir / "test" / "theta_gamma.npy", gamma_te)
    np.save(data_dir / "train" / "U.npy", np.zeros(n_train, dtype=np.float64))
    np.save(data_dir / "test" / "U.npy", np.zeros(n_test, dtype=np.float64))

    sigma_y = float(
        (cfg.raw.get("observation") or {}).get(
            "noise_sigma", sec.get("likelihood_sigma", 1.0)
        )
    )
    dt = float(t_grid[1] - t_grid[0])
    meta = {
        "system": system_name_from_cfg(cfg),
        "domain": "sir_ode",
        "model": "deterministic_sir_ode_rk4",
        "likelihood": "gaussian_explicit",
        "bank_content": "full_I_vs_time_per_theta",
        "reference": (
            "iDAD (Ivanova et al. NeurIPS 2021) App. D.6 / "
            "epidemic_simulate_data.py settings: N=500, I0=2, "
            "t=linspace(0,100,10000), lognormal prior, observe I(τ) count, "
            "mean-I≥1 filter. Dynamics: ODE drift only (no SDE diffusion). "
            "Explicit y~N(I(τ),σ²) for Foster et al. DAD / sPCE."
        ),
        "n_actions": int(design_times.shape[0]),
        "obs_dim": 1,
        "n_grid": int(n_grid),
        "n_time": int(t_grid.shape[0]),
        "dt": dt,
        "measurement_times": design_times.tolist(),
        "design_indices": design_idx.tolist(),
        "population": population,
        "i0": i0,
        "s0": population - i0,
        "t_end": t_end,
        "prior": {
            "family": "lognormal_independent",
            "beta_mean": beta_mean,
            "gamma_mean": gamma_mean,
            "log_std": log_std,
            "note": "log θ ~ N([log 0.5, log 0.1], 0.5² I) as in iDAD github",
        },
        "min_mean_infected": min_mean_infected,
        "observation": "infected_count_I_tau",
        "design": "measurement_time_chronological_on_idad_grid",
        "stored_arrays": {
            "meta/t.npy": "(n_grid,) linspace(0, t_end, n_grid)",
            "train/I.npy": "(n_theta, n_grid) infected count vs time",
            "train/theta_beta.npy": "(n_theta,)",
            "train/theta_gamma.npy": "(n_theta,)",
        },
        "extra_observation_noise": (
            f"Gaussian N(0, σ_y²) on infected count with σ_y={sigma_y} "
            "(iDAD itself adds no Poisson layer; σ_y enables explicit DAD/sPCE)"
        ),
        "train_theta_count": n_train,
        "test_theta_count": n_test,
        "train_seed": train_seed,
        "test_seed": test_seed,
        "sigma_y": sigma_y,
        "config_path": str(cfg.config_path),
        "chronological_designs": True,
    }
    (data_dir / "meta" / "meta.json").write_text(json.dumps(meta, indent=2))
    elapsed = float(time.perf_counter() - started)
    print(
        f"[sir-bank] wrote {data_dir} train={n_train} test={n_test} "
        f"n_t={t_grid.shape[0]} dt={dt:.6g} I={list(I_tr.shape)} "
        f"A={design_times.shape[0]} ({elapsed:.1f}s)"
    )
    return {
        "reused": False,
        "path": str(data_dir),
        "system": system_name_from_cfg(cfg),
        "n_actions": int(design_times.shape[0]),
        "obs_dim": 1,
        "train_theta_count": n_train,
        "test_theta_count": n_test,
        "bank_shape_train": list(I_tr.shape),
        "bank_shape_test": list(I_te.shape),
        "n_time": int(t_grid.shape[0]),
        "elapsed_seconds": elapsed,
        "N_sim": 1,
        "reference": meta["reference"],
    }


def load_sir_bank(path: Path) -> dict[str, Any]:
    """Load full I(t) tables and derive design-time centres (infected counts)."""
    path = Path(path)
    if not sir_bank_is_complete(path):
        raise FileNotFoundError(f"Incomplete SIR bank at {path}")
    meta = json.loads((path / "meta" / "meta.json").read_text())
    t = np.load(path / "meta" / "t.npy")
    I_train = np.load(path / "train" / "I.npy")
    I_test = np.load(path / "test" / "I.npy")
    beta_train = np.load(path / "train" / "theta_beta.npy")
    gamma_train = np.load(path / "train" / "theta_gamma.npy")
    beta_test = np.load(path / "test" / "theta_beta.npy")
    gamma_test = np.load(path / "test" / "theta_gamma.npy")
    measurement_times = np.asarray(meta["measurement_times"], dtype=np.float64)
    design_idx = None
    idx_path = path / "meta" / "design_indices.npy"
    if idx_path.is_file():
        design_idx = np.load(idx_path)
    centres_train = centres_from_trajectories(
        t, I_train, measurement_times, grid_indices=design_idx
    )
    centres_test = centres_from_trajectories(
        t, I_test, measurement_times, grid_indices=design_idx
    )
    return {
        "path": path,
        "meta": meta,
        "t": t,
        "I_train": I_train,
        "I_test": I_test,
        "beta_train": beta_train,
        "gamma_train": gamma_train,
        "beta_test": beta_test,
        "gamma_test": gamma_test,
        "centres_train": centres_train,
        "centres_test": centres_test,
    }
