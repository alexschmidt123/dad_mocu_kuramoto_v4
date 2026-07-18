"""Build nested / permuted posterior supports from master banks (offline arrays only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config_for_run, repo_root
from src.control.particle_posterior_adequacy import (
    MASTER_N,
    NESTED_SUPPORT_SIZES,
    SYSTEM_CONFIGS,
)
from src.control.terminal_rule import load_frozen_terminal_rule
from src.contrastive.spce import log_prior_uniform_discrete
from src.data import data_dir, get_systems, load_tables
from src.run_context import load_experiment_run
from src.swing_equation_ode.design import build_catalog


SUPPORT_SEEDS = (101, 202, 303, 404, 505)
WEIGHT_THRESHOLD = 1e-6
GLOBAL_HISTORY_SEED = 44117  # match objective_adaptive_value keyed noise


@dataclass(frozen=True)
class MasterArrays:
    system: str
    data_path: Path
    Y_sim: np.ndarray  # (N_master, n_actions)
    Y: np.ndarray
    U: np.ndarray  # (N_master,)
    M: np.ndarray
    K: np.ndarray
    n_actions: int
    n_buses: int
    amplitudes: list[float]
    buses: list[int]
    probe_duration: float
    sigma_y: float
    alpha: float
    margin: float
    u_grid: np.ndarray
    catalog: list[Any]
    latent_dim: int
    train_theta_count_production: int
    test_theta_count_production: int


def master_data_path(system: str, project_root: Path | None = None) -> Path:
    root = project_root or repo_root()
    cfg = load_config_for_run(SYSTEM_CONFIGS[system], root, step_number=3)
    path = data_dir(root, cfg)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Master bank missing at {path}. Run generate-master first."
        )
    return path


def load_master_arrays(system: str, project_root: Path | None = None) -> MasterArrays:
    root = project_root or repo_root()
    cfg = load_config_for_run(SYSTEM_CONFIGS[system], root, step_number=3)
    data_path = master_data_path(system, root)
    Y_sim = np.load(data_path / "Y_bank_sim.npy")
    Y = np.load(data_path / "Y_bank.npy")
    U = np.load(data_path / "U_bank.npy").reshape(-1)
    M = np.load(data_path / "theta_M.npy")
    K = np.load(data_path / "theta_K.npy")
    if Y_sim.shape[0] < MASTER_N:
        raise RuntimeError(f"{data_path}: master train rows {Y_sim.shape[0]} < {MASTER_N}")
    catalog = build_catalog(cfg)
    n_actions = len(catalog)
    if Y_sim.shape[1] != n_actions:
        raise RuntimeError(
            f"Y_bank actions {Y_sim.shape[1]} != catalog {n_actions}"
        )
    # Terminal rule from historical production experiment (snap_up=True official).
    prod_exp = root / "experiments" / f"{system}_T3"
    frozen = load_frozen_terminal_rule(prod_exp)
    # Production true-θ counts (diagnostic only; not support size).
    prod_run = load_experiment_run(prod_exp, root)
    train_n = len(prod_run.train_systems)
    test_n = len(prod_run.test_systems)
    amps = [float(a) for a in cfg.probe_amplitudes]
    buses = list(range(int(cfg.N)))
    return MasterArrays(
        system=system,
        data_path=data_path,
        Y_sim=Y_sim[:MASTER_N],
        Y=Y[:MASTER_N],
        U=U[:MASTER_N],
        M=M[:MASTER_N],
        K=K[:MASTER_N],
        n_actions=n_actions,
        n_buses=int(cfg.N),
        amplitudes=amps,
        buses=buses,
        probe_duration=float(cfg.probe_duration),
        sigma_y=float(cfg.sigma_y),
        alpha=float(frozen.alpha),
        margin=float(frozen.margin),
        u_grid=np.asarray(frozen.u_candidates, dtype=np.float64),
        catalog=catalog,
        latent_dim=2 * int(cfg.N),
        train_theta_count_production=train_n,
        test_theta_count_production=test_n,
    )


def nested_indices(n_master: int, n_particles: int, support_seed: int) -> np.ndarray:
    """Permute master once per seed; take nested prefix of length n_particles."""
    if n_particles > n_master:
        raise ValueError(f"n_particles={n_particles} > n_master={n_master}")
    rng = np.random.default_rng(int(support_seed))
    perm = rng.permutation(int(n_master))
    return perm[: int(n_particles)].astype(np.int64)


@dataclass(frozen=True)
class ParticleSupport:
    system: str
    n_particles: int
    support_seed: int
    indices: np.ndarray
    centres: np.ndarray  # (n_actions, n_particles) = Y_sim[idx].T
    U: np.ndarray
    log_p0: np.ndarray
    selection_rule: str


def build_support(
    master: MasterArrays,
    n_particles: int,
    support_seed: int,
) -> ParticleSupport:
    idx = nested_indices(MASTER_N, n_particles, support_seed)
    centres = master.Y_sim[idx].T.copy()  # (A, N)
    U = master.U[idx].copy()
    return ParticleSupport(
        system=master.system,
        n_particles=int(n_particles),
        support_seed=int(support_seed),
        indices=idx,
        centres=centres,
        U=U,
        log_p0=log_prior_uniform_discrete(int(n_particles)),
        selection_rule=f"permute(master,{support_seed})[:{n_particles}]",
    )


def assert_scientific_invariants(master: MasterArrays) -> None:
    if master.latent_dim != 2 * master.n_buses:
        raise AssertionError("latent dim must be 2N")
    if abs(master.probe_duration - 0.2) > 1e-12:
        raise AssertionError(f"duration must be 0.2 s, got {master.probe_duration}")
    if len(master.amplitudes) != 6:
        raise AssertionError(f"expected 6 amplitudes, got {master.amplitudes}")
    if master.n_actions != len(master.amplitudes) * master.n_buses:
        raise AssertionError("n_actions != 6 * n_buses")


def production_true_theta_systems(system: str, project_root: Path | None = None) -> list[dict]:
    """Train+validation true-θ from production experiment (history generation only)."""
    root = project_root or repo_root()
    from src.control.pilot import load_pilot_splits

    exp = root / "experiments" / f"{system}_T3"
    run = load_experiment_run(exp, root)
    splits = load_pilot_splits(exp, run)
    return list(splits["support_systems"]) + list(splits["validation_systems"])
