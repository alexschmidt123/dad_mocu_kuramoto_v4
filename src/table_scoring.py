"""
Train-table support and Foster sPCE-EIG (banked ``y_sim`` centres, noisy ``y`` data).

- ``y``: noisy observation — policy rollouts and fixed term in sPCE.
- ``y_sim``: ODE output before noise — likelihood centre only (sPCE / myopic / ΔH).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config import SBOEDConfig
from src.contrastive.spce import (
    log_likelihood_scalar,
    log_prior_uniform_discrete,
    spce_step_scalar_likelihoods,
    spce_total_from_f_tensor,
)
from src.data import lookup_prefix_y, lookup_prefix_y_sim, lookup_sequence_y, lookup_sequence_y_sim


@dataclass
class TableThetaSupport:
    """Subsample of train latent θ entries (discrete prior support at eval)."""

    systems: list[dict[str, Any]]
    log_p0: np.ndarray

    def __len__(self) -> int:
        return len(self.systems)

    @classmethod
    def from_train(
        cls,
        train_systems: list[dict[str, Any]],
        cfg: SBOEDConfig,
        rng: np.random.Generator,
    ) -> TableThetaSupport:
        n = min(int(cfg.prior.get("mc_samples", 128)), len(train_systems))
        idx = rng.choice(len(train_systems), size=n, replace=False)
        picked = [train_systems[int(i)] for i in idx]
        return cls(systems=picked, log_p0=log_prior_uniform_discrete(n))


def y_sequence_from_table(system: dict[str, Any], sequence: list[int]) -> np.ndarray:
    seq = [int(a) for a in sequence]
    try:
        return np.asarray(lookup_sequence_y(system, seq), dtype=np.float64)
    except KeyError:
        return np.asarray(
            [lookup_prefix_y(system, seq[: t + 1])[-1] for t in range(len(seq))],
            dtype=np.float64,
        )


def y_sim_sequence_from_table(system: dict[str, Any], sequence: list[int]) -> np.ndarray:
    seq = [int(a) for a in sequence]
    try:
        return np.asarray(lookup_sequence_y_sim(system, seq), dtype=np.float64)
    except KeyError:
        return np.asarray(
            [lookup_prefix_y_sim(system, seq[: t + 1])[-1] for t in range(len(seq))],
            dtype=np.float64,
        )


def y_sim_steps_from_tables(
    support: TableThetaSupport,
    sequence: list[int],
) -> np.ndarray:
    """Shape ``(T, n_support)`` — banked ``y_sim`` (likelihood centres)."""
    seq = [int(a) for a in sequence]
    T = len(seq)
    out = np.zeros((T, len(support.systems)), dtype=np.float64)
    for i, sys in enumerate(support.systems):
        ys = y_sim_sequence_from_table(sys, seq)
        if ys.shape[0] != T:
            raise ValueError(f"y_sim length mismatch for support index {i}")
        out[:, i] = ys
    return out


def y_sim_last_step_from_tables(
    support: TableThetaSupport,
    sequence: list[int],
) -> np.ndarray:
    return y_sim_steps_from_tables(support, sequence)[-1, :].copy()


def y_steps_from_tables(
    support: TableThetaSupport,
    sequence: list[int],
) -> np.ndarray:
    """Shape ``(T, n_support)`` — banked noisy ``y`` (optional approximate posterior)."""
    seq = [int(a) for a in sequence]
    T = len(seq)
    out = np.zeros((T, len(support.systems)), dtype=np.float64)
    for i, sys in enumerate(support.systems):
        y = y_sequence_from_table(sys, seq)
        if y.shape[0] != T:
            raise ValueError(f"y length mismatch for support index {i}")
        out[:, i] = y
    return out


def likelihood_centre_tensor_for_spce(
    sequence: list[int],
    theta0_system: dict[str, Any],
    support: TableThetaSupport,
    rng: np.random.Generator,
    L: int,
    *,
    contrastive_systems: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """
    Shape ``(L+1, T)`` of banked ``y_sim`` for Foster sPCE (θ₀ row + contrastive train θ).
    """
    seq = [int(a) for a in sequence]
    rows = [y_sim_sequence_from_table(theta0_system, seq)]

    pool = list(contrastive_systems if contrastive_systems is not None else support.systems)
    others = [s for s in pool if s is not theta0_system]
    if not others:
        others = pool
    if not others:
        raise ValueError("need at least one contrastive latent θ for sPCE")
    n_pick = min(L, len(others))
    pick = rng.choice(len(others), size=n_pick, replace=False)
    for i in pick:
        rows.append(y_sim_sequence_from_table(others[int(i)], seq))
    return np.stack(rows, axis=0)


def spce_eig_from_rollout(
    cfg: SBOEDConfig,
    sequence: list[int],
    y_obs: list[float] | np.ndarray,
    theta0_system: dict[str, Any],
    support: TableThetaSupport,
    rng: np.random.Generator,
    L: int | None = None,
) -> tuple[list[float], float, float]:
    """
    Foster sPCE-EIG: fixed noisy ``y_obs``; centres from banked ``y_sim`` (train contrastives only at train).

    Returns ``(step_eigs, mean_step_eig, total_eig)``.
    """
    if L is None:
        L = int(cfg.spce.get("L", 4))
    y_seq = np.asarray(y_obs, dtype=np.float64)
    seq = [int(a) for a in sequence]
    centre = likelihood_centre_tensor_for_spce(
        seq, theta0_system, support, rng, L,
        contrastive_systems=support.systems,
    )
    step_eigs: list[float] = []
    for t in range(len(y_seq)):
        log_L0 = log_likelihood_scalar(float(y_seq[t]), float(centre[0, t]), cfg.sigma_y)
        log_c = [
            log_likelihood_scalar(float(y_seq[t]), float(centre[ell + 1, t]), cfg.sigma_y)
            for ell in range(centre.shape[0] - 1)
        ]
        step_eigs.append(spce_step_scalar_likelihoods(log_L0, np.asarray(log_c)))
    total = float(spce_total_from_f_tensor(y_seq, centre, cfg.sigma_y, positive_idx=0))
    mean_step = float(np.mean(step_eigs)) if step_eigs else 0.0
    return step_eigs, mean_step, total


def spce_eig_train_row(
    cfg: SBOEDConfig,
    sequence: list[int],
    y_obs: list[float] | np.ndarray,
    theta0_system: dict[str, Any],
    support: TableThetaSupport,
    rng: np.random.Generator,
    L: int | None = None,
) -> tuple[list[float], float, float]:
    """sPCE-EIG for one train row (θ₀ = row latent θ; uses ``y_sim`` centres only in the loss)."""
    return spce_eig_from_rollout(
        cfg, sequence, y_obs, theta0_system, support, rng, L=L,
    )
