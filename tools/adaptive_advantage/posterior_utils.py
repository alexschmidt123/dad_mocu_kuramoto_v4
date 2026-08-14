"""Production posterior + terminal-control helpers (read-only reuse)."""

from __future__ import annotations

import numpy as np

from src.control.posterior_ctrl import (
    normalize_log_weights,
    posterior_safe_u_ctrl,
    weighted_quantile,
)
from src.observations.likelihood import vector_gaussian_loglik


def uniform_log_prior(n: int) -> np.ndarray:
    return np.full(n, -np.log(float(n)), dtype=np.float64)


def update_log_weights(
    log_w: np.ndarray,
    y_obs: np.ndarray | float,
    centres: np.ndarray,
    sigma_y: float,
) -> np.ndarray:
    """Bayesian particle update using production Gaussian likelihood.

    centres: (n_particles, obs_dim) for one design.
    """
    y = np.asarray(y_obs, dtype=np.float64).reshape(-1)
    c = np.asarray(centres, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, None]
    if y.size == 1 and c.shape[1] != 1:
        raise ValueError("y_obs dim mismatch with centres")
    return np.asarray(log_w, dtype=np.float64) + vector_gaussian_loglik(y, c, sigma_y)


def weights_from_log(log_w: np.ndarray) -> np.ndarray:
    return normalize_log_weights(np.asarray(log_w, dtype=np.float64))


def terminal_u(
    U: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
) -> float:
    """Production posterior-safe snapped control."""
    return float(
        posterior_safe_u_ctrl(
            np.asarray(U, dtype=np.float64),
            np.asarray(weights, dtype=np.float64),
            float(alpha),
            margin=float(margin),
            u_grid=np.asarray(u_grid, dtype=np.float64),
            snap_up=True,
        )
    )


def raw_quantile(
    U: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    margin: float,
) -> float:
    q = 1.0 - float(alpha)
    return float(weighted_quantile(U, weights, q) + float(margin))


def observe_from_bank(
    Y: np.ndarray,
    theta_index: int,
    design_id: int,
    sigma_y: float,
    rng: np.random.Generator,
) -> float:
    """y_obs = Y[theta, design] + N(0, sigma_y^2). No simulator."""
    clean = float(Y[theta_index, design_id])
    return clean + float(rng.normal(0.0, sigma_y))
