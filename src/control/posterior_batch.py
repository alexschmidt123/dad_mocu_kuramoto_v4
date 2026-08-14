"""Shared offline posterior / terminal-control batch helpers.

Used by diagnostics and by objective_rl_sboed. All methods must ultimately
evaluate terminal control via ``posterior_ctrl``; these helpers only provide
vectorized bank-based Monte Carlo convenience.
"""

from __future__ import annotations

import math

import numpy as np

from src.control.posterior_ctrl import normalize_log_weights
from src.inference.scoring import TableThetaSupport, y_sim_last_step_from_tables


def centres_matrix(table_support: TableThetaSupport, n_actions: int) -> np.ndarray:
    """Return (n_actions, n_particles) banked clean observation centres."""
    return np.stack(
        [
            np.asarray(y_sim_last_step_from_tables(table_support, [a]), dtype=np.float64)
            for a in range(n_actions)
        ],
        axis=0,
    )


def batch_u_ctrl(
    U: np.ndarray,
    log_w: np.ndarray,
    *,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    snap_up: bool = True,
) -> np.ndarray:
    """Vectorized terminal control. ``log_w``: (..., N) → (...).

    ``snap_up=True`` (historical): snap_up(Q + margin).
    ``snap_up=False`` (continuous studies): Q + margin.
    """
    flat = log_w.reshape(-1, log_w.shape[-1])
    m = np.max(flat, axis=1, keepdims=True)
    w = np.exp(flat - m)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-300, None)
    order = np.argsort(U, kind="mergesort")
    U_sorted = U[order]
    w_sorted = w[:, order]
    cdf = np.cumsum(w_sorted, axis=1)
    q = 1.0 - float(alpha)
    idx = np.sum(cdf < q, axis=1)
    idx = np.clip(idx, 0, U.size - 1)
    u0 = U_sorted[idx] + float(margin)
    if not snap_up:
        return u0.reshape(log_w.shape[:-1])
    gi = np.searchsorted(u_grid, u0, side="left")
    gi = np.clip(gi, 0, u_grid.size - 1)
    return u_grid[gi].reshape(log_w.shape[:-1])


def expected_u_after_action(
    action: int,
    log_w: np.ndarray,
    weights: np.ndarray,
    *,
    centres: np.ndarray,
    U: np.ndarray,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    idx: np.ndarray,
    noise: np.ndarray,
    snap_up: bool = True,
) -> float:
    """E_y[u_ctrl | h, a] with shared CRN (idx, noise). Offline banks only."""
    del weights  # CRN uses idx drawn from weights by the caller.
    c = centres[int(action)]
    y = c[idx] + noise
    s2 = float(sigma_y) ** 2
    log_L = (
        -0.5 * math.log(2.0 * math.pi * s2)
        - 0.5 * ((y[:, None] - c[None, :]) ** 2) / s2
    )
    log_w_h = log_w[None, :] + log_L
    return float(
        np.mean(
            batch_u_ctrl(
                U,
                log_w_h,
                alpha=alpha,
                margin=margin,
                u_grid=u_grid,
                snap_up=snap_up,
            )
        )
    )


def update_posterior(
    log_w: np.ndarray,
    y: float,
    centres_a: np.ndarray,
    sigma_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One Gaussian likelihood update; returns (log_w, normalized weights)."""
    s2 = float(sigma_y) ** 2
    log_L = (
        -0.5 * math.log(2.0 * math.pi * s2)
        - 0.5 * ((float(y) - centres_a) ** 2) / s2
    )
    log_w1 = log_w + log_L
    return log_w1, normalize_log_weights(log_w1)


# Backward-compatible private aliases used by older call sites.
_centres_matrix = centres_matrix
_batch_u_ctrl = batch_u_ctrl
