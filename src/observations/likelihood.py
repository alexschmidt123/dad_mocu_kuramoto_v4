"""Full Δf bank compression and vector observation likelihood.

Physical banks store ``delta_f_full`` of length ``N_sim`` (ODE steps).
Methods only ever see ``delta_f_obs`` of length ``N_obs`` selected by
deterministic evenly-spaced indices. Unselected points must not enter
posterior updates, policies, or design selection.
"""

from __future__ import annotations

import math

import numpy as np


def evenly_spaced_indices(n_sim: int, n_obs: int) -> np.ndarray:
    """Deterministic unique indices in ``[0, n_sim-1]`` of length ``n_obs``."""
    n_sim = int(n_sim)
    n_obs = int(n_obs)
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    if n_obs > n_sim:
        raise ValueError(f"n_obs={n_obs} > n_sim={n_sim}")
    if n_obs == 1:
        return np.asarray([0], dtype=np.int64)
    raw = np.linspace(0, n_sim - 1, n_obs)
    idx = np.round(raw).astype(np.int64)
    # Enforce uniqueness while preserving order / coverage.
    for i in range(1, n_obs):
        if idx[i] <= idx[i - 1]:
            idx[i] = idx[i - 1] + 1
    if idx[-1] >= n_sim:
        # Shift left from the end.
        idx[-1] = n_sim - 1
        for i in range(n_obs - 2, -1, -1):
            if idx[i] >= idx[i + 1]:
                idx[i] = idx[i + 1] - 1
    if idx[0] < 0 or len(np.unique(idx)) != n_obs:
        # Fallback: floor-linspace then unique repair.
        idx = np.unique(np.linspace(0, n_sim - 1, n_obs, dtype=int))
        if idx.size < n_obs:
            missing = [i for i in range(n_sim) if i not in set(idx.tolist())]
            need = n_obs - idx.size
            idx = np.sort(np.concatenate([idx, np.asarray(missing[:need], dtype=np.int64)]))
    return np.asarray(idx, dtype=np.int64)


def compress_delta_f(delta_f_full: np.ndarray, obs_indices: np.ndarray) -> np.ndarray:
    """Select method-visible points. Never pass ``delta_f_full`` to methods."""
    full = np.asarray(delta_f_full, dtype=np.float64)
    idx = np.asarray(obs_indices, dtype=np.int64)
    if full.ndim == 1:
        return full[idx].copy()
    if full.ndim == 2:
        return full[:, idx].copy()
    if full.ndim == 3:
        return full[:, :, idx].copy()
    raise ValueError(f"unsupported delta_f_full ndim={full.ndim}")


def vector_gaussian_loglik(
    y_obs: np.ndarray,
    centres: np.ndarray,
    sigma_y: float,
) -> np.ndarray:
    """
    Isotropic Gaussian log-density on ``R^{N_obs}``.

    ``y_obs``: (N_obs,)
    ``centres``: (n_particles, N_obs) or (N_obs,)
    Returns log p(y|θ_n) shape ``(n_particles,)``.
    """
    y = np.asarray(y_obs, dtype=np.float64).reshape(-1)
    c = np.asarray(centres, dtype=np.float64)
    if c.ndim == 1:
        c = c.reshape(1, -1)
    if c.shape[-1] != y.size:
        raise ValueError(f"centre dim {c.shape[-1]} != y dim {y.size}")
    s2 = float(sigma_y) ** 2
    if s2 <= 0:
        raise ValueError("sigma_y must be positive")
    d = y.size
    resid = c - y[None, :]
    quad = np.sum(resid * resid, axis=-1) / s2
    return -0.5 * d * math.log(2.0 * math.pi * s2) - 0.5 * quad
