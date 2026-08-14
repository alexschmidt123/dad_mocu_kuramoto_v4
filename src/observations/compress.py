"""Unified observation interface for all SBOED methods.

Physical banks store full probe-bus Δf trajectories (and ideally max |ROCOF|).
Methods only ever see the representation selected by ``observation.N_obs``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.observations.likelihood import compress_delta_f, evenly_spaced_indices


def validate_n_obs(n_obs: int, n_sim: int) -> int:
    """Validate ``N_obs``; returns the normalized non-negative integer."""
    n_obs = int(n_obs)
    n_sim = int(n_sim)
    if n_obs < 0:
        raise ValueError(f"N_obs must be >= 0, got {n_obs}")
    if n_sim <= 0:
        raise ValueError(f"N_sim must be positive, got {n_sim}")
    if n_obs > n_sim:
        raise ValueError(f"N_obs={n_obs} exceeds N_sim={n_sim}")
    return n_obs


def observation_mode(n_obs: int) -> str:
    return "max_rocof" if int(n_obs) == 0 else "sampled_delta_f"


def compression_ratio(n_obs: int, n_sim: int) -> float | None:
    if int(n_obs) == 0:
        return None
    return float(n_obs) / float(n_sim)


def obs_indices_for_n_obs(n_sim: int, n_obs: int) -> np.ndarray:
    """
    Deterministic indices for method-visible Δf samples.

    ``N_obs == 0`` → empty index array (max-ROCOF path; no Δf points exposed).
    """
    n_obs = validate_n_obs(n_obs, n_sim)
    if n_obs == 0:
        return np.asarray([], dtype=np.int64)
    return evenly_spaced_indices(n_sim, n_obs)


def build_observation_clean(
    full_delta_f: np.ndarray | None,
    max_rocof: np.ndarray | float | None,
    n_obs: int,
    *,
    n_sim: int | None = None,
    obs_indices: np.ndarray | None = None,
) -> np.ndarray:
    """
    Method-visible clean observation (no noise).

    ``N_obs == 0`` → shape ``(1,)`` scalar max |ROCOF|.
    ``N_obs > 0`` → shape ``(N_obs,)`` evenly spaced Δf samples.

    Never returns unselected Δf points.
    """
    if n_sim is None:
        if full_delta_f is not None:
            full = np.asarray(full_delta_f, dtype=np.float64)
            n_sim = int(full.shape[-1]) if full.ndim >= 1 else 0
        else:
            n_sim = 0
    n_obs = validate_n_obs(n_obs, max(int(n_sim), 1) if n_obs == 0 else int(n_sim))

    if n_obs == 0:
        if max_rocof is None:
            raise ValueError("N_obs=0 requires max_rocof")
        return np.asarray([float(np.asarray(max_rocof).reshape(-1)[0])], dtype=np.float64)

    if full_delta_f is None:
        raise ValueError("N_obs>0 requires full_delta_f")
    full = np.asarray(full_delta_f, dtype=np.float64)
    if full.ndim != 1:
        raise ValueError(f"expected 1-D full_delta_f for single (θ,ξ), got shape {full.shape}")
    idx = (
        np.asarray(obs_indices, dtype=np.int64)
        if obs_indices is not None
        else obs_indices_for_n_obs(full.size, n_obs)
    )
    if idx.size != n_obs:
        raise ValueError(f"obs_indices length {idx.size} != N_obs={n_obs}")
    return compress_delta_f(full, idx)


def build_centres_bank(
    full_delta_f_bank: np.ndarray | None,
    max_rocof_bank: np.ndarray | None,
    n_obs: int,
    *,
    obs_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Build method-visible clean centres for the support bank.

    Inputs:
      full_delta_f_bank: (n_theta, n_actions, N_sim) or None
      max_rocof_bank: (n_theta, n_actions) or None

    Returns:
      centres: (n_actions, n_theta, obs_dim)
      indices: (N_obs,) or empty
      mode: observation_mode string
    """
    if n_obs == 0:
        if max_rocof_bank is None:
            raise ValueError("N_obs=0 requires max_rocof_bank")
        rocof = np.asarray(max_rocof_bank, dtype=np.float64)
        if rocof.ndim != 2:
            raise ValueError(f"max_rocof_bank must be 2-D, got {rocof.shape}")
        # (n_actions, n_theta, 1)
        centres = np.transpose(rocof, (1, 0))[:, :, None]
        return centres, np.asarray([], dtype=np.int64), observation_mode(0)

    if full_delta_f_bank is None:
        raise ValueError("N_obs>0 requires full_delta_f_bank")
    full = np.asarray(full_delta_f_bank, dtype=np.float64)
    if full.ndim != 3:
        raise ValueError(f"full_delta_f_bank must be 3-D, got {full.shape}")
    n_sim = int(full.shape[-1])
    n_obs = validate_n_obs(n_obs, n_sim)
    idx = (
        np.asarray(obs_indices, dtype=np.int64)
        if obs_indices is not None
        else obs_indices_for_n_obs(n_sim, n_obs)
    )
    obs = compress_delta_f(full, idx)  # (n_theta, n_actions, N_obs)
    centres = np.transpose(obs, (1, 0, 2))
    return centres, idx, observation_mode(n_obs)


def observation_report_fields(
    *,
    n_obs: int,
    n_sim: int,
    obs_indices: np.ndarray,
) -> dict[str, Any]:
    mode = observation_mode(n_obs)
    ratio = compression_ratio(n_obs, n_sim)
    return {
        "N_obs": int(n_obs),
        "N_sim": int(n_sim),
        "observation_mode": mode,
        "compression_ratio": ratio if ratio is not None else "n/a_scalar_max_rocof",
        "obs_indices": np.asarray(obs_indices, dtype=np.int64).tolist(),
        "sampling": "uniform" if n_obs > 0 else "n/a",
    }
