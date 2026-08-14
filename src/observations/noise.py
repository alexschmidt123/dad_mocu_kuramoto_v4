"""Keyed observation noise (shared by bank lookup and carry-state observers)."""

from __future__ import annotations

import numpy as np


def keyed_noise_vector(
    *,
    global_seed: int,
    theta_id: int,
    rollout_id: int,
    step: int,
    action_id: int,
    n_obs: int,
) -> np.ndarray:
    dim = max(int(n_obs), 1)
    seed = (
        int(global_seed) * 1_000_003
        + int(theta_id) * 97_451
        + int(rollout_id) * 1_039
        + int(step) * 31
        + int(action_id) * 17
        + int(dim) * 991
    ) % (2**31 - 1)
    return np.random.default_rng(seed).normal(size=dim)
