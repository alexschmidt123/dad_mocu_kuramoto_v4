"""Carry-state scalar max-RoCoF observations (no reset between probes).

Used by continuous-duration power-grid experiments. Posterior centres still
come from the independent-duration bank ``R(θ,d)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.domains.swing.design import Design, build_catalog, build_simulator
from src.observations.noise import keyed_noise_vector


class CarryStateRoCoFObserver:
    """Sequential max-|RoCoF| observer with optional state carry."""

    _sim_cache: dict[int, tuple[Any, list[Design]]] = {}

    def __init__(
        self,
        *,
        cfg,
        M: np.ndarray | list[float],
        K: np.ndarray | list[float],
        reset_after_probe: bool = False,
    ) -> None:
        key = id(cfg)
        cached = CarryStateRoCoFObserver._sim_cache.get(key)
        if cached is None:
            cached = (build_simulator(cfg), build_catalog(cfg))
            CarryStateRoCoFObserver._sim_cache[key] = cached
        self.sim, self.catalog = cached
        self.M = np.asarray(M, dtype=np.float64)
        self.K = np.asarray(K, dtype=np.float64)
        self.reset_after_probe = bool(reset_after_probe)
        self.state: np.ndarray | None = None

    def observe_clean(self, action: int) -> float:
        design = self.catalog[int(action)]
        y, self.state = self.sim.simulate_step(
            self.M,
            self.K,
            design,
            None if self.reset_after_probe else self.state,
        )
        return float(y)

    def observe_noisy(
        self,
        action: int,
        *,
        sigma_y: float,
        global_seed: int,
        theta_id: int,
        rollout_id: int,
        step: int,
        n_obs: int = 0,
    ) -> np.ndarray:
        clean = self.observe_clean(action)
        z = keyed_noise_vector(
            global_seed=global_seed,
            theta_id=theta_id,
            rollout_id=rollout_id,
            step=step,
            action_id=int(action),
            n_obs=max(int(n_obs), 1),
        )
        return np.asarray([clean], dtype=np.float64) + float(sigma_y) * z.reshape(-1)[:1]


def make_carry_observer(ctx, system_row: dict[str, Any]) -> CarryStateRoCoFObserver:
    return CarryStateRoCoFObserver(
        cfg=ctx.cfg,
        M=system_row["M"],
        K=system_row["K"],
        reset_after_probe=bool(getattr(ctx, "reset_after_probe", True)),
    )


def use_carry_state_observation(ctx, *, for_training: bool = False) -> bool:
    cont = bool(getattr(ctx, "continuous_duration_mode", False)) or (
        not bool(getattr(ctx, "reset_after_probe", True))
    ) or str(getattr(ctx, "observation_mode", "")).startswith("continuous_duration")
    if not cont:
        return False
    obs = dict(getattr(ctx.cfg, "raw", {}).get("observation") or {})
    src = str(obs.get("true_source") or obs.get("true_observation") or "carry_state").lower()
    if src in {"bank", "lookup", "independent"}:
        return False
    if for_training:
        train = dict(getattr(ctx.cfg, "raw", {}).get("training") or {})
        if bool(train.get("use_bank_observations", True)):
            return False
    return True
