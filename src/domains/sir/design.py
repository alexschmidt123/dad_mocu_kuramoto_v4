"""Categorical measurement-time catalog for SIR ODE EIG.

iDAD (Ivanova et al., 2021, App. D.6) uses continuous τ ∈ (0, 100) with an
increasing-time transform. We discretize that interval into sorted categorical
actions and enforce chronological feasibility (only later times remain).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SIRDesignCatalog:
    """Discrete observation times; action id ``a`` probes ``times[a]``."""

    times: np.ndarray  # (n_actions,) sorted ascending

    @property
    def n_actions(self) -> int:
        return int(self.times.shape[0])

    def time_of(self, action: int) -> float:
        return float(self.times[int(action)])


def build_measurement_catalog(
    *,
    t_min: float,
    t_max: float,
    n_actions: int,
) -> SIRDesignCatalog:
    if int(n_actions) < 2:
        raise ValueError(f"n_actions must be >= 2, got {n_actions}")
    if float(t_max) <= float(t_min):
        raise ValueError(f"need t_max > t_min, got [{t_min}, {t_max}]")
    times = np.linspace(float(t_min), float(t_max), int(n_actions), dtype=np.float64)
    return SIRDesignCatalog(times=times)


def chronological_feasible(
    n_actions: int,
    used_actions: list[int] | set[int],
    *,
    remaining_steps: int | None = None,
) -> np.ndarray:
    """Return action ids still allowed under strictly increasing time.

    Action ids index a sorted time grid ``τ_0 < τ_1 < …``. After selecting
    action ``a``, only ``a' > a`` remain, so every rollout satisfies
    ``ξ_1 < ξ_2 < … < ξ_T``.

    When ``remaining_steps`` is set (actions still to choose *including* the
    current step), also reserve enough later indices so the horizon cannot
    strand: the largest selectable index is ``n_actions - remaining_steps``.
    """
    used = {int(a) for a in used_actions}
    last = max(used) if used else -1
    n = int(n_actions)
    if remaining_steps is None:
        upper = n
    else:
        rem = int(remaining_steps)
        if rem < 1:
            return np.asarray([], dtype=int)
        # Need rem distinct indices > last; max first pick is n-rem.
        upper = n - rem + 1
    return np.asarray(
        [
            a
            for a in range(n)
            if a not in used and a > last and a < upper
        ],
        dtype=int,
    )
