"""
One-step design space for sBOED (documents/sBOED_design.tex).

Each design xi = (a, b, d): probe amplitude, bus (0-indexed), duration.
step_number and amplitudes come from the experiment config, not hard-coded here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

import numpy as np

from src.config import SBOEDConfig


def hann_window(t: float, T: float) -> float:
    """
    Hann window function: s(t; T) = 0.5 * (1 - cos(2πt/T))
    
    Args:
        t: Time (scalar)
        T: Duration (scalar)
    
    Returns:
        Window value (0 if t > T, otherwise Hann window)
    """
    if t > T:
        return 0.0
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * t / T))


@dataclass(frozen=True)
class Design:
    """One probing operation xi = (amplitude, bus, duration)."""

    amplitude: float
    bus: int
    duration: float

    def as_tuple(self) -> tuple[float, int, float]:
        return (self.amplitude, self.bus, self.duration)

    def index_in(self, catalog: list[Design]) -> int:
        for i, d in enumerate(catalog):
            if d == self:
                return i
        raise ValueError(f"Design {self} not in catalog")


def build_design_catalog(
    n_buses: int,
    amplitudes: Iterable[float],
    duration: float,
) -> list[Design]:
    """All one-step designs: len(amplitudes) * n_buses actions."""
    catalog: list[Design] = []
    for amp in amplitudes:
        for bus in range(n_buses):
            catalog.append(Design(amplitude=float(amp), bus=int(bus), duration=float(duration)))
    return catalog


def count_no_repeat_sequences(n_actions: int, step_number: int) -> int:
    """Number of ordered sequences of distinct action indices (length ``step_number``)."""
    if step_number < 0:
        raise ValueError("step_number must be non-negative")
    if step_number == 0:
        return 1
    if step_number > n_actions:
        return 0
    return math.perm(n_actions, step_number)


def masked_action_indices(used_actions: set[int], catalog: list[Design]) -> np.ndarray:
    """Feasible action indices excluding already-used action indices."""
    return np.array(
        [i for i in range(len(catalog)) if i not in used_actions],
        dtype=np.int64,
    )


def enumerate_no_repeat_sequences(catalog: list[Design], step_number: int) -> list[tuple[int, ...]]:
    """All ordered no-repeat action sequences of length ``step_number``."""
    n = len(catalog)
    if step_number > n:
        return []
    if step_number == 0:
        return [tuple()]
    return list(permutations(range(n), step_number))


def random_valid_sequence(
    catalog: list[Design],
    step_number: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample a valid no-repeat sequence of action indices."""
    n = len(catalog)
    return list(rng.choice(n, size=step_number, replace=False))


def build_catalog(cfg: SBOEDConfig) -> list[Design]:
    return build_design_catalog(cfg.N, cfg.probe_amplitudes, cfg.probe_duration)


def build_simulator(cfg: SBOEDConfig):
    from src.swing_equation_ode.simulator import SwingSimulator

    return SwingSimulator(cfg.swing, fs_hz=cfg.fs_hz, T_obs_sec=cfg.T_obs_sec)
