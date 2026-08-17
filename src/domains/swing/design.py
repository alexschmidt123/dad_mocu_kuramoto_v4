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
    *,
    buses: Iterable[int] | None = None,
    durations: Iterable[float] | None = None,
) -> list[Design]:
    """One-step designs: amplitudes × durations × buses.

    Ordering: outer duration, middle amplitude, inner bus.  Multiple durations
    create non-scale waveform diversity (unlike multi-amp, which is often pure
    ROCOF scaling and does not trap Myopic).
    """
    if buses is None:
        bus_list = list(range(int(n_buses)))
    else:
        bus_list = [int(b) for b in buses]
        if not bus_list:
            raise ValueError("probe bus list is empty")
        for b in bus_list:
            if b < 0 or b >= int(n_buses):
                raise ValueError(f"probe bus {b} out of range for N={n_buses}")
    if durations is None:
        dur_list = [float(duration)]
    else:
        dur_list = [float(d) for d in durations]
        if not dur_list:
            raise ValueError("probe duration list is empty")
        if any(d <= 0.0 for d in dur_list):
            raise ValueError(f"probe durations must be positive, got {dur_list}")
    catalog: list[Design] = []
    for dur in dur_list:
        for amp in amplitudes:
            for bus in bus_list:
                catalog.append(
                    Design(amplitude=float(amp), bus=int(bus), duration=float(dur))
                )
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


def unrank_no_repeat_sequence(n: int, step_number: int, rank: int) -> tuple[int, ...]:
    """``rank``-th length-``step_number`` sequence in lex order (0-based)."""
    if step_number > n:
        raise ValueError("step_number cannot exceed n_actions")
    if step_number < 0:
        raise ValueError("step_number must be non-negative")
    total = count_no_repeat_sequences(n, step_number)
    if rank < 0 or rank >= total:
        raise IndexError(f"rank {rank} out of range for P({n},{step_number})={total}")
    if step_number == 0:
        return tuple()
    available = list(range(n))
    seq: list[int] = []
    r = int(rank)
    for i in range(step_number):
        tail = step_number - i - 1
        denom = count_no_repeat_sequences(len(available) - 1, tail) if tail > 0 else 1
        j = r // denom
        r %= denom
        seq.append(available.pop(j))
    return tuple(seq)


def unrank_sequence_chunk(
    n: int,
    step_number: int,
    start: int,
    count: int,
) -> list[tuple[int, ...]]:
    return [unrank_no_repeat_sequence(n, step_number, start + i) for i in range(count)]


def enumerate_no_repeat_sequences(catalog: list[Design], step_number: int) -> list[tuple[int, ...]]:
    """All ordered no-repeat action sequences of length ``step_number`` (small T only)."""
    n = len(catalog)
    if step_number > n:
        return []
    if step_number == 0:
        return [tuple()]
    total = count_no_repeat_sequences(n, step_number)
    if total > 500_000:
        raise MemoryError(
            f"P({n},{step_number})={total} sequences is too large to list in RAM. "
            "Use reset one-step observation banks for current experiments."
        )
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
    durations = getattr(cfg, "probe_durations", None)
    design_bus_count = int(cfg.swing.get("physical_bus_count", cfg.N))
    return build_design_catalog(
        design_bus_count,
        cfg.probe_amplitudes,
        cfg.probe_duration,
        buses=cfg.probe_buses,
        durations=durations,
    )


def build_simulator(cfg: SBOEDConfig):
    from src.domains.swing.simulator import SwingSimulator

    return SwingSimulator(
        cfg.swing,
        fs_hz=cfg.fs_hz,
        T_obs_sec=cfg.T_obs_sec,
        ode_dt=float(cfg.swing.get("ode_dt", 1.0 / 160.0)),
    )
