"""Deterministic SIR ODE matched to iDAD SIR settings (explicit likelihood).

Matches Ivanova et al., NeurIPS 2021 (iDAD) Appendix D.6 and
``epidemic_simulate_data.py``, except we integrate the deterministic drift
(no diffusion) so DAD/sPCE has an explicit Gaussian likelihood.

iDAD settings:
    N = 500, I0 = 2, S0 = N − I0
    t = linspace(0, 100, 10000)          # Δτ ≈ 10⁻²
    prior: log(β,γ) ~ N([log 0.5, log 0.1], 0.5² I)
    observe infected count I(τ)           # not I/N; no Poisson layer
    keep paths with mean_t I(t) ≥ 1

Design ξ = measurement time (strictly increasing). Observation model for
this codebase (explicit):

    y | θ, τ  ~  N( I(τ; θ), σ_y² )
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def sample_lognormal_theta(
    n: int,
    *,
    beta_mean: float = 0.5,
    gamma_mean: float = 0.1,
    log_std: float = 0.5,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """iDAD prior: diagonal MVN on log-θ with loc=[log β̄, log γ̄]."""
    rng = np.random.default_rng(int(seed))
    loc = np.log(np.asarray([float(beta_mean), float(gamma_mean)], dtype=np.float64))
    z = rng.normal(size=(int(n), 2)) * float(log_std) + loc
    theta = np.exp(z)
    return theta[:, 0].astype(np.float64), theta[:, 1].astype(np.float64)


def _rhs(s: float, i: float, beta: float, gamma: float, population: float) -> tuple[float, float]:
    n = float(population)
    p_inf = float(beta) * s * i / n
    p_rec = float(gamma) * i
    return -p_inf, p_inf - p_rec


def integrate_sir_ode(
    beta: float,
    gamma: float,
    *,
    population: float = 500.0,
    i0: float = 2.0,
    t_end: float = 100.0,
    n_grid: int = 10000,
    dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RK4-integrate the SIR ODE on the iDAD time grid.

    Default ``n_grid=10000`` matches ``torch.linspace(0, 100, 10000)`` in
    ``epidemic_simulate_data.py``. If ``dt`` is given, ``n_grid`` is ignored
    and the grid is ``arange``-style with that step (legacy).
    """
    if beta < 0.0 or gamma < 0.0:
        raise ValueError(f"β, γ must be ≥ 0, got β={beta}, γ={gamma}")
    n_pop = float(population)
    if n_pop <= 0.0:
        raise ValueError(f"population must be positive, got {n_pop}")

    if dt is not None:
        if float(dt) <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        n_steps = int(np.floor(float(t_end) / float(dt))) + 1
        t = np.linspace(0.0, float(dt) * (n_steps - 1), n_steps, dtype=np.float64)
    else:
        if int(n_grid) < 2:
            raise ValueError(f"n_grid must be >= 2, got {n_grid}")
        t = np.linspace(0.0, float(t_end), int(n_grid), dtype=np.float64)

    n_steps = int(t.shape[0])
    S = np.zeros(n_steps, dtype=np.float64)
    I = np.zeros(n_steps, dtype=np.float64)
    # iDAD github: X(0) = (N − I0, I0); paper "(0,2)" is a typo for X=(S,I).
    S[0] = n_pop - float(i0)
    I[0] = float(i0)
    for k in range(n_steps - 1):
        h = float(t[k + 1] - t[k])
        s = float(np.clip(S[k], 0.0, n_pop))
        i = float(np.clip(I[k], 0.0, n_pop))
        k1s, k1i = _rhs(s, i, beta, gamma, n_pop)
        k2s, k2i = _rhs(s + 0.5 * h * k1s, i + 0.5 * h * k1i, beta, gamma, n_pop)
        k3s, k3i = _rhs(s + 0.5 * h * k2s, i + 0.5 * h * k2i, beta, gamma, n_pop)
        k4s, k4i = _rhs(s + h * k3s, i + h * k3i, beta, gamma, n_pop)
        s_next = float(np.clip(s + (h / 6.0) * (k1s + 2 * k2s + 2 * k3s + k4s), 0.0, n_pop))
        i_next = float(np.clip(i + (h / 6.0) * (k1i + 2 * k2i + 2 * k3i + k4i), 0.0, n_pop))
        if s_next + i_next > n_pop:
            scale = n_pop / (s_next + i_next)
            s_next *= scale
            i_next *= scale
        S[k + 1] = s_next
        I[k + 1] = i_next
    return t, S, I


def infected_count_at_times(
    beta: float,
    gamma: float,
    times: Sequence[float],
    *,
    population: float = 500.0,
    i0: float = 2.0,
    t_end: float = 100.0,
    n_grid: int = 10000,
) -> np.ndarray:
    """Return clean infected count ``I(τ; θ)`` at each design time, shape ``(n_times, 1)``."""
    times_arr = np.asarray(times, dtype=np.float64).reshape(-1)
    if times_arr.size == 0:
        raise ValueError("times must be non-empty")
    t, _S, I = integrate_sir_ode(
        float(beta),
        float(gamma),
        population=float(population),
        i0=float(i0),
        t_end=max(float(t_end), float(times_arr.max())),
        n_grid=int(n_grid),
    )
    vals = np.interp(times_arr, t, I).astype(np.float64)
    return vals.reshape(-1, 1)


def trajectory_is_good(I: np.ndarray, *, min_mean_infected: float = 1.0) -> bool:
    """iDAD filter: keep paths whose mean infected count is at least 1."""
    return float(np.mean(I)) >= float(min_mean_infected)
