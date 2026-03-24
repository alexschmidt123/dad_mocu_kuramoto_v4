"""
**MOCU from a weighted particle posterior** (NumPy only).

You already have support points ``θ_i`` and **normalized weights** ``w_i`` (e.g. from
:mod:`discrete_bayes`). This module:

1. Evaluates ``γ*(θ_i)`` for each particle via :func:`gamma_star.gamma_star` (Torch ODE inside).
2. Aggregates with :func:`discrete_bayes.mocu_gamma_star` (weighted median ``\\hatγ``, then
   ``\\sum_i w_i |γ*_i - \\hatγ|``).

**Contrast with :mod:`swing_equation_mocu`:** that module **draws i.i.d. ``(M,K)``** from a box
and estimates MOCU by Monte Carlo (``MOCU_swing_equation``). Here the distribution is **explicit
particles + weights**, not uniform sampling.

**See also:** :mod:`mocu_pycuda` (GPU C++ kernels), :mod:`mocu_torchdiffeq` (torchdiffeq helpers).
"""

from __future__ import annotations

import numpy as np

from .gamma_star import gamma_star
from .discrete_bayes import mocu_gamma_star


def compute_mocu(
    particles_theta: np.ndarray,
    weights: np.ndarray,
    B: np.ndarray,
    P_m: np.ndarray,
    D: float,
    g: np.ndarray,
    r_max: float = 0.1,
    f_min: float = 49.8,
    h: float = 1.0 / 160.0,
    T: float = 10.0,
    M_steps=None,
    gamma_min: float = 0.0,
    gamma_max: float = 100.0,
    max_iterations: int = 20,
    tol: float = 0.01,
    reference_probe_bus=None,
    reference_probe_amplitude=None,
    reference_probe_duration=2.0,
    device: str = "cuda",
) -> float:
    """
    MOCU(p) for a **particle** posterior (design §5.9).

    MOCU(p) = E[J(γ̂(p), ϑ)] = E[|γ*(ϑ) − γ̂(p)|]; γ̂(p) = weighted median of γ*(θ_i).

    Same formula as :func:`swing_equation_mocu.MOCU_swing_equation` when weights are uniform
    on samples; here ``weights`` can be arbitrary (e.g. posterior masses).

    Args:
        particles_theta: [N_particles, 2] array of (M, K) values
        weights: [N_particles] normalized posterior weights
        B, P_m, D, g: System parameters
        r_max, f_min: Frequency constraints
        h, T, M_steps: Time parameters
        gamma_min, gamma_max, max_iterations, tol: Binary search parameters
        device: 'cuda' or 'cpu'

    Returns:
        mocu: MOCU(p) value (float), or NaN if no valid γ*
    """
    N_particles = len(particles_theta)

    gamma_star_values = np.full(N_particles, np.nan)
    for i in range(N_particles):
        theta = particles_theta[i]
        try:
            gamma = gamma_star(
                (float(theta[0]), float(theta[1])),
                B,
                P_m,
                D,
                g,
                r_max=r_max,
                f_min=f_min,
                h=h,
                T=T,
                M_steps=M_steps,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                max_iterations=max_iterations,
                tol=tol,
                reference_probe_bus=reference_probe_bus,
                reference_probe_amplitude=reference_probe_amplitude,
                reference_probe_duration=reference_probe_duration,
                device=device,
            )
            if not (np.isnan(gamma) or np.isinf(gamma)):
                gamma_star_values[i] = gamma
        except Exception:
            continue

    valid = np.isfinite(gamma_star_values)
    if not np.any(valid):
        return np.nan
    # pseudocode.tex eq:mocu_gamma (same as discrete_bayes.mocu_gamma_star)
    w = np.array(weights, dtype=np.float64)
    w[~valid] = 0.0
    sw = float(np.sum(w))
    if sw <= 0:
        return np.nan
    w = w / sw
    mocu_val, _ = mocu_gamma_star(w, gamma_star_values)
    return float(mocu_val)
