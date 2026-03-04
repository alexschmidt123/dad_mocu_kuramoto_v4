"""
MOCU computation using particle-based posterior (weighted particles).

Notation (design §5): J(γ, ϑ) = operational cost = |γ − γ*(ϑ)|; γ̂(p) = Bayes decision = median;
MOCU(p) = E[J(γ̂(p), ϑ)] = E[|γ*(ϑ) − γ̂(p)|]. For weighted particles, γ̂(p) is the weighted median.
"""

import numpy as np
from typing import List, Tuple
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from .gamma_star import gamma_star


def compute_mocu(particles_theta: np.ndarray, weights: np.ndarray,
                 B: np.ndarray, P_m: np.ndarray, D: float, g: np.ndarray,
                 r_max: float = 0.1, f_min: float = 49.8,
                 h: float = 1.0/160.0, T: float = 10.0, M_steps=None,
                 gamma_min: float = 0.0, gamma_max: float = 100.0,
                 max_iterations: int = 20, tol: float = 0.01,
                 reference_probe_bus=None, reference_probe_amplitude=None, reference_probe_duration=2.0,
                 device: str = 'cuda') -> float:
    """
    Compute MOCU(p) using particle-based posterior (design §5.9).
    
    MOCU(p) = E[J(γ̂(p), ϑ)] = E[|γ*(ϑ) − γ̂(p)|]; J(γ, ϑ) = cost.
    γ̂(p) = weighted median of γ*(θ_i). Same formula as MOCU_swing_equation(); here p is
    represented by weighted particles (θ_i, w_i).
    
    Args:
        particles_theta: [N_particles, 2] array of (M, K) values
        weights: [N_particles] normalized posterior weights
        B, P_m, D, g: System parameters
        r_max, f_min: Frequency constraints
        h, T, M_steps: Time parameters
        gamma_min, gamma_max, max_iterations, tol: Binary search parameters
        device: 'cuda' or 'cpu'
    
    Returns:
        mocu: MOCU(p) value (float)
    """
    N_particles = len(particles_theta)
    
    # Compute γ*(θ) for all particles
    gamma_star_values = np.full(N_particles, np.nan)
    for i in range(N_particles):
        theta = particles_theta[i]
        try:
            gamma = gamma_star(
                (float(theta[0]), float(theta[1])),
                B, P_m, D, g,
                r_max=r_max, f_min=f_min,
                h=h, T=T, M_steps=M_steps,
                gamma_min=gamma_min, gamma_max=gamma_max,
                max_iterations=max_iterations, tol=tol,
                reference_probe_bus=reference_probe_bus, reference_probe_amplitude=reference_probe_amplitude, reference_probe_duration=reference_probe_duration,
                device=device
            )
            if not (np.isnan(gamma) or np.isinf(gamma)):
                gamma_star_values[i] = gamma
        except Exception:
            continue
    
    valid = np.isfinite(gamma_star_values)
    if not np.any(valid):
        return np.nan
    g_vals = gamma_star_values[valid]
    w_vals = weights[valid]
    w_vals = w_vals / np.sum(w_vals)  # renormalize over valid
    # Weighted median: smallest m s.t. cumulative weight >= 0.5
    order = np.argsort(g_vals)
    g_sorted = g_vals[order]
    w_sorted = w_vals[order]
    cumw = np.cumsum(w_sorted)
    idx = np.searchsorted(cumw, 0.5, side="left")
    gamma_hat = float(g_sorted[min(idx, len(g_sorted) - 1)])
    # MOCU(p) = E[J(γ̂(p), ϑ)] = E[|γ* − γ̂(p)|]
    mocu_val = np.sum(w_vals * np.abs(g_vals - gamma_hat))
    return float(mocu_val)
