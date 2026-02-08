"""
MOCU computation using particle-based posterior.

Based on documents/design_part1.tex Section 6:
MOCU(h_T) = E_{θ~p(θ|h_T)}[γ*(A_T) - γ*(θ)]
where A_T is the credible set of p(θ | h_T)
"""

import numpy as np
from typing import List, Tuple
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from .gamma_star import gamma_star
from .posterior_particles import get_credible_set


def compute_mocu(particles_theta: np.ndarray, weights: np.ndarray,
                 B: np.ndarray, P_m: np.ndarray, D: float, g: np.ndarray,
                 r_max: float = 0.5, f_min: float = 59.5,
                 h: float = 1.0/160.0, T: float = 10.0, M_steps=None,
                 gamma_min: float = 0.0, gamma_max: float = 100.0,
                 max_iterations: int = 20, tol: float = 0.01,
                 credible_mass: float = 0.95,
                 device: str = 'cuda') -> float:
    """
    Compute MOCU using particle-based posterior.
    
    Based on design_part1.tex Section 6:
    MOCU(h_T) = E_{θ~p(θ|h_T)}[γ*(A_T) - γ*(θ)]
    
    where:
    - A_T is the credible set of p(θ | h_T)
    - γ*(A_T) = max_{θ∈A_T} γ*(θ)
    - γ*(θ) is computed via binary search
    
    Args:
        particles_theta: [N_particles, 2] array of (M, K) values
        weights: [N_particles] normalized posterior weights
        B, P_m, D, g: System parameters
        r_max, f_min: Frequency constraints
        h, T, M_steps: Time parameters
        gamma_min, gamma_max, max_iterations, tol: Binary search parameters
        credible_mass: Credible mass for credible set (default 0.95)
        device: 'cuda' or 'cpu'
    
    Returns:
        mocu: MOCU value (float)
    """
    N_particles = len(particles_theta)
    
    # Get credible set A_T
    credible_set = get_credible_set(particles_theta, weights, credible_mass)
    credible_particles = particles_theta[credible_set]
    
    # Compute γ*(θ) for all particles in credible set
    gamma_star_credible = []
    for theta in credible_particles:
        try:
            gamma = gamma_star(
                (float(theta[0]), float(theta[1])),
                B, P_m, D, g,
                r_max=r_max, f_min=f_min,
                h=h, T=T, M_steps=M_steps,
                gamma_min=gamma_min, gamma_max=gamma_max,
                max_iterations=max_iterations, tol=tol,
                device=device
            )
            if not (np.isnan(gamma) or np.isinf(gamma)):
                gamma_star_credible.append(gamma)
        except Exception as e:
            # Skip failed computations
            print(f"[WARNING] gamma_star failed for theta={theta}: {e}")
            continue
    
    if len(gamma_star_credible) == 0:
        return np.nan
    
    # γ*(A_T) = max_{θ∈A_T} γ*(θ)
    gamma_star_A = np.max(gamma_star_credible)
    
    # Compute γ*(θ) for all particles (for expectation)
    gamma_star_values = []
    for i, theta in enumerate(particles_theta):
        try:
            gamma = gamma_star(
                (float(theta[0]), float(theta[1])),
                B, P_m, D, g,
                r_max=r_max, f_min=f_min,
                h=h, T=T, M_steps=M_steps,
                gamma_min=gamma_min, gamma_max=gamma_max,
                max_iterations=max_iterations, tol=tol,
                device=device
            )
            if not (np.isnan(gamma) or np.isinf(gamma)):
                gamma_star_values.append((i, gamma))
        except Exception as e:
            # Skip failed computations
            continue
    
    if len(gamma_star_values) == 0:
        return np.nan
    
    # Compute MOCU: E[γ*(A_T) - γ*(θ)] = Σ_n w[n] * (γ*(A_T) - γ*(θ_n))
    mocu_val = 0.0
    for i, gamma_star_theta in gamma_star_values:
        mocu_val += weights[i] * (gamma_star_A - gamma_star_theta)
    
    return float(mocu_val)
