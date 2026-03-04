"""
Compute γ*(ϑ) — minimal safe gain (design §3).

γ*(ϑ) = inf{γ ≥ 0 : under reference contingency, ∀t |ḟ| ≤ r_max and f ≥ f_min}.
Used in cost J(γ, ϑ) = |γ − γ*(ϑ)| and MOCU(p) = E[J(γ̂(p), ϑ)].
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from .swing_equation_mocu import binary_search_gamma_star

__all__ = ['gamma_star']


def gamma_star(theta, B, P_m, D, g, r_max=0.1, f_min=49.8,
               h=1.0/160.0, T=10.0, M_steps=None,
               gamma_min=0.0, gamma_max=100.0, max_iterations=20, tol=0.01,
               reference_probe_bus=None, reference_probe_amplitude=None, reference_probe_duration=2.0,
               device='cuda'):
    """
    Compute γ*(θ) for a given parameter θ = (M, K).
    
    Args:
        theta: (M, K) tuple
        B: Coupling matrix [N, N]
        P_m: Mechanical power [N]
        D: Damping coefficient
        g: Control allocation [N]
        r_max: Maximum ROCOF constraint (Hz/s)
        f_min: Minimum frequency constraint (Hz)
        h, T, M_steps: Time parameters
        gamma_min, gamma_max: Binary search bounds
        max_iterations, tol: Binary search parameters
        device: 'cuda' or 'cpu'
    
    Returns:
        gamma_star: Optimal control capacity (float)
    """
    M, K = theta
    return binary_search_gamma_star(
        B, P_m, D, M, K, g,
        r_max=r_max, f_min=f_min,
        h=h, T=T, M_steps=M_steps,
        gamma_min=gamma_min, gamma_max=gamma_max,
        max_iterations=max_iterations, tol=tol,
        reference_probe_bus=reference_probe_bus, reference_probe_amplitude=reference_probe_amplitude, reference_probe_duration=reference_probe_duration,
        device=device
    )
