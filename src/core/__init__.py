"""
Core functionality for swing equation simulation, observation, inference, and decision.

This module consolidates:
- Simulator (swing equation, probe signals)
- Observation extraction (ROCOF)
- Inference (likelihood, posterior)
- Decision: γ*, γ̂(p), cost J(γ,ϑ), MOCU(p)
"""

# Re-export key functions for convenience
from .swing_equation_ode import (
    solve_swing_equation_ode,
    solve_swing_equation_ode_batch,
    extract_frequency_features,
    extract_frequency_features_batch,
)
from .swing_equation_mocu import MOCU_swing_equation, MOCU_swing_equation_design_j, binary_search_gamma_star
from .rocof import extract_rocof
from .likelihood import log_likelihood, mu_theta_xi
from .posterior_particles import posterior_weights, get_credible_set
from .gamma_star import gamma_star
from .mocu import compute_mocu

__all__ = [
    'solve_swing_equation_ode',
    'solve_swing_equation_ode_batch',
    'extract_frequency_features',
    'extract_frequency_features_batch',
    'MOCU_swing_equation',
    'MOCU_swing_equation_design_j',
    'binary_search_gamma_star',
    'extract_rocof',
    'extract_max_rocof',
    'log_likelihood',
    'mu_theta_xi',
    'posterior_weights',
    'get_credible_set',
    'gamma_star',
    'compute_mocu',
]
