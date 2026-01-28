"""
Particle-based posterior computation.

Based on new_plan.tex Section 5:
- Posterior: p(θ | h_T) ∝ p(θ) ∏_{t=1}^T p(y_t | θ, ξ_t)
- Uses particle weights computed from likelihood
"""

import numpy as np
from typing import List, Tuple, Optional
from .likelihood import log_likelihood_batch


def log_prior(theta: Tuple[float, float], 
              M_lower: float, M_upper: float, 
              K_lower: float, K_upper: float) -> float:
    """
    Compute log prior p(θ) for uniform prior over bounds.
    
    Args:
        theta: (M, K) tuple
        M_lower, M_upper: Inertia bounds
        K_lower, K_upper: Control gain bounds
    
    Returns:
        log_prior: log p(θ) (uniform prior: constant if in bounds, -inf if out)
    """
    M, K = theta
    
    # Uniform prior: constant if within bounds, -inf otherwise
    if (M_lower <= M <= M_upper) and (K_lower <= K <= K_upper):
        # Uniform: p(θ) = 1 / ((M_upper - M_lower) * (K_upper - K_lower))
        # log p(θ) = -log((M_upper - M_lower) * (K_upper - K_lower))
        area = (M_upper - M_lower) * (K_upper - K_lower)
        return -np.log(area) if area > 0 else 0.0
    else:
        return -np.inf


def posterior_weights(particles_theta: np.ndarray, history: List[Tuple[Tuple, float]],
                     sigma: float, B: np.ndarray, P_m: np.ndarray, D: float, g: np.ndarray,
                     M_lower: float, M_upper: float, K_lower: float, K_upper: float,
                     h: float = 1.0/160.0, T: float = 10.0, M_steps: Optional[int] = None,
                     fs: float = 12.0, device: str = 'cuda', timeout: float = 5.0) -> np.ndarray:
    """
    Compute posterior weights for particles given history.
    
    Based on new_plan.tex Section 5:
    p(θ | h_T) ∝ p(θ) ∏_{t=1}^T p(y_t | θ, ξ_t)
    
    Weights: w[n] ∝ p(θ_n) ∏_{t=1}^T p(y_t | θ_n, ξ_t)
    log w[n] = log p(θ_n) + Σ_{t=1}^T log p(y_t | θ_n, ξ_t)
    
    Args:
        particles_theta: [N_particles, 2] array of (M, K) values
        history: List of (xi_t, y_t) tuples where:
            - xi_t: (b, A, T_p) tuple
            - y_t: Observed ROCOF_max (scalar)
        sigma: Measurement noise standard deviation
        B, P_m, D, g: System parameters
        M_lower, M_upper, K_lower, K_upper: Prior bounds
        h, T, M_steps, fs, device, timeout: Simulation parameters
    
    Returns:
        weights: [N_particles] normalized weights (sum to 1)
    """
    N_particles = len(particles_theta)
    
    # Initialize log weights with log prior
    log_weights = np.zeros(N_particles)
    for n in range(N_particles):
        theta = (float(particles_theta[n, 0]), float(particles_theta[n, 1]))
        log_weights[n] = log_prior(theta, M_lower, M_upper, K_lower, K_upper)
    
    # Add log-likelihood for each observation in history
    for xi_t, y_t in history:
        # Compute log-likelihood for all particles
        log_likelihoods = log_likelihood_batch(
            y_t, particles_theta, xi_t, sigma, B, P_m, D, g,
            h, T, M_steps, fs, device, timeout
        )
        log_weights += log_likelihoods
    
    # Normalize weights using log-sum-exp trick for numerical stability
    # w[n] = exp(log_w[n] - log_sum_exp(log_w))
    log_w_max = np.max(log_weights)
    log_weights_shifted = log_weights - log_w_max
    weights = np.exp(log_weights_shifted)
    weights = weights / np.sum(weights)
    
    return weights


def get_credible_set(particles_theta: np.ndarray, weights: np.ndarray, 
                     credible_mass: float = 0.95) -> np.ndarray:
    """
    Select credible set A_T from particles by cumulative weight.
    
    Based on new_plan.tex Section 6:
    A_T = credible set of p(θ | h_T)
    
    Args:
        particles_theta: [N_particles, 2] array of (M, K) values
        weights: [N_particles] normalized weights
        credible_mass: Credible mass (default 0.95 = 95%)
    
    Returns:
        credible_set: Boolean array [N_particles] indicating which particles are in credible set
    """
    # Sort particles by weight (descending)
    sorted_indices = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_indices]
    
    # Cumulative weights
    cum_weights = np.cumsum(sorted_weights)
    
    # Find particles that sum to credible_mass
    n_credible = np.searchsorted(cum_weights, credible_mass) + 1
    n_credible = min(n_credible, len(particles_theta))
    
    # Create boolean mask
    credible_set = np.zeros(len(particles_theta), dtype=bool)
    credible_set[sorted_indices[:n_credible]] = True
    
    return credible_set
