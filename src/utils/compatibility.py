"""
Compatibility utilities for merging old and new code structures.

Provides helper functions to work with both:
- Old: Bounds-based uncertainty (M_lower, M_upper, K_lower, K_upper)
- New: Particle-based uncertainty (particles_theta, weights)
"""

import numpy as np
from typing import Tuple, List


def bounds_to_particles(M_lower: float, M_upper: float, 
                       K_lower: float, K_upper: float,
                       N_particles: int = 1000, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert bounds to particles for compatibility with new structure.
    
    Args:
        M_lower, M_upper: Inertia bounds
        K_lower, K_upper: Control gain bounds
        N_particles: Number of particles to generate
        seed: Random seed
    
    Returns:
        particles_theta: [N_particles, 2] array of (M, K) values
        weights: [N_particles] uniform weights (1/N_particles)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample uniformly from bounds
    M_samples = np.random.uniform(M_lower, M_upper, size=N_particles)
    K_samples = np.random.uniform(K_lower, K_upper, size=N_particles)
    
    particles_theta = np.column_stack([M_samples, K_samples])
    weights = np.ones(N_particles) / N_particles
    
    return particles_theta, weights


def particles_to_bounds(particles_theta: np.ndarray, weights: np.ndarray,
                        credible_mass: float = 0.95) -> Tuple[float, float, float, float]:
    """
    Convert particles to bounds (credible interval) for compatibility with old structure.
    
    Args:
        particles_theta: [N_particles, 2] array of (M, K) values
        weights: [N_particles] normalized weights
        credible_mass: Credible mass for interval (default 0.95)
    
    Returns:
        M_lower, M_upper, K_lower, K_upper: Credible interval bounds
    """
    # Sort by weights (descending)
    sorted_indices = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_indices]
    
    # Cumulative weights
    cum_weights = np.cumsum(sorted_weights)
    
    # Find particles that sum to credible_mass
    n_credible = np.searchsorted(cum_weights, credible_mass) + 1
    n_credible = min(n_credible, len(particles_theta))
    
    # Get credible particles
    credible_indices = sorted_indices[:n_credible]
    credible_particles = particles_theta[credible_indices]
    
    # Compute bounds
    M_lower = np.min(credible_particles[:, 0])
    M_upper = np.max(credible_particles[:, 0])
    K_lower = np.min(credible_particles[:, 1])
    K_upper = np.max(credible_particles[:, 1])
    
    return M_lower, M_upper, K_lower, K_upper


def history_to_new_format(old_history: List) -> List:
    """
    Convert old history format to new format.
    
    Old: List of (probe_bus, probe_amplitude, probe_duration) or (probe_bus, probe_amplitude)
    New: List of ((b, A, T_p), y_t) tuples where y_t is ROCOF_max
    
    Args:
        old_history: Old format history
    
    Returns:
        new_history: New format history (requires observations, so may be incomplete)
    """
    # This is a placeholder - actual conversion requires observations
    # which may not be available in old history
    new_history = []
    for item in old_history:
        if isinstance(item, tuple) and len(item) >= 2:
            if len(item) == 3:
                xi_t = item  # (b, A, T_p)
            elif len(item) == 2:
                b, A = item
                xi_t = (b, A, 2.0)  # Default T_p = 2.0
            else:
                continue
            
            # Note: y_t is missing in old format, so this is incomplete
            # In practice, you'd need to recompute observations
            new_history.append((xi_t, None))  # Placeholder
    
    return new_history
