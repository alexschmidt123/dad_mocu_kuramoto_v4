"""
Probe signal generation (Hann window).

Based on documents/design_part1.tex Section 2:
- Probe injection: u_probe_{ξ,i}(τ) = A_t * s(τ; T_p) if i = b_t, else 0
- Hann window: s(τ; T_p) = 0.5 * (1 - cos(2πτ/T_p))
"""

import numpy as np


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


def apply_probe_signal(b: int, A: float, T_p: float, time_grid: np.ndarray) -> np.ndarray:
    """
    Generate probe signal for all buses.
    
    Args:
        b: Bus index (0-indexed) where probe is applied
        A: Probe amplitude
        T_p: Probe duration
        time_grid: [M] time points
    
    Returns:
        u_probe: [M, N] probe signal (zeros except at bus b)
    """
    M = len(time_grid)
    # Assume N is determined by context (will be set by caller)
    # For now, return signal at bus b only
    u_probe_b = np.array([A * hann_window(t, T_p) for t in time_grid])
    return u_probe_b
