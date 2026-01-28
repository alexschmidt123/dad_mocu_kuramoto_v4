"""
ROCOF (Rate of Change of Frequency) extraction.

Based on new_plan.tex Section 4:
- PMU-like frequency measurement: Δf_i(t) = ω_i(t) / (2π)
- ROCOF-only observation: y_t = ROCOF_max = max_n |Δf[n+1] - Δf[n]| / Δt
- Sampling rate: f_s = 12 Hz
- Observation window: t ∈ [0, 10] s

This module provides ROCOF-only observation extraction, compatible with
the existing extract_frequency_features() function in src/core/swing_equation_ode.py
"""

import numpy as np
import sys
import os
from typing import Tuple

# Add project root to path for compatibility
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def extract_rocof(omega_trajectory: np.ndarray, h: float, fs: float = 12.0) -> float:
    """
    Extract ROCOF_max from frequency trajectory.
    
    Based on new_plan.tex Section 4:
    - Δf_i(t) = ω_i(t) / (2π)
    - ROCOF: d(Δf)/dt estimated as finite difference
    - y_t = ROCOF_max = max over n in window of |Δf[n+1] - Δf[n]| / Δt
    
    Args:
        omega_trajectory: [M, N] frequency trajectory from ODE (ω values)
        h: ODE time step (float, e.g., 1/160 s)
        fs: Observation sampling frequency (float, default 12.0 Hz)
    
    Returns:
        rocof_max: Maximum ROCOF (scalar float, Hz/s)
    """
    M, N = omega_trajectory.shape
    
    # Convert ω to frequency deviation: Δf = ω / (2π)
    freq_trajectory = omega_trajectory / (2.0 * np.pi)  # [M, N]
    
    # Downsample to observation sampling frequency fs (PMU-like)
    # ODE uses fine time step h, but observations should be at fs = 12 Hz
    h_obs = 1.0 / fs  # Observation time step (1/12 ≈ 0.0833 s)
    downsample_factor = int(h_obs / h)  # How many ODE steps per observation step
    
    if downsample_factor > 1:
        # Downsample: take every downsample_factor-th sample
        indices = np.arange(0, M, downsample_factor)
        freq_trajectory_obs = freq_trajectory[indices, :]  # [M_obs, N]
    else:
        # ODE time step is already coarser than observation rate (shouldn't happen)
        freq_trajectory_obs = freq_trajectory
    
    M_obs = freq_trajectory_obs.shape[0]
    
    # Compute ROCOF: d(Δf)/dt estimated as finite difference
    # rocof[n] = (Δf[n+1] - Δf[n]) / Δt
    # Using np.gradient for numerical stability
    rocof = np.gradient(freq_trajectory_obs, axis=0) / h_obs  # [M_obs, N]
    
    # ROCOF_max = max over n in window of |rocof[n]|
    rocof_max = np.max(np.abs(rocof))  # Maximum ROCOF across all buses and time
    
    return float(rocof_max)


def extract_rocof_from_features(features: dict) -> float:
    """
    Extract ROCOF_max from existing features dictionary.
    
    Compatibility function for existing code that uses extract_frequency_features().
    
    Args:
        features: Dictionary from extract_frequency_features() with 'ROCOF_max' key
    
    Returns:
        rocof_max: Maximum ROCOF (scalar float, Hz/s)
    """
    if isinstance(features, dict) and 'ROCOF_max' in features:
        return float(features['ROCOF_max'])
    else:
        # Fallback: try to extract from features if it's a dict-like object
        try:
            return float(features.get('ROCOF_max', 0.0))
        except (AttributeError, TypeError):
            # If features is not dict-like, assume it's already ROCOF_max
            return float(features) if np.isscalar(features) else 0.0


def extract_rocof_with_trajectory(omega_trajectory: np.ndarray, h: float, 
                                  fs: float = 12.0) -> Tuple[float, np.ndarray]:
    """
    Extract ROCOF_max and return downsampled frequency trajectory.
    
    Args:
        omega_trajectory: [M, N] frequency trajectory from ODE
        h: ODE time step
        fs: Observation sampling frequency
    
    Returns:
        rocof_max: Maximum ROCOF (scalar)
        freq_trajectory_obs: [M_obs, N] downsampled frequency deviation trajectory
    """
    M, N = omega_trajectory.shape
    
    # Convert ω to frequency deviation: Δf = ω / (2π)
    freq_trajectory = omega_trajectory / (2.0 * np.pi)  # [M, N]
    
    # Downsample to observation sampling frequency fs
    h_obs = 1.0 / fs
    downsample_factor = int(h_obs / h)
    
    if downsample_factor > 1:
        indices = np.arange(0, M, downsample_factor)
        freq_trajectory_obs = freq_trajectory[indices, :]
    else:
        freq_trajectory_obs = freq_trajectory
    
    M_obs = freq_trajectory_obs.shape[0]
    
    # Compute ROCOF
    rocof = np.gradient(freq_trajectory_obs, axis=0) / h_obs
    rocof_max = np.max(np.abs(rocof))
    
    return float(rocof_max), freq_trajectory_obs
