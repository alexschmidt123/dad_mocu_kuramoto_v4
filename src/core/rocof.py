"""
ROCOF (Rate of Change of Frequency) extraction.

Based on documents/pseucocode _parameter_list.md and design_part1.tex Section 4:
- PMU-like frequency measurement: Δf_i(t) = ω_i(t) / (2π)
- Sampling rate: f_s = 12 Hz (ENTSO-E, NASPI standards)
- Two modes:
  1. extract_max_rocof: Full observation window, numerical derivative (doc-compliant).
     y_t = ROCOF_max = max over window of |diff(Δf)/dt|. Matches pseudocode.
  2. extract_rocof: Sliding window (0.5s) with linear fit; eval over first 1s (legacy option).

Reference: pseucocode _parameter_list.md; design_part1.tex Section 4; HICSS 2024 (Peng et al.)
"""

import numpy as np
import sys
import os
from typing import Tuple

# Add project root to path for compatibility
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def extract_max_rocof(omega_series: np.ndarray, fs: float = 12.0,
                      window_sec: float = 10.0, h: float = None,
                      probe_bus: int = None) -> float:
    """
    Extract peak ROCOF from frequency deviation over the observation window.
    Matches documents/pseucocode _parameter_list.md (Design Part 1, Section 4):
    - delta_f = omega_series / (2π)
    - dt = 1/fs, rocof_series = diff(delta_f, axis=0) / dt
    - Return max |rocof| within the first window_sec seconds.

    Args:
        omega_series: [M, N] or [M] frequency trajectory (ω in rad/s)
        fs: Sampling frequency (Hz), default 12.0 (PMU standard)
        window_sec: Observation window in seconds (default 10.0, T_obs in doc)
        h: ODE time step; if provided, downsample to fs first (indices = 0, step, 2*step, ...)
        probe_bus: If provided (0-based index), use ROCOF at that bus only (makes probe choice matter)

    Returns:
        rocof_max: Maximum absolute ROCOF (Hz/s)
    """
    if omega_series.ndim == 1:
        omega_series = omega_series[:, np.newaxis]
    M, N = omega_series.shape
    dt = 1.0 / fs
    if h is not None and h > 0 and (1.0 / h) > fs:
        downsample = max(1, int(round((1.0 / h) / fs)))
        indices = np.arange(0, M, downsample)
        omega_series = omega_series[indices, :]
        M = omega_series.shape[0]
    n_window = min(M, int(round(window_sec * fs)))
    omega_series = omega_series[:n_window, :]
    delta_f = omega_series / (2.0 * np.pi)
    rocof_series = np.diff(delta_f, axis=0) / dt
    if probe_bus is not None and 0 <= probe_bus < rocof_series.shape[1]:
        rocof_max = float(np.max(np.abs(rocof_series[:, probe_bus])))
    else:
        rocof_max = float(np.max(np.abs(rocof_series)))
    return rocof_max


def extract_rocof(omega_trajectory: np.ndarray, h: float, fs: float = 12.0,
                  rocof_window_sec: float = 0.5, rocof_eval_sec: float = 1.0) -> float:
    """
    Extract ROCOF_max from frequency trajectory using sliding window with linear fit.
    
    Based on documents/pseucocode _parameter_list.txt:
    - Sliding window: 0.5s (rocof_window_sec)
    - Evaluation horizon: First 1.0s only (rocof_eval_sec)
    - Method: Linear fit (least squares slope) in each window
    - ROCOF_max = max |slope| over all windows in first 1s
    
    Args:
        omega_trajectory: [M, N] frequency trajectory from ODE (ω values)
        h: ODE time step (float, e.g., 1/160 s)
        fs: Observation sampling frequency (float, default 12.0 Hz)
        rocof_window_sec: Sliding window duration (float, default 0.5s)
        rocof_eval_sec: Evaluation horizon (float, default 1.0s - only first 1s)
    
    Returns:
        rocof_max: Maximum ROCOF (scalar float, Hz/s)
    """
    M, N = omega_trajectory.shape
    
    # Convert ω to frequency deviation: Δf = ω / (2π)
    freq_trajectory = omega_trajectory / (2.0 * np.pi)  # [M, N]
    
    # Downsample to observation sampling frequency fs (PMU-like)
    h_obs = 1.0 / fs  # Observation time step (1/12 ≈ 0.0833 s)
    downsample_factor = int(h_obs / h)
    
    if downsample_factor > 1:
        indices = np.arange(0, M, downsample_factor)
        freq_trajectory_obs = freq_trajectory[indices, :]  # [M_obs, N]
    else:
        freq_trajectory_obs = freq_trajectory
    
    M_obs = freq_trajectory_obs.shape[0]
    
    # Limit to first rocof_eval_sec seconds
    N_eval = int(rocof_eval_sec * fs)  # Number of samples in evaluation window
    N_eval = min(N_eval, M_obs)  # Don't exceed available samples
    freq_trajectory_eval = freq_trajectory_obs[:N_eval, :]  # [N_eval, N]
    
    # Sliding window size (in samples)
    W = int(rocof_window_sec * fs)  # Window size in samples
    W = max(1, W)  # At least 1 sample
    
    # Compute ROCOF using sliding window with linear fit
    rocof_vals = []
    for i in range(max(1, N_eval - W + 1)):
        # Extract segment for this window
        segment = freq_trajectory_eval[i:i+W, :]  # [W, N]
        
        # Linear fit: f(t) = a + b*t, where b is the slope (ROCOF)
        # Use least squares: slope = Σ(t - t_mean)(f - f_mean) / Σ(t - t_mean)²
        t_segment = np.arange(W) * h_obs  # Time array for this segment
        t_mean = np.mean(t_segment)
        
        # Compute slope for each bus
        for bus in range(N):
            f_segment = segment[:, bus]
            f_mean = np.mean(f_segment)
            
            # Least squares slope
            numerator = np.sum((t_segment - t_mean) * (f_segment - f_mean))
            denominator = np.sum((t_segment - t_mean) ** 2)
            
            if denominator > 1e-10:  # Avoid division by zero
                slope = numerator / denominator
                rocof_vals.append(abs(slope))
    
    # ROCOF_max = max over all windows and buses
    rocof_max = max(rocof_vals) if rocof_vals else 0.0
    
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
