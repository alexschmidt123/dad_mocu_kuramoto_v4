"""
Scalar observation y_t for sBOED (pseudocode.tex §Observation): mainly ROCOF_max.

Optional ``format='full'`` returns the full frequency-feature dict (legacy / debugging).
"""

import numpy as np

from src.core.swing_equation_ode import extract_frequency_features
from .rocof import extract_rocof, extract_max_rocof, extract_rocof_from_features


def get_observation(
    omega_trajectory,
    h,
    fs=12.0,
    format="rocof_only",
    rocof_method="full_window",
    T_obs_sec=10.0,
    rocof_window_sec=0.5,
    rocof_eval_sec=1.0,
):
    """
    Map trajectory → observation y (default: scalar ROCOF_max per pseudocode).

    ``rocof_method='full_window'``: max |d(Δf)/dt| over T_obs_sec (12 Hz).
    ``sliding_window``: legacy linear fit in first rocof_eval_sec (uses rocof_window_sec).
    """
    if format == 'rocof_only':
        if rocof_method == 'full_window':
            return extract_max_rocof(omega_trajectory, fs=fs, window_sec=T_obs_sec, h=h)
        return extract_rocof(
            omega_trajectory, h, fs=fs,
            rocof_window_sec=rocof_window_sec, rocof_eval_sec=rocof_eval_sec
        )
    elif format == 'full':
        return extract_frequency_features(omega_trajectory, h, fs=fs)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'rocof_only' or 'full'")


def observation_to_rocof(observation):
    """
    Convert observation to ROCOF_max scalar.
    
    Handles both dictionary (old format) and scalar (new format) inputs.
    
    Args:
        observation: Either dict with 'ROCOF_max' key or scalar ROCOF_max
    
    Returns:
        rocof_max: Scalar ROCOF_max
    """
    if isinstance(observation, dict):
        return extract_rocof_from_features(observation)
    elif isinstance(observation, (int, float, np.number)):
        return float(observation)
    else:
        raise TypeError(f"Cannot convert observation type {type(observation)} to ROCOF")
