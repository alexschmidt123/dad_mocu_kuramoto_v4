"""
Observation module - compatibility layer.

Provides both old (dictionary) and new (ROCOF-only scalar) observation formats.
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.swing_equation_ode import extract_frequency_features
from .rocof import extract_rocof, extract_max_rocof, extract_rocof_from_features


def get_observation(omega_trajectory, h, fs=12.0, format='rocof_only',
                   rocof_method='full_window', T_obs_sec=10.0,
                   rocof_window_sec=0.5, rocof_eval_sec=1.0):
    """
    Get observation in specified format.
    Follows documents/pseucocode _parameter_list.md and design_part1.tex Section 4.

    Args:
        omega_trajectory: [M, N] frequency trajectory
        h: ODE time step
        fs: Observation sampling frequency (default 12.0 Hz)
        format: 'rocof_only' (scalar) or 'full' (dictionary)
        rocof_method: 'full_window' (doc-compliant: max |diff(Δf)/dt| over T_obs)
                      or 'sliding_window' (legacy: linear fit over rocof_window_sec in first rocof_eval_sec)
        T_obs_sec: Observation window in seconds (default 10.0, design table)
        rocof_window_sec, rocof_eval_sec: Used only when rocof_method='sliding_window'

    Returns:
        If format='rocof_only': scalar y_t = ROCOF_max
        If format='full': dictionary with all features
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
