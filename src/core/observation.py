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
from .rocof import extract_rocof, extract_rocof_from_features


def get_observation(omega_trajectory, h, fs=12.0, format='rocof_only'):
    """
    Get observation in specified format.
    
    Args:
        omega_trajectory: [M, N] frequency trajectory
        h: ODE time step
        fs: Observation sampling frequency
        format: 'rocof_only' (scalar) or 'full' (dictionary)
    
    Returns:
        If format='rocof_only': scalar y_t = ROCOF_max
        If format='full': dictionary with all features
    """
    if format == 'rocof_only':
        return extract_rocof(omega_trajectory, h, fs=fs)
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
