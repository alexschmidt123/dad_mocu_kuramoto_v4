"""
Unified interface for first-order and second-order Kuramoto models.

This module provides a unified interface that maintains the original project structure
while supporting both first-order and second-order Kuramoto models with active probing.

Based on: "Probing Signal-Based Inertia and Frequency Response Estimation for
Power Systems with High Penetration of Inverter-Based Resources"
"""

import numpy as np
from typing import Union, Optional, Dict, Any, Tuple
import warnings

# Import first-order model
from .mocu_torchdiffeq import (
    solve_kuramoto_ode as solve_first_order_ode,
    check_synchronization as check_first_order_sync,
    MOCU_torchdiffeq as MOCU_first_order
)

# Import second-order model
try:
    from .swing_equation_ode import (
        solve_swing_equation_ode,
        check_frequency_synchronization,
        extract_frequency_features
    )
    from .swing_equation_mocu import (
        MOCU_swing_equation,
        binary_search_gamma_star
    )
    from .swing_equation_params import (
        get_default_swing_equation_params,
        sample_uncertain_parameters
    )
    SWING_EQUATION_AVAILABLE = True
except ImportError:
    SWING_EQUATION_AVAILABLE = False
    warnings.warn("Swing equation modules not available. Second-order model disabled.")


def get_model_type(config: Dict[str, Any]) -> str:
    """
    Get model type from config, defaulting to 'second_order'.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        'first_order' or 'second_order'
    """
    # Check for explicit model_type setting
    if 'model' in config and 'type' in config['model']:
        return config['model']['type']
    
    # Check for legacy flag
    if 'use_swing_equation' in config:
        return 'second_order' if config['use_swing_equation'] else 'first_order'
    
    # Default to second-order (new model)
    return 'second_order'


def solve_ode(model_type: str, **kwargs):
    """
    Unified ODE solver interface.
    
    Args:
        model_type: 'first_order' or 'second_order'
        **kwargs: Model-specific parameters
    
    Returns:
        Trajectory (phase for first-order, [phase, frequency] for second-order)
    """
    if model_type == 'first_order':
        # First-order: w, a, h, M, device, method, timeout
        return solve_first_order_ode(
            kwargs['w'],
            kwargs['a'],
            kwargs['h'],
            kwargs['M'],
            device=kwargs.get('device', 'cuda'),
            method=kwargs.get('method', 'rk4'),
            timeout=kwargs.get('timeout', 5.0)
        )
    
    elif model_type == 'second_order':
        if not SWING_EQUATION_AVAILABLE:
            raise RuntimeError("Second-order model not available. Install required modules.")
        
        # Second-order: B, P_m, D, M, K, g, and optional probe/control
        return solve_swing_equation_ode(
            kwargs['B'],
            kwargs['P_m'],
            kwargs['D'],
            kwargs['M'],
            kwargs['K'],
            kwargs['g'],
            gamma=kwargs.get('gamma', None),
            probe_bus=kwargs.get('probe_bus', None),
            probe_amplitude=kwargs.get('probe_amplitude', None),
            probe_duration=kwargs.get('probe_duration', 2.0),
            h=kwargs.get('h', 1.0/160.0),
            M_steps=kwargs.get('M_steps', None),
            T=kwargs.get('T', 10.0),
            theta0=kwargs.get('theta0', None),
            omega0=kwargs.get('omega0', None),
            device=kwargs.get('device', 'cuda'),
            method=kwargs.get('method', 'rk4'),
            timeout=kwargs.get('timeout', 5.0)
        )
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def check_sync(model_type: str, trajectory: np.ndarray, M_steps: int, **kwargs) -> int:
    """
    Unified synchronization check interface.
    
    Args:
        model_type: 'first_order' or 'second_order'
        trajectory: Trajectory array
        M_steps: Number of time steps
        **kwargs: Additional model-specific parameters
    
    Returns:
        1 if synchronized, 0 if not
    """
    if model_type == 'first_order':
        return check_first_order_sync(trajectory, M_steps)
    
    elif model_type == 'second_order':
        if not SWING_EQUATION_AVAILABLE:
            raise RuntimeError("Second-order model not available.")
        # For second-order, trajectory is [M_steps, 2*N], extract frequency part
        N = trajectory.shape[1] // 2
        omega_trajectory = trajectory[:, N:]
        return check_frequency_synchronization(omega_trajectory, M_steps, 
                                               tol=kwargs.get('tol', 1e-3))
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compute_mocu(model_type: str, **kwargs) -> float:
    """
    Unified MOCU computation interface.
    
    Args:
        model_type: 'first_order' or 'second_order'
        **kwargs: Model-specific parameters
    
    Returns:
        MOCU value (float)
    """
    if model_type == 'first_order':
        # First-order MOCU: K_max, w, N, h, M, T, aLowerBoundIn, aUpperBoundIn, seed, device
        return MOCU_first_order(
            kwargs['K_max'],
            kwargs['w'],
            kwargs['N'],
            kwargs['h'],
            kwargs['M'],
            kwargs['T'],
            kwargs['aLowerBoundIn'],
            kwargs['aUpperBoundIn'],
            seed=kwargs.get('seed', 0),
            device=kwargs.get('device', 'cuda')
        )
    
    elif model_type == 'second_order':
        if not SWING_EQUATION_AVAILABLE:
            raise RuntimeError("Second-order model not available.")
        
        # Second-order MOCU: K_max, B, P_m, D, M_lower, M_upper, K_lower, K_upper, g, ...
        return MOCU_swing_equation(
            kwargs['K_max'],
            kwargs['B'],
            kwargs['P_m'],
            kwargs['D'],
            kwargs['M_lower'],
            kwargs['M_upper'],
            kwargs['K_lower'],
            kwargs['K_upper'],
            kwargs['g'],
            r_max=kwargs.get('r_max', 0.5),
            f_min=kwargs.get('f_min', 49.5),
            h=kwargs.get('h', 1.0/160.0),
            T=kwargs.get('T', 10.0),
            M_steps=kwargs.get('M_steps', None),
            seed=kwargs.get('seed', 0),
            device=kwargs.get('device', 'cuda')
        )
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def perform_experiment(model_type: str, **kwargs) -> Union[int, Dict[str, Any]]:
    """
    Perform experiment and return observation.
    
    For first-order: Returns binary sync observation (0 or 1)
    For second-order: Returns frequency features dictionary
    
    Args:
        model_type: 'first_order' or 'second_order'
        **kwargs: Model-specific parameters
    
    Returns:
        First-order: int (0 or 1)
        Second-order: dict with frequency features
    """
    if model_type == 'first_order':
        # First-order: solve ODE and check sync
        trajectory = solve_ode(model_type, **kwargs)
        sync_result = check_sync(model_type, trajectory, kwargs['M'])
        return sync_result
    
    elif model_type == 'second_order':
        if not SWING_EQUATION_AVAILABLE:
            raise RuntimeError("Second-order model not available.")
        
        # Second-order: solve ODE and extract frequency features
        trajectory = solve_ode(model_type, **kwargs)
        N = len(kwargs['P_m'])
        omega_trajectory = trajectory[:, N:]  # Extract frequency part
        
        features = extract_frequency_features(
            omega_trajectory,
            h=kwargs.get('h', 1.0/160.0),
            fs=kwargs.get('fs', 12.0)
        )
        return features
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
