"""
**torchdiffeq helpers** for the swing equation (re-uses :mod:`swing_equation_mocu` for ``γ*`` / MOCU).

Not the main entry point: use :func:`swing_equation_mocu.MOCU_swing_equation` for standard MOCU.
For **PyCUDA + embedded CUDA C++**, see :mod:`mocu_pycuda`. For **weighted particles**, see :mod:`mocu_particles`.

Dynamics (design_part1.tex): ``dθ/dt = ω``,
``M dω/dt = P_m - Σ B_ij sin(θ_i-θ_j) - Dω - Kω + u_probe + u_ctrl``.
"""

import numpy as np
import torch
from typing import Union, Optional, Dict, Any

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("[WARNING] torchdiffeq not available. Install with: pip install torchdiffeq")

# Import second-order model functions
from .swing_equation_ode import (
    solve_swing_equation_ode,
    check_frequency_synchronization,
    extract_frequency_features
)
from .swing_equation_mocu import (
    binary_search_gamma_star,
    MOCU_swing_equation
)
from .swing_equation_params import (
    get_default_swing_equation_params,
    sample_uncertain_parameters
)


def solve_kuramoto_ode(B, P_m, D, M, K, g, h, M_steps, device='cuda', method='rk4', timeout=5.0,
                       gamma=None, probe_bus=None, probe_amplitude=None, probe_duration=2.0,
                       theta0=None, omega0=None, T=None):
    """
    Solve swing equation ODE using torchdiffeq.
    
    Args:
        B: Coupling matrix [N, N] (numpy array)
        P_m: Mechanical power [N] (numpy array)
        D: Damping coefficient (scalar)
        M: Inertia (scalar)
        K: Control gain (scalar)
        g: Control allocation [N] (numpy array, sum to 1)
        h: Time step (float)
        M_steps: Number of time steps (int)
        device: 'cuda' or 'cpu'
        method: ODE solver method ('rk4', 'euler', etc.)
        timeout: Maximum time in seconds for ODE solving (default: 5.0)
        gamma: Control capacity (scalar, optional)
        probe_bus: Bus index for probing (int, optional, 0-indexed)
        probe_amplitude: Probe amplitude A (float, optional)
        probe_duration: Probe duration T (float, default 2.0s)
        theta0: Initial phases [N] (numpy array, optional)
        omega0: Initial frequencies [N] (numpy array, optional)
        T: Time horizon (float, optional, computed from h*M_steps if not provided)
    
    Returns:
        state_trajectory: [M_steps, 2*N] where first N columns are θ, last N are ω (numpy array)
    """
    if T is None:
        T = h * M_steps
    
    state_traj = solve_swing_equation_ode(
        B, P_m, D, M, K, g, gamma=gamma,
        probe_bus=probe_bus, probe_amplitude=probe_amplitude, probe_duration=probe_duration,
        h=h, M_steps=M_steps, T=T,
        theta0=theta0, omega0=omega0,
        device=device, method=method, timeout=timeout
    )
    
    return state_traj


def check_synchronization(state_trajectory, M_steps):
    """
    Check if system is frequency-synchronized based on ω trajectory.
    
    Args:
        state_trajectory: [M_steps, 2*N] state trajectory (numpy array)
        M_steps: Number of time steps (int)
    
    Returns:
        is_synchronized: 1 if synchronized, 0 if not
    """
    # Extract frequency trajectory (last N columns are ω)
    N = state_trajectory.shape[1] // 2
    omega_trajectory = state_trajectory[:, N:]  # [M_steps, N]
    
    return check_frequency_synchronization(omega_trajectory, M_steps)


def MOCU_torchdiffeq(K_max: int, B: np.ndarray, P_m: np.ndarray, D: float,
                     M_lower: float, M_upper: float, K_lower: float, K_upper: float,
                     g: np.ndarray, h: float, M: int, T: float,
                     seed: int = 0, device: str = 'cuda',
                     r_max: float = 0.5, f_min: float = 59.5) -> float:
    """
    Compute MOCU(p) for swing equation using torchdiffeq (design §5.9).
    MOCU(p) = E[J(γ̂(p), ϑ)]; J(γ, ϑ) = cost.
    
    Args:
        K_max: Number of Monte Carlo samples
        B: Coupling matrix [N, N] (numpy array, known/fixed)
        P_m: Mechanical power [N] (numpy array, known/fixed)
        D: Damping coefficient (scalar, known/fixed)
        M_lower, M_upper: Inertia bounds (scalars)
        K_lower, K_upper: Control gain bounds (scalars)
        g: Control allocation [N] (numpy array, known/fixed, sum to 1)
        h: Time step (float)
        M: Number of time steps (int)
        T: Time horizon (float)
        seed: Random seed (0 = no seed)
        device: 'cuda' or 'cpu'
        r_max: Maximum ROCOF constraint (Hz/s, default 0.5)
        f_min: Minimum frequency constraint (Hz, default 49.5 for 50 Hz)
    
    Returns:
        MOCU(p) value (float)
    """
    return MOCU_swing_equation(
        K_max, B, P_m, D, M_lower, M_upper, K_lower, K_upper, g,
        r_max=r_max, f_min=f_min,
        h=h, T=T, M_steps=M,
        seed=seed, device=device
    )

