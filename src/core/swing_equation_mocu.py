"""
MOCU computation for second-order Kuramoto (swing equation).

Based on new_plan.tex:
- MOCU measures expected excess primary control due to uncertainty in (M,K)
- MOCU(p_t) = E_{(M,K)~p_t}[γ*(A_t) - γ*(M,K)]
- γ*(M,K) is computed via binary search over γ to satisfy frequency constraints
"""

import numpy as np
import torch
from typing import Tuple, Optional

from .swing_equation_ode import (
    solve_swing_equation_ode, 
    check_frequency_synchronization,
    extract_frequency_features
)

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("[WARNING] torchdiffeq not available. Install with: pip install torchdiffeq")


def binary_search_gamma_star(B, P_m, D, M, K, g, 
                             r_max=0.5, f_min=49.5,  # Frequency constraints (Hz)
                             h=1.0/160.0, T=10.0, M_steps=None,
                             gamma_min=0.0, gamma_max=100.0, 
                             max_iterations=20, tol=0.01,
                             device='cuda'):
    """
    Binary search for γ*(M,K) - minimum control capacity to satisfy frequency constraints.
    
    Based on new_plan.tex:
    γ*(M,K) = min_γ s.t. frequency constraints are satisfied
    Constraints: max_t |df/dt| <= r_max, min_t f(t) >= f_min
    
    Args:
        B: Coupling matrix [N, N] (numpy array)
        P_m: Mechanical power [N] (numpy array)
        D: Damping coefficient (scalar)
        M: Inertia (scalar)
        K: Control gain (scalar)
        g: Control allocation [N] (numpy array, sum to 1)
        r_max: Maximum ROCOF constraint (Hz/s, default 0.5)
        f_min: Minimum frequency constraint (Hz, default 49.5)
        h: Time step (float)
        T: Time horizon (float, default 10.0s)
        M_steps: Number of time steps (int, optional)
        gamma_min: Lower bound for binary search (float)
        gamma_max: Upper bound for binary search (float)
        max_iterations: Maximum binary search iterations (int)
        tol: Tolerance for convergence (float)
        device: 'cuda' or 'cpu'
    
    Returns:
        gamma_star: Optimal control capacity γ*(M,K) (float)
    """
    if not TORCHDIFFEQ_AVAILABLE:
        raise RuntimeError("torchdiffeq not available. Install with: pip install torchdiffeq")
    
    # Compute number of steps if not provided
    if M_steps is None:
        M_steps = int(T / h)
    
    # Binary search for γ*
    gamma_lower = gamma_min
    gamma_upper = gamma_max
    
    # First, check if gamma_max satisfies constraints
    state_traj = solve_swing_equation_ode(
        B, P_m, D, M, K, g, gamma=gamma_max,
        h=h, M_steps=M_steps, T=T, device=device
    )
    
    # Extract frequency trajectory (last N columns are ω)
    N = len(P_m)
    omega_traj = state_traj[:, N:]  # [M_steps, N]
    
    # Check constraints
    features = extract_frequency_features(omega_traj, h)
    rocof_max = features['ROCOF_max']
    f_min_actual = features['f_min']
    
    # If gamma_max doesn't satisfy, return a large value
    if rocof_max > r_max or f_min_actual < f_min:
        return gamma_max * 2.0  # System cannot be stabilized
    
    # Check if gamma_min satisfies constraints
    state_traj = solve_swing_equation_ode(
        B, P_m, D, M, K, g, gamma=gamma_min,
        h=h, M_steps=M_steps, T=T, device=device
    )
    omega_traj = state_traj[:, N:]
    features = extract_frequency_features(omega_traj, h)
    rocof_max = features['ROCOF_max']
    f_min_actual = features['f_min']
    
    if rocof_max <= r_max and f_min_actual >= f_min:
        # gamma_min already satisfies, return it
        return gamma_min
    
    # Binary search
    for iteration in range(max_iterations):
        gamma_mid = (gamma_lower + gamma_upper) / 2.0
        
        # Solve ODE with gamma_mid
        state_traj = solve_swing_equation_ode(
            B, P_m, D, M, K, g, gamma=gamma_mid,
            h=h, M_steps=M_steps, T=T, device=device
        )
        omega_traj = state_traj[:, N:]
        features = extract_frequency_features(omega_traj, h)
        rocof_max = features['ROCOF_max']
        f_min_actual = features['f_min']
        
        # Check if constraints are satisfied
        constraints_satisfied = (rocof_max <= r_max) and (f_min_actual >= f_min)
        
        if constraints_satisfied:
            gamma_upper = gamma_mid
        else:
            gamma_lower = gamma_mid
        
        # Check convergence
        if (gamma_upper - gamma_lower) < tol:
            break
    
    return gamma_upper


def MOCU_swing_equation(K_max: int, B: np.ndarray, P_m: np.ndarray, D: float,
                       M_lower: float, M_upper: float, K_lower: float, K_upper: float,
                       g: np.ndarray, r_max=0.5, f_min=49.5,
                       h=1.0/160.0, T=10.0, M_steps=None,
                       seed: int = 0, device: str = 'cuda') -> float:
    """
    Compute MOCU for second-order Kuramoto (swing equation).
    
    Based on new_plan.tex:
    MOCU(p_t) = E_{(M,K)~p_t}[γ*(A_t) - γ*(M,K)]
    
    where:
    - A_t is the support/credible set of p_t(M,K)
    - γ*(A_t) = max_{(M,K)∈A_t} γ*(M,K)
    - γ*(M,K) is computed via binary search
    
    Args:
        K_max: Number of Monte Carlo samples
        B: Coupling matrix [N, N] (numpy array, known/fixed)
        P_m: Mechanical power [N] (numpy array, known/fixed)
        D: Damping coefficient (scalar, known/fixed)
        M_lower, M_upper: Inertia bounds (scalars)
        K_lower, K_upper: Control gain bounds (scalars)
        g: Control allocation [N] (numpy array, known/fixed, sum to 1)
        r_max: Maximum ROCOF constraint (Hz/s, default 0.5)
        f_min: Minimum frequency constraint (Hz, default 49.5)
        h: Time step (float)
        T: Time horizon (float, default 10.0s)
        M_steps: Number of time steps (int, optional)
        seed: Random seed (0 = no seed)
        device: 'cuda' or 'cpu'
    
    Returns:
        MOCU value (float)
    """
    if not TORCHDIFFEQ_AVAILABLE:
        raise RuntimeError("torchdiffeq not available. Install with: pip install torchdiffeq")
    
    # Set random seed
    if seed != 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    N = len(P_m)
    
    # Compute number of steps if not provided
    if M_steps is None:
        M_steps = int(T / h)
    
    # Sample (M, K) from uniform distribution over bounds
    gamma_star_values = []
    
    for k in range(K_max):
        # Sample M and K uniformly from bounds
        M_sample = np.random.uniform(M_lower, M_upper)
        K_sample = np.random.uniform(K_lower, K_upper)
        
        # Compute γ*(M, K) via binary search
        gamma_star = binary_search_gamma_star(
            B, P_m, D, M_sample, K_sample, g,
            r_max=r_max, f_min=f_min,
            h=h, T=T, M_steps=M_steps,
            device=device
        )
        gamma_star_values.append(gamma_star)
    
    gamma_star_values = np.array(gamma_star_values)
    
    # Compute γ*(A_t) = max_{(M,K)∈A_t} γ*(M,K)
    # A_t is the support (bounds), so we compute max over corner cases
    # For simplicity, we use the max of sampled values
    # In practice, we might want to check corner cases explicitly
    gamma_star_A = np.max(gamma_star_values)
    
    # Compute MOCU: E[γ*(A_t) - γ*(M,K)]
    MOCU_val = np.mean(gamma_star_A - gamma_star_values)
    
    return float(MOCU_val)
