"""
MOCU computation for second-order Kuramoto (swing equation).

Based on documents/design_part1.tex:
- MOCU measures expected excess primary control due to uncertainty in (M,K)
- MOCU(p_t) = E_{(M,K)~p_t}[γ*(A_t) - γ*(M,K)]
- γ*(M,K) is computed via binary search over γ to satisfy frequency constraints
"""

import numpy as np
import torch
from typing import Tuple, Optional

from .swing_equation_ode import (
    solve_swing_equation_ode,
    solve_swing_equation_ode_batch,
    check_frequency_synchronization,
    extract_frequency_features,
    extract_frequency_features_batch,
)

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("[WARNING] torchdiffeq not available. Install with: pip install torchdiffeq")


def binary_search_gamma_star(B, P_m, D, M, K, g, 
                             r_max=0.5, f_min=49.5,  # Frequency constraints (Hz, 50 Hz nominal, aligned with MATLAB .mdl)
                             h=1.0/160.0, T=10.0, M_steps=None,
                             gamma_min=0.0, gamma_max=100.0, 
                             max_iterations=20, tol=0.01,
                             device='cuda'):
    """
    Binary search for γ*(M,K) - minimum control capacity to satisfy frequency constraints.
    
    Based on documents/design_part1.tex:
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
        f_min: Minimum frequency constraint (Hz, default 49.5 for 50 Hz)
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
    
    N = len(P_m)
    
    # First, check if gamma_min satisfies constraints (early exit if it does)
    try:
        state_traj = solve_swing_equation_ode(
            B, P_m, D, M, K, g, gamma=gamma_min,
            h=h, M_steps=M_steps, T=T, device=device
        )
        omega_traj = state_traj[:, N:]
        features = extract_frequency_features(omega_traj, h, fs=12.0)
        rocof_max = features['ROCOF_max']
        f_min_actual = features['f_min']
        
        # Validate features
        if not (np.isnan(rocof_max) or np.isinf(rocof_max) or np.isnan(f_min_actual) or np.isinf(f_min_actual)):
            if rocof_max <= r_max and f_min_actual >= f_min:
                # gamma_min already satisfies, return it
                return gamma_min
    except (RuntimeError, ValueError) as e:
        # If ODE solving fails, continue to binary search
        if "NaN" not in str(e) and "Inf" not in str(e) and "timeout" not in str(e).lower():
            raise  # Re-raise non-numerical errors
    
    # Check if gamma_max satisfies constraints
    try:
        state_traj = solve_swing_equation_ode(
            B, P_m, D, M, K, g, gamma=gamma_max,
            h=h, M_steps=M_steps, T=T, device=device
        )
        
        # Extract frequency trajectory (last N columns are ω)
        omega_traj = state_traj[:, N:]  # [M_steps, N]
        
        # Check constraints
        features = extract_frequency_features(omega_traj, h, fs=12.0)
        rocof_max = features['ROCOF_max']
        f_min_actual = features['f_min']
        
        # Validate features
        if np.isnan(rocof_max) or np.isinf(rocof_max) or np.isnan(f_min_actual) or np.isinf(f_min_actual):
            # If NaN/Inf, assume system cannot be stabilized with gamma_max
            # But continue to binary search (gamma_min might work)
            gamma_upper = gamma_max  # Keep original upper bound
        elif rocof_max > r_max or f_min_actual < f_min:
            # gamma_max doesn't satisfy, but continue to binary search
            gamma_upper = gamma_max  # Keep original upper bound
        else:
            # gamma_max satisfies, so we know solution is <= gamma_max
            gamma_upper = gamma_max
    except (RuntimeError, ValueError) as e:
        # If ODE solving fails (NaN/Inf/timeout), continue to binary search
        if "NaN" in str(e) or "Inf" in str(e) or "timeout" in str(e).lower():
            gamma_upper = gamma_max  # Keep original upper bound, continue search
        else:
            raise  # Re-raise other errors
    
    # Binary search
    for iteration in range(max_iterations):
        gamma_mid = (gamma_lower + gamma_upper) / 2.0
        
        # Solve ODE with gamma_mid
        try:
            state_traj = solve_swing_equation_ode(
                B, P_m, D, M, K, g, gamma=gamma_mid,
                h=h, M_steps=M_steps, T=T, device=device
            )
            omega_traj = state_traj[:, N:]
            features = extract_frequency_features(omega_traj, h, fs=12.0)  # PMU-like sampling at 12 Hz
            rocof_max = features['ROCOF_max']
            f_min_actual = features['f_min']
            
            # Validate features
            if np.isnan(rocof_max) or np.isinf(rocof_max) or np.isnan(f_min_actual) or np.isinf(f_min_actual):
                # If NaN/Inf, assume constraints not satisfied
                constraints_satisfied = False
            else:
                # Check if constraints are satisfied
                constraints_satisfied = (rocof_max <= r_max) and (f_min_actual >= f_min)
        except (RuntimeError, ValueError) as e:
            # If ODE solving fails (NaN/Inf/timeout), assume constraints not satisfied
            if "NaN" in str(e) or "Inf" in str(e) or "timeout" in str(e).lower():
                constraints_satisfied = False
            else:
                raise  # Re-raise other errors
        
        if constraints_satisfied:
            gamma_upper = gamma_mid
        else:
            gamma_lower = gamma_mid
        
        # Check convergence
        if (gamma_upper - gamma_lower) < tol:
            break
    
    return gamma_upper


def binary_search_gamma_star_batch(M_batch, K_batch, B, P_m, D, g,
                                   r_max=0.5, f_min=49.5,
                                   h=1.0/160.0, T=10.0, M_steps=None,
                                   gamma_min=0.0, gamma_max=100.0,
                                   max_iterations=20, tol=0.01,
                                   device='cuda'):
    """
    Batched binary search for γ*(M,K) over a batch of (M, K) using one ODE solve per iteration.
    M_batch, K_batch: arrays of shape [batch_size].
    Returns: gamma_star [batch_size].
    """
    if not TORCHDIFFEQ_AVAILABLE:
        raise RuntimeError("torchdiffeq not available. Install with: pip install torchdiffeq")
    if M_steps is None:
        M_steps = int(T / h)
    batch_size = len(M_batch)
    N = len(P_m)
    gamma_lower = np.full(batch_size, gamma_min, dtype=np.float64)
    gamma_upper = np.full(batch_size, gamma_max, dtype=np.float64)
    # Optional: early check at gamma_min (batch)
    try:
        state_traj = solve_swing_equation_ode_batch(
            B, P_m, D, M_batch, K_batch, g, gamma=gamma_min,
            h=h, M_steps=M_steps, T=T, device=device
        )
        omega_traj = state_traj[:, :, N:]
        features = extract_frequency_features_batch(omega_traj, h, fs=12.0)
        rocof = features['ROCOF_max']
        f_min_actual = features['f_min']
        satisfied = (~np.isnan(rocof)) & (~np.isinf(rocof)) & (~np.isnan(f_min_actual)) & (~np.isinf(f_min_actual)) & (rocof <= r_max) & (f_min_actual >= f_min)
        gamma_upper[satisfied] = gamma_min
    except Exception:
        pass
    for iteration in range(max_iterations):
        gamma_mid = (gamma_lower + gamma_upper) / 2.0
        try:
            state_traj = solve_swing_equation_ode_batch(
                B, P_m, D, M_batch, K_batch, g, gamma=gamma_mid,
                h=h, M_steps=M_steps, T=T, device=device
            )
            omega_traj = state_traj[:, :, N:]
            features = extract_frequency_features_batch(omega_traj, h, fs=12.0)
            rocof = features['ROCOF_max']
            f_min_actual = features['f_min']
            valid = (~np.isnan(rocof)) & (~np.isinf(rocof)) & (~np.isnan(f_min_actual)) & (~np.isinf(f_min_actual))
            constraints_satisfied = valid & (rocof <= r_max) & (f_min_actual >= f_min)
        except Exception:
            constraints_satisfied = np.zeros(batch_size, dtype=bool)
        gamma_upper[constraints_satisfied] = gamma_mid[constraints_satisfied]
        gamma_lower[~constraints_satisfied] = gamma_mid[~constraints_satisfied]
        if np.all((gamma_upper - gamma_lower) < tol):
            break
    return gamma_upper.astype(np.float64)


def MOCU_swing_equation(K_max: int, B: np.ndarray, P_m: np.ndarray, D: float,
                       M_lower: float, M_upper: float, K_lower: float, K_upper: float,
                       g: np.ndarray, r_max=0.5, f_min=49.5,
                       h=1.0/160.0, T=10.0, M_steps=None,
                       seed: int = 0, device: str = 'cuda') -> float:
    """
    Compute MOCU for second-order Kuramoto (swing equation).
    
    Based on documents/design_part1.tex:
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
        f_min: Minimum frequency constraint (Hz, default 49.5 for 50 Hz)
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
    
    # Sample all (M, K) from uniform distribution over bounds
    M_batch = np.random.uniform(M_lower, M_upper, size=K_max).astype(np.float64)
    K_batch = np.random.uniform(K_lower, K_upper, size=K_max).astype(np.float64)
    
    # Batched binary search: one ODE solve per iteration over all samples
    try:
        gamma_star_values = binary_search_gamma_star_batch(
            M_batch, K_batch, B, P_m, D, g,
            r_max=r_max, f_min=f_min,
            h=h, T=T, M_steps=M_steps,
            gamma_min=0.0, gamma_max=100.0,
            max_iterations=20, tol=0.01,
            device=device
        )
    except Exception as e:
        raise RuntimeError(f"Batched MOCU failed: {e}") from e
    
    # Filter invalid (NaN/Inf/negative)
    valid = (~np.isnan(gamma_star_values)) & (~np.isinf(gamma_star_values)) & (gamma_star_values >= 0)
    if np.sum(valid) == 0:
        raise RuntimeError(f"All {K_max} samples failed. Check parameter bounds and system stability.")
    if np.sum(valid) < K_max * 0.5:
        import warnings
        warnings.warn(f"Only {np.sum(valid)}/{K_max} samples succeeded. Some parameter combinations may be unstable.")
    gamma_star_values = gamma_star_values[valid]
    
    # Compute γ*(A_t) = max_{(M,K)∈A_t} γ*(M,K)
    # A_t is the support (bounds), so we compute max over corner cases
    # For simplicity, we use the max of sampled values
    # In practice, we might want to check corner cases explicitly
    gamma_star_A = np.max(gamma_star_values)
    
    # Compute MOCU: E[γ*(A_t) - γ*(M,K)]
    MOCU_val = np.mean(gamma_star_A - gamma_star_values)
    
    return float(MOCU_val)
