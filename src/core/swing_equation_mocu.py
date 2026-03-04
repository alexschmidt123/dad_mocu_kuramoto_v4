"""
MOCU computation for second-order Kuramoto (swing equation).

Notation (design §5, ChatGPT-aligned):
  J(γ, ϑ) = operational cost = |γ − γ*(ϑ)|.
  γ̂(p) = Bayes-optimal decision = median over p of γ*(ϑ).
  MOCU(p) = expected objective cost of uncertainty = E_p[J(γ̂(p), ϑ)] = E_p[|γ*(ϑ) − γ̂(p)|].

Implemented by MOCU_swing_equation() and MOCU_swing_equation_design_j() (same value).
γ*(ϑ) is computed via binary search under a reference contingency (r_max, f_min).
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
                             r_max=0.1, f_min=49.8,  # Design §3: 0.1 Hz/s, 49.8 Hz (50 Hz nominal)
                             h=1.0/160.0, T=10.0, M_steps=None,
                             gamma_min=0.0, gamma_max=200.0, 
                             max_iterations=20, tol=0.01,
                             reference_probe_bus=None, reference_probe_amplitude=None, reference_probe_duration=2.0,
                             device='cuda'):
    """
    Binary search for γ*(M,K) - minimum control capacity to satisfy frequency constraints.
    
    Design §3 and documents/gamma_star_and_mocu_math.md: γ* is defined under a
    reference contingency. If reference_probe_bus and reference_probe_amplitude
    are set, the ODE is run with that probe; otherwise nominal (no disturbance).
    Constraints: max_t |df/dt| <= r_max, min_t f(t) >= f_min.
    
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
    
    use_reference = (
        reference_probe_bus is not None
        and reference_probe_amplitude is not None
        and reference_probe_amplitude > 0
    )
    probe_kw = dict(probe_bus=reference_probe_bus, probe_amplitude=reference_probe_amplitude, probe_duration=reference_probe_duration) if use_reference else {}

    # First, check if gamma_min satisfies constraints (early exit if it does)
    try:
        state_traj = solve_swing_equation_ode(
            B, P_m, D, M, K, g, gamma=gamma_min,
            h=h, M_steps=M_steps, T=T, device=device, **probe_kw
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
            h=h, M_steps=M_steps, T=T, device=device, **probe_kw
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
                h=h, M_steps=M_steps, T=T, device=device, **probe_kw
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
                                   r_max=0.1, f_min=49.8,
                                   h=1.0/160.0, T=10.0, M_steps=None,
                                   gamma_min=0.0, gamma_max=200.0,
                                   max_iterations=20, tol=0.01,
                                   reference_probe_bus=None, reference_probe_amplitude=None, reference_probe_duration=2.0,
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
    use_ref = (
        reference_probe_bus is not None
        and reference_probe_amplitude is not None
        and reference_probe_amplitude > 0
    )
    probe_kw_batch = dict(probe_bus=reference_probe_bus, probe_amplitude=reference_probe_amplitude, probe_duration=reference_probe_duration) if use_ref else {}
    gamma_lower = np.full(batch_size, gamma_min, dtype=np.float64)
    gamma_upper = np.full(batch_size, gamma_max, dtype=np.float64)
    # Optional: early check at gamma_min (batch)
    try:
        state_traj = solve_swing_equation_ode_batch(
            B, P_m, D, M_batch, K_batch, g, gamma=gamma_min,
            h=h, M_steps=M_steps, T=T, device=device, **probe_kw_batch
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
                h=h, M_steps=M_steps, T=T, device=device, **probe_kw_batch
            )
            omega_traj = state_traj[:, :, N:]
            features = extract_frequency_features_batch(omega_traj, h, fs=12.0)
            rocof = features['ROCOF_max']
            f_min_actual = features['f_min']
            valid = (~np.isnan(rocof)) & (~np.isinf(rocof)) & (~np.isnan(f_min_actual)) & (~np.isinf(f_min_actual))
            constraints_satisfied = valid & (rocof <= r_max) & (f_min_actual >= f_min)
        except Exception:
            # Batch failed (NaN/timeout): fall back to per-sample ODE with scalar gamma
            gamma_scalar = float(np.median(gamma_mid))
            constraints_satisfied = np.zeros(batch_size, dtype=bool)
            for i in range(batch_size):
                try:
                    st = solve_swing_equation_ode(
                        B, P_m, D, float(M_batch[i]), float(K_batch[i]), g, gamma=gamma_scalar,
                        h=h, M_steps=M_steps, T=T, device=device, **probe_kw_batch
                    )
                    om = st[:, N:]
                    feats = extract_frequency_features(om, h, fs=12.0)
                    rocof_i = feats['ROCOF_max']
                    f_i = feats['f_min']
                    if not (np.isnan(rocof_i) or np.isinf(rocof_i) or np.isnan(f_i) or np.isinf(f_i)):
                        constraints_satisfied[i] = (rocof_i <= r_max) and (f_i >= f_min)
                except Exception:
                    pass
            gamma_mid = np.full(batch_size, gamma_scalar)  # Use scalar for fallback update
        gamma_upper[constraints_satisfied] = gamma_mid[constraints_satisfied]
        gamma_lower[~constraints_satisfied] = gamma_mid[~constraints_satisfied]
        if np.all((gamma_upper - gamma_lower) < tol):
            break
    return gamma_upper.astype(np.float64)


def MOCU_swing_equation(K_max: int, B: np.ndarray, P_m: np.ndarray, D: float,
                       M_lower: float, M_upper: float, K_lower: float, K_upper: float,
                       g: np.ndarray, r_max=0.1, f_min=49.8,
                       h=1.0/160.0, T=10.0, M_steps=None,
                       reference_probe_bus=None, reference_probe_amplitude=None, reference_probe_duration=2.0,
                       seed: int = 0, device: str = 'cuda') -> float:
    """
    Compute MOCU for second-order Kuramoto (swing equation).
    
    Formula (design §5.9): same as design document.
        J(γ, ϑ) = |γ − γ*(ϑ)|  (operational cost)
        γ̂(p) = median of γ*(ϑ) under belief p
        MOCU(p) = E_p[J(γ̂(p), ϑ)] = E_p[|γ*(ϑ) − γ̂(p)|]
    
    Here the belief p is represented by the box [M_lower, M_upper] × [K_lower, K_upper]:
    we sample (M, K) uniformly from that box, compute γ*(M, K) for each via binary search
    (under the reference contingency), then γ̂ = median(γ*), MOCU = mean(|γ* − γ̂|).
    So the formula is correct; if MOCU stays ~constant across steps, the bound-update
    heuristic in evaluation may be keeping the box large (see update_bounds in generate_dad_data).
    
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
    
    # Batched binary search: one ODE solve per iteration over all samples (under reference contingency if set)
    try:
        gamma_star_values = binary_search_gamma_star_batch(
            M_batch, K_batch, B, P_m, D, g,
            r_max=r_max, f_min=f_min,
            h=h, T=T, M_steps=M_steps,
            reference_probe_bus=reference_probe_bus, reference_probe_amplitude=reference_probe_amplitude, reference_probe_duration=reference_probe_duration,
            gamma_min=0.0, gamma_max=200.0,
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
    
    # MOCU(p) = E[J(γ̂(p), ϑ)] = E[|γ*(ϑ) − γ̂(p)|], γ̂(p) = median (design §5.9)
    gamma_hat = np.median(gamma_star_values)
    MOCU_val = np.mean(np.abs(gamma_star_values - gamma_hat))
    
    return float(MOCU_val)


def get_mocu_swing_computer(use_pycuda: bool = None):
    """
    Return MOCU computer: PyCUDA (fast) if available and working, else PyTorch.
    use_pycuda: True=prefer PyCUDA, False=use PyTorch, None=env USE_PYCUDA or True.
    Runs a one-time probe when PyCUDA is requested; on failure falls back to torchdiffeq
    so the reported backend matches what is actually used.
    """
    import os
    import warnings
    if use_pycuda is None:
        use_pycuda = os.environ.get('USE_PYCUDA', '1') in ('1', 'true', 'yes')
    if use_pycuda:
        try:
            from . import mocu_cuda_swing
            from .swing_equation_params import get_default_swing_equation_params

            # One-time probe: run a minimal MOCU to verify PyCUDA compiles and runs
            try:
                params = get_default_swing_equation_params(N=14, topology='ieee14')
                B = np.ascontiguousarray(params['B'].astype(np.float64))
                P_m = np.ascontiguousarray(params['P_m'].astype(np.float64))
                g = np.ascontiguousarray(params['g'].astype(np.float64))
                mocu_cuda_swing.MOCU_swing_pycuda(
                    4, B, P_m, params['D'],
                    params['M_lower'], params['M_upper'], params['K_lower'], params['K_upper'],
                    g, r_max=0.5, f_min=49.8, seed=42)
            except Exception as e:
                warnings.warn(
                    f"PyCUDA probe failed ({type(e).__name__}: {e}). "
                    "Using torchdiffeq for MOCU. Set USE_PYCUDA=0 to silence.",
                    UserWarning,
                    stacklevel=2,
                )
                return (MOCU_swing_equation, 'torchdiffeq')

            def _mocu_pycuda(K_max, B, P_m, D, M_lower, M_upper, K_lower, K_upper, g,
                            r_max=0.5, f_min=49.8, h=1.0/160.0, T=10.0, M_steps=None,
                            reference_probe_bus=0, reference_probe_amplitude=0.5, reference_probe_duration=2.0,
                            seed=0, device='cuda', **kwargs):
                ref_bus = 0 if reference_probe_bus is None else int(reference_probe_bus)
                ref_amp = 0.5 if reference_probe_amplitude is None else float(reference_probe_amplitude)
                ref_dur = 2.0 if reference_probe_duration is None else float(reference_probe_duration)
                return mocu_cuda_swing.MOCU_swing_pycuda(
                    K_max, B, P_m, D, M_lower, M_upper, K_lower, K_upper, g,
                    r_max=r_max, f_min=f_min, h=h, T=T,
                    reference_probe_bus=ref_bus, reference_probe_amplitude=ref_amp, reference_probe_duration=ref_dur,
                    seed=seed)
            return (_mocu_pycuda, 'PyCUDA')
        except Exception:
            pass
    return (MOCU_swing_equation, 'torchdiffeq')


def MOCU_swing_equation_design_j(K_max: int, B: np.ndarray, P_m: np.ndarray, D: float,
                                  M_lower: float, M_upper: float, K_lower: float, K_upper: float,
                                  g: np.ndarray, r_max=0.1, f_min=49.8,
                                  h=1.0/160.0, T=10.0, M_steps=None,
                                  reference_probe_bus=None, reference_probe_amplitude=None, reference_probe_duration=2.0,
                                  seed: int = 0, device: str = 'cuda') -> float:
    """
    MOCU(p) per design §5.9. Same as MOCU_swing_equation() (alias).
    MOCU(p) = E[J(γ̂(p), ϑ)] = E[|γ*(ϑ) − γ̂(p)|], γ̂(p) = median; J(γ, ϑ) = cost.
    """
    return MOCU_swing_equation(
        K_max, B, P_m, D, M_lower, M_upper, K_lower, K_upper, g,
        r_max=r_max, f_min=f_min, h=h, T=T, M_steps=M_steps,
        reference_probe_bus=reference_probe_bus, reference_probe_amplitude=reference_probe_amplitude, reference_probe_duration=reference_probe_duration,
        seed=seed, device=device
    )
