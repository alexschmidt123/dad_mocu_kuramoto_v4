"""
Explicit likelihood model for measurement-level uncertainty.

Based on documents/design_part1.tex Section 5:
- Deterministic simulator mean: μ(θ, ξ_t) = ROCOF_max(Δf(·; θ, ξ_t))
- Explicit likelihood: p(y_t | θ, ξ_t) = N(μ(θ, ξ_t), σ²)
- Measurement noise and unmodeled effects enter via σ²
"""

import numpy as np
from typing import Tuple, Optional
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from .swing_equation_ode import solve_swing_equation_ode, solve_swing_equation_ode_batch
from .rocof import extract_rocof


def mu_theta_xi(theta: Tuple[float, float], xi: Tuple[int, float, float],
                B: np.ndarray, P_m: np.ndarray, D: float, g: np.ndarray,
                h: float = 1.0/160.0, T: float = 10.0, M_steps: Optional[int] = None,
                fs: float = 12.0, device: str = 'cuda', timeout: float = 5.0) -> float:
    """
    Compute deterministic simulator mean μ(θ, ξ_t).
    
    Based on documents/design_part1.tex Section 5:
    μ(θ, ξ_t) = ROCOF_max(Δf(·; θ, ξ_t))
    
    Args:
        theta: (M, K) tuple - uncertain parameters
        xi: (b, A, T_p) tuple - probe design
            - b: bus index (0-indexed)
            - A: probe amplitude
            - T_p: probe duration (usually 2.0 s)
        B: Coupling matrix [N, N]
        P_m: Mechanical power [N]
        D: Damping coefficient
        g: Control allocation [N]
        h: ODE time step
        T: Time horizon
        M_steps: Number of time steps (optional)
        fs: Observation sampling frequency
        device: 'cuda' or 'cpu'
        timeout: Maximum time for ODE solving
    
    Returns:
        mu: Deterministic simulator mean (ROCOF_max, scalar)
    """
    M, K = theta
    b, A, T_p = xi
    
    # Solve swing equation with probe (no control: gamma=0)
    try:
        state_traj = solve_swing_equation_ode(
            B, P_m, D, M, K, g,
            gamma=None,  # No control for observation
            probe_bus=b,
            probe_amplitude=A,
            probe_duration=T_p,
            h=h, M_steps=M_steps, T=T,
            device=device, timeout=timeout
        )
        
        # Extract frequency trajectory (last N columns are ω)
        N = len(P_m)
        omega_traj = state_traj[:, N:]  # [M_steps, N]
        
        # Extract ROCOF_max
        rocof_max = extract_rocof(omega_traj, h, fs=fs)
        
        return float(rocof_max)
    except Exception as e:
        # If simulation fails, return a default value
        # In practice, might want to handle this differently
        print(f"[WARNING] mu_theta_xi failed for theta={theta}, xi={xi}: {e}")
        return 0.0


def log_likelihood(y: float, theta: Tuple[float, float], xi: Tuple[int, float, float],
                   sigma: float, B: np.ndarray, P_m: np.ndarray, D: float, g: np.ndarray,
                   h: float = 1.0/160.0, T: float = 10.0, M_steps: Optional[int] = None,
                   fs: float = 12.0, device: str = 'cuda', timeout: float = 5.0) -> float:
    """
    Compute log-likelihood log p(y_t | θ, ξ_t).
    
    Based on documents/design_part1.tex Section 5:
    p(y_t | θ, ξ_t) = N(μ(θ, ξ_t), σ²)
    log p(y_t | θ, ξ_t) = -(y - μ)²/(2σ²) - 0.5*log(2πσ²)
    
    Args:
        y: Observed ROCOF_max (scalar)
        theta: (M, K) tuple
        xi: (b, A, T_p) tuple
        sigma: Measurement noise standard deviation
        B, P_m, D, g: System parameters
        h, T, M_steps, fs, device, timeout: Simulation parameters
    
    Returns:
        log_likelihood: log p(y | θ, ξ) (scalar)
    """
    # Compute deterministic mean
    mu = mu_theta_xi(theta, xi, B, P_m, D, g, h, T, M_steps, fs, device, timeout)
    
    # Compute log-likelihood: log N(y; μ, σ²)
    logp = -0.5 * (y - mu)**2 / (sigma**2) - 0.5 * np.log(2.0 * np.pi * sigma**2)
    
    return float(logp)


def log_likelihood_batch(y: float, thetas: np.ndarray, xi: Tuple[int, float, float],
                         sigma: float, B: np.ndarray, P_m: np.ndarray, D: float, g: np.ndarray,
                         h: float = 1.0/160.0, T: float = 10.0, M_steps: Optional[int] = None,
                         fs: float = 12.0, device: str = 'cuda', timeout: float = 5.0) -> np.ndarray:
    """
    Compute log-likelihood for a batch of theta values using a single batched ODE solve.
    
    Args:
        y: Observed ROCOF_max (scalar)
        thetas: [N_particles, 2] array of (M, K) values
        xi: (b, A, T_p) tuple
        sigma: Measurement noise standard deviation
        B, P_m, D, g: System parameters
        h, T, M_steps, fs, device, timeout: Simulation parameters
    
    Returns:
        log_likelihoods: [N_particles] array of log p(y | θ, ξ)
    """
    N_particles = len(thetas)
    N = len(P_m)
    b, A, T_p = xi
    M_batch = thetas[:, 0].astype(np.float64)
    K_batch = thetas[:, 1].astype(np.float64)
    try:
        state_traj = solve_swing_equation_ode_batch(
            B, P_m, D, M_batch, K_batch, g,
            gamma=None,
            probe_bus=b,
            probe_amplitude=A,
            probe_duration=T_p,
            h=h, M_steps=M_steps, T=T,
            device=device, timeout=timeout,
        )
        # state_traj [M_steps, batch, 2*N]; omega = last N columns
        omega_traj = state_traj[:, :, N:]
        mu_batch = np.zeros(N_particles)
        for i in range(N_particles):
            rocof_max = extract_rocof(omega_traj[:, i, :], h, fs=fs)
            mu_batch[i] = float(rocof_max)
        logp = -0.5 * (y - mu_batch) ** 2 / (sigma ** 2) - 0.5 * np.log(2.0 * np.pi * sigma ** 2)
        return logp.astype(np.float64)
    except Exception as e:
        # Fallback: per-particle sequential (e.g. if batch ODE fails)
        log_likelihoods = np.zeros(N_particles)
        for i in range(N_particles):
            theta = (float(thetas[i, 0]), float(thetas[i, 1]))
            log_likelihoods[i] = log_likelihood(
                y, theta, xi, sigma, B, P_m, D, g,
                h, T, M_steps, fs, device, timeout
            )
        return log_likelihoods
