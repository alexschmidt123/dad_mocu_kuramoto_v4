"""ROCOF extraction, observations, and swing-equation forward model."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp

from src.domains.swing.design import Design, hann_window


def as_bus_vectors(
    M: np.ndarray | float | list[float],
    K: np.ndarray | float | list[float],
    n_buses: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize θ to length-``n_buses`` vectors (broadcast scalars for legacy data)."""
    M_v = np.asarray(M, dtype=np.float64).reshape(-1)
    K_v = np.asarray(K, dtype=np.float64).reshape(-1)
    if M_v.size == 1:
        M_v = np.full(n_buses, float(M_v[0]))
    if K_v.size == 1:
        K_v = np.full(n_buses, float(K_v[0]))
    if M_v.shape != (n_buses,) or K_v.shape != (n_buses,):
        raise ValueError(f"M and K must be length {n_buses} or scalars")
    return M_v, K_v


def system_mk(system: dict[str, Any], n_buses: int) -> tuple[np.ndarray, np.ndarray]:
    return as_bus_vectors(system["M"], system["K"], n_buses)


def equilibrium_power_injections(B: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Return injections that make ``(theta, omega=0)`` an exact equilibrium."""
    B_v = np.asarray(B, dtype=np.float64)
    theta_v = np.asarray(theta, dtype=np.float64).reshape(-1)
    if B_v.shape != (theta_v.size, theta_v.size):
        raise ValueError("B and theta dimensions do not match")
    angle_diff = theta_v[:, None] - theta_v[None, :]
    return np.sum(B_v * np.sin(angle_diff), axis=1)


def mk_to_json(M: np.ndarray, K: np.ndarray) -> tuple[list[float], list[float]]:
    return [float(x) for x in M], [float(x) for x in K]


def generate_ieee14_coupling_matrix(coupling_strength: float = 1.0) -> np.ndarray:
    """
    Coupling graph for IEEE 14-bus (MATPOWER ``case14`` / UW PSTCA).

    Branch list matches
    https://github.com/MATPOWER/matpower/blob/master/data/case14.m
    (20 lines/transformers; 0-indexed bus ids below).
    """
    N = 14
    B = np.zeros((N, N))
    # MATPOWER case14 branches (fbus,tbus) → 0-indexed
    connections = [
        (0, 1),   # 1-2
        (0, 4),   # 1-5
        (1, 2),   # 2-3
        (1, 3),   # 2-4
        (1, 4),   # 2-5
        (2, 3),   # 3-4
        (3, 4),   # 4-5
        (3, 6),   # 4-7
        (3, 8),   # 4-9
        (4, 5),   # 5-6
        (5, 10),  # 6-11
        (5, 11),  # 6-12
        (5, 12),  # 6-13
        (6, 7),   # 7-8
        (6, 8),   # 7-9
        (8, 9),   # 9-10
        (8, 13),  # 9-14
        (9, 10),  # 10-11
        (11, 12), # 12-13
        (12, 13), # 13-14
    ]
    for i, j in connections:
        B[i, j] = coupling_strength
        B[j, i] = coupling_strength
    return B


def generate_ieee5_coupling_matrix(coupling_strength: float = 1.0) -> np.ndarray:
    """MATPOWER ``case5`` (modified PJM 5-bus) line graph."""
    N = 5
    B = np.zeros((N, N))
    connections = [
        (0, 1),  # 1-2
        (0, 3),  # 1-4
        (0, 4),  # 1-5
        (1, 2),  # 2-3
        (2, 3),  # 3-4
        (3, 4),  # 4-5
    ]
    for i, j in connections:
        B[i, j] = coupling_strength
        B[j, i] = coupling_strength
    return B


def generate_ieee9_coupling_matrix(coupling_strength: float = 1.0) -> np.ndarray:
    """MATPOWER ``case9`` (Chow 9-bus) line graph."""
    N = 9
    B = np.zeros((N, N))
    connections = [
        (0, 3),  # 1-4
        (3, 4),  # 4-5
        (4, 5),  # 5-6
        (2, 5),  # 3-6
        (5, 6),  # 6-7
        (6, 7),  # 7-8
        (7, 1),  # 8-2
        (7, 8),  # 8-9
        (8, 3),  # 9-4
    ]
    for i, j in connections:
        B[i, j] = coupling_strength
        B[j, i] = coupling_strength
    return B


def generate_default_coupling_matrix(N: int, topology: str = 'fully_connected', 
                                     coupling_strength: float = 1.0) -> np.ndarray:
    """
    Generate coupling matrix B for swing equation.
    
    Args:
        N: Number of buses/oscillators
        topology: Network topology ('fully_connected', 'ring', 'star', 'random',
            'ieee5', 'ieee9', 'ieee14')
        coupling_strength: Base coupling strength (float)
    
    Returns:
        B: Coupling matrix [N, N] (numpy array, symmetric)
    """
    if topology == 'ieee5':
        if N != 5:
            raise ValueError(f"IEEE-5 topology requires N=5, got N={N}")
        return generate_ieee5_coupling_matrix(coupling_strength)
    if topology == 'ieee9':
        if N != 9:
            raise ValueError(f"IEEE-9 topology requires N=9, got N={N}")
        return generate_ieee9_coupling_matrix(coupling_strength)
    if topology == 'ieee14':
        if N != 14:
            raise ValueError(f"IEEE-14 topology requires N=14, got N={N}")
        return generate_ieee14_coupling_matrix(coupling_strength)
    
    B = np.zeros((N, N))
    
    if topology == 'fully_connected':
        # All-to-all coupling (similar to first-order model)
        for i in range(N):
            for j in range(i + 1, N):
                B[i, j] = coupling_strength
                B[j, i] = coupling_strength
    
    elif topology == 'ring':
        # Ring topology: each bus connected to neighbors
        for i in range(N):
            j = (i + 1) % N
            B[i, j] = coupling_strength
            B[j, i] = coupling_strength
    
    elif topology == 'star':
        # Star topology: bus 0 is hub
        for i in range(1, N):
            B[0, i] = coupling_strength
            B[i, 0] = coupling_strength
    
    elif topology == 'random':
        # Random topology (Erdős–Rényi-like)
        np.random.seed(42)  # For reproducibility
        p = 0.5  # Connection probability
        for i in range(N):
            for j in range(i + 1, N):
                if np.random.random() < p:
                    B[i, j] = coupling_strength
                    B[j, i] = coupling_strength
    
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    return B


def generate_default_mechanical_power(N: int, method: str = 'uniform',
                                       base_power: float = 1.0) -> np.ndarray:
    """
    Generate mechanical power P_m for each bus.
    
    Args:
        N: Number of buses
        method: Generation method ('uniform', 'random', 'degree_based')
        base_power: Base power value (float)
    
    Returns:
        P_m: Mechanical power [N] (numpy array)
    """
    if method == 'uniform':
        P_m = np.ones(N) * base_power
    
    elif method == 'random':
        np.random.seed(42)
        P_m = base_power * (0.5 + np.random.random(N))
    
    elif method == 'degree_based':
        # Power proportional to node degree (if using degree-based coupling)
        # For now, use uniform as placeholder
        P_m = np.ones(N) * base_power
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return P_m


def generate_default_control_allocation(N: int, method: str = 'uniform') -> np.ndarray:
    """
    Generate control allocation g (spatial allocation of control across buses).
    
    Based on design_part1.tex: sum_i g_i = 1, g_i >= 0
    
    Args:
        N: Number of buses
        method: Allocation method ('uniform', 'random', 'hub_based')
    
    Returns:
        g: Control allocation [N] (numpy array, sum to 1)
    """
    if method == 'uniform':
        g = np.ones(N) / N
    
    elif method == 'random':
        np.random.seed(42)
        g = np.random.random(N)
        g = g / np.sum(g)  # Normalize to sum to 1
    
    elif method == 'hub_based':
        # More control at hub (bus 0)
        g = np.ones(N) * 0.5 / (N - 1)
        g[0] = 0.5
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return g


def get_default_swing_equation_params(N: int, 
                                     topology: str = 'fully_connected',
                                     coupling_strength: float = 1.0,
                                     damping: float = 0.1,
                                     base_power: float = 1.0,
                                     M_lower: float = 0.01,
                                     M_upper: float = 0.06,
                                     K_lower: float = 0.05,
                                     K_upper: float = 0.50) -> dict:
    """
    Generate default system parameters for swing equation.
    
    Args:
        N: Number of buses/oscillators
        topology: Network topology ('fully_connected', 'ring', 'star', 'random')
        coupling_strength: Base coupling strength (float)
        damping: Damping coefficient D (float)
        base_power: Base mechanical power (float)
        M_lower, M_upper: Inertia bounds (floats)
        K_lower, K_upper: Control gain bounds (floats)
    
    Returns:
        params: Dictionary with all system parameters
    """
    B = generate_default_coupling_matrix(N, topology, coupling_strength)
    P_m = generate_default_mechanical_power(N, method='uniform', base_power=base_power)
    g = generate_default_control_allocation(N, method='uniform')
    
    params = {
        'B': B,
        'P_m': P_m,
        'D': damping,
        'g': g,
        'M_lower': M_lower,
        'M_upper': M_upper,
        'K_lower': K_lower,
        'K_upper': K_upper,
        'N': N,
    }
    
    return params


def sample_uncertain_parameters(M_lower: float, M_upper: float,
                               K_lower: float, K_upper: float,
                               seed: Optional[int] = None) -> Tuple[float, float]:
    """
    Sample uncertain parameters (M, K) from uniform distribution.
    
    Args:
        M_lower, M_upper: Inertia bounds
        K_lower, K_upper: Control gain bounds
        seed: Random seed (optional)
    
    Returns:
        (M, K): Sampled inertia and control gain
    """
    if seed is not None:
        np.random.seed(seed)
    
    M = np.random.uniform(M_lower, M_upper)
    K = np.random.uniform(K_lower, K_upper)
    
    return M, K

BusMK = Union[float, np.ndarray]


# --- ROCOF (Rate of Change of Frequency) extraction ---
"""
ROCOF (Rate of Change of Frequency) extraction.

Based on documents/pseucocode _parameter_list.md and design_part1.tex Section 4:
- PMU-like frequency measurement: Δf_i(t) = ω_i(t) / (2π)
- Sampling rate: f_s = 12 Hz (ENTSO-E, NASPI standards)
- Two modes:
  1. extract_max_rocof: Full observation window, numerical derivative (doc-compliant).
     y_t = ROCOF_max = max over window of |diff(Δf)/dt|. Matches pseudocode.
  2. extract_rocof: Sliding window (0.5s) with linear fit; eval over first 1s (legacy option).

Reference: documents/sBOED_design.tex (max-ROCOF observation).
"""

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


"""
Scalar observation y_t for sBOED (documents/sBOED_design.tex Eq. 164–168).

Observation: max absolute ROCOF over the observation window.
"""


def get_observation(
    omega_trajectory,
    h,
    fs=12.0,
    format="rocof_only",
    rocof_method="full_window",
    T_obs_sec=10.0,
    rocof_window_sec=0.5,
    rocof_eval_sec=1.0,
    probe_bus=None,
):
    """
    Map trajectory → observation y (default: scalar ROCOF_max).

    ``rocof_method='full_window'``: max |d(Δf)/dt| over T_obs_sec (12 Hz).
    ``sliding_window``: legacy linear fit in first rocof_eval_sec (uses rocof_window_sec).
    """
    if format == 'rocof_only':
        if rocof_method == 'full_window':
            return extract_max_rocof(
                omega_trajectory, fs=fs, window_sec=T_obs_sec, h=h, probe_bus=probe_bus
            )
        return extract_rocof(
            omega_trajectory, h, fs=fs,
            rocof_window_sec=rocof_window_sec, rocof_eval_sec=rocof_eval_sec
        )
    elif format == 'full':
        raise NotImplementedError("Full feature dict format removed; use rocof_only.")
    else:
        raise ValueError(f"Unknown format: {format}. Use 'rocof_only' or 'full'")


def observation_to_rocof(observation):
    """Convert observation to ROCOF_max scalar."""
    if isinstance(observation, dict):
        return extract_rocof_from_features(observation)
    elif isinstance(observation, (int, float, np.number)):
        return float(observation)
    else:
        raise TypeError(f"Cannot convert observation type {type(observation)} to ROCOF")


"""
Swing-equation simulator for sBOED (documents/sBOED_design.tex Eq. 96–101).

Reduced NREL-style deviation dynamics on IEEE-14 with scalar uncertain (M, K):
    M dΔω_i/dt = P_m_i - Σ_j B_ij sin(θ_i-θ_j) - (K/(2π)+D) Δω_i + u_probe_i(t; ξ)

Each probe is a reset experiment from the configured baseline state. Sequential
order matters for posterior history and policy decisions, not for the physical
forward response of a fixed (θ, ξ) pair.
"""


def _build_system_params(config_swing: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config_swing or {}
    N = int(cfg.get("N", 14))
    params = get_default_swing_equation_params(
        N=N,
        topology=str(cfg.get("topology", "ieee14")),
        coupling_strength=float(cfg.get("coupling_strength", 1.0)),
        damping=float(cfg.get("damping", 0.1)),
        base_power=float(cfg.get("base_power", 1.0)),
        M_lower=float(cfg.get("M_lower", 0.01)),
        M_upper=float(cfg.get("M_upper", 0.06)),
        K_lower=float(cfg.get("K_lower", 0.05)),
        K_upper=float(cfg.get("K_upper", 0.50)),
    )
    if cfg.get("P_m_nodes") is not None:
        params["P_m"] = np.asarray(cfg["P_m_nodes"], dtype=np.float64)
    if cfg.get("theta0_nodes") is not None:
        params["theta0"] = np.asarray(cfg["theta0_nodes"], dtype=np.float64)
    if cfg.get("omega0_nodes") is not None:
        params["omega0"] = np.asarray(cfg["omega0_nodes"], dtype=np.float64)
    if cfg.get("D_nodes") is not None and cfg.get("use_node_specific_damping"):
        params["D_nodes"] = np.asarray(cfg["D_nodes"], dtype=np.float64)
    params["enforce_initial_equilibrium"] = bool(
        cfg.get("enforce_initial_equilibrium", False)
    )
    return params


def probe_input(t: float, design: Design) -> float:
    """Scalar Hann-window probe amplitude at time t (sign: u_probe = -ΔP_e)."""
    if t < 0 or t > design.duration:
        return 0.0
    return design.amplitude * hann_window(t, design.duration)


class SwingSimulator:
    """Forward model F(θ, ξ) → max-ROCOF observation."""

    def __init__(
        self,
        config_swing: dict[str, Any] | None = None,
        *,
        fs_hz: float = 12.0,
        T_obs_sec: float = 10.0,
        ode_dt: float = 1.0 / 160.0,
    ):
        self.params = _build_system_params(config_swing)
        self.N = int(self.params["N"])
        self.B = np.asarray(self.params["B"], dtype=np.float64)
        self.P_m = np.asarray(self.params["P_m"], dtype=np.float64)
        self.g = np.asarray(self.params["g"], dtype=np.float64)
        self.D_scalar = float(self.params["D"])
        self.D_nodes = np.asarray(
            self.params.get("D_nodes", np.full(self.N, self.D_scalar)),
            dtype=np.float64,
        )
        self.theta0 = np.asarray(
            self.params.get("theta0", np.zeros(self.N)),
            dtype=np.float64,
        )
        self.omega0 = np.asarray(
            self.params.get("omega0", np.zeros(self.N)),
            dtype=np.float64,
        )
        self.configured_P_m = self.P_m.copy()
        self.initial_equilibrium_injections = equilibrium_power_injections(
            self.B, self.theta0
        )
        self.initial_equilibrium_residual = (
            self.configured_P_m - self.initial_equilibrium_injections
        )
        if bool(self.params.get("enforce_initial_equilibrium", False)):
            # The reduced lossless swing model must start at an equilibrium.
            # Rough MATPOWER Pg-Pd values are not consistent with the simplified
            # unit-coupling B matrix and otherwise create an artificial drift.
            self.P_m = self.initial_equilibrium_injections.copy()
        self.fs_hz = float(fs_hz)
        self.T_obs_sec = float(T_obs_sec)
        self.ode_dt = float(ode_dt)

    def _rhs(
        self,
        t: float,
        y: np.ndarray,
        M: BusMK,
        K: BusMK,
        design: Design,
    ) -> np.ndarray:
        N = self.N
        M_v, K_v = as_bus_vectors(M, K, N)
        theta = y[:N]
        domega = y[N:]
        decay = K_v / (2.0 * np.pi) + self.D_nodes

        coupling = np.zeros(N, dtype=np.float64)
        for i in range(N):
            for j in range(N):
                if self.B[i, j] != 0:
                    coupling[i] += self.B[i, j] * np.sin(theta[i] - theta[j])

        u = np.zeros(N, dtype=np.float64)
        u[design.bus] = probe_input(t, design)

        dtheta = domega
        ddomega = (self.P_m - coupling - decay * domega + u) / M_v
        return np.concatenate([dtheta, ddomega])

    def simulate_step(
        self,
        M: BusMK,
        K: BusMK,
        design: Design,
        state: np.ndarray | None = None,
        *,
        add_noise: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, np.ndarray]:
        """
        One probing step from ``state`` (equilibrium if None).

        Returns max-ROCOF observation and post-step ODE state.
        """
        y0 = (
            np.concatenate([self.theta0.copy(), self.omega0.copy()])
            if state is None
            else np.asarray(state, dtype=np.float64).copy()
        )
        t_end = max(self.T_obs_sec, design.duration + 0.5)
        t_eval = np.arange(0.0, t_end + self.ode_dt, self.ode_dt)

        sol = solve_ivp(
            fun=lambda t, y: self._rhs(t, y, M, K, design),
            t_span=(0.0, t_end),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        omega_traj = sol.y[self.N :, :].T
        y_clean = get_observation(
            omega_traj,
            h=self.ode_dt,
            fs=self.fs_hz,
            format="rocof_only",
            rocof_method="full_window",
            T_obs_sec=self.T_obs_sec,
            probe_bus=design.bus,
        )
        if add_noise is not None and add_noise > 0:
            if rng is None:
                rng = np.random.default_rng()
            y_clean = float(y_clean) + float(rng.normal(0.0, add_noise))
        return float(y_clean), sol.y[:, -1].copy()

    def simulate_sequence(
        self,
        M: BusMK,
        K: BusMK,
        designs: list[Design],
        *,
        add_noise: float | None = None,
        rng: np.random.Generator | None = None,
        reset_after_probe: bool = True,
    ) -> list[float]:
        """Ordered multi-probe rollout.

        ``reset_after_probe=True`` (Plan-2): each probe starts from equilibrium.
        ``reset_after_probe=False`` (continuous-duration mode): carry ODE state
        across probes so the physical trajectory is one continuous experiment.
        """
        if rng is None:
            rng = np.random.default_rng()
        observations: list[float] = []
        state: np.ndarray | None = None
        for design in designs:
            y, state = self.simulate_step(
                M,
                K,
                design,
                None if reset_after_probe else state,
                add_noise=add_noise,
                rng=rng,
            )
            observations.append(y)
        return observations

    def simulate(
        self,
        M: BusMK,
        K: BusMK,
        design: Design,
        *,
        add_noise: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Simulate one probe from equilibrium; scalar max-ROCOF observation."""
        y, _ = self.simulate_step(M, K, design, None, add_noise=add_noise, rng=rng)
        return y

    def map_batch(
        self,
        M: np.ndarray,
        K: np.ndarray,
        design: Design,
    ) -> np.ndarray:
        """F(θ_n, ξ) for each support point (scalar M,K per grid node → uniform bus vector)."""
        return np.array([
            self.simulate(float(m), float(k), design) for m, k in zip(M, K)
        ])
