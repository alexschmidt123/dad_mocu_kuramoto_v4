"""
Second-order Kuramoto model (swing equation) for power systems.

Based on documents/design_part1.tex:
- State: [θ, ω] where θ is phase and ω is frequency
- Dynamics:
  dθ/dt = ω
  M dω/dt = P_m - Σ B_ij sin(θ_i - θ_j) - D ω - K ω + u_probe + u_ctrl

This replaces the first-order Kuramoto model to enable DAD methods.
"""

import numpy as np
import torch
from typing import Union, Optional, Tuple

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("[WARNING] torchdiffeq not available. Install with: pip install torchdiffeq")


class SwingEquationODE(torch.nn.Module):
    """
    Second-order Kuramoto (swing equation) system.
    
    State: [θ, ω] where:
    - θ: phase angles [N]
    - ω: frequencies [N]
    
    Dynamics:
    - dθ/dt = ω
    - M dω/dt = P_m - Σ B_ij sin(θ_i - θ_j) - D ω - K ω + u_probe + u_ctrl
    """
    
    def __init__(self, B, P_m, D, M, K, g, device='cuda'):
        """
        Args:
            B: Coupling matrix [N, N] (known, fixed)
            P_m: Mechanical power [N] (known, fixed)
            D: Damping coefficient (scalar, known, fixed)
            M: Inertia (scalar, uncertain parameter)
            K: Control gain (scalar, uncertain parameter)
            g: Control allocation [N] (known, fixed, sum to 1)
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        
        # Convert to tensors if needed
        if isinstance(B, np.ndarray):
            B = torch.tensor(B, dtype=torch.float32)
        if isinstance(P_m, np.ndarray):
            P_m = torch.tensor(P_m, dtype=torch.float32)
        if isinstance(g, np.ndarray):
            g = torch.tensor(g, dtype=torch.float32)
        
        # Known parameters (buffers)
        self.register_buffer('B', B.to(device))
        self.register_buffer('P_m', P_m.to(device))
        self.register_buffer('D', torch.tensor(D, dtype=torch.float32, device=device))
        self.register_buffer('g', g.to(device))
        
        # Uncertain parameters (buffers, will be set per simulation)
        self.register_buffer('M', torch.tensor(M, dtype=torch.float32, device=device))
        self.register_buffer('K', torch.tensor(K, dtype=torch.float32, device=device))
        
        self.N = len(P_m)
        self.device = device
        
        # Control and probe inputs (set via set_inputs method)
        self.gamma = None  # Control capacity (planning decision)
        self.probe_bus = None  # Bus index for probing (0-indexed)
        self.probe_amplitude = None  # Probe amplitude A
        self.probe_duration = None  # Probe duration T (default 2.0s)
    
    def set_parameters(self, M, K):
        """Update uncertain parameters M and K."""
        self.M = torch.tensor(M, dtype=torch.float32, device=self.device)
        self.K = torch.tensor(K, dtype=torch.float32, device=self.device)
    
    def set_control(self, gamma):
        """Set control capacity γ (planning decision)."""
        self.gamma = gamma
    
    def set_probe(self, bus_idx, amplitude, duration=2.0):
        """
        Set probe input parameters.
        
        Args:
            bus_idx: Bus index (0-indexed) where probe is applied
            amplitude: Probe amplitude A
            duration: Probe duration T (default 2.0s)
        """
        self.probe_bus = bus_idx
        self.probe_amplitude = amplitude
        self.probe_duration = duration
    
    def hann_window(self, t, T):
        """
        Hann window function: s(t;T) = 0.5 * (1 - cos(2πt/T))
        
        Args:
            t: Time (scalar or tensor)
            T: Duration (scalar)
        
        Returns:
            Window value (0 if t > T, otherwise Hann window)
        """
        if isinstance(t, torch.Tensor):
            # Vectorized version
            mask = (t <= T).float()
            window = 0.5 * (1.0 - torch.cos(2.0 * np.pi * t / T))
            return mask * window
        else:
            # Scalar version
            if t > T:
                return 0.0
            return 0.5 * (1.0 - np.cos(2.0 * np.pi * t / T))
    
    def forward(self, t, state):
        """
        Compute derivatives for swing equation.
        
        Args:
            t: Time (scalar, required by torchdiffeq)
            state: [θ, ω] concatenated [2*N] (torch tensor)
        
        Returns:
            dstate_dt: [dθ/dt, dω/dt] concatenated [2*N]
        """
        # Split state into phase and frequency
        theta = state[:self.N]  # [N]
        omega = state[self.N:]  # [N]
        
        # dθ/dt = ω
        dtheta_dt = omega
        
        # Compute coupling term: Σ B_ij sin(θ_i - θ_j)
        theta_i = theta.unsqueeze(1)  # [N, 1]
        theta_j = theta.unsqueeze(0)   # [1, N]
        coupling = torch.sum(self.B * torch.sin(theta_j - theta_i), dim=1)  # [N]
        
        # Control input: u_ctrl = γ * g_i * ω_i
        if self.gamma is not None:
            u_ctrl = self.gamma * self.g * omega  # [N]
        else:
            u_ctrl = torch.zeros_like(omega)
        
        # Probe input: u_probe = A * s(t;T) at bus b, 0 elsewhere
        u_probe = torch.zeros_like(omega)
        if self.probe_bus is not None and self.probe_amplitude is not None:
            # t is a scalar tensor from torchdiffeq
            if isinstance(t, torch.Tensor):
                t_val = t.item() if t.numel() == 1 else float(t)
            else:
                t_val = float(t)
            
            window_val = self.hann_window(t_val, self.probe_duration)
            u_probe[self.probe_bus] = self.probe_amplitude * window_val
        
        # M dω/dt = P_m - coupling - D*ω - K*ω + u_probe + u_ctrl
        # dω/dt = (P_m - coupling - D*ω - K*ω + u_probe + u_ctrl) / M
        domega_dt = (self.P_m - coupling - self.D * omega - self.K * omega + 
                     u_probe + u_ctrl) / self.M
        
        # Concatenate derivatives
        dstate_dt = torch.cat([dtheta_dt, domega_dt], dim=0)  # [2*N]
        
        return dstate_dt


class SwingEquationODEBatch(torch.nn.Module):
    """
    Batched second-order Kuramoto (swing equation) for torchdiffeq.
    State shape: [batch, 2*N]; M, K, gamma can be [batch] or scalars.
    """

    def __init__(self, B, P_m, D, M, K, g, gamma=None, device='cuda'):
        """
        Args:
            B, P_m, g: [N, N], [N], [N] (shared)
            D: scalar
            M, K: [batch] or scalar (per-batch or shared)
            gamma: [batch] or scalar or None
        """
        super().__init__()
        if isinstance(B, np.ndarray):
            B = torch.tensor(B, dtype=torch.float32)
        if isinstance(P_m, np.ndarray):
            P_m = torch.tensor(P_m, dtype=torch.float32)
        if isinstance(g, np.ndarray):
            g = torch.tensor(g, dtype=torch.float32)
        self.register_buffer('B', B.to(device))
        self.register_buffer('P_m', P_m.to(device))
        self.register_buffer('D', torch.tensor(D, dtype=torch.float32, device=device))
        self.register_buffer('g', g.to(device))
        self.N = len(P_m)
        self.device = device
        # M, K, gamma: ensure 1d [batch]
        self._set_param('M', M, device)
        self._set_param('K', K, device)
        self._set_param('gamma', gamma, device)
        self.probe_bus = None
        self.probe_amplitude = None
        self.probe_duration = None

    def _set_param(self, name, val, device):
        if val is None:
            setattr(self, name, None)
            return
        if isinstance(val, torch.Tensor):
            t = val.detach().to(dtype=torch.float32, device=device)
        else:
            t = torch.tensor(val, dtype=torch.float32, device=device)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        setattr(self, name, t)

    def set_parameters(self, M, K, gamma=None):
        """Update M [batch], K [batch], optional gamma [batch] or scalar."""
        self._set_param('M', M, self.device)
        self._set_param('K', K, self.device)
        if gamma is not None:
            self._set_param('gamma', gamma, self.device)

    def set_probe(self, bus_idx, amplitude, duration=2.0):
        self.probe_bus = bus_idx
        self.probe_amplitude = amplitude
        self.probe_duration = duration

    def hann_window(self, t, T):
        if isinstance(t, torch.Tensor):
            t_val = t.item() if t.numel() == 1 else float(t)
        else:
            t_val = float(t)
        if t_val > T:
            return 0.0
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * t_val / T))

    def forward(self, t, state):
        # state [batch, 2*N]
        batch = state.shape[0]
        theta = state[:, :self.N]   # [batch, N]
        omega = state[:, self.N:]   # [batch, N]
        dtheta_dt = omega
        # Coupling: for each b, coupling[b,i] = sum_j B[i,j]*sin(theta[b,j]-theta[b,i])
        theta_i = theta.unsqueeze(2)   # [batch, N, 1]
        theta_j = theta.unsqueeze(1)   # [batch, 1, N]
        diff = theta_j - theta_i       # [batch, N, N]
        coupling = (self.B.unsqueeze(0) * torch.sin(diff)).sum(dim=2)  # [batch, N]
        M = self.M
        K = self.K
        if M.dim() == 0 or M.shape[0] != batch:
            M = M.expand(batch)
        if K.dim() == 0 or K.shape[0] != batch:
            K = K.expand(batch)
        M = M.view(batch, 1)
        K = K.view(batch, 1)
        u_ctrl = torch.zeros_like(omega)
        if self.gamma is not None:
            gam = self.gamma
            if gam.dim() == 0 or gam.shape[0] != batch:
                gam = gam.expand(batch)
            u_ctrl = gam.view(batch, 1) * self.g.unsqueeze(0) * omega
        u_probe = torch.zeros_like(omega)
        if self.probe_bus is not None and self.probe_amplitude is not None:
            window_val = self.hann_window(t, self.probe_duration)
            u_probe[:, self.probe_bus] = self.probe_amplitude * window_val
        domega_dt = (self.P_m.unsqueeze(0) - coupling - self.D * omega - K * omega + u_probe + u_ctrl) / M
        dstate_dt = torch.cat([dtheta_dt, domega_dt], dim=1)
        return dstate_dt


def solve_swing_equation_ode_batch(B, P_m, D, M_batch, K_batch, g, gamma=None,
                                   probe_bus=None, probe_amplitude=None, probe_duration=2.0,
                                   h=1.0/160.0, M_steps=None, T=10.0,
                                   theta0=None, omega0=None, device='cuda',
                                   method='rk4', timeout=5.0):
    """
    Solve swing equation ODE for a batch of (M, K) using torchdiffeq.
    state0 has shape [batch, 2*N]; odeint returns [M_steps, batch, 2*N].
    gamma can be scalar or [batch].
    """
    if not TORCHDIFFEQ_AVAILABLE:
        raise RuntimeError("torchdiffeq not available. Install with: pip install torchdiffeq")
    import time
    N = len(P_m)
    if M_steps is None:
        M_steps = int(T / h)
    batch_size = len(M_batch)
    if isinstance(B, np.ndarray):
        B_tensor = torch.tensor(B, dtype=torch.float32, device=device)
    else:
        B_tensor = B.to(device)
    if isinstance(P_m, np.ndarray):
        P_m_tensor = torch.tensor(P_m, dtype=torch.float32, device=device)
    else:
        P_m_tensor = P_m.to(device)
    if isinstance(g, np.ndarray):
        g_tensor = torch.tensor(g, dtype=torch.float32, device=device)
    else:
        g_tensor = g.to(device)
    M_t = torch.tensor(np.asarray(M_batch, dtype=np.float32), device=device)
    K_t = torch.tensor(np.asarray(K_batch, dtype=np.float32), device=device)
    if gamma is not None:
        g_val = np.atleast_1d(gamma)
        gamma_t = torch.tensor(g_val.astype(np.float32), device=device)
        if gamma_t.numel() != batch_size:
            gamma_t = gamma_t.expand(batch_size)
    else:
        gamma_t = None
    ode_system = SwingEquationODEBatch(B_tensor, P_m_tensor, D, M_t, K_t, g_tensor, gamma=gamma_t, device=device)
    if probe_bus is not None and probe_amplitude is not None:
        ode_system.set_probe(probe_bus, probe_amplitude, probe_duration)
    if theta0 is None:
        theta0 = np.zeros((batch_size, N), dtype=np.float32)
    else:
        theta0 = np.broadcast_to(np.asarray(theta0, dtype=np.float32), (batch_size, N))
    if omega0 is None:
        omega0 = np.zeros((batch_size, N), dtype=np.float32)
    else:
        omega0 = np.broadcast_to(np.asarray(omega0, dtype=np.float32), (batch_size, N))
    state0 = torch.tensor(np.concatenate([theta0, omega0], axis=1), dtype=torch.float32, device=device)
    t = torch.linspace(0, T, M_steps, dtype=torch.float32, device=device)
    start_time = time.time()
    try:
        with torch.no_grad():
            state_trajectory = odeint(ode_system, state0, t, method=method)
            if device == 'cuda' and torch.cuda.is_available():
                _ = state_trajectory[0, 0, 0].item()
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise RuntimeError(f"ODE solving exceeded timeout ({timeout}s): {elapsed:.2f}s")
            result = state_trajectory.cpu().numpy()
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                raise RuntimeError("ODE solution contains NaN or Inf values")
            return result
    except RuntimeError as e:
        if "timeout" in str(e).lower() or "nan" in str(e).lower() or "inf" in str(e).lower():
            raise
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise RuntimeError(f"ODE solving exceeded timeout ({timeout}s): {elapsed:.2f}s") from e
        raise RuntimeError(f"ODE solving failed after {elapsed:.2f}s: {e}") from e
    except Exception as e:
        elapsed = time.time() - start_time
        raise RuntimeError(f"ODE solving failed after {elapsed:.2f}s: {e}") from e


def solve_swing_equation_ode(B, P_m, D, M, K, g, gamma=None, 
                             probe_bus=None, probe_amplitude=None, probe_duration=2.0,
                             h=1.0/160.0, M_steps=None, T=10.0, 
                             theta0=None, omega0=None, device='cuda', 
                             method='rk4', timeout=5.0):
    """
    Solve swing equation ODE using torchdiffeq.
    
    Args:
        B: Coupling matrix [N, N] (numpy array)
        P_m: Mechanical power [N] (numpy array)
        D: Damping coefficient (scalar)
        M: Inertia (scalar, uncertain parameter)
        K: Control gain (scalar, uncertain parameter)
        g: Control allocation [N] (numpy array, sum to 1)
        gamma: Control capacity (scalar, optional, planning decision)
        probe_bus: Bus index for probing (int, optional, 0-indexed)
        probe_amplitude: Probe amplitude A (float, optional)
        probe_duration: Probe duration T (float, default 2.0s)
        h: Time step (float, default 1/160)
        M_steps: Number of time steps (int, optional, computed from T/h if not provided)
        T: Time horizon (float, default 10.0s, matches design_part1.tex: t∈[0,10]s)
        theta0: Initial phases [N] (numpy array, optional, default zeros)
        omega0: Initial frequencies [N] (numpy array, optional, default zeros)
        device: 'cuda' or 'cpu'
        method: ODE solver method ('rk4', 'euler', etc.)
        timeout: Maximum time in seconds for ODE solving (default: 5.0)
    
    Returns:
        state_trajectory: [M_steps, 2*N] where first N columns are θ, last N are ω (numpy array)
    """
    if not TORCHDIFFEQ_AVAILABLE:
        raise RuntimeError("torchdiffeq not available. Install with: pip install torchdiffeq")
    
    import time
    
    N = len(P_m)
    
    # Compute number of steps if not provided
    if M_steps is None:
        M_steps = int(T / h)
    
    # Convert to tensors
    if isinstance(B, np.ndarray):
        B_tensor = torch.tensor(B, dtype=torch.float32, device=device)
    else:
        B_tensor = B.to(device)
    
    if isinstance(P_m, np.ndarray):
        P_m_tensor = torch.tensor(P_m, dtype=torch.float32, device=device)
    else:
        P_m_tensor = P_m.to(device)
    
    if isinstance(g, np.ndarray):
        g_tensor = torch.tensor(g, dtype=torch.float32, device=device)
    else:
        g_tensor = g.to(device)
    
    # Create ODE system
    ode_system = SwingEquationODE(B_tensor, P_m_tensor, D, M, K, g_tensor, device=device)
    
    # Set control and probe inputs
    if gamma is not None:
        ode_system.set_control(gamma)
    if probe_bus is not None and probe_amplitude is not None:
        ode_system.set_probe(probe_bus, probe_amplitude, probe_duration)
    
    # Initial conditions
    if theta0 is None:
        theta0 = np.zeros(N)
    if omega0 is None:
        omega0 = np.zeros(N)
    
    # Concatenate initial state [θ, ω]
    state0 = torch.tensor(np.concatenate([theta0, omega0]), 
                          dtype=torch.float32, device=device)
    
    # Time points
    t = torch.linspace(0, T, M_steps, dtype=torch.float32, device=device)
    
    # Solve ODE with timeout protection
    start_time = time.time()
    try:
        with torch.no_grad():  # No gradients needed for MOCU computation
            state_trajectory = odeint(ode_system, state0, t, method=method)
            
            # Explicit synchronization before CPU transfer
            if device == 'cuda' and torch.cuda.is_available():
                _ = state_trajectory[0, 0].item()  # Force computation to complete
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise RuntimeError(f"ODE solving exceeded timeout ({timeout}s): {elapsed:.2f}s")
            
            # Convert to numpy and return
            result = state_trajectory.cpu().numpy()
            
            # Verify result is valid
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                raise RuntimeError("ODE solution contains NaN or Inf values")
            
            return result
            
    except RuntimeError as e:
        # Re-raise timeout or validation errors
        if "timeout" in str(e).lower() or "nan" in str(e).lower() or "inf" in str(e).lower():
            raise
        # For other RuntimeErrors, check if it's a CUDA issue
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise RuntimeError(f"ODE solving exceeded timeout ({timeout}s): {elapsed:.2f}s") from e
        raise RuntimeError(f"ODE solving failed after {elapsed:.2f}s: {e}") from e
    except Exception as e:
        elapsed = time.time() - start_time
        raise RuntimeError(f"ODE solving failed after {elapsed:.2f}s: {e}") from e


def extract_frequency_features(omega_trajectory, h, fs=12.0):
    """
    Extract frequency features from ω trajectory.
    
Based on design_part1.tex Section 4:
- Δf_i(t) = ω_i(t) / (2π)
    - Observations at fs = 12 Hz, t ∈ [0, 10] s
    - Features: [ROCOF_max, f_min, t_settle, ...]
    
    Note: ODE solving uses fine time step h (e.g., 1/160 s) for numerical accuracy.
    Observations are downsampled to fs = 12 Hz (1/12 ≈ 0.0833 s) to match PMU-like sampling.
    
    Args:
        omega_trajectory: [M, N] frequency trajectory from ODE (numpy array)
        h: ODE time step (float, e.g., 1/160 s)
        fs: Observation sampling frequency (float, default 12.0 Hz, matches design_part1.tex)
    
    Returns:
        features: Dictionary with extracted features
    """
    M, N = omega_trajectory.shape
    
    # Convert ω to frequency: Δf = ω / (2π)
    freq_trajectory = omega_trajectory / (2.0 * np.pi)  # [M, N]
    
    # Downsample to observation sampling frequency fs (PMU-like, matches design_part1.tex)
    # ODE uses fine time step h, but observations should be at fs = 12 Hz
    h_obs = 1.0 / fs  # Observation time step (1/12 ≈ 0.0833 s)
    downsample_factor = int(h_obs / h)  # How many ODE steps per observation step
    
    if downsample_factor > 1:
        # Downsample: take every downsample_factor-th sample
        indices = np.arange(0, M, downsample_factor)
        freq_trajectory_obs = freq_trajectory[indices, :]  # [M_obs, N]
        t_obs = np.arange(len(indices)) * h_obs  # Time array at observation rate
    else:
        # ODE time step is already coarser than observation rate (shouldn't happen)
        # Use all samples
        freq_trajectory_obs = freq_trajectory
        t_obs = np.arange(M) * h
    
    M_obs = freq_trajectory_obs.shape[0]
    
    # Compute ROCOF (Rate of Change of Frequency): d(Δf)/dt at observation rate
    rocof = np.gradient(freq_trajectory_obs, axis=0) / h_obs  # [M_obs, N]
    rocof_max = np.max(np.abs(rocof))  # Maximum ROCOF across all buses and time
    
    # Minimum absolute frequency (from downsampled observations)
    # freq_trajectory_obs is frequency deviation Δf, so absolute frequency = f_nominal + Δf
    # 60 Hz nominal (USA): f_absolute = 60.0 + Δf
    f_nominal = 60.0  # Nominal frequency (Hz)
    f_absolute_trajectory = f_nominal + freq_trajectory_obs  # [M_obs, N]
    f_min = np.min(f_absolute_trajectory)  # Minimum absolute frequency
    
    # Settling time (time when frequency deviation is within 1% of final value)
    # Use last 10% of trajectory to estimate final value
    final_window = max(1, int(0.1 * M_obs))
    final_freq = np.mean(freq_trajectory_obs[-final_window:, :])
    
    # Find when frequency settles (within 1% of final)
    freq_deviation = np.abs(freq_trajectory_obs - final_freq)
    threshold = 0.01 * np.abs(final_freq) if final_freq != 0 else 0.01
    settled_mask = freq_deviation < threshold
    
    # Settling time is first time when all buses are settled
    all_settled = np.all(settled_mask, axis=1)
    if np.any(all_settled):
        settle_idx = np.where(all_settled)[0][0]
        t_settle = t_obs[settle_idx]
    else:
        t_settle = t_obs[-1]  # Never settled
    
    features = {
        'ROCOF_max': rocof_max,
        'f_min': f_min,  # Minimum absolute frequency (Hz)
        'f_min_deviation': np.min(freq_trajectory_obs),  # Minimum frequency deviation (Hz)
        't_settle': t_settle,
        'freq_trajectory': freq_trajectory_obs,  # Downsampled trajectory (frequency deviation Δf) at fs Hz
        'freq_absolute_trajectory': f_absolute_trajectory,  # Absolute frequency trajectory (f_nominal + Δf)
        'fs': fs,  # Observation sampling frequency
        'h_obs': h_obs,  # Observation time step
    }
    
    return features


def extract_frequency_features_batch(omega_trajectory, h, fs=12.0):
    """
    Extract frequency features for a batch of trajectories.
    omega_trajectory: [M_steps, batch, N]
    Returns dict with ROCOF_max [batch], f_min [batch].
    """
    M_steps, batch_size, N = omega_trajectory.shape
    freq_trajectory = omega_trajectory / (2.0 * np.pi)
    h_obs = 1.0 / fs
    downsample_factor = max(1, int(h_obs / h))
    indices = np.arange(0, M_steps, downsample_factor)
    freq_trajectory_obs = freq_trajectory[indices, :, :]  # [M_obs, batch, N]
    M_obs = freq_trajectory_obs.shape[0]
    rocof = np.gradient(freq_trajectory_obs, axis=0) / h_obs
    rocof_max = np.max(np.abs(rocof), axis=(0, 2))  # [batch]
    f_nominal = 60.0
    f_absolute = f_nominal + freq_trajectory_obs
    f_min = np.min(f_absolute, axis=(0, 2))  # [batch]
    return {'ROCOF_max': rocof_max, 'f_min': f_min}


def check_frequency_synchronization(omega_trajectory, M_steps, tol=1e-3):
    """
    Check if system is frequency-synchronized based on ω trajectory.
    
    For second-order model, we check if frequencies converge (not phases).
    
    Args:
        omega_trajectory: [M, N] frequency trajectory (numpy array)
        M_steps: Number of time steps (int)
        tol: Tolerance for synchronization (float, default 1e-3)
    
    Returns:
        is_synchronized: 1 if synchronized, 0 if not
    """
    # Use second half of trajectory to check stability
    second_half = omega_trajectory[M_steps//2:, :]
    
    # Compute frequency differences (omega - omega_old)
    diff_t = np.diff(second_half, axis=0)
    
    # Check if all differences are small (synchronized)
    # Original logic: max - min <= tol
    freq_tol = np.max(diff_t) - np.min(diff_t)
    
    # Also check if frequencies themselves converge (not just derivatives)
    # Check variance in final portion
    final_window = int(0.1 * len(second_half))
    final_freqs = second_half[-final_window:, :]
    freq_variance = np.var(final_freqs)
    
    # Synchronized if both derivative differences and frequency variance are small
    is_sync = (freq_tol <= tol) and (freq_variance <= tol)
    
    return 1 if is_sync else 0
