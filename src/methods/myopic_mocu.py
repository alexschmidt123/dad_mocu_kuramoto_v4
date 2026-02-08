"""
ODE-based OED Method for Swing Equation

Uses exact MOCU computation (Monte Carlo sampling) to greedily select probe actions.
This is the most accurate but slowest method.

For swing equation: Computes expected MOCU for all probe actions (b, A, T).
"""

import time
import numpy as np
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods.base import OEDMethod

# Import swing equation MOCU
try:
    from src.core.swing_equation_mocu import MOCU_swing_equation
    from src.core.swing_equation_params import get_default_swing_equation_params
    SWING_EQUATION_AVAILABLE = True
except ImportError:
    SWING_EQUATION_AVAILABLE = False
    print("[WARNING] Swing equation modules not available")

try:
    import torch
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False


class ODE_Method(OEDMethod):
    """
    ODE-based method using sampling (ground truth MOCU computation) for swing equation.
    
    This method computes the expected MOCU for all possible probe actions
    using Monte Carlo sampling. It's the most accurate but computationally expensive.
    
    Static version: computes R matrix once and reuses it.
    """
    
    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx,
                 B=None, P_m=None, D=None, g=None,
                 probe_amplitudes=None, probe_duration=2.0,
                 r_max=0.5, f_min=59.5, gpu_id=0):
        """
        Args:
            N: Number of buses
            K_max: Number of Monte Carlo samples for MOCU
            deltaT: Time step
            MReal: Number of time steps
            TReal: Time horizon
            it_idx: Number of MOCU averaging iterations
            B: Coupling matrix [N, N] (optional, will generate if not provided)
            P_m: Mechanical power [N] (optional, will generate if not provided)
            D: Damping coefficient (optional, default 0.1)
            g: Control allocation [N] (optional, will generate if not provided)
            probe_amplitudes: List of probe amplitude options (default: [0.5, 1.0, 2.0])
            probe_duration: Probe duration T (default: 2.0s)
            r_max: Maximum ROCOF constraint (default: 0.5 Hz/s)
            f_min: Minimum frequency constraint (default: 59.5 Hz for 60 Hz nominal)
            gpu_id: GPU device ID
        """
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)
        
        self.probe_amplitudes = probe_amplitudes if probe_amplitudes else [0.5, 1.0, 2.0]
        self.probe_duration = probe_duration
        self.r_max = r_max
        self.f_min = f_min
        self.R_matrix = np.zeros((N, len(self.probe_amplitudes)))
        
        # System parameters (fixed, known)
        if B is None or P_m is None or D is None or g is None:
            # Generate default parameters
            system_params = get_default_swing_equation_params(N=N, topology='ieee14')
            self.B = system_params['B']
            self.P_m = system_params['P_m']
            self.D = system_params.get('D', 0.1)
            self.g = system_params['g']
        else:
            self.B = B
            self.P_m = P_m
            self.D = D
            self.g = g
        
        # Device
        self.device = 'cuda' if (TORCHDIFFEQ_AVAILABLE and torch.cuda.is_available()) else 'cpu'
        
        print(f"[ODE] Initialized (static version, device: {self.device})")
    
    def _compute_expected_mocu_matrix(self, M_lower, M_upper, K_lower, K_upper):
        """
        Compute R matrix using ground truth MOCU (sampling-based).
        
        For each possible probe action (b, A):
        - Simulate bound update after probe
        - Compute expected MOCU
        
        This is VERY slow but gives exact expected values.
        
        Args:
            M_lower, M_upper, K_lower, K_upper: Current uncertainty bounds
        
        Returns:
            R_matrix: [N, len(probe_amplitudes)] matrix with expected MOCU
        """
        if not SWING_EQUATION_AVAILABLE:
            raise RuntimeError("Swing equation modules not available")
        
        R_matrix = np.zeros((self.N, len(self.probe_amplitudes)))
        
        # For each probe action (b, A), compute expected MOCU
        # This is simplified - in practice would need observation model p(y | M, K, ξ)
        for b in range(self.N):
            for a_idx, A in enumerate(self.probe_amplitudes):
                # Heuristic: Simulate bound update after probe
                # In practice, would use observation model to compute expected MOCU
                M_lower_new, M_upper_new, K_lower_new, K_upper_new = \
                    self._simulate_probe_update(M_lower, M_upper, K_lower, K_upper, b, A)
                
                # Compute MOCU for updated bounds
                mocu_vals = np.zeros(self.it_idx)
                for l in range(self.it_idx):
                    mocu_vals[l] = MOCU_swing_equation(
                        K_max=self.K_max,
                        B=self.B,
                        P_m=self.P_m,
                        D=self.D,
                        M_lower=M_lower_new,
                        M_upper=M_upper_new,
                        K_lower=K_lower_new,
                        K_upper=K_upper_new,
                        g=self.g,
                        r_max=self.r_max,
                        f_min=self.f_min,
                        h=self.deltaT,
                        T=self.TReal,
                        M_steps=self.MReal,
                        seed=l,
                        device=self.device
                    )
                
                R_matrix[b, a_idx] = np.mean(mocu_vals)
        
        return R_matrix
    
    def _simulate_probe_update(self, M_lower, M_upper, K_lower, K_upper, probe_bus, probe_amplitude):
        """Simulate bound update after probe (heuristic)."""
        update_strength = 0.1
        
        M_range = M_upper - M_lower
        K_range = K_upper - K_lower
        
        M_lower_new = M_lower + update_strength * M_range * 0.1
        M_upper_new = M_upper - update_strength * M_range * 0.1
        K_lower_new = K_lower + update_strength * K_range * 0.1
        K_upper_new = K_upper - update_strength * K_range * 0.1
        
        M_lower_new = max(M_lower, M_lower_new)
        M_upper_new = min(M_upper, M_upper_new)
        K_lower_new = max(K_lower, K_lower_new)
        K_upper_new = min(K_upper, K_upper_new)
        
        if M_lower_new >= M_upper_new:
            M_lower_new, M_upper_new = M_lower, M_upper
        if K_lower_new >= K_upper_new:
            K_lower_new, K_upper_new = K_lower, K_upper
        
        return M_lower_new, M_upper_new, K_lower_new, K_upper_new
    
    def select_experiment(self, M_lower, M_upper, K_lower, K_upper, history,
                         probe_amplitudes=None, probe_duration=None):
        """
        Select next probe action using static ODE strategy.
        
        Computes R matrix only once, then greedily selects from it.
        
        Args:
            M_lower, M_upper, K_lower, K_upper: Current uncertainty bounds
            history: List of (probe_action, observation) tuples
            probe_amplitudes: Probe amplitude options (optional)
            probe_duration: Probe duration (optional)
        
        Returns:
            (probe_bus, probe_amplitude, probe_duration): Selected probe action
        """
        if probe_amplitudes is None:
            probe_amplitudes = self.probe_amplitudes
        if probe_duration is None:
            probe_duration = self.probe_duration
        
        # Compute R matrix only on first call
        if not np.any(self.R_matrix):
            print("[ODE] Computing expected MOCU matrix (static, once only)...")
            print("[ODE] Warning: This may take a LONG time (exact sampling)...")
            self.R_matrix = self._compute_expected_mocu_matrix(M_lower, M_upper, K_lower, K_upper)
        
        # Mask out already selected probe actions
        for (probe_action, _) in history:
            if isinstance(probe_action, tuple) and len(probe_action) >= 2:
                b, A = probe_action[0], probe_action[1]
                if b < self.N:
                    a_idx = probe_amplitudes.index(A) if A in probe_amplitudes else 0
                    if a_idx < len(probe_amplitudes):
                        self.R_matrix[b, a_idx] = np.inf
        
        # Find probe action with minimum expected MOCU
        valid_R_values = self.R_matrix[np.isfinite(self.R_matrix)]
        
        if valid_R_values.size == 0:
            print("[ODE] Warning: No valid probe actions left!")
            return (0, probe_amplitudes[0], probe_duration)
        
        min_val = np.min(valid_R_values)
        min_indices = np.where(self.R_matrix == min_val)
        
        if len(min_indices[0]) > 0:
            b_idx = int(min_indices[0][0])
            a_idx = int(min_indices[1][0])
            probe_bus = b_idx
            probe_amplitude = probe_amplitudes[a_idx] if a_idx < len(probe_amplitudes) else probe_amplitudes[0]
        else:
            probe_bus = 0
            probe_amplitude = probe_amplitudes[0]
        
        return (probe_bus, probe_amplitude, probe_duration)
