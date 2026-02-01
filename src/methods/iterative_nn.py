"""
iNN (Iterative Neural Network) OED Method for Swing Equation

Paper (first-order Kuramoto) used MPNN for MOCU estimation.
Iterative: re-computes expected MOCU for all probe actions at each step.
"""

import time
import numpy as np
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods.base import OEDMethod
from src.models.predictors.swing_predictor_utils import (
    load_swing_mocu_predictor,
    predict_swing_mocu,
)


class iNN_Method(OEDMethod):
    """
    Iterative Neural Network (iNN) method for OED with swing equation.
    Uses MPNN predictor to compute expected MOCU iteratively at each step.
    """
    
    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx, model_name,
                 probe_amplitudes=None, probe_duration=2.0, gpu_id=0, B=None):
        """
        Args:
            N, K_max, deltaT, MReal, TReal, it_idx: as in base
            model_name: Name of trained MPNN predictor directory
            probe_amplitudes: List of probe amplitude options
            probe_duration: Probe duration T
            gpu_id: GPU device ID
            B: Coupling matrix [N,N] (required for MPNN)
        """
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.probe_amplitudes = probe_amplitudes if probe_amplitudes else [0.5, 1.0, 2.0]
        self.probe_duration = probe_duration
        if B is None:
            raise ValueError("B (coupling matrix) required for iNN MPNN predictor.")
        self.B = B
        self.model = None
        self.mean = None
        self.std = None
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self._load_model_and_stats()
    
    def _load_model_and_stats(self):
        """Load MPNN MOCU predictor."""
        try:
            self.model, self.mean, self.std = load_swing_mocu_predictor(
                model_name=self.model_name,
                device=self.device,
                B=self.B,
                N=self.N,
            )
            print(f"[iNN] Loaded MPNN predictor '{self.model_name}' on {self.device}")
        except Exception as e:
            raise FileNotFoundError(
                f"MOCU predictor (MPNN) not found for {self.model_name}. Error: {e}"
            )
    
    def _compute_expected_mocu_matrix(self, M_lower, M_upper, K_lower, K_upper):
        """
        Compute expected MOCU matrix for all probe actions.
        
        For each probe action (b, A), simulate bound update and predict MOCU.
        
        Args:
            M_lower, M_upper, K_lower, K_upper: Current uncertainty bounds
        
        Returns:
            R_matrix: [N, len(probe_amplitudes)] matrix with expected MOCU for each probe
        """
        R_matrix = np.zeros((self.N, len(self.probe_amplitudes)))
        
        # For each probe action (b, A), compute expected MOCU
        # This is a simplified heuristic - in practice, would need observation model
        for b in range(self.N):
            for a_idx, A in enumerate(self.probe_amplitudes):
                # Heuristic: Simulate bound update after probe
                # In practice, this would use an observation model p(y | M, K, ξ)
                # For now, use simple heuristic: probe narrows bounds
                
                # Simulate bound update (simplified - would use actual observation model)
                M_lower_new, M_upper_new, K_lower_new, K_upper_new = \
                    self._simulate_probe_update(M_lower, M_upper, K_lower, K_upper, b, A)
                
                # Predict MOCU (MPNN can condition on probe (b, A); paper used MPNN)
                mocu_pred = predict_swing_mocu(
                    self.model, self.mean, self.std,
                    M_lower_new, M_upper_new, K_lower_new, K_upper_new,
                    device=self.device,
                    probe_bus=b, probe_amplitude=A,
                )
                
                if isinstance(mocu_pred, torch.Tensor):
                    mocu_pred = mocu_pred.cpu().item()
                
                R_matrix[b, a_idx] = float(mocu_pred)
        
        return R_matrix
    
    def _simulate_probe_update(self, M_lower, M_upper, K_lower, K_upper, probe_bus, probe_amplitude):
        """
        Simulate bound update after probe (heuristic).
        
        In practice, this would use observation model: p(y | M, K, ξ)
        For now, use simple heuristic: probe narrows bounds.
        
        Args:
            M_lower, M_upper, K_lower, K_upper: Current bounds
            probe_bus: Bus where probe is applied
            probe_amplitude: Probe amplitude
        
        Returns:
            Updated bounds (M_lower_new, M_upper_new, K_lower_new, K_upper_new)
        """
        # Simple heuristic: narrow bounds by small amount
        # In practice, would use actual observation model
        update_strength = 0.1  # How much to narrow bounds
        
        M_range = M_upper - M_lower
        K_range = K_upper - K_lower
        
        M_lower_new = M_lower + update_strength * M_range * 0.1
        M_upper_new = M_upper - update_strength * M_range * 0.1
        K_lower_new = K_lower + update_strength * K_range * 0.1
        K_upper_new = K_upper - update_strength * K_range * 0.1
        
        # Ensure valid bounds
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
        Select next probe action using iterative iNN strategy.
        
        Re-computes expected MOCU matrix at every step based on current bounds.
        
        Args:
            M_lower, M_upper, K_lower, K_upper: Current uncertainty bounds
            history: List of (probe_action, observation) tuples
            probe_amplitudes: Probe amplitude options (optional, uses self.probe_amplitudes)
            probe_duration: Probe duration (optional, uses self.probe_duration)
        
        Returns:
            (probe_bus, probe_amplitude, probe_duration): Selected probe action
        """
        if probe_amplitudes is None:
            probe_amplitudes = self.probe_amplitudes
        if probe_duration is None:
            probe_duration = self.probe_duration
        
        # Re-compute R matrix at every step (iterative)
        print(f"[iNN] Computing expected MOCU matrix (step {len(history) + 1})...")
        R_matrix = self._compute_expected_mocu_matrix(M_lower, M_upper, K_lower, K_upper)
        
        # Mask out already selected probe actions
        for (probe_action, _) in history:
            if isinstance(probe_action, tuple) and len(probe_action) >= 2:
                b, A = probe_action[0], probe_action[1]
                if b < self.N:
                    a_idx = probe_amplitudes.index(A) if A in probe_amplitudes else 0
                    if a_idx < len(probe_amplitudes):
                        R_matrix[b, a_idx] = np.inf  # Mask out
        
        # Find probe action with minimum expected MOCU
        valid_R_values = R_matrix[np.isfinite(R_matrix)]
        
        if valid_R_values.size == 0:
            print("[iNN] Warning: No valid probe actions left!")
            return (0, probe_amplitudes[0], probe_duration)
        
        min_val = np.min(valid_R_values)
        min_indices = np.where(R_matrix == min_val)
        
        if len(min_indices[0]) > 0:
            b_idx = int(min_indices[0][0])
            a_idx = int(min_indices[1][0])
            probe_bus = b_idx
            probe_amplitude = probe_amplitudes[a_idx] if a_idx < len(probe_amplitudes) else probe_amplitudes[0]
        else:
            # Fallback
            probe_bus = 0
            probe_amplitude = probe_amplitudes[0]
        
        return (probe_bus, probe_amplitude, probe_duration)
