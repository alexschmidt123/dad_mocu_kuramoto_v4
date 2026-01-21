"""
NN (Static Neural Network) OED Method for Swing Equation

This is the static (non-iterative) version using Swing MLP predictor.
It computes the expected MOCU matrix once and reuses it for all selections.

For swing equation model: Uses Swing MLP predictor to predict MOCU from (M, K) bounds.
"""

import time
import numpy as np
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods.base import OEDMethod
from src.models.predictors.swing_predictor_utils import (
    load_swing_mlp_predictor,
    predict_swing_mocu
)


class NN_Method(OEDMethod):
    """
    Static Neural Network (NN) method for OED with swing equation.
    
    Uses Swing MLP predictor to compute expected MOCU once at the beginning,
    then greedily selects probe actions without re-evaluation.
    
    This is faster than iNN but less adaptive.
    """
    
    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx, model_name,
                 probe_amplitudes=None, probe_duration=2.0, gpu_id=0):
        """
        Args:
            N: Number of buses
            K_max: Number of Monte Carlo samples for MOCU
            deltaT: Time step
            MReal: Number of time steps
            TReal: Time horizon
            it_idx: Number of MOCU averaging iterations
            model_name: Name of trained Swing MLP model directory
            probe_amplitudes: List of probe amplitude options (default: [0.5, 1.0, 2.0])
            probe_duration: Probe duration T (default: 2.0s)
            gpu_id: GPU device ID
        """
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.probe_amplitudes = probe_amplitudes if probe_amplitudes else [0.5, 1.0, 2.0]
        self.probe_duration = probe_duration
        self.model = None
        self.mean = None
        self.std = None
        self.R_matrix = np.zeros((N, len(self.probe_amplitudes)))
        
        # Device
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Load model once
        self._load_model_and_stats()
    
    def _load_model_and_stats(self):
        """Load trained Swing MLP model and normalization statistics."""
        try:
            self.model, self.mean, self.std = load_swing_mlp_predictor(
                model_name=self.model_name,
                device=self.device
            )
            print(f"[NN] Loaded Swing MLP predictor '{self.model_name}' on {self.device}")
        except Exception as e:
            raise FileNotFoundError(
                f"Swing MLP model not found for {self.model_name}. "
                f"Please ensure the model is trained and saved. Error: {e}"
            )
    
    def _compute_expected_mocu_matrix(self, M_lower, M_upper, K_lower, K_upper):
        """
        Compute expected MOCU matrix for all probe actions.
        
        Args:
            M_lower, M_upper, K_lower, K_upper: Current uncertainty bounds
        
        Returns:
            R_matrix: [N, len(probe_amplitudes)] matrix with expected MOCU for each probe
        """
        R_matrix = np.zeros((self.N, len(self.probe_amplitudes)))
        
        # For each probe action (b, A), compute expected MOCU
        for b in range(self.N):
            for a_idx, A in enumerate(self.probe_amplitudes):
                # Heuristic: Simulate bound update after probe
                M_lower_new, M_upper_new, K_lower_new, K_upper_new = \
                    self._simulate_probe_update(M_lower, M_upper, K_lower, K_upper, b, A)
                
                # Predict MOCU with Swing MLP
                mocu_pred = predict_swing_mocu(
                    self.model, self.mean, self.std,
                    M_lower_new, M_upper_new, K_lower_new, K_upper_new,
                    device=self.device
                )
                
                if isinstance(mocu_pred, torch.Tensor):
                    mocu_pred = mocu_pred.cpu().item()
                
                R_matrix[b, a_idx] = float(mocu_pred)
        
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
        Select next probe action using static NN strategy.
        
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
            print("[NN] Computing expected MOCU matrix (static, once only)...")
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
            print("[NN] Warning: No valid probe actions left!")
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
