"""
DAD-MOCU Method (Deep Adaptive Design for MOCU) - Swing Equation

Uses a trained policy network to sequentially select probe experiments
that minimize terminal MOCU. The policy is learned via REINFORCE 
(reinforcement learning) that directly optimizes terminal MOCU as the loss.

Training: REINFORCE policy gradient with terminal MOCU as reward signal.
This directly optimizes the true objective (minimize terminal MOCU) rather
than mimicking a suboptimal expert policy.

In the paper: "DAD" method (proposed)
"""

import time
import numpy as np
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods.base import OEDMethod


class DAD_MOCU_Method(OEDMethod):
    """
    Deep Adaptive Design method with MOCU objective for swing equation.
    
    Uses a learned policy network to make sequential experimental selections.
    The policy is trained to minimize terminal MOCU over K steps.
    
    This is analogous to the original DAD paper (Foster et al. 2021) but with
    MOCU as the objective instead of Expected Information Gain (EIG).
    """
    
    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx,
                 policy_model_path=None, probe_amplitudes=None, probe_duration=2.0,
                 gpu_id=0, B=None):
        """
        Args:
            N: Number of buses
            K_max: Number of Monte Carlo samples for MOCU
            deltaT: Time step
            MReal: Number of time steps
            TReal: Time horizon
            it_idx: Number of MOCU averaging iterations
            policy_model_path: Path to trained policy checkpoint (.pth)
            probe_amplitudes: List of probe amplitude options (default: [0.5, 1.0, 2.0])
            probe_duration: Probe duration T (default: 2.0s)
            gpu_id: GPU device ID
            B: Coupling matrix [N,N] for MPNN MOCU predictor (optional; built from default params if None)
        """
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)
        
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.policy_net = None
        self.mocu_predictor = None
        self.mocu_mean = None
        self.mocu_std = None
        self.use_expected_mocu = True
        
        self.probe_amplitudes = probe_amplitudes if probe_amplitudes else [0.5, 1.0, 2.0]
        self.probe_duration = probe_duration
        self.B = B
        
        self._load_policy(policy_model_path)
        self._load_mocu_predictor()
        
        print(f"[DAD-MOCU] Initialized with policy on {self.device}")
        if self.mocu_predictor is not None:
            print(f"[DAD-MOCU] MPNN MOCU predictor loaded - will use expected MOCU features (enhanced mode)")
        else:
            print(f"[DAD-MOCU] MPNN MOCU predictor not available - using standard mode")
    
    def _load_policy(self, policy_model_path):
        """Load trained DAD policy network."""
        # For now, policy network loading is deferred until we have swing equation policy networks
        # This is a placeholder that will be updated when policy networks are retrained for swing equation
        if policy_model_path is None:
            # Try to find policy in new structure
            models_root = PROJECT_ROOT / 'models'
            found_paths = []
            
            if models_root.exists():
                for config_dir in models_root.iterdir():
                    if config_dir.is_dir():
                        # Try pattern with K: dad_policy_N{N}_K*.pth
                        for k_file in config_dir.glob(f'dad_policy_N{self.N}_K*.pth'):
                            if '_best.pth' in k_file.name:
                                found_paths.append((k_file, k_file.stat().st_mtime, True))
                            else:
                                found_paths.append((k_file, k_file.stat().st_mtime, False))
                        
                        # Try pattern without K: dad_policy_N{N}.pth
                        candidate = config_dir / f'dad_policy_N{self.N}.pth'
                        if candidate.exists():
                            found_paths.append((candidate, candidate.stat().st_mtime, False))
            
            if found_paths:
                found_paths.sort(key=lambda x: (not x[2], -x[1]))
                policy_model_path = found_paths[0][0]
                print(f"[DAD-MOCU] Found policy at: {policy_model_path}")
            else:
                # Policy not found - DAD will use random selection as fallback
                print(f"[DAD-MOCU] Warning: Policy model not found. DAD will use random selection.")
                print(f"[DAD-MOCU] Please train a DAD policy first using:")
                print(f"[DAD-MOCU]   python scripts/train_dad_policy.py --data-path <data> --name dad_policy_N{self.N}_K<K>")
                self.policy_net = None
                return
        
        print(f"[DAD-MOCU] Loading policy from: {policy_model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(policy_model_path, map_location=self.device)
            model_config = checkpoint['config']
            
            # Check if this is a swing equation policy
            if model_config.get('model_type') != 'swing_equation':
                print(f"[DAD-MOCU] Warning: Policy was trained for first-order model, not swing equation.")
                print(f"[DAD-MOCU] Policy may not work correctly. Please retrain for swing equation.")
            
            # Import policy network (will need swing equation version)
            from src.models.policy_networks import DADPolicyNetworkSwing
            
            # Create model
            self.policy_net = DADPolicyNetworkSwing(
                N=model_config['N'],
                num_probe_amplitudes=len(self.probe_amplitudes),
                hidden_dim=model_config.get('hidden_dim', 64),
                encoding_dim=model_config.get('encoding_dim', 32),
                num_message_passing=model_config.get('num_message_passing', 3)
            )
            
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.policy_net.to(self.device)
            self.policy_net.eval()
            
            print(f"[DAD-MOCU] Policy loaded successfully")
        except Exception as e:
            print(f"[DAD-MOCU] Error loading policy: {e}")
            print(f"[DAD-MOCU] DAD will use random selection as fallback.")
            self.policy_net = None
    
    def _load_mocu_predictor(self):
        """Load MPNN MOCU predictor for expected MOCU features (like iNN/NN)."""
        try:
            from src.models.predictors.swing_predictor_utils import load_swing_mocu_predictor
            from src.core.swing_equation_params import get_default_swing_equation_params
            import os
            
            if self.B is None:
                params = get_default_swing_equation_params(N=self.N)
                self.B = params['B']
            model_name = os.getenv('MOCU_MODEL_NAME', f'cons{self.N}')
            
            self.mocu_predictor, self.mocu_mean, self.mocu_std = load_swing_mocu_predictor(
                model_name=model_name, device=str(self.device), B=self.B, N=self.N
            )
            if self.mocu_predictor is not None:
                self.mocu_predictor.eval()
                self.mocu_predictor = self.mocu_predictor.to(self.device)
        except Exception as e:
            print(f"[DAD-MOCU] Warning: Could not load MPNN MOCU predictor: {e}")
            self.mocu_predictor = None
            self.mocu_mean = None
            self.mocu_std = None
    
    def _compute_expected_mocu_matrix(self, M_lower, M_upper, K_lower, K_upper):
        """
        Compute R matrix (expected remaining MOCU) for all possible probe designs.
        
        This is the same computation that iNN/NN uses - gives DAD the same information.
        """
        if self.mocu_predictor is None:
            return None
        
        R_matrix = np.zeros((self.N, len(self.probe_amplitudes)))
        
        try:
            from src.models.predictors.swing_predictor_utils import predict_swing_mocu
            
            for bus_idx in range(self.N):
                for amp_idx, probe_amplitude in enumerate(self.probe_amplitudes):
                    M_lower_new, M_upper_new, K_lower_new, K_upper_new = \
                        self._simulate_probe_update(M_lower, M_upper, K_lower, K_upper, bus_idx, probe_amplitude)
                    
                    mocu_pred = predict_swing_mocu(
                        self.mocu_predictor, self.mocu_mean, self.mocu_std,
                        M_lower_new, M_upper_new, K_lower_new, K_upper_new,
                        device=str(self.device),
                        probe_bus=bus_idx, probe_amplitude=probe_amplitude,
                    )
                    
                    if hasattr(mocu_pred, 'item'):
                        mocu_pred = mocu_pred.item()
                    
                    R_matrix[bus_idx, amp_idx] = float(mocu_pred)
        except Exception as e:
            print(f"[DAD-MOCU] Warning: Failed to compute expected MOCU matrix: {e}")
            return None
        
        return R_matrix
    
    def _simulate_probe_update(self, M_lower, M_upper, K_lower, K_upper, probe_bus, probe_amplitude):
        """
        Simulate bound update after a probe design (heuristic).
        
        This is a simplified version - in practice, we'd need to simulate
        the actual observation and update bounds accordingly.
        """
        # Heuristic: reduce uncertainty by a small amount
        # In practice, this would be based on the actual observation
        reduction_factor = 0.1  # Reduce uncertainty by 10%
        
        M_range = M_upper - M_lower
        K_range = K_upper - K_lower
        
        M_lower_new = M_lower + reduction_factor * M_range * 0.5
        M_upper_new = M_upper - reduction_factor * M_range * 0.5
        K_lower_new = K_lower + reduction_factor * K_range * 0.5
        K_upper_new = K_upper - reduction_factor * K_range * 0.5
        
        # Ensure valid bounds
        M_lower_new = max(M_lower, M_lower_new)
        M_upper_new = min(M_upper, M_upper_new)
        K_lower_new = max(K_lower, K_lower_new)
        K_upper_new = min(K_upper, K_upper_new)
        
        return M_lower_new, M_upper_new, K_lower_new, K_upper_new
    
    def select_experiment(self, M_lower, M_upper, K_lower, K_upper, history,
                          probe_amplitudes=None, probe_duration=None):
        """
        Select next probe design using learned DAD policy.
        
        The policy network takes the current state (M, K bounds, history)
        and outputs a probability distribution over available probe designs (ξ).
        We select the design with highest probability (greedy/deterministic).
        
        Args:
            M_lower, M_upper, K_lower, K_upper: Current uncertainty bounds (scalars)
            history: List of ((probe_bus, probe_amplitude, probe_duration), observation) tuples
            probe_amplitudes: Probe amplitude options (optional, uses self.probe_amplitudes)
            probe_duration: Probe duration (optional, uses self.probe_duration)
        
        Returns:
            (probe_bus, probe_amplitude, probe_duration): Selected probe design ξ
        """
        if probe_amplitudes is None:
            probe_amplitudes = self.probe_amplitudes
        if probe_duration is None:
            probe_duration = self.probe_duration
        
        # If no policy network, use random selection as fallback
        if self.policy_net is None:
            import random
            probe_bus = random.randint(0, self.N - 1)
            probe_amplitude = random.choice(probe_amplitudes)
            return (probe_bus, probe_amplitude, probe_duration)
        
        # Convert history format: from [((bus, amp, dur), obs), ...] to [(bus, amp, obs), ...]
        history_list = []
        observed_designs = set()
        if history:
            for xi_obs in history:
                if isinstance(xi_obs, tuple) and len(xi_obs) == 2:
                    xi, obs = xi_obs
                    if isinstance(xi, tuple) and len(xi) >= 2:
                        bus, amp = xi[0], xi[1]
                        # Convert observation to scalar (use ROCOF_max if available)
                        if isinstance(obs, dict):
                            obs_scalar = obs.get('ROCOF_max', 0.0)
                        else:
                            obs_scalar = float(obs) if isinstance(obs, (int, float)) else 0.0
                        history_list.append((bus, amp, obs_scalar))
                        observed_designs.add((bus, amp))
        
        # Create state data for policy network (swing equation version)
        try:
            from src.models.policy_networks import create_swing_state_data
            state_data = create_swing_state_data(
                M_lower, M_upper, K_lower, K_upper, self.N, device=self.device
            )
        except ImportError:
            # Fallback: create simple state representation
            state_data = {
                'M_lower': torch.tensor([M_lower], dtype=torch.float32, device=self.device),
                'M_upper': torch.tensor([M_upper], dtype=torch.float32, device=self.device),
                'K_lower': torch.tensor([K_lower], dtype=torch.float32, device=self.device),
                'K_upper': torch.tensor([K_upper], dtype=torch.float32, device=self.device),
            }
        
        # Get available designs (not yet observed)
        available_designs = []
        for bus_idx in range(self.N):
            for amp_idx, amp in enumerate(probe_amplitudes):
                if (bus_idx, amp) not in observed_designs:
                    available_designs.append((bus_idx, amp_idx))
        
        if not available_designs:
            print("[DAD-MOCU] Warning: No available probe designs left!")
            return (0, probe_amplitudes[0], probe_duration)
        
        # Create available designs mask
        num_designs = self.N * len(probe_amplitudes)
        available_mask = np.zeros(num_designs, dtype=np.float32)
        for bus_idx, amp_idx in available_designs:
            xi_idx = bus_idx * len(probe_amplitudes) + amp_idx
            available_mask[xi_idx] = 1.0
        
        available_mask_tensor = torch.tensor([available_mask], dtype=torch.float32, device=self.device)
        
        # Convert history to tensor format
        if len(history_list) == 0:
            history_tensor = None
        else:
            history_tensor = torch.tensor([history_list], dtype=torch.float32, device=self.device)
        
        # Compute expected MOCU features (like iNN/NN) if MPNN predictor is available
        expected_mocu_features = None
        if self.use_expected_mocu and self.mocu_predictor is not None:
            try:
                R_matrix = self._compute_expected_mocu_matrix(M_lower, M_upper, K_lower, K_upper)
                if R_matrix is not None:
                    # Extract expected MOCU values for available designs
                    expected_mocu_values = []
                    for bus_idx, amp_idx in available_designs:
                        expected_mocu_values.append(R_matrix[bus_idx, amp_idx])
                    
                    # Convert to tensor: [1, num_designs]
                    num_designs = self.N * len(probe_amplitudes)
                    expected_mocu_array = np.zeros(num_designs, dtype=np.float32)
                    for idx, (bus_idx, amp_idx) in enumerate(available_designs):
                        xi_idx = bus_idx * len(probe_amplitudes) + amp_idx
                        expected_mocu_array[xi_idx] = expected_mocu_values[idx]
                    
                    # Normalize expected MOCU features
                    if len(expected_mocu_values) > 0:
                        mocu_min = min(expected_mocu_values)
                        mocu_max = max(expected_mocu_values)
                        mocu_range = mocu_max - mocu_min
                        if mocu_range > 1e-6:
                            expected_mocu_array = (expected_mocu_array - mocu_min) / mocu_range
                    
                    expected_mocu_features = torch.tensor([expected_mocu_array], dtype=torch.float32, device=self.device)
            except Exception as e:
                print(f"[DAD-MOCU] Warning: Failed to compute expected MOCU features: {e}")
                expected_mocu_features = None
        
        # Get policy ξ (design) probabilities (greedy/deterministic)
        with torch.no_grad():
            self.policy_net.eval()
            xi_logits, xi_probs = self.policy_net(
                state_data, history_tensor, available_mask_tensor, 
                expected_mocu_features=expected_mocu_features
            )
        
        # Select ξ with highest probability (deterministic)
        xi_probs = xi_probs.squeeze(0)  # [num_designs]
        xi_idx = torch.argmax(xi_probs).item()
        
        # Convert ξ index to (bus, amplitude)
        bus_idx = xi_idx // len(probe_amplitudes)
        amp_idx = xi_idx % len(probe_amplitudes)
        probe_bus = bus_idx
        probe_amplitude = probe_amplitudes[amp_idx]
        
        return (probe_bus, probe_amplitude, probe_duration)
