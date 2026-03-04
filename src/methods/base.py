"""
Base class for OED methods.

Provides common interface for all experimental design algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
import time
import os
from typing import Tuple, List, Dict, Any

# Import torch for device checks
# Lazy import torch to avoid initializing PyTorch CUDA unnecessarily
# Import will happen lazily when needed (for PyTorch-based methods)
torch = None


class OEDMethod(ABC):
    """
    Abstract base class for Optimal Experimental Design methods.
    
    All OED methods must implement select_experiment() which chooses
    the next experiment given current state.
    """
    
    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx):
        """
        Args:
            N: Number of oscillators
            K_max: Number of Monte Carlo samples for MOCU
            deltaT: Time step
            MReal: Number of time steps for MOCU evaluation
            TReal: Time horizon for MOCU evaluation
            it_idx: Number of MOCU averaging iterations
        """
        self.N = N
        self.K_max = K_max
        self.deltaT = deltaT
        self.MReal = MReal
        self.TReal = TReal
        self.it_idx = it_idx
        self.name = self.__class__.__name__
    
    @abstractmethod
    def select_experiment(self, M_lower, M_upper, K_lower, K_upper, history,
                         probe_amplitudes=None, probe_duration=None):
        """
        Select next probe action to perform.
        
        Args:
            M_lower, M_upper, K_lower, K_upper: Current uncertainty bounds (scalars)
            history: List of (probe_action, observation) tuples
            probe_amplitudes: List of probe amplitude options (optional)
            probe_duration: Probe duration T (optional)
        
        Returns:
            (probe_bus, probe_amplitude, probe_duration): Selected probe action
        """
        pass
    
    def run_sequential_design(self, 
                             initial_state: Dict[str, Any],
                             ground_truth: Dict[str, Any],
                             num_iterations: int,
                             mocu_computer) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        Run full sequential experimental design process.
        
        Args:
            initial_state: Initial state with uncertainty bounds
            ground_truth: Ground truth for simulating observations
                - 'a_true': True coupling strengths
                - 'is_synchronized': Synchronization matrix
                - 'critical_k': Critical coupling thresholds
            num_iterations: Number of experiments to perform
            mocu_computer: Function to compute ground truth MOCU
        
        Returns:
            mocu_curve: MOCU values at each iteration [num_iterations+1]
            sequence: List of selected (i, j) pairs
            times: Computation time per iteration [num_iterations]
        """
        N = len(initial_state['w'])
        
        # Initialize
        mocu_curve = np.zeros(num_iterations + 1)
        sequence = []
        times = np.zeros(num_iterations)
        
        # Compute initial MOCU
        mocu_curve[0] = mocu_computer(
            initial_state['w'],
            initial_state['a_lower'],
            initial_state['a_upper']
        )
        
        # Current state
        state = {
            'w': initial_state['w'].copy(),
            'a_lower': initial_state['a_lower'].copy(),
            'a_upper': initial_state['a_upper'].copy(),
            'history': []
        }
        
        observed_pairs = set()
        
        # Sequential experimental design
        for iteration in range(num_iterations):
            start_time = time.time()
            
            # Get available pairs
            available_pairs = [
                (i, j) for i in range(N) for j in range(i+1, N)
                if (i, j) not in observed_pairs
            ]
            
            # Select experiment
            (i_sel, j_sel), info = self.select_experiment(state, available_pairs)
            
            # Simulate observation
            observation = int(ground_truth['is_synchronized'][i_sel, j_sel])
            f_critical = ground_truth['critical_k'][i_sel, j_sel]
            
            # Update bounds
            if observation == 0:  # Not synchronized
                state['a_upper'][i_sel, j_sel] = min(state['a_upper'][i_sel, j_sel], f_critical)
                state['a_upper'][j_sel, i_sel] = state['a_upper'][i_sel, j_sel]
            else:  # Synchronized
                state['a_lower'][i_sel, j_sel] = max(state['a_lower'][i_sel, j_sel], f_critical)
                state['a_lower'][j_sel, i_sel] = state['a_lower'][i_sel, j_sel]
            
            # Update history
            state['history'].append((i_sel, j_sel, observation))
            observed_pairs.add((i_sel, j_sel))
            
            # Record
            sequence.append((i_sel, j_sel))
            times[iteration] = time.time() - start_time
            
            # Compute MOCU
            mocu_new = mocu_computer(state['w'], state['a_lower'], state['a_upper'])
            
            # Ensure monotonicity (MOCU should decrease or stay same)
            mocu_curve[iteration + 1] = min(mocu_new, mocu_curve[iteration])
        
        return mocu_curve, sequence, times
    
    def run_episode(self, M_lower_init, M_upper_init, K_lower_init, K_upper_init,
                    M_true, K_true, B, P_m, D, g, probe_amplitudes, probe_duration,
                    r_max=0.1, f_min=49.8, update_cnt=10, initial_mocu=None,
                    reference_probe_bus=None, reference_probe_amplitude=None, reference_probe_duration=2.0,
                    sigma=0.0, update_strength=0.05):
        """
        Run a complete OED episode for swing equation model.
        
        This is the main entry point for evaluation. It runs the sequential
        experimental design process and tracks:
        - MOCU curve over iterations
        - Selected probe action sequence  
        - Time complexity per iteration
        
        Args:
            M_lower_init, M_upper_init, K_lower_init, K_upper_init: Initial uncertainty bounds (scalars)
            M_true, K_true: True parameters (unknown to agent, used for simulation)
            B: Coupling matrix [N, N] (fixed, known)
            P_m: Mechanical power [N] (fixed, known)
            D: Damping coefficient (fixed, known)
            g: Control allocation [N] (fixed, known)
            probe_amplitudes: List of probe amplitude options
            probe_duration: Probe duration T (seconds)
            r_max: Maximum ROCOF constraint (Hz/s, default 0.1, design §3)
            f_min: Minimum frequency constraint (Hz, default 49.8 for 50 Hz nominal)
            update_cnt: Number of experiments to perform
            initial_mocu: Pre-computed initial MOCU (optional)
        
        Returns:
            MOCUCurve: MOCU values at each step [update_cnt+1]
            experimentSequence: List of (probe_bus, probe_amplitude, probe_duration) tuples
            timeComplexity: Time per iteration [update_cnt]
        """
        
        # Declare torch as global at the start of the function
        global torch
        
        N = len(P_m)
        # Initialize MOCUCurve - will be filled by computation
        MOCUCurve = np.zeros(update_cnt + 1)
        experimentSequence = []
        timeComplexity = np.zeros(update_cnt)
        history = []
        
        # Compute initial MOCU
        # initial_mocu is passed from evaluate.py (computed with swing equation MOCU)
        # Use it to avoid redundant computation
        if initial_mocu is not None:
            MOCUCurve[0] = max(float(initial_mocu), 1e-10)
            self._last_valid_mocu = MOCUCurve[0]
        else:
            # Fallback: Compute initial MOCU if not provided
            try:
                if torch is None:
                    try:
                        import torch as _torch
                        torch = _torch
                    except ImportError:
                        torch = None
                
                device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
                
                from ..core.swing_equation_mocu import get_mocu_swing_computer
                _mocu_fn, _ = get_mocu_swing_computer()
                it_temp_val = np.zeros(self.it_idx)
                for l in range(self.it_idx):
                    it_temp_val[l] = _mocu_fn(
                        K_max=self.K_max,
                        B=B,
                        P_m=P_m,
                        D=D,
                        M_lower=M_lower_init,
                        M_upper=M_upper_init,
                        K_lower=K_lower_init,
                        K_upper=K_upper_init,
                        g=g,
                        r_max=r_max,
                        f_min=f_min,
                        h=self.deltaT,
                        T=self.TReal,
                        M_steps=self.MReal,
                        reference_probe_bus=reference_probe_bus,
                        reference_probe_amplitude=reference_probe_amplitude,
                        reference_probe_duration=reference_probe_duration,
                        seed=l,
                        device=device
                    )
                MOCUCurve[0] = max(np.mean(it_temp_val), 1e-10)
                self._last_valid_mocu = MOCUCurve[0]
            except Exception as e:
                if not hasattr(self, '_mocu_warned'):
                    print(f"[WARNING] Failed to compute initial MOCU: {e}")
                    self._mocu_warned = True
                # Do not use 0 (would be misleading). Use small positive fallback so curve is valid.
                MOCUCurve[0] = 1e-6
                self._last_valid_mocu = 1e-6
        
        # Sequential experimental design
        method_name = self.__class__.__name__
        
        # Import torch for PyTorch-based methods
        if torch is None:
            try:
                import torch as _torch
                globals()['torch'] = _torch
            except ImportError:
                torch = None
        
        device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
        
        # Import swing equation functions
        try:
            from scripts.data_generation.generate_dad_data import (
                perform_probe_experiment,
                update_bounds_bayesian,
            )
        except ImportError as e:
            raise ImportError(f"Failed to import swing equation functions: {e}")
        
        # Get base bounds from config (for update_bounds)
        # These are the maximum possible bounds
        M_lower_base = M_lower_init  # Use initial as base (could be from config)
        M_upper_base = M_upper_init
        K_lower_base = K_lower_init
        K_upper_base = K_upper_init
        
        # Current uncertainty bounds
        M_lower_current = M_lower_init
        M_upper_current = M_upper_init
        K_lower_current = K_lower_init
        K_upper_current = K_upper_init
        
        # Time parameters for probe experiments
        h = self.deltaT
        T = self.TReal
        M_steps = self.MReal
        
        for iteration in range(update_cnt):
            iterationStartTime = time.time()
            
            # Select probe action using the method's specific logic
            probe_action = self.select_experiment(
                M_lower_current, M_upper_current, K_lower_current, K_upper_current,
                history,
                probe_amplitudes=probe_amplitudes,
                probe_duration=probe_duration
            )
            
            if not isinstance(probe_action, tuple) or len(probe_action) < 2:
                print(f"[WARNING] Invalid probe action: {probe_action}, using default")
                probe_bus = 0
                probe_amplitude = probe_amplitudes[0] if probe_amplitudes else 1.0
            else:
                probe_bus, probe_amplitude = probe_action[0], probe_action[1]
                if len(probe_action) >= 3:
                    probe_duration = probe_action[2]
            
            # Ensure valid probe bus
            if probe_bus < 0 or probe_bus >= N:
                probe_bus = 0
            
            # Perform probe experiment (simulate observation using true parameters)
            try:
                observation = perform_probe_experiment(
                    B, P_m, D, M_true, K_true, g,
                    probe_bus, probe_amplitude, probe_duration,
                    h, T, M_steps, device=device, sigma=sigma
                )
            except TypeError:
                # Backward compatibility: older perform_probe_experiment may not accept sigma
                observation = perform_probe_experiment(
                    B, P_m, D, M_true, K_true, g,
                    probe_bus, probe_amplitude, probe_duration,
                    h, T, M_steps, device=device
                )
            
            iterationTime = time.time() - iterationStartTime
            timeComplexity[iteration] = iterationTime
            
            experimentSequence.append((probe_bus, probe_amplitude, probe_duration))
            history.append(((probe_bus, probe_amplitude, probe_duration), observation))

            # Bayesian update: posterior over (M,K) from full history so order matters
            (M_lower_current, M_upper_current, K_lower_current, K_upper_current) = \
                update_bounds_bayesian(
                    M_lower_current, M_upper_current, K_lower_current, K_upper_current,
                    history, M_lower_base, M_upper_base, K_lower_base, K_upper_base,
                    B=B, P_m=P_m, D=D, g=g, h=self.deltaT, T=self.TReal, M_steps=self.MReal,
                    sigma=sigma, n_particles=128, device=device
                )
            
            # Re-compute MOCU for the updated bounds (PyCUDA if available, else PyTorch)
            try:
                from ..core.swing_equation_mocu import get_mocu_swing_computer
                _mocu_fn, _ = get_mocu_swing_computer()
                it_temp_val = np.zeros(self.it_idx)
                for l in range(self.it_idx):
                    it_temp_val[l] = _mocu_fn(
                        K_max=self.K_max,
                        B=B,
                        P_m=P_m,
                        D=D,
                        M_lower=M_lower_current,
                        M_upper=M_upper_current,
                        K_lower=K_lower_current,
                        K_upper=K_upper_current,
                        g=g,
                        r_max=r_max,
                        f_min=f_min,
                        h=self.deltaT,
                        T=self.TReal,
                        M_steps=self.MReal,
                        reference_probe_bus=reference_probe_bus,
                        reference_probe_amplitude=reference_probe_amplitude,
                        reference_probe_duration=reference_probe_duration,
                        seed=l,
                        device=device
                    )
                raw_mocu = np.mean(it_temp_val)
                MOCUCurve[iteration + 1] = max(float(raw_mocu), 1e-10)
                self._last_valid_mocu = MOCUCurve[iteration + 1]
                # MOCU must never increase as steps progress
                if MOCUCurve[iteration + 1] > MOCUCurve[iteration]:
                    MOCUCurve[iteration + 1] = MOCUCurve[iteration]
                    
            except ImportError:
                # MOCU computation not available
                if not hasattr(self, '_mocu_import_warned'):
                    print(f"[{method_name}] Warning: Swing equation MOCU not available (ImportError)")
                    self._mocu_import_warned = True
                # Use last valid MOCU or previous value
                if hasattr(self, '_last_valid_mocu') and self._last_valid_mocu is not None:
                    MOCUCurve[iteration + 1] = self._last_valid_mocu
                else:
                    MOCUCurve[iteration + 1] = MOCUCurve[iteration]
                    
            except Exception as e:
                # MOCU computation failed
                if not hasattr(self, '_mocu_iter_warned'):
                    print(f"[{method_name}] ERROR: MOCU computation failed (iteration {iteration+1}): {type(e).__name__}: {e}")
                    self._mocu_iter_warned = True
                # Use last valid MOCU or previous value
                if hasattr(self, '_last_valid_mocu') and self._last_valid_mocu is not None:
                    MOCUCurve[iteration + 1] = self._last_valid_mocu
                else:
                    MOCUCurve[iteration + 1] = MOCUCurve[iteration]
        
        return MOCUCurve, experimentSequence, timeComplexity
    
    def get_name(self) -> str:
        """Return method name."""
        return self.name

