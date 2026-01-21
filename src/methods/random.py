"""
RANDOM OED Method for Swing Equation

Baseline method that selects probe actions uniformly at random.
Used as a comparison baseline to show the value of intelligent selection.
"""

import time
import numpy as np
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods.base import OEDMethod


class RANDOM_Method(OEDMethod):
    """
    Random baseline method for OED with swing equation.
    
    Selects probe actions uniformly at random from available options.
    This provides a lower bound on performance.
    """
    
    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx, 
                 probe_amplitudes=None, probe_duration=2.0, seed=None):
        """
        Args:
            N: Number of buses
            K_max: Number of Monte Carlo samples for MOCU
            deltaT: Time step
            MReal: Number of time steps
            TReal: Time horizon
            it_idx: Number of MOCU averaging iterations
            probe_amplitudes: List of probe amplitude options (default: [0.5, 1.0, 2.0])
            probe_duration: Probe duration T (default: 2.0s)
            seed: Random seed for reproducibility (optional)
        """
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)
        
        self.probe_amplitudes = probe_amplitudes if probe_amplitudes else [0.5, 1.0, 2.0]
        self.probe_duration = probe_duration
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Pre-generate random sequence of all possible probe actions
        self.experiment_sequence = []
        for b in range(N):
            for A in self.probe_amplitudes:
                self.experiment_sequence.append((b, A, probe_duration))
        
        random.shuffle(self.experiment_sequence)
        self.current_index = 0
        
        print(f"[RANDOM] Initialized (seed={seed}, {len(self.experiment_sequence)} probe actions)")
    
    def select_experiment(self, M_lower, M_upper, K_lower, K_upper, history,
                         probe_amplitudes=None, probe_duration=None):
        """
        Select next probe action randomly.
        
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
        
        if self.current_index >= len(self.experiment_sequence):
            # Fallback: random selection
            probe_bus = random.randint(0, self.N - 1)
            probe_amplitude = random.choice(probe_amplitudes)
            return (probe_bus, probe_amplitude, probe_duration)
        
        selected_action = self.experiment_sequence[self.current_index]
        self.current_index += 1
        
        return selected_action
