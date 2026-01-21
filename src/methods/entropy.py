"""
ENTROPY OED Method for Swing Equation

Simple heuristic-based method that selects probe actions based on uncertainty.
Selects probe at bus with maximum uncertainty in (M, K) bounds.

This method does NOT use any prediction model - it's purely based on current bounds.
"""

import time
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods.base import OEDMethod


class ENTROPY_Method(OEDMethod):
    """
    Entropy-based (uncertainty-based) method for OED with swing equation.
    
    Greedy heuristic: select probe at bus with maximum uncertainty.
    Uses degree-based selection (probe buses with highest connectivity).
    
    This is the fastest method but not necessarily the most effective.
    """
    
    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx,
                 probe_amplitudes=None, probe_duration=2.0, B=None):
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
            B: Coupling matrix [N, N] (optional, for degree-based selection)
        """
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)
        self.probe_amplitudes = probe_amplitudes if probe_amplitudes else [0.5, 1.0, 2.0]
        self.probe_duration = probe_duration
        self.B = B  # Coupling matrix for degree-based selection
        print(f"[ENTROPY] Initialized (degree-based probe selection)")
    
    def select_experiment(self, M_lower, M_upper, K_lower, K_upper, history,
                         probe_amplitudes=None, probe_duration=None):
        """
        Select next probe action using entropy (uncertainty) strategy.
        
        Selects bus with maximum uncertainty or highest degree.
        
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
        
        # Compute uncertainty: (M_upper - M_lower) + (K_upper - K_lower)
        uncertainty = (M_upper - M_lower) + (K_upper - K_lower)
        
        # Select bus: use degree-based if B is available, otherwise random
        if self.B is not None:
            # Degree-based: probe bus with highest connectivity
            degrees = np.sum(self.B > 0, axis=1)  # Count connections
            # Mask out already probed buses
            probed_buses = set()
            for (probe_action, _) in history:
                if isinstance(probe_action, tuple) and len(probe_action) > 0:
                    probed_buses.add(probe_action[0])
            
            # Select bus with highest degree that hasn't been probed
            available_buses = [b for b in range(self.N) if b not in probed_buses]
            if available_buses:
                bus_degrees = [(b, degrees[b]) for b in available_buses]
                bus_degrees.sort(key=lambda x: x[1], reverse=True)
                probe_bus = bus_degrees[0][0]
            else:
                probe_bus = np.argmax(degrees)  # All probed, use max degree
        else:
            # Random bus selection
            import random
            probe_bus = random.randint(0, self.N - 1)
        
        # Select amplitude: use middle value (or random)
        probe_amplitude = probe_amplitudes[len(probe_amplitudes) // 2] if len(probe_amplitudes) > 0 else probe_amplitudes[0]
        
        print(f"[ENTROPY] Selected probe bus {probe_bus}, A={probe_amplitude}, uncertainty={uncertainty:.4f}")
        
        return (probe_bus, probe_amplitude, probe_duration)
