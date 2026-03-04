"""
ENTROPY OED Method for Swing Equation

Aligns with accelerateOED entropy_strategy: pick design with maximum uncertainty.
In accelerateOED: pick edge (i,j) with max a_diff = a_upper - a_lower.
In swing: (M,K) bounds are global. Analogue: pick (bus, amplitude) used least
in history (max exploration / max entropy in design space).
"""

import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods.base import OEDMethod


class ENTROPY_Method(OEDMethod):
    """
    Entropy-based method for OED with swing equation.

    Swing analogue to accelerateOED entropy: pick (bus, amplitude) with
    minimum usage count in history (max exploration / diversity).
    When tied, prefers bus with higher degree (if B given) or lexicographic order.
    """

    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx,
                 probe_amplitudes=None, probe_duration=2.0, B=None):
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)
        self.probe_amplitudes = probe_amplitudes if probe_amplitudes else [0.5, 1.0, 2.0]
        self.probe_duration = probe_duration
        self.B = B
        print(f"[ENTROPY] Initialized (max-exploration selection, {len(self.probe_amplitudes)} amplitudes)")

    def select_experiment(self, M_lower, M_upper, K_lower, K_upper, history,
                         probe_amplitudes=None, probe_duration=None):
        """
        Pick (bus, amplitude) with minimum usage in history (max entropy/exploration).
        """
        if probe_amplitudes is None:
            probe_amplitudes = self.probe_amplitudes
        if probe_duration is None:
            probe_duration = self.probe_duration

        # Count usage of each (bus, amplitude)
        from collections import defaultdict
        counts = defaultdict(int)
        for (probe_action, _) in history:
            if isinstance(probe_action, tuple) and len(probe_action) >= 2:
                bus, amp = probe_action[0], probe_action[1]
                counts[(bus, amp)] += 1

        # All designs
        designs = [(b, A) for b in range(self.N) for A in probe_amplitudes]
        # Sort by (count, -degree, bus, amp) so we pick min-count, then max-degree bus, then lex
        degrees = np.sum(self.B > 0, axis=1) if self.B is not None else np.zeros(self.N)
        designs.sort(key=lambda da: (counts[da], -degrees[da[0]], da[0], da[1]))
        bus, amp = designs[0]

        return (bus, amp, probe_duration)
