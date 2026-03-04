"""
ODE-based OED Method for Swing Equation

Aligns with accelerateOED mocu_strategy: greedy selection minimizing expected MOCU.
Recomputes R matrix each step using current bounds and proper observation model.
"""

import time
import numpy as np
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods.base import OEDMethod

# Import swing equation MOCU and likelihood
try:
    from src.core.swing_equation_mocu import MOCU_swing_equation, get_mocu_swing_computer
    from src.core.swing_equation_params import get_default_swing_equation_params
    from src.core.likelihood import mu_theta_xi
    from scripts.data_generation.generate_dad_data import update_bounds_bayesian
    SWING_EQUATION_AVAILABLE = True
except ImportError as e:
    SWING_EQUATION_AVAILABLE = False
    mu_theta_xi = None
    update_bounds_bayesian = None
    print(f"[WARNING] Swing equation modules not available: {e}")

try:
    import torch
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False


class ODE_Method(OEDMethod):
    """
    ODE-based method: greedy selection minimizing expected MOCU (accelerateOED style).
    Uses TRUE MOCU (MOCU_swing_equation) throughout—no MPNN or other estimates.
    Order matters: E[MOCU|ξ] is recomputed each step from current belief (history);
    δ MOCU for a design ξ at step t differs from its value at step t' because the
    prior (belief state) differs.
    """

    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx,
                 B=None, P_m=None, D=None, g=None,
                 probe_amplitudes=None, probe_duration=2.0,
                 r_max=0.5, f_min=49.8, sigma=0.05,
                 reference_probe_bus=0, reference_probe_amplitude=0.5, reference_probe_duration=2.0,
                 M_lower_base=0.01, M_upper_base=0.06, K_lower_base=0.05, K_upper_base=0.50,
                 n_mc_samples=4, gpu_id=0):
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)

        self.probe_amplitudes = probe_amplitudes if probe_amplitudes else [0.5, 1.0, 2.0]
        self.probe_duration = probe_duration
        self.r_max = r_max
        self.f_min = f_min
        self.sigma = sigma
        self.reference_probe_bus = reference_probe_bus
        self.reference_probe_amplitude = reference_probe_amplitude
        self.reference_probe_duration = reference_probe_duration
        self.M_lower_base = M_lower_base
        self.M_upper_base = M_upper_base
        self.K_lower_base = K_lower_base
        self.K_upper_base = K_upper_base
        self.n_mc_samples = max(1, n_mc_samples)  # MC over y; 1=fast, 4+=accurate

        # System parameters
        if B is None or P_m is None or D is None or g is None:
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

        self.device = 'cuda' if (TORCHDIFFEQ_AVAILABLE and torch.cuda.is_available()) else 'cpu'

        print(f"[ODE] Initialized (iterative E[MOCU], n_mc={n_mc_samples}, device={self.device})")
    
    def _compute_expected_mocu_matrix(self, M_lower, M_upper, K_lower, K_upper, history):
        """
        Compute R[b,a] = E[MOCU | design (b,A)] using TRUE MOCU (no approximation).

        Full Monte Carlo: for each (b,A), sample (M,K)~prior, y~N(μ(M,K,ξ),σ²),
        compute posterior p(θ|history,y), compute TRUE MOCU via MOCU_swing_equation, average.
        Order matters: history encodes the sequence [ξ_1, ξ_2, ...]; posterior and thus
        δ MOCU for ξ at step t depends on the current belief state.
        """
        if not SWING_EQUATION_AVAILABLE or mu_theta_xi is None or update_bounds_bayesian is None:
            raise RuntimeError("Swing equation and likelihood modules required for ODE")

        R_matrix = np.zeros((self.N, len(self.probe_amplitudes)))

        for b in range(self.N):
            for a_idx, A in enumerate(self.probe_amplitudes):
                xi = (b + 1, float(A), float(self.probe_duration))
                probe_action = (b, float(A), self.probe_duration)
                mocu_sum = 0.0
                n_ok = 0
                for _ in range(self.n_mc_samples):
                    M_sample = np.random.uniform(M_lower, M_upper)
                    K_sample = np.random.uniform(K_lower, K_upper)
                    try:
                        mu = mu_theta_xi(
                            (M_sample, K_sample), xi,
                            self.B, self.P_m, self.D, self.g,
                            h=self.deltaT, T=self.TReal, M_steps=self.MReal,
                            device=self.device
                        )
                        y = float(mu + self.sigma * np.random.randn())
                    except Exception:
                        continue
                    obs_tuples = list(history) + [(probe_action, {'ROCOF_max': y})]
                    try:
                        Ml, Mu, Kl, Ku = update_bounds_bayesian(
                            M_lower, M_upper, K_lower, K_upper, obs_tuples,
                            self.M_lower_base, self.M_upper_base,
                            self.K_lower_base, self.K_upper_base,
                            B=self.B, P_m=self.P_m, D=self.D, g=self.g,
                            h=self.deltaT, T=self.TReal, M_steps=self.MReal,
                            sigma=self.sigma, n_particles=64, device=self.device
                        )
                        _mocu_fn, _ = get_mocu_swing_computer()
                        m = _mocu_fn(
                            K_max=self.K_max, B=self.B, P_m=self.P_m, D=self.D,
                            M_lower=Ml, M_upper=Mu, K_lower=Kl, K_upper=Ku,
                            g=self.g, r_max=self.r_max, f_min=self.f_min,
                            h=self.deltaT, T=self.TReal, M_steps=self.MReal,
                            reference_probe_bus=self.reference_probe_bus,
                            reference_probe_amplitude=self.reference_probe_amplitude,
                            reference_probe_duration=self.reference_probe_duration,
                            seed=np.random.randint(0, 2**31),
                            device=self.device
                        )
                        mocu_sum += float(m)
                        n_ok += 1
                    except Exception:
                        pass
                R_matrix[b, a_idx] = mocu_sum / n_ok if n_ok > 0 else np.inf

        return R_matrix
    
    def select_experiment(self, M_lower, M_upper, K_lower, K_upper, history,
                         probe_amplitudes=None, probe_duration=None):
        """
        Select probe action that minimizes expected MOCU (greedy, iterative).
        Recomputes R each step using current bounds and Bayesian observation model.
        """
        if probe_amplitudes is None:
            probe_amplitudes = self.probe_amplitudes
        if probe_duration is None:
            probe_duration = self.probe_duration

        R_matrix = self._compute_expected_mocu_matrix(
            M_lower, M_upper, K_lower, K_upper, history
        )

        # Mask already-selected designs
        for (probe_action, _) in history:
            if isinstance(probe_action, tuple) and len(probe_action) >= 2:
                b, A = probe_action[0], probe_action[1]
                if 0 <= b < self.N and A in probe_amplitudes:
                    a_idx = probe_amplitudes.index(A)
                    R_matrix[b, a_idx] = np.inf

        valid = np.isfinite(R_matrix)
        if not np.any(valid):
            return (0, probe_amplitudes[0], probe_duration)

        min_val = np.min(R_matrix[valid])
        idx = np.where(R_matrix == min_val)
        b_idx = int(idx[0][0])
        a_idx = int(idx[1][0])
        return (b_idx, probe_amplitudes[a_idx], probe_duration)
