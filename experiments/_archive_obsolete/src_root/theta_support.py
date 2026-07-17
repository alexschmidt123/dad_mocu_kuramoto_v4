"""Monte Carlo discrete support for per-bus θ = (M_{1:N}, K_{1:N})."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import SBOEDConfig
from src.contrastive.spce import (
    log_prior_uniform_discrete,
    predict_f_sequence,
    sample_mk_prior,
)


@dataclass
class ThetaMCSupport:
    """
    Uniform prior on ``N`` independent θ draws, each with shape ``(n_buses,)`` for M and K.

    Used for eval ΔH, myopic scoring, and posterior means over 28-D θ (not scalar grid).
    """

    M: np.ndarray  # (N, n_buses)
    K: np.ndarray
    log_p0: np.ndarray

    def __len__(self) -> int:
        return int(self.M.shape[0])

    @classmethod
    def build(cls, cfg: SBOEDConfig, rng: np.random.Generator) -> ThetaMCSupport:
        n = int(cfg.prior.get("mc_samples", 128))
        sw = cfg.swing
        M, K = sample_mk_prior(
            float(sw["M_lower"]),
            float(sw["M_upper"]),
            float(sw["K_lower"]),
            float(sw["K_upper"]),
            n,
            rng,
            n_buses=cfg.N,
        )
        return cls(M=M, K=K, log_p0=log_prior_uniform_discrete(n))

    def f_steps(self, sim, catalog, sequence: list[int]) -> np.ndarray:
        """Reset-based noiseless F; shape ``(T, N)`` for posterior updates."""
        seq = [int(a) for a in sequence]
        n, t_len = len(self.M), len(seq)
        out = np.zeros((t_len, n), dtype=np.float64)
        for i in range(n):
            out[:, i] = predict_f_sequence(sim, self.M[i], self.K[i], catalog, seq)
        return out

    def one_step_f(self, sim, catalog, action: int) -> np.ndarray:
        """Equilibrium one-step F(θ_n, ξ) ``(N,)`` for myopic ΔH scoring."""
        a = int(action)
        f_vals = np.empty(len(self.M), dtype=np.float64)
        for i in range(len(self.M)):
            f_vals[i] = float(
                predict_f_sequence(sim, self.M[i], self.K[i], catalog, [a])[0]
            )
        return f_vals
