"""One-step myopic selector minimizing expected posterior-safe u_ctrl."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.control.posterior_ctrl import normalize_log_weights, posterior_safe_u_ctrl
from src.contrastive.spce import log_gaussian_observation_density
from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables


@dataclass
class MyopicControlSelector:
    """
    J(ξ | h) = E_{y|h,ξ}[ u_ctrl(h, ξ, y) ].

    Draws ``n_hypothetical`` predictive observations from the mixture of Gaussians
    over support centres; no multi-step lookahead.
    """

    table_support: TableThetaSupport
    U_support: np.ndarray
    n_actions: int
    sigma_y: float
    alpha: float
    n_hypothetical: int = 32
    safety_margin: float = 0.0
    u_candidates: tuple[float, ...] = ()

    def select(
        self,
        *,
        used: set[int],
        log_weights: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
        **_kwargs,
    ) -> int:
        feasible = [a for a in range(self.n_actions) if a not in used]
        if not feasible:
            raise RuntimeError("no feasible actions for myopic")

        # Common random numbers across candidate actions (shared particle + noise draws).
        n_h = max(1, int(self.n_hypothetical))
        idx = rng.choice(len(weights), size=n_h, p=weights)
        noise = rng.normal(0.0, self.sigma_y, size=n_h)

        best_a = feasible[0]
        best_j = float("inf")
        self.last_scores: dict[int, float] = {}
        tie_count = 0
        near_tie_count = 0
        for a in feasible:
            j = self._expected_u_ctrl(a, log_weights, weights, idx=idx, noise=noise)
            self.last_scores[int(a)] = float(j)
        # Recompute ties relative to best after all scores known.
        ordered = sorted(self.last_scores, key=lambda x: (self.last_scores[x], x))
        best_a = ordered[0]
        best_j = float(self.last_scores[best_a])
        for a, j in self.last_scores.items():
            if a == best_a:
                continue
            if abs(j - best_j) <= 1e-15:
                tie_count += 1
            if abs(j - best_j) <= 1e-3:
                near_tie_count += 1
        self.last_selected = int(best_a)
        self.last_selected_score = float(best_j)
        self.last_tie_count = int(tie_count)
        self.last_near_tie_count = int(near_tie_count)
        self.last_candidate_count = int(len(feasible))
        self.last_tie_break_used = bool(tie_count > 0)
        self.last_score_gap = (
            float(self.last_scores[ordered[1]] - best_j) if len(ordered) > 1 else float("nan")
        )
        return int(best_a)

    def _expected_u_ctrl(
        self,
        action: int,
        log_weights: np.ndarray,
        weights: np.ndarray,
        *,
        idx: np.ndarray,
        noise: np.ndarray,
    ) -> float:
        centres = y_sim_last_step_from_tables(self.table_support, [action])
        # Predictive under shared CRN: y = centre[particle] + shared noise.
        y_draw = centres[idx] + noise

        # Vectorized hypothetical posterior updates for all draws.
        s2 = float(self.sigma_y) ** 2
        log_L = -0.5 * np.log(2.0 * np.pi * s2) - 0.5 * (
            (y_draw[:, None] - centres[None, :]) ** 2
        ) / s2
        log_w_h = log_weights[None, :] + log_L
        c = np.max(log_w_h, axis=1, keepdims=True)
        w_h = np.exp(log_w_h - c)
        w_h = w_h / np.clip(w_h.sum(axis=1, keepdims=True), 1e-300, None)

        u_vals = np.empty(len(idx), dtype=np.float64)
        for k in range(len(idx)):
            u_vals[k] = posterior_safe_u_ctrl(
                self.U_support,
                w_h[k],
                self.alpha,
                margin=self.safety_margin,
                u_grid=self.u_candidates if self.u_candidates else None,
            )
        return float(np.mean(u_vals))


def score_all_actions_myopic(
    selector: MyopicControlSelector,
    *,
    used: set[int],
    log_weights: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> dict[int, float]:
    """Debug helper: expected u_ctrl for every feasible action."""
    out: dict[int, float] = {}
    n_h = max(1, int(selector.n_hypothetical))
    idx = rng.choice(len(weights), size=n_h, p=weights)
    noise = rng.normal(0.0, selector.sigma_y, size=n_h)
    for a in range(selector.n_actions):
        if a in used:
            continue
        out[a] = selector._expected_u_ctrl(
            a, log_weights, weights, idx=idx, noise=noise
        )
    return out
