"""Configuration for control adaptive-advantage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
# Outputs live under reports/ so the package is usable without tests/.
SUITE_ROOT = REPO_ROOT / "reports" / "control_adaptive_advantage"
RESULTS_DIR = SUITE_ROOT / "results"
FIGURES_DIR = SUITE_ROOT / "figures"
REPORTS_DIR = SUITE_ROOT / "reports"

DEFAULT_SYSTEMS = ("ieee5", "ieee9", "ieee14")


@dataclass
class SuiteConfig:
    systems: tuple[str, ...] = DEFAULT_SYSTEMS
    horizon: int = 2
    noise_replicates: int = 64
    bootstrap_replicates: int = 2000
    seed: int = 20260722
    quick: bool = False
    # Belief / scoring support size (train subsample).
    support_size: int = 128
    # Held-out true-θ rollouts (test subsample).
    eval_size: int = 64
    # Hypothetical observations for one-step / T=2 expectations.
    n_hyp_y: int = 32
    # Exact T=2 search if n_actions <= this; else screen then reduced search.
    exact_design_threshold: int = 36
    # After screening, keep this many first-design / second-design candidates.
    screen_top_k: int = 16
    # Exhaustive Fixed when C(n,T) <= this (mirrors production default spirit).
    fixed_exhaustive_threshold: int = 30000
    # Mutual-information bins for design–control relevance.
    u_bins: int = 8
    y_bins: int = 12
    # Minimum continuation-value gap for *meaningful* branching / routing.
    # None => auto = 0.5 * min positive u_grid spacing.
    meaningful_gap_eps: float | None = None
    # Tolerance for planner sanity J_planning <= J_myopic.
    planner_sanity_tol: float = 1e-6

    def resolved(self) -> "SuiteConfig":
        if not self.quick:
            return self
        return SuiteConfig(
            systems=self.systems,
            horizon=self.horizon,
            noise_replicates=16,
            bootstrap_replicates=200,
            seed=self.seed,
            quick=True,
            support_size=64,
            eval_size=24,
            n_hyp_y=12,
            exact_design_threshold=self.exact_design_threshold,
            screen_top_k=8,
            fixed_exhaustive_threshold=self.fixed_exhaustive_threshold,
            u_bins=self.u_bins,
            y_bins=self.y_bins,
            meaningful_gap_eps=self.meaningful_gap_eps,
            planner_sanity_tol=self.planner_sanity_tol,
        )


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
