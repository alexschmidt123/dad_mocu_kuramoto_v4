"""Shared helpers for consistency audit (existing banks only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.adaptive_advantage.config import SuiteConfig
from tools.adaptive_advantage.loaders import SystemBank, load_system_bank
from tools.adaptive_advantage.planning_utils import CRNBundle, make_crn_bundle
from tools.adaptive_advantage.posterior_utils import (
    terminal_u,
    uniform_log_prior,
    update_log_weights,
    weights_from_log,
)
from tools.adaptive_advantage.statistics_utils import paired_delta_ci

AUDIT_ROOT = Path(__file__).resolve().parent
AUDIT_RESULTS = AUDIT_ROOT / "results"
AUDIT_REPORTS = AUDIT_ROOT / "reports"


def ensure_audit_dirs() -> None:
    AUDIT_RESULTS.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORTS.mkdir(parents=True, exist_ok=True)


def load_audit_bank(system: str, cfg: SuiteConfig) -> SystemBank:
    return load_system_bank(
        system,
        support_size=cfg.support_size,
        eval_size=cfg.eval_size,
        support_seed=cfg.seed,
        eval_seed=cfg.seed + 1,
    )


def make_audit_crn(bank: SystemBank, cfg: SuiteConfig) -> CRNBundle:
    return make_crn_bundle(
        n_eval=bank.n_eval,
        n_rep=cfg.noise_replicates,
        horizon=2,
        n_support=bank.n_support,
        n_hyp=cfg.n_hyp_y,
        sigma_y=bank.sigma_y,
        rng=np.random.default_rng(cfg.seed + 50),
    )


def design_dict(bank: SystemBank, a: int) -> dict[str, Any]:
    return bank.designs[int(a)].as_dict()


def centres(Y: np.ndarray, a: int) -> np.ndarray:
    return np.asarray(Y[:, int(a)], dtype=np.float64)[:, None]


def observe(Y_eval: np.ndarray, i: int, a: int, eps: float) -> float:
    return float(Y_eval[int(i), int(a)]) + float(eps)


def terminal_after_sequence(
    bank: SystemBank,
    i: int,
    seq: tuple[int, ...],
    eps_steps: np.ndarray,
) -> float:
    log_w = uniform_log_prior(bank.n_support)
    for t, a in enumerate(seq):
        y = observe(bank.Y_eval, i, a, float(eps_steps[t]))
        log_w = update_log_weights(log_w, y, centres(bank.Y_support, a), bank.sigma_y)
    w = weights_from_log(log_w)
    return terminal_u(
        bank.U_support,
        w,
        alpha=bank.alpha,
        margin=bank.safety_margin,
        u_grid=bank.u_grid,
    )


def score_second_designs(
    bank: SystemBank,
    log_w: np.ndarray,
    used: set[int],
    *,
    hyp_idx_raw: np.ndarray,
    hyp_noise: np.ndarray,
) -> dict[int, float]:
    """E_y2[u] for every feasible second design under shared hyp CRN."""
    from tools.adaptive_advantage.planning_utils import (
        expected_u_one_step_posterior_crn,
    )

    out: dict[int, float] = {}
    for a in range(bank.n_designs):
        if a in used:
            continue
        out[a] = expected_u_one_step_posterior_crn(
            bank.Y_support,
            bank.U_support,
            a,
            log_w=log_w,
            sigma_y=bank.sigma_y,
            alpha=bank.alpha,
            margin=bank.safety_margin,
            u_grid=bank.u_grid,
            hyp_idx_raw=hyp_idx_raw,
            hyp_noise=hyp_noise,
        )
    return out


def gap_eps(bank: SystemBank, override: float | None = None) -> float:
    if override is not None:
        return float(override)
    g = np.asarray(bank.u_grid, dtype=np.float64)
    diffs = np.diff(np.unique(np.round(g, 12)))
    diffs = diffs[diffs > 0]
    return float(0.5 * float(diffs.min())) if diffs.size else 1e-6


def save_json(path: Path, obj: Any) -> None:
    def _default(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def mean_matrix(u: np.ndarray) -> float:
    return float(np.asarray(u, dtype=np.float64).mean())


def paired_ci(a: np.ndarray, b: np.ndarray, *, n_boot: int, seed: int) -> dict[str, Any]:
    return paired_delta_ci(a, b, n_boot=n_boot, seed=seed)
