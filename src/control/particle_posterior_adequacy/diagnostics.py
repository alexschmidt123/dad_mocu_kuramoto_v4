"""Posterior ESS, u_ctrl/u_cont, design stability and regret diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.control.particle_posterior_adequacy.supports import (
    WEIGHT_THRESHOLD,
    MasterArrays,
    ParticleSupport,
)
from src.control.posterior_batch import expected_u_after_action, update_posterior
from src.control.posterior_ctrl import (
    normalize_log_weights,
    posterior_control_decision,
    posterior_ess,
)


def degeneracy_flag(norm_ess: float, max_weight: float) -> str:
    """Documented diagnostic flags (not a sole scientific conclusion)."""
    if norm_ess < 0.05 or max_weight > 0.5:
        return "severe_degeneracy"
    if norm_ess < 0.15 or max_weight > 0.25:
        return "moderate_degeneracy"
    return "stable_support"


def posterior_weight_stats(weights: np.ndarray) -> dict[str, float]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / max(float(w.sum()), 1e-300)
    order = np.argsort(-w)
    top1 = float(w[order[0]]) if w.size else 0.0
    top5 = float(w[order[: min(5, w.size)]].sum())
    top10 = float(w[order[: min(10, w.size)]].sum())
    # entropy in nats
    nz = w[w > 0]
    ent = float(-np.sum(nz * np.log(nz))) if nz.size else 0.0
    n_eff = int(np.sum(w > WEIGHT_THRESHOLD))
    ess = float(posterior_ess(w))
    return {
        "ESS": ess,
        "normalized_ESS": ess / max(w.size, 1),
        "max_weight": float(w.max()) if w.size else 0.0,
        "top1_mass": top1,
        "top5_mass": top5,
        "top10_mass": top10,
        "posterior_entropy": ent,
        "n_nonnegligible": n_eff,
        "weight_threshold": WEIGHT_THRESHOLD,
        "degeneracy_flag": degeneracy_flag(ess / max(w.size, 1), float(w.max()) if w.size else 1.0),
    }


def posterior_at_history(
    support: ParticleSupport,
    actions: list[int],
    observations: list[float],
    sigma_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    log_w = np.asarray(support.log_p0, dtype=np.float64).copy()
    for a, y in zip(actions, observations):
        log_w, _ = update_posterior(log_w, float(y), support.centres[int(a)], sigma_y)
    return log_w, normalize_log_weights(log_w)


def terminal_controls(
    support: ParticleSupport,
    weights: np.ndarray,
    *,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
) -> dict[str, float]:
    dec = posterior_control_decision(
        support.U,
        weights,
        alpha,
        margin=margin,
        u_grid=u_grid,
        snap_up=True,
    )
    return {
        "u_cont": float(dec.u_raw),
        "u_ctrl": float(dec.u_ctrl_snapped),
        "u_quantile": float(dec.u_quantile),
    }


def score_designs(
    support: ParticleSupport,
    log_w: np.ndarray,
    weights: np.ndarray,
    used_actions: set[int],
    *,
    master: MasterArrays,
    idx: np.ndarray,
    noise: np.ndarray,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for action in range(master.n_actions):
        if action in used_actions:
            continue
        d = master.catalog[action]
        bus, amp = int(d.bus), float(d.amplitude)
        j_s = expected_u_after_action(
            action,
            log_w,
            weights,
            centres=support.centres,
            U=support.U,
            sigma_y=master.sigma_y,
            alpha=master.alpha,
            margin=master.margin,
            u_grid=master.u_grid,
            idx=idx,
            noise=noise,
            snap_up=True,
        )
        j_c = expected_u_after_action(
            action,
            log_w,
            weights,
            centres=support.centres,
            U=support.U,
            sigma_y=master.sigma_y,
            alpha=master.alpha,
            margin=master.margin,
            u_grid=master.u_grid,
            idx=idx,
            noise=noise,
            snap_up=False,
        )
        out.append(
            {
                "action": int(action),
                "bus": bus,
                "amplitude": amp,
                "J_snapped": float(j_s),
                "J_continuous": float(j_c),
            }
        )
    return out


def analyze_history_support(
    master: MasterArrays,
    support: ParticleSupport,
    history: dict[str, Any],
    history_step: int,
    *,
    n_hyp: int,
    rng: np.random.Generator,
    reference_scores: dict[int, float] | None = None,
    reference_optimal: dict[str, Any] | None = None,
    score_designs_flag: bool = True,
) -> dict[str, Any]:
    step = history["steps"].get(history_step)
    if step is None:
        # history shorter than requested step
        max_t = max(history["steps"])
        step = history["steps"][max_t]
        history_step = int(max_t)
    actions = list(step["actions"])
    observations = list(step["observations"])
    log_w, weights = posterior_at_history(
        support, actions, observations, master.sigma_y
    )
    stats = posterior_weight_stats(weights)
    ctrls = terminal_controls(
        support,
        weights,
        alpha=master.alpha,
        margin=master.margin,
        u_grid=master.u_grid,
    )
    scored: list[dict[str, Any]] = []
    if score_designs_flag:
        # Shared CRN for design scoring (keyed by support seed + history + step)
        idx = rng.choice(len(weights), size=n_hyp, p=weights)
        noise = rng.normal(0.0, master.sigma_y, size=n_hyp)
        scored = score_designs(
            support,
            log_w,
            weights,
            set(actions),
            master=master,
            idx=idx,
            noise=noise,
        )
    if not scored:
        opt = {
            "optimal_design": None,
            "optimal_bus": None,
            "optimal_amplitude": None,
            "J_star_snapped": math.nan,
            "J_star_continuous": math.nan,
        }
        design_rows = []
    else:
        best = min(scored, key=lambda r: (r["J_snapped"], r["action"]))
        opt = {
            "optimal_design": int(best["action"]),
            "optimal_bus": int(best["bus"]),
            "optimal_amplitude": float(best["amplitude"]),
            "J_star_snapped": float(best["J_snapped"]),
            "J_star_continuous": float(best["J_continuous"]),
        }
        design_rows = scored

    regret = math.nan
    design_agreement = amplitude_agreement = bus_agreement = None
    ref_opt = None
    if reference_optimal is not None and opt["optimal_design"] is not None:
        ref_opt = reference_optimal.get("optimal_design")
        design_agreement = int(opt["optimal_design"]) == int(ref_opt) if ref_opt is not None else None
        bus_agreement = (
            int(opt["optimal_bus"]) == int(reference_optimal["optimal_bus"])
            if reference_optimal.get("optimal_bus") is not None
            else None
        )
        amplitude_agreement = (
            abs(float(opt["optimal_amplitude"]) - float(reference_optimal["optimal_amplitude"]))
            < 1e-12
            if reference_optimal.get("optimal_amplitude") is not None
            else None
        )
    if reference_scores is not None and opt["optimal_design"] is not None:
        j_ref_sel = float(reference_scores[int(opt["optimal_design"])])
        j_ref_min = float(min(reference_scores.values()))
        regret = j_ref_sel - j_ref_min

    particle_row = {
        "system": master.system,
        "particle_count": support.n_particles,
        "support_seed": support.support_seed,
        "history_id": history["history_id"],
        "history_step": history_step,
        "theta_id": history.get("theta_id"),
        "selection_rule": support.selection_rule,
        **stats,
        **ctrls,
    }
    design_row = {
        "system": master.system,
        "particle_count": support.n_particles,
        "support_seed": support.support_seed,
        "history_id": history["history_id"],
        "history_step": history_step,
        "optimal_bus": opt["optimal_bus"],
        "optimal_amplitude": opt["optimal_amplitude"],
        "optimal_design": opt["optimal_design"],
        "reference_optimal_design": ref_opt,
        "design_agreement": design_agreement,
        "bus_agreement": bus_agreement,
        "amplitude_agreement": amplitude_agreement,
        "reference_regret": regret,
        "J_star_snapped": opt["J_star_snapped"],
        "J_star_continuous": opt["J_star_continuous"],
    }
    return {
        "particle_row": particle_row,
        "design_row": design_row,
        "scores_snapped": {int(r["action"]): float(r["J_snapped"]) for r in design_rows},
        "scores_continuous": {
            int(r["action"]): float(r["J_continuous"]) for r in design_rows
        },
        "opt": opt,
    }


def error_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean_abs_error": math.nan,
            "median_abs_error": math.nan,
            "p95_abs_error": math.nan,
            "max_abs_error": math.nan,
            "n": 0,
        }
    return {
        "mean_abs_error": float(np.mean(arr)),
        "median_abs_error": float(np.median(arr)),
        "p95_abs_error": float(np.quantile(arr, 0.95)),
        "max_abs_error": float(np.max(arr)),
        "n": int(arr.size),
    }
