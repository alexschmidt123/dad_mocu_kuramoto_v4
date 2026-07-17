"""Shared experiment context and belief features for the RL-sBOED study."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.control.legacy.adaptive_value_diagnosis import _batch_u_ctrl, _centres_matrix
from src.control.banks import extract_U_bank
from src.control.objective_rl_sboed import OUT, ROOT
from src.control.pilot import load_pilot_splits
from src.control.posterior_ctrl import (
    normalize_log_weights,
    posterior_control_decision,
)
from src.control.terminal_rule import load_frozen_terminal_rule, observe_with_keyed_noise
from src.contrastive.spce import log_prior_uniform_discrete
from src.data import lookup_action_y_sim
from src.run_context import load_experiment_run
from src.swing_equation_ode.design import build_catalog
from src.table_scoring import TableThetaSupport


BELIEF_DIM = 33


@dataclass
class StudyContext:
    system: str
    horizon: int
    centres: np.ndarray
    U: np.ndarray
    log_p0: np.ndarray
    sigma_y: float
    alpha: float
    margin: float
    u_grid: np.ndarray
    n_actions: int
    support: TableThetaSupport
    train_systems: list[dict[str, Any]]
    validation_systems: list[dict[str, Any]]
    confirmation_systems: list[dict[str, Any]]
    M_support: np.ndarray
    K_support: np.ndarray
    obs_mean: float
    obs_std: float
    particle_features: np.ndarray
    fixed_sequence: list[int]
    terminal_rule_hash: str
    exp_dir: Path


def load_fixed_sequence(system: str) -> list[int]:
    path = ROOT / "experiments" / f"{system}_T3" / "eval" / "fixed" / "subset_meta.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = [int(x) for x in payload["selected_action_ids"]]
    if len(ids) != 3:
        raise ValueError(f"expected T=3 Fixed sequence for {system}, got {ids}")
    return ids


def load_study_context(system: str, horizon: int = 3) -> StudyContext:
    if horizon != 3:
        raise ValueError("This study currently supports T=3 only")
    exp = ROOT / "experiments" / f"{system}_T3"
    run = load_experiment_run(exp, ROOT)
    splits = load_pilot_splits(exp, run)
    frozen = load_frozen_terminal_rule(exp)
    support_systems = list(splits["support_systems"])
    support = TableThetaSupport(
        systems=support_systems,
        log_p0=log_prior_uniform_discrete(len(support_systems)),
    )
    M = np.asarray([row["M"] for row in support.systems], dtype=np.float64)
    K = np.asarray([row["K"] for row in support.systems], dtype=np.float64)
    if M.ndim > 1:
        M = M.mean(axis=1)
    if K.ndim > 1:
        K = K.mean(axis=1)
    M = np.asarray(M, dtype=np.float64).reshape(-1)
    K = np.asarray(K, dtype=np.float64).reshape(-1)
    U = np.asarray(extract_U_bank(support.systems), dtype=np.float64).reshape(-1)
    if not (len(M) == len(K) == len(U)):
        raise ValueError(f"M/K/U length mismatch: {len(M)}, {len(K)}, {len(U)}")
    n_actions = len(build_catalog(run.cfg))
    centres = _centres_matrix(support, n_actions)
    train_like = list(splits["support_systems"]) + list(splits["calibration_systems"])
    obs = np.asarray(
        [
            lookup_action_y_sim(system_row, action)
            for system_row in train_like
            for action in range(n_actions)
        ],
        dtype=np.float64,
    )
    raw_particles = np.column_stack([M, K, U]).astype(np.float64)
    mean = raw_particles.mean(axis=0)
    std = np.maximum(raw_particles.std(axis=0), 1e-8)
    particles = ((raw_particles - mean) / std).astype(np.float32)
    meta = frozen.metadata()
    return StudyContext(
        system=system,
        horizon=horizon,
        centres=centres,
        U=U,
        log_p0=np.asarray(support.log_p0, dtype=np.float64),
        sigma_y=float(run.cfg.sigma_y),
        alpha=float(frozen.alpha),
        margin=float(frozen.margin),
        u_grid=np.asarray(frozen.u_candidates, dtype=np.float64),
        n_actions=n_actions,
        support=support,
        train_systems=list(splits["support_systems"]),
        validation_systems=list(splits["validation_systems"]),
        confirmation_systems=list(splits["test_systems"]),
        M_support=M,
        K_support=K,
        obs_mean=float(obs.mean()),
        obs_std=float(max(obs.std(), 1e-8)),
        particle_features=particles,
        fixed_sequence=load_fixed_sequence(system),
        terminal_rule_hash=str(meta.get("terminal_rule_hash", "")),
        exp_dir=exp,
    )


def control_from_log_weights(ctx: StudyContext, log_w: np.ndarray):
    w = normalize_log_weights(log_w)
    return posterior_control_decision(
        ctx.U, w, ctx.alpha, margin=ctx.margin, u_grid=ctx.u_grid
    )


def batch_u_ctrl(ctx: StudyContext, log_w: np.ndarray) -> np.ndarray:
    return _batch_u_ctrl(
        ctx.U, log_w, alpha=ctx.alpha, margin=ctx.margin, u_grid=ctx.u_grid
    )


def belief_summary(ctx: StudyContext, log_w: np.ndarray, observations: list[float]) -> np.ndarray:
    """33-d summary belief used by the production policy backbone."""
    w = normalize_log_weights(log_w)
    feats = np.zeros(BELIEF_DIM, dtype=np.float32)
    feats[0] = float(len(observations)) / float(ctx.horizon)
    ess = float(1.0 / np.sum(w * w))
    feats[1] = ess / float(len(w))
    feats[2] = float(np.max(w))
    feats[3] = float(np.sum(w * ctx.M_support))
    feats[4] = float(np.sqrt(max(np.sum(w * (ctx.M_support - feats[3]) ** 2), 0.0)))
    feats[5] = float(np.sum(w * ctx.K_support))
    feats[6] = float(np.sqrt(max(np.sum(w * (ctx.K_support - feats[5]) ** 2), 0.0)))
    order = np.argsort(ctx.U, kind="mergesort")
    u_sorted = ctx.U[order]
    cdf = np.cumsum(w[order])
    for i, q in enumerate((0.05, 0.25, 0.50, 0.75, 0.95)):
        idx = int(np.searchsorted(cdf, q, side="left"))
        idx = min(max(idx, 0), u_sorted.size - 1)
        feats[7 + i] = float(u_sorted[idx])
    decision = posterior_control_decision(
        ctx.U, w, ctx.alpha, margin=ctx.margin, u_grid=ctx.u_grid
    )
    feats[12] = float(decision.u_ctrl)
    # Mass on each grid level (16 slots).
    for i, level in enumerate(ctx.u_grid[:16]):
        feats[13 + i] = float(np.sum(w[np.isclose(ctx.U, level)]))
    if observations:
        y = np.asarray(observations, dtype=np.float64)
        feats[29] = float(y.mean())
        feats[30] = float(y.std() if y.size > 1 else 0.0)
        feats[31] = float(y.min())
        feats[32] = float(y.max())
    return feats


def update_log_weights(
    ctx: StudyContext,
    log_w: np.ndarray,
    action: int,
    y_obs: float,
) -> np.ndarray:
    centre = ctx.centres[int(action)]
    resid = (float(y_obs) - centre) / float(ctx.sigma_y)
    return log_w - 0.5 * resid * resid


def observe_bank(
    system_row: dict[str, Any],
    action: int,
    *,
    sigma_y: float,
    global_seed: int,
    theta_id: int,
    step: int,
    rollout_id: int = 0,
) -> float:
    return float(
        observe_with_keyed_noise(
            system_row,
            int(action),
            sigma_y=sigma_y,
            global_seed=global_seed,
            theta_id=theta_id,
            rollout_id=rollout_id,
            step=step,
        )
    )


def ensure_out_dirs() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT
