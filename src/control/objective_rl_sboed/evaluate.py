"""Baselines and paired comparison for the objective RL-sBOED study."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.control.myopic import MyopicControlSelector
from src.control.objective_rl_sboed.context import (
    StudyContext,
    control_from_log_weights,
    observe_bank,
    update_log_weights,
)
from src.control.u_req import ControlSpec
from src.rollout import FixedSelector, RandomSelector


GLOBAL_SEED = 77311


def _rollout_with_selector(
    ctx: StudyContext,
    system: dict[str, Any],
    *,
    theta_id: int,
    rollout_id: int,
    selector,
) -> dict[str, Any]:
    log_w = ctx.log_p0.copy()
    actions: list[int] = []
    observations: list[float] = []
    u_path = [control_from_log_weights(ctx, log_w).u_ctrl]
    for step in range(ctx.horizon):
        # Selectors in this repo expect history-like APIs; Fixed/Random are simple.
        if hasattr(selector, "select"):
            # MyopicControlSelector / FixedSelector / RandomSelector interfaces differ.
            try:
                action = int(
                    selector.select(
                        log_weights=log_w,
                        used=set(actions),
                        step=step,
                    )
                )
            except TypeError:
                action = int(selector.select(used=set(actions), step=step))
        else:
            raise TypeError(f"unsupported selector {type(selector)}")
        y = observe_bank(
            system,
            action,
            sigma_y=ctx.sigma_y,
            global_seed=GLOBAL_SEED,
            theta_id=theta_id,
            step=step,
            rollout_id=rollout_id,
        )
        actions.append(action)
        observations.append(y)
        log_w = update_log_weights(ctx, log_w, action, y)
        u_path.append(control_from_log_weights(ctx, log_w).u_ctrl)
    return {
        "sequence": actions,
        "y_obs": observations,
        "u_ctrl": u_path[-1],
        "u_path": u_path,
        "theta_id": theta_id,
        "rollout_id": rollout_id,
    }


def evaluate_fixed(ctx: StudyContext, systems: list[dict[str, Any]], n_rollouts: int) -> list[dict[str, Any]]:
    rows = []
    selector = FixedSelector(sequence=list(ctx.fixed_sequence))
    for rid in range(n_rollouts):
        tid = rid % len(systems)
        # FixedSelector API from rollout.py
        log_w = ctx.log_p0.copy()
        actions: list[int] = []
        observations: list[float] = []
        u_path = [control_from_log_weights(ctx, log_w).u_ctrl]
        for step in range(ctx.horizon):
            action = int(ctx.fixed_sequence[step])
            y = observe_bank(
                systems[tid],
                action,
                sigma_y=ctx.sigma_y,
                global_seed=GLOBAL_SEED,
                theta_id=tid,
                step=step,
                rollout_id=rid,
            )
            actions.append(action)
            observations.append(y)
            log_w = update_log_weights(ctx, log_w, action, y)
            u_path.append(control_from_log_weights(ctx, log_w).u_ctrl)
        rows.append(
            {
                "method": "Fixed",
                "rollout_id": rid,
                "theta_id": tid,
                "sequence": " ".join(map(str, actions)),
                "u_ctrl": u_path[-1],
            }
        )
    return rows


def evaluate_random(ctx: StudyContext, systems: list[dict[str, Any]], n_rollouts: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for rid in range(n_rollouts):
        tid = rid % len(systems)
        log_w = ctx.log_p0.copy()
        actions: list[int] = []
        observations: list[float] = []
        u_path = [control_from_log_weights(ctx, log_w).u_ctrl]
        available = list(range(ctx.n_actions))
        for step in range(ctx.horizon):
            action = int(rng.choice(available))
            available.remove(action)
            y = observe_bank(
                systems[tid],
                action,
                sigma_y=ctx.sigma_y,
                global_seed=GLOBAL_SEED,
                theta_id=tid,
                step=step,
                rollout_id=rid,
            )
            actions.append(action)
            observations.append(y)
            log_w = update_log_weights(ctx, log_w, action, y)
            u_path.append(control_from_log_weights(ctx, log_w).u_ctrl)
        rows.append(
            {
                "method": "Random",
                "rollout_id": rid,
                "theta_id": tid,
                "sequence": " ".join(map(str, actions)),
                "u_ctrl": u_path[-1],
            }
        )
    return rows


def evaluate_myopic(ctx: StudyContext, systems: list[dict[str, Any]], n_rollouts: int) -> list[dict[str, Any]]:
    """One-step expected-u_ctrl minimization using bank centres (no ODE)."""
    from src.control.legacy.adaptive_value_diagnosis import expected_u_after_action
    from src.control.posterior_ctrl import normalize_log_weights

    rows = []
    n_hyp = 32
    for rid in range(n_rollouts):
        tid = rid % len(systems)
        log_w = ctx.log_p0.copy()
        actions: list[int] = []
        observations: list[float] = []
        u_path = [control_from_log_weights(ctx, log_w).u_ctrl]
        for step in range(ctx.horizon):
            w = normalize_log_weights(log_w)
            rng = np.random.default_rng(GLOBAL_SEED + rid * 17 + step * 13)
            idx = rng.choice(len(w), size=n_hyp, p=w)
            noise = rng.normal(0.0, ctx.sigma_y, size=n_hyp)
            best_a = None
            best_score = float("inf")
            for a in range(ctx.n_actions):
                if a in actions:
                    continue
                score = float(
                    expected_u_after_action(
                        a,
                        log_w,
                        w,
                        centres=ctx.centres,
                        U=ctx.U,
                        sigma_y=ctx.sigma_y,
                        alpha=ctx.alpha,
                        margin=ctx.margin,
                        u_grid=ctx.u_grid,
                        idx=idx,
                        noise=noise,
                    )
                )
                if score < best_score - 1e-15 or (
                    abs(score - best_score) <= 1e-15 and (best_a is None or a < best_a)
                ):
                    best_score = score
                    best_a = a
            action = int(best_a)
            y = observe_bank(
                systems[tid],
                action,
                sigma_y=ctx.sigma_y,
                global_seed=GLOBAL_SEED,
                theta_id=tid,
                step=step,
                rollout_id=rid,
            )
            actions.append(action)
            observations.append(y)
            log_w = update_log_weights(ctx, log_w, action, y)
            u_path.append(control_from_log_weights(ctx, log_w).u_ctrl)
        rows.append(
            {
                "method": "Myopic",
                "rollout_id": rid,
                "theta_id": tid,
                "sequence": " ".join(map(str, actions)),
                "u_ctrl": u_path[-1],
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    u = np.asarray([float(r["u_ctrl"]) for r in rows], dtype=np.float64)
    seqs = [r["sequence"] for r in rows]
    counts = Counter(seqs)
    dom, n = counts.most_common(1)[0]
    return {
        "method": method,
        "mean_u_ctrl": float(u.mean()),
        "median_u_ctrl": float(np.median(u)),
        "std_u_ctrl": float(u.std()),
        "n_rollouts": len(rows),
        "n_unique_sequences": int(len(counts)),
        "dominant_sequence": dom,
        "dominant_sequence_fraction": float(n / len(rows)),
    }


def paired_bootstrap_ci(
    diff: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 123,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(diff[idx].mean()))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "mean_diff": float(diff.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
