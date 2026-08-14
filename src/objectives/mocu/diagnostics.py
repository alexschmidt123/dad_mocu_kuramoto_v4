"""Myopic selection and policy diagnostic helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from src.control.posterior_ctrl import normalize_log_weights
from src.objectives.mocu.context import (
    GLOBAL_SEED,
    ExperimentContext,
    expected_u_after_action_vector,
    normalize_method_key,
    observe_compressed,
    update_posterior_vector,
)


def select_myopic_action(
    ctx: ExperimentContext,
    log_w: np.ndarray,
    used_actions: Iterable[int],
    *,
    rollout_id: int,
    step: int,
    seed: int = GLOBAL_SEED,
    n_hypothetical: int | None = None,
) -> int:
    """One-step myopic: minimize expected posterior-safe u_ctrl."""
    ctrl = dict(ctx.cfg.raw.get("control") or {})
    n_hyp = int(n_hypothetical or ctrl.get("myopic_hypothetical", 64))
    used = {int(a) for a in used_actions}
    w = normalize_log_weights(log_w)
    rng = np.random.default_rng(int(seed) + int(rollout_id) * 17 + int(step) * 13)
    idx = rng.choice(len(w), size=min(n_hyp, len(w)), p=w, replace=True)
    noise = rng.normal(0.0, ctx.sigma_y, size=(len(idx), ctx.obs_dim))

    if bool(getattr(ctx, "continuous_duration_mode", False)) or str(
        getattr(ctx, "observation_mode", "")
    ).startswith("continuous_duration"):
        from src.domains.sir.design import chronological_feasible

        remaining = max(int(ctx.horizon) - len(used), 1)
        feasible = chronological_feasible(
            ctx.n_actions, list(used), remaining_steps=remaining
        )
    else:
        feasible = np.asarray(
            [a for a in range(ctx.n_actions) if a not in used], dtype=int
        )
    if feasible.size == 0:
        raise RuntimeError("No feasible action remains for myopic selection")

    best_a = None
    best_score = float("inf")
    for a in feasible.tolist():
        score = expected_u_after_action_vector(
            int(a),
            log_w,
            centres=ctx.centres_support,
            U=ctx.U_support,
            sigma_y=ctx.sigma_y,
            alpha=ctx.alpha,
            margin=ctx.margin,
            u_grid=ctx.u_grid,
            idx=idx,
            noise=noise,
        )
        if score < best_score - 1e-15 or (
            abs(score - best_score) <= 1e-15 and (best_a is None or a < best_a)
        ):
            best_score = score
            best_a = int(a)
    assert best_a is not None
    return int(best_a)


def diagnose_conditional_action_diversity(
    ctx: ExperimentContext,
    *,
    method: str,
    n_rollouts: int,
    seed: int = 7,
    device: str = "cpu",
) -> dict[str, Any]:
    """Lightweight open-loop / sequence-diversity diagnostic for a trained policy."""
    from src.objectives.mocu.train import load_trained_policy, _tensors_from_state

    key = normalize_method_key(method)
    policy = load_trained_policy(ctx, key, device=torch.device(device))
    sequences = []
    torch_device = torch.device(device)
    for rid in range(int(n_rollouts)):
        tid = rid % len(ctx.test_systems)
        system = ctx.test_systems[tid]
        log_w = ctx.log_p0.copy()
        actions: list[int] = []
        observations: list[np.ndarray] = []
        for step in range(ctx.horizon):
            tensors = _tensors_from_state(
                ctx,
                actions=actions,
                observations=observations,
                log_w=log_w,
                step=step,
                device=torch_device,
            )
            with torch.no_grad():
                action = int(policy(*tensors[:-1]).argmax(dim=-1).item())
            y = observe_compressed(
                system,
                action,
                sigma_y=ctx.sigma_y,
                n_obs=ctx.n_obs,
                global_seed=int(seed),
                theta_id=tid,
                rollout_id=rid,
                step=step,
            )
            actions.append(action)
            observations.append(y)
            log_w = update_posterior_vector(ctx, log_w, action, y)
        sequences.append(tuple(actions))
    unique = len(set(sequences))
    report = {
        "method": method,
        "n_rollouts": int(n_rollouts),
        "n_unique_sequences": unique,
        "open_loop": unique <= 1,
    }
    out = Path(ctx.out_dir) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "conditional_action_diversity.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(path)
    return report


def distill_myopic_to_dad(
    ctx: ExperimentContext,
    *,
    n_rollouts: int = 128,
    seed: int = 101,
    device: str = "cpu",
) -> dict[str, Any]:
    """Placeholder BC distillation entry (myopic labels → DAD architecture)."""
    return {
        "status": "not_implemented_in_recovered_layout",
        "n_rollouts": int(n_rollouts),
        "seed": int(seed),
        "device": str(device),
        "note": "Use train dad / evaluate myopic for production baselines.",
    }
