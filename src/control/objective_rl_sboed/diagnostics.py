"""Adaptivity and action-regret diagnostics for DAD vs RL-sBOED."""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.control.objective_rl_sboed import OUT, ROOT
from src.control.objective_rl_sboed.context import (
    StudyContext,
    belief_summary,
    load_study_context,
    update_log_weights,
)
from src.control.objective_rl_sboed.layout import method_key
from src.neural.rl_policy import AdaptiveExperimentPolicy, PolicyConfig


def _load_policy(ctx: StudyContext, checkpoint: Path) -> AdaptiveExperimentPolicy:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["policy"] if isinstance(payload, dict) and "policy" in payload else payload
    policy = AdaptiveExperimentPolicy(ctx.n_actions, PolicyConfig(max_steps=ctx.horizon))
    policy.load_state_dict(state)
    policy.eval()
    return policy


@torch.no_grad()
def policy_second_action(
    ctx: StudyContext,
    policy: AdaptiveExperimentPolicy,
    *,
    xi1: int,
    y1: float,
) -> int:
    """Select xi_2 given h_1 = (xi1, y1) under the learned policy (argmax)."""
    log_w = update_log_weights(ctx, ctx.log_p0.copy(), int(xi1), float(y1))
    length = int(ctx.horizon)
    action_idx = torch.zeros(1, length, dtype=torch.long)
    obs = torch.zeros(1, length, dtype=torch.float32)
    mask = torch.zeros(1, length, dtype=torch.float32)
    y_norm = (float(y1) - ctx.obs_mean) / ctx.obs_std
    action_idx[0, 0] = int(xi1)
    obs[0, 0] = float(y_norm)
    mask[0, 0] = 1.0
    belief = torch.as_tensor(
        belief_summary(ctx, log_w, [float(y1)])[None, :], dtype=torch.float32
    )
    steps = torch.as_tensor([1], dtype=torch.long)
    particles = torch.as_tensor(ctx.particle_features[None, :], dtype=torch.float32)
    weights = torch.as_tensor(np.exp(log_w - np.max(log_w))[None, :], dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    feasible = torch.ones(1, ctx.n_actions, dtype=torch.bool)
    feasible[0, int(xi1)] = False
    logits = policy(action_idx, obs, mask, belief, steps, particles, weights, feasible)
    return int(torch.argmax(logits, dim=-1).item())


def adaptivity_from_confirmation(rows: list[dict[str, Any]], method: str, init_mode: str, seed: int) -> dict[str, Any]:
    seqs = [r["sequence"] for r in rows]
    counts = Counter(seqs)
    dom, n = counts.most_common(1)[0]
    # Observation-dependent proxy: more than one unique sequence under keyed noise.
    return {
        "method": method,
        "init_mode": init_mode,
        "seed": seed,
        "n_unique_sequences": int(len(counts)),
        "dominant_sequence": dom,
        "dominant_sequence_fraction": float(n / max(len(rows), 1)),
        "observation_dependent_rate": float(1.0 - (n / max(len(rows), 1))),
        "n_rollouts": len(rows),
    }


def collect_adaptivity(system: str, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    root = OUT / f"{system}_T3"
    for method, init in (
        ("DAD", "random"),
        ("DAD", "fixed"),
        ("RL-sBOED", "random"),
        ("RL-sBOED", "fixed"),
    ):
        for seed in seeds:
            path = root / "train" / method_key(method, init) / f"seed_{seed}" / "confirmation_rollouts.csv"
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            out.append(adaptivity_from_confirmation(rows, method, init, seed))
    return out


def _parse_scores(raw: str) -> dict[int, float]:
    data = json.loads(raw) if raw.strip().startswith("{") else ast.literal_eval(raw)
    return {int(k): float(v) for k, v in data.items()}


def action_regret_rows(
    system: str,
    *,
    method: str,
    init_mode: str,
    seed: int,
    max_histories: int = 200,
) -> list[dict[str, Any]]:
    """Regret vs xi2_star from frozen objective_adaptive_value histories."""
    hist_path = (
        ROOT
        / "experiments"
        / "objective_adaptive_value"
        / f"{system}_T3"
        / "first_history_results.csv"
    )
    if not hist_path.is_file():
        return []
    ckpt = (
        OUT
        / f"{system}_T3"
        / "train"
        / method_key(method, init_mode)
        / f"seed_{seed}"
        / "best_checkpoint.pt"
    )
    if not ckpt.is_file():
        return []
    ctx = load_study_context(system)
    policy = _load_policy(ctx, ckpt)
    rows_out: list[dict[str, Any]] = []
    with hist_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if i >= max_histories:
                break
            xi1 = int(row["xi1"])
            y1 = float(row["y1"])
            xi2_star = int(row["xi2_star"])
            scores = _parse_scores(row["all_scores_json"])
            xi2_pol = policy_second_action(ctx, policy, xi1=xi1, y1=y1)
            j_star = float(row["best_expected_u_ctrl"])
            j_pol = float(scores.get(xi2_pol, np.nan))
            if not np.isfinite(j_pol):
                continue
            rows_out.append(
                {
                    "system": system,
                    "method": method,
                    "init_mode": init_mode,
                    "seed": seed,
                    "history_id": int(row["history_id"]),
                    "xi1": xi1,
                    "y1": y1,
                    "xi2_star": xi2_star,
                    "xi2_policy": xi2_pol,
                    "agree": int(xi2_pol == xi2_star),
                    "J_star": j_star,
                    "J_policy": j_pol,
                    "regret": j_pol - j_star,
                }
            )
    return rows_out


def collect_action_regret(
    system: str,
    seeds: tuple[int, ...],
    selected: dict[str, dict[str, Any]],
    *,
    max_histories: int = 200,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, meta in selected.items():
        init = str(meta["selected_init"])
        for seed in seeds:
            rows.extend(
                action_regret_rows(
                    system,
                    method=method,
                    init_mode=init,
                    seed=seed,
                    max_histories=max_histories,
                )
            )
    return rows


def reward_diagnostics(system: str, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    """Aggregate stepwise reward sparsity from training metrics."""
    out: list[dict[str, Any]] = []
    root = OUT / f"{system}_T3"
    for method, init in (("RL-sBOED", "random"), ("RL-sBOED", "fixed"), ("DAD", "random"), ("DAD", "fixed")):
        for seed in seeds:
            path = root / "train" / method_key(method, init) / f"seed_{seed}" / "training_metrics.csv"
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                continue
            zeros = [
                float(r["zero_intermediate_reward_fraction"])
                for r in rows
                if r.get("zero_intermediate_reward_fraction") not in (None, "")
            ]
            out.append(
                {
                    "system": system,
                    "method": method,
                    "init_mode": init,
                    "seed": seed,
                    "mean_zero_intermediate_reward_fraction": float(np.mean(zeros)) if zeros else float("nan"),
                    "n_updates_logged": len(rows),
                }
            )
    return out


def ensure_sensitivity_audit_alias() -> Path:
    """Spec path ``sensitivity_audit/`` aliases ``diagnostics/sensitivity_audit/``."""
    canonical = OUT / "diagnostics" / "sensitivity_audit"
    alias = OUT / "sensitivity_audit"
    canonical.mkdir(parents=True, exist_ok=True)
    if alias.exists() or alias.is_symlink():
        return alias
    try:
        alias.symlink_to(canonical.relative_to(OUT), target_is_directory=True)
    except OSError:
        # Fallback: copy key files if symlink fails.
        alias.mkdir(parents=True, exist_ok=True)
        for path in canonical.glob("*"):
            dest = alias / path.name
            if not dest.exists():
                dest.write_bytes(path.read_bytes())
    return alias
