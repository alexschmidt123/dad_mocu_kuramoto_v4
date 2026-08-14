"""LEGACY Fixed/Myopic/DAD fusion router — NOT publication MoE-sBOED.

Publication MoE-sBOED is ``BeliefConditionedMoEPolicy`` in
``src/policies/rl_sboed.py``: a learned shared base + belief-conditioned
residual experts. Experts are never named Fixed/Myopic/DAD.

This module remains only for older stepwise-EIG fusion checkpoints that
score Fixed/Myopic/DAD proposals and fuse them. Do not use it for new
continuous-duration / vector-EIG / MOCU runs.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class MoERouterConfig:
    """Continuous reliability router; coefficients are validation parameters."""

    temperature: float = 0.75
    fixed_ess_coefficient: float = 2.0
    myopic_progress_coefficient: float = 2.0
    myopic_dad_uncertainty_coefficient: float = 1.0
    dad_future_coefficient: float = 2.0
    dad_confidence_coefficient: float = 2.0
    fixed_bias: float = 0.0
    myopic_bias: float = 2.0
    dad_bias: float = 0.0
    residual_scale: float = 1.0
    residual_enabled: bool = True


class ResidualValueCritic(torch.nn.Module):
    """Small action-value residual trained from counterfactual continuations."""

    def __init__(self, feature_dim: int = 8, hidden_dim: int = 48):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def residual_critic_path(exp_dir: Path) -> Path:
    from src.reporting.run_context import model_dir

    return model_dir(exp_dir) / "moe_sboed_residual.pth"


def _action_features(
    fixed_scores: np.ndarray,
    myopic_scores: np.ndarray,
    dad_scores: np.ndarray,
    feasible: np.ndarray,
    *,
    step: int,
    horizon: int,
    ess_fraction: float,
    dad_normalized_entropy: float,
    belief_features: np.ndarray | None = None,
) -> np.ndarray:
    """State-and-action features shared by MOCU and EIG critics."""
    calibrated = np.stack(
        [
            _rank_utility(fixed_scores, feasible),
            _rank_utility(myopic_scores, feasible),
            _rank_utility(dad_scores, feasible),
        ],
        axis=1,
    )
    belief = (
        np.asarray(belief_features, dtype=np.float32).reshape(-1)
        if belief_features is not None
        else np.zeros(0, dtype=np.float32)
    )
    features = np.zeros((len(fixed_scores), 8 + len(belief)), dtype=np.float32)
    progress = float(step) / float(max(horizon - 1, 1))
    features[:, 0] = progress
    features[:, 1] = float(ess_fraction)
    features[:, 2] = float(dad_normalized_entropy)
    features[:, 3:6] = np.where(np.isfinite(calibrated), calibrated, -1.0)
    features[:, 6] = np.mean(features[:, 3:6], axis=1)
    features[:, 7] = np.std(features[:, 3:6], axis=1)
    if len(belief):
        features[:, 8:] = belief[None, :]
    return features


def _eig_belief_features(table_support, weights: np.ndarray) -> np.ndarray:
    """Posterior parameter moments without imposing an ordering on particles."""
    M = np.asarray([s["M"] for s in table_support.systems], dtype=np.float64)
    K = np.asarray([s["K"] for s in table_support.systems], dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mean_m = np.sum(w[:, None] * M, axis=0)
    mean_k = np.sum(w[:, None] * K, axis=0)
    std_m = np.sqrt(np.maximum(np.sum(w[:, None] * (M - mean_m) ** 2, axis=0), 0.0))
    std_k = np.sqrt(np.maximum(np.sum(w[:, None] * (K - mean_k) ** 2, axis=0), 0.0))
    scale_m = max(float(np.ptp(M)), 1e-8)
    scale_k = max(float(np.ptp(K)), 1e-8)
    return np.concatenate(
        [
            (mean_m - float(np.mean(M))) / scale_m,
            std_m / scale_m,
            (mean_k - float(np.mean(K))) / scale_k,
            std_k / scale_k,
            np.asarray(
                [
                    float(np.max(w)),
                    float(-np.sum(w * np.log(np.clip(w, 1e-300, None))))
                    / max(math.log(len(w)), 1e-12),
                ]
            ),
        ]
    ).astype(np.float32)


def _load_residual_critic(
    exp_dir: Path, device: torch.device
) -> tuple[ResidualValueCritic | None, dict[str, float]]:
    path = residual_critic_path(exp_dir)
    if not path.is_file():
        return None, {"target_mean": 0.0, "target_scale": 1.0}
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    critic = ResidualValueCritic(
        feature_dim=int(checkpoint.get("feature_dim", 8)),
        hidden_dim=int(checkpoint.get("hidden_dim", 48)),
    ).to(device)
    critic.load_state_dict(checkpoint["state_dict"])
    critic.eval()
    return critic, {
        "target_mean": float(checkpoint.get("target_mean", 0.0)),
        "target_scale": float(checkpoint.get("target_scale", 1.0)),
    }


def _critic_values(
    critic: ResidualValueCritic | None,
    normalization: dict[str, float],
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray | None:
    if critic is None:
        return None
    with torch.no_grad():
        standardized = critic(
            torch.as_tensor(features, dtype=torch.float32, device=device)
        ).cpu().numpy()
    return (
        standardized * float(normalization["target_scale"])
        + float(normalization["target_mean"])
    )


def moe_config_path(exp_dir: Path) -> Path:
    from src.reporting.run_context import model_dir

    return model_dir(exp_dir) / "moe_sboed.json"


def prepare_moe(
    exp_dir: Path,
    *,
    experiment_type: str,
    config: MoERouterConfig | None = None,
) -> dict[str, Any]:
    """Save the objective-specific router definition; experts remain frozen."""
    cfg = config or MoERouterConfig()
    report = {
        "method": "MoE-sBOED",
        "experiment_type": str(experiment_type),
        "experts": ["Fixed", "Myopic", "DAD"],
        "routing": (
            "residual policy improvement over every feasible action; "
            "Fixed/Myopic/DAD scores are proposal features, not hard choices"
        ),
        "config": asdict(cfg),
        "objective": (
            "terminal MOCU"
            if experiment_type == "objective_based"
            else "terminal entropy/EIG"
        ),
    }
    path = moe_config_path(exp_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["path"] = str(path)
    return report


def load_moe_config(exp_dir: Path) -> MoERouterConfig:
    path = moe_config_path(exp_dir)
    if not path.is_file():
        return MoERouterConfig()
    doc = json.loads(path.read_text(encoding="utf-8"))
    return MoERouterConfig(**dict(doc.get("config") or {}))


def router_weights(
    *,
    step: int,
    horizon: int,
    ess_fraction: float,
    dad_normalized_entropy: float,
    config: MoERouterConfig,
) -> np.ndarray:
    """Return continuous Fixed/Myopic/DAD reliability weights."""
    progress = float(step) / float(max(horizon - 1, 1))
    future = 1.0 - progress
    confidence = float(np.clip(1.0 - dad_normalized_entropy, 0.0, 1.0))
    logits = np.asarray(
        [
            config.fixed_bias
            + config.fixed_ess_coefficient * float(ess_fraction),
            config.myopic_bias
            + config.myopic_progress_coefficient * progress
            + config.myopic_dad_uncertainty_coefficient
            * float(dad_normalized_entropy),
            config.dad_bias
            + config.dad_future_coefficient * future
            + config.dad_confidence_coefficient * confidence,
        ],
        dtype=np.float64,
    )
    logits /= max(float(config.temperature), 1e-6)
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    return weights / float(np.sum(weights))


def _rank_utility(values: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    """Convert arbitrary expert scores (higher=better) to common rank units."""
    out = np.full(len(values), -np.inf, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)[feasible]
    unique = np.unique(vals)
    if len(unique) == 1:
        ranks = np.zeros(len(vals), dtype=np.float64)
    else:
        levels = np.linspace(-1.0, 1.0, num=len(unique))
        ranks = levels[np.searchsorted(unique, vals)]
    out[feasible] = ranks
    return out


def _fixed_utility(
    n_actions: int, fixed_sequence: list[int], used: list[int]
) -> np.ndarray:
    """Full action preference induced by the remaining fixed sequence."""
    used_set = set(map(int, used))
    remaining = [int(a) for a in fixed_sequence if int(a) not in used_set]
    values = np.full(n_actions, -1.0, dtype=np.float64)
    for rank, action in enumerate(remaining):
        values[action] = 1.0 - rank / float(max(len(remaining), 1))
    return values


def fuse_action_scores(
    fixed_scores: np.ndarray,
    myopic_scores: np.ndarray,
    dad_scores: np.ndarray,
    feasible: np.ndarray,
    weights: np.ndarray,
    residual_values: np.ndarray | None = None,
    residual_scale: float = 1.0,
) -> tuple[int, np.ndarray]:
    """Fuse calibrated full-action rankings; output may differ from every top-1."""
    calibrated = np.stack(
        [
            _rank_utility(fixed_scores, feasible),
            _rank_utility(myopic_scores, feasible),
            _rank_utility(dad_scores, feasible),
        ],
        axis=0,
    )
    fused = np.full(calibrated.shape[1], -np.inf, dtype=np.float64)
    fused[feasible] = np.sum(
        weights[:, None] * calibrated[:, feasible], axis=0
    )
    if residual_values is not None:
        residual_rank = _rank_utility(
            np.asarray(residual_values, dtype=np.float64), feasible
        )
        fused[feasible] += float(residual_scale) * residual_rank[feasible]
    return int(np.argmax(fused)), fused


def _next_fixed(sequence: list[int], used: list[int]) -> int:
    used_set = set(map(int, used))
    for action in sequence:
        if int(action) not in used_set:
            return int(action)
    raise RuntimeError("Fixed expert has no unused action")


def _fit_residual_critic(
    exp_dir: Path,
    features: list[np.ndarray],
    targets: list[float],
    *,
    objective: str,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    """Fit the residual action-value critic and save one compact checkpoint."""
    if not features:
        raise RuntimeError("No counterfactual samples were generated for MoE critic")
    torch.manual_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    y_raw = np.asarray(targets, dtype=np.float32)
    target_mean = float(np.mean(y_raw))
    target_scale = max(float(np.std(y_raw)), 1e-6)
    y = torch.as_tensor(
        (y_raw - target_mean) / target_scale,
        dtype=torch.float32,
        device=device,
    )
    critic = ResidualValueCritic(feature_dim=x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        critic.parameters(), lr=2e-3, weight_decay=1e-4
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    final_loss = float("nan")
    for _ in range(int(epochs)):
        order = torch.randperm(len(x), generator=generator, device=device)
        for start in range(0, len(x), 128):
            idx = order[start : start + 128]
            pred = critic(x[idx])
            loss = torch.nn.functional.smooth_l1_loss(pred, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            optimizer.step()
            final_loss = float(loss.detach().item())
    path = residual_critic_path(exp_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": critic.state_dict(),
            "feature_dim": int(x.shape[1]),
            "hidden_dim": 48,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "objective": objective,
            "n_counterfactual_samples": len(features),
            "final_training_loss": final_loss,
        },
        path,
    )
    return {
        "path": str(path),
        "n_counterfactual_samples": len(features),
        "final_training_loss": final_loss,
        "target_mean": target_mean,
        "target_scale": target_scale,
    }


def _counterfactual_candidates(
    feature_matrix: np.ndarray,
    feasible: list[int],
    *,
    budget: int | None = None,
) -> list[int]:
    """Return all feasible actions by default for genuine policy improvement."""
    feasible_a = np.asarray(feasible, dtype=int)
    if budget is None or int(budget) >= len(feasible_a):
        return list(map(int, feasible_a))
    selected: set[int] = set()
    for score_column in (3, 4, 5):
        order = feasible_a[
            np.argsort(feature_matrix[feasible_a, score_column])[::-1]
        ]
        selected.update(map(int, order[:4]))
    if len(selected) < min(budget, len(feasible)):
        coverage = np.linspace(
            0, len(feasible_a) - 1, num=min(budget, len(feasible_a)), dtype=int
        )
        for idx in coverage:
            selected.add(int(feasible_a[idx]))
            if len(selected) >= budget:
                break
    return sorted(selected)[: max(budget, len(selected))]


@torch.no_grad()
def rollout_objective_moe(
    ctx,
    policy,
    system: dict[str, Any],
    *,
    theta_id: int,
    rollout_id: int,
    device: torch.device,
    config: MoERouterConfig,
    forced_prefix: list[int] | None = None,
) -> dict[str, Any]:
    """Objective/MOCU MoE with vector observations and control-cost Myopic."""
    from src.objectives.mocu.context import (
        GLOBAL_SEED,
        belief_summary,
        control_from_log_weights,
        observe_compressed,
        update_posterior_vector,
    )
    from src.objectives.mocu.diagnostics import select_myopic_action
    from src.objectives.mocu.train import _tensors_from_state
    from src.control.posterior_ctrl import normalize_log_weights

    log_w = ctx.log_p0.copy()
    actions: list[int] = []
    observations: list[np.ndarray] = []
    router_trace: list[dict[str, Any]] = []
    critic, critic_norm = _load_residual_critic(ctx.out_dir, device)
    for step in range(ctx.horizon):
        feasible = np.asarray(
            [a for a in range(ctx.n_actions) if a not in set(actions)], dtype=int
        )
        fixed_scores = _fixed_utility(
            ctx.n_actions, ctx.fixed_sequence, actions
        )
        n_hyp = int((ctx.cfg.raw.get("control") or {}).get("myopic_hypothetical", 64))
        if ctx.obs_dim >= 100:
            n_hyp = min(n_hyp, 4)
        w = normalize_log_weights(log_w)
        rng = np.random.default_rng(
            GLOBAL_SEED + int(rollout_id) * 17 + int(step) * 13
        )
        idx = rng.choice(len(w), size=n_hyp, p=w)
        noise = rng.normal(0.0, ctx.sigma_y, size=(n_hyp, ctx.obs_dim))
        myopic_scores = np.full(ctx.n_actions, -np.inf, dtype=np.float64)
        from src.objectives.mocu.context import expected_u_all_actions_torch
        expected_u = expected_u_all_actions_torch(
            log_w,
            centres=ctx.centres_support,
            U=ctx.U_support,
            sigma_y=ctx.sigma_y,
            alpha=ctx.alpha,
            margin=ctx.margin,
            u_grid=ctx.u_grid,
            idx=idx,
            noise=noise,
            feasible=feasible,
            device=str(device),
        )
        myopic_scores[feasible] = -expected_u[feasible]
        state = _tensors_from_state(
            ctx,
            actions=actions,
            observations=observations,
            log_w=log_w,
            step=step,
            device=device,
        )
        logits = policy(*state).squeeze(0)
        if actions:
            logits[torch.as_tensor(actions, dtype=torch.long, device=device)] = -1e9
        dad_scores = logits.detach().cpu().numpy().astype(np.float64)
        dad_action = int(torch.argmax(logits).item())
        probs = torch.softmax(logits, dim=-1)
        dad_entropy = float(
            -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
        )
        n_feasible = max(ctx.n_actions - len(actions), 1)
        entropy_norm = dad_entropy / max(math.log(n_feasible), 1e-12)
        ess_fraction = float(1.0 / np.sum(w * w)) / float(len(w))
        weights = router_weights(
            step=step,
            horizon=ctx.horizon,
            ess_fraction=ess_fraction,
            dad_normalized_entropy=entropy_norm,
            config=config,
        )
        features = _action_features(
            fixed_scores,
            myopic_scores,
            dad_scores,
            feasible,
            step=step,
            horizon=ctx.horizon,
            ess_fraction=ess_fraction,
            dad_normalized_entropy=entropy_norm,
            belief_features=belief_summary(ctx, log_w, observations),
        )
        residual_values = (
            _critic_values(critic, critic_norm, features, device)
            if config.residual_enabled
            else None
        )
        action, _ = fuse_action_scores(
            fixed_scores,
            myopic_scores,
            dad_scores,
            feasible,
            weights,
            residual_values=residual_values,
            residual_scale=config.residual_scale,
        )
        if forced_prefix is not None and step < len(forced_prefix):
            forced = int(forced_prefix[step])
            if forced not in set(feasible.tolist()):
                raise ValueError(f"Forced action {forced} is infeasible at step {step}")
            action = forced
        fixed_action = int(np.argmax(fixed_scores[feasible]))
        fixed_action = int(feasible[fixed_action])
        myopic_action = int(np.argmax(myopic_scores))
        y = observe_compressed(
            system,
            action,
            sigma_y=ctx.sigma_y,
            n_obs=ctx.n_obs,
            global_seed=GLOBAL_SEED,
            theta_id=theta_id,
            rollout_id=rollout_id,
            step=step,
        )
        router_trace.append(
            {
                "step": step,
                "router_weights": {
                    "fixed": float(weights[0]),
                    "myopic": float(weights[1]),
                    "dad": float(weights[2]),
                },
                "action": action,
                "fixed_action": fixed_action,
                "myopic_action": myopic_action,
                "dad_action": dad_action,
                "ess_fraction": ess_fraction,
                "dad_normalized_entropy": entropy_norm,
                "action_features": features.tolist(),
                "critic_values": (
                    residual_values.tolist()
                    if residual_values is not None
                    else None
                ),
            }
        )
        actions.append(action)
        observations.append(y)
        log_w = update_posterior_vector(ctx, log_w, action, y)
    decision = control_from_log_weights(ctx, log_w)
    return {
        "actions": actions,
        "observations": observations,
        "u_ctrl": float(decision.u_ctrl),
        "log_w": log_w,
        "router_trace": router_trace,
    }


def calibrate_objective_moe(
    ctx,
    *,
    smoke: bool = False,
    seed: int = 101,
) -> dict[str, Any]:
    """Train and calibrate the MOCU residual router off support."""
    started = time.perf_counter()
    from src.objectives.mocu.train import load_trained_policy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_trained_policy(ctx, "DAD", device=device)
    n_val = min(4 if smoke else 16, len(ctx.validation_systems))
    systems = ctx.validation_systems[:n_val]
    rows = []
    for name, candidate in _router_candidates():
        mocu = []
        safe = []
        for rollout_id, system in enumerate(systems):
            out = rollout_objective_moe(
                ctx,
                policy,
                system,
                theta_id=rollout_id,
                rollout_id=rollout_id,
                device=device,
                config=candidate,
            )
            u_ctrl = float(out["u_ctrl"])
            mocu.append(u_ctrl - float(system["u_req"]))
            safe.append(u_ctrl + 1e-12 >= float(system["u_req"]))
        rows.append(
            {
                "name": name,
                "mean_validation_mocu": float(np.mean(mocu)),
                "validation_safety_rate": float(np.mean(safe)),
                "config": asdict(candidate),
            }
        )
    eligible = [row for row in rows if row["validation_safety_rate"] >= 0.95]
    selected = (
        min(eligible, key=lambda row: row["mean_validation_mocu"])
        if eligible
        else next(row for row in rows if row["name"] == "myopic_anchor")
    )
    base_config = MoERouterConfig(**selected["config"])
    critic_systems = systems[: max(1, n_val // 2)]
    scale_systems = systems[max(1, n_val // 2) :] or systems[-1:]
    critic_features: list[np.ndarray] = []
    critic_targets: list[float] = []
    for local_id, system in enumerate(critic_systems):
        prefix: list[int] = []
        base = rollout_objective_moe(
            ctx,
            policy,
            system,
            theta_id=local_id,
            rollout_id=10_000 + local_id,
            device=device,
            config=base_config,
        )
        for step, trace in enumerate(base["router_trace"]):
            feature_matrix = np.asarray(trace["action_features"], dtype=np.float32)
            feasible = [
                a for a in range(ctx.n_actions) if a not in set(prefix)
            ]
            candidates = _counterfactual_candidates(
                feature_matrix, feasible
            )
            for candidate_action in candidates:
                counterfactual = rollout_objective_moe(
                    ctx,
                    policy,
                    system,
                    theta_id=local_id,
                    rollout_id=10_000 + local_id,
                    device=device,
                    config=base_config,
                    forced_prefix=prefix + [int(candidate_action)],
                )
                critic_features.append(feature_matrix[int(candidate_action)])
                u_ctrl = float(counterfactual["u_ctrl"])
                u_req = float(system["u_req"])
                shortfall = max(u_req - u_ctrl, 0.0)
                realized_cost = u_ctrl + 10.0 * shortfall + 0.10 * float(
                    shortfall > 0.0
                )
                critic_targets.append(-(realized_cost - u_req))
            prefix.append(int(base["actions"][step]))
    critic_report = _fit_residual_critic(
        ctx.out_dir,
        critic_features,
        critic_targets,
        objective="negative_safety_aware_terminal_mocu",
        seed=seed,
        epochs=80 if smoke else 180,
    )
    scale_rows = []
    for scale in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        candidate = MoERouterConfig(
            **{
                **asdict(base_config),
                "residual_enabled": bool(scale > 0.0),
                "residual_scale": float(scale),
            }
        )
        values = []
        safety = []
        for local_id, system in enumerate(scale_systems):
            out = rollout_objective_moe(
                ctx,
                policy,
                system,
                theta_id=local_id,
                rollout_id=20_000 + local_id,
                device=device,
                config=candidate,
            )
            values.append(float(out["u_ctrl"]) - float(system["u_req"]))
            safety.append(
                float(out["u_ctrl"]) + 1e-12 >= float(system["u_req"])
            )
        scale_rows.append(
            {
                "residual_scale": float(scale),
                "mean_validation_mocu": float(np.mean(values)),
                "validation_safety_rate": float(np.mean(safety)),
                "per_system_metric": [float(v) for v in values],
                "config": asdict(candidate),
            }
        )
    eligible_scales = [
        row for row in scale_rows if row["validation_safety_rate"] >= 0.95
    ]
    baseline_scale = next(
        row for row in scale_rows if row["residual_scale"] == 0.0
    )
    selected_scale = baseline_scale
    for row in eligible_scales:
        if row["residual_scale"] == 0.0:
            continue
        improvement = np.asarray(baseline_scale["per_system_metric"]) - np.asarray(
            row["per_system_metric"]
        )
        se = (
            float(np.std(improvement, ddof=1)) / math.sqrt(len(improvement))
            if len(improvement) > 1
            else float("inf")
        )
        lower = float(np.mean(improvement)) - 1.645 * se
        row["paired_improvement_lcb90"] = lower
        if lower > 0.0 and row["mean_validation_mocu"] < selected_scale["mean_validation_mocu"]:
            selected_scale = row
    report = prepare_moe(
        ctx.out_dir,
        experiment_type="objective_based",
        config=MoERouterConfig(**selected_scale["config"]),
    )
    report.update(
        {
            "selection_split": "strict off-support validation",
            "n_validation_systems": n_val,
            "seed": int(seed),
            "selected_candidate": selected["name"],
            "validation_mean_mocu": selected_scale["mean_validation_mocu"],
            "validation_safety_rate": selected["validation_safety_rate"],
            "candidates": rows,
            "critic": critic_report,
            "residual_scale_candidates": scale_rows,
            "selected_residual_scale": selected_scale["residual_scale"],
            "do_no_harm_fallback": "myopic_anchor",
            "calibration_seconds": float(time.perf_counter() - started),
            "device": str(device),
        }
    )
    path = moe_config_path(ctx.out_dir)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def rollout_eig_moe(
    cfg,
    exp_dir: Path,
    test_systems: list[dict[str, Any]],
    catalog,
    table_support,
    *,
    config_override: MoERouterConfig | None = None,
    forced_prefix: list[int] | None = None,
) -> list[dict[str, Any]]:
    """EIG MoE with scalar table likelihood and a separately trained EIG-DAD."""
    from src.inference.spce import (
        clamp_info_gain,
        log_gaussian_observation_density,
        normalize_log_weights,
        posterior_entropy,
    )
    from src.banks.tables import lookup_action_y
    from src.objectives.eig.pipeline import default_fixed_sequence
    from src.policies.dad import DADPolicy
    from src.reporting.run_context import model_dir
    from src.inference.scoring import y_sim_last_step_from_tables

    config = config_override or load_moe_config(exp_dir)
    policy_path = model_dir(exp_dir) / "dad_eig.pth"
    if not policy_path.is_file():
        policy_path = model_dir(exp_dir) / "dad_delta_h.pth"
    if not policy_path.is_file():
        policy_path = model_dir(exp_dir) / "dad_spce.pth"
    if not policy_path.is_file():
        raise FileNotFoundError(f"MoE EIG-DAD checkpoint missing under {model_dir(exp_dir)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(policy_path, map_location=device, weights_only=False)
    policy = DADPolicy(len(catalog), max_steps=cfg.step_number).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    critic, critic_norm = _load_residual_critic(exp_dir, device)
    fixed_sequence = default_fixed_sequence(cfg, catalog)
    outputs = []
    for system in test_systems:
        log_w = np.array(table_support.log_p0, dtype=np.float64)
        actions: list[int] = []
        observations: list[float] = []
        trace: list[dict[str, Any]] = []
        for step in range(cfg.step_number):
            used = set(actions)
            feasible = np.asarray(
                [a for a in range(len(catalog)) if a not in used], dtype=int
            )
            fixed_scores = _fixed_utility(
                len(catalog), fixed_sequence, actions
            )
            p_before = normalize_log_weights(log_w)
            h_before = posterior_entropy(p_before)
            best_dh = -np.inf
            myopic_action = int(feasible[0])
            myopic_scores = np.full(len(catalog), -np.inf, dtype=np.float64)
            for action in feasible:
                centres = y_sim_last_step_from_tables(table_support, [int(action)])
                y_hat = float(np.sum(p_before * centres))
                after = normalize_log_weights(
                    log_w
                    + log_gaussian_observation_density(
                        y_hat, centres, cfg.sigma_y
                    )
                )
                dh = clamp_info_gain(h_before - posterior_entropy(after))
                myopic_scores[int(action)] = float(dh)
                if dh > best_dh:
                    best_dh = dh
                    myopic_action = int(action)
            if actions:
                act_t = torch.tensor([actions], dtype=torch.long, device=device)
                obs_t = torch.tensor(
                    [observations], dtype=torch.float32, device=device
                )
                mask_t = torch.ones(1, len(actions), device=device)
            else:
                act_t = torch.zeros(1, 0, dtype=torch.long, device=device)
                obs_t = torch.zeros(1, 0, device=device)
                mask_t = torch.zeros(1, 0, device=device)
            feasible_t = torch.ones(
                1, len(catalog), dtype=torch.bool, device=device
            )
            if actions:
                feasible_t[:, torch.as_tensor(actions, device=device)] = False
            logits_t = policy(act_t, obs_t, mask_t, feasible_t)
            probs_t = torch.softmax(logits_t, dim=-1)
            log_probs_t = torch.log(probs_t.clamp_min(1e-12))
            dad_entropy_t = -(probs_t * log_probs_t).sum(dim=-1)
            dad_scores = logits_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
            dad_action = int(torch.argmax(logits_t, dim=-1).item())
            n_feasible = max(len(feasible), 1)
            entropy_norm = float(dad_entropy_t.item()) / max(
                math.log(n_feasible), 1e-12
            )
            ess_fraction = float(1.0 / np.sum(p_before * p_before)) / float(
                len(p_before)
            )
            weights = router_weights(
                step=step,
                horizon=cfg.step_number,
                ess_fraction=ess_fraction,
                dad_normalized_entropy=entropy_norm,
                config=config,
            )
            features = _action_features(
                fixed_scores,
                myopic_scores,
                dad_scores,
                feasible,
                step=step,
                horizon=cfg.step_number,
                ess_fraction=ess_fraction,
                dad_normalized_entropy=entropy_norm,
                belief_features=_eig_belief_features(
                    table_support, p_before
                ),
            )
            residual_values = (
                _critic_values(critic, critic_norm, features, device)
                if config.residual_enabled
                else None
            )
            action, _ = fuse_action_scores(
                fixed_scores,
                myopic_scores,
                dad_scores,
                feasible,
                weights,
                residual_values=residual_values,
                residual_scale=config.residual_scale,
            )
            if forced_prefix is not None and step < len(forced_prefix):
                forced = int(forced_prefix[step])
                if forced not in set(feasible.tolist()):
                    raise ValueError(
                        f"Forced action {forced} is infeasible at step {step}"
                    )
                action = forced
            fixed_action = int(feasible[np.argmax(fixed_scores[feasible])])
            y = float(lookup_action_y(system, action))
            centres = y_sim_last_step_from_tables(table_support, [action])
            log_w = log_w + log_gaussian_observation_density(
                y, centres, cfg.sigma_y
            )
            trace.append(
                {
                    "step": step,
                    "router_weights": {
                        "fixed": float(weights[0]),
                        "myopic": float(weights[1]),
                        "dad": float(weights[2]),
                    },
                    "action": action,
                    "fixed_action": fixed_action,
                    "myopic_action": myopic_action,
                    "dad_action": dad_action,
                    "ess_fraction": ess_fraction,
                    "dad_normalized_entropy": entropy_norm,
                    "action_features": features.tolist(),
                    "critic_values": (
                        residual_values.tolist()
                        if residual_values is not None
                        else None
                    ),
                }
            )
            actions.append(action)
            observations.append(y)
        outputs.append(
            {
                "M": system["M"],
                "K": system["K"],
                "sequence": actions,
                "y": observations,
                "router_trace": trace,
            }
        )
    return outputs


def _router_candidates() -> list[tuple[str, MoERouterConfig]]:
    """Small predeclared family; includes exact near-pure parent fallbacks."""
    common = {
        "fixed_ess_coefficient": 0.0,
        "myopic_progress_coefficient": 0.0,
        "myopic_dad_uncertainty_coefficient": 0.0,
        "dad_future_coefficient": 0.0,
        "dad_confidence_coefficient": 0.0,
        "temperature": 0.5,
        "residual_enabled": False,
    }
    return [
        (
            "myopic_anchor",
            MoERouterConfig(**common, fixed_bias=-12.0, myopic_bias=12.0, dad_bias=-12.0),
        ),
        (
            "fixed_anchor",
            MoERouterConfig(**common, fixed_bias=12.0, myopic_bias=-12.0, dad_bias=-12.0),
        ),
        (
            "dad_anchor",
            MoERouterConfig(**common, fixed_bias=-12.0, myopic_bias=-12.0, dad_bias=12.0),
        ),
        (
            "myopic_dad",
            MoERouterConfig(
                fixed_bias=-3.0, myopic_bias=2.0, dad_bias=1.0,
                residual_enabled=False,
            ),
        ),
        (
            "myopic_fixed_dad",
            MoERouterConfig(
                fixed_bias=0.0, myopic_bias=2.0, dad_bias=1.0,
                residual_enabled=False,
            ),
        ),
        (
            "balanced",
            MoERouterConfig(
                fixed_bias=0.0, myopic_bias=0.0, dad_bias=0.0,
                residual_enabled=False,
            ),
        ),
    ]


def calibrate_eig_moe(
    run,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train and calibrate the EIG residual router on held-out systems."""
    from src.inference.spce import (
        log_gaussian_observation_density,
        normalize_log_weights,
        posterior_entropy,
    )
    from src.domains.swing.design import build_catalog
    from src.inference.scoring import TableThetaSupport, y_sim_last_step_from_tables

    n_val = min(8 if smoke else 32, max(len(run.train_systems) // 8, 1))
    validation_systems = run.train_systems[-n_val:]
    support_systems = run.train_systems[:-n_val] or run.train_systems
    support = TableThetaSupport.from_train(
        support_systems,
        run.cfg,
        np.random.default_rng(int(run.cfg.prior.get("mc_support_seed", 1))),
    )
    catalog = build_catalog(run.cfg)
    h0 = posterior_entropy(normalize_log_weights(support.log_p0))
    rows = []
    for name, candidate in _router_candidates():
        rollouts = rollout_eig_moe(
            run.cfg,
            run.exp_dir,
            validation_systems,
            catalog,
            support,
            config_override=candidate,
        )
        terminal = []
        for rollout in rollouts:
            log_w = np.array(support.log_p0, dtype=np.float64)
            for action, y_obs in zip(rollout["sequence"], rollout["y"]):
                centres = y_sim_last_step_from_tables(support, [int(action)])
                log_w += log_gaussian_observation_density(
                    float(y_obs), centres, run.cfg.sigma_y
                )
            terminal.append(
                h0 - posterior_entropy(normalize_log_weights(log_w))
            )
        rows.append(
            {
                "name": name,
                "mean_validation_terminal_eig": float(np.mean(terminal)),
                "config": asdict(candidate),
            }
        )
    selected = max(rows, key=lambda row: row["mean_validation_terminal_eig"])
    base_config = MoERouterConfig(**selected["config"])
    critic_systems = validation_systems[: max(1, n_val // 2)]
    scale_systems = validation_systems[max(1, n_val // 2) :] or validation_systems[-1:]

    def terminal_eig(rollout: dict[str, Any]) -> float:
        log_w = np.array(support.log_p0, dtype=np.float64)
        for action, y_obs in zip(rollout["sequence"], rollout["y"]):
            centres = y_sim_last_step_from_tables(support, [int(action)])
            log_w += log_gaussian_observation_density(
                float(y_obs), centres, run.cfg.sigma_y
            )
        return float(
            h0 - posterior_entropy(normalize_log_weights(log_w))
        )

    critic_features: list[np.ndarray] = []
    critic_targets: list[float] = []
    for system in critic_systems:
        base = rollout_eig_moe(
            run.cfg,
            run.exp_dir,
            [system],
            catalog,
            support,
            config_override=base_config,
        )[0]
        prefix: list[int] = []
        for step, trace in enumerate(base["router_trace"]):
            feature_matrix = np.asarray(trace["action_features"], dtype=np.float32)
            feasible = [
                a for a in range(len(catalog)) if a not in set(prefix)
            ]
            candidates = _counterfactual_candidates(
                feature_matrix, feasible
            )
            for candidate_action in candidates:
                counterfactual = rollout_eig_moe(
                    run.cfg,
                    run.exp_dir,
                    [system],
                    catalog,
                    support,
                    config_override=base_config,
                    forced_prefix=prefix + [int(candidate_action)],
                )[0]
                critic_features.append(feature_matrix[int(candidate_action)])
                critic_targets.append(terminal_eig(counterfactual))
            prefix.append(int(base["sequence"][step]))
    critic_report = _fit_residual_critic(
        run.exp_dir,
        critic_features,
        critic_targets,
        objective="terminal_eig",
        seed=int(run.meta.train_seed),
        epochs=80 if smoke else 180,
    )
    scale_rows = []
    for scale in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        candidate = MoERouterConfig(
            **{
                **asdict(base_config),
                "residual_enabled": bool(scale > 0.0),
                "residual_scale": float(scale),
            }
        )
        rollouts = rollout_eig_moe(
            run.cfg,
            run.exp_dir,
            scale_systems,
            catalog,
            support,
            config_override=candidate,
        )
        scale_rows.append(
            {
                "residual_scale": float(scale),
                "mean_validation_terminal_eig": float(
                    np.mean([terminal_eig(r) for r in rollouts])
                ),
                "per_system_metric": [
                    float(terminal_eig(r)) for r in rollouts
                ],
                "config": asdict(candidate),
            }
        )
    baseline_scale = next(
        row for row in scale_rows if row["residual_scale"] == 0.0
    )
    selected_scale = baseline_scale
    for row in scale_rows:
        if row["residual_scale"] == 0.0:
            continue
        improvement = np.asarray(row["per_system_metric"]) - np.asarray(
            baseline_scale["per_system_metric"]
        )
        se = (
            float(np.std(improvement, ddof=1)) / math.sqrt(len(improvement))
            if len(improvement) > 1
            else float("inf")
        )
        lower = float(np.mean(improvement)) - 1.645 * se
        row["paired_improvement_lcb90"] = lower
        if (
            lower > 0.0
            and row["mean_validation_terminal_eig"]
            > selected_scale["mean_validation_terminal_eig"]
        ):
            selected_scale = row
    report = prepare_moe(
        run.exp_dir,
        experiment_type="eig_based",
        config=MoERouterConfig(**selected_scale["config"]),
    )
    report.update(
        {
            "selection_split": "held-out training tail",
            "n_validation_systems": n_val,
            "selected_candidate": selected["name"],
            "validation_terminal_eig": selected_scale[
                "mean_validation_terminal_eig"
            ],
            "candidates": rows,
            "critic": critic_report,
            "residual_scale_candidates": scale_rows,
            "selected_residual_scale": selected_scale["residual_scale"],
            "do_no_harm_fallback": "myopic_anchor",
        }
    )
    path = moe_config_path(run.exp_dir)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
