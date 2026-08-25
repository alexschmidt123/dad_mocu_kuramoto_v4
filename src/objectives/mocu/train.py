"""PPO training for DAD (terminal) and RL-sBOED (stepwise) with vector Δf obs."""

from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.objectives.mocu.context import (
    ExperimentContext,
    belief_summary,
    control_from_log_weights,
    observe_compressed,
    posterior_mocu,
    update_posterior_vector,
)
from src.objectives.mocu.rewards import (
    dad_rewards,
    safety_aware_control_cost,
    verify_rl_sboed_rollout,
)
from src.policies.rl_sboed import (
    AdaptiveExperimentPolicy,
    PolicyConfig,
    StateValueCritic,
)
from src.policies.moe import (
    BeliefConditionedMoEPolicy,
    SharedBaseResidualMoEPolicy,
    parameter_matched_expert_hidden,
)
from src.layout import ensure_result_layout, model_dir

RewardMode = Literal["dad_terminal", "rl_sboed_stepwise"]


def _architecture_tag(policy: Any) -> str:
    if isinstance(policy, SharedBaseResidualMoEPolicy):
        return "shared_base_top2_residual_moe_v2"
    if isinstance(policy, BeliefConditionedMoEPolicy):
        if int(policy.n_experts) == 1:
            return "parameter_matched_dense_control_v1"
        # Shared base + belief-conditioned residual experts (softplus scale).
        return "shared_base_belief_residual_top2_moe_v3"
    return "dense_policy"


def _method_stem(method: str) -> str:
    """Canonical filename stem for a method display name or key."""
    from src.objectives.mocu.context import (
        METHOD_ALIASES,
        normalize_method_key,
    )

    try:
        return normalize_method_key(method)
    except ValueError:
        lowered = method.replace("-", "_").lower()
        for key, display in METHOD_ALIASES.items():
            if display.lower().replace("-", "_") == lowered:
                return key
        return lowered


@dataclass
class TrainConfig:
    """PPO budget for objective-based DAD / RL-sBOED (offline bank rollouts)."""

    updates: int = 120
    # Effective updates = max(updates, min_updates_per_horizon * T).
    min_updates_per_horizon: int = 0
    trajectories_per_update: int = 16
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    gae_lambda: float = 1.0
    # Default kept high enough to avoid open-loop collapse (was 0.005).
    entropy_coefficient: float = 0.05
    # Prespecified linear annealing keeps early exploration while allowing the
    # final deterministic policy to specialize.  A fraction <= 0 preserves the
    # historical constant-coefficient behavior.
    entropy_final_coefficient: float = 0.05
    entropy_anneal_fraction: float = 0.0
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    max_grad_norm: float = 1.0
    validation_interval: int = 10
    validation_rollouts: int = 64
    patience: int = 8
    gamma: float = 1.0
    device: str = "auto"
    # Joint checkpoint score = mean_mocu - weight * unique_frac (minimize).
    # Soft diversity bonus in MOCU units (unique_frac ∈ [0,1]).
    checkpoint_diversity_weight: float = 0.1
    # Soft unique floor: max(2, ceil(frac * n_rollouts)) when n_rollouts >= 2.
    min_unique_sequence_fraction: float = 0.05
    # When true, prefer floor-ok over collapsed only if raw MOCU is within
    # ``unique_floor_mocu_slack`` (does not allow arbitrarily worse MOCU).
    prefer_unique_sequence_floor: bool = False
    # Max raw-MOCU degradation allowed when upgrading collapsed → floor-ok.
    unique_floor_mocu_slack: float = 0.005
    # Warn if deterministic unique_seq stays <= 1 after this many updates (0=off).
    open_loop_warn_after_updates: int = 100
    # Goal-oriented utility C=u + lambda_gap*(u_req-u)_+ + lambda_event*1[unsafe].
    undercontrol_penalty: float = 10.0
    violation_penalty: float = 0.10
    # Checkpoint score adds this penalty per unit empirical under-control rate.
    checkpoint_safety_penalty: float = 10.0
    min_valid_safety_rate: float = 0.95
    # Dense DAD/RL policies otherwise see only the sampled action's return out
    # of a large discrete action set.  This auxiliary uses the same simulated
    # rollout batch to rank every feasible next experiment under the current
    # belief; it is objective supervision, not imitation of another method.
    dense_counterfactual_coefficient: float = 0.0
    dense_counterfactual_temperature: float = 0.35
    # Decision-sensitive post-prior branching. Pairs are regularized only when
    # their all-action MOCU rankings disagree; identical-prior states (step 0)
    # are explicitly excluded.
    dense_branching_coefficient: float = 0.0
    dense_branching_similarity_threshold: float = 0.5
    dense_branching_margin: float = 0.35
    # Supervised all-action counterfactual decision-value objective for MoE.
    # Default 0: IEEE-5 T=3 σ=0.01 ablations found pure PPO better than CF
    # warm-start / floor variants on terminal MOCU.  Set >0 to re-enable.
    moe_counterfactual_coefficient: float = 0.0
    # Linearly decay the counterfactual coefficient over this fraction of the
    # update budget when the coefficient is >0.  <= 0 keeps it constant.
    moe_counterfactual_anneal_fraction: float = 0.33
    # Annealed CF weight floor as a fraction of moe_counterfactual_coefficient.
    moe_counterfactual_floor_fraction: float = 0.0
    # Bootstrap the counterfactual targets with the state-value critic:
    # q(a) = -MOCU_{t+1}(a) + gamma * V(h_{t+1}(a)), an estimate of negative
    # expected terminal MOCU.  Without the bootstrap the targets equal the
    # Myopic one-step criterion and cap the supervised signal at myopic
    # behaviour.
    moe_counterfactual_bootstrap: bool = True
    # Weight of the observation-branching regularizer: same-step states whose
    # counterfactual rankings disagree must receive different action
    # distributions.  Default 0 with CF-off training; only meaningful when CF
    # targets are computed.
    moe_branching_coefficient: float = 0.0
    # Re-run farthest-point prototype initialization once at this fraction of
    # the update budget, replacing prototypes anchored on noisy early
    # fingerprints.  <= 0 disables the reset.
    moe_prototype_reset_fraction: float = 0.2
    # Optional weak Fixed anchor on the shared base only (OFF by default).
    # Nonzero Fixed-BC makes MoE look like Fixed; keep 0 for distinctive MoE.
    moe_fixed_bc_coefficient: float = 0.0
    moe_fixed_bc_anneal_fraction: float = 0.5
    moe_fixed_bc_floor_fraction: float = 0.0
    # belief_summary[:, 1] is ESS / n_particles.  Gate weight rises above this.
    moe_fixed_bc_ess_threshold: float = 0.4
    # Low-ESS residual influence: maximize KL(π_final || π_base) when ESS is
    # below the threshold so experts must move the decision once the posterior
    # is informative.
    moe_low_ess_residual_coefficient: float = 0.20
    moe_low_ess_threshold: float = 0.4
    # Softly keep softplus(logit_scale) near a target so residual experts can
    # outrank the base without Fixed cloning.
    moe_residual_scale_coefficient: float = 0.02
    moe_logit_scale_target: float = 3.5
    # Initial softplus(logit_scale) ≈ softplus(init).
    moe_logit_scale_init: float = 3.0
    # Soft router load-balancing and expert-redundancy penalties.
    moe_balance_coefficient: float = 0.001
    moe_redundancy_coefficient: float = 0.01
    # Optimizer for DAD.  "reinforce" reproduces the published DAD update
    # (single score-function pass); "ppo" runs the same clipped multi-epoch
    # update as RL-sBOED.  Setting this to "ppo" while keeping DAD's terminal
    # reward isolates the credit-assignment effect (hypothesis H4) from the
    # optimizer, which the REINFORCE-vs-PPO default otherwise confounds.
    dad_optimizer: str = "reinforce"
    # MoE architecture knobs.  moe_n_experts=1 with a widened moe_expert_hidden
    # is the parameter-matched dense control that isolates the mixture's
    # contribution from raw capacity and the shared supervision losses.
    moe_n_experts: int = 4
    moe_top_k: int = 2
    moe_expert_hidden: int = 0  # 0 => default to policy hidden width

    @classmethod
    def from_cfg(cls, training: dict[str, Any] | None) -> "TrainConfig":
        """Load MOCU knobs from an already-resolved ``objective_based`` training dict.

        Callers must pass ``cfg.training_for("objective_based")`` (or the nested
        ``training.objective_based`` subtree). EIG keys must not appear here.
        """
        raw = dict(training or {})
        if any(str(k).startswith("eig_") for k in raw) or "eig_based" in raw:
            raise ValueError(
                "MOCU TrainConfig received EIG training keys; pass "
                "cfg.training_for('objective_based') so MOCU and EIG stay independent"
            )
        cfg = cls()
        updates = raw.get("updates", raw.get("epochs"))
        if updates is not None:
            cfg.updates = int(updates)
        if "min_updates_per_horizon" in raw:
            cfg.min_updates_per_horizon = int(raw["min_updates_per_horizon"])
        traj = raw.get("trajectories_per_update", raw.get("batch_size"))
        if traj is not None:
            cfg.trajectories_per_update = int(traj)
        if "ppo_epochs" in raw:
            cfg.ppo_epochs = int(raw["ppo_epochs"])
        if "ppo_clip" in raw:
            cfg.ppo_clip = float(raw["ppo_clip"])
        if "gae_lambda" in raw:
            cfg.gae_lambda = float(raw["gae_lambda"])
        ent = raw.get("entropy_coefficient", raw.get("entropy_coef"))
        if ent is not None:
            cfg.entropy_coefficient = float(ent)
        if "entropy_final_coefficient" in raw:
            cfg.entropy_final_coefficient = float(
                raw["entropy_final_coefficient"]
            )
        if "entropy_anneal_fraction" in raw:
            cfg.entropy_anneal_fraction = float(raw["entropy_anneal_fraction"])
        lr = raw.get("learning_rate")
        if "actor_lr" in raw:
            cfg.actor_lr = float(raw["actor_lr"])
        elif lr is not None:
            cfg.actor_lr = float(lr)
        if "critic_lr" in raw:
            cfg.critic_lr = float(raw["critic_lr"])
        elif lr is not None:
            cfg.critic_lr = float(lr)
        grad = raw.get("max_grad_norm", raw.get("grad_clip"))
        if grad is not None:
            cfg.max_grad_norm = float(grad)
        if "validation_interval" in raw:
            cfg.validation_interval = int(raw["validation_interval"])
        if "validation_rollouts" in raw:
            cfg.validation_rollouts = int(raw["validation_rollouts"])
        if "patience" in raw:
            cfg.patience = int(raw["patience"])
        if "gamma" in raw:
            cfg.gamma = float(raw["gamma"])
        if "device" in raw:
            cfg.device = str(raw["device"])
        if "checkpoint_diversity_weight" in raw:
            cfg.checkpoint_diversity_weight = float(raw["checkpoint_diversity_weight"])
        if "min_unique_sequence_fraction" in raw:
            cfg.min_unique_sequence_fraction = float(raw["min_unique_sequence_fraction"])
        if "prefer_unique_sequence_floor" in raw:
            cfg.prefer_unique_sequence_floor = bool(
                raw["prefer_unique_sequence_floor"]
            )
        if "unique_floor_mocu_slack" in raw:
            cfg.unique_floor_mocu_slack = float(raw["unique_floor_mocu_slack"])
        warn_u = raw.get(
            "open_loop_warn_after_updates", raw.get("open_loop_warn_updates")
        )
        if warn_u is not None:
            cfg.open_loop_warn_after_updates = int(warn_u)
        if "undercontrol_penalty" in raw:
            cfg.undercontrol_penalty = float(raw["undercontrol_penalty"])
        if "violation_penalty" in raw:
            cfg.violation_penalty = float(raw["violation_penalty"])
        if "checkpoint_safety_penalty" in raw:
            cfg.checkpoint_safety_penalty = float(raw["checkpoint_safety_penalty"])
        if "min_valid_safety_rate" in raw:
            cfg.min_valid_safety_rate = float(raw["min_valid_safety_rate"])
        if "dense_counterfactual_coefficient" in raw:
            cfg.dense_counterfactual_coefficient = float(
                raw["dense_counterfactual_coefficient"]
            )
        if "dense_counterfactual_temperature" in raw:
            cfg.dense_counterfactual_temperature = float(
                raw["dense_counterfactual_temperature"]
            )
        if "dense_branching_coefficient" in raw:
            cfg.dense_branching_coefficient = float(
                raw["dense_branching_coefficient"]
            )
        if "dense_branching_similarity_threshold" in raw:
            cfg.dense_branching_similarity_threshold = float(
                raw["dense_branching_similarity_threshold"]
            )
        if "dense_branching_margin" in raw:
            cfg.dense_branching_margin = float(raw["dense_branching_margin"])
        if "moe_counterfactual_coefficient" in raw:
            cfg.moe_counterfactual_coefficient = float(
                raw["moe_counterfactual_coefficient"]
            )
        if "moe_counterfactual_anneal_fraction" in raw:
            cfg.moe_counterfactual_anneal_fraction = float(
                raw["moe_counterfactual_anneal_fraction"]
            )
        if "moe_counterfactual_floor_fraction" in raw:
            cfg.moe_counterfactual_floor_fraction = float(
                raw["moe_counterfactual_floor_fraction"]
            )
        if "moe_counterfactual_bootstrap" in raw:
            cfg.moe_counterfactual_bootstrap = bool(
                raw["moe_counterfactual_bootstrap"]
            )
        if "moe_branching_coefficient" in raw:
            cfg.moe_branching_coefficient = float(
                raw["moe_branching_coefficient"]
            )
        if "moe_prototype_reset_fraction" in raw:
            cfg.moe_prototype_reset_fraction = float(
                raw["moe_prototype_reset_fraction"]
            )
        if "moe_balance_coefficient" in raw:
            cfg.moe_balance_coefficient = float(raw["moe_balance_coefficient"])
        if "moe_redundancy_coefficient" in raw:
            cfg.moe_redundancy_coefficient = float(
                raw["moe_redundancy_coefficient"]
            )
        if "moe_logit_scale_init" in raw:
            cfg.moe_logit_scale_init = float(raw["moe_logit_scale_init"])
        if "moe_fixed_bc_coefficient" in raw:
            cfg.moe_fixed_bc_coefficient = float(raw["moe_fixed_bc_coefficient"])
        if "moe_fixed_bc_anneal_fraction" in raw:
            cfg.moe_fixed_bc_anneal_fraction = float(
                raw["moe_fixed_bc_anneal_fraction"]
            )
        if "moe_fixed_bc_floor_fraction" in raw:
            cfg.moe_fixed_bc_floor_fraction = float(
                raw["moe_fixed_bc_floor_fraction"]
            )
        if "moe_fixed_bc_ess_threshold" in raw:
            cfg.moe_fixed_bc_ess_threshold = float(
                raw["moe_fixed_bc_ess_threshold"]
            )
        if "moe_low_ess_residual_coefficient" in raw:
            cfg.moe_low_ess_residual_coefficient = float(
                raw["moe_low_ess_residual_coefficient"]
            )
        if "moe_low_ess_threshold" in raw:
            cfg.moe_low_ess_threshold = float(raw["moe_low_ess_threshold"])
        if "moe_residual_scale_coefficient" in raw:
            cfg.moe_residual_scale_coefficient = float(
                raw["moe_residual_scale_coefficient"]
            )
        if "moe_logit_scale_target" in raw:
            cfg.moe_logit_scale_target = float(raw["moe_logit_scale_target"])
        if "dad_optimizer" in raw:
            optimizer = str(raw["dad_optimizer"]).lower()
            if optimizer not in ("reinforce", "ppo"):
                raise ValueError(
                    f"dad_optimizer must be 'reinforce' or 'ppo', got {optimizer!r}"
                )
            cfg.dad_optimizer = optimizer
        if "moe_n_experts" in raw:
            cfg.moe_n_experts = int(raw["moe_n_experts"])
        if "moe_top_k" in raw:
            cfg.moe_top_k = int(raw["moe_top_k"])
        if "moe_expert_hidden" in raw:
            cfg.moe_expert_hidden = int(raw["moe_expert_hidden"])
        return cfg


def _annealed_weight(
    *,
    coefficient: float,
    anneal_fraction: float,
    floor_fraction: float,
    updates: int,
    update: int,
) -> float:
    """Linear anneal of ``coefficient`` toward ``floor_fraction * coefficient``."""
    base = float(coefficient)
    if base <= 0.0:
        return 0.0
    floor = base * max(0.0, float(floor_fraction))
    fraction = float(anneal_fraction)
    if fraction <= 0.0:
        return base
    anneal_updates = max(1, int(round(fraction * int(updates))))
    annealed = base * max(0.0, 1.0 - (int(update) - 1) / anneal_updates)
    return max(annealed, floor)


def entropy_weight(config: "TrainConfig", update: int) -> float:
    """Prespecified linear entropy schedule for either reference optimizer.

    This changes only the exploration regularizer, not DAD's terminal
    REINFORCE objective or RL-sBOED's PPO/telescoping-return objective.
    """
    start = float(config.entropy_coefficient)
    final = float(config.entropy_final_coefficient)
    if start < 0.0 or final < 0.0:
        raise ValueError("entropy coefficients must be non-negative")
    fraction = float(config.entropy_anneal_fraction)
    if fraction <= 0.0:
        return start
    anneal_updates = max(1, int(round(fraction * int(config.updates))))
    progress = min(max((int(update) - 1) / anneal_updates, 0.0), 1.0)
    return start + progress * (final - start)


def moe_counterfactual_weight(config: "TrainConfig", update: int) -> float:
    """Annealed counterfactual-loss weight for training update ``update`` (1-based)."""
    return _annealed_weight(
        coefficient=config.moe_counterfactual_coefficient,
        anneal_fraction=config.moe_counterfactual_anneal_fraction,
        floor_fraction=config.moe_counterfactual_floor_fraction,
        updates=config.updates,
        update=update,
    )


def moe_fixed_bc_weight(config: "TrainConfig", update: int) -> float:
    """Annealed Fixed behavioral-cloning weight for update ``update`` (1-based)."""
    return _annealed_weight(
        coefficient=config.moe_fixed_bc_coefficient,
        anneal_fraction=config.moe_fixed_bc_anneal_fraction,
        floor_fraction=config.moe_fixed_bc_floor_fraction,
        updates=config.updates,
        update=update,
    )


def moe_residual_release_fraction(config: "TrainConfig", update: int) -> float:
    """How freely residual experts may act (1 = fully released).

    When Fixed-BC is disabled (default), residuals are fully released from the
    first update.  When Fixed-BC is enabled, release rises as Fixed-BC anneals.
    """
    base = float(config.moe_fixed_bc_coefficient)
    if base <= 0.0:
        return 1.0
    current = moe_fixed_bc_weight(config, update)
    return float(max(0.0, min(1.0, 1.0 - current / base)))


def low_ess_residual_influence_loss(
    policy: BeliefConditionedMoEPolicy,
    *inputs: torch.Tensor,
    feasible_mask: torch.Tensor,
    ess_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Encourage residual experts to move π away from the base when ESS is low."""
    return policy.low_ess_residual_loss(
        *inputs,
        feasible_mask=feasible_mask,
        ess_threshold=ess_threshold,
    )


def residual_scale_floor_loss(
    policy: BeliefConditionedMoEPolicy,
    *,
    target: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hinge penalty if softplus(logit_scale) falls below ``target``."""
    return policy.residual_scale_floor_loss(target=target)


def fixed_sequence_bc_loss(
    logits: torch.Tensor,
    steps: torch.Tensor,
    feasible_mask: torch.Tensor,
    fixed_sequence: list[int],
    *,
    belief_summary: torch.Tensor | None = None,
    ess_threshold: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy toward Fixed, optionally weighted by high ESS.

    When ``ess_threshold > 0`` and ``belief_summary`` is provided, each sample
    is weighted by ``clamp((ess - threshold) / (1 - threshold), 0, 1)`` so
    prior-like beliefs anchor to Fixed while peaked posteriors do not.
    """
    if not fixed_sequence:
        zero = logits.sum() * 0.0
        return zero, {
            "fixed_bc_loss": 0.0,
            "fixed_bc_n": 0.0,
            "fixed_bc_gate_mean": 0.0,
        }
    step = steps.reshape(-1).long()
    targets = torch.zeros_like(step)
    valid = torch.zeros_like(step, dtype=torch.bool)
    n_fixed = len(fixed_sequence)
    for s, action in enumerate(fixed_sequence):
        mask = step == int(s)
        if not bool(mask.any()):
            continue
        targets = torch.where(mask, torch.full_like(targets, int(action)), targets)
        # Feasible if the Fixed bus is still available at this state.
        feas = feasible_mask[mask, int(action)]
        valid_idx = mask.nonzero(as_tuple=False).reshape(-1)
        valid[valid_idx] = feas
    if not bool(valid.any()):
        zero = logits.sum() * 0.0
        return zero, {
            "fixed_bc_loss": 0.0,
            "fixed_bc_n": 0.0,
            "fixed_bc_gate_mean": 0.0,
        }
    masked_logits = logits.masked_fill(~feasible_mask, -1e9)
    thr = float(ess_threshold)
    if belief_summary is not None and thr > 0.0:
        ess = belief_summary.reshape(belief_summary.shape[0], -1)[:, 1]
        gate = ((ess - thr) / max(1e-6, 1.0 - thr)).clamp(0.0, 1.0)
        log_probs = F.log_softmax(masked_logits[valid], dim=-1)
        nll = -log_probs.gather(1, targets[valid].unsqueeze(1)).squeeze(1)
        w = gate[valid]
        w_sum = w.sum()
        if float(w_sum) <= 1e-8:
            zero = logits.sum() * 0.0
            return zero, {
                "fixed_bc_loss": 0.0,
                "fixed_bc_n": 0.0,
                "fixed_bc_gate_mean": 0.0,
            }
        loss = (nll * w).sum() / w_sum
        gate_mean = float(w.mean().detach())
    else:
        loss = F.cross_entropy(masked_logits[valid], targets[valid])
        gate_mean = 1.0
    return loss, {
        "fixed_bc_loss": float(loss.detach()),
        "fixed_bc_n": float(valid.sum().detach()),
        "fixed_bc_gate_mean": gate_mean,
    }


def sequence_diversity_stats(sequences: list[str]) -> dict[str, float | int]:
    """Unique-count and Shannon entropy of action-sequence strings."""
    n = len(sequences)
    if n == 0:
        return {
            "n_unique_sequences": 0,
            "sequence_entropy": 0.0,
            "unique_frac": 0.0,
        }
    unique = set(sequences)
    n_unique = len(unique)
    counts: dict[str, int] = {}
    for s in sequences:
        counts[s] = counts.get(s, 0) + 1
    probs = np.asarray([c / n for c in counts.values()], dtype=np.float64)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-300, 1.0))).sum())
    return {
        "n_unique_sequences": int(n_unique),
        "sequence_entropy": entropy,
        "unique_frac": float(n_unique / n),
    }


def dense_counterfactual_ranking_loss(
    logits: torch.Tensor,
    target_utility: torch.Tensor,
    feasible_mask: torch.Tensor,
    *,
    temperature: float = 0.35,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train a dense policy from belief-specific all-action utilities.

    Utilities are standardized within each state before forming a soft target,
    because absolute MOCU differences are small and vary with posterior depth.
    Infeasible (already selected) actions receive no probability mass.
    """
    feasible = feasible_mask.bool() & torch.isfinite(target_utility)
    count = feasible.sum(dim=-1, keepdim=True).clamp_min(1)
    safe = torch.where(feasible, target_utility.detach(), torch.zeros_like(target_utility))
    mean = safe.sum(dim=-1, keepdim=True) / count
    centered = torch.where(feasible, safe - mean, torch.zeros_like(safe))
    scale = torch.sqrt(
        (centered.square().sum(dim=-1, keepdim=True) / count).clamp_min(1e-8)
    )
    standardized = centered / scale
    target_logits = (standardized / max(float(temperature), 1e-3)).masked_fill(
        ~feasible, -1e9
    )
    target_prob = torch.softmax(target_logits, dim=-1)
    predicted = logits.masked_fill(~feasible, -1e9)
    loss = -(target_prob * F.log_softmax(predicted, dim=-1)).sum(dim=-1).mean()
    agreement = (
        predicted.argmax(dim=-1) == target_logits.argmax(dim=-1)
    ).float().mean()
    return loss, {
        "dense_cf_loss": float(loss.detach()),
        "dense_cf_top1_agreement": float(agreement.detach()),
    }


def dense_post_prior_branching_loss(
    logits: torch.Tensor,
    target_utility: torch.Tensor,
    feasible_mask: torch.Tensor,
    steps: torch.Tensor,
    *,
    similarity_threshold: float = 0.5,
    margin: float = 0.35,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Separate policies only for post-prior states needing different actions.

    Counterfactual utility fingerprints determine whether two same-stage
    posterior states disagree. The loss never acts at step zero and never
    rewards diversity between states with compatible MOCU rankings.
    """
    feasible = feasible_mask.bool() & torch.isfinite(target_utility)
    count = feasible.sum(dim=-1, keepdim=True).clamp_min(1)
    safe = torch.where(feasible, target_utility.detach(), torch.zeros_like(target_utility))
    mean = safe.sum(dim=-1, keepdim=True) / count
    centered = torch.where(feasible, safe - mean, torch.zeros_like(safe))
    scale = torch.sqrt(
        (centered.square().sum(dim=-1, keepdim=True) / count).clamp_min(1e-8)
    )
    fingerprints = torch.where(feasible, centered / scale, torch.zeros_like(centered))
    unit = F.normalize(fingerprints, dim=-1, eps=1e-8)
    similarity = unit @ unit.transpose(0, 1)

    masked_logits = logits.clamp(-50.0, 50.0).masked_fill(~feasible, -1e9)
    probs = torch.softmax(masked_logits, dim=-1)
    stage = steps.reshape(-1)
    same_stage = stage[:, None] == stage[None, :]
    post_prior = (stage > 0)[:, None] & (stage > 0)[None, :]
    off_diagonal = ~torch.eye(
        probs.shape[0], dtype=torch.bool, device=probs.device
    )
    disagree = (
        (similarity < float(similarity_threshold))
        & same_stage
        & post_prior
        & off_diagonal
    )
    if not bool(disagree.any()):
        zero = logits.sum() * 0.0
        return zero, {
            "dense_branching_pairs": 0.0,
            "dense_branching_loss": 0.0,
            "dense_branching_step0_pairs": 0.0,
        }
    total_variation = 0.5 * (
        probs[:, None, :] - probs[None, :, :]
    ).abs().sum(dim=-1)
    loss = F.relu(float(margin) - total_variation)[disagree].mean()
    return loss, {
        "dense_branching_pairs": float(disagree.sum().detach()) / 2.0,
        "dense_branching_loss": float(loss.detach()),
        "dense_branching_step0_pairs": 0.0,
        "dense_branching_tv": float(total_variation[disagree].mean().detach()),
    }


def checkpoint_score(
    mean_u_ctrl: float,
    n_unique: int,
    n_rollouts: int,
    *,
    diversity_weight: float,
    min_unique_fraction: float,
    under_control_rate: float = 0.0,
    safety_penalty: float = 0.0,
    min_safety_rate: float = 0.95,
) -> tuple[float, bool]:
    """Joint checkpoint score (lower is better).

    ``score = mean_mocu - diversity_weight * unique_frac`` for safety-valid
    policies, so modest diversity is rewarded without ignoring MOCU. Unsafe
    policies stay in the 1000+ band. ``meets_floor`` is a soft-gate diagnostic
    for ``should_replace_checkpoint``.
    """
    n = max(int(n_rollouts), 1)
    unique_frac = float(n_unique) / float(n)
    min_unique = 1
    if n >= 2:
        min_unique = max(2, int(np.ceil(float(min_unique_fraction) * n)))
    meets_floor = int(n_unique) >= int(min_unique)
    safety_rate = 1.0 - float(under_control_rate)
    if safety_rate < float(min_safety_rate):
        # Invalid checkpoints cannot beat any valid checkpoint.
        score = (
            1_000.0
            + float(safety_penalty) * (float(min_safety_rate) - safety_rate)
            + float(mean_u_ctrl)
        )
    else:
        score = float(mean_u_ctrl) - float(diversity_weight) * unique_frac
    return score, meets_floor


# Safety-invalid checkpoints use score = 1000 + … in ``checkpoint_score``.
_CHECKPOINT_UNSAFE_SCORE = 500.0


def checkpoint_rank_key(
    *,
    score: float,
    n_unique: int,
) -> tuple:
    """Lexicographic rank on joint score (lower tuple is better)."""
    unsafe = float(score) >= _CHECKPOINT_UNSAFE_SCORE
    return (unsafe, float(score), -int(n_unique))


def should_replace_checkpoint(
    *,
    score: float,
    meets_floor: bool,
    n_unique: int,
    best_score: float,
    best_meets_floor: bool,
    best_unique: int,
    prefer_unique_floor: bool,
    mean_mocu: float | None = None,
    best_mean_mocu: float | None = None,
    floor_mocu_slack: float = 0.0,
) -> bool:
    """Whether the candidate validation checkpoint should replace the best.

    Primary rule: lower joint ``score`` (MOCU − λ·unique_frac) wins.

    Soft unique floor (``prefer_unique_floor``): a floor-ok candidate may beat a
    collapsed best only if its raw MOCU is within ``floor_mocu_slack`` of the
    collapsed MOCU. A collapsed candidate may displace a floor-ok best only if
    its raw MOCU is better by more than the slack (and joint score is better).
    """
    cand_mocu = float(score if mean_mocu is None else mean_mocu)
    best_mocu = float(best_score if best_mean_mocu is None else best_mean_mocu)
    slack = max(float(floor_mocu_slack), 0.0)

    if prefer_unique_floor and bool(meets_floor) != bool(best_meets_floor):
        cand_unsafe = float(score) >= _CHECKPOINT_UNSAFE_SCORE
        best_unsafe = float(best_score) >= _CHECKPOINT_UNSAFE_SCORE
        if cand_unsafe and not best_unsafe:
            return False
        if meets_floor and not best_meets_floor:
            # Upgrade collapsed → diverse only if MOCU is preserved within slack.
            if cand_mocu <= best_mocu + slack + 1e-12:
                return True
            # Otherwise allow only if joint score is still better.
            return checkpoint_rank_key(
                score=float(score), n_unique=int(n_unique)
            ) < checkpoint_rank_key(
                score=float(best_score), n_unique=int(best_unique)
            )
        # meets_floor False, best_meets_floor True: demote only for clear MOCU win.
        if (
            float(score) < float(best_score) - 1e-12
            and cand_mocu + slack < best_mocu - 1e-12
        ):
            return True
        return False

    return checkpoint_rank_key(
        score=float(score), n_unique=int(n_unique)
    ) < checkpoint_rank_key(
        score=float(best_score), n_unique=int(best_unique)
    )


def _resolve_device(spec: str) -> torch.device:
    s = (spec or "auto").strip().lower()
    if s == "cpu":
        return torch.device("cpu")
    if s.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"training.device={spec!r} but CUDA is unavailable")
        return torch.device(s if s != "cuda" else "cuda")
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown training.device {spec!r} (use auto|cpu|cuda)")


def _policy_config(ctx: ExperimentContext) -> PolicyConfig:
    return PolicyConfig(
        max_steps=ctx.horizon,
        obs_dim=ctx.obs_dim,
        summary_dim=33,
        particle_dim=int(ctx.particle_features.shape[1]),
    )


def _tensors_from_state(
    ctx: ExperimentContext,
    *,
    actions: list[int],
    observations: list[np.ndarray],
    log_w: np.ndarray,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    length = int(ctx.horizon)
    action_idx = torch.zeros(1, length, dtype=torch.long, device=device)
    obs = torch.zeros(1, length, ctx.obs_dim, dtype=torch.float32, device=device)
    mask = torch.zeros(1, length, dtype=torch.float32, device=device)
    if actions:
        n = len(actions)
        action_idx[0, :n] = torch.as_tensor(actions, dtype=torch.long)
        y = np.stack(
            [
                (np.asarray(o, dtype=np.float64) - ctx.obs_mean) / ctx.obs_std
                for o in observations
            ],
            axis=0,
        )
        obs[0, :n] = torch.as_tensor(y, dtype=torch.float32)
        mask[0, :n] = 1.0
    belief = torch.as_tensor(
        belief_summary(ctx, log_w, observations)[None, :],
        dtype=torch.float32,
        device=device,
    )
    steps = torch.as_tensor([step], dtype=torch.long, device=device)
    particles = torch.as_tensor(
        ctx.particle_features[None, :], dtype=torch.float32, device=device
    )
    weights = torch.as_tensor(
        np.exp(log_w - np.max(log_w))[None, :], dtype=torch.float32, device=device
    )
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    # Measurement times in SIR are chronological. Power-grid duration probes
    # are independent experiments and may be selected in any order; selected
    # durations are simply removed below by the non-chronological mask.
    chrono = str(getattr(ctx, "observation_mode", "")).startswith("sir_")
    if chrono:
        from src.domains.sir.design import chronological_feasible

        remaining = max(int(ctx.horizon) - int(step), 0)
        allowed = set(
            chronological_feasible(
                ctx.n_actions,
                list(actions),
                remaining_steps=remaining,
            ).tolist()
        )
        feasible = torch.zeros(1, ctx.n_actions, dtype=torch.bool, device=device)
        for a in allowed:
            feasible[0, int(a)] = True
    else:
        feasible = torch.ones(1, ctx.n_actions, dtype=torch.bool, device=device)
        for a in actions:
            feasible[0, int(a)] = False
    return action_idx, obs, mask, belief, steps, particles, weights, feasible


def sample_trajectory(
    ctx: ExperimentContext,
    policy: AdaptiveExperimentPolicy,
    system_row: dict[str, Any],
    *,
    theta_id: int,
    rollout_id: int,
    global_seed: int,
    reward_mode: RewardMode,
    device: torch.device,
    deterministic: bool = False,
) -> dict[str, Any]:
    from src.observations.carry_state import (
        make_carry_observer,
        use_carry_state_observation,
    )

    log_w = ctx.log_p0.copy()
    actions: list[int] = []
    observations: list[np.ndarray] = []
    u_path = [control_from_log_weights(ctx, log_w).u_ctrl]
    mocu_path = [posterior_mocu(ctx, log_w)]
    states: list[tuple[torch.Tensor, ...]] = []
    log_probs: list[float] = []
    carry = (
        make_carry_observer(ctx, system_row)
        if use_carry_state_observation(ctx, for_training=True)
        else None
    )
    policy.eval()
    for step in range(ctx.horizon):
        tensors = _tensors_from_state(
            ctx,
            actions=actions,
            observations=observations,
            log_w=log_w,
            step=step,
            device=device,
        )
        states.append(tensors)
        with torch.no_grad():
            dist = policy.distribution(*tensors[:-1], tensors[-1])
            if deterministic:
                action = int(torch.argmax(dist.probs, dim=-1).item())
            else:
                action = int(dist.sample().item())
            log_probs.append(
                float(dist.log_prob(torch.tensor(action, device=device)).item())
            )
        if carry is not None:
            y = carry.observe_noisy(
                action,
                sigma_y=ctx.sigma_y,
                global_seed=global_seed,
                theta_id=theta_id,
                rollout_id=rollout_id,
                step=step,
                n_obs=ctx.n_obs,
            )
        else:
            y = observe_compressed(
                system_row,
                action,
                sigma_y=ctx.sigma_y,
                n_obs=ctx.n_obs,
                global_seed=global_seed,
                theta_id=theta_id,
                rollout_id=rollout_id,
                step=step,
            )
        actions.append(action)
        observations.append(y)
        log_w = update_posterior_vector(ctx, log_w, action, y)
        u_path.append(control_from_log_weights(ctx, log_w).u_ctrl)
        mocu_path.append(posterior_mocu(ctx, log_w))

    if reward_mode == "dad_terminal":
        trace = dad_rewards(mocu_path)
    else:
        trace = verify_rl_sboed_rollout(mocu_path)
    return {
        "actions": actions,
        "observations": observations,
        "u_path": u_path,
        "posterior_mocu_path": mocu_path,
        "rewards": list(trace.rewards),
        "log_probs": log_probs,
        "states": states,
        "terminal_u_ctrl": float(u_path[-1]),
        "terminal_posterior_mocu": float(mocu_path[-1]),
        "theta_id": theta_id,
        "log_w": log_w,
    }


def _gae(
    rewards: np.ndarray, values: np.ndarray, lam: float, gamma: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        advantages[t] = last
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


@torch.no_grad()
def evaluate_policy(
    ctx: ExperimentContext,
    policy: AdaptiveExperimentPolicy,
    systems: list[dict[str, Any]],
    *,
    n_rollouts: int,
    global_seed: int,
    reward_mode: RewardMode,
    device: torch.device,
    deterministic: bool = True,
) -> dict[str, Any]:
    rows = []
    for rid in range(n_rollouts):
        tid = int(rid % len(systems))
        traj = sample_trajectory(
            ctx,
            policy,
            systems[tid],
            theta_id=tid,
            rollout_id=rid,
            global_seed=global_seed,
            reward_mode=reward_mode,
            device=device,
            deterministic=deterministic,
        )
        rows.append(
            {
                "rollout_id": rid,
                "theta_id": tid,
                "sequence": " ".join(map(str, traj["actions"])),
                "u_ctrl": traj["terminal_u_ctrl"],
                "u_req": float(systems[tid]["u_req"]),
                "eval_mode": "deterministic" if deterministic else "stochastic",
            }
        )
    u = np.asarray([r["u_ctrl"] for r in rows], dtype=np.float64)
    u_req = np.asarray([r["u_req"] for r in rows], dtype=np.float64)
    realized_mocu = np.asarray(
        [
            safety_aware_control_cost(
                r["u_ctrl"],
                r["u_req"],
                undercontrol_penalty=ctx.undercontrol_penalty,
                violation_penalty=ctx.violation_penalty,
            )
            - r["u_req"]
            for r in rows
        ],
        dtype=np.float64,
    )
    div = sequence_diversity_stats([r["sequence"] for r in rows])
    return {
        "mean_u_ctrl": float(u.mean()) if u.size else float("nan"),
        "median_u_ctrl": float(np.median(u)) if u.size else float("nan"),
        "std_u_ctrl": float(u.std()) if u.size else float("nan"),
        "under_control_rate": float(np.mean(u + 1e-12 < u_req)),
        "mean_mocu": (
            float(realized_mocu.mean()) if realized_mocu.size else float("nan")
        ),
        "mean_shortfall": float(np.maximum(u_req - u, 0.0).mean()),
        "n_unique_sequences": int(div["n_unique_sequences"]),
        "sequence_entropy": float(div["sequence_entropy"]),
        "unique_frac": float(div["unique_frac"]),
        "eval_mode": "deterministic" if deterministic else "stochastic",
        "rows": rows,
    }


def _posterior_control_gpu(
    log_w: torch.Tensor,
    u_support: torch.Tensor,
    u_grid: torch.Tensor,
    *,
    alpha: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched posterior weights and conservative control decision on-device."""
    weights = torch.softmax(log_w, dim=-1)
    order = torch.argsort(u_support)
    u_sorted = u_support[order]
    cdf = torch.cumsum(weights[:, order], dim=-1)
    quantile_index = torch.sum(cdf < (1.0 - float(alpha)), dim=-1).clamp(
        max=len(u_support) - 1
    )
    target = u_sorted[quantile_index] + float(margin)
    grid_index = torch.sum(
        u_grid[None, :] + 1e-12 < target[:, None], dim=-1
    ).clamp(max=len(u_grid) - 1)
    return u_grid[grid_index], weights


def _posterior_mocu_gpu(
    log_w: torch.Tensor,
    u_support: torch.Tensor,
    u_grid: torch.Tensor,
    *,
    alpha: float,
    margin: float,
    undercontrol_penalty: float,
    violation_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u_ctrl, weights = _posterior_control_gpu(
        log_w, u_support, u_grid, alpha=alpha, margin=margin
    )
    shortfall = (u_support[None, :] - u_ctrl[:, None]).clamp_min(0.0)
    realized_cost = (
        u_ctrl[:, None]
        + float(undercontrol_penalty) * shortfall
        + float(violation_penalty) * (shortfall > 0.0).float()
    )
    regret = realized_cost - u_support[None, :]
    return (weights * regret).sum(dim=-1), u_ctrl, weights


def _belief_summary_gpu(
    ctx: ExperimentContext,
    weights: torch.Tensor,
    u_ctrl: torch.Tensor,
    raw_history: torch.Tensor,
    history_mask: torch.Tensor,
    *,
    step: int,
    m_support: torch.Tensor,
    k_support: torch.Tensor,
    u_support: torch.Tensor,
    u_grid: torch.Tensor,
) -> torch.Tensor:
    """Vectorized equivalent of ``belief_summary`` for a rollout batch."""
    batch = weights.shape[0]
    feats = torch.zeros(batch, 33, dtype=torch.float32, device=weights.device)
    feats[:, 0] = float(step) / float(ctx.horizon)
    feats[:, 1] = 1.0 / (
        weights.square().sum(dim=-1).clamp_min(1e-12) * weights.shape[-1]
    )
    feats[:, 2] = weights.max(dim=-1).values
    mean_m = (weights * m_support[None, :]).sum(dim=-1)
    mean_k = (weights * k_support[None, :]).sum(dim=-1)
    feats[:, 3] = mean_m
    feats[:, 4] = torch.sqrt(
        (weights * (m_support[None, :] - mean_m[:, None]).square())
        .sum(dim=-1)
        .clamp_min(0.0)
    )
    feats[:, 5] = mean_k
    feats[:, 6] = torch.sqrt(
        (weights * (k_support[None, :] - mean_k[:, None]).square())
        .sum(dim=-1)
        .clamp_min(0.0)
    )
    order = torch.argsort(u_support)
    u_sorted = u_support[order]
    cdf = torch.cumsum(weights[:, order], dim=-1)
    for offset, quantile in enumerate((0.05, 0.25, 0.50, 0.75, 0.95)):
        index = torch.sum(cdf < quantile, dim=-1).clamp(max=len(u_support) - 1)
        feats[:, 7 + offset] = u_sorted[index]
    feats[:, 12] = u_ctrl
    for offset, level in enumerate(u_grid[:16]):
        support_mask = torch.isclose(u_support, level)
        feats[:, 13 + offset] = weights[:, support_mask].sum(dim=-1)
    if step:
        valid = history_mask[:, :, None].expand_as(raw_history) > 0
        count = valid.sum(dim=(1, 2)).clamp_min(1)
        values = torch.where(valid, raw_history, torch.zeros_like(raw_history))
        mean = values.sum(dim=(1, 2)) / count
        centered = torch.where(valid, raw_history - mean[:, None, None], 0.0)
        feats[:, 29] = mean
        feats[:, 30] = torch.sqrt(
            centered.square().sum(dim=(1, 2)) / count
        )
        feats[:, 31] = torch.where(
            valid, raw_history, torch.full_like(raw_history, float("inf"))
        ).amin(dim=(1, 2))
        feats[:, 32] = torch.where(
            valid, raw_history, torch.full_like(raw_history, -float("inf"))
        ).amax(dim=(1, 2))
    return feats


@torch.no_grad()
def _collect_batched_rollouts(
    ctx: ExperimentContext,
    policy: AdaptiveExperimentPolicy,
    critic: StateValueCritic,
    config: TrainConfig,
    *,
    reward_mode: RewardMode,
    update: int,
    seed: int,
    device: torch.device,
    collect_counterfactual: bool = True,
) -> dict[str, Any]:
    """Collect one complete training update without leaving the torch device."""
    batch = int(config.trajectories_per_update)
    horizon = int(ctx.horizon)
    generator = torch.Generator(device=device).manual_seed(
        int(seed) + 1_000_003 * int(update)
    )
    clean_bank = torch.as_tensor(
        np.stack([s["obs_clean"] for s in ctx.train_systems]),
        dtype=torch.float32,
        device=device,
    )
    required_bank = torch.as_tensor(
        [s["u_req"] for s in ctx.train_systems],
        dtype=torch.float32,
        device=device,
    )
    system_ids = torch.randint(
        len(ctx.train_systems), (batch,), generator=generator, device=device
    )
    log_w = torch.as_tensor(
        ctx.log_p0, dtype=torch.float32, device=device
    )[None, :].expand(batch, -1).clone()
    centres = torch.as_tensor(
        ctx.centres_support, dtype=torch.float32, device=device
    )
    u_support = torch.as_tensor(ctx.U_support, dtype=torch.float32, device=device)
    m_support = torch.as_tensor(ctx.M_support, dtype=torch.float32, device=device)
    k_support = torch.as_tensor(ctx.K_support, dtype=torch.float32, device=device)
    u_grid = torch.as_tensor(ctx.u_grid, dtype=torch.float32, device=device)
    particles = torch.as_tensor(
        ctx.particle_features, dtype=torch.float32, device=device
    )[None, :, :].expand(batch, -1, -1)
    action_history = torch.zeros(
        batch, horizon, dtype=torch.long, device=device
    )
    raw_history = torch.zeros(
        batch, horizon, ctx.obs_dim, dtype=torch.float32, device=device
    )
    normalized_history = torch.zeros_like(raw_history)
    history_mask = torch.zeros(batch, horizon, device=device)
    feasible = torch.ones(
        batch, ctx.n_actions, dtype=torch.bool, device=device
    )
    states: list[tuple[torch.Tensor, ...]] = []
    actions_by_step: list[torch.Tensor] = []
    old_lp_by_step: list[torch.Tensor] = []
    values_by_step: list[torch.Tensor] = []
    mocu_by_step: list[torch.Tensor] = []
    counterfactual_utility_by_step: list[torch.Tensor] = []
    mocu0, u_ctrl, weights = _posterior_mocu_gpu(
        log_w,
        u_support,
        u_grid,
        alpha=ctx.alpha,
        margin=ctx.margin,
        undercontrol_penalty=config.undercontrol_penalty,
        violation_penalty=config.violation_penalty,
    )
    mocu_by_step.append(mocu0)
    batch_index = torch.arange(batch, device=device)
    for step in range(horizon):
        belief = _belief_summary_gpu(
            ctx,
            weights,
            u_ctrl,
            raw_history,
            history_mask,
            step=step,
            m_support=m_support,
            k_support=k_support,
            u_support=u_support,
            u_grid=u_grid,
        )
        state = (
            action_history.clone(),
            normalized_history.clone(),
            history_mask.clone(),
            belief,
            torch.full((batch,), step, dtype=torch.long, device=device),
            particles,
            weights,
            feasible.clone(),
        )
        states.append(state)
        if collect_counterfactual:
            # All-action counterfactual target on the sampled training system.
            # Across systems/noise draws this is an unbiased supervised signal
            # for the belief-conditioned expected reduction.  It defines the
            # decision fingerprints used to discover expert regimes.
            # Collection is skipped when the annealed counterfactual weight is
            # zero, since simulating every action is the dominant rollout cost.
            bootstrap = bool(config.moe_counterfactual_bootstrap) and (
                step + 1 < horizon
            )
            cf_utility = torch.empty(
                batch, ctx.n_actions, dtype=clean_bank.dtype, device=device
            )
            # Chunk actions to keep B x A x particles x N_obs memory bounded.
            for action_start in range(0, ctx.n_actions, 16):
                action_stop = min(action_start + 16, ctx.n_actions)
                chunk = action_stop - action_start
                cf_clean = clean_bank[system_ids, action_start:action_stop]
                cf_observation = cf_clean + torch.randn(
                    cf_clean.shape,
                    generator=generator,
                    dtype=cf_clean.dtype,
                    device=device,
                ) * float(ctx.sigma_y)
                cf_residual = (
                    cf_observation[:, :, None, :]
                    - centres[None, action_start:action_stop, :, :]
                )
                cf_log_w = log_w[:, None, :] - 0.5 * cf_residual.square().sum(-1) / (
                    float(ctx.sigma_y) ** 2
                )
                cf_mocu, cf_u_ctrl, cf_weights = _posterior_mocu_gpu(
                    cf_log_w.reshape(-1, cf_log_w.shape[-1]),
                    u_support,
                    u_grid,
                    alpha=ctx.alpha,
                    margin=ctx.margin,
                    undercontrol_penalty=config.undercontrol_penalty,
                    violation_penalty=config.violation_penalty,
                )
                # Policy utilities are maximized, hence negative MOCU.
                utilities = -cf_mocu.reshape(batch, chunk)
                if bootstrap:
                    # Critic bootstrap on the fantasy next state turns the
                    # one-step target (the Myopic criterion) into an estimate
                    # of negative expected terminal MOCU:
                    #   q(a) = -MOCU_{t+1}(a) + gamma * V(h_{t+1}(a)).
                    flat = batch * chunk
                    acts = torch.arange(
                        action_start, action_stop, dtype=torch.long, device=device
                    )
                    fa_actions = (
                        action_history[:, None, :].expand(batch, chunk, horizon).clone()
                    )
                    fa_actions[:, :, step] = acts[None, :]
                    fa_norm = (
                        normalized_history[:, None, :, :]
                        .expand(batch, chunk, horizon, ctx.obs_dim)
                        .clone()
                    )
                    fa_norm[:, :, step, :] = (
                        cf_observation - float(ctx.obs_mean)
                    ) / max(float(ctx.obs_std), 1e-8)
                    fa_raw = (
                        raw_history[:, None, :, :]
                        .expand(batch, chunk, horizon, ctx.obs_dim)
                        .clone()
                    )
                    fa_raw[:, :, step, :] = cf_observation
                    fa_mask = (
                        history_mask[:, None, :].expand(batch, chunk, horizon).clone()
                    )
                    fa_mask[:, :, step] = 1.0
                    fa_belief = _belief_summary_gpu(
                        ctx,
                        cf_weights,
                        cf_u_ctrl,
                        fa_raw.reshape(flat, horizon, ctx.obs_dim),
                        fa_mask.reshape(flat, horizon),
                        step=step + 1,
                        m_support=m_support,
                        k_support=k_support,
                        u_support=u_support,
                        u_grid=u_grid,
                    )
                    fa_steps = torch.full(
                        (flat,), step + 1, dtype=torch.long, device=device
                    )
                    fa_particles = (
                        particles[:, None, :, :]
                        .expand(batch, chunk, *particles.shape[1:])
                        .reshape(flat, *particles.shape[1:])
                    )
                    next_value = critic(
                        fa_actions.reshape(flat, horizon),
                        fa_norm.reshape(flat, horizon, ctx.obs_dim),
                        fa_mask.reshape(flat, horizon),
                        fa_belief,
                        fa_steps,
                        fa_particles,
                        cf_weights,
                    )
                    utilities = utilities + float(config.gamma) * next_value.reshape(
                        batch, chunk
                    )
                cf_utility[:, action_start:action_stop] = utilities
            counterfactual_utility_by_step.append(cf_utility)
        dist = policy.distribution(*state[:-1], state[-1])
        action = torch.multinomial(
            dist.probs, 1, generator=generator
        ).squeeze(-1)
        actions_by_step.append(action)
        old_lp_by_step.append(dist.log_prob(action))
        values_by_step.append(critic(*state[:-1]))
        clean = clean_bank[system_ids, action]
        observation = clean + torch.randn(
            clean.shape,
            generator=generator,
            dtype=clean.dtype,
            device=device,
        ) * float(ctx.sigma_y)
        action_history[:, step] = action
        raw_history[:, step] = observation
        normalized_history[:, step] = (
            observation - float(ctx.obs_mean)
        ) / max(float(ctx.obs_std), 1e-8)
        history_mask[:, step] = 1.0
        feasible[batch_index, action] = False
        action_centres = centres[action]  # B,P,D
        residual = observation[:, None, :] - action_centres
        log_w = log_w - 0.5 * residual.square().sum(dim=-1) / (
            float(ctx.sigma_y) ** 2
        )
        mocu, u_ctrl, weights = _posterior_mocu_gpu(
            log_w,
            u_support,
            u_grid,
            alpha=ctx.alpha,
            margin=ctx.margin,
            undercontrol_penalty=config.undercontrol_penalty,
            violation_penalty=config.violation_penalty,
        )
        mocu_by_step.append(mocu)
    mocu_path = torch.stack(mocu_by_step, dim=1)
    if reward_mode == "dad_terminal":
        rewards = torch.zeros(batch, horizon, device=device)
        rewards[:, -1] = -mocu_path[:, -1]
    else:
        rewards = mocu_path[:, :-1] - mocu_path[:, 1:]
    values = torch.stack(values_by_step, dim=1)
    advantages = torch.zeros_like(rewards)
    last = torch.zeros(batch, device=device)
    for step in reversed(range(horizon)):
        next_value = values[:, step + 1] if step + 1 < horizon else 0.0
        delta = rewards[:, step] + float(config.gamma) * next_value - values[:, step]
        last = delta + float(config.gamma * config.gae_lambda) * last
        advantages[:, step] = last
    returns = advantages + values
    required = required_bank[system_ids]
    shortfall = (required - u_ctrl).clamp_min(0.0)
    terminal_cost = (
        u_ctrl
        + float(config.undercontrol_penalty) * shortfall
        + float(config.violation_penalty) * (shortfall > 0.0).float()
    )
    return {
        "states": states,
        "actions": torch.stack(actions_by_step, dim=1),
        "old_log_probs": torch.stack(old_lp_by_step, dim=1),
        "advantages": advantages,
        "returns": returns,
        "terminal_u_ctrl": u_ctrl,
        "terminal_cost": terminal_cost,
        "counterfactual_utility": (
            torch.cat(counterfactual_utility_by_step, dim=0)
            if counterfactual_utility_by_step
            else None
        ),
    }


def train_policy(
    ctx: ExperimentContext,
    *,
    method: Literal["DAD", "RL-sBOED", "MoE-sBOED", "MatchedDense"],
    seed: int = 101,
    smoke: bool = False,
) -> dict[str, Any]:
    training_block = ctx.cfg.training_for(
        getattr(ctx, "experiment_type", "objective_based")
    )
    config = TrainConfig.from_cfg(training_block)
    if (
        abs(float(config.undercontrol_penalty) - float(ctx.undercontrol_penalty))
        > 1e-12
        or abs(float(config.violation_penalty) - float(ctx.violation_penalty))
        > 1e-12
    ):
        raise RuntimeError(
            "MOCU cost mismatch between context and trainer; all methods must "
            "share undercontrol_penalty and violation_penalty"
        )
    if bool(training_block.get("reference_method_fidelity", False)):
        if method in ("DAD", "RL-sBOED") and (
            float(config.dense_counterfactual_coefficient) != 0.0
            or float(config.dense_branching_coefficient) != 0.0
        ):
            raise ValueError(
                "reference_method_fidelity forbids dense counterfactual/branching "
                "auxiliaries for DAD and RL-sBOED"
            )
        if method == "DAD" and str(config.dad_optimizer).lower() != "reinforce":
            raise ValueError(
                "reference_method_fidelity requires DAD terminal-reward REINFORCE"
            )
    # MoE default: soft unique-floor + joint MOCU/diversity score when YAML omits keys.
    if method == "MoE-sBOED":
        if "prefer_unique_sequence_floor" not in training_block:
            config = replace(config, prefer_unique_sequence_floor=True)
            print(
                "[train] MoE-sBOED: enabling prefer_unique_sequence_floor "
                "(soft unique-floor + MOCU slack)"
            )
    n_val = max(int(config.validation_rollouts), 1)
    min_u = (
        max(2, int(np.ceil(float(config.min_unique_sequence_fraction) * n_val)))
        if n_val >= 2
        else 1
    )
    print(
        f"[train] {method} checkpoint selection: "
        f"joint_score=MOCU-{config.checkpoint_diversity_weight:g}*unique_frac; "
        f"soft_floor={'ON' if config.prefer_unique_sequence_floor else 'OFF'} "
        f"(uniq>={min_u}/{config.validation_rollouts}, "
        f"mocu_slack={config.unique_floor_mocu_slack:g})"
    )
    if not smoke and config.min_updates_per_horizon > 0:
        horizon_floor = int(config.min_updates_per_horizon) * int(ctx.horizon)
        if config.updates < horizon_floor:
            print(
                f"[train] increasing updates {config.updates} → {horizon_floor} "
                f"for T={ctx.horizon} "
                f"(min_updates_per_horizon={config.min_updates_per_horizon})"
            )
            config.updates = horizon_floor
    if smoke:
        # Keep MoE ablation knobs from the YAML; only shrink the PPO budget.
        config = replace(
            config,
            updates=3,
            trajectories_per_update=4,
            ppo_epochs=2,
            validation_interval=2,
            validation_rollouts=4,
            patience=2,
            open_loop_warn_after_updates=max(
                1, min(2, int(config.open_loop_warn_after_updates or 2))
            ),
        )
    # The stepwise MOCU reductions telescope to the same terminal objective but
    # give PPO substantially denser credit.  DAD retains its published terminal
    # return; RL-sBOED and MoE-sBOED use the aligned stepwise decomposition.
    reward_mode: RewardMode = (
        "rl_sboed_stepwise"
        if method in ("RL-sBOED", "MoE-sBOED", "MatchedDense")
        else "dad_terminal"
    )
    stem = _method_stem(method)
    ensure_result_layout(ctx.out_dir)
    out = model_dir(ctx.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy_path = out / f"{stem}.pth"
    metrics_csv = out / f"{stem}_training_metrics.csv"
    loss_png = out / f"{stem}_loss_curve.png"
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    total_traj_budget = int(config.updates) * int(config.trajectories_per_update)
    print(
        f"[train] {method} device={device} updates={config.updates} "
        f"traj/update={config.trajectories_per_update} "
        f"total_trajectories={total_traj_budget} "
        f"val_every={config.validation_interval} "
        f"val_rollouts={config.validation_rollouts} "
        f"entropy_coef={config.entropy_coefficient}→"
        f"{config.entropy_final_coefficient} "
        f"entropy_anneal_fraction={config.entropy_anneal_fraction} "
        f"diversity_w={config.checkpoint_diversity_weight} "
        f"dense_cf={config.dense_counterfactual_coefficient} "
        f"dense_branch={config.dense_branching_coefficient}"
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    pcfg = _policy_config(ctx)
    if method in ("MoE-sBOED", "MatchedDense"):
        if method == "MatchedDense":
            n_experts = 1
            top_k = 1
            expert_hidden = (
                int(config.moe_expert_hidden)
                if int(config.moe_expert_hidden) > 0
                else parameter_matched_expert_hidden(
                    hidden=pcfg.hidden,
                    n_actions=ctx.n_actions,
                    reference_experts=max(2, int(config.moe_n_experts)),
                )
            )
        else:
            n_experts = int(config.moe_n_experts)
            top_k = int(config.moe_top_k)
            expert_hidden = (
                int(config.moe_expert_hidden)
                if int(config.moe_expert_hidden) > 0
                else None
            )
        policy = BeliefConditionedMoEPolicy(
            ctx.n_actions,
            pcfg,
            n_experts=n_experts,
            top_k=top_k,
            expert_hidden=expert_hidden,
            logit_scale_init=float(config.moe_logit_scale_init),
            balance_coefficient=float(config.moe_balance_coefficient),
            redundancy_coefficient=float(config.moe_redundancy_coefficient),
        ).to(device)
        n_params = sum(p.numel() for p in policy.parameters())
        print(
            f"[train] {method} architecture n_experts={policy.n_experts} "
            f"top_k={policy.top_k} expert_hidden={policy.expert_hidden} "
            f"total_params={n_params} "
            f"cf_coef={config.moe_counterfactual_coefficient:g} "
            f"cf_floor={config.moe_counterfactual_floor_fraction:g} "
            f"balance={config.moe_balance_coefficient:g} "
            f"logit_scale_init={config.moe_logit_scale_init:g}"
            + (
                " (parameter-matched dense control)"
                if method == "MatchedDense" or policy.n_experts == 1
                else ""
            )
        )
    else:
        policy = AdaptiveExperimentPolicy(ctx.n_actions, pcfg).to(device)
    critic = StateValueCritic(ctx.n_actions, pcfg).to(device)
    actor_opt = torch.optim.Adam(policy.parameters(), lr=config.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    best_state = None
    best_score = float("inf")
    best_val = float("inf")
    best_unique = 0
    best_meets_floor = False
    open_loop_warned = False
    history: list[dict[str, Any]] = []
    t0 = time.time()
    # patience <= 0 disables early stopping (run full update budget).
    early_stop = int(config.patience) > 0
    patience_left = int(config.patience) if early_stop else 10**9
    trajectories_sampled = 0

    prototype_reset_update = (
        max(2, int(round(config.moe_prototype_reset_fraction * config.updates)))
        if config.moe_prototype_reset_fraction > 0.0
        else 0
    )
    for update in range(1, config.updates + 1):
        update_t0 = time.time()
        policy.train()
        critic.train()
        current_entropy_weight = entropy_weight(config, update)
        cf_weight = (
            moe_counterfactual_weight(config, update)
            if isinstance(policy, BeliefConditionedMoEPolicy)
            else 0.0
        )
        branch_weight = (
            float(config.moe_branching_coefficient)
            if isinstance(policy, BeliefConditionedMoEPolicy)
            else 0.0
        )
        dense_cf_weight = (
            float(config.dense_counterfactual_coefficient)
            if not isinstance(policy, BeliefConditionedMoEPolicy)
            else 0.0
        )
        dense_branch_weight = (
            float(config.dense_branching_coefficient)
            if not isinstance(policy, BeliefConditionedMoEPolicy)
            else 0.0
        )
        # Branching needs CF fingerprints even when CF ranking loss is off.
        need_counterfactual = (
            (cf_weight > 0.0)
            or (branch_weight > 0.0)
            or (dense_cf_weight > 0.0)
            or (dense_branch_weight > 0.0)
        )
        fixed_bc_weight = (
            moe_fixed_bc_weight(config, update)
            if isinstance(policy, BeliefConditionedMoEPolicy)
            else 0.0
        )
        if (
            isinstance(policy, BeliefConditionedMoEPolicy)
            and update == prototype_reset_update
        ):
            # Re-anchor regime prototypes on fingerprints from the warmed-up
            # policy/critic instead of the noisy first-batch initialization.
            policy.reset_regime_prototypes()
            print(
                f"[train] {method} regime prototypes reset at update={update} "
                f"for re-initialization from warmed-up fingerprints"
            )
        rollout = _collect_batched_rollouts(
            ctx,
            policy,
            critic,
            config,
            reward_mode=reward_mode,
            update=update,
            seed=seed,
            device=device,
            collect_counterfactual=need_counterfactual,
        )
        trajectories_sampled += int(config.trajectories_per_update)

        def cat_field(idx: int) -> torch.Tensor:
            return torch.cat([s[idx] for s in rollout["states"]], dim=0)

        action_t = rollout["actions"].transpose(0, 1).reshape(-1)
        old_lp_t = rollout["old_log_probs"].transpose(0, 1).reshape(-1)
        adv_t = rollout["advantages"].transpose(0, 1).reshape(-1)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        # Normalize once across trajectories.  Per-trajectory normalization
        # erases the between-trajectory terminal-cost signal, especially for
        # DAD where every action shares the same terminal return.
        ret_t = rollout["returns"].transpose(0, 1).reshape(-1)
        inputs = tuple(cat_field(i) for i in range(8))
        terminals = rollout["terminal_u_ctrl"].detach().cpu().numpy()
        terminal_costs = rollout["terminal_cost"].detach().cpu().numpy()

        last_entropy = 0.0
        moe_stats: dict[str, float] = {}
        # DAD defaults to a one-pass score-function (REINFORCE) update, matching
        # the discrete-design extension of DAD. RL-sBOED retains clipped PPO.
        # dad_optimizer="ppo" runs DAD's terminal reward under PPO so the H4
        # diagnostic can separate credit assignment from the optimizer.
        dad_reinforce = method == "DAD" and config.dad_optimizer == "reinforce"
        optimization_epochs = 1 if dad_reinforce else config.ppo_epochs
        for _ in range(optimization_epochs):
            dist = policy.distribution(*inputs[:-1], inputs[-1])
            new_lp = dist.log_prob(action_t)
            entropy = dist.entropy().mean()
            last_entropy = float(entropy.detach().item())
            if dad_reinforce:
                policy_loss = -(
                    (new_lp * adv_t.detach()).mean()
                    + current_entropy_weight * entropy
                )
            else:
                ratio = torch.exp(new_lp - old_lp_t)
                surr1 = ratio * adv_t
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip
                    )
                    * adv_t
                )
                policy_loss = -(
                    torch.min(surr1, surr2).mean()
                    + current_entropy_weight * entropy
                )
            if dense_cf_weight > 0.0 and rollout["counterfactual_utility"] is not None:
                dense_cf_loss, dense_cf_stats = dense_counterfactual_ranking_loss(
                    policy(*inputs[:-1], inputs[-1]),
                    rollout["counterfactual_utility"],
                    inputs[-1],
                    temperature=float(config.dense_counterfactual_temperature),
                )
                policy_loss = policy_loss + dense_cf_weight * dense_cf_loss
                moe_stats.update(dense_cf_stats)
                moe_stats["dense_cf_weight"] = float(dense_cf_weight)
            if dense_branch_weight > 0.0 and rollout["counterfactual_utility"] is not None:
                dense_branch_loss, dense_branch_stats = dense_post_prior_branching_loss(
                    policy(*inputs[:-1], inputs[-1]),
                    rollout["counterfactual_utility"],
                    inputs[-1],
                    inputs[4],
                    similarity_threshold=float(
                        config.dense_branching_similarity_threshold
                    ),
                    margin=float(config.dense_branching_margin),
                )
                policy_loss = policy_loss + dense_branch_weight * dense_branch_loss
                moe_stats.update(dense_branch_stats)
                moe_stats["dense_branching_weight"] = float(dense_branch_weight)
            if isinstance(policy, BeliefConditionedMoEPolicy):
                auxiliary, moe_stats = policy.specialization_loss(*inputs[:-1])
                policy_loss = policy_loss + auxiliary
                # Belief-gated Fixed anchor on the shared base only: high-ESS
                # (prior-like) states recover Fixed; low-ESS states leave the
                # residual experts free to adapt.
                if fixed_bc_weight > 0.0:
                    base_for_bc = policy.base_logits(*inputs[:-1])
                    fixed_loss, fixed_stats = fixed_sequence_bc_loss(
                        base_for_bc,
                        inputs[4],
                        inputs[-1],
                        list(ctx.fixed_sequence),
                        belief_summary=inputs[3],
                        ess_threshold=float(config.moe_fixed_bc_ess_threshold),
                    )
                    policy_loss = policy_loss + fixed_bc_weight * fixed_loss
                    moe_stats.update(fixed_stats)
                moe_stats["moe_fixed_bc_weight"] = float(fixed_bc_weight)
                moe_stats["moe_fixed_bc_ess_threshold"] = float(
                    config.moe_fixed_bc_ess_threshold
                )
                # Residual release: as Fixed-BC anneals down, push experts to
                # matter on low-ESS beliefs and keep logit_scale from collapsing.
                release = moe_residual_release_fraction(config, update)
                moe_stats["moe_residual_release_fraction"] = float(release)
                low_ess_coef = (
                    float(config.moe_low_ess_residual_coefficient) * release
                )
                if low_ess_coef > 0.0:
                    low_ess_loss, low_ess_stats = low_ess_residual_influence_loss(
                        policy,
                        *inputs[:-1],
                        feasible_mask=inputs[-1],
                        ess_threshold=float(config.moe_low_ess_threshold),
                    )
                    policy_loss = policy_loss + low_ess_coef * low_ess_loss
                    moe_stats.update(low_ess_stats)
                moe_stats["moe_low_ess_residual_weight"] = float(low_ess_coef)
                scale_coef = float(config.moe_residual_scale_coefficient) * release
                if scale_coef > 0.0:
                    scale_loss, scale_stats = residual_scale_floor_loss(
                        policy,
                        target=float(config.moe_logit_scale_target),
                    )
                    policy_loss = policy_loss + scale_coef * scale_loss
                    moe_stats.update(scale_stats)
                moe_stats["moe_residual_scale_weight"] = float(scale_coef)
                # Optional counterfactual regime supervision (off by default).
                if cf_weight > 0.0 and rollout["counterfactual_utility"] is not None:
                    cf_loss, cf_stats = policy.counterfactual_loss(
                        *inputs[:-1],
                        target_utility=rollout["counterfactual_utility"],
                        feasible_mask=inputs[-1],
                    )
                    policy_loss = policy_loss + cf_weight * cf_loss
                    moe_stats.update(cf_stats)
                # Branching can run with CF fingerprints alone (no CF ranking loss).
                if (
                    branch_weight > 0.0
                    and rollout["counterfactual_utility"] is not None
                ):
                    branch_loss, branch_stats = policy.branching_loss(
                        *inputs[:-1],
                        target_utility=rollout["counterfactual_utility"],
                        feasible_mask=inputs[-1],
                    )
                    policy_loss = policy_loss + branch_weight * branch_loss
                    moe_stats.update(branch_stats)
                moe_stats["moe_counterfactual_weight"] = float(cf_weight)
                moe_stats["moe_branching_weight"] = float(branch_weight)
            values_pred = critic(*inputs[:-1])
            value_loss = F.huber_loss(values_pred, ret_t)
            actor_opt.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            actor_opt.step()
            critic_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            critic_opt.step()

        update_elapsed = time.time() - update_t0
        row: dict[str, Any] = {
            "update": update,
            "mean_train_u_ctrl": float(np.mean(terminals)),
            "mean_train_control_cost": float(np.mean(terminal_costs)),
            "method": method,
            "update_seconds": float(update_elapsed),
            "trajectories_sampled": int(trajectories_sampled),
            "policy_entropy": last_entropy,
            "entropy_coefficient": float(current_entropy_weight),
            **moe_stats,
        }
        log_every = max(1, min(25, int(config.validation_interval)))
        if update % log_every == 0 or update == 1 or update == config.updates:
            elapsed = time.time() - t0
            rate = trajectories_sampled / max(elapsed, 1e-9)
            eta = (
                (total_traj_budget - trajectories_sampled) / rate
                if rate > 0
                else float("nan")
            )
            print(
                f"[train] {method} update={update}/{config.updates} "
                f"mean_u={row['mean_train_u_ctrl']:.4f} "
                f"entropy={last_entropy:.3f} "
                f"entropy_coef={current_entropy_weight:.5f} "
                f"update_s={update_elapsed:.2f} "
                f"traj={trajectories_sampled}/{total_traj_budget} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )
        if update % config.validation_interval == 0 or update == config.updates:
            val = evaluate_policy(
                ctx,
                policy,
                ctx.validation_systems,
                n_rollouts=config.validation_rollouts,
                global_seed=seed + 17,
                reward_mode=reward_mode,
                device=device,
                deterministic=True,
            )
            score, meets_floor = checkpoint_score(
                float(val["mean_mocu"]),
                int(val["n_unique_sequences"]),
                int(config.validation_rollouts),
                diversity_weight=config.checkpoint_diversity_weight,
                min_unique_fraction=config.min_unique_sequence_fraction,
                under_control_rate=float(val["under_control_rate"]),
                safety_penalty=config.checkpoint_safety_penalty,
                min_safety_rate=config.min_valid_safety_rate,
            )
            row["validation_mean_u_ctrl"] = val["mean_u_ctrl"]
            row["validation_mean_mocu"] = val["mean_mocu"]
            row["validation_n_unique_sequences"] = val["n_unique_sequences"]
            row["validation_sequence_entropy"] = val["sequence_entropy"]
            row["validation_unique_frac"] = val["unique_frac"]
            row["validation_under_control_rate"] = val["under_control_rate"]
            row["validation_valid"] = int(
                float(val["under_control_rate"])
                <= 1.0 - config.min_valid_safety_rate + 1e-12
            )
            row["validation_mean_shortfall"] = val["mean_shortfall"]
            row["validation_checkpoint_score"] = score
            row["validation_meets_unique_floor"] = int(meets_floor)
            if method == "MoE-sBOED":
                row["moe_adaptivity_constraint"] = (
                    "soft_unique_floor_joint_score"
                    if config.prefer_unique_sequence_floor
                    else "joint_score_only"
                )
            print(
                f"[train] {method} val update={update} "
                f"mean_u={val['mean_u_ctrl']:.4f} "
                f"mocu={val['mean_mocu']:.4f} "
                f"under={val['under_control_rate']:.3f} "
                f"unique_seq={val['n_unique_sequences']}/{config.validation_rollouts} "
                f"seq_H={val['sequence_entropy']:.3f} "
                f"joint_score={score:.4f} floor={'ok' if meets_floor else 'MISS'}"
            )
            warn_after = int(config.open_loop_warn_after_updates)
            if (
                warn_after > 0
                and update >= warn_after
                and int(val["n_unique_sequences"]) <= 1
                and not open_loop_warned
            ):
                open_loop_warned = True
                print(
                    f"[train] WARNING {method}: deterministic unique_seq="
                    f"{val['n_unique_sequences']} after update={update} "
                    f"(open-loop collapse). Consider higher entropy_coefficient "
                    f"or checking belief conditioning."
                )
            if should_replace_checkpoint(
                score=float(score),
                meets_floor=bool(meets_floor),
                n_unique=int(val["n_unique_sequences"]),
                best_score=float(best_score),
                best_meets_floor=bool(best_meets_floor),
                best_unique=int(best_unique),
                prefer_unique_floor=bool(config.prefer_unique_sequence_floor),
                mean_mocu=float(val["mean_mocu"]),
                best_mean_mocu=float(best_val),
                floor_mocu_slack=float(config.unique_floor_mocu_slack),
            ):
                best_score = float(score)
                best_val = float(val["mean_mocu"])
                best_unique = int(val["n_unique_sequences"])
                best_meets_floor = bool(meets_floor)
                best_state = copy.deepcopy(policy.state_dict())
                if early_stop:
                    patience_left = int(config.patience)
                torch.save(
                    {
                        "policy": best_state,
                        "method": method,
                        "seed": seed,
                        "obs_dim": ctx.obs_dim,
                        "n_obs": ctx.n_obs,
                        "obs_indices": ctx.obs_indices.tolist(),
                        "checkpoint_score": best_score,
                        "validation_mean_mocu": best_val,
                        "validation_n_unique_sequences": best_unique,
                        "validation_meets_unique_floor": best_meets_floor,
                        "parent_initialization": None,
                        "moe_counterfactual_coefficient": (
                            float(config.moe_counterfactual_coefficient)
                            if isinstance(policy, BeliefConditionedMoEPolicy) else 0.0
                        ),
                        "moe_counterfactual_anneal_fraction": (
                            float(config.moe_counterfactual_anneal_fraction)
                            if isinstance(policy, BeliefConditionedMoEPolicy) else 0.0
                        ),
                        "architecture": _architecture_tag(policy),
                        "moe_n_experts": (
                            int(policy.n_experts)
                            if isinstance(policy, BeliefConditionedMoEPolicy)
                            else 0
                        ),
                        "moe_top_k": (
                            int(policy.top_k)
                            if isinstance(policy, BeliefConditionedMoEPolicy)
                            else 0
                        ),
                        "moe_expert_hidden": (
                            int(policy.expert_hidden)
                            if isinstance(policy, BeliefConditionedMoEPolicy)
                            else 0
                        ),
                    },
                    policy_path,
                )
            elif early_stop:
                patience_left -= 1
        history.append(row)
        if early_stop and patience_left <= 0:
            print(
                f"[train] early stop at update={update} "
                f"(patience={config.patience}, best_val={best_val}, "
                f"best_unique={best_unique})"
            )
            break

    if best_state is not None:
        policy.load_state_dict(best_state)
    torch.save(
        {
            "policy": policy.state_dict(),
            "method": method,
            "seed": seed,
            "obs_dim": ctx.obs_dim,
            "n_obs": ctx.n_obs,
            "obs_indices": ctx.obs_indices.tolist(),
            "checkpoint_score": best_score,
            "validation_mean_mocu": best_val,
            "validation_n_unique_sequences": best_unique,
            "validation_meets_unique_floor": best_meets_floor,
            "parent_initialization": None,
            "entropy_coefficient": float(config.entropy_coefficient),
            "entropy_final_coefficient": float(
                config.entropy_final_coefficient
            ),
            "entropy_anneal_fraction": float(config.entropy_anneal_fraction),
            "moe_counterfactual_coefficient": (
                float(config.moe_counterfactual_coefficient)
                if isinstance(policy, BeliefConditionedMoEPolicy) else 0.0
            ),
            "moe_counterfactual_anneal_fraction": (
                float(config.moe_counterfactual_anneal_fraction)
                if isinstance(policy, BeliefConditionedMoEPolicy) else 0.0
            ),
            "architecture": _architecture_tag(policy),
            "moe_n_experts": (
                int(policy.n_experts)
                if isinstance(policy, BeliefConditionedMoEPolicy)
                else 0
            ),
            "moe_top_k": (
                int(policy.top_k)
                if isinstance(policy, BeliefConditionedMoEPolicy)
                else 0
            ),
            "moe_expert_hidden": (
                int(policy.expert_hidden)
                if isinstance(policy, BeliefConditionedMoEPolicy)
                else 0
            ),
        },
        policy_path,
    )
    _write_csv(metrics_csv, history)
    _write_loss_curve(loss_png, history, method=method)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_total = time.time() - t0
    if best_state is not None and not best_meets_floor:
        print(
            f"[train] WARNING {method}: best checkpoint still misses unique-seq "
            f"floor (unique={best_unique}/{config.validation_rollouts}). "
            f"Policy may be open-loop."
        )
    print(
        f"[train] {method} finished updates={len(history)} "
        f"trajectories={trajectories_sampled} "
        f"elapsed={elapsed_total:.1f}s "
        f"best_val_mocu={best_val:.4f} unique={best_unique} "
        f"score={best_score:.4f} → {policy_path}"
    )
    result = {
        "method": method,
        "seed": seed,
        "best_validation_mean_mocu": best_val,
        "best_validation_n_unique_sequences": best_unique,
        "best_checkpoint_score": best_score,
        "best_meets_unique_floor": best_meets_floor,
        "open_loop_warned": open_loop_warned,
        "elapsed_seconds": elapsed_total,
        "n_updates_ran": len(history),
        "trajectories_sampled": int(trajectories_sampled),
        "trajectories_budget": int(total_traj_budget),
        "entropy_coefficient": float(config.entropy_coefficient),
        "entropy_final_coefficient": float(config.entropy_final_coefficient),
        "entropy_anneal_fraction": float(config.entropy_anneal_fraction),
        "config": asdict(config),
        "n_obs": ctx.n_obs,
        "obs_indices": ctx.obs_indices.tolist(),
        "checkpoint": str(policy_path),
        "training_metrics_csv": str(metrics_csv),
        "loss_curve_png": str(loss_png),
        "device": str(device),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "parent_initialization": None,
        "moe_counterfactual_coefficient": (
            float(config.moe_counterfactual_coefficient)
            if isinstance(policy, BeliefConditionedMoEPolicy)
            else 0.0
        ),
    }
    # Persist a compact training summary into run_config.json (no extra JSON files).
    try:
        from src.layout import load_run_config_doc, run_config_path

        doc = load_run_config_doc(ctx.out_dir)
        train_block = dict(doc.get("training_results") or {})
        train_block[stem] = {
            "seed": int(seed),
            "elapsed_seconds": result["elapsed_seconds"],
            "n_updates_ran": result["n_updates_ran"],
            "trajectories_sampled": result["trajectories_sampled"],
            "trajectories_budget": result["trajectories_budget"],
            "best_validation_mean_mocu": result["best_validation_mean_mocu"],
            "best_validation_n_unique_sequences": result[
                "best_validation_n_unique_sequences"
            ],
            "best_checkpoint_score": result["best_checkpoint_score"],
            "best_meets_unique_floor": result["best_meets_unique_floor"],
            "open_loop_warned": result["open_loop_warned"],
            "device": str(device),
            "checkpoint": str(policy_path),
            "parent_initialization": None,
            "moe_counterfactual_coefficient": (
                float(config.moe_counterfactual_coefficient)
                if isinstance(policy, BeliefConditionedMoEPolicy)
                else 0.0
            ),
        }
        doc["training_results"] = train_block
        run_config_path(ctx.out_dir).write_text(
            json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8"
        )
    except Exception:
        pass
    return result


def load_trained_policy(
    ctx: ExperimentContext, method: str, device: torch.device | None = None
) -> AdaptiveExperimentPolicy | BeliefConditionedMoEPolicy:
    device = device or torch.device("cpu")
    stem = _method_stem(method)
    mdir = model_dir(ctx.out_dir)
    candidates = [
        mdir / f"{stem}.pth",
        # Legacy layout fallbacks
        ctx.out_dir / "train" / stem / "best_checkpoint.pt",
        ctx.out_dir / "train" / stem / "final_checkpoint.pt",
        mdir / "best_checkpoint.pt",
        mdir / "final_checkpoint.pt",
    ]
    ckpt_path = next((p for p in candidates if p.is_file()), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No trained policy for {method!r} under {mdir} "
            f"(expected {stem}.pth)"
        )
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    is_moe = (
        str(payload.get("architecture", "")).endswith("moe")
        or "moe" in str(payload.get("architecture", ""))
        or "matched_dense" in str(payload.get("architecture", ""))
        or stem in ("moe_sboed", "matched_dense")
    )
    if is_moe:
        # Rebuild with the checkpoint's actual architecture (a 1-expert dense
        # control or a widened expert head differs from the 4-expert default).
        sd = payload["policy"]
        expert_ids = [
            int(k.split(".")[1]) for k in sd if k.startswith("experts.")
        ]
        n_experts = (max(expert_ids) + 1) if expert_ids else 4
        expert_hidden = (
            int(sd["experts.0.0.weight"].shape[0])
            if "experts.0.0.weight" in sd
            else None
        )
        top_k = int(payload.get("moe_top_k", min(2, n_experts)))
        arch = str(payload.get("architecture", ""))
        # Poster-era residual MoE used sigmoid(expert_scale).  Current v3 uses
        # softplus(logit_scale) + shared base inside BeliefConditionedMoEPolicy.
        legacy_residual = (
            "expert_scale" in sd
            or arch.startswith("shared_base_top2_residual")
        ) and "logit_scale" not in sd
        if legacy_residual:
            policy = SharedBaseResidualMoEPolicy(
                ctx.n_actions,
                _policy_config(ctx),
                n_experts=n_experts,
                top_k=top_k,
                expert_hidden=expert_hidden,
            ).to(device)
            policy.load_state_dict(payload["policy"])
        else:
            policy = BeliefConditionedMoEPolicy(
                ctx.n_actions,
                _policy_config(ctx),
                n_experts=n_experts,
                top_k=top_k,
                expert_hidden=expert_hidden,
            ).to(device)
            # Pure-mixture checkpoints (no base_head) load with a zero base so
            # behaviour stays scale * routed experts.  Newer residual_gate keys
            # are optional for older v3 checkpoints.
            missing_gate = any(
                k.startswith("residual_gate.") for k in policy.state_dict()
            ) and not any(k.startswith("residual_gate.") for k in sd)
            if "base_head.0.weight" not in sd or missing_gate:
                if "base_head.0.weight" not in sd:
                    nn.init.zeros_(policy.base_head[-1].weight)
                    nn.init.zeros_(policy.base_head[-1].bias)
                policy.load_state_dict(payload["policy"], strict=False)
            else:
                policy.load_state_dict(payload["policy"])
    else:
        policy = AdaptiveExperimentPolicy(ctx.n_actions, _policy_config(ctx)).to(device)
        policy.load_state_dict(payload["policy"])
    policy.eval()
    return policy


def _write_loss_curve(path: Path, history: list[dict[str, Any]], *, method: str) -> None:
    if not history:
        return
    updates = [int(r["update"]) for r in history]
    train_y = [float(r["mean_train_u_ctrl"]) for r in history]
    val_pts = [
        (int(r["update"]), float(r["validation_mean_u_ctrl"]))
        for r in history
        if r.get("validation_mean_u_ctrl") is not None
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(updates, train_y, "b-o", markersize=3, label="train mean u_ctrl")
    if val_pts:
        ax.plot(
            [u for u, _ in val_pts],
            [v for _, v in val_pts],
            "g-s",
            markersize=3,
            label="val mean u_ctrl",
        )
    ax.set_xlabel("Update")
    ax.set_ylabel("mean u_ctrl")
    ax.set_title(f"{method} training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
