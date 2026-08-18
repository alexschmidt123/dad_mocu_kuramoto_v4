"""Vector-observation EIG training/evaluation on the physical delta-f bank."""

from __future__ import annotations

import csv
import json
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.objectives.mocu.context import (
    GLOBAL_SEED,
    ExperimentContext,
    update_posterior_vector,
)
from src.control.posterior_ctrl import normalize_log_weights
from src.policies.rl_sboed import (
    AdaptiveExperimentPolicy,
    PolicyConfig,
    StateValueCritic,
)
from src.policies.moe import (
    BeliefConditionedMoEPolicy,
    parameter_matched_expert_hidden,
)
from src.layout import model_dir
from src.domains.sir.design import chronological_feasible


def _eig_feasible(
    ctx: ExperimentContext,
    actions: list[int],
    *,
    remaining_steps: int | None = None,
) -> np.ndarray:
    """Return actions allowed by the domain's sequential-design constraints.

    SIR measurement times are chronological. Power-grid probe durations are
    independent experimental designs: they may be selected in any order, but
    an already-used duration is masked.

    ``remaining_steps`` counts actions still to choose *including* the current
    step (defaults to ``horizon - len(actions)``).
    """
    chronological = str(getattr(ctx, "observation_mode", "")).startswith("sir_")
    if chronological:
        rem = (
            int(remaining_steps)
            if remaining_steps is not None
            else max(int(ctx.horizon) - len(actions), 0)
        )
        return chronological_feasible(
            ctx.n_actions, actions, remaining_steps=rem
        )
    return np.asarray(
        [a for a in range(ctx.n_actions) if a not in set(actions)],
        dtype=int,
    )


METHODS = (
    "dad_eig",
    "rl_sboed_eig",
    "moe_sboed",
    "myopic_delta_h",
    "random",
    "fixed_open_loop",
)


def _soft_bc_loss(
    logits: torch.Tensor,
    scores: np.ndarray,
    feasible: np.ndarray,
    *,
    temperature: float,
) -> torch.Tensor:
    """Distill a full action-value ranking (soft labels), not only the argmax.

    Hard CE to argmax makes DAD/RL/MoE myopic clones. Soft KL preserves near-ties
    that matter for sequential EIG on SIR and continuous-duration grids.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    feasible_scores = np.asarray(scores, dtype=np.float64)[feasible]
    if feasible_scores.size == 0 or not np.all(np.isfinite(feasible_scores)):
        raise RuntimeError(
            "Non-finite EIG scores reached behavioral cloning for feasible "
            f"actions {np.asarray(feasible, dtype=int).tolist()}"
        )
    temp = max(float(temperature), 1e-3)
    feas = torch.as_tensor(np.asarray(feasible, dtype=int), device=logits.device)
    feasible_logits = logits[:, feas]
    if bool((feasible_logits <= -1e8).any()):
        raise RuntimeError(
            "EIG behavioral-cloning feasible actions disagree with the policy "
            "feasible-action mask; refusing to optimize invalid targets."
        )
    raw = torch.as_tensor(
        feasible_scores,
        dtype=torch.float32,
        device=logits.device,
    )
    # Soft target over the feasible set only.
    target = torch.softmax((raw - raw.max()) / temp, dim=-1)
    masked = torch.full(
        (logits.shape[0], logits.shape[1]),
        -1e9,
        device=logits.device,
        dtype=logits.dtype,
    )
    masked[:, feas] = feasible_logits
    log_p = torch.log_softmax(masked, dim=-1)[:, feas]
    return torch.sum(
        target * (torch.log(target.clamp_min(1e-8)) - log_p.squeeze(0))
    )


class VectorEIGEngine:
    """CUDA-batched posterior and expected one-step EIG calculations."""

    def __init__(self, ctx: ExperimentContext, device: torch.device):
        self.ctx = ctx
        self.device = device
        self.centres = torch.as_tensor(
            np.transpose(ctx.centres_support, (1, 0, 2)),
            dtype=torch.float32,
            device=device,
        )  # P,A,D
        self.log_p0 = torch.as_tensor(
            ctx.log_p0, dtype=torch.float32, device=device
        )
        self.sigma = float(ctx.sigma_y)
        self.sigma2 = self.sigma**2

    @staticmethod
    def entropy(log_w: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(log_w, dim=-1)
        return -(p * torch.log(p.clamp_min(1e-30))).sum(dim=-1)

    def update(
        self, log_w: torch.Tensor, action: int, observation: torch.Tensor
    ) -> torch.Tensor:
        diff = self.centres[:, int(action), :] - observation
        return log_w - 0.5 * torch.sum(diff * diff, dim=-1) / self.sigma2

    @torch.no_grad()
    def action_scores(
        self,
        log_w: torch.Tensor,
        feasible: np.ndarray,
        *,
        n_fantasies: int,
        seed: int,
    ) -> np.ndarray:
        """Expected one-step entropy reduction for all feasible actions."""
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        p = torch.softmax(log_w, dim=-1)
        h0 = self.entropy(log_w)
        n_particles = len(p)
        sample_ids = torch.multinomial(
            p, int(n_fantasies), replacement=True, generator=generator
        )
        out = np.full(self.ctx.n_actions, -np.inf, dtype=np.float64)
        # Action chunks cap peak memory: C,S,P,D.
        for start in range(0, len(feasible), 32):
            acts_np = feasible[start : start + 32]
            acts = torch.as_tensor(acts_np, dtype=torch.long, device=self.device)
            clean = self.centres[sample_ids[:, None], acts[None, :], :]
            clean = clean.permute(1, 0, 2)  # C,S,D
            noise = torch.randn(
                clean.shape,
                generator=generator,
                device=self.device,
                dtype=clean.dtype,
            ) * self.sigma
            y = clean + noise
            centres = self.centres[:, acts, :].permute(1, 0, 2)  # C,P,D
            distances = torch.cdist(
                y,
                centres,
                p=2.0,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            quad = distances * distances
            ll = -0.5 * quad / self.sigma2
            post_h = self.entropy(log_w[None, None, :] + ll)
            gains = h0 - post_h.mean(dim=1)
            if not bool(torch.isfinite(gains).all()):
                raise RuntimeError(
                    "Non-finite one-step EIG encountered; refusing to train on "
                    f"invalid targets (actions={acts_np.tolist()}, seed={seed})."
                )
            out[acts_np] = gains.detach().cpu().numpy()
        return out

    @torch.no_grad()
    def two_step_scores(
        self,
        log_w: torch.Tensor,
        feasible: np.ndarray,
        *,
        n_fantasies: int,
        seed: int,
        has_future_step: bool,
    ) -> np.ndarray:
        """One-step EIG plus a posterior-conditioned continuation value.

        The score is defined for every feasible action, so the selected action
        is not restricted to any expert's top-1 proposal.
        """
        immediate = self.action_scores(
            log_w, feasible, n_fantasies=n_fantasies, seed=seed
        )
        if not has_future_step or len(feasible) <= 1:
            return immediate
        generator = torch.Generator(device=self.device).manual_seed(int(seed) + 37)
        p = torch.softmax(log_w, dim=-1)
        sample_ids = torch.multinomial(
            p, max(2, int(n_fantasies) // 2), replacement=True, generator=generator
        )
        out = immediate.copy()
        for action in feasible:
            clean = self.centres[sample_ids, int(action), :]
            noise = torch.randn(
                clean.shape,
                generator=generator,
                device=self.device,
                dtype=clean.dtype,
            ) * self.sigma
            continuation = []
            if str(getattr(self.ctx, "observation_mode", "")).startswith("sir_"):
                next_feasible = feasible[feasible > int(action)]
            else:
                next_feasible = feasible[feasible != int(action)]
            if next_feasible.size == 0:
                continue
            for fantasy_id, observation in enumerate(clean + noise):
                post = self.update(log_w, int(action), observation)
                next_scores = self.action_scores(
                    post,
                    next_feasible,
                    n_fantasies=max(2, int(n_fantasies) // 2),
                    seed=int(seed) + 1009 * (int(action) + 1) + fantasy_id,
                )
                best_next = float(np.max(next_scores[next_feasible]))
                if not np.isfinite(best_next):
                    raise RuntimeError(
                        "Non-finite two-step EIG continuation encountered "
                        f"(action={int(action)}, seed={seed})."
                    )
                continuation.append(best_next)
            continuation_mean = float(np.mean(continuation))
            if not np.isfinite(continuation_mean):
                raise RuntimeError(
                    "Non-finite mean two-step EIG continuation encountered "
                    f"(action={int(action)}, seed={seed})."
                )
            out[int(action)] += continuation_mean
        if not np.all(np.isfinite(out[feasible])):
            raise RuntimeError(
                "Non-finite two-step EIG scores encountered for feasible actions "
                f"(seed={seed})."
            )
        return out


def _policy_tensors(
    ctx: ExperimentContext,
    actions: list[int],
    observations: list[np.ndarray],
    log_w: torch.Tensor,
    *,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Build an information-only EIG policy state.

    The shared objective-policy encoder also exposes posterior summaries of
    ``U`` and the derived control decision.  Those are valid for MOCU design
    but would blur the intended separation of the pure-EIG baseline.  Keep the
    common tensor shapes, while masking every objective-specific feature:

    * belief[7:29] -- U quantiles, u_ctrl, and U-level posterior masses;
    * legacy 3-column particles[..., 2] -- standardized MOCU-only U.

    New EIG contexts contain only the full machine-wise [M..., K...] latent
    vector, so no particle coordinate is masked in that representation.

    EIG policies retain history, ESS, maximum posterior mass, M/K summaries,
    posterior particle weights, and the feasible-action mask.
    """
    from src.objectives.mocu.train import _tensors_from_state

    tensors = _tensors_from_state(
        ctx,
        actions=actions,
        observations=observations,
        log_w=log_w.detach().cpu().numpy(),
        step=step,
        device=device,
    )
    action_idx, obs, mask, belief, steps, particles, weights, feasible = tensors
    expected_actions = _eig_feasible(
        ctx,
        actions,
        remaining_steps=max(int(ctx.horizon) - int(step), 0),
    )
    expected_feasible = torch.zeros_like(feasible)
    expected_feasible[
        0, torch.as_tensor(expected_actions, dtype=torch.long, device=device)
    ] = True
    if not torch.equal(feasible, expected_feasible):
        raise RuntimeError(
            "EIG scoring and policy feasible-action masks disagree "
            f"at step={step}, history={actions}."
        )
    belief = belief.clone()
    belief[..., 7:29] = 0.0
    particles = particles.clone()
    if particles.shape[-1] == 3:
        # Backward compatibility for legacy/SIR contexts built as [M, K, U].
        particles[..., 2] = 0.0
    return action_idx, obs, mask, belief, steps, particles, weights, feasible


def _observe(
    system: dict[str, Any],
    action: int,
    *,
    sigma: float,
    rollout_id: int,
    step: int,
    eval_seed: int | None = None,
) -> np.ndarray:
    """Additive Gaussian noise. Training keeps GLOBAL_SEED; eval uses eval_seed."""
    clean = np.asarray(system["obs_clean"][int(action)], dtype=np.float32)
    base = int(GLOBAL_SEED if eval_seed is None else eval_seed)
    rng = np.random.default_rng(
        base + 97_451 * int(rollout_id) + 104_729 * int(step)
    )
    return clean + rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)


def _load_policy(
    ctx: ExperimentContext, name: str, device: torch.device
) -> AdaptiveExperimentPolicy:
    path = model_dir(ctx.out_dir) / f"{name}.pth"
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = dict(payload.get("meta", {}))
    policy_cls = (
        BeliefConditionedMoEPolicy
        if "moe" in str(meta.get("architecture", ""))
        or "matched_dense" in str(meta.get("architecture", ""))
        or name in {"moe_sboed", "matched_dense"}
        else AdaptiveExperimentPolicy
    )
    training = dict(ctx.cfg.raw.get("training") or {})
    hidden = int(
        meta.get("policy_hidden")
        or training.get("policy_hidden", 128)
    )
    config = PolicyConfig(
        max_steps=ctx.horizon,
        obs_dim=ctx.obs_dim,
        summary_dim=33,
        hidden=hidden,
        particle_dim=int(ctx.particle_features.shape[1]),
    )
    if policy_cls is BeliefConditionedMoEPolicy:
        policy = policy_cls(
            ctx.n_actions,
            config,
            n_experts=int(meta.get("n_experts", 4)),
            top_k=int(meta.get("top_k", 2)),
            expert_hidden=int(meta.get("expert_hidden", hidden)),
        ).to(device)
    else:
        policy = policy_cls(ctx.n_actions, config).to(device)
    sd = payload["state_dict"]
    # Older MoE checkpoints lack belief residual_gate; load non-strictly.
    if isinstance(policy, BeliefConditionedMoEPolicy) and not any(
        k.startswith("residual_gate.") for k in sd
    ):
        policy.load_state_dict(sd, strict=False)
    else:
        policy.load_state_dict(sd)
    policy.eval()
    return policy


def train_vector_eig_policy(
    ctx: ExperimentContext,
    *,
    method: str,
    smoke: bool,
    seed: int,
) -> dict[str, Any]:
    """Train a dense or belief-conditioned MoE policy for vector EIG."""
    if method not in {"dad_eig", "rl_sboed_eig", "moe_sboed", "matched_dense"}:
        raise ValueError(method)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = VectorEIGEngine(ctx, device)
    training = dict(ctx.cfg.raw.get("training") or {})
    hidden = int(training.get("policy_hidden", 128))
    config = PolicyConfig(
        max_steps=ctx.horizon,
        obs_dim=ctx.obs_dim,
        summary_dim=33,
        hidden=hidden,
        particle_dim=int(ctx.particle_features.shape[1]),
    )
    is_residual_policy = method in {"moe_sboed", "matched_dense"}
    if is_residual_policy:
        reference_experts = int(training.get("eig_moe_n_experts", 4))
        matched = method == "matched_dense"
        expert_hidden = (
            parameter_matched_expert_hidden(
                hidden=hidden,
                n_actions=ctx.n_actions,
                reference_experts=reference_experts,
                reference_expert_hidden=hidden,
            )
            if matched
            else hidden
        )
        policy = BeliefConditionedMoEPolicy(
            ctx.n_actions,
            config,
            n_experts=1 if matched else reference_experts,
            top_k=1 if matched else int(training.get("eig_moe_top_k", 2)),
            expert_hidden=expert_hidden,
            logit_scale_init=float(training.get("eig_moe_logit_scale_init", 3.0)),
            balance_coefficient=float(
                training.get("eig_moe_balance_coefficient", 0.001)
            ),
            routing_information_coefficient=float(
                training.get("eig_moe_routing_information_coefficient", 0.0)
            ),
            redundancy_coefficient=float(
                training.get("eig_moe_redundancy_coefficient", 0.01)
            ),
        ).to(device)
    else:
        policy = AdaptiveExperimentPolicy(ctx.n_actions, config).to(device)
    epochs = 2 if smoke else int(training.get("eig_epochs", 20))
    steps_per_epoch = 16 if smoke else int(
        training.get("eig_steps_per_epoch", len(ctx.train_systems))
    )
    batch_size = 4 if smoke else int(training.get("batch_size", 16))
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=1e-4,
    )
    critic = None
    critic_optimizer = None
    ppo_epochs = 2 if smoke else int(training.get("eig_moe_ppo_epochs", 4))
    ppo_clip = float(training.get("eig_moe_ppo_clip", 0.2))
    entropy_coef = float(training.get("entropy_coef", 0.01))
    # Counterfactual teacher ranking is OFF by default. Using two-step /
    # myopic argmax CE on fused logits made MoE copy other methods. Residuals
    # must change the ranking through PPO + belief-conditioned losses.
    cf_coefficient = float(training.get("eig_moe_cf_coefficient", 0.0))
    cf_anneal_fraction = float(training.get("eig_moe_cf_anneal_fraction", 0.0))
    cf_floor_fraction = float(training.get("eig_moe_cf_floor_fraction", 0.0))
    cf_rollouts_per_batch = int(training.get("eig_moe_cf_rollouts_per_batch", 2))
    branching_coefficient = float(
        training.get("eig_moe_branching_coefficient", 0.15)
    )
    low_ess_residual_coefficient = float(
        training.get("eig_moe_low_ess_residual_coefficient", 0.10)
    )
    low_ess_threshold = float(training.get("eig_moe_low_ess_threshold", 0.4))
    residual_scale_coefficient = float(
        training.get("eig_moe_residual_scale_coefficient", 0.05)
    )
    residual_scale_target = float(
        training.get("eig_moe_logit_scale_target", 3.5)
    )
    freeze_base_after_bc = bool(
        training.get("eig_moe_freeze_base_after_bc", False)
    )
    leave_base_coefficient = float(
        training.get("eig_moe_leave_base_coefficient", 0.0)
    )
    leave_base_margin = float(training.get("eig_moe_leave_base_margin", 0.5))
    # Soft BC + optional two-step labels beat hard myopic CE (SIR was stuck ≈ myopic).
    bc_temperature = float(training.get("eig_bc_temperature", 0.5))
    bc_lookahead = str(training.get("eig_bc_lookahead", "two_step")).lower()
    bc_fantasies = int(training.get("eig_bc_fantasies", 16 if not smoke else 4))
    # RL-sBOED uses PPO with stepwise reward-to-go.  DAD may use the same
    # variance-reduction machinery while retaining its distinct terminal-EIG
    # return at every step; this changes the estimator, not DAD's objective.
    rl_use_ppo = bool(training.get("eig_rl_use_ppo", True))
    dad_use_ppo = bool(training.get("eig_dad_use_ppo", False))
    use_actor_critic = isinstance(policy, BeliefConditionedMoEPolicy) or (
        method == "rl_sboed_eig" and rl_use_ppo
    ) or (
        method == "dad_eig" and dad_use_ppo
    )
    if use_actor_critic:
        critic = StateValueCritic(
            ctx.n_actions,
            PolicyConfig(
                max_steps=ctx.horizon,
                obs_dim=ctx.obs_dim,
                summary_dim=33,
                hidden=hidden,
                particle_dim=int(ctx.particle_features.shape[1]),
            ),
        ).to(device)
        critic_optimizer = torch.optim.AdamW(
            critic.parameters(),
            lr=float(training.get("eig_moe_critic_lr", 1e-3)),
            weight_decay=1e-4,
        )
    # Soft unique floor for EIG checkpointing (same idea as MOCU MoE).
    min_unique_frac = float(training.get("eig_min_unique_sequence_fraction", 0.05))
    unique_eig_slack = float(training.get("eig_unique_floor_slack", 0.02))
    prefer_unique_floor = bool(training.get("eig_prefer_unique_sequence_floor", True))
    rng = np.random.default_rng(int(seed))
    baseline = np.zeros(ctx.horizon, dtype=np.float64)
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    # Behavioral-cloning warm start: soft distillation of (optional) two-step
    # scores gives all amortized policies a strong, observation-conditioned
    # initialization before high-variance policy gradients.
    bc_trajectories = 12 if smoke else int(
        training.get("eig_bc_trajectories", 128)
    )
    bc_losses = []
    policy.train()
    print(
        f"[eig:{method}] BC warm-start trajectories={bc_trajectories} "
        f"horizon={ctx.horizon} n_actions={ctx.n_actions} "
        f"lookahead={bc_lookahead} temp={bc_temperature} "
        f"actor_critic={use_actor_critic} obs_seed={int(seed)}",
        flush=True,
    )
    for bc_id in range(bc_trajectories):
        if bc_id == 0 or (bc_id + 1) % max(1, bc_trajectories // 8) == 0 or (
            bc_id + 1
        ) == bc_trajectories:
            print(
                f"[eig:{method}] BC {bc_id + 1}/{bc_trajectories}",
                flush=True,
            )
        system = ctx.train_systems[int(rng.integers(len(ctx.train_systems)))]
        actions: list[int] = []
        observations: list[np.ndarray] = []
        log_w = engine.log_p0.clone()
        trajectory_losses = []
        for step in range(ctx.horizon):
            feasible = _eig_feasible(ctx, actions)
            if bc_lookahead in {"two_step", "2step", "two-step"}:
                scores = engine.two_step_scores(
                    log_w,
                    feasible,
                    n_fantasies=bc_fantasies,
                    seed=int(seed) + bc_id * 1009 + step,
                    has_future_step=step < ctx.horizon - 1,
                )
            else:
                scores = engine.action_scores(
                    log_w,
                    feasible,
                    n_fantasies=bc_fantasies,
                    seed=int(seed) + bc_id * 1009 + step,
                )
            label = int(np.argmax(scores))
            tensors = _policy_tensors(
                ctx,
                actions,
                observations,
                log_w,
                step=step,
                device=device,
            )
            if isinstance(policy, BeliefConditionedMoEPolicy):
                # Distill generalist expert 0 only. Fused MoE logits are never
                # trained toward the two-step / myopic teacher.
                logits = policy.base_logits(*tensors[:-1]).masked_fill(
                    ~tensors[-1], -1e9
                )
            else:
                logits = policy(*tensors)
            imitation = _soft_bc_loss(
                logits, scores, feasible, temperature=bc_temperature
            )
            if isinstance(policy, BeliefConditionedMoEPolicy):
                # Soft KL alone left expert-0 greedy on a weak 0.5s probe.
                # A small CE locks t=0 to the two-step argmax (same first
                # action DAD/RL use); fused logits are still not cloned.
                imitation = imitation + 0.5 * F.cross_entropy(
                    logits,
                    torch.as_tensor([label], device=logits.device),
                )
            trajectory_losses.append(imitation)
            y_np = _observe(
                system,
                label,
                sigma=ctx.sigma_y,
                rollout_id=50_000 + bc_id,
                step=step,
                eval_seed=int(seed),
            )
            log_w = engine.update(
                log_w, label, torch.as_tensor(y_np, device=device)
            )
            actions.append(label)
            observations.append(y_np)
        optimizer.zero_grad(set_to_none=True)
        bc_loss = torch.stack(trajectory_losses).mean()
        bc_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        bc_losses.append(float(bc_loss.detach().item()))
    if isinstance(policy, BeliefConditionedMoEPolicy):
        policy.reinitialize_residual_experts()
        if freeze_base_after_bc:
            policy.freeze_base_head()
            print(
                f"[eig:{method}] froze generalist expert 0; specialists+router "
                "remain the fused MoE policy",
                flush=True,
            )
        else:
            print(
                f"[eig:{method}] specialist experts reinitialized; fused policy "
                "is the routed mixture (no teacher residual)",
                flush=True,
            )
        policy.reset_regime_prototypes()

    # Checkpoint differences between epochs are small (~0.02 nats), so a small
    # validation set makes model selection a lottery.  Default to the full
    # held-out validation bank unless the config restricts it.
    n_validation = 4 if smoke else int(training.get("eig_validation_systems", 128))
    validation_systems = ctx.validation_systems[:n_validation]
    validation_fixed = _fixed_sequence(
        ctx, engine, n_fantasies=4 if smoke else 12, seed=int(seed)
    )
    moe_step0_action: int | None = None
    if isinstance(policy, BeliefConditionedMoEPolicy):
        moe_step0_action = _prior_two_step_action(
            ctx,
            engine,
            n_fantasies=4 if smoke else 12,
            seed=int(seed),
        )
        print(
            f"[eig:{method}] pin t=0 to prior one-step action="
            f"{moe_step0_action} (MoE adaptive only for t>0)",
            flush=True,
        )

    def validation_eig() -> tuple[float, int]:
        policy.eval()
        rollouts = [
            _rollout(
                ctx,
                engine,
                system,
                rollout_id=70_000 + i,
                method=method,
                dad=policy,
                fixed_sequence=validation_fixed,
                n_fantasies=4 if smoke else 12,
                moe_step0_action=moe_step0_action,
                eval_seed=int(seed),
            )
            for i, system in enumerate(validation_systems)
        ]
        policy.train()
        return (
            float(np.mean([row["terminal_eig"] for row in rollouts])),
            len({tuple(row["sequence"]) for row in rollouts}),
        )

    # Terminal policy gradients are noisy. Preserve the strongest held-out
    # checkpoint, including the distilled initialization, rather than blindly
    # saving the final epoch.
    best_validation_eig, best_validation_unique = validation_eig()
    n_val_rollouts = max(len(validation_systems), 1)
    minimum_unique = max(
        2 if isinstance(policy, BeliefConditionedMoEPolicy) else 1,
        int(math.ceil(min_unique_frac * n_val_rollouts)),
    )
    best_meets_adaptivity = best_validation_unique >= minimum_unique
    best_admissible_eig = (
        best_validation_eig if best_meets_adaptivity else float("-inf")
    )
    best_stage = "behavioral_cloning"
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }
    fallback_eig = best_validation_eig
    fallback_unique = best_validation_unique
    fallback_stage = best_stage
    fallback_state = best_state
    for epoch in range(epochs):
        epoch_gains = []
        epoch_losses = []
        epoch_moe_stats: dict[str, float] = {}
        if cf_coefficient > 0.0 and isinstance(policy, BeliefConditionedMoEPolicy):
            if cf_anneal_fraction > 0.0:
                cf_anneal_epochs = max(1, int(round(cf_anneal_fraction * epochs)))
                cf_weight = max(
                    cf_coefficient * max(0.0, 1.0 - epoch / cf_anneal_epochs),
                    cf_coefficient * max(0.0, cf_floor_fraction),
                )
            else:
                cf_weight = cf_coefficient
        else:
            cf_weight = 0.0
        for batch_start in range(0, steps_per_epoch, batch_size):
            losses = []
            ppo_states: list[tuple[torch.Tensor, ...]] = []
            ppo_actions: list[torch.Tensor] = []
            ppo_old_log_probs: list[torch.Tensor] = []
            ppo_returns: list[float] = []
            cf_states: list[tuple[torch.Tensor, ...]] = []
            cf_targets: list[torch.Tensor] = []
            for sample_offset in range(
                min(batch_size, steps_per_epoch - batch_start)
            ):
                rollout_id = epoch * steps_per_epoch + batch_start + sample_offset
                system = ctx.train_systems[
                    int(rng.integers(len(ctx.train_systems)))
                ]
                actions: list[int] = []
                observations: list[np.ndarray] = []
                log_probs = []
                entropies = []
                rewards = []
                trajectory_states: list[tuple[torch.Tensor, ...]] = []
                trajectory_actions: list[torch.Tensor] = []
                trajectory_old_lp: list[torch.Tensor] = []
                log_w = engine.log_p0.clone()
                entropy_before = float(engine.entropy(log_w).item())
                collect_cf = (
                    cf_weight > 0.0
                    and isinstance(policy, BeliefConditionedMoEPolicy)
                    and sample_offset < cf_rollouts_per_batch
                )
                for step in range(ctx.horizon):
                    tensors = _policy_tensors(
                        ctx,
                        actions,
                        observations,
                        log_w,
                        step=step,
                        device=device,
                    )
                    if collect_cf:
                        feasible_np = _eig_feasible(ctx, actions)
                        cf_scores = engine.two_step_scores(
                            log_w,
                            feasible_np,
                            n_fantasies=bc_fantasies,
                            seed=int(seed) + 7919 * rollout_id + step,
                            has_future_step=step < ctx.horizon - 1,
                        )
                        cf_target = torch.zeros(1, ctx.n_actions, device=device)
                        cf_target[
                            0, torch.as_tensor(feasible_np, device=device)
                        ] = torch.as_tensor(
                            cf_scores[feasible_np], dtype=torch.float32, device=device
                        )
                        cf_states.append(tuple(t.detach() for t in tensors))
                        cf_targets.append(cf_target)
                    if use_actor_critic:
                        with torch.no_grad():
                            dist = policy.distribution(*tensors)
                            if (
                                moe_step0_action is not None
                                and step == 0
                            ):
                                action_t = torch.as_tensor(
                                    [moe_step0_action],
                                    device=device,
                                    dtype=torch.long,
                                )
                            else:
                                action_t = dist.sample()
                            log_prob = dist.log_prob(action_t)
                            entropy = dist.entropy()
                        trajectory_states.append(tuple(t.detach() for t in tensors))
                        trajectory_actions.append(action_t.detach())
                        trajectory_old_lp.append(log_prob.detach())
                    else:
                        dist = policy.distribution(*tensors)
                        if moe_step0_action is not None and step == 0:
                            action_t = torch.as_tensor(
                                [moe_step0_action],
                                device=device,
                                dtype=torch.long,
                            )
                        else:
                            action_t = dist.sample()
                        log_prob = dist.log_prob(action_t)
                        entropy = dist.entropy()
                    action = int(action_t.item())
                    y_np = _observe(
                        system,
                        action,
                        sigma=ctx.sigma_y,
                        rollout_id=rollout_id,
                        step=step,
                        eval_seed=int(seed),
                    )
                    y = torch.as_tensor(y_np, device=device)
                    log_w = engine.update(log_w, action, y)
                    entropy_after = float(engine.entropy(log_w).item())
                    rewards.append(entropy_before - entropy_after)
                    entropy_before = entropy_after
                    actions.append(action)
                    observations.append(y_np)
                    log_probs.append(log_prob.squeeze(0))
                    entropies.append(entropy.squeeze(0))
                rewards_a = np.asarray(rewards)
                # Entropy reductions telescope exactly to terminal EIG.
                returns = (
                    np.asarray([rewards_a.sum()] * ctx.horizon)
                    if method == "dad_eig"
                    else np.cumsum(rewards_a[::-1])[::-1].copy()
                )
                if use_actor_critic:
                    ppo_states.extend(trajectory_states)
                    ppo_actions.extend(trajectory_actions)
                    ppo_old_log_probs.extend(trajectory_old_lp)
                    ppo_returns.extend(float(v) for v in returns)
                else:
                    baseline = 0.9 * baseline + 0.1 * returns
                    advantage = torch.as_tensor(
                        returns - baseline, dtype=torch.float32, device=device
                    )
                    lp = torch.stack(log_probs)
                    ent = torch.stack(entropies)
                    losses.append(
                        -torch.sum(lp * advantage) - entropy_coef * ent.sum()
                    )
                epoch_gains.append(float(rewards_a.sum()))
            if use_actor_critic:
                assert critic is not None and critic_optimizer is not None
                inputs = tuple(
                    torch.cat([state[i] for state in ppo_states], dim=0)
                    for i in range(len(ppo_states[0]))
                )
                action_t = torch.cat(ppo_actions).long()
                old_lp_t = torch.cat(ppo_old_log_probs).detach()
                return_t = torch.as_tensor(
                    ppo_returns, dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    old_values = critic(*inputs[:-1])
                    advantage_t = return_t - old_values
                    advantage_t = (advantage_t - advantage_t.mean()) / (
                        advantage_t.std() + 1e-8
                    )
                cf_inputs: tuple[torch.Tensor, ...] | None = None
                cf_target_t: torch.Tensor | None = None
                if (
                    cf_weight > 0.0
                    and cf_states
                    and isinstance(policy, BeliefConditionedMoEPolicy)
                ):
                    cf_inputs = tuple(
                        torch.cat([state[i] for state in cf_states], dim=0)
                        for i in range(len(cf_states[0]))
                    )
                    cf_target_t = torch.cat(cf_targets, dim=0)
                last_loss = None
                for _ in range(ppo_epochs):
                    dist = policy.distribution(*inputs)
                    new_lp = dist.log_prob(action_t)
                    ratio = torch.exp(new_lp - old_lp_t)
                    clipped = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
                    policy_loss = -torch.min(
                        ratio * advantage_t, clipped * advantage_t
                    ).mean()
                    policy_loss = policy_loss - entropy_coef * dist.entropy().mean()
                    actor_loss = policy_loss
                    moe_stats: dict[str, float] = {}
                    if isinstance(policy, BeliefConditionedMoEPolicy):
                        auxiliary, moe_stats = policy.specialization_loss(
                            *inputs[:-1]
                        )
                        actor_loss = actor_loss + auxiliary
                        if low_ess_residual_coefficient > 0.0:
                            low_ess_loss, low_ess_stats = policy.low_ess_residual_loss(
                                *inputs[:-1],
                                feasible_mask=inputs[-1],
                                ess_threshold=low_ess_threshold,
                            )
                            actor_loss = (
                                actor_loss
                                + low_ess_residual_coefficient * low_ess_loss
                            )
                            moe_stats.update(low_ess_stats)
                        if residual_scale_coefficient > 0.0:
                            scale_loss, scale_stats = policy.residual_scale_floor_loss(
                                target=residual_scale_target,
                            )
                            actor_loss = (
                                actor_loss
                                + residual_scale_coefficient * scale_loss
                            )
                            moe_stats.update(scale_stats)
                        if branching_coefficient > 0.0:
                            branch_loss, branch_stats = policy.belief_branching_loss(
                                *inputs[:-1],
                                feasible_mask=inputs[-1],
                            )
                            actor_loss = (
                                actor_loss + branching_coefficient * branch_loss
                            )
                            moe_stats.update(branch_stats)
                        if leave_base_coefficient > 0.0:
                            leave_loss, leave_stats = policy.greedy_leave_base_loss(
                                *inputs[:-1],
                                feasible_mask=inputs[-1],
                                margin=leave_base_margin,
                            )
                            actor_loss = (
                                actor_loss + leave_base_coefficient * leave_loss
                            )
                            moe_stats.update(leave_stats)
                    if cf_inputs is not None and cf_target_t is not None:
                        cf_loss, cf_stats = policy.counterfactual_loss(
                            *cf_inputs[:-1],
                            target_utility=cf_target_t,
                            feasible_mask=cf_inputs[-1],
                        )
                        actor_loss = actor_loss + cf_weight * cf_loss
                        moe_stats.update(cf_stats)
                    optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()

                    value_loss = torch.nn.functional.huber_loss(
                        critic(*inputs[:-1]), return_t
                    )
                    critic_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    critic_optimizer.step()
                    last_loss = actor_loss.detach()
                assert last_loss is not None
                epoch_losses.append(float(last_loss.item()))
                epoch_moe_stats = moe_stats
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().item()))
        epoch_validation_eig, epoch_validation_unique = validation_eig()
        epoch_meets_adaptivity = epoch_validation_unique >= minimum_unique
        history.append(
            {
                "epoch": epoch + 1,
                "mean_terminal_eig": float(np.mean(epoch_gains)),
                "mean_loss": float(np.mean(epoch_losses)),
                "validation_terminal_eig": epoch_validation_eig,
                "validation_n_unique_sequences": epoch_validation_unique,
                "validation_meets_adaptivity": int(epoch_meets_adaptivity),
                "moe_cf_weight": float(cf_weight),
                **epoch_moe_stats,
            }
        )
        prefer_diverse_fallback = isinstance(
            policy, BeliefConditionedMoEPolicy
        ) and minimum_unique > 1
        better_fallback = (
            epoch_validation_unique > fallback_unique
            or (
                epoch_validation_unique == fallback_unique
                and epoch_validation_eig > fallback_eig
            )
            if prefer_diverse_fallback
            else epoch_validation_eig > fallback_eig
        )
        if better_fallback:
            fallback_eig = epoch_validation_eig
            fallback_unique = epoch_validation_unique
            fallback_stage = f"epoch_{epoch + 1}"
            fallback_state = {
                name: value.detach().cpu().clone()
                for name, value in policy.state_dict().items()
            }
        # Soft unique floor: diverse checkpoint may replace collapsed one if
        # EIG stays within slack; collapsed may demote diverse only for a clear win.
        if prefer_unique_floor:
            diverse_ok = epoch_meets_adaptivity and (
                epoch_validation_eig
                >= best_admissible_eig - unique_eig_slack
                or not best_meets_adaptivity
            )
            if diverse_ok and (
                epoch_validation_eig > best_admissible_eig
                or (
                    not best_meets_adaptivity
                    and epoch_validation_eig >= best_validation_eig - unique_eig_slack
                )
            ):
                best_admissible_eig = epoch_validation_eig
                best_validation_eig = epoch_validation_eig
                best_validation_unique = epoch_validation_unique
                best_meets_adaptivity = True
                best_stage = f"epoch_{epoch + 1}"
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in policy.state_dict().items()
                }
        elif epoch_meets_adaptivity and epoch_validation_eig > best_admissible_eig:
            best_admissible_eig = epoch_validation_eig
            best_validation_eig = epoch_validation_eig
            best_validation_unique = epoch_validation_unique
            best_meets_adaptivity = True
            best_stage = f"epoch_{epoch + 1}"
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in policy.state_dict().items()
            }
    if not best_meets_adaptivity:
        best_validation_eig = fallback_eig
        best_validation_unique = fallback_unique
        best_stage = fallback_stage
        best_state = fallback_state
    policy.load_state_dict(best_state)
    policy.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    path = model_dir(ctx.out_dir) / f"{method}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "critic_state_dict": (
                critic.state_dict() if critic is not None else None
            ),
            "meta": {
                "method": method,
                "objective": (
                    "stepwise_entropy_reduction"
                    if method == "rl_sboed_eig"
                    else "terminal_eig"
                ),
                "training_seed": int(seed),
                "policy_hidden": hidden,
                "particle_dim": int(ctx.particle_features.shape[1]),
                "obs_dim": ctx.obs_dim,
                "n_obs": ctx.n_obs,
                "policy_input": "eig_information_only_spatial_latent_v2",
                "policy_input_retained": (
                    "history,ESS,max_weight,M_summary,K_summary,"
                    "machinewise_MK_particles,posterior_weights,feasible_actions"
                ),
                "policy_input_masked": (
                    "U_quantiles,u_ctrl,U_level_masses"
                ),
                "experiment_dir": str(ctx.out_dir.resolve()),
                "architecture": (
                    "parameter_matched_dense_control_v1"
                    if method == "matched_dense"
                    else "belief_topk_mixture_moe_v4"
                    if method == "moe_sboed"
                    else "dense_policy"
                ),
                "n_experts": int(getattr(policy, "n_experts", 0)),
                "top_k": int(getattr(policy, "top_k", 0)),
                "expert_hidden": int(getattr(policy, "expert_hidden", 0)),
                "optimizer": (
                    "ppo_actor_critic" if use_actor_critic else "reinforce"
                ),
                "ppo_epochs": (ppo_epochs if use_actor_critic else 0),
                "ppo_clip": (ppo_clip if use_actor_critic else 0.0),
                # Stepwise entropy reductions telescope to terminal EIG, so
                # the MoE/RL return definition keeps the terminal objective while
                # restoring per-step credit assignment.
                "returns": (
                    "stepwise_returns_to_go"
                    if method in {"rl_sboed_eig", "moe_sboed", "matched_dense"}
                    else "terminal_broadcast"
                ),
                "eig_bc_lookahead": bc_lookahead,
                "eig_bc_temperature": bc_temperature,
                "eig_rl_use_ppo": bool(use_actor_critic and method == "rl_sboed_eig"),
                "eig_dad_use_ppo": bool(use_actor_critic and method == "dad_eig"),
                "eig_moe_cf_coefficient": (
                    cf_coefficient
                    if isinstance(policy, BeliefConditionedMoEPolicy)
                    else 0.0
                ),
                "eig_moe_cf_anneal_fraction": (
                    cf_anneal_fraction
                    if isinstance(policy, BeliefConditionedMoEPolicy)
                    else 0.0
                ),
                "eig_moe_cf_floor_fraction": (
                    cf_floor_fraction
                    if isinstance(policy, BeliefConditionedMoEPolicy)
                    else 0.0
                ),
                "eig_moe_cf_rollouts_per_batch": (
                    cf_rollouts_per_batch
                    if isinstance(policy, BeliefConditionedMoEPolicy)
                    else 0
                ),
                "eig_moe_branching_coefficient": (
                    branching_coefficient
                    if isinstance(policy, BeliefConditionedMoEPolicy)
                    else 0.0
                ),
                "eig_moe_balance_coefficient": float(
                    getattr(policy, "balance_coefficient", 0.0)
                ),
                "eig_moe_routing_information_coefficient": float(
                    getattr(policy, "routing_information_coefficient", 0.0)
                ),
                "eig_moe_redundancy_coefficient": float(
                    getattr(policy, "redundancy_coefficient", 0.0)
                ),
                "eig_moe_low_ess_residual_coefficient": (
                    low_ess_residual_coefficient
                    if isinstance(policy, BeliefConditionedMoEPolicy)
                    else 0.0
                ),
                "eig_moe_residual_scale_coefficient": (
                    residual_scale_coefficient
                    if isinstance(policy, BeliefConditionedMoEPolicy)
                    else 0.0
                ),
                "eig_moe_freeze_base_after_bc": (
                    freeze_base_after_bc
                    if isinstance(policy, BeliefConditionedMoEPolicy)
                    else False
                ),
                "eig_moe_leave_base_coefficient": (
                    leave_base_coefficient
                    if isinstance(policy, BeliefConditionedMoEPolicy)
                    else 0.0
                ),
                "moe_step0_action": moe_step0_action,
                "eig_moe_prototype_reset_after_warm_start": isinstance(
                    policy, BeliefConditionedMoEPolicy
                ),
            },
            "elapsed_seconds": elapsed,
            "history": history,
            "behavioral_cloning": {
                "trajectories": bc_trajectories,
                "mean_loss": float(np.mean(bc_losses)),
            },
            "model_selection": {
                "criterion": "held_out_terminal_eig",
                "best_stage": best_stage,
                "best_validation_terminal_eig": best_validation_eig,
                "best_validation_n_unique_sequences": best_validation_unique,
                "minimum_unique_sequences": minimum_unique,
                "best_meets_adaptivity": best_meets_adaptivity,
                "n_validation_systems": len(validation_systems),
            },
        },
        path,
    )
    return {
        "method": method,
        "checkpoint": str(path),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "history": history,
        "behavioral_cloning_trajectories": bc_trajectories,
        "behavioral_cloning_mean_loss": float(np.mean(bc_losses)),
        "best_stage": best_stage,
        "best_validation_terminal_eig": best_validation_eig,
        "best_validation_n_unique_sequences": best_validation_unique,
        "minimum_unique_sequences": minimum_unique,
        "best_meets_adaptivity": best_meets_adaptivity,
    }


def _prior_two_step_action(
    ctx: ExperimentContext,
    engine: VectorEIGEngine,
    *,
    n_fantasies: int,
    seed: int,
) -> int:
    """Open-loop first probe: one-step EIG at the prior (myopic ξ₁).

    Two-step scores at the prior collapsed to a weak 0.5s probe on IEEE9;
    myopic t=0 is always a 3.0s injection. MoE only adapts after this probe.
    """
    feasible = _eig_feasible(ctx, [])
    scores = engine.action_scores(
        engine.log_p0.clone(),
        feasible,
        n_fantasies=n_fantasies,
        seed=int(seed),
    )
    return int(np.argmax(scores))


def _fixed_sequence(
    ctx: ExperimentContext,
    engine: VectorEIGEngine,
    *,
    n_fantasies: int,
    seed: int | None = None,
) -> list[int]:
    """Open-loop Fixed: greedy one-step prior EIG (chronological on SIR)."""
    base = int(GLOBAL_SEED if seed is None else seed)
    seq: list[int] = []
    log_w = engine.log_p0.clone()
    for step in range(int(ctx.horizon)):
        feasible = _eig_feasible(ctx, seq)
        if feasible.size == 0:
            break
        scores = engine.action_scores(
            log_w,
            feasible,
            n_fantasies=n_fantasies,
            seed=base + step,
        )
        seq.append(int(np.argmax(scores)))
    return seq


@torch.no_grad()
def _rollout(
    ctx: ExperimentContext,
    engine: VectorEIGEngine,
    system: dict[str, Any],
    *,
    rollout_id: int,
    method: str,
    dad: AdaptiveExperimentPolicy | None,
    fixed_sequence: list[int],
    n_fantasies: int,
    moe_step0_action: int | None = None,
    eval_seed: int | None = None,
) -> dict[str, Any]:
    actions: list[int] = []
    observations: list[np.ndarray] = []
    log_w = engine.log_p0.clone()
    h0 = float(engine.entropy(log_w).item())
    step_eig = []
    trace = []
    rng_base = int(GLOBAL_SEED if eval_seed is None else eval_seed)
    for step in range(ctx.horizon):
        feasible = _eig_feasible(ctx, actions)
        myopic_scores = None
        if method == "myopic_delta_h":
            myopic_scores = engine.action_scores(
                log_w,
                feasible,
                n_fantasies=n_fantasies,
                seed=rng_base + 1009 * rollout_id + step,
            )
        if feasible.size == 0:
            raise RuntimeError(
                f"No chronologically feasible actions left at step {step} "
                f"(history={actions}). SIR requires ξ1 < ξ2 < … < ξT."
            )
        if method == "random":
            rng = np.random.default_rng(rng_base + rollout_id * 101 + step)
            action = int(rng.choice(feasible))
        elif (
            method == "moe_sboed"
            and step == 0
            and moe_step0_action is not None
        ):
            action = int(moe_step0_action)
        elif method == "fixed_open_loop":
            # Stay inside the chronological feasible set (SIR: increasing times).
            feasible_set = {int(a) for a in feasible.tolist()}
            action = next(
                int(a) for a in fixed_sequence if int(a) in feasible_set
            )
        elif method == "myopic_delta_h":
            action = int(np.argmax(myopic_scores))
        else:
            assert dad is not None
            tensors = _policy_tensors(
                ctx,
                actions,
                observations,
                log_w,
                step=step,
                device=engine.device,
            )
            logits = dad(*tensors).squeeze(0)
            # Mask already applied in policy tensors / feasible; argmax on logits.
            action = int(torch.argmax(logits).item())
            if int(action) not in {int(a) for a in feasible.tolist()}:
                # Defensive: fall back to best feasible logit if mask was missed.
                action = int(
                    max(feasible.tolist(), key=lambda a: float(logits[int(a)]))
                )
        y_np = _observe(
            system,
            action,
            sigma=ctx.sigma_y,
            rollout_id=rollout_id,
            step=step,
            eval_seed=eval_seed,
        )
        h_before = float(engine.entropy(log_w).item())
        log_w = engine.update(
            log_w, action, torch.as_tensor(y_np, device=engine.device)
        )
        h_after = float(engine.entropy(log_w).item())
        step_eig.append(h_before - h_after)
        actions.append(action)
        observations.append(y_np)
    return {
        "sequence": actions,
        "terminal_eig": h0 - float(engine.entropy(log_w).item()),
        "step_eig": step_eig,
        "router_trace": trace,
    }


_VECTOR_CHECKPOINTS = {
    "dad_eig": "dad_eig.pth",
    "rl_sboed_eig": "rl_sboed_eig.pth",
    "moe_sboed": "moe_sboed.pth",
    "matched_dense": "matched_dense.pth",
}


def evaluate_vector_eig(
    ctx: ExperimentContext,
    *,
    smoke: bool,
    methods: tuple[str, ...] | None = None,
    eval_seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate selected methods with common vector noise and write compact tables."""
    from src.layout import resolve_eval_seed

    if eval_seed is None:
        eval_seed = resolve_eval_seed(ctx.out_dir)
    eval_seed = int(eval_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = VectorEIGEngine(ctx, device)
    matched_path = model_dir(ctx.out_dir) / "matched_dense.pth"
    available_methods = METHODS + (
        ("matched_dense",) if matched_path.exists() else ()
    )
    selected_methods = available_methods if methods is None else tuple(methods)
    unknown = sorted(set(selected_methods) - set(available_methods))
    if unknown:
        raise ValueError(f"Unavailable vector-EIG methods: {unknown}")
    kept: list[str] = []
    for method in selected_methods:
        ckpt = _VECTOR_CHECKPOINTS.get(method)
        if ckpt is not None and not (model_dir(ctx.out_dir) / ckpt).is_file():
            print(f"[evaluate] skip {method}: missing {ckpt}")
            continue
        kept.append(method)
    selected_methods = tuple(kept)
    if not selected_methods:
        raise ValueError("No vector-EIG methods left to evaluate")
    dad = (
        _load_policy(ctx, "dad_eig", device)
        if "dad_eig" in selected_methods
        else None
    )
    rl = (
        _load_policy(ctx, "rl_sboed_eig", device)
        if "rl_sboed_eig" in selected_methods
        else None
    )
    moe = (
        _load_policy(ctx, "moe_sboed", device)
        if "moe_sboed" in selected_methods
        else None
    )
    matched = (
        _load_policy(ctx, "matched_dense", device)
        if "matched_dense" in selected_methods
        else None
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fixed: tuple[int, ...] = ()
    fixed_preparation_seconds = 0.0
    if "fixed_open_loop" in selected_methods:
        fixed_started = time.perf_counter()
        fixed = _fixed_sequence(
            ctx, engine, n_fantasies=4 if smoke else 16, seed=eval_seed
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fixed_preparation_seconds = float(time.perf_counter() - fixed_started)
    training_seconds = {}
    for method_name, checkpoint_name in (
        ("dad_eig", "dad_eig"),
        ("rl_sboed_eig", "rl_sboed_eig"),
    ):
        if method_name in selected_methods:
            payload = torch.load(
                model_dir(ctx.out_dir) / f"{checkpoint_name}.pth",
                map_location="cpu",
                weights_only=False,
            )
            training_seconds[method_name] = float(payload.get("elapsed_seconds", 0.0))
    moe_step0_action: int | None = None
    if "moe_sboed" in selected_methods:
        moe_payload = torch.load(
            model_dir(ctx.out_dir) / "moe_sboed.pth",
            map_location="cpu",
            weights_only=False,
        )
        training_seconds["moe_sboed"] = float(
            moe_payload.get("elapsed_seconds", 0.0)
        )
        meta = dict(moe_payload.get("meta") or {})
        if meta.get("moe_step0_action") is not None:
            moe_step0_action = int(meta["moe_step0_action"])
        else:
            moe_step0_action = _prior_two_step_action(
                ctx,
                engine,
                n_fantasies=4 if smoke else 16,
                seed=int(meta.get("training_seed", GLOBAL_SEED)),
            )
    if matched is not None:
        matched_payload = torch.load(
            matched_path, map_location="cpu", weights_only=False
        )
        training_seconds["matched_dense"] = float(
            matched_payload.get("elapsed_seconds", 0.0)
        )
    if "fixed_open_loop" in selected_methods:
        training_seconds["fixed_open_loop"] = fixed_preparation_seconds
    evaluation = dict(ctx.cfg.raw.get("evaluation") or {})
    n = min(
        4 if smoke else int(evaluation.get("eig_test_systems", 128)),
        len(ctx.test_systems),
    )
    summaries = []
    all_rows = []
    for method in selected_methods:
        policy = (
            rl
            if method == "rl_sboed_eig"
            else moe
            if method == "moe_sboed"
            else matched
            if method == "matched_dense"
            else dad
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        replicates = (8 if smoke else 32) if method == "random" else 1
        rows = []
        for i in range(n):
            for replicate in range(replicates):
                rollout_id = i * replicates + replicate
                row = _rollout(
                    ctx,
                    engine,
                    ctx.test_systems[i],
                    rollout_id=rollout_id,
                    method=method,
                    dad=policy
                    if method
                    not in {"random", "fixed_open_loop", "myopic_delta_h"}
                    else None,
                    fixed_sequence=fixed,
                    n_fantasies=4 if smoke else 16,
                    moe_step0_action=(
                        moe_step0_action if method == "moe_sboed" else None
                    ),
                    eval_seed=eval_seed,
                )
                row["theta_id"] = i
                row["design_replicate"] = replicate
                rows.append(row)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_method = float(time.perf_counter() - started)
        per_theta = np.asarray(
            [
                np.mean(
                    [
                        r["terminal_eig"]
                        for r in rows
                        if int(r["theta_id"]) == i
                    ]
                )
                for i in range(n)
            ],
            dtype=np.float64,
        )
        sem = (
            float(per_theta.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        )
        summaries.append(
            {
                "method": method,
                "n": n,
                "n_design_replicates": len(rows),
                "terminal_eig_mean": float(per_theta.mean()),
                "terminal_eig_std": (
                    float(per_theta.std(ddof=1)) if n > 1 else 0.0
                ),
                "ci95_low": float(per_theta.mean() - 1.96 * sem),
                "ci95_high": float(per_theta.mean() + 1.96 * sem),
                "seconds": elapsed_method,
                "offline_training_or_calibration_seconds": training_seconds.get(
                    method, 0.0
                ),
                # Short alias used by publication tables.  Fixed records its
                # offline design preparation; learned policies record training.
                "training_time_seconds": training_seconds.get(method, 0.0),
                "online_seconds_per_rollout": float(
                    elapsed_method / max(len(rows), 1)
                ),
                "n_unique_sequences": len({tuple(r["sequence"]) for r in rows}),
                "eval_seed": int(eval_seed),
                "timing_scope": (
                    "offline=method-specific preparation; online=CUDA-synchronized warm "
                    "action selection + observation lookup + posterior update; shared "
                    "physical-bank generation excluded"
                ),
            }
        )
        for i, row in enumerate(rows):
            all_rows.append({"method": method, "rollout_id": i, **row})
    eval_dir = ctx.out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "terminal_eig_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    observation_name = (
        "sir_infected_count"
        if str(getattr(ctx, "observation_mode", "")).startswith("sir_")
        else "sampled_delta_f_vector"
    )
    (eval_dir / "vector_eig_results.json").write_text(
        json.dumps(
            {
                "observation": observation_name,
                "n_obs": ctx.n_obs,
                "device": str(device),
                "eval_seed": int(eval_seed),
                "summaries": summaries,
                "rollouts": all_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "observation": observation_name,
        "n_obs": ctx.n_obs,
        "device": str(device),
        "eval_seed": int(eval_seed),
        "summaries": summaries,
    }


@torch.no_grad()
def diagnose_vector_eig_moe(
    ctx: ExperimentContext,
    *,
    n_rollouts: int = 128,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Measure whether the EIG MoE uses belief-conditioned residual regimes."""
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    engine = VectorEIGEngine(ctx, device)
    policy = _load_policy(ctx, "moe_sboed", device)
    if not isinstance(policy, BeliefConditionedMoEPolicy):
        raise RuntimeError("moe_sboed checkpoint is not a belief-conditioned MoE")
    policy.eval()
    records: list[dict[str, Any]] = []
    sequences: list[tuple[int, ...]] = []
    systems = ctx.test_systems[: min(int(n_rollouts), len(ctx.test_systems))]
    global_scale = float(
        torch.nn.functional.softplus(policy.logit_scale).clamp(max=20.0).item()
    )
    for rollout_id, system in enumerate(systems):
        actions: list[int] = []
        observations: list[np.ndarray] = []
        log_w = engine.log_p0.clone()
        for step in range(ctx.horizon):
            tensors = _policy_tensors(
                ctx,
                actions,
                observations,
                log_w,
                step=step,
                device=device,
            )
            feasible = tensors[-1]
            fused, router, experts, scale, base = policy._components(*tensors[:-1])
            fused = fused.masked_fill(~feasible, -1e9)
            base = base.masked_fill(~feasible, -1e9)
            experts = experts.masked_fill(~feasible[:, None, :], -1e9)
            fused_action = int(fused.argmax(-1).item())
            base_action = int(base.argmax(-1).item())
            expert_actions = experts.argmax(-1).squeeze(0).cpu().tolist()
            weights = torch.softmax(log_w, dim=-1)
            ess = float(1.0 / weights.square().sum().item())
            router_row = router.squeeze(0)
            router_entropy = float(
                -(router_row * router_row.clamp_min(1e-8).log()).sum().item()
            )
            top2_mass = float(
                router_row.topk(min(2, policy.n_experts)).values.sum().item()
            )
            pair_disagree = []
            for left in range(policy.n_experts):
                for right in range(left + 1, policy.n_experts):
                    pair_disagree.append(
                        float(expert_actions[left] != expert_actions[right])
                    )
            records.append(
                {
                    "rollout_id": rollout_id,
                    "step": step,
                    "ess": ess,
                    "router_entropy": router_entropy,
                    "top2_router_mass": top2_mass,
                    "dominant_expert": int(router_row.argmax().item()),
                    "residual_gate": float(scale.item() / max(global_scale, 1e-8)),
                    "base_action": base_action,
                    "fused_action": fused_action,
                    "base_fused_disagree": int(base_action != fused_action),
                    "n_distinct_expert_actions": len(set(expert_actions)),
                    "pairwise_expert_disagreement": (
                        float(np.mean(pair_disagree)) if pair_disagree else 0.0
                    ),
                    **{
                        f"router_weight_{i}": float(router_row[i].item())
                        for i in range(policy.n_experts)
                    },
                    **{
                        f"expert_action_{i}": int(expert_actions[i])
                        for i in range(policy.n_experts)
                    },
                }
            )
            y_np = _observe(
                system,
                fused_action,
                sigma=ctx.sigma_y,
                rollout_id=rollout_id,
                step=step,
            )
            log_w = engine.update(
                log_w, fused_action, torch.as_tensor(y_np, device=device)
            )
            actions.append(fused_action)
            observations.append(y_np)
        sequences.append(tuple(actions))
    usage = np.bincount(
        [int(row["dominant_expert"]) for row in records],
        minlength=policy.n_experts,
    )
    summary = {
        "method": "moe_sboed",
        "experiment_type": "eig_based",
        "n_rollouts": len(systems),
        "n_records": len(records),
        "n_experts": policy.n_experts,
        "top_k": policy.top_k,
        "active_dominant_experts": int((usage > 0).sum()),
        "dominant_expert_counts": usage.tolist(),
        "mean_router_entropy": float(np.mean([r["router_entropy"] for r in records])),
        "mean_top2_router_mass": float(np.mean([r["top2_router_mass"] for r in records])),
        "mean_residual_gate": float(np.mean([r["residual_gate"] for r in records])),
        "base_fused_disagreement_rate": float(
            np.mean([r["base_fused_disagree"] for r in records])
        ),
        "mean_pairwise_expert_disagreement": float(
            np.mean([r["pairwise_expert_disagreement"] for r in records])
        ),
        "mean_distinct_expert_actions": float(
            np.mean([r["n_distinct_expert_actions"] for r in records])
        ),
        "n_unique_fused_sequences": len(set(sequences)),
    }
    diagnostics_dir = ctx.out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    with (diagnostics_dir / "eig_moe_router_records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (diagnostics_dir / "eig_moe_mechanism_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
