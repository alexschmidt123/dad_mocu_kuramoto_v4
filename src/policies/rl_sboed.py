"""Production policies for DAD and RL-sBOED (shared history + belief backbone).

Manuscript methods: DAD, RL-sBOED.
This module is self-contained. Experimental Stage-2 encoder screens live under
``src.policies.legacy``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_SUMMARY_DIM = 33
DEFAULT_PARTICLE_DIM = 3


@dataclass(frozen=True)
class PolicyConfig:
    """Production policy configuration."""

    hidden: int = 128
    max_steps: int = 3
    summary_dim: int = DEFAULT_SUMMARY_DIM
    particle_dim: int = DEFAULT_PARTICLE_DIM
    obs_dim: int = 1


class _HistoryEncoder(nn.Module):
    """Encode the ordered probe-observation history."""

    def __init__(
        self,
        n_actions: int,
        hidden: int = 128,
        max_steps: int = 3,
        obs_dim: int = 1,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.hidden = hidden
        self.obs_dim = int(obs_dim)
        self.position = nn.Embedding(max_steps, 16)
        self.pair = nn.Sequential(
            nn.Linear(n_actions + self.obs_dim + 16, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden))

    def forward(
        self,
        action_indices: torch.Tensor,
        normalized_observations: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = action_indices.shape
        if length == 0:
            return torch.zeros(
                batch,
                self.hidden,
                device=action_indices.device,
                dtype=normalized_observations.dtype,
            )
        one_hot = F.one_hot(
            action_indices.clamp(min=0), num_classes=self.n_actions
        ).float()
        positions = self.position(
            torch.arange(length, device=action_indices.device)
        ).unsqueeze(0).expand(batch, -1, -1)
        if normalized_observations.ndim == 2:
            obs = normalized_observations.clamp(-8.0, 8.0).unsqueeze(-1)
        else:
            obs = normalized_observations.clamp(-8.0, 8.0)
        pair_input = torch.cat(
            [
                one_hot,
                obs,
                positions,
            ],
            dim=-1,
        )
        encoded = self.pair(pair_input)
        scores = torch.matmul(
            self.query.expand(batch, -1, -1), encoded.transpose(1, 2)
        ) * (self.hidden**-0.5)
        scores = scores.masked_fill(history_mask.unsqueeze(1) <= 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0)
        pooled = torch.matmul(attention, encoded).squeeze(1)
        pooled = pooled.masked_fill(
            (history_mask.sum(dim=1) <= 0).unsqueeze(-1), 0.0
        )
        return self.output(pooled)


class _BeliefEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        return self.network(summary)


class _ParticleSetEncoder(nn.Module):
    """Weighted pool of per-particle features (uses current posterior weights)."""

    def __init__(self, particle_dim: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(particle_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

    def forward(
        self, particle_features: torch.Tensor, posterior_weights: torch.Tensor
    ) -> torch.Tensor:
        # particle_features: (B, N, D), posterior_weights: (B, N)
        h = self.network(particle_features)
        w = posterior_weights.clamp_min(0.0)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return (h * w.unsqueeze(-1)).sum(dim=1)


class _PolicyBackbone(nn.Module):
    def __init__(self, n_actions: int, config: PolicyConfig):
        super().__init__()
        hidden = config.hidden
        self.history_encoder = _HistoryEncoder(
            n_actions, hidden, config.max_steps, obs_dim=config.obs_dim
        )
        self.belief_encoder = _BeliefEncoder(config.summary_dim, hidden)
        self.particle_encoder = _ParticleSetEncoder(config.particle_dim, hidden)
        self.step_encoder = nn.Embedding(config.max_steps + 1, 32)
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden + 32, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

    def forward(
        self,
        action_indices: torch.Tensor,
        normalized_observations: torch.Tensor,
        history_mask: torch.Tensor,
        belief_summary: torch.Tensor,
        steps: torch.Tensor,
        particle_features: torch.Tensor,
        posterior_weights: torch.Tensor,
    ) -> torch.Tensor:
        history = self.history_encoder(
            action_indices, normalized_observations, history_mask
        )
        belief = self.belief_encoder(belief_summary)
        particles = self.particle_encoder(particle_features, posterior_weights)
        return self.fusion(
            torch.cat(
                [history, belief, particles, self.step_encoder(steps.long())],
                dim=-1,
            )
        )


class AdaptiveExperimentPolicy(nn.Module):
    """History + belief policy used by DAD and RL-sBOED."""

    def __init__(self, n_actions: int, config: PolicyConfig | None = None):
        super().__init__()
        self.config = config or PolicyConfig()
        self.n_actions = n_actions
        self.backbone = _PolicyBackbone(n_actions, self.config)
        self.action_head = nn.Linear(self.config.hidden, n_actions)

    def forward(
        self,
        action_indices: torch.Tensor,
        normalized_observations: torch.Tensor,
        history_mask: torch.Tensor,
        belief_summary: torch.Tensor,
        steps: torch.Tensor,
        particle_features: torch.Tensor,
        posterior_weights: torch.Tensor,
        feasible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.action_head(
            self.backbone(
                action_indices,
                normalized_observations,
                history_mask,
                belief_summary,
                steps,
                particle_features,
                posterior_weights,
            )
        ).clamp(-50.0, 50.0)
        if feasible_mask is not None:
            logits = logits.masked_fill(~feasible_mask, -1e9)
        return logits

    def distribution(self, *inputs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self(*inputs))


def parameter_matched_expert_hidden(
    *,
    hidden: int,
    n_actions: int,
    reference_experts: int = 4,
    reference_expert_hidden: int | None = None,
) -> int:
    """Expert mid-width that matches a multi-expert MoE's trainable capacity.

    The backbone is shared across architectures, so only the expert heads and
    router are equalized.  A one-expert policy with this width therefore has
    approximately the same number of trainable parameters as the reference
    ``reference_experts``-expert MoE: any remaining performance gap is
    attributable to the mixture, not to capacity.
    """
    h = int(hidden)
    a = int(n_actions)
    e = int(reference_experts)
    ref_mid = int(reference_expert_hidden) if reference_expert_hidden else h
    r_mid = max(1, h // 2)

    def expert_params(n_exp: int, mid: int) -> int:
        return n_exp * (h * mid + mid + mid * a + a)

    def router_params(n_exp: int) -> int:
        return h * r_mid + r_mid + r_mid * n_exp + n_exp

    target = expert_params(e, ref_mid) + router_params(e)
    # One-expert router is smaller; put the residual capacity into the head.
    residual = target - router_params(1) - a
    denom = h + a + 1
    return max(h, int(round(residual / denom)))


class SharedBaseResidualMoEPolicy(nn.Module):
    """Legacy shared-base + top-2 residual MoE (architecture tag v2).

    Kept for loading poster-era checkpoints (``shared_base_top2_residual_moe_v2``)
    that store ``base_head`` and ``expert_scale`` instead of the current
    counterfactual-regime MoE.  New training uses ``BeliefConditionedMoEPolicy``.
    """

    def __init__(
        self,
        n_actions: int,
        config: PolicyConfig | None = None,
        *,
        n_experts: int = 4,
        top_k: int = 2,
        expert_hidden: int | None = None,
    ):
        super().__init__()
        self.config = config or PolicyConfig()
        self.n_actions = int(n_actions)
        self.n_experts = int(n_experts)
        self.top_k = min(int(top_k), self.n_experts)
        hidden = self.config.hidden
        self.expert_hidden = int(expert_hidden) if expert_hidden else hidden
        self.balance_coefficient = 0.001
        self.redundancy_coefficient = 0.005
        self.backbone = _PolicyBackbone(self.n_actions, self.config)
        self.base_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.n_actions),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, self.expert_hidden),
                    nn.SiLU(),
                    nn.Linear(self.expert_hidden, self.n_actions),
                )
                for _ in range(self.n_experts)
            ]
        )
        self.router = nn.Sequential(
            nn.Linear(hidden, max(1, hidden // 2)),
            nn.SiLU(),
            nn.Linear(max(1, hidden // 2), self.n_experts),
        )
        # Stored as a free parameter; forward applies sigmoid (matches v2).
        self.expert_scale = nn.Parameter(torch.tensor(0.0))

    def _components(self, *inputs: torch.Tensor):
        features = self.backbone(*inputs)
        base_logits = self.base_head(features)
        expert_values = torch.stack([head(features) for head in self.experts], dim=1)
        dense_weights = torch.softmax(self.router(features), dim=-1)
        top_values, top_indices = torch.topk(dense_weights, self.top_k, dim=-1)
        sparse_weights = torch.zeros_like(dense_weights).scatter(
            -1, top_indices, top_values
        )
        sparse_weights = sparse_weights / sparse_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        routed = (sparse_weights.unsqueeze(-1) * expert_values).sum(dim=1)
        scale = torch.sigmoid(self.expert_scale)
        logits = base_logits + scale * routed
        return logits, dense_weights, expert_values, scale

    def forward(
        self,
        action_indices: torch.Tensor,
        normalized_observations: torch.Tensor,
        history_mask: torch.Tensor,
        belief_summary: torch.Tensor,
        steps: torch.Tensor,
        particle_features: torch.Tensor,
        posterior_weights: torch.Tensor,
        feasible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = (
            action_indices,
            normalized_observations,
            history_mask,
            belief_summary,
            steps,
            particle_features,
            posterior_weights,
        )
        logits, _, _, _ = self._components(*inputs)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        logits = logits.clamp(-50.0, 50.0)
        if feasible_mask is not None:
            logits = logits.masked_fill(~feasible_mask, -1e9)
        return logits

    def distribution(self, *inputs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self(*inputs))

    def specialization_loss(
        self, *inputs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        _, weights, expert_values, scale = self._components(*inputs)
        mean_usage = weights.mean(dim=0)
        target = torch.full_like(mean_usage, 1.0 / self.n_experts)
        balance = self.n_experts * torch.sum((mean_usage - target).square())
        centered = expert_values - expert_values.mean(-1, keepdim=True)
        normalized = F.normalize(centered, dim=-1, eps=1e-8)
        similarity = torch.matmul(normalized, normalized.transpose(1, 2))
        off_diagonal = ~torch.eye(
            self.n_experts, dtype=torch.bool, device=similarity.device
        )
        if self.n_experts < 2:
            redundancy = similarity.sum() * 0.0
        else:
            redundancy = F.relu(similarity[:, off_diagonal]).square().mean()
        router_entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1).mean()
        loss = (
            float(self.balance_coefficient) * balance
            + float(self.redundancy_coefficient) * redundancy
        )
        return loss, {
            "router_entropy": float(router_entropy.detach()),
            "expert_redundancy": float(redundancy.detach()),
            "max_expert_usage": float(mean_usage.max().detach()),
            "expert_residual_scale": float(scale.detach()),
        }


class BeliefConditionedMoEPolicy(nn.Module):
    """Belief-conditioned shared-base + residual MoE for sequential design.

    Innovation (not a fusion of Fixed/Myopic/DAD):
      * shared base head = general probe ranking;
      * learned residual experts = regime-specific corrections;
      * belief-conditioned router (top-k) selects which experts apply;
      * belief-gated residual scale lets corrections grow when the posterior
        is informative / remaining horizon is non-trivial.

    External baselines remain comparison methods only. They are never expert
    heads inside this policy.
    """

    def __init__(
        self,
        n_actions: int,
        config: PolicyConfig | None = None,
        *,
        n_experts: int = 4,
        top_k: int = 2,
        expert_hidden: int | None = None,
        logit_scale_init: float = 3.0,
        balance_coefficient: float = 0.001,
        redundancy_coefficient: float = 0.01,
    ):
        super().__init__()
        self.config = config or PolicyConfig()
        self.n_actions = int(n_actions)
        self.n_experts = int(n_experts)
        self.top_k = min(int(top_k), self.n_experts)
        hidden = self.config.hidden
        # expert_hidden widens each expert head.  Defaults to ``hidden`` so the
        # standard 4-expert model is unchanged; a 1-expert parameter-matched
        # dense control sets it larger so its capacity matches the mixture.
        self.expert_hidden = int(expert_hidden) if expert_hidden else hidden
        self.balance_coefficient = float(balance_coefficient)
        self.redundancy_coefficient = float(redundancy_coefficient)
        self.backbone = _PolicyBackbone(self.n_actions, self.config)
        self.base_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.n_actions),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, self.expert_hidden),
                    nn.SiLU(),
                    nn.Linear(self.expert_hidden, self.n_actions),
                )
                for _ in range(self.n_experts)
            ]
        )
        # Start as the shared base: residual experts contribute nothing until
        # PPO / regime losses learn belief-dependent corrections.
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)
        self.router = nn.Sequential(
            nn.Linear(hidden, max(1, hidden // 2)),
            nn.SiLU(),
            nn.Linear(max(1, hidden // 2), self.n_experts),
        )
        # Belief-dependent residual strength: sigmoid gate on the shared
        # latent. Bias +2 ⇒ gate≈0.88 at init so residuals can learn early
        # without Fixed/Myopic cloning.
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden, max(1, hidden // 2)),
            nn.SiLU(),
            nn.Linear(max(1, hidden // 2), 1),
        )
        nn.init.zeros_(self.residual_gate[-1].weight)
        nn.init.constant_(self.residual_gate[-1].bias, 2.0)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))
        self.register_buffer(
            "regime_prototypes", torch.zeros(self.n_experts, self.n_actions)
        )
        self.register_buffer("prototypes_initialized", torch.tensor(False))
        self.prototype_temperature = 0.35
        self.prototype_momentum = 0.95
    @staticmethod
    def _fingerprint(values: torch.Tensor, feasible: torch.Tensor) -> torch.Tensor:
        """Normalize action values so regimes encode rankings, not scale.

        Variance is clamped *inside* ``sqrt`` so zero-centered expert heads
        (the zero-init residual experts at BC start) do not emit NaN through
        ``SqrtBackward``. Non-finite utilities are treated as infeasible.
        """
        finite = torch.isfinite(values)
        usable = feasible & finite
        safe_values = torch.where(usable, values, torch.zeros_like(values))
        count = usable.sum(-1, keepdim=True).clamp_min(1)
        mean = safe_values.sum(-1, keepdim=True) / count
        centered = torch.where(usable, values - mean, torch.zeros_like(values))
        var = centered.square().sum(-1, keepdim=True) / count
        scale = torch.sqrt(var.clamp_min(1e-8))
        return torch.where(usable, centered / scale, torch.zeros_like(values))

    @torch.no_grad()
    def reset_regime_prototypes(self) -> None:
        """Discard prototypes so the next counterfactual batch re-initializes them.

        Farthest-point initialization normally runs on the very first batch,
        when the policy is untrained and counterfactual estimates are at their
        noisiest.  Calling this after a warm-up phase re-anchors the regime
        prototypes on fingerprints produced by a meaningful policy.
        """
        self.prototypes_initialized.fill_(False)

    @torch.no_grad()
    def _initialize_prototypes(self, fingerprints: torch.Tensor) -> None:
        """Deterministic farthest-point initialization in decision-value space."""
        if bool(self.prototypes_initialized) or fingerprints.shape[0] == 0:
            return
        chosen = [int(torch.argmax(fingerprints.square().sum(-1)).item())]
        for _ in range(1, self.n_experts):
            centres = fingerprints[chosen]
            distance = torch.cdist(fingerprints, centres).square().amin(dim=1)
            chosen.append(int(torch.argmax(distance).item()))
        self.regime_prototypes.copy_(fingerprints[chosen])
        self.prototypes_initialized.fill_(True)

    @torch.no_grad()
    def _update_prototypes(
        self, fingerprints: torch.Tensor, responsibilities: torch.Tensor
    ) -> None:
        mass = responsibilities.sum(0)
        means = responsibilities.transpose(0, 1) @ fingerprints
        means = means / mass[:, None].clamp_min(1e-6)
        active = mass > 1e-4
        self.regime_prototypes[active].mul_(self.prototype_momentum).add_(
            means[active], alpha=1.0 - self.prototype_momentum
        )

    def _components(self, *inputs: torch.Tensor):
        features = self.backbone(*inputs)
        base_logits = self.base_head(features)
        expert_values = torch.stack([head(features) for head in self.experts], dim=1)
        dense_weights = torch.softmax(self.router(features), dim=-1)
        top_values, top_indices = torch.topk(dense_weights, self.top_k, dim=-1)
        sparse_weights = torch.zeros_like(dense_weights).scatter(-1, top_indices, top_values)
        sparse_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        routed_value = (sparse_weights.unsqueeze(-1) * expert_values).sum(dim=1)
        global_scale = F.softplus(self.logit_scale).clamp(max=20.0)
        # Per-state belief gate ∈ (0,1): residual experts matter only when the
        # latent belief state requests a correction (not a Fixed/Myopic clone).
        belief_gate = torch.sigmoid(self.residual_gate(features))
        scale = global_scale * belief_gate
        logits = base_logits + scale * routed_value
        return logits, dense_weights, expert_values, scale, base_logits

    def forward(
        self,
        action_indices: torch.Tensor,
        normalized_observations: torch.Tensor,
        history_mask: torch.Tensor,
        belief_summary: torch.Tensor,
        steps: torch.Tensor,
        particle_features: torch.Tensor,
        posterior_weights: torch.Tensor,
        feasible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = (
            action_indices, normalized_observations, history_mask, belief_summary,
            steps, particle_features, posterior_weights,
        )
        logits, _, _, _, _ = self._components(*inputs)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        logits = logits.clamp(-50.0, 50.0)
        if feasible_mask is not None:
            logits = logits.masked_fill(~feasible_mask, -1e9)
        return logits

    def base_logits(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Shared-base action scores (no residual expert correction)."""
        return self._components(*inputs)[4]

    def distribution(self, *inputs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self(*inputs))

    def specialization_loss(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Balance routing while discouraging functionally identical experts."""
        _, weights, expert_values, scale = self._components(*inputs)[:4]
        mean_usage = weights.mean(dim=0)
        target = torch.full_like(mean_usage, 1.0 / self.n_experts)
        balance = self.n_experts * torch.sum((mean_usage - target).square())
        centered = expert_values - expert_values.mean(-1, keepdim=True)
        # eps avoids NaN from zero-init residual experts (norm 0).
        normalized = F.normalize(centered, dim=-1, eps=1e-8)
        similarity = torch.matmul(normalized, normalized.transpose(1, 2))
        off_diagonal = ~torch.eye(self.n_experts, dtype=torch.bool, device=similarity.device)
        # Penalize positively duplicated experts. Anti-correlated corrections
        # are useful and should not be penalized as redundancy.  A single-expert
        # dense control has no off-diagonal pairs, so redundancy is zero.
        if self.n_experts < 2:
            redundancy = similarity.sum() * 0.0
        else:
            redundancy = F.relu(similarity[:, off_diagonal]).square().mean()
        router_entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1).mean()
        # Weak load balancing prevents dead experts without forcing a uniform
        # router when one posterior regime legitimately dominates.
        loss = (
            float(self.balance_coefficient) * balance
            + float(self.redundancy_coefficient) * redundancy
        )
        global_scale = float(F.softplus(self.logit_scale).clamp(max=20.0).detach())
        gate_mean = float((scale.detach().reshape(-1) / max(global_scale, 1e-6)).mean())
        stats = {
            "router_entropy": float(router_entropy.detach()),
            "expert_redundancy": float(redundancy.detach()),
            "max_expert_usage": float(mean_usage.max().detach()),
            "expert_logit_scale": global_scale,
            "belief_residual_gate_mean": gate_mean,
            "moe_balance_coefficient": float(self.balance_coefficient),
            "moe_redundancy_coefficient": float(self.redundancy_coefficient),
        }
        return loss, stats

    def counterfactual_loss(
        self,
        *inputs: torch.Tensor,
        target_utility: torch.Tensor,
        feasible_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Supervise experts/router from all-action counterfactual utilities."""
        fused, router_weights, expert_values, _, _ = self._components(*inputs)
        target = self._fingerprint(target_utility.detach(), feasible_mask)
        self._initialize_prototypes(target)
        distance = torch.cdist(target, self.regime_prototypes).square()
        responsibilities = torch.softmax(
            -distance / self.prototype_temperature, dim=-1
        )
        if self.training:
            self._update_prototypes(target, responsibilities)

        predicted = torch.stack(
            [self._fingerprint(expert_values[:, e], feasible_mask)
             for e in range(self.n_experts)], dim=1
        )
        per_expert = F.smooth_l1_loss(
            predicted, target[:, None, :].expand_as(predicted), reduction="none"
        ).sum(-1) / feasible_mask.sum(-1, keepdim=True).clamp_min(1)
        value_loss = (responsibilities * per_expert).sum(-1).mean()
        router_loss = -(responsibilities * router_weights.clamp_min(1e-8).log()).sum(-1).mean()
        masked_fused = fused.masked_fill(~feasible_mask, -1e9)
        best_action = target_utility.masked_fill(~feasible_mask, -1e9).argmax(-1)
        ranking_loss = F.cross_entropy(masked_fused, best_action)
        loss = value_loss + router_loss + 0.25 * ranking_loss
        assignment = responsibilities.argmax(-1)
        used = torch.bincount(assignment, minlength=self.n_experts).float()
        return loss, {
            "cf_value_loss": float(value_loss.detach()),
            "cf_router_loss": float(router_loss.detach()),
            "cf_ranking_loss": float(ranking_loss.detach()),
            "cf_active_regimes": float((used > 0).sum().detach()),
        }

    def branching_loss(
        self,
        *inputs: torch.Tensor,
        target_utility: torch.Tensor,
        feasible_mask: torch.Tensor,
        similarity_threshold: float = 0.5,
        margin: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Penalize observation-invariant behaviour only where it costs utility.

        For pairs of same-step states whose counterfactual action-value
        rankings disagree (fingerprint cosine similarity below the threshold),
        an adaptive policy must produce different action distributions; such
        pairs pay a hinge penalty on total-variation distance.  Pairs whose
        counterfactual rankings agree are never penalized, so the regularizer
        cannot corrupt the objective when several beliefs legitimately share
        one optimal action.  Step-0 pairs are excluded because their policy
        inputs are identical and no history-conditioned policy can separate
        them.
        """
        logits, _, _, _, _ = self._components(*inputs)
        logits = logits.clamp(-50.0, 50.0).masked_fill(~feasible_mask, -1e9)
        probs = torch.softmax(logits, dim=-1)
        fingerprints = self._fingerprint(target_utility.detach(), feasible_mask)
        unit = F.normalize(fingerprints, dim=-1, eps=1e-8)
        similarity = unit @ unit.transpose(0, 1)
        steps = inputs[4].reshape(-1)
        same_step = steps[:, None] == steps[None, :]
        informative = (steps > 0)[:, None] & (steps > 0)[None, :]
        n = probs.shape[0]
        off_diagonal = ~torch.eye(n, dtype=torch.bool, device=probs.device)
        disagree = (
            (similarity < similarity_threshold)
            & same_step
            & informative
            & off_diagonal
        )
        if not bool(disagree.any()):
            zero = logits.sum() * 0.0
            return zero, {"branching_pairs": 0.0, "branching_loss": 0.0}
        total_variation = 0.5 * (
            probs[:, None, :] - probs[None, :, :]
        ).abs().sum(-1)
        penalty = F.relu(margin - total_variation)[disagree].mean()
        return penalty, {
            "branching_pairs": float(disagree.sum().detach()) / 2.0,
            "branching_loss": float(penalty.detach()),
        }


class StateValueCritic(nn.Module):
    """Value critic mirroring the policy backbone."""

    def __init__(self, n_actions: int, config: PolicyConfig | None = None):
        super().__init__()
        self.config = config or PolicyConfig()
        self.backbone = _PolicyBackbone(n_actions, self.config)
        self.head = nn.Sequential(
            nn.Linear(self.config.hidden, self.config.hidden),
            nn.SiLU(),
            nn.Linear(self.config.hidden, 1),
        )

    def forward(
        self,
        action_indices: torch.Tensor,
        normalized_observations: torch.Tensor,
        history_mask: torch.Tensor,
        belief_summary: torch.Tensor,
        steps: torch.Tensor,
        particle_features: torch.Tensor,
        posterior_weights: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(
            action_indices,
            normalized_observations,
            history_mask,
            belief_summary,
            steps,
            particle_features,
            posterior_weights,
        )
        return self.head(features).squeeze(-1)


# Production aliases (scientific method names only).
# Note: REINFORCE DAD uses ``src.policies.dad.DADPolicy``; PPO DAD/RL-sBOED use these.
DADPolicy = AdaptiveExperimentPolicy
RLSBOEDPolicy = AdaptiveExperimentPolicy


class PolicyTrainer:
    """Marker namespace for the shared PPO trainer used by DAD and RL-sBOED."""

    pass
