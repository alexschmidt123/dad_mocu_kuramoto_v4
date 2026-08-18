"""MoE-sBOED policy networks.

Production method: ``BeliefConditionedMoEPolicy`` (train with ``--method moe_sboed``).
``SharedBaseResidualMoEPolicy`` is kept only to load older checkpoints.
Shared history/belief backbone lives in ``src.policies.rl_sboed``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.policies.rl_sboed import PolicyConfig, _PolicyBackbone


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
    """Belief-conditioned mixture of expert probe-rankers.

    This is a real MoE, not a teacher clone and not a residual on myopic/DAD:
      * E expert heads each emit a full action ranking;
      * a belief-conditioned router picks a sparse top-k mixture;
      * fused logits = Σ_e ŵ_e Expert_e(h)  (no added two-step/myopic base).

    Expert 0 is a generalist warm-started by BC. Experts 1..E-1 are specialists.
    External baselines (Fixed / Myopic / DAD / RL) are never expert heads.
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
        routing_information_coefficient: float = 0.0,
        redundancy_coefficient: float = 0.01,
    ):
        super().__init__()
        self.config = config or PolicyConfig()
        self.n_actions = int(n_actions)
        self.n_experts = int(n_experts)
        self.top_k = min(int(top_k), self.n_experts)
        hidden = self.config.hidden
        self.expert_hidden = int(expert_hidden) if expert_hidden else hidden
        self.balance_coefficient = float(balance_coefficient)
        self.routing_information_coefficient = float(
            routing_information_coefficient
        )
        self.redundancy_coefficient = float(redundancy_coefficient)
        self.backbone = _PolicyBackbone(self.n_actions, self.config)
        # Kept for older checkpoints; fused logits do not add this head.
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
        # Unused in the fused path; kept so older residual_gate checkpoints load.
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
        self.detach_base = False

    def freeze_base_head(self) -> None:
        """Freeze generalist expert 0. Specialists and the router stay live."""
        for parameter in self.experts[0].parameters():
            parameter.requires_grad = False
        for parameter in self.base_head.parameters():
            parameter.requires_grad = False
        self.detach_base = True

    def reinitialize_residual_experts(self, std: float = 0.08) -> None:
        """Re-init specialist heads only; keep BC generalist (expert 0)."""
        for expert in self.experts[1:]:
            nn.init.normal_(expert[-1].weight, mean=0.0, std=float(std))
            nn.init.zeros_(expert[-1].bias)

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
        expert_values = torch.stack([head(features) for head in self.experts], dim=1)
        generalist = expert_values[:, 0]
        if self.detach_base:
            generalist = generalist.detach()
            expert_values = expert_values.clone()
            expert_values[:, 0] = generalist
        dense_weights = torch.softmax(self.router(features), dim=-1)
        steps = inputs[4].reshape(-1)
        at_prior = steps <= 0
        if bool(at_prior.any()):
            # Identical prior: specialists must not override the generalist
            # first probe (that collapse is what wrecked IEEE9 EIG).
            prior_w = torch.zeros_like(dense_weights)
            prior_w[:, 0] = 1.0
            dense_weights = torch.where(at_prior[:, None], prior_w, dense_weights)
        top_values, top_indices = torch.topk(dense_weights, self.top_k, dim=-1)
        sparse_weights = torch.zeros_like(dense_weights).scatter(-1, top_indices, top_values)
        sparse_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        routed_value = (sparse_weights.unsqueeze(-1) * expert_values).sum(dim=1)
        temperature = F.softplus(self.logit_scale).clamp(max=20.0)
        logits = temperature * routed_value
        return logits, dense_weights, expert_values, temperature, generalist

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
        """Generalist expert 0 only. Used for BC; not the fused MoE policy."""
        features = self.backbone(*inputs)
        return self.experts[0](features)

    def distribution(self, *inputs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self(*inputs))

    def specialization_loss(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Balance routing while discouraging functionally identical experts."""
        logits, weights, expert_values, scale, base = self._components(*inputs)
        steps = inputs[4].reshape(-1)
        post = steps > 0
        usage_w = weights[post] if bool(post.any()) else weights
        mean_usage = usage_w.mean(dim=0)
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
        conditional_entropy = -(
            weights * weights.clamp_min(1e-8).log()
        ).sum(-1).mean()
        marginal_entropy = -(
            mean_usage * mean_usage.clamp_min(1e-8).log()
        ).sum()
        # H(E|belief)-H(E) is the negative routing mutual information.  Its
        # minimum is attained by confident, belief-dependent assignments that
        # use several experts; both one-expert collapse and uniform routing
        # have zero mutual information.  The separate load term supplies a
        # gradient when a saturated router has already collapsed.
        negative_routing_information = conditional_entropy - marginal_entropy
        loss = (
            float(self.balance_coefficient) * balance
            + float(self.routing_information_coefficient)
            * negative_routing_information
            + float(self.redundancy_coefficient) * redundancy
        )
        global_scale = float(F.softplus(self.logit_scale).clamp(max=20.0).detach())
        gate_mean = float((scale.detach().reshape(-1) / max(global_scale, 1e-6)).mean())
        fused_top = logits.argmax(-1)
        base_top = base.argmax(-1)
        steps = inputs[4].reshape(-1)
        post = steps > 0
        if bool(post.any()):
            disagree = float((fused_top[post] != base_top[post]).float().mean().detach())
        else:
            disagree = 0.0
        stats = {
            "router_entropy": float(conditional_entropy.detach()),
            "router_marginal_entropy": float(marginal_entropy.detach()),
            "router_mutual_information": float(
                (-negative_routing_information).detach()
            ),
            "expert_redundancy": float(redundancy.detach()),
            "max_expert_usage": float(mean_usage.max().detach()),
            "expert_logit_scale": global_scale,
            "belief_residual_gate_mean": gate_mean,
            "fused_vs_base_argmax_disagree": disagree,
            "moe_balance_coefficient": float(self.balance_coefficient),
            "moe_routing_information_coefficient": float(
                self.routing_information_coefficient
            ),
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
        _fused, router_weights, expert_values, _, _ = self._components(*inputs)
        target = self._fingerprint(target_utility.detach(), feasible_mask)
        self._initialize_prototypes(target)
        distance = torch.cdist(target, self.regime_prototypes).square()
        # Balanced soft clustering prevents the decision-regime targets from
        # silently assigning every posterior state to one prototype.  This is
        # an entropic optimal-transport assignment over the current batch,
        # not supervision from any external BOED method.
        assignment_logits = -distance / self.prototype_temperature
        responsibilities = torch.exp(
            assignment_logits - assignment_logits.max()
        ).transpose(0, 1)
        responsibilities = responsibilities.clamp_min(1e-8)
        responsibilities = responsibilities / responsibilities.sum()
        for _ in range(3):
            responsibilities = responsibilities / responsibilities.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            responsibilities = responsibilities / self.n_experts
            responsibilities = responsibilities / responsibilities.sum(
                dim=0, keepdim=True
            ).clamp_min(1e-8)
            responsibilities = responsibilities / max(
                int(assignment_logits.shape[0]), 1
            )
        responsibilities = (
            responsibilities
            * max(int(assignment_logits.shape[0]), 1)
        ).transpose(0, 1).detach()
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
        # Do not CE-train fused logits toward a teacher argmax: that clones
        # myopic / two-step / Fixed into the mixture. Experts cluster on
        # fingerprints; the fused ranking is left to PPO + residual losses.
        loss = value_loss + router_loss
        assignment = responsibilities.argmax(-1)
        used = torch.bincount(assignment, minlength=self.n_experts).float()
        return loss, {
            "cf_value_loss": float(value_loss.detach()),
            "cf_router_loss": float(router_loss.detach()),
            "cf_ranking_loss": 0.0,
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

    def belief_branching_loss(
        self,
        *inputs: torch.Tensor,
        feasible_mask: torch.Tensor,
        similarity_threshold: float = 0.85,
        margin: float = 0.35,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Force distinct π on distinct posteriors, without a teacher ranking.

        Pairs at the same step>0 whose belief summaries disagree must have
        different action distributions. This is not myopic/Fixed/DAD cloning:
        the signal is the posterior itself.
        """
        logits, _, _, _, _ = self._components(*inputs)
        logits = logits.clamp(-50.0, 50.0).masked_fill(~feasible_mask, -1e9)
        probs = torch.softmax(logits, dim=-1)
        belief = inputs[3].reshape(inputs[3].shape[0], -1)
        unit = F.normalize(belief, dim=-1, eps=1e-8)
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
            return zero, {
                "belief_branching_pairs": 0.0,
                "belief_branching_loss": 0.0,
            }
        total_variation = 0.5 * (
            probs[:, None, :] - probs[None, :, :]
        ).abs().sum(-1)
        penalty = F.relu(margin - total_variation)[disagree].mean()
        return penalty, {
            "belief_branching_pairs": float(disagree.sum().detach()) / 2.0,
            "belief_branching_loss": float(penalty.detach()),
        }

    def low_ess_residual_loss(
        self,
        *inputs: torch.Tensor,
        feasible_mask: torch.Tensor,
        ess_threshold: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Residual experts must move π away from the frozen base when ESS is low."""
        logits, _, _, scale, base = self._components(*inputs)
        belief = inputs[3]
        ess = belief.reshape(belief.shape[0], -1)[:, 1]
        thr = float(ess_threshold)
        if thr <= 0.0:
            zero = logits.sum() * 0.0
            return zero, {
                "low_ess_residual_kl": 0.0,
                "low_ess_gate_mean": 0.0,
                "expert_logit_scale": float(scale.detach().reshape(-1).mean()),
            }
        gate = ((thr - ess) / max(thr, 1e-6)).clamp(0.0, 1.0)
        if float(gate.sum()) <= 1e-8:
            zero = logits.sum() * 0.0
            return zero, {
                "low_ess_residual_kl": 0.0,
                "low_ess_gate_mean": 0.0,
                "expert_logit_scale": float(scale.detach().reshape(-1).mean()),
            }
        masked_final = logits.clamp(-50.0, 50.0).masked_fill(~feasible_mask, -1e9)
        masked_base = base.clamp(-50.0, 50.0).masked_fill(~feasible_mask, -1e9)
        log_p = F.log_softmax(masked_final, dim=-1)
        log_q = F.log_softmax(masked_base, dim=-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
        w_sum = gate.sum().clamp_min(1e-8)
        loss = -(gate * kl).sum() / w_sum
        return loss, {
            "low_ess_residual_kl": float((gate * kl).sum().detach() / w_sum.detach()),
            "low_ess_gate_mean": float(gate.mean().detach()),
            "expert_logit_scale": float(scale.detach().reshape(-1).mean()),
        }

    def residual_scale_floor_loss(
        self,
        *,
        target: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Hinge penalty if softplus(logit_scale) falls below ``target``."""
        scale = F.softplus(self.logit_scale).clamp(max=20.0)
        gap = F.relu(float(target) - scale)
        loss = gap.square()
        return loss, {
            "residual_scale": float(scale.detach()),
            "residual_scale_target": float(target),
            "residual_scale_gap": float(gap.detach()),
        }

    def greedy_leave_base_loss(
        self,
        *inputs: torch.Tensor,
        feasible_mask: torch.Tensor,
        margin: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Force greedy fused argmax off the base action after step 0.

        KL-to-base can rise while argmax stays put; eval is greedy, so the
        residual must beat the base action by a margin on later steps.
        """
        logits, _, _, _, base = self._components(*inputs)
        steps = inputs[4].reshape(-1)
        post = (steps > 0).to(dtype=logits.dtype)
        if float(post.sum()) <= 1e-8:
            zero = logits.sum() * 0.0
            return zero, {
                "leave_base_gap": 0.0,
                "leave_base_disagree": 0.0,
            }
        masked_fused = logits.clamp(-50.0, 50.0).masked_fill(~feasible_mask, -1e9)
        masked_base = base.clamp(-50.0, 50.0).masked_fill(~feasible_mask, -1e9)
        base_top = masked_base.argmax(-1)
        base_score = masked_fused.gather(1, base_top.unsqueeze(-1)).squeeze(-1)
        alt = masked_fused.scatter(
            1, base_top.unsqueeze(-1), torch.full_like(base_score, -1e9).unsqueeze(-1)
        )
        alt_best = alt.max(dim=-1).values
        gap = F.relu(base_score + float(margin) - alt_best)
        w = post.sum().clamp_min(1.0)
        loss = (post * gap).sum() / w
        disagree = ((alt_best > base_score) & (steps > 0)).to(dtype=logits.dtype)
        return loss, {
            "leave_base_gap": float((post * gap).sum().detach() / w.detach()),
            "leave_base_disagree": float(
                disagree.sum().detach() / post.sum().detach()
            ),
        }

