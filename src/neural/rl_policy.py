"""Production policies for DAD and RL-sBOED (shared history + belief backbone).

Manuscript methods: DAD, RL-sBOED.
This module is self-contained. Experimental Stage-2 encoder screens live under
``src.neural.legacy``.
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


class _HistoryEncoder(nn.Module):
    """Encode the ordered probe-observation history."""

    def __init__(self, n_actions: int, hidden: int = 128, max_steps: int = 3):
        super().__init__()
        self.n_actions = n_actions
        self.hidden = hidden
        self.position = nn.Embedding(max_steps, 16)
        self.pair = nn.Sequential(
            nn.Linear(n_actions + 1 + 16, hidden),
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
        pair_input = torch.cat(
            [
                one_hot,
                normalized_observations.clamp(-8.0, 8.0).unsqueeze(-1),
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


class _PolicyBackbone(nn.Module):
    def __init__(self, n_actions: int, config: PolicyConfig):
        super().__init__()
        hidden = config.hidden
        self.history_encoder = _HistoryEncoder(
            n_actions, hidden, config.max_steps
        )
        self.belief_encoder = _BeliefEncoder(config.summary_dim, hidden)
        self.step_encoder = nn.Embedding(config.max_steps + 1, 32)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden + 32, hidden),
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
        del particle_features, posterior_weights  # reserved for API compatibility
        history = self.history_encoder(
            action_indices, normalized_observations, history_mask
        )
        belief = self.belief_encoder(belief_summary)
        return self.fusion(
            torch.cat([history, belief, self.step_encoder(steps.long())], dim=-1)
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
# Note: REINFORCE DAD uses ``src.neural.policy.DADPolicy``; PPO DAD/RL-sBOED use these.
DADPolicy = AdaptiveExperimentPolicy
RLSBOEDPolicy = AdaptiveExperimentPolicy


class PolicyTrainer:
    """Marker namespace for the shared PPO trainer used by DAD and RL-sBOED."""

    pass
