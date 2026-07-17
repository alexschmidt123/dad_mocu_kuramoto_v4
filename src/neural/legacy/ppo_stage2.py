"""Stage-2 DAD policy used only for diagnostic action comparison."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.neural.legacy.advanced_dad import NormalizedHistoryEncoder


@dataclass(frozen=True)
class Stage2Architecture:
    belief_encoder: str = "B0"
    history_encoder: str = "H0"
    hidden: int = 128
    particle_embedding: int = 96
    attention_heads: int = 4
    history_layers: int = 2


class TokenHistoryEncoder(nn.Module):
    def __init__(
        self,
        n_actions: int,
        hidden: int,
        max_steps: int,
        *,
        mode: str,
        heads: int,
        layers: int,
    ):
        super().__init__()
        self.hidden = hidden
        self.mode = mode
        self.action = nn.Embedding(n_actions, 32)
        self.observation = nn.Sequential(nn.Linear(1, 16), nn.SiLU())
        self.position = nn.Embedding(max_steps, 16)
        self.token = nn.Sequential(
            nn.Linear(64, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        if mode == "H1":
            self.attention = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        hidden, heads, batch_first=True, dropout=0.0
                    )
                    for _ in range(layers)
                ]
            )
            self.feed_forward = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden, 2 * hidden),
                        nn.SiLU(),
                        nn.Linear(2 * hidden, hidden),
                    )
                    for _ in range(layers)
                ]
            )
            self.norm1 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.norm2 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        elif mode == "H2":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=2 * hidden,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=layers, enable_nested_tensor=False
            )
        else:
            raise ValueError(f"unknown token history mode {mode}")
        self.query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.output = nn.LayerNorm(hidden)

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
        positions = self.position(
            torch.arange(length, device=action_indices.device)
        ).unsqueeze(0).expand(batch, -1, -1)
        tokens = self.token(
            torch.cat(
                [
                    self.action(action_indices.clamp(min=0)),
                    self.observation(
                        normalized_observations.clamp(-8.0, 8.0).unsqueeze(-1)
                    ),
                    positions,
                ],
                dim=-1,
            )
        )
        padding = history_mask <= 0
        all_masked = history_mask.sum(dim=1) <= 0
        if torch.any(all_masked):
            padding = padding.clone()
            tokens = tokens.clone()
            padding[all_masked, 0] = False
            tokens[all_masked, 0] = 0.0
        if self.mode == "H1":
            for attention, feed_forward, norm1, norm2 in zip(
                self.attention, self.feed_forward, self.norm1, self.norm2
            ):
                update, _ = attention(
                    tokens, tokens, tokens, key_padding_mask=padding
                )
                tokens = norm1(tokens + update)
                tokens = norm2(tokens + feed_forward(tokens))
        else:
            tokens = self.transformer(tokens, src_key_padding_mask=padding)
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        scores = torch.matmul(
            self.query.expand(batch, -1, -1), tokens.transpose(1, 2)
        ) * (self.hidden**-0.5)
        scores = scores.masked_fill(padding.unsqueeze(1), -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        pooled = torch.matmul(attention_weights, tokens).squeeze(1)
        pooled = pooled.masked_fill(all_masked.unsqueeze(-1), 0.0)
        return self.output(pooled)


class SummaryBeliefEncoder(nn.Module):
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


class ParticleBeliefEncoder(nn.Module):
    def __init__(self, particle_dim: int, hidden: int, embedding: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(particle_dim, embedding),
            nn.LayerNorm(embedding),
            nn.SiLU(),
            nn.Linear(embedding, embedding),
            nn.SiLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(embedding, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

    def forward(
        self, particles: torch.Tensor, posterior_weights: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.phi(particles)
        pooled = torch.sum(
            encoded * posterior_weights.clamp_min(0.0).unsqueeze(-1), dim=1
        )
        return self.rho(pooled)


class Stage2Backbone(nn.Module):
    def __init__(
        self,
        n_actions: int,
        summary_dim: int,
        particle_dim: int,
        architecture: Stage2Architecture,
        max_steps: int = 3,
    ):
        super().__init__()
        self.architecture = architecture
        hidden = architecture.hidden
        if architecture.history_encoder == "H0":
            self.history_encoder = NormalizedHistoryEncoder(
                n_actions, hidden, max_steps
            )
        else:
            self.history_encoder = TokenHistoryEncoder(
                n_actions,
                hidden,
                max_steps,
                mode=architecture.history_encoder,
                heads=architecture.attention_heads,
                layers=architecture.history_layers,
            )
        if architecture.belief_encoder in {"B0", "B1"}:
            self.belief_encoder = SummaryBeliefEncoder(summary_dim, hidden)
        elif architecture.belief_encoder == "B2":
            self.belief_encoder = ParticleBeliefEncoder(
                particle_dim, hidden, architecture.particle_embedding
            )
        else:
            raise ValueError(f"unknown belief encoder {architecture.belief_encoder}")
        self.step_encoder = nn.Embedding(max_steps + 1, 32)
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
        history = self.history_encoder(
            action_indices, normalized_observations, history_mask
        )
        if self.architecture.belief_encoder == "B2":
            belief = self.belief_encoder(particle_features, posterior_weights)
        else:
            belief = self.belief_encoder(belief_summary)
        return self.fusion(
            torch.cat([history, belief, self.step_encoder(steps.long())], dim=-1)
        )


class Stage2Policy(nn.Module):
    def __init__(
        self,
        n_actions: int,
        summary_dim: int,
        particle_dim: int,
        architecture: Stage2Architecture,
        max_steps: int = 3,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.architecture = architecture
        self.backbone = Stage2Backbone(
            n_actions, summary_dim, particle_dim, architecture, max_steps
        )
        self.action_head = nn.Linear(architecture.hidden, n_actions)

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
