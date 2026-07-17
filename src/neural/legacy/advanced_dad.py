"""Minimal history encoder reused by Stage-2 DAD (H0 branch)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedHistoryEncoder(nn.Module):
    """Encode the complete ordered history using z-scored observations."""

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
            [one_hot, normalized_observations.clamp(-8.0, 8.0).unsqueeze(-1), positions],
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
