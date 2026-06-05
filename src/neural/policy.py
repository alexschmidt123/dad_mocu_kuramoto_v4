"""
History-dependent DAD policy for sequential probe selection.

Simple MLP over encoded history (design one-hot + normalized observation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoryEncoder(nn.Module):
    """Encode h_t = {(ξ_i, y_i)}_{i=1}^t into a fixed-size vector."""

    def __init__(self, n_actions: int, hidden: int = 128, max_steps: int = 3):
        super().__init__()
        self.n_actions = n_actions
        self.max_steps = max_steps
        self.step_mlp = nn.Sequential(
            nn.Linear(n_actions + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pool = nn.Linear(hidden, hidden)

    def forward(self, action_indices: torch.Tensor, observations: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_indices: (B, T) long, -1 for padding
            observations: (B, T) float, 0 for padding
            mask: (B, T) float, 1 for valid steps
        """
        B, T = action_indices.shape
        one_hot = F.one_hot(action_indices.clamp(min=0), num_classes=self.n_actions).float()
        obs = observations.unsqueeze(-1)
        step_in = torch.cat([one_hot, obs], dim=-1)
        h = self.step_mlp(step_in)
        h = h * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = h.sum(dim=1) / denom
        return self.pool(pooled)


class DADPolicy(nn.Module):
    """π_φ(h_{t-1}) → logits over feasible actions."""

    def __init__(self, n_actions: int, hidden: int = 128, max_steps: int = 3):
        super().__init__()
        self.n_actions = n_actions
        self.encoder = HistoryEncoder(n_actions, hidden, max_steps)
        self.head = nn.Linear(hidden, n_actions)

    def forward(
        self,
        action_indices: torch.Tensor,
        observations: torch.Tensor,
        mask: torch.Tensor,
        feasible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns action logits (B, n_actions). Infeasible actions masked to -inf.
        """
        h = self.encoder(action_indices, observations, mask)
        logits = self.head(h)
        if feasible_mask is not None:
            logits = logits.masked_fill(~feasible_mask, float("-inf"))
        return logits

    def select_action(
        self,
        action_indices: torch.Tensor,
        observations: torch.Tensor,
        mask: torch.Tensor,
        feasible_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(action_indices, observations, mask, feasible_mask)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)
        return action, log_prob
