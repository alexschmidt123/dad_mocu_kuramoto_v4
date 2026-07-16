"""
History-dependent DAD policy for sequential probe selection.

Per-step MLP embeddings + single-head attention pooling (replacing mean-pool).
Observations are lightly squashed before the MLP to avoid overflow on large ROCOF values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoryEncoder(nn.Module):
    """Encode h_t = {(ξ_i, y_i)}_{i=1}^t into a fixed-size vector via attention pooling."""

    def __init__(self, n_actions: int, hidden: int = 128, max_steps: int = 3):
        super().__init__()
        self.n_actions = n_actions
        self.max_steps = max_steps
        self.hidden = hidden
        self.step_mlp = nn.Sequential(
            nn.Linear(n_actions + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Learned query for attention pooling over past step embeddings.
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.attn_scale = hidden**-0.5
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, action_indices: torch.Tensor, observations: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_indices: (B, T) long, -1 for padding
            observations: (B, T) float, 0 for padding
            mask: (B, T) float, 1 for valid steps
        """
        B, T = action_indices.shape
        if T == 0:
            return torch.zeros(B, self.hidden, device=action_indices.device, dtype=observations.dtype)

        one_hot = F.one_hot(action_indices.clamp(min=0), num_classes=self.n_actions).float()
        # Squash ROCOF magnitudes so MLP / attention scores stay finite.
        obs = torch.tanh(observations / 10.0).unsqueeze(-1)
        step_in = torch.cat([one_hot, obs], dim=-1)
        h = self.step_mlp(step_in)  # (B, T, H)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        # Single-head attention: learned query attends over past steps.
        q = self.pool_query.expand(B, -1, -1)
        scores = torch.matmul(q, h.transpose(1, 2)) * self.attn_scale  # (B, 1, T)
        # Prefer a large negative over -inf to avoid softmax all-masked NaNs.
        scores = scores.masked_fill(mask.unsqueeze(1) <= 0, -1e9)
        all_masked = mask.sum(dim=1) <= 0
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        pooled = torch.matmul(attn, h).squeeze(1)  # (B, H)
        pooled = pooled.masked_fill(all_masked.unsqueeze(-1), 0.0)
        return self.out_proj(pooled)


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
        Returns action logits (B, n_actions). Infeasible actions masked with a large negative.
        """
        h = self.encoder(action_indices, observations, mask)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        logits = self.head(h)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        logits = logits.clamp(-50.0, 50.0)
        if feasible_mask is not None:
            # Large negative (not -inf) keeps softmax numerically safe for multinomial.
            logits = logits.masked_fill(~feasible_mask, -1e9)
        return logits

    def select_action(
        self,
        action_indices: torch.Tensor,
        observations: torch.Tensor,
        mask: torch.Tensor,
        feasible_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns ``(action, log_prob, entropy)`` where entropy is over feasible actions only.
        """
        logits = self.forward(action_indices, observations, mask, feasible_mask)
        # Softmax over feasible set; replace any residual NaNs and renormalize.
        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs * feasible_mask.float()
        denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        probs = probs / denom

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)

        log_probs = torch.log(probs.clamp(min=1e-12))
        log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0)
        return action, log_prob, entropy
