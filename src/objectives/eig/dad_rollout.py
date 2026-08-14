"""On-policy / deterministic DADPolicy rollouts used by EIG train and eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.config import SBOEDConfig
from src.banks.tables import lookup_action_y
from src.policies.dad import DADPolicy


def feasible_mask(
    used: set[int], n_actions: int, device: torch.device
) -> torch.Tensor:
    m = torch.ones(n_actions, dtype=torch.bool, device=device)
    for i in used:
        m[i] = False
    return m


def training_horizon(meta: dict, cfg: SBOEDConfig) -> int:
    if meta.get("experiment_step_number") is not None:
        return int(meta["experiment_step_number"])
    if meta.get("training_horizon") is not None:
        return int(meta["training_horizon"])
    return int(cfg.step_number)


def policy_rollout(
    policy: DADPolicy,
    device: torch.device,
    sys: dict[str, Any],
    step_number: int,
    n_actions: int,
) -> tuple[list[int], list[float], torch.Tensor, torch.Tensor]:
    """On-policy rollout over the complete history {(ξ_i, y_i)}."""
    used: set[int] = set()
    seq: list[int] = []
    y_list: list[float] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    act_h: list[int] = []
    obs_h: list[float] = []

    for _ in range(step_number):
        if not act_h:
            act_t = torch.zeros(1, 0, dtype=torch.long, device=device)
            obs_t = torch.zeros(1, 0, device=device)
            mask_t = torch.zeros(1, 0, device=device)
        else:
            act_t = torch.tensor([act_h], dtype=torch.long, device=device)
            obs_t = torch.tensor([obs_h], dtype=torch.float32, device=device)
            mask_t = torch.ones(1, len(act_h), device=device)
        feas = feasible_mask(used, n_actions, device).unsqueeze(0)
        a, log_p, ent = policy.select_action(
            act_t, obs_t, mask_t, feas, deterministic=False
        )
        a_idx = int(a.item())
        seq.append(a_idx)
        y = lookup_action_y(sys, a_idx)
        y_list.append(float(y))
        log_probs.append(log_p.squeeze(0))
        entropies.append(ent.squeeze(0))
        act_h.append(a_idx)
        obs_h.append(float(y))
        used.add(a_idx)

    return seq, y_list, torch.stack(log_probs), torch.stack(entropies)


# Compatibility aliases used by older call sites.
_feasible_mask = feasible_mask
_policy_rollout = policy_rollout
_training_horizon = training_horizon


def rollout_dad(
    cfg: SBOEDConfig,
    test_systems: list[dict],
    policy_path: Path,
    meta: dict,
    rng: Any,
    *,
    expected_experiment_dir: Path | str | None = None,
) -> list[dict]:
    del rng
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(policy_path, map_location=device, weights_only=False)
    ckpt_meta = ckpt.get("meta") or {}
    saved_exp = ckpt_meta.get("experiment_dir")
    if expected_experiment_dir is not None and saved_exp:
        if Path(saved_exp).resolve() != Path(expected_experiment_dir).resolve():
            raise ValueError(
                f"Policy {policy_path} belongs to {saved_exp!r}, "
                f"not this run ({Path(expected_experiment_dir).resolve()})."
            )
    horizon = training_horizon({**meta, **ckpt_meta}, cfg)
    policy = DADPolicy(meta["n_actions"], max_steps=horizon).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    out = []
    for sys in test_systems:
        used: set[int] = set()
        seq, act_h, obs_h = [], [], []
        for _ in range(horizon):
            if not act_h:
                act_t = torch.zeros(1, 0, dtype=torch.long, device=device)
                obs_t = torch.zeros(1, 0, device=device)
                mask_t = torch.zeros(1, 0, device=device)
            else:
                act_t = torch.tensor([act_h], dtype=torch.long, device=device)
                obs_t = torch.tensor([obs_h], dtype=torch.float32, device=device)
                mask_t = torch.ones(1, len(act_h), device=device)
            feas = feasible_mask(used, meta["n_actions"], device).unsqueeze(0)
            a, _, _ = policy.select_action(
                act_t, obs_t, mask_t, feas, deterministic=True
            )
            a_idx = int(a.item())
            seq.append(a_idx)
            y = lookup_action_y(sys, a_idx)
            act_h.append(a_idx)
            obs_h.append(float(y))
            used.add(a_idx)
        out.append({"M": sys["M"], "K": sys["K"], "sequence": seq, "y": list(obs_h)})
    return out
