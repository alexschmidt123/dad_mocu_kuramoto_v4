"""DAD training: RL-style REINFORCE with sPCE or delta_H reward."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.neural.policy import DADPolicy
from src.config import SBOEDConfig
from src.data import (
    clear_trajectory_sim_context,
    lookup_prefix_y,
    set_trajectory_sim_context,
    validate_trajectory_y_sim,
)
from src.contrastive.spce import posterior_after_gaussian_observations, posterior_entropy
from src.table_scoring import (
    TableThetaSupport,
    spce_eig_train_row,
    y_sim_steps_from_tables,
)


def _feasible_mask(used: set[int], n_actions: int, device: torch.device) -> torch.Tensor:
    m = torch.ones(n_actions, dtype=torch.bool, device=device)
    for i in used:
        m[i] = False
    return m


def _policy_rollout(
    policy: DADPolicy,
    device: torch.device,
    sys: dict[str, Any],
    step_number: int,
    n_actions: int,
) -> tuple[list[int], list[float], torch.Tensor]:
    """On-policy rollout: π sees only noisy ``y`` from the train table."""
    used: set[int] = set()
    seq: list[int] = []
    y_list: list[float] = []
    log_probs: list[torch.Tensor] = []
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
        feas = _feasible_mask(used, n_actions, device).unsqueeze(0)
        a, log_p = policy.select_action(act_t, obs_t, mask_t, feas, deterministic=False)
        a_idx = int(a.item())
        seq.append(a_idx)
        y_hist = lookup_prefix_y(sys, seq)
        y_list.append(float(y_hist[-1]))
        log_probs.append(log_p.squeeze(0))
        act_h.append(a_idx)
        obs_h.append(float(y_hist[-1]))
        used.add(a_idx)

    return seq, y_list, torch.stack(log_probs).sum()


def train_dad_policy(
    cfg: SBOEDConfig,
    train_systems: list[dict],
    meta: dict,
    output_dir: Path,
    *,
    data_dir: Path | None = None,
    run_tag: str = "dad_spce",
) -> Path:
    """
    Foster DAD with table bank (no ODE at train time). RL-style REINFORCE only.

    - **Policy (student):** on-policy rollout; history is (action, noisy **y**) only.
    - **Reward (oracle):** scalar from the same rollout — uses realized noisy **y** plus
      banked **y_sim** centres on the particle support (for sPCE or ΔH). Not imitation.
    - **objective:** ``reinforce`` (dad_spce) or ``delta_h`` (dad_delta_h).
    """
    del data_dir
    validate_trajectory_y_sim(train_systems, split="train")
    set_trajectory_sim_context(cfg, int(cfg.data.get("train_seed", 0)))

    output_dir.mkdir(parents=True, exist_ok=True)
    n_actions = meta["n_actions"]
    step_number = int(meta.get("step_number", meta.get("horizon", cfg.step_number)))
    tr = cfg.raw.setdefault("training", {})
    epochs = int(tr.get("epochs", 20))
    batch_size = int(tr.get("batch_size", 8))
    lr = float(tr.get("learning_rate", 1e-3))
    objective = str(tr.get("objective", "reinforce")).lower()
    if objective not in {"reinforce", "delta_h"}:
        raise ValueError(
            f"Unsupported training.objective='{objective}'. "
            "Use one of: ['reinforce', 'delta_h']."
        )
    grad_clip = float(tr.get("grad_clip", 1.0))
    reinforce_ema = float(tr.get("reinforce_baseline_ema", 0.9))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DADPolicy(n_actions).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(int(cfg.data.get("train_seed", 0)))
    mc_seed = int(cfg.prior.get("mc_support_seed", int(cfg.data.get("train_seed", 0))))
    support = TableThetaSupport.from_train(
        train_systems, cfg, np.random.default_rng(mc_seed),
    )

    steps_per_epoch = int(tr.get("steps_per_epoch", max(batch_size * 4, len(train_systems))))
    epoch_losses: list[float] = []
    epoch_reward: list[float] = []
    baseline_spce = 0.0

    print(
        f"  DAD train: objective={objective} epochs={epochs} batch={batch_size} "
        f"support={len(support)} | on-policy π: noisy y only | reward: y + y_sim centres"
    )
    t0 = time.perf_counter()

    for epoch in range(epochs):
        losses: list[float] = []
        reward_vals: list[float] = []
        for start in range(0, steps_per_epoch, batch_size):
            bl: list[torch.Tensor] = []
            n_batch = min(batch_size, steps_per_epoch - start)

            for _ in range(n_batch):
                i_sys = int(rng.integers(len(train_systems)))
                sys = train_systems[i_sys]
                seq, y_list, log_p = _policy_rollout(
                    policy, device, sys, step_number, n_actions,
                )
                if objective == "delta_h":
                    centre_steps = y_sim_steps_from_tables(support, seq)
                    p_final, p_trace = posterior_after_gaussian_observations(
                        centre_steps,
                        np.asarray(y_list, dtype=np.float64),
                        cfg.sigma_y,
                        support.log_p0,
                    )
                    reward = float(posterior_entropy(p_trace[0]) - posterior_entropy(p_final))
                else:
                    _, _, reward = spce_eig_train_row(cfg, seq, y_list, sys, support, rng)
                reward_vals.append(float(reward))
                baseline_spce = reinforce_ema * baseline_spce + (1.0 - reinforce_ema) * reward
                adv = float(reward - baseline_spce)
                loss = -log_p * adv

                bl.append(loss)

            opt.zero_grad()
            total = torch.stack(bl).mean()
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            opt.step()
            losses.append(float(total.item()))

        epoch_losses.append(float(np.mean(losses)))
        epoch_reward.append(float(np.mean(reward_vals)) if reward_vals else 0.0)
        print(
            f"  epoch {epoch + 1}/{epochs} loss={epoch_losses[-1]:.4f} "
            f"mean_reward={epoch_reward[-1]:.4f}"
        )

    policy_path = output_dir / f"{run_tag}.pth"
    torch.save({"state_dict": policy.state_dict(), "meta": meta}, policy_path)

    metrics = {
        "epochs": epochs,
        "objective": objective,
        "support_size": len(support),
        "epoch_losses": epoch_losses,
        "epoch_mean_reward": epoch_reward,
        "elapsed_seconds": float(time.perf_counter() - t0),
        "device": str(device),
    }
    with (output_dir / f"{run_tag}_training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, "b-o", markersize=3, label="loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"DAD training ({objective})")
    ax.grid(True, alpha=0.3)
    if epoch_reward:
        ax2 = ax.twinx()
        ax2.plot(range(1, len(epoch_reward) + 1), epoch_reward, "g--", markersize=3)
        ax2.set_ylabel("mean reward")
    fig.savefig(output_dir / f"{run_tag}_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    clear_trajectory_sim_context()
    return policy_path


def rollout_dad(
    cfg: SBOEDConfig,
    test_systems: list[dict],
    policy_path: Path,
    meta: dict,
    rng: np.random.Generator,
) -> list[dict]:
    del rng
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(policy_path, map_location=device, weights_only=False)
    policy = DADPolicy(meta["n_actions"]).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    out = []
    for sys in test_systems:
        used: set[int] = set()
        seq, act_h, obs_h = [], [], []
        for _ in range(cfg.step_number):
            if not act_h:
                act_t = torch.zeros(1, 0, dtype=torch.long, device=device)
                obs_t = torch.zeros(1, 0, device=device)
                mask_t = torch.zeros(1, 0, device=device)
            else:
                act_t = torch.tensor([act_h], dtype=torch.long, device=device)
                obs_t = torch.tensor([obs_h], dtype=torch.float32, device=device)
                mask_t = torch.ones(1, len(act_h), device=device)
            feas = _feasible_mask(used, meta["n_actions"], device).unsqueeze(0)
            a, _ = policy.select_action(act_t, obs_t, mask_t, feas, deterministic=True)
            a_idx = int(a.item())
            seq.append(a_idx)
            y_hist = lookup_prefix_y(sys, seq)
            act_h.append(a_idx)
            obs_h.append(float(y_hist[-1]))
            used.add(a_idx)
        out.append({"M": sys["M"], "K": sys["K"], "sequence": seq, "y": y_hist})
    return out
