"""PPO training for DAD (terminal reward) and RL-sBOED (stepwise u_ctrl reward)."""

from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from src.control.objective_rl_sboed.context import (
    StudyContext,
    belief_summary,
    control_from_log_weights,
    observe_bank,
    update_log_weights,
)
from src.control.objective_rl_sboed.layout import (
    prepare_system_experiment,
    publish_checkpoint_to_model,
    training_output_dir,
)
from src.control.objective_rl_sboed.rewards import (
    GAMMA,
    dad_rewards,
    verify_rl_sboed_rollout,
)
from src.experiment_layout import RunMetadata, git_commit_hash, utc_now_stamp, write_run_metadata
from src.control.objective_rl_sboed import ROOT
from src.neural.rl_policy import AdaptiveExperimentPolicy, PolicyConfig, StateValueCritic

RewardMode = Literal["dad_terminal", "rl_sboed_stepwise"]
InitMode = Literal["random", "fixed"]


@dataclass
class TrainConfig:
    updates: int = 120
    trajectories_per_update: int = 16
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    gae_lambda: float = 1.0
    entropy_coefficient: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    max_grad_norm: float = 1.0
    validation_interval: int = 10
    validation_rollouts: int = 64
    patience: int = 8
    gamma: float = GAMMA


def _tensors_from_state(
    ctx: StudyContext,
    *,
    actions: list[int],
    observations: list[float],
    log_w: np.ndarray,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    length = int(ctx.horizon)
    action_idx = torch.zeros(1, length, dtype=torch.long, device=device)
    obs = torch.zeros(1, length, dtype=torch.float32, device=device)
    mask = torch.zeros(1, length, dtype=torch.float32, device=device)
    if actions:
        n = len(actions)
        action_idx[0, :n] = torch.as_tensor(actions, dtype=torch.long)
        y = (np.asarray(observations, dtype=np.float64) - ctx.obs_mean) / ctx.obs_std
        obs[0, :n] = torch.as_tensor(y, dtype=torch.float32)
        mask[0, :n] = 1.0
    belief = torch.as_tensor(
        belief_summary(ctx, log_w, observations)[None, :], dtype=torch.float32, device=device
    )
    steps = torch.as_tensor([step], dtype=torch.long, device=device)
    particles = torch.as_tensor(ctx.particle_features[None, :], dtype=torch.float32, device=device)
    weights = torch.as_tensor(
        np.exp(log_w - np.max(log_w))[None, :], dtype=torch.float32, device=device
    )
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    feasible = torch.ones(1, ctx.n_actions, dtype=torch.bool, device=device)
    for a in actions:
        feasible[0, int(a)] = False
    return action_idx, obs, mask, belief, steps, particles, weights, feasible


def sample_trajectory(
    ctx: StudyContext,
    policy: AdaptiveExperimentPolicy,
    system_row: dict[str, Any],
    *,
    theta_id: int,
    rollout_id: int,
    global_seed: int,
    reward_mode: RewardMode,
    device: torch.device,
    deterministic: bool = False,
) -> dict[str, Any]:
    log_w = ctx.log_p0.copy()
    actions: list[int] = []
    observations: list[float] = []
    u_path = [control_from_log_weights(ctx, log_w).u_ctrl]
    states: list[tuple[torch.Tensor, ...]] = []
    log_probs: list[float] = []
    values: list[float] = []
    policy.eval()
    for step in range(ctx.horizon):
        tensors = _tensors_from_state(
            ctx,
            actions=actions,
            observations=observations,
            log_w=log_w,
            step=step,
            device=device,
        )
        states.append(tensors)
        with torch.no_grad():
            dist = policy.distribution(*tensors[:-1], tensors[-1])
            if deterministic:
                action = int(torch.argmax(dist.probs, dim=-1).item())
            else:
                action = int(dist.sample().item())
            log_probs.append(float(dist.log_prob(torch.tensor(action, device=device)).item()))
        # Value estimated separately when critic provided by caller; placeholder 0 here.
        values.append(0.0)
        y = observe_bank(
            system_row,
            action,
            sigma_y=ctx.sigma_y,
            global_seed=global_seed,
            theta_id=theta_id,
            step=step,
            rollout_id=rollout_id,
        )
        actions.append(action)
        observations.append(y)
        log_w = update_log_weights(ctx, log_w, action, y)
        u_path.append(control_from_log_weights(ctx, log_w).u_ctrl)

    if reward_mode == "dad_terminal":
        trace = dad_rewards(u_path)
    else:
        trace = verify_rl_sboed_rollout(u_path)
    return {
        "actions": actions,
        "observations": observations,
        "u_path": list(trace.u_path),
        "rewards": list(trace.rewards),
        "log_probs": log_probs,
        "states": states,
        "terminal_u_ctrl": trace.terminal_u_ctrl,
        "theta_id": theta_id,
    }


def _gae(rewards: np.ndarray, values: np.ndarray, lam: float, gamma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        advantages[t] = last
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def behavior_clone_fixed(
    ctx: StudyContext,
    policy: AdaptiveExperimentPolicy,
    *,
    seed: int,
    epochs: int = 40,
    batch_size: int = 32,
    device: torch.device,
) -> dict[str, Any]:
    """BC onto Fixed sequence targets (observations vary; actions fixed)."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    fixed = list(ctx.fixed_sequence)
    metrics: list[dict[str, Any]] = []
    policy.train()
    for epoch in range(1, epochs + 1):
        losses = []
        correct = 0
        total = 0
        full_match = 0
        for _ in range(batch_size):
            system = ctx.train_systems[int(rng.integers(0, len(ctx.train_systems)))]
            theta_id = int(rng.integers(0, 10_000))
            log_w = ctx.log_p0.copy()
            actions: list[int] = []
            observations: list[float] = []
            epoch_ok = True
            for step, target in enumerate(fixed):
                tensors = _tensors_from_state(
                    ctx,
                    actions=actions,
                    observations=observations,
                    log_w=log_w,
                    step=step,
                    device=device,
                )
                logits = policy(*tensors[:-1], tensors[-1])
                loss = F.cross_entropy(
                    logits, torch.as_tensor([target], dtype=torch.long, device=device)
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
                pred = int(torch.argmax(logits, dim=-1).item())
                correct += int(pred == target)
                total += 1
                if pred != target:
                    epoch_ok = False
                y = observe_bank(
                    system,
                    target,
                    sigma_y=ctx.sigma_y,
                    global_seed=seed,
                    theta_id=theta_id,
                    step=step,
                    rollout_id=epoch,
                )
                actions.append(target)
                observations.append(y)
                log_w = update_log_weights(ctx, log_w, target, y)
            full_match += int(epoch_ok)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "step_accuracy": correct / max(total, 1),
            "full_sequence_accuracy": full_match / batch_size,
        }
        metrics.append(row)
        if row["full_sequence_accuracy"] >= 0.999 and row["step_accuracy"] >= 0.999:
            break
    return {"metrics": metrics, "final": metrics[-1] if metrics else {}}


@torch.no_grad()
def evaluate_policy(
    ctx: StudyContext,
    policy: AdaptiveExperimentPolicy,
    systems: list[dict[str, Any]],
    *,
    n_rollouts: int,
    global_seed: int,
    reward_mode: RewardMode,
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    rows = []
    for rid in range(n_rollouts):
        tid = int(rid % len(systems))
        traj = sample_trajectory(
            ctx,
            policy,
            systems[tid],
            theta_id=tid,
            rollout_id=rid,
            global_seed=global_seed,
            reward_mode=reward_mode,
            device=device,
            deterministic=True,
        )
        rows.append(
            {
                "split": split_name,
                "rollout_id": rid,
                "theta_id": tid,
                "sequence": " ".join(map(str, traj["actions"])),
                "u_ctrl": traj["terminal_u_ctrl"],
                "u_path": " ".join(f"{x:.6f}" for x in traj["u_path"]),
                "rewards": " ".join(f"{x:.6f}" for x in traj["rewards"]),
            }
        )
    u = np.asarray([r["u_ctrl"] for r in rows], dtype=np.float64)
    seqs = [r["sequence"] for r in rows]
    from collections import Counter

    counts = Counter(seqs)
    dom, dom_n = counts.most_common(1)[0]
    return {
        "mean_u_ctrl": float(u.mean()),
        "median_u_ctrl": float(np.median(u)),
        "std_u_ctrl": float(u.std()),
        "n_unique_sequences": int(len(counts)),
        "dominant_sequence": dom,
        "dominant_sequence_fraction": float(dom_n / len(rows)),
        "rows": rows,
    }


def train_policy(
    ctx: StudyContext,
    *,
    method: Literal["DAD", "RL-sBOED"],
    init_mode: InitMode,
    seed: int,
    output_dir: Path | None = None,
    config: TrainConfig | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    config = config or TrainConfig()
    if smoke:
        config = TrainConfig(
            updates=4,
            trajectories_per_update=4,
            ppo_epochs=2,
            validation_interval=2,
            validation_rollouts=8,
            patience=3,
        )
    reward_mode: RewardMode = (
        "dad_terminal" if method == "DAD" else "rl_sboed_stepwise"
    )
    if abs(config.gamma - 1.0) > 0.0 and method == "RL-sBOED":
        raise ValueError("RL-sBOED requires gamma=1")

    exp_dir = prepare_system_experiment(
        ctx.system,
        horizon=ctx.horizon,
        terminal_rule_hash=ctx.terminal_rule_hash,
        entry_point="run.sh",
    )
    if output_dir is None:
        output_dir = training_output_dir(
            ctx.system, method, init_mode, seed, horizon=ctx.horizon
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(
        exp_dir,
        RunMetadata(
            experiment_name=f"objective_rl_sboed/{ctx.system}_T{ctx.horizon}",
            entry_point="run.sh",
            timestamp_utc=utc_now_stamp(),
            system=ctx.system,
            horizon=ctx.horizon,
            method=method,
            seed=seed,
            git_commit=git_commit_hash(ROOT),
            terminal_rule_hash=ctx.terminal_rule_hash,
            data_dir=str(ctx.exp_dir),
            initialization=init_mode,
            extra={"reward_mode": reward_mode, "train_dir": str(output_dir)},
        ),
    )
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    policy = AdaptiveExperimentPolicy(ctx.n_actions, PolicyConfig(max_steps=ctx.horizon))
    critic = StateValueCritic(ctx.n_actions, PolicyConfig(max_steps=ctx.horizon))
    policy.to(device)
    critic.to(device)

    bc_info: dict[str, Any] | None = None
    if init_mode == "fixed":
        bc_info = behavior_clone_fixed(
            ctx, policy, seed=seed, epochs=20 if smoke else 60, device=device
        )
        (output_dir / "bc_metrics.json").write_text(
            json.dumps(bc_info, indent=2), encoding="utf-8"
        )

    actor_opt = torch.optim.Adam(policy.parameters(), lr=config.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    best_state = None
    best_val = float("inf")
    best_update = 0
    patience_left = config.patience
    history: list[dict[str, Any]] = []
    t0 = time.time()

    for update in range(1, config.updates + 1):
        policy.train()
        critic.train()
        batch_states: list[tuple[torch.Tensor, ...]] = []
        batch_actions: list[int] = []
        batch_old_lp: list[float] = []
        batch_adv: list[float] = []
        batch_ret: list[float] = []
        terminals: list[float] = []
        zero_intermediate = 0

        for traj_i in range(config.trajectories_per_update):
            tid = int(rng.integers(0, len(ctx.train_systems)))
            system = ctx.train_systems[tid]
            # Collect with value estimates.
            log_w = ctx.log_p0.copy()
            actions: list[int] = []
            observations: list[float] = []
            u_path = [control_from_log_weights(ctx, log_w).u_ctrl]
            states = []
            old_lp = []
            values = []
            for step in range(ctx.horizon):
                tensors = _tensors_from_state(
                    ctx,
                    actions=actions,
                    observations=observations,
                    log_w=log_w,
                    step=step,
                    device=device,
                )
                states.append(tensors)
                dist = policy.distribution(*tensors[:-1], tensors[-1])
                action = int(dist.sample().item())
                old_lp.append(float(dist.log_prob(torch.tensor(action, device=device)).item()))
                with torch.no_grad():
                    values.append(float(critic(*tensors[:-1]).item()))
                y = observe_bank(
                    system,
                    action,
                    sigma_y=ctx.sigma_y,
                    global_seed=seed,
                    theta_id=tid,
                    step=step,
                    rollout_id=update * 1000 + traj_i,
                )
                actions.append(action)
                observations.append(y)
                log_w = update_log_weights(ctx, log_w, action, y)
                u_path.append(control_from_log_weights(ctx, log_w).u_ctrl)

            if reward_mode == "dad_terminal":
                trace = dad_rewards(u_path)
            else:
                trace = verify_rl_sboed_rollout(u_path)
            rewards = np.asarray(trace.rewards, dtype=np.float64)
            if reward_mode == "rl_sboed_stepwise" and np.allclose(rewards[:-1], 0.0):
                zero_intermediate += 1
            values_arr = np.asarray(values, dtype=np.float64)
            adv, ret = _gae(rewards, values_arr, config.gae_lambda, gamma=config.gamma)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_old_lp.extend(old_lp)
            batch_adv.extend(adv.tolist())
            batch_ret.extend(ret.tolist())
            terminals.append(float(trace.terminal_u_ctrl))

        # Stack batch
        def cat_field(idx: int) -> torch.Tensor:
            return torch.cat([s[idx] for s in batch_states], dim=0)

        action_t = torch.as_tensor(batch_actions, dtype=torch.long, device=device)
        old_lp_t = torch.as_tensor(batch_old_lp, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(batch_adv, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(batch_ret, dtype=torch.float32, device=device)
        inputs = tuple(cat_field(i) for i in range(8))

        approx_kl = 0.0
        policy_loss_v = 0.0
        value_loss_v = 0.0
        entropy_v = 0.0
        for _ in range(config.ppo_epochs):
            dist = policy.distribution(*inputs[:-1], inputs[-1])
            new_lp = dist.log_prob(action_t)
            ratio = torch.exp(new_lp - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * adv_t
            entropy = dist.entropy().mean()
            policy_loss = -(torch.min(surr1, surr2).mean() + config.entropy_coefficient * entropy)
            values_pred = critic(*inputs[:-1])
            value_loss = F.huber_loss(values_pred, ret_t)
            actor_opt.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            actor_opt.step()
            critic_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            critic_opt.step()
            approx_kl = float((old_lp_t - new_lp.detach()).mean().item())
            policy_loss_v = float(policy_loss.item())
            value_loss_v = float(value_loss.item())
            entropy_v = float(entropy.item())
            if approx_kl > 0.03:
                break

        row: dict[str, Any] = {
            "update": update,
            "mean_train_u_ctrl": float(np.mean(terminals)),
            "policy_loss": policy_loss_v,
            "value_loss": value_loss_v,
            "entropy": entropy_v,
            "approximate_kl": approx_kl,
            "zero_intermediate_reward_fraction": zero_intermediate
            / max(config.trajectories_per_update, 1),
            "reward_mode": reward_mode,
            "method": method,
            "init_mode": init_mode,
        }

        if update % config.validation_interval == 0 or update == config.updates:
            val = evaluate_policy(
                ctx,
                policy,
                ctx.validation_systems,
                n_rollouts=config.validation_rollouts,
                global_seed=seed + 17,
                reward_mode=reward_mode,
                device=device,
                split_name="validation",
            )
            row.update(
                {
                    "validation_mean_u_ctrl": val["mean_u_ctrl"],
                    "validation_unique_sequences": val["n_unique_sequences"],
                    "validation_dominant_fraction": val["dominant_sequence_fraction"],
                }
            )
            # Safety eligibility proxy: finite mean.
            if np.isfinite(val["mean_u_ctrl"]) and val["mean_u_ctrl"] < best_val - 1e-12:
                best_val = float(val["mean_u_ctrl"])
                best_update = update
                best_state = copy.deepcopy(policy.state_dict())
                patience_left = config.patience
                torch.save(
                    {
                        "policy": best_state,
                        "method": method,
                        "init_mode": init_mode,
                        "seed": seed,
                        "update": update,
                        "validation_mean_u_ctrl": best_val,
                        "reward_mode": reward_mode,
                        "scientific_method": method,
                    },
                    output_dir / "best_checkpoint.pt",
                )
                publish_checkpoint_to_model(
                    exp_dir,
                    method,
                    init_mode,
                    seed,
                    output_dir / "best_checkpoint.pt",
                )
            else:
                patience_left -= 1

        history.append(row)
        if patience_left <= 0:
            break

    final_state = copy.deepcopy(policy.state_dict())
    torch.save(
        {
            "policy": final_state,
            "method": method,
            "init_mode": init_mode,
            "seed": seed,
            "checkpoint_kind": "final",
        },
        output_dir / "final_checkpoint.pt",
    )
    if best_state is not None:
        policy.load_state_dict(best_state)

    conf = evaluate_policy(
        ctx,
        policy,
        ctx.confirmation_systems,
        n_rollouts=min(128, max(32, len(ctx.confirmation_systems) * 2)),
        global_seed=seed + 91,
        reward_mode=reward_mode,
        device=device,
        split_name="confirmation",
    )
    _write_csv(output_dir / "training_metrics.csv", history)
    _write_csv(output_dir / "confirmation_rollouts.csv", conf["rows"])
    result = {
        "method": method,
        "init_mode": init_mode,
        "seed": seed,
        "reward_mode": reward_mode,
        "best_update": best_update,
        "best_validation_mean_u_ctrl": best_val,
        "confirmation_mean_u_ctrl": conf["mean_u_ctrl"],
        "confirmation_median_u_ctrl": conf["median_u_ctrl"],
        "confirmation_std_u_ctrl": conf["std_u_ctrl"],
        "confirmation_unique_sequences": conf["n_unique_sequences"],
        "confirmation_dominant_sequence": conf["dominant_sequence"],
        "confirmation_dominant_fraction": conf["dominant_sequence_fraction"],
        "bc": bc_info["final"] if bc_info else None,
        "elapsed_seconds": time.time() - t0,
        "config": asdict(config),
        "terminal_rule_hash": ctx.terminal_rule_hash,
        "uses_offline_banks_only": True,
        "gamma": config.gamma,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
