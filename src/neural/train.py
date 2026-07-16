"""DAD training: REINFORCE minimizing terminal posterior-safe u_ctrl."""

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

try:  # pragma: no cover
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)

from src.control.banks import extract_U_bank
from src.control.posterior_ctrl import normalize_log_weights, posterior_safe_u_ctrl
from src.control.u_req import ControlSpec
from src.contrastive.spce import log_gaussian_observation_density
from src.data import lookup_action_y, validate_trajectory_y_sim
from src.neural.policy import DADPolicy
from src.config import SBOEDConfig
from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables


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
) -> tuple[list[int], list[float], torch.Tensor, torch.Tensor]:
    """On-policy rollout over the *complete* history {(ξ_i, y_i)}."""
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
        feas = _feasible_mask(used, n_actions, device).unsqueeze(0)
        a, log_p, ent = policy.select_action(act_t, obs_t, mask_t, feas, deterministic=False)
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


def _terminal_u_ctrl(
    support: TableThetaSupport,
    U_support: np.ndarray,
    seq: list[int],
    y_list: list[float],
    sigma_y: float,
    alpha: float,
    *,
    margin: float = 0.0,
    u_grid=None,
) -> float:
    """Posterior from full probe history → posterior-safe u_ctrl."""
    log_w = np.asarray(support.log_p0, dtype=np.float64).copy()
    for a, y in zip(seq, y_list):
        centres = y_sim_last_step_from_tables(support, [int(a)])
        log_L = log_gaussian_observation_density(float(y), centres, sigma_y)
        log_w = log_w + log_L
    w = normalize_log_weights(log_w)
    return float(
        posterior_safe_u_ctrl(U_support, w, alpha, margin=margin, u_grid=u_grid)
    )


def _entropy_coef_at_epoch(entropy_coef: float, epoch: int, epochs: int) -> float:
    if epochs <= 1 or entropy_coef <= 0:
        return float(max(0.0, entropy_coef))
    frac = 1.0 - float(epoch) / float(epochs - 1)
    return float(max(0.0, entropy_coef * frac))


def _training_horizon(meta: dict, cfg: SBOEDConfig) -> int:
    if meta.get("experiment_step_number") is not None:
        return int(meta["experiment_step_number"])
    if meta.get("training_horizon") is not None:
        return int(meta["training_horizon"])
    return int(cfg.step_number)


def train_dad_policy(
    cfg: SBOEDConfig,
    train_systems: list[dict],
    meta: dict,
    output_dir: Path,
    *,
    data_dir: Path | None = None,
    run_tag: str = "dad",
    validation_systems: list[dict] | None = None,
    support_systems: list[dict] | None = None,
) -> Path:
    """
    Foster-style DAD with table bank. REINFORCE minimizing E[u_ctrl(H_T)].

    Terminal cost = posterior-safe u_ctrl after the full T-step history.
    Checkpoint selected by smallest validation mean u_ctrl when validation_systems given.
    Posterior particles come from support_systems when provided (never test).
    """
    del data_dir
    validate_trajectory_y_sim(train_systems, split="train")
    for i, sys in enumerate(train_systems):
        if "u_req" not in sys:
            raise KeyError(f"train system[{i}] missing u_req; regenerate control bank")

    output_dir.mkdir(parents=True, exist_ok=True)
    n_actions = meta["n_actions"]
    step_number = _training_horizon(meta, cfg)
    tr = cfg.raw.setdefault("training", {})
    epochs = int(tr.get("epochs", 20))
    batch_size = int(tr.get("batch_size", 8))
    lr = float(tr.get("learning_rate", 1e-3))
    grad_clip = float(tr.get("grad_clip", 1.0))
    reinforce_ema = float(tr.get("reinforce_baseline_ema", 0.9))
    entropy_coef0 = float(tr.get("entropy_coef", 0.01))
    eval_interval = max(1, int(tr.get("evaluation_interval", 1)))
    control_spec = ControlSpec.from_cfg(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DADPolicy(n_actions, max_steps=step_number).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(int(cfg.data.get("train_seed", 0)))
    mc_seed = int(cfg.prior.get("mc_support_seed", int(cfg.data.get("train_seed", 0))))
    particle_systems = support_systems if support_systems is not None else train_systems
    support = TableThetaSupport.from_train(
        particle_systems, cfg, np.random.default_rng(mc_seed),
    )
    U_support = extract_U_bank(support.systems)

    steps_per_epoch = int(tr.get("steps_per_epoch", max(batch_size * 4, len(train_systems))))
    epoch_losses: list[float] = []
    epoch_cost: list[float] = []
    epoch_entropy: list[float] = []
    epoch_val: list[float] = []
    epoch_grad_norm: list[float] = []
    epoch_baseline: list[float] = []
    epoch_checkpoint_log: list[dict[str, Any]] = []
    baseline = 0.0
    best_val = float("inf")
    best_state: dict[str, Any] | None = None

    print(
        f"  DAD train: T={step_number} objective=min E[u_ctrl] epochs={epochs} "
        f"batch={batch_size} support={len(support)} α={control_spec.alpha} | "
        f"complete history encoder"
    )
    t0 = time.perf_counter()
    epoch_iter = tqdm(range(epochs), desc=f"train {run_tag}", unit="epoch")

    for epoch in epoch_iter:
        losses: list[float] = []
        cost_vals: list[float] = []
        entropy_vals: list[float] = []
        grad_norms: list[float] = []
        lambda_ent = _entropy_coef_at_epoch(entropy_coef0, epoch, epochs)

        for start in range(0, steps_per_epoch, batch_size):
            bl: list[torch.Tensor] = []
            n_batch = min(batch_size, steps_per_epoch - start)
            for _ in range(n_batch):
                i_sys = int(rng.integers(len(train_systems)))
                sys = train_systems[i_sys]
                seq, y_list, log_ps, ents = _policy_rollout(
                    policy, device, sys, step_number, n_actions,
                )
                entropy_vals.append(float(ents.detach().mean().item()))
                cost = _terminal_u_ctrl(
                    support,
                    U_support,
                    seq,
                    y_list,
                    cfg.sigma_y,
                    control_spec.alpha,
                    margin=float(getattr(control_spec, "safety_margin", 0.0)),
                    u_grid=control_spec.u_candidates,
                )
                cost_vals.append(cost)
                # Minimize cost ⇒ advantage = baseline - cost (lower cost → positive adv).
                baseline = reinforce_ema * baseline + (1.0 - reinforce_ema) * cost
                adv = float(baseline - cost)
                pg_loss = -log_ps.sum() * adv
                loss = pg_loss - lambda_ent * ents.sum()
                bl.append(loss)

            opt.zero_grad()
            total = torch.stack(bl).mean()
            total.backward()
            if grad_clip > 0:
                gn = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip))
            else:
                gn = float(
                    torch.sqrt(
                        sum(
                            (p.grad.detach() ** 2).sum()
                            for p in policy.parameters()
                            if p.grad is not None
                        )
                    )
                )
            grad_norms.append(gn)
            opt.step()
            losses.append(float(total.item()))

        epoch_losses.append(float(np.mean(losses)))
        epoch_cost.append(float(np.mean(cost_vals)) if cost_vals else 0.0)
        epoch_entropy.append(float(np.mean(entropy_vals)) if entropy_vals else 0.0)
        epoch_grad_norm.append(float(np.mean(grad_norms)) if grad_norms else 0.0)
        epoch_baseline.append(float(baseline))

        val_mean = epoch_cost[-1]
        val_safety = float("nan")
        do_val = (epoch % eval_interval == 0) or (epoch == epochs - 1)
        if validation_systems and do_val:
            val_costs = []
            val_safe_flags: list[float] = []
            policy.eval()
            with torch.no_grad():
                for sys in validation_systems[: min(64, len(validation_systems))]:
                    # Deterministic validation rollout.
                    used: set[int] = set()
                    seq_d, y_d, act_h, obs_h = [], [], [], []
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
                        a, _, _ = policy.select_action(
                            act_t, obs_t, mask_t, feas, deterministic=True,
                        )
                        a_idx = int(a.item())
                        seq_d.append(a_idx)
                        y = float(lookup_action_y(sys, a_idx))
                        y_d.append(y)
                        act_h.append(a_idx)
                        obs_h.append(y)
                        used.add(a_idx)
                    u_c = _terminal_u_ctrl(
                        support,
                        U_support,
                        seq_d,
                        y_d,
                        cfg.sigma_y,
                        control_spec.alpha,
                        margin=float(getattr(control_spec, "safety_margin", 0.0)),
                        u_grid=control_spec.u_candidates,
                    )
                    val_costs.append(u_c)
                    # Safety-first proxy: u_ctrl >= u_req (same implication used in calibration).
                    u_req = float(sys.get("u_req", np.nan))
                    val_safe_flags.append(1.0 if u_c + 1e-12 >= u_req else 0.0)
            policy.train()
            val_mean = float(np.mean(val_costs)) if val_costs else val_mean
            val_safety = float(np.mean(val_safe_flags)) if val_safe_flags else float("nan")
        elif not validation_systems:
            val_mean = epoch_cost[-1]
            val_safety = 1.0
        # Keep last validation value between evaluation intervals.
        if not epoch_val:
            epoch_val.append(float(val_mean))
        elif do_val or not validation_systems:
            epoch_val.append(float(val_mean))
        else:
            epoch_val.append(float(epoch_val[-1]))

        # Safety-first checkpoint: reject validation_safety < 1.0; else min mean u_ctrl.
        if do_val or not validation_systems:
            ckpt_row = {
                "epoch": int(epoch),
                "validation_mean_u_ctrl": float(val_mean),
                "validation_safety_rate": float(val_safety),
                "admissible": bool(
                    (not validation_systems) or abs(val_safety - 1.0) < 1e-12
                ),
                "rejection_reason": (
                    ""
                    if (not validation_systems) or abs(val_safety - 1.0) < 1e-12
                    else "validation_safety_below_1"
                ),
            }
            epoch_checkpoint_log.append(ckpt_row)
            if ckpt_row["admissible"] and val_mean < best_val:
                best_val = val_mean
                best_state = {
                    "state_dict": {
                        k: v.detach().cpu().clone() for k, v in policy.state_dict().items()
                    },
                    "val_u_ctrl": best_val,
                    "val_safety": float(val_safety),
                    "epoch": epoch,
                }

        epoch_iter.set_postfix(
            loss=f"{epoch_losses[-1]:.4f}",
            u_ctrl=f"{epoch_cost[-1]:.4f}",
            val=f"{val_mean:.4f}",
            Hπ=f"{epoch_entropy[-1]:.3f}",
        )

    if best_state is not None:
        policy.load_state_dict(best_state["state_dict"])

    policy_path = output_dir / f"{run_tag}.pth"
    save_meta = {
        **meta,
        "step_number": step_number,
        "training_horizon": step_number,
        "experiment_step_number": step_number,
        "encoder": "attention_pool",
        "entropy_coef": entropy_coef0,
        "objective": "min_u_ctrl",
        "best_val_u_ctrl": best_val if np.isfinite(best_val) else None,
        "best_epoch": None if best_state is None else best_state.get("epoch"),
    }
    torch.save({"state_dict": policy.state_dict(), "meta": save_meta}, policy_path)

    metrics = {
        "epochs": epochs,
        "objective": "min_u_ctrl",
        "training_horizon": step_number,
        "support_size": len(support),
        "entropy_coef": entropy_coef0,
        "encoder": "attention_pool",
        "epoch_losses": epoch_losses,
        "training_loss": epoch_losses,
        "epoch_mean_u_ctrl": epoch_cost,
        "mean_training_u_ctrl": epoch_cost,
        "epoch_val_u_ctrl": epoch_val,
        "validation_mean_u_ctrl": epoch_val,
        "epoch_mean_policy_entropy": epoch_entropy,
        "policy_entropy": epoch_entropy,
        "gradient_norm": epoch_grad_norm,
        "baseline_value": epoch_baseline,
        "best_val_u_ctrl": best_val if np.isfinite(best_val) else None,
        "best_val_safety": (
            None if best_state is None else best_state.get("val_safety")
        ),
        "checkpoint_metric": "safety_first_then_validation_mean_u_ctrl",
        "checkpoint_log": epoch_checkpoint_log,
        "elapsed_seconds": float(time.perf_counter() - t0),
        "device": str(device),
    }
    with (output_dir / f"{run_tag}_training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    # Stable alias for pilot loaders.
    with (output_dir / "dad_training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    # Alias policy path.
    alias = output_dir / "dad.pth"
    if policy_path.resolve() != alias.resolve():
        import shutil

        shutil.copy2(policy_path, alias)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(epoch_cost) + 1), epoch_cost, "g-o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mean train u_ctrl")
    ax.set_title("DAD training (min u_ctrl)")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / f"{run_tag}_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return policy_path


def rollout_dad(
    cfg: SBOEDConfig,
    test_systems: list[dict],
    policy_path: Path,
    meta: dict,
    rng: np.random.Generator,
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
    horizon = _training_horizon({**meta, **ckpt_meta}, cfg)
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
            feas = _feasible_mask(used, meta["n_actions"], device).unsqueeze(0)
            a, _, _ = policy.select_action(act_t, obs_t, mask_t, feas, deterministic=True)
            a_idx = int(a.item())
            seq.append(a_idx)
            y = lookup_action_y(sys, a_idx)
            act_h.append(a_idx)
            obs_h.append(float(y))
            used.add(a_idx)
        out.append({"M": sys["M"], "K": sys["K"], "sequence": seq, "y": list(obs_h)})
    return out
