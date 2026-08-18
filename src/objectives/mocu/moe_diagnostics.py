"""Mechanism diagnostics for the belief-conditioned MoE policy.

These routines answer the "why does it work?" question a reviewer asks of a
mixture-of-experts model: do the experts specialize to distinct belief regimes,
and does the router select among them as a function of belief?  Nothing here
trains or mutates a checkpoint; it rolls out a trained MoE deterministically and
records router weights, expert action rankings, and belief features per step.

Debug suite (collapse / influence / fingerprint structure):
  1. Expert utilization / collapse metrics from top-k router mass.
  2. Mixture influence: does the routed policy disagree with the dominant
     expert alone or a uniform expert average?
  3. Belief/fingerprint PCA: are router decisions structured in
     (ESS, u_ctrl, step, router masses) space, or one amorphous blob?
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.objectives.mocu.context import (
    GLOBAL_SEED,
    ExperimentContext,
    control_from_log_weights,
    observe_compressed,
    posterior_ess,
    update_posterior_vector,
)
from src.objectives.mocu.train import (
    _resolve_device,
    _tensors_from_state,
    load_trained_policy,
)
from src.policies.moe import (
    BeliefConditionedMoEPolicy,
    SharedBaseResidualMoEPolicy,
)
from src.layout import model_dir


def _expert_argmax(expert_values: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    """Argmax action of each expert restricted to the feasible set."""
    masked = np.where(feasible[None, :], expert_values, -np.inf)
    return masked.argmax(axis=-1)


def _pairwise_disagreement(top_actions: np.ndarray) -> float:
    """Fraction of expert pairs whose top feasible action differs."""
    n = len(top_actions)
    if n < 2:
        return 0.0
    diff = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            diff += int(top_actions[i] != top_actions[j])
    return diff / max(total, 1)


def _feasible_argmax(values: np.ndarray, feasible: np.ndarray) -> int:
    masked = np.where(feasible, values, -np.inf)
    return int(masked.argmax())


def collect_moe_router_records(
    ctx: ExperimentContext,
    *,
    n_rollouts: int,
    seed: int = 101,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Deterministically roll out the trained MoE, logging router internals."""
    del seed  # observation noise keys are GLOBAL_SEED + rollout ids, as in eval
    device = device or _resolve_device("auto")
    policy = load_trained_policy(ctx, "MoE-sBOED", device=device)
    if not isinstance(policy, (BeliefConditionedMoEPolicy, SharedBaseResidualMoEPolicy)):
        raise TypeError(
            "moe_diagnostics requires a MoE policy checkpoint "
            "(BeliefConditionedMoEPolicy or SharedBaseResidualMoEPolicy)"
        )
    n_experts = policy.n_experts
    top_k = policy.top_k
    records: list[dict[str, Any]] = []
    for rid in range(int(n_rollouts)):
        tid = rid % len(ctx.test_systems)
        system = ctx.test_systems[tid]
        log_w = ctx.log_p0.copy()
        actions: list[int] = []
        observations: list[np.ndarray] = []
        for step in range(ctx.horizon):
            tensors = _tensors_from_state(
                ctx,
                actions=actions,
                observations=observations,
                log_w=log_w,
                step=step,
                device=device,
            )
            feasible_t = tensors[-1]
            with torch.no_grad():
                components = policy._components(*tensors[:-1])
                # BeliefConditionedMoEPolicy v3 returns
                # (logits, weights, experts, scale, base); legacy residual
                # returns four values with logits = base + sigmoid*routed.
                logits_scaled = components[0]
                dense_weights = components[1]
                expert_values = components[2]
                scale = components[3]
                masked = logits_scaled.masked_fill(~feasible_t, -1e9)
                action = int(torch.argmax(masked, dim=-1).item())
            dense = dense_weights.squeeze(0).cpu().numpy()
            experts = expert_values.squeeze(0).cpu().numpy()  # n_experts, n_actions
            feasible = feasible_t.squeeze(0).cpu().numpy().astype(bool)
            top_idx = np.argsort(-dense)[:top_k]
            top_mass = dense[top_idx]
            top_mass = top_mass / max(top_mass.sum(), 1e-8)
            sparse = np.zeros(n_experts, dtype=np.float64)
            sparse[top_idx] = top_mass
            router_entropy = float(
                -(dense * np.log(np.clip(dense, 1e-12, 1.0))).sum()
            )
            expert_top = _expert_argmax(experts, feasible)
            dominant = int(top_idx[0])
            routed = (sparse[:, None] * experts).sum(axis=0)
            uniform = experts.mean(axis=0)
            action_dominant = _feasible_argmax(experts[dominant], feasible)
            action_uniform = _feasible_argmax(uniform, feasible)
            action_routed_unit = _feasible_argmax(routed, feasible)
            ctrl = control_from_log_weights(ctx, log_w)
            n_particles = len(log_w)
            ess_frac = posterior_ess(log_w) / float(n_particles)
            w_norm = np.exp(log_w - log_w.max())
            w_norm = w_norm / w_norm.sum()
            # Compact expert fingerprint: z-scored feasible logits of dominant
            # and second expert (for PCA clustering diagnostics).
            fp_feats: list[float] = []
            for e in list(top_idx)[: min(2, len(top_idx))]:
                vals = experts[int(e)][feasible]
                if vals.size == 0:
                    continue
                centered = vals - vals.mean()
                denom = float(np.sqrt((centered * centered).mean())) + 1e-4
                # Keep a short prefix so PCA stays cheap across action spaces.
                prefix = centered / denom
                fp_feats.extend(prefix[:8].tolist())
            while len(fp_feats) < 16:
                fp_feats.append(0.0)
            record: dict[str, Any] = {
                "rollout_id": rid,
                "theta_id": tid,
                "step": step,
                "ess_fraction": float(ess_frac),
                "max_weight": float(w_norm.max()),
                "u_ctrl": float(ctrl.u_ctrl),
                "chosen_action": action,
                "dominant_expert": dominant,
                "router_entropy": router_entropy,
                "expert_disagreement": _pairwise_disagreement(expert_top),
                "logit_scale": float(scale.item()),
                "action_matches_dominant_expert": int(action == action_dominant),
                "action_matches_uniform_experts": int(action == action_uniform),
                "action_matches_routed_unscaled": int(action == action_routed_unit),
                "flip_vs_dominant": int(action != action_dominant),
                "flip_vs_uniform": int(action != action_uniform),
            }
            for e in range(n_experts):
                record[f"router_dense_{e}"] = float(dense[e])
                record[f"router_top2_{e}"] = float(sparse[e])
                record[f"expert_top_action_{e}"] = int(expert_top[e])
            for i, v in enumerate(fp_feats[:16]):
                record[f"fingerprint_{i}"] = float(v)
            records.append(record)
            y = observe_compressed(
                system,
                action,
                sigma_y=ctx.sigma_y,
                n_obs=ctx.n_obs,
                global_seed=GLOBAL_SEED,
                theta_id=tid,
                rollout_id=rid,
                step=step,
            )
            actions.append(action)
            observations.append(y)
            log_w = update_posterior_vector(ctx, log_w, action, y)
    return records


def _pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D PCA embedding and explained-variance ratios (no sklearn)."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        return np.zeros((x.shape[0], 2)), np.asarray([0.0, 0.0])
    x = x - x.mean(axis=0, keepdims=True)
    # Guard against constant columns.
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    x = x / std
    # Economy SVD.
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    components = min(2, vt.shape[0])
    embedding = x @ vt[:components].T
    if components == 1:
        embedding = np.concatenate(
            [embedding, np.zeros((embedding.shape[0], 1))], axis=1
        )
    total = float((singular ** 2).sum())
    if total <= 1e-12:
        evr = np.asarray([0.0, 0.0])
    else:
        ev = (singular[:components] ** 2) / total
        evr = np.zeros(2, dtype=np.float64)
        evr[: components] = ev
    return embedding, evr


def summarize_router_records(
    records: list[dict[str, Any]], n_experts: int, horizon: int
) -> dict[str, Any]:
    """Aggregate per-step router/expert records into publication statistics."""
    if not records:
        return {"n_records": 0}
    top2 = np.asarray(
        [[r[f"router_top2_{e}"] for e in range(n_experts)] for r in records]
    )
    dense = np.asarray(
        [[r[f"router_dense_{e}"] for e in range(n_experts)] for r in records]
    )
    overall_usage = top2.mean(axis=0)
    usage_by_step: dict[int, list[float]] = {}
    for step in range(horizon):
        mask = np.asarray([r["step"] == step for r in records])
        if mask.any():
            usage_by_step[step] = top2[mask].mean(axis=0).tolist()
    dominant = np.asarray([r["dominant_expert"] for r in records])
    # Active experts: those carrying >5% mean top-2 mass.
    active = int((overall_usage > 0.05).sum())
    mean_disagreement = float(np.mean([r["expert_disagreement"] for r in records]))
    mean_router_entropy = float(np.mean([r["router_entropy"] for r in records]))
    max_entropy = float(np.log(max(n_experts, 1)))
    # Belief dependence: does the dominant expert change with ESS / step?
    ess = np.asarray([r["ess_fraction"] for r in records])
    step_arr = np.asarray([r["step"] for r in records])
    # Simple dependence measure: variance of ESS conditioned on dominant expert.
    ess_by_expert = {
        int(e): float(np.mean(ess[dominant == e]))
        for e in np.unique(dominant)
    }
    flip_dom = float(np.mean([r["flip_vs_dominant"] for r in records]))
    flip_uni = float(np.mean([r["flip_vs_uniform"] for r in records]))
    match_dom = float(np.mean([r["action_matches_dominant_expert"] for r in records]))

    # Collapse diagnostics on dense router weights.
    mean_dense = dense.mean(axis=0)
    max_usage = float(mean_dense.max())
    # Gini of mean usage (0=uniform, ~1=one expert).
    sorted_u = np.sort(mean_dense)
    n = len(sorted_u)
    gini = float(
        (2.0 * np.sum((np.arange(1, n + 1) * sorted_u)) / (n * sorted_u.sum() + 1e-12))
        - (n + 1) / n
    )
    collapsed = bool(max_usage >= 0.90 or active <= 1)

    # Fingerprint / belief PCA features.
    belief_feats = np.asarray(
        [
            [
                r["ess_fraction"],
                r["u_ctrl"],
                float(r["step"]) / max(horizon - 1, 1),
                r["max_weight"],
                *[r[f"router_dense_{e}"] for e in range(n_experts)],
            ]
            for r in records
        ],
        dtype=np.float64,
    )
    fp_cols = [c for c in records[0] if c.startswith("fingerprint_")]
    fp_feats = np.asarray(
        [[r[c] for c in fp_cols] for r in records], dtype=np.float64
    )
    belief_xy, belief_evr = _pca_2d(belief_feats)
    fp_xy, fp_evr = _pca_2d(fp_feats)
    # Silhouette-like separation proxy: between/within distance of dominant labels.
    def _separation(xy: np.ndarray, labels: np.ndarray) -> float:
        labs = np.unique(labels)
        if len(labs) < 2 or xy.shape[0] < 4:
            return 0.0
        centres = {int(e): xy[labels == e].mean(axis=0) for e in labs}
        within = []
        for e in labs:
            pts = xy[labels == e]
            if len(pts) == 0:
                continue
            within.append(float(np.mean(np.linalg.norm(pts - centres[int(e)], axis=1))))
        between = []
        lab_list = list(labs)
        for i in range(len(lab_list)):
            for j in range(i + 1, len(lab_list)):
                between.append(
                    float(
                        np.linalg.norm(
                            centres[int(lab_list[i])] - centres[int(lab_list[j])]
                        )
                    )
                )
        w = float(np.mean(within)) if within else 0.0
        b = float(np.mean(between)) if between else 0.0
        return float(b / (w + 1e-6))

    return {
        "n_records": len(records),
        "n_experts": n_experts,
        "active_experts": active,
        "overall_top2_usage": overall_usage.tolist(),
        "usage_by_step": {int(k): v for k, v in usage_by_step.items()},
        "mean_expert_disagreement": mean_disagreement,
        "mean_router_entropy": mean_router_entropy,
        "router_entropy_ratio": float(mean_router_entropy / max(max_entropy, 1e-12)),
        "max_dense_usage": max_usage,
        "usage_gini": gini,
        "collapsed_to_one_expert": collapsed,
        "dominant_expert_distribution": {
            int(e): int((dominant == e).sum()) for e in np.unique(dominant)
        },
        "mean_ess_by_dominant_expert": ess_by_expert,
        "dominant_changes_with_step": bool(len(np.unique(dominant)) > 1),
        "action_flip_rate_vs_dominant_expert": flip_dom,
        "action_flip_rate_vs_uniform_experts": flip_uni,
        "action_match_rate_dominant_expert": match_dom,
        "mixture_moves_actions": bool(flip_dom > 0.05 or flip_uni > 0.05),
        "belief_pca_explained_variance": belief_evr.tolist(),
        "fingerprint_pca_explained_variance": fp_evr.tolist(),
        "belief_pca_cluster_separation": _separation(belief_xy, dominant),
        "fingerprint_pca_cluster_separation": _separation(fp_xy, dominant),
        "_belief_pca_xy": belief_xy.tolist(),
        "_fingerprint_pca_xy": fp_xy.tolist(),
    }


def _plot_router(
    summary: dict[str, Any], records: list[dict[str, Any]], out_png: Path
) -> None:
    n_experts = int(summary["n_experts"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    usage_by_step = summary["usage_by_step"]
    steps = sorted(int(s) for s in usage_by_step)
    for e in range(n_experts):
        axes[0, 0].plot(
            steps,
            [usage_by_step[s][e] for s in steps],
            marker="o",
            markersize=3,
            label=f"expert {e}",
        )
    axes[0, 0].set_xlabel("experiment step")
    axes[0, 0].set_ylabel("mean top-2 router mass")
    axes[0, 0].set_title("Router utilization by step")
    axes[0, 0].legend(fontsize=8)

    usage = np.asarray(summary["overall_top2_usage"])
    axes[0, 1].bar(range(n_experts), usage, color="steelblue")
    axes[0, 1].set_xlabel("expert")
    axes[0, 1].set_ylabel("overall top-2 usage")
    axes[0, 1].set_title(
        f"Usage (active={summary['active_experts']}, "
        f"maxDense={summary['max_dense_usage']:.2f}, "
        f"collapse={summary['collapsed_to_one_expert']})"
    )

    # Belief-regime scatter: which expert dominates as ESS / u_ctrl change?
    ess = np.asarray([r["ess_fraction"] for r in records], dtype=np.float64)
    u_ctrl = np.asarray([r["u_ctrl"] for r in records], dtype=np.float64)
    dominant = np.asarray([r["dominant_expert"] for r in records], dtype=int)
    step_arr = np.asarray([r["step"] for r in records], dtype=int)
    cmap = plt.get_cmap("tab10")
    for e in range(n_experts):
        mask = dominant == e
        if not mask.any():
            continue
        axes[0, 2].scatter(
            ess[mask],
            u_ctrl[mask],
            s=12,
            alpha=0.45,
            color=cmap(e % 10),
            label=f"expert {e}",
        )
    axes[0, 2].set_xlabel("posterior ESS fraction")
    axes[0, 2].set_ylabel("posterior-safe u_ctrl")
    axes[0, 2].set_title("Dominant expert vs belief (ESS, u_ctrl)")
    axes[0, 2].legend(fontsize=8, loc="best")

    # Expert specialization: disagreement and router entropy vs step.
    for step in steps:
        mask = step_arr == step
        if not mask.any():
            continue
        axes[1, 0].scatter(
            [step] * int(mask.sum()),
            [r["expert_disagreement"] for r, m in zip(records, mask) if m],
            s=10,
            alpha=0.35,
            color="0.4",
        )
    mean_dis_by_step = [
        float(
            np.mean(
                [r["expert_disagreement"] for r in records if int(r["step"]) == step]
            )
        )
        for step in steps
    ]
    axes[1, 0].plot(steps, mean_dis_by_step, "r-o", markersize=4, label="mean disagreement")
    axes[1, 0].set_xlabel("experiment step")
    axes[1, 0].set_ylabel("expert top-action disagreement")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set_title(
        f"Specialization (flip vs dom={summary['action_flip_rate_vs_dominant_expert']:.2f})"
    )
    axes[1, 0].legend(fontsize=8)

    # Belief PCA colored by dominant expert.
    belief_xy = np.asarray(summary.get("_belief_pca_xy", []), dtype=np.float64)
    if belief_xy.size:
        for e in range(n_experts):
            mask = dominant == e
            if not mask.any():
                continue
            axes[1, 1].scatter(
                belief_xy[mask, 0],
                belief_xy[mask, 1],
                s=12,
                alpha=0.45,
                color=cmap(e % 10),
                label=f"expert {e}",
            )
        ev = summary.get("belief_pca_explained_variance", [0, 0])
        axes[1, 1].set_title(
            f"Belief PCA (sep={summary['belief_pca_cluster_separation']:.2f}, "
            f"EVR={ev[0]:.2f}/{ev[1]:.2f})"
        )
        axes[1, 1].set_xlabel("PC1")
        axes[1, 1].set_ylabel("PC2")
        axes[1, 1].legend(fontsize=8, loc="best")

    fp_xy = np.asarray(summary.get("_fingerprint_pca_xy", []), dtype=np.float64)
    if fp_xy.size:
        for e in range(n_experts):
            mask = dominant == e
            if not mask.any():
                continue
            axes[1, 2].scatter(
                fp_xy[mask, 0],
                fp_xy[mask, 1],
                s=12,
                alpha=0.45,
                color=cmap(e % 10),
                label=f"expert {e}",
            )
        ev = summary.get("fingerprint_pca_explained_variance", [0, 0])
        axes[1, 2].set_title(
            f"Expert-fingerprint PCA "
            f"(sep={summary['fingerprint_pca_cluster_separation']:.2f}, "
            f"EVR={ev[0]:.2f}/{ev[1]:.2f})"
        )
        axes[1, 2].set_xlabel("PC1")
        axes[1, 2].set_ylabel("PC2")
        axes[1, 2].legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def moe_mechanism_report(
    ctx: ExperimentContext,
    *,
    n_rollouts: int = 128,
    seed: int = 101,
    device: str = "auto",
) -> dict[str, Any]:
    """Run the full MoE mechanism diagnostic and write CSV/JSON/figure."""
    resolved = _resolve_device(device)
    records = collect_moe_router_records(
        ctx, n_rollouts=n_rollouts, seed=seed, device=resolved
    )
    policy = load_trained_policy(ctx, "MoE-sBOED", device=resolved)
    assert isinstance(
        policy, (BeliefConditionedMoEPolicy, SharedBaseResidualMoEPolicy)
    )
    summary = summarize_router_records(records, policy.n_experts, ctx.horizon)
    diagnostics_dir = Path(ctx.out_dir) / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostics_dir / "moe_router_records.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    png_path = diagnostics_dir / "moe_router_mechanism.png"
    _plot_router(summary, records, png_path)
    # Drop bulky PCA coordinates from the JSON report (kept only for plotting).
    public = {
        k: v
        for k, v in summary.items()
        if not str(k).startswith("_")
    }
    public["records_csv"] = str(csv_path)
    public["figure_png"] = str(png_path)
    public["checkpoint"] = str(model_dir(ctx.out_dir) / "moe_sboed.pth")
    # Compact debug verdict for quick reading.
    public["debug_verdict"] = {
        "expert_collapse": public["collapsed_to_one_expert"],
        "mixture_moves_actions": public["mixture_moves_actions"],
        "belief_regimes_separated": public["belief_pca_cluster_separation"] > 1.0,
        "fingerprint_regimes_separated": (
            public["fingerprint_pca_cluster_separation"] > 1.0
        ),
    }
    json_path = diagnostics_dir / "moe_mechanism_report.json"
    json_path.write_text(json.dumps(public, indent=2), encoding="utf-8")
    public["report_json"] = str(json_path)
    return public
