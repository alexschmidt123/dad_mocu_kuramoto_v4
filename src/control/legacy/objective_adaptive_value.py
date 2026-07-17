"""Objective-based adaptive-value diagnostic for IEEE5/IEEE9 T=3.

Measures whether the optimal second probe changes with the first-step history
and whether that change reduces expected terminal u_ctrl.

This is a diagnostic only. Scientific methods remain DAD / Myopic / Fixed / Random.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.control.legacy.adaptive_value_diagnosis import _batch_u_ctrl, _centres_matrix
from src.control.banks import extract_U_bank
from src.control.pilot import load_pilot_splits
from src.control.posterior_ctrl import normalize_log_weights, posterior_ess
from src.control.terminal_rule import keyed_noise, load_frozen_terminal_rule
from src.contrastive.spce import log_prior_uniform_discrete
from src.data import lookup_action_y_sim
from src.neural.legacy.ppo_stage2 import Stage2Architecture, Stage2Policy
from src.run_context import load_experiment_run
from src.swing_equation_ode.design import build_catalog
from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "objective_adaptive_value"
NEAR_TIE_TOL = 1e-4
GLOBAL_SEED = 44117
N_HISTORIES = 256
N_HYP_OUTER = 96
N_HYP_INNER = 48
N_BOOT = 10000
DAD_SEEDS = (101, 202, 303, 404, 505)
# Split-sample MC is expensive; default worker count keeps CPU busy without thrashing.
DEFAULT_WORKERS = 8


@dataclass
class SystemBundle:
    system: str
    centres: np.ndarray
    U: np.ndarray
    log_p0: np.ndarray
    p0: np.ndarray
    sigma_y: float
    alpha: float
    margin: float
    u_grid: np.ndarray
    n_actions: int
    diagnostic_systems: list[dict[str, Any]]
    support: TableThetaSupport
    M_support: np.ndarray
    K_support: np.ndarray
    obs_mean: float
    obs_std: float
    particle_features: np.ndarray


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating,)):
        return _jsonable(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_bundle(system: str) -> SystemBundle:
    exp = ROOT / "experiments" / f"{system}_T3"
    run = load_experiment_run(exp, ROOT)
    splits = load_pilot_splits(exp, run)
    frozen = load_frozen_terminal_rule(exp)
    support_systems = list(splits["support_systems"])
    support = TableThetaSupport(
        systems=support_systems,
        log_p0=log_prior_uniform_discrete(len(support_systems)),
    )
    U = extract_U_bank(support.systems)
    n_actions = len(build_catalog(run.cfg))
    centres = _centres_matrix(support, n_actions)
    log_p0 = np.asarray(support.log_p0, dtype=np.float64)
    p0 = normalize_log_weights(log_p0)
    # Dedicated diagnostic split: validation only (never confirmation/test).
    diagnostic = list(splits["validation_systems"])
    M = np.asarray([row["M"] for row in support.systems], dtype=np.float64)
    K = np.asarray([row["K"] for row in support.systems], dtype=np.float64)
    train_like = list(splits["support_systems"]) + list(splits["calibration_systems"])
    obs = np.asarray(
        [
            lookup_action_y_sim(system_row, action)
            for system_row in train_like
            for action in range(n_actions)
        ],
        dtype=np.float64,
    )
    raw_particles = np.concatenate([M, K, U[:, None]], axis=1)
    mean = raw_particles.mean(axis=0)
    std = np.maximum(raw_particles.std(axis=0), 1e-8)
    particles = ((raw_particles - mean) / std).astype(np.float32)
    return SystemBundle(
        system=system,
        centres=centres,
        U=U,
        log_p0=log_p0,
        p0=p0,
        sigma_y=float(run.cfg.sigma_y),
        alpha=float(frozen.alpha),
        margin=float(frozen.margin),
        u_grid=np.asarray(frozen.u_candidates, dtype=np.float64),
        n_actions=n_actions,
        diagnostic_systems=diagnostic,
        support=support,
        M_support=M,
        K_support=K,
        obs_mean=float(obs.mean()),
        obs_std=float(max(obs.std(), 1e-8)),
        particle_features=particles,
    )


def _normalize_rows(log_w: np.ndarray) -> np.ndarray:
    flat = np.asarray(log_w, dtype=np.float64)
    if flat.ndim == 1:
        return normalize_log_weights(flat)
    m = flat.max(axis=1, keepdims=True)
    w = np.exp(flat - m)
    return w / np.clip(w.sum(axis=1, keepdims=True), 1e-300, None)


def _sample_indices(weights: np.ndarray, n_hyp: int, rng: np.random.Generator) -> np.ndarray:
    """weights: (B, N) -> idx (B, n_hyp)."""
    cdf = np.cumsum(weights, axis=1)
    cdf[:, -1] = 1.0
    uniforms = rng.random((weights.shape[0], n_hyp))
    return (uniforms[..., None] >= cdf[:, None, :]).sum(axis=-1).astype(np.int64)


def batch_min_expected_u(
    log_w: np.ndarray,
    forbidden: set[int],
    *,
    centres: np.ndarray,
    U: np.ndarray,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_actions: int,
    n_hyp: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """For each posterior, min_a E_y[u_ctrl|h,a] with shared CRN across actions."""
    if log_w.ndim == 1:
        log_w = log_w[None, :]
    weights = _normalize_rows(log_w)
    idx = _sample_indices(weights, n_hyp, rng)
    noise = rng.normal(0.0, sigma_y, size=idx.shape)
    s2 = float(sigma_y) ** 2
    log_const = -0.5 * math.log(2.0 * math.pi * s2)
    best = np.full(log_w.shape[0], np.inf, dtype=np.float64)
    for action in range(n_actions):
        if action in forbidden:
            continue
        centre = centres[action]
        y = centre[idx] + noise
        log_L = log_const - 0.5 * ((y[..., None] - centre[None, None, :]) ** 2) / s2
        log_w_h = log_w[:, None, :] + log_L
        costs = _batch_u_ctrl(
            U,
            log_w_h.reshape(-1, log_w.shape[-1]),
            alpha=alpha,
            margin=margin,
            u_grid=u_grid,
        ).reshape(log_w.shape[0], n_hyp)
        best = np.minimum(best, costs.mean(axis=1))
    return best


def j_h1_xi2(
    log_w1: np.ndarray,
    xi1: int,
    xi2: int,
    *,
    centres: np.ndarray,
    U: np.ndarray,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_actions: int,
    n_hyp_outer: int,
    n_hyp_inner: int,
    rng: np.random.Generator,
) -> float:
    """Two-step lookahead J(h1, xi2) under terminal u_ctrl."""
    w1 = _normalize_rows(log_w1)
    idx2 = rng.choice(len(w1), size=n_hyp_outer, p=w1)
    y2 = centres[xi2, idx2] + rng.normal(0.0, sigma_y, size=n_hyp_outer)
    s2 = float(sigma_y) ** 2
    log_L2 = (
        -0.5 * math.log(2.0 * math.pi * s2)
        - 0.5 * ((y2[:, None] - centres[xi2][None, :]) ** 2) / s2
    )
    log_w2 = log_w1[None, :] + log_L2
    leaf = batch_min_expected_u(
        log_w2,
        {xi1, xi2},
        centres=centres,
        U=U,
        sigma_y=sigma_y,
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        n_actions=n_actions,
        n_hyp=n_hyp_inner,
        rng=rng,
    )
    return float(leaf.mean())


def generate_first_histories(
    bundle: SystemBundle,
    xi1: int,
    n_histories: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Diagnostic first-step histories from validation systems × keyed noise."""
    systems = bundle.diagnostic_systems
    replicas = int(math.ceil(n_histories / max(len(systems), 1)))
    rows: list[dict[str, Any]] = []
    history_id = 0
    for replica in range(replicas):
        for theta_id, system_row in enumerate(systems):
            if history_id >= n_histories:
                return rows
            z = keyed_noise(
                global_seed=GLOBAL_SEED,
                theta_id=theta_id,
                rollout_id=replica,
                step=0,
                action_id=xi1,
            )
            y1 = float(lookup_action_y_sim(system_row, xi1)) + float(
                bundle.sigma_y
            ) * float(z)
            s2 = float(bundle.sigma_y) ** 2
            log_L = (
                -0.5 * math.log(2.0 * math.pi * s2)
                - 0.5 * ((y1 - bundle.centres[xi1]) ** 2) / s2
            )
            log_w1 = bundle.log_p0 + log_L
            w1 = normalize_log_weights(log_w1)
            rows.append(
                {
                    "history_id": history_id,
                    "xi1": xi1,
                    "theta_id": theta_id,
                    "rollout_id": replica,
                    "y1": y1,
                    "log_w1": log_w1,
                    "ess": float(posterior_ess(w1)),
                }
            )
            history_id += 1
    return rows


def _score_all_xi2(
    log_w1: np.ndarray,
    xi1: int,
    *,
    centres: np.ndarray,
    U: np.ndarray,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_actions: int,
    n_hyp_outer: int,
    n_hyp_inner: int,
    rng: np.random.Generator,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for xi2 in range(n_actions):
        if xi2 == xi1:
            continue
        scores[xi2] = j_h1_xi2(
            log_w1,
            xi1,
            xi2,
            centres=centres,
            U=U,
            sigma_y=sigma_y,
            alpha=alpha,
            margin=margin,
            u_grid=u_grid,
            n_actions=n_actions,
            n_hyp_outer=n_hyp_outer,
            n_hyp_inner=n_hyp_inner,
            rng=rng,
        )
    return scores


def score_history(
    history: dict[str, Any],
    *,
    centres: np.ndarray,
    U: np.ndarray,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_actions: int,
    n_hyp_outer: int,
    n_hyp_inner: int,
    seed: int,
) -> dict[str, Any]:
    """Split-sample selection/evaluation to avoid optimistic min-of-noise bias."""
    xi1 = int(history["xi1"])
    log_w1 = history["log_w1"]
    select_scores = _score_all_xi2(
        log_w1,
        xi1,
        centres=centres,
        U=U,
        sigma_y=sigma_y,
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        n_actions=n_actions,
        n_hyp_outer=n_hyp_outer,
        n_hyp_inner=n_hyp_inner,
        rng=np.random.default_rng(seed),
    )
    eval_scores = _score_all_xi2(
        log_w1,
        xi1,
        centres=centres,
        U=U,
        sigma_y=sigma_y,
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        n_actions=n_actions,
        n_hyp_outer=n_hyp_outer,
        n_hyp_inner=n_hyp_inner,
        rng=np.random.default_rng(seed + 917_411),
    )
    ordered_select = sorted(
        select_scores.items(), key=lambda item: (item[1], item[0])
    )
    best_a = int(ordered_select[0][0])
    ordered_eval = sorted(eval_scores.items(), key=lambda item: (item[1], item[0]))
    second_a, second_j = ordered_eval[1]
    best_j_eval = float(eval_scores[best_a])
    gap = float(second_j - ordered_eval[0][1])
    co_tied = sum(
        1
        for _, value in ordered_select
        if abs(value - ordered_select[0][1]) <= 1e-12
    )
    near_tied = sum(
        1
        for _, value in ordered_select
        if abs(value - ordered_select[0][1]) <= NEAR_TIE_TOL
    )
    return {
        "history_id": history["history_id"],
        "xi1": xi1,
        "theta_id": history["theta_id"],
        "rollout_id": history["rollout_id"],
        "y1": history["y1"],
        "posterior_ess": history["ess"],
        "xi2_star": best_a,
        "best_expected_u_ctrl": best_j_eval,
        "xi2_second": int(second_a),
        "second_best_expected_u_ctrl": float(second_j),
        "gap_best_second": gap,
        "n_co_tied": int(co_tied),
        "n_near_tied": int(near_tied),
        # Independent evaluation scores used for J_common / Delta.
        "all_scores": eval_scores,
        "select_scores": select_scores,
    }


def _worker_score(payload: dict[str, Any]) -> dict[str, Any]:
    return score_history(
        payload["history"],
        centres=payload["centres"],
        U=payload["U"],
        sigma_y=payload["sigma_y"],
        alpha=payload["alpha"],
        margin=payload["margin"],
        u_grid=payload["u_grid"],
        n_actions=payload["n_actions"],
        n_hyp_outer=payload["n_hyp_outer"],
        n_hyp_inner=payload["n_hyp_inner"],
        seed=payload["seed"],
    )


def evaluate_xi1(
    bundle: SystemBundle,
    xi1: int,
    *,
    n_histories: int,
    n_hyp_outer: int,
    n_hyp_inner: int,
    workers: int,
) -> list[dict[str, Any]]:
    histories = generate_first_histories(
        bundle, xi1, n_histories, np.random.default_rng(GLOBAL_SEED + 17 * xi1)
    )
    payloads = [
        {
            "history": {
                "history_id": history["history_id"],
                "xi1": history["xi1"],
                "theta_id": history["theta_id"],
                "rollout_id": history["rollout_id"],
                "y1": history["y1"],
                "ess": history["ess"],
                "log_w1": history["log_w1"],
            },
            "centres": bundle.centres,
            "U": bundle.U,
            "sigma_y": bundle.sigma_y,
            "alpha": bundle.alpha,
            "margin": bundle.margin,
            "u_grid": bundle.u_grid,
            "n_actions": bundle.n_actions,
            "n_hyp_outer": n_hyp_outer,
            "n_hyp_inner": n_hyp_inner,
            "seed": GLOBAL_SEED + 10007 * xi1 + history["history_id"],
        }
        for history in histories
    ]
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        for payload in payloads:
            rows.append(_worker_score(payload))
        return rows
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker_score, payload) for payload in payloads]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: int(row["history_id"]))
    return rows


def adaptive_gain_for_xi1(scored: list[dict[str, Any]]) -> dict[str, Any]:
    j_adaptive_values = np.asarray(
        [float(row["best_expected_u_ctrl"]) for row in scored], dtype=np.float64
    )
    j_adaptive = float(j_adaptive_values.mean())
    # Reconstruct common score from stored all_scores.
    actions = sorted(
        {
            int(action)
            for row in scored
            for action in row["all_scores"].keys()
        }
    )
    common_means = {
        action: float(
            np.mean([float(row["all_scores"][action]) for row in scored])
        )
        for action in actions
    }
    common_action = min(common_means, key=lambda action: (common_means[action], action))
    j_common_values = np.asarray(
        [float(row["all_scores"][common_action]) for row in scored], dtype=np.float64
    )
    j_common = float(j_common_values.mean())
    paired = j_common_values - j_adaptive_values
    delta = float(paired.mean())
    rng = np.random.default_rng(GLOBAL_SEED + 91)
    boots = np.empty(N_BOOT, dtype=np.float64)
    n = len(paired)
    for index in range(N_BOOT):
        sample = rng.integers(0, n, size=n)
        boots[index] = float(paired[sample].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    xi2_star = [int(row["xi2_star"]) for row in scored]
    counts = Counter(xi2_star)
    dominant, dominant_count = counts.most_common(1)[0]
    return {
        "xi1": int(scored[0]["xi1"]),
        "n_histories": len(scored),
        "J_adaptive": j_adaptive,
        "J_common": j_common,
        "Delta_adaptive": delta,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "common_xi2": int(common_action),
        "n_unique_xi2_star": len(counts),
        "dominant_xi2_star": int(dominant),
        "dominant_fraction": dominant_count / len(scored),
        "near_tie_rate": float(
            np.mean([float(row["n_near_tied"]) > 1 for row in scored])
        ),
        "mean_gap_best_second": float(
            np.mean([float(row["gap_best_second"]) for row in scored])
        ),
        "xi2_star_distribution": dict(counts),
    }


def y1_bin_distribution(scored: list[dict[str, Any]], n_bins: int = 5) -> list[dict[str, Any]]:
    y1 = np.asarray([float(row["y1"]) for row in scored], dtype=np.float64)
    edges = np.unique(np.quantile(y1, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        edges = np.asarray([y1.min() - 1e-9, y1.max() + 1e-9])
    labels = ["low", "medium_low", "medium", "medium_high", "high"]
    rows = []
    for bin_id in range(min(n_bins, max(1, edges.size - 1))):
        if bin_id == edges.size - 1:
            break
        left, right = edges[bin_id], edges[bin_id + 1]
        if bin_id == edges.size - 2:
            mask = (y1 >= left) & (y1 <= right)
        else:
            mask = (y1 >= left) & (y1 < right)
        members = [row for keep, row in zip(mask, scored) if keep]
        if not members:
            continue
        counts = Counter(int(row["xi2_star"]) for row in members)
        dominant, count = counts.most_common(1)[0]
        rows.append(
            {
                "xi1": int(scored[0]["xi1"]),
                "y1_bin": labels[min(bin_id, len(labels) - 1)],
                "y1_bin_id": bin_id,
                "y1_low": float(left),
                "y1_high": float(right),
                "n_histories": len(members),
                "dominant_xi2_star": int(dominant),
                "dominant_fraction": count / len(members),
                "n_unique_xi2_star": len(counts),
                "xi2_star_distribution": json.dumps(dict(counts)),
            }
        )
    return rows


def verify_n_hyp_stability(
    bundle: SystemBundle,
    xi1_list: list[int],
    *,
    n_histories: int = 12,
) -> list[dict[str, Any]]:
    """Compare MC budgets via split-sample Delta and argmin agreement."""
    rows = []
    for xi1 in xi1_list:
        histories = generate_first_histories(
            bundle, xi1, n_histories, np.random.default_rng(GLOBAL_SEED + xi1)
        )
        scored_by_budget: dict[str, list[dict[str, Any]]] = {}
        for label, outer, inner in (
            ("96_48", 96, 48),
            ("128_64", 128, 64),
        ):
            scored = [
                score_history(
                    history,
                    centres=bundle.centres,
                    U=bundle.U,
                    sigma_y=bundle.sigma_y,
                    alpha=bundle.alpha,
                    margin=bundle.margin,
                    u_grid=bundle.u_grid,
                    n_actions=bundle.n_actions,
                    n_hyp_outer=outer,
                    n_hyp_inner=inner,
                    seed=GLOBAL_SEED + history["history_id"],
                )
                for history in histories
            ]
            scored_by_budget[label] = scored
            gain = adaptive_gain_for_xi1(scored)
            rows.append(
                {
                    "xi1": xi1,
                    "budget": label,
                    "n_histories": n_histories,
                    "J_adaptive": gain["J_adaptive"],
                    "J_common": gain["J_common"],
                    "Delta_adaptive": gain["Delta_adaptive"],
                    "ci95_low": gain["ci95_low"],
                    "ci95_high": gain["ci95_high"],
                    "n_unique_xi2_star": gain["n_unique_xi2_star"],
                    "mean_gap_best_second": gain["mean_gap_best_second"],
                    "xi2_star_agreement": float("nan"),
                }
            )
        low = scored_by_budget["96_48"]
        high = scored_by_budget["128_64"]
        agree = float(
            np.mean(
                [int(a["xi2_star"] == b["xi2_star"]) for a, b in zip(low, high)]
            )
        )
        rows.append(
            {
                "xi1": xi1,
                "budget": "agreement_96_48_vs_128_64",
                "n_histories": n_histories,
                "J_adaptive": float("nan"),
                "J_common": float("nan"),
                "Delta_adaptive": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
                "n_unique_xi2_star": int(
                    len({int(row["xi2_star"]) for row in low})
                ),
                "mean_gap_best_second": float(
                    np.mean([row["gap_best_second"] for row in low])
                ),
                "xi2_star_agreement": agree,
            }
        )
    return rows


def _load_dad_policy(bundle: SystemBundle, checkpoint: Path) -> Stage2Policy:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = Stage2Architecture(**payload["config"]["architecture"])
    policy = Stage2Policy(
        bundle.n_actions,
        summary_dim=33,
        particle_dim=bundle.particle_features.shape[1],
        architecture=architecture,
    )
    policy.load_state_dict(payload["policy"])
    policy.eval()
    return policy


def dad_second_action(
    bundle: SystemBundle,
    policy: Stage2Policy,
    xi1: int,
    y1: float,
    log_w1: np.ndarray,
) -> int:
    device = torch.device("cpu")
    actions = torch.tensor([[xi1]], dtype=torch.long, device=device)
    observations = torch.tensor(
        [[(y1 - bundle.obs_mean) / bundle.obs_std]],
        dtype=torch.float32,
        device=device,
    )
    history_mask = torch.ones(1, 1, dtype=torch.float32, device=device)
    summary = torch.zeros(1, 33, dtype=torch.float32, device=device)
    steps = torch.tensor([1], dtype=torch.long, device=device)
    particles = torch.as_tensor(
        bundle.particle_features[None, :, :], dtype=torch.float32, device=device
    )
    weights = torch.as_tensor(
        normalize_log_weights(log_w1)[None, :], dtype=torch.float32, device=device
    )
    feasible = torch.ones(1, bundle.n_actions, dtype=torch.bool, device=device)
    feasible[0, xi1] = False
    with torch.no_grad():
        logits = policy(
            actions,
            observations,
            history_mask,
            summary,
            steps,
            particles,
            weights,
            feasible,
        )
        return int(torch.argmax(logits, dim=-1).item())


def dad_regret_rows(
    bundle: SystemBundle,
    scored: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in DAD_SEEDS:
        checkpoint = (
            OUT.parent
            / "dad_ppo_stage2"
            / f"{bundle.system}_T3"
            / "final_8seeds"
            / f"seed_{seed}"
            / "best_checkpoint.pt"
        )
        if not checkpoint.is_file():
            continue
        policy = _load_dad_policy(bundle, checkpoint)
        for row in scored:
            # Need log_w1 again from y1/xi1.
            xi1 = int(row["xi1"])
            y1 = float(row["y1"])
            s2 = float(bundle.sigma_y) ** 2
            log_L = (
                -0.5 * math.log(2.0 * math.pi * s2)
                - 0.5 * ((y1 - bundle.centres[xi1]) ** 2) / s2
            )
            log_w1 = bundle.log_p0 + log_L
            xi2_dad = dad_second_action(bundle, policy, xi1, y1, log_w1)
            j_dad = float(row["all_scores"].get(xi2_dad, np.nan))
            if np.isnan(j_dad):
                # Should not happen; recompute if needed.
                continue
            rows.append(
                {
                    "system": bundle.system,
                    "dad_seed": seed,
                    "history_id": row["history_id"],
                    "xi1": xi1,
                    "y1": y1,
                    "xi2_star": row["xi2_star"],
                    "xi2_DAD": xi2_dad,
                    "agree": int(xi2_dad == int(row["xi2_star"])),
                    "J_star": row["best_expected_u_ctrl"],
                    "J_DAD": j_dad,
                    "regret": j_dad - float(row["best_expected_u_ctrl"]),
                }
            )
    return rows


def make_plots(
    system: str,
    history_rows: list[dict[str, Any]],
    gain_rows: list[dict[str, Any]],
    regret_rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # xi2_star versus y1 for a few high-delta first probes
    top = sorted(gain_rows, key=lambda row: -float(row["Delta_adaptive"]))[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    for axis, gain in zip(axes.ravel(), top):
        members = [
            row for row in history_rows if int(row["xi1"]) == int(gain["xi1"])
        ]
        axis.scatter(
            [float(row["y1"]) for row in members],
            [int(row["xi2_star"]) for row in members],
            s=12,
            alpha=0.7,
        )
        axis.set_title(f"xi1={gain['xi1']}")
        axis.set_xlabel("y1")
        axis.set_ylabel("xi2_star")
    fig.tight_layout()
    fig.savefig(plots / "xi2_star_versus_y1.png", dpi=150)
    plt.close(fig)

    # optimal xi2 frequency for best xi1
    best = min(gain_rows, key=lambda row: float(row["J_adaptive"]))
    members = [row for row in history_rows if int(row["xi1"]) == int(best["xi1"])]
    counts = Counter(int(row["xi2_star"]) for row in members)
    fig, ax = plt.subplots(figsize=(8, 4))
    actions = sorted(counts)
    ax.bar(actions, [counts[action] for action in actions])
    ax.set_xlabel("xi2_star")
    ax.set_ylabel("frequency")
    ax.set_title(f"{system}: optimal xi2 frequency for best xi1={best['xi1']}")
    fig.tight_layout()
    fig.savefig(plots / "optimal_xi2_frequency.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    order = sorted(gain_rows, key=lambda row: int(row["xi1"]))
    ax.bar(
        [int(row["xi1"]) for row in order],
        [float(row["Delta_adaptive"]) for row in order],
        yerr=[
            [
                float(row["Delta_adaptive"]) - float(row["ci95_low"])
                for row in order
            ],
            [
                float(row["ci95_high"]) - float(row["Delta_adaptive"])
                for row in order
            ],
        ],
        capsize=2,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("xi1")
    ax.set_ylabel("Delta_adaptive")
    ax.set_title(f"{system}: Delta_adaptive by first probe")
    fig.tight_layout()
    fig.savefig(plots / "delta_adaptive_by_xi1.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        [float(row["J_common"]) for row in gain_rows],
        [float(row["J_adaptive"]) for row in gain_rows],
        s=20,
    )
    lims = [
        min(float(row["J_adaptive"]) for row in gain_rows)
        - 0.01,
        max(float(row["J_common"]) for row in gain_rows)
        + 0.01,
    ]
    ax.plot(lims, lims, "--", color="gray")
    ax.set_xlabel("J_common")
    ax.set_ylabel("J_adaptive")
    ax.set_title(f"{system}: J_adaptive vs J_common")
    fig.tight_layout()
    fig.savefig(plots / "j_adaptive_vs_j_common.png", dpi=150)
    plt.close(fig)

    if regret_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        regrets = [float(row["regret"]) for row in regret_rows]
        ax.hist(regrets, bins=30, color="#4c78a8")
        ax.set_xlabel("DAD next-action regret")
        ax.set_ylabel("count")
        ax.set_title(f"{system}: DAD action regret")
        fig.tight_layout()
        fig.savefig(plots / "dad_action_regret.png", dpi=150)
        plt.close(fig)


def classify_case(gain_rows: list[dict[str, Any]]) -> str:
    deltas = np.asarray(
        [float(row["Delta_adaptive"]) for row in gain_rows], dtype=np.float64
    )
    unique = np.asarray(
        [int(row["n_unique_xi2_star"]) for row in gain_rows], dtype=np.float64
    )
    significant = [
        row
        for row in gain_rows
        if float(row["ci95_low"]) > 0 and float(row["Delta_adaptive"]) > 1e-4
    ]
    changing = unique.mean() > 1.5
    if len(significant) == 0 and not changing:
        return "A"
    if changing and len(significant) == 0:
        return "B"
    if len(significant) >= max(1, int(0.25 * len(gain_rows))):
        return "C"
    if 0 < len(significant) < max(1, int(0.25 * len(gain_rows))):
        return "D"
    return "B"


def run_system(
    system: str,
    *,
    n_histories: int = N_HISTORIES,
    n_hyp_outer: int = N_HYP_OUTER,
    n_hyp_inner: int = N_HYP_INNER,
    workers: int = 4,
    xi1_subset: list[int] | None = None,
    skip_stability: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    bundle = load_bundle(system)
    out_dir = OUT / f"{system}_T3"
    out_dir.mkdir(parents=True, exist_ok=True)
    xi1_list = xi1_subset or list(range(bundle.n_actions))

    if skip_stability:
        agree = float("nan")
    else:
        print(f"[{system}] n_hyp stability check on subset...", flush=True)
        stability = verify_n_hyp_stability(
            bundle,
            xi1_list[: min(2, len(xi1_list))],
            n_histories=4,
        )
        _write_csv(out_dir / "n_hyp_stability.csv", stability)
        agreement_rows = [
            row
            for row in stability
            if str(row["budget"]).startswith("agreement_")
        ]
        agree = (
            float(np.mean([row["xi2_star_agreement"] for row in agreement_rows]))
            if agreement_rows
            else float("nan")
        )
        print(
            f"[{system}] n_hyp stability: xi2_star agreement={agree:.3f}",
            flush=True,
        )

    history_path = out_dir / "first_history_results.csv"
    gain_path = out_dir / "adaptive_gain.csv"
    dist_path = out_dir / "xi2_distribution.csv"
    history_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    done_xi1: set[int] = set()
    if gain_path.is_file():
        with gain_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                gain_rows.append(row)
                done_xi1.add(int(row["xi1"]))
        if history_path.is_file():
            with history_path.open(encoding="utf-8", newline="") as stream:
                history_rows = list(csv.DictReader(stream))
        if dist_path.is_file():
            with dist_path.open(encoding="utf-8", newline="") as stream:
                distribution_rows = list(csv.DictReader(stream))
        print(
            f"[{system}] resuming with {len(done_xi1)} completed first probes",
            flush=True,
        )

    for index, xi1 in enumerate(xi1_list):
        if xi1 in done_xi1:
            continue
        print(
            f"[{system}] scoring xi1={xi1} ({index + 1}/{len(xi1_list)})",
            flush=True,
        )
        scored = evaluate_xi1(
            bundle,
            xi1,
            n_histories=n_histories,
            n_hyp_outer=n_hyp_outer,
            n_hyp_inner=n_hyp_inner,
            workers=workers,
        )
        gain = adaptive_gain_for_xi1(scored)
        gain_rows.append(
            {k: v for k, v in gain.items() if k != "xi2_star_distribution"}
            | {
                "xi2_star_distribution": json.dumps(gain["xi2_star_distribution"])
            }
        )
        distribution_rows.extend(y1_bin_distribution(scored))
        for row in scored:
            history_rows.append(
                {
                    "history_id": row["history_id"],
                    "xi1": row["xi1"],
                    "theta_id": row["theta_id"],
                    "rollout_id": row["rollout_id"],
                    "y1": row["y1"],
                    "posterior_ess": row["posterior_ess"],
                    "xi2_star": row["xi2_star"],
                    "best_expected_u_ctrl": row["best_expected_u_ctrl"],
                    "xi2_second": row["xi2_second"],
                    "second_best_expected_u_ctrl": row["second_best_expected_u_ctrl"],
                    "gap_best_second": row["gap_best_second"],
                    "n_co_tied": row["n_co_tied"],
                    "n_near_tied": row["n_near_tied"],
                    "all_scores_json": json.dumps(
                        {str(k): float(v) for k, v in row["all_scores"].items()}
                    ),
                }
            )
        _write_csv(history_path, history_rows)
        _write_csv(gain_path, gain_rows)
        _write_csv(dist_path, distribution_rows)

    # Rebuild scored all_scores for DAD regret from saved JSON.
    scored_for_dad = []
    for row in history_rows:
        scored_for_dad.append(
            {
                **row,
                "all_scores": {
                    int(k): float(v)
                    for k, v in json.loads(row["all_scores_json"]).items()
                },
                "best_expected_u_ctrl": float(row["best_expected_u_ctrl"]),
                "xi2_star": int(row["xi2_star"]),
            }
        )
    print(f"[{system}] DAD action regret...", flush=True)
    regret_rows = dad_regret_rows(bundle, scored_for_dad)
    _write_csv(out_dir / "dad_action_regret.csv", regret_rows)
    make_plots(system, history_rows, gain_rows, regret_rows, out_dir)

    best = min(gain_rows, key=lambda row: float(row["J_adaptive"]))
    case = classify_case(gain_rows)
    summary = {
        "system": system,
        "horizon": 3,
        "n_histories_per_xi1": n_histories,
        "n_hyp_outer": n_hyp_outer,
        "n_hyp_inner": n_hyp_inner,
        "n_hyp_stability_agreement": agree,
        "case": case,
        "best_xi1_for_adaptive_planning": int(best["xi1"]),
        "J_adaptive_best_xi1": float(best["J_adaptive"]),
        "J_common_best_xi1": float(best["J_common"]),
        "Delta_adaptive_best_xi1": float(best["Delta_adaptive"]),
        "ci95_low_best_xi1": float(best["ci95_low"]),
        "ci95_high_best_xi1": float(best["ci95_high"]),
        "n_unique_optimal_xi2_best_xi1": int(best["n_unique_xi2_star"]),
        "dominant_xi2_fraction_best_xi1": float(best["dominant_fraction"]),
        "near_tie_rate_best_xi1": float(best["near_tie_rate"]),
        "mean_Delta_adaptive_all_xi1": float(
            np.mean([float(row["Delta_adaptive"]) for row in gain_rows])
        ),
        "fraction_xi1_with_significant_Delta": float(
            np.mean(
                [
                    float(row["ci95_low"]) > 0
                    for row in gain_rows
                ]
            )
        ),
        "dad_action_agreement": (
            float(np.mean([row["agree"] for row in regret_rows]))
            if regret_rows
            else float("nan")
        ),
        "dad_mean_regret": (
            float(np.mean([row["regret"] for row in regret_rows]))
            if regret_rows
            else float("nan")
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "uses_offline_banks_only": True,
        "used_confirmation_split": False,
        "methods_in_project": ["DAD", "Myopic", "Fixed", "Random"],
    }
    _write_json(out_dir / "system_summary.json", summary)
    return summary


def write_final_report(summaries: list[dict[str, Any]]) -> Path:
    summary_dir = OUT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_dir / "system_comparison.csv", summaries)
    by_system = {row["system"]: row for row in summaries}
    lines = [
        "# Objective-based adaptive-value test",
        "",
        "Scientific methods in the main project remain only:",
        "",
        "- DAD",
        "- Myopic",
        "- Fixed",
        "- Random",
        "",
        "This directory is a diagnostic of intrinsic adaptive value under terminal "
        "`u_ctrl`. It is not a new experiment-design method.",
        "",
        "## Answers",
        "",
    ]
    ieee5 = by_system.get("ieee5", {})
    ieee9 = by_system.get("ieee9", {})

    def change_text(row: dict[str, Any]) -> str:
        if not row:
            return "not run"
        if int(row["n_unique_optimal_xi2_best_xi1"]) <= 1 and float(
            row["dominant_xi2_fraction_best_xi1"]
        ) > 0.95:
            return "mostly one second probe"
        return (
            f"yes ({int(row['n_unique_optimal_xi2_best_xi1'])} unique xi2_star; "
            f"dominant fraction {float(row['dominant_xi2_fraction_best_xi1']):.3f})"
        )

    lines += [
        f"1. IEEE5 best-second-probe changes across first-step histories: "
        f"{change_text(ieee5)}.",
        f"2. IEEE9 best-second-probe changes across first-step histories: "
        f"{change_text(ieee9)}.",
        "3. Systematic vs noise/ties: see near-tie rates and y1-bin distributions "
        "in each system's `xi2_distribution.csv`. Changes are treated as systematic "
        "only when y1 bins prefer different second probes and `Delta_adaptive` is "
        "material.",
        "4. History-dependent second-probe selection reduces expected final "
        f"`u_ctrl` when `Delta_adaptive` CI excludes 0. IEEE5 case "
        f"{ieee5.get('case', 'n/a')}; IEEE9 case {ieee9.get('case', 'n/a')}.",
        f"5. IEEE5 Delta_adaptive (best xi1={ieee5.get('best_xi1_for_adaptive_planning')}): "
        f"{ieee5.get('Delta_adaptive_best_xi1', float('nan')):.6f}, "
        f"95% CI [{ieee5.get('ci95_low_best_xi1', float('nan')):.6f}, "
        f"{ieee5.get('ci95_high_best_xi1', float('nan')):.6f}].",
        f"6. IEEE9 Delta_adaptive (best xi1={ieee9.get('best_xi1_for_adaptive_planning')}): "
        f"{ieee9.get('Delta_adaptive_best_xi1', float('nan')):.6f}, "
        f"95% CI [{ieee9.get('ci95_low_best_xi1', float('nan')):.6f}, "
        f"{ieee9.get('ci95_high_best_xi1', float('nan')):.6f}].",
        "7. Fixed is naturally strong when one common second probe is near-optimal "
        "for most histories (`Delta_adaptive ≈ 0`).",
        f"8. DAD second-action agreement with xi2_star: IEEE5 "
        f"{ieee5.get('dad_action_agreement', float('nan')):.3f}, IEEE9 "
        f"{ieee9.get('dad_action_agreement', float('nan')):.3f}.",
        f"9. DAD mean next-action regret: IEEE5 "
        f"{ieee5.get('dad_mean_regret', float('nan')):.6f}, IEEE9 "
        f"{ieee9.get('dad_mean_regret', float('nan')):.6f}.",
        "10. Main limitation: if case A/B, low intrinsic adaptive value; if case C "
        "and DAD regret is large, imperfect DAD training; if case D, first-probe "
        "selection is the bottleneck.",
        "",
        "## Notes",
        "",
        "- Outer histories use the validation diagnostic split with keyed observation "
        "noise; confirmation/test systems are never used.",
        "- All physical responses come from offline banks; no ODE calls.",
        "- Objective is terminal u_ctrl only.",
    ]
    path = summary_dir / "final_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("ieee5", "ieee9", "both"), default="both")
    parser.add_argument("--n-histories", type=int, default=N_HISTORIES)
    parser.add_argument("--n-hyp-outer", type=int, default=N_HYP_OUTER)
    parser.add_argument("--n-hyp-inner", type=int, default=N_HYP_INNER)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument(
        "--xi1",
        type=str,
        default="",
        help="Optional comma-separated first-probe subset for debugging",
    )
    args = parser.parse_args()
    systems = ("ieee5", "ieee9") if args.system == "both" else (args.system,)
    xi1_subset = (
        [int(value) for value in args.xi1.split(",") if value.strip() != ""]
        if args.xi1
        else None
    )
    summaries = []
    for system in systems:
        summaries.append(
            run_system(
                system,
                n_histories=args.n_histories,
                n_hyp_outer=args.n_hyp_outer,
                n_hyp_inner=args.n_hyp_inner,
                workers=args.workers,
                xi1_subset=xi1_subset,
                skip_stability=args.skip_stability,
            )
        )
    report = write_final_report(summaries)
    print(json.dumps({"summaries": summaries, "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
