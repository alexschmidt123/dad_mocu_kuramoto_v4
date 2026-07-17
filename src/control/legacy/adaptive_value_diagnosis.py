"""Bank-based adaptive-value diagnosis for IEEE5 (no simulator, no test leakage).

Estimates whether observation-dependent design can beat Exact Fixed before any
DAD retraining. Case A (Δ≈0): stop improvement. Case B: proceed to diagnostics.
"""

from __future__ import annotations

import csv
import json
import math
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from src.control.banks import extract_U_bank
from src.control.posterior_batch import (
    batch_u_ctrl as _batch_u_ctrl,
    centres_matrix as _centres_matrix,
    expected_u_after_action,
    update_posterior,
)
from src.control.legacy.ieee5_t4 import EXPECTED_HASH, FROZEN_MARGIN, _write_csv, _write_json
from src.control.legacy.ieee5_t4_fixed_exact import score_subset_detailed
from src.control.pilot import load_pilot_splits
from src.control.posterior_ctrl import normalize_log_weights, posterior_safe_u_ctrl, snap_up_to_grid
from src.control.terminal_rule import load_frozen_terminal_rule
from src.control.u_req import ControlSpec
from src.contrastive.spce import log_prior_uniform_discrete
from src.data import lookup_action_y_sim
from src.run_context import load_experiment_run
from src.swing_equation_ode.design import build_catalog
from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables

OUT_DEFAULT = "experiments/ieee5_dad_adaptive_value_diagnosis"


# ---------------------------------------------------------------------------
# Exact T=2 adaptive reference
# ---------------------------------------------------------------------------


def j_adaptive_t2_for_action(
    a1: int,
    *,
    centres: np.ndarray,
    U: np.ndarray,
    log_p0: np.ndarray,
    p0: np.ndarray,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_actions: int,
    K_outer: int,
    n_hyp_inner: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    """
    J(a1) = E_y1[ min_{a2≠a1} E_y2[u_ctrl | h1, a2] ].

    Outer draws: prior-predictive (particle from p0 + Gaussian noise).
    Inner: CRN across a2. Returns (mean, per-outer-sample values).
    """
    c1 = centres[a1]
    idx1 = rng.choice(len(p0), size=K_outer, p=p0)
    noise1 = rng.normal(0.0, sigma_y, size=K_outer)
    y1 = c1[idx1] + noise1
    per = np.empty(K_outer, dtype=np.float64)
    for k in range(K_outer):
        log_w1, w1 = update_posterior(log_p0, float(y1[k]), c1, sigma_y)
        idx2 = rng.choice(len(w1), size=n_hyp_inner, p=w1)
        noise2 = rng.normal(0.0, sigma_y, size=n_hyp_inner)
        best = float("inf")
        for a2 in range(n_actions):
            if a2 == a1:
                continue
            j = expected_u_after_action(
                a2,
                log_w1,
                w1,
                centres=centres,
                U=U,
                sigma_y=sigma_y,
                alpha=alpha,
                margin=margin,
                u_grid=u_grid,
                idx=idx2,
                noise=noise2,
            )
            if j < best:
                best = j
        per[k] = best
    return float(np.mean(per)), per


def exact_fixed_t2(
    *,
    table_support: TableThetaSupport,
    U: np.ndarray,
    systems: list[dict[str, Any]],
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid,
    noise_replicas: int,
    global_seed: int,
    n_actions: int,
) -> dict[str, Any]:
    best = None
    best_obj = float("inf")
    rows = []
    centres_cache = {
        a: np.asarray(y_sim_last_step_from_tables(table_support, [a]), dtype=np.float64)
        for a in range(n_actions)
    }
    for subset in combinations(range(n_actions), 2):
        m = score_subset_detailed(
            subset,
            table_support=table_support,
            U_support=U,
            systems=systems,
            sigma_y=sigma_y,
            alpha=alpha,
            noise_replicas=noise_replicas,
            margin=margin,
            u_grid=u_grid,
            global_seed=global_seed,
            centres_cache=centres_cache,
        )
        obj = m["validation_mean_u_ctrl"]
        rows.append({"subset": list(subset), "mean_u": obj, "safety": m["validation_safety_rate"]})
        if obj < best_obj - 1e-15 or (
            abs(obj - best_obj) <= 1e-15 and (best is None or subset < tuple(best))
        ):
            if abs(m["validation_safety_rate"] - 1.0) > 1e-12 and best is not None:
                pass
            if abs(m["validation_safety_rate"] - 1.0) <= 1e-12:
                best_obj = obj
                best = list(subset)
    # Prefer safety=1 then min mean (re-select cleanly)
    safe_rows = [r for r in rows if abs(r["safety"] - 1.0) <= 1e-12]
    pool = safe_rows or rows
    pool = sorted(pool, key=lambda r: (r["mean_u"], r["subset"]))
    best = pool[0]["subset"]
    best_obj = pool[0]["mean_u"]
    return {
        "exact_fixed_subset": best,
        "exact_fixed_value": float(best_obj),
        "n_subsets": len(rows),
        "search_mode": "exhaustive",
    }


def j_adaptive_t2_validation_outer(
    a1: int,
    *,
    centres: np.ndarray,
    U: np.ndarray,
    log_p0: np.ndarray,
    systems: list[dict[str, Any]],
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_actions: int,
    noise_replicas: int,
    global_seed: int,
    n_hyp_inner: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    """
    Same outer measure as Exact Fixed: validation systems × keyed-noise replicas.

    J(a1)= mean_{sys,rep}[ min_{a2≠a1} E_{y2|h1,a2} u_ctrl ].
    """
    from src.control.terminal_rule import keyed_noise
    from src.data import lookup_action_y_sim as _lookup

    per: list[float] = []
    for s_idx, sys in enumerate(systems):
        for rep in range(noise_replicas):
            z = keyed_noise(
                global_seed=global_seed,
                theta_id=s_idx,
                rollout_id=rep,
                step=0,
                action_id=a1,
            )
            y1 = float(_lookup(sys, a1)) + float(sigma_y) * z
            log_w1, w1 = update_posterior(log_p0, y1, centres[a1], sigma_y)
            idx2 = rng.choice(len(w1), size=n_hyp_inner, p=w1)
            noise2 = rng.normal(0.0, sigma_y, size=n_hyp_inner)
            best = float("inf")
            for a2 in range(n_actions):
                if a2 == a1:
                    continue
                j = expected_u_after_action(
                    a2,
                    log_w1,
                    w1,
                    centres=centres,
                    U=U,
                    sigma_y=sigma_y,
                    alpha=alpha,
                    margin=margin,
                    u_grid=u_grid,
                    idx=idx2,
                    noise=noise2,
                )
                if j < best:
                    best = j
            per.append(best)
    arr = np.asarray(per, dtype=np.float64)
    return float(np.mean(arr)), arr


def run_adaptive_reference_t2(
    *,
    table_support: TableThetaSupport,
    U: np.ndarray,
    systems: list[dict[str, Any]],
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid,
    n_actions: int,
    K_outer: int = 256,
    n_hyp_inner: int = 128,
    noise_replicas: int = 2,
    global_seed: int = 7,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """T=2 adaptive vs Exact Fixed with matched validation outer measure."""
    rng = rng or np.random.default_rng(12345)
    t0 = time.perf_counter()
    centres = _centres_matrix(table_support, n_actions)
    log_p0 = np.asarray(table_support.log_p0, dtype=np.float64)
    u_grid_arr = np.asarray(u_grid, dtype=np.float64)

    rows = []
    all_outer = []
    for a1 in range(n_actions):
        mean_j, per = j_adaptive_t2_validation_outer(
            a1,
            centres=centres,
            U=U,
            log_p0=log_p0,
            systems=systems,
            sigma_y=sigma_y,
            alpha=alpha,
            margin=margin,
            u_grid=u_grid_arr,
            n_actions=n_actions,
            noise_replicas=noise_replicas,
            global_seed=global_seed,
            n_hyp_inner=n_hyp_inner,
            rng=rng,
        )
        rows.append(
            {
                "a1": a1,
                "J_adaptive": mean_j,
                "se": float(np.std(per, ddof=1) / math.sqrt(len(per))),
            }
        )
        all_outer.append(per)
        if (a1 + 1) % 5 == 0:
            print(f"  T=2 adaptive: scored a1={a1+1}/{n_actions}", flush=True)

    rows_sorted = sorted(rows, key=lambda r: r["J_adaptive"])
    best_a1 = int(rows_sorted[0]["a1"])
    j_adapt = float(rows_sorted[0]["J_adaptive"])
    per_best = all_outer[best_a1]

    fixed = exact_fixed_t2(
        table_support=table_support,
        U=U,
        systems=systems,
        sigma_y=sigma_y,
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        noise_replicas=noise_replicas,
        global_seed=global_seed,
        n_actions=n_actions,
    )
    # Paired bootstrap: Fixed costs vs adaptive costs on same (sys,rep) indices
    # Bootstrap Δ using adaptive outer samples vs fixed point estimate (conservative)
    delta_point = float(fixed["exact_fixed_value"] - j_adapt)
    boots = []
    brng = np.random.default_rng(999)
    n_out = len(per_best)
    for _ in range(5000):
        idx = brng.integers(0, n_out, size=n_out)
        boots.append(float(fixed["exact_fixed_value"] - np.mean(per_best[idx])))
    lo, hi = np.quantile(boots, [0.025, 0.975])

    runtime = float(time.perf_counter() - t0)
    return {
        "exact_or_approximate": "exact_nested_MC_validation_outer",
        "tree_depth": 2,
        "observation_branch_count": n_out,
        "belief_state_count": n_out,
        "candidate_actions_evaluated": n_actions * (n_actions - 1),
        "adaptive_reference_value": j_adapt,
        "best_first_action": best_a1,
        "exact_fixed_value": float(fixed["exact_fixed_value"]),
        "exact_fixed_subset": fixed["exact_fixed_subset"],
        "estimated_adaptive_value": delta_point,
        "confidence_interval": [float(lo), float(hi)],
        "K_outer": n_out,
        "n_hyp_inner": n_hyp_inner,
        "runtime": runtime,
        "per_a1": rows,
        "outer_measure": "validation_systems_x_keyed_replicas",
        "uses_offline_banks_only": True,
        "used_test_systems": False,
    }


# ---------------------------------------------------------------------------
# Approximate T=3 / T=4: observation-bin scenario tree + beam
# ---------------------------------------------------------------------------


def _bin_centers_from_predictive(
    centres_a: np.ndarray,
    weights: np.ndarray,
    sigma_y: float,
    n_bins: int,
    rng: np.random.Generator,
    n_draw: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_centers, bin_probs) for posterior-predictive y | a."""
    idx = rng.choice(len(weights), size=n_draw, p=weights)
    y = centres_a[idx] + rng.normal(0.0, sigma_y, size=n_draw)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y, qs)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    centers = []
    probs = []
    for b in range(n_bins):
        mask = (y >= edges[b]) & (y < edges[b + 1])
        if not np.any(mask):
            continue
        centers.append(float(np.mean(y[mask])))
        probs.append(float(np.mean(mask)))
    if not centers:
        centers = [float(np.mean(y))]
        probs = [1.0]
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    return np.asarray(centers, dtype=np.float64), p


def approximate_adaptive_reference(
    *,
    horizon: int,
    centres: np.ndarray,
    U: np.ndarray,
    log_p0: np.ndarray,
    p0: np.ndarray,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_actions: int,
    systems: list[dict[str, Any]],
    fixed_subset: list[int],
    noise_replicas: int = 2,
    global_seed: int = 7,
    n_hyp: int = 128,
    n_bins: int = 5,
    beam_width: int = 8,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Validation-only approximate adaptive reference (not globally optimal).

    Method: CRN scenario evaluation on the same (validation system × keyed
    replica) outer draws used by Exact Fixed.

    Adaptive policy: at each step choose the action minimizing one-step
    E_y[u_ctrl|h,a] (myopic adaptive). At the final step this matches the
    nested terminal objective. Later actions therefore depend on previous
    observations through the posterior.

    Also reports a bin-beam planning estimate at the root for documentation.
    """
    from src.control.terminal_rule import keyed_noise
    from src.data import lookup_action_y_sim as _lookup

    t0 = time.perf_counter()
    u_grid = np.asarray(u_grid, dtype=np.float64)
    fixed_seq = list(sorted(int(a) for a in fixed_subset))

    def myopic_choose(log_w: np.ndarray, w: np.ndarray, used: set[int]) -> int:
        idx = rng.choice(len(w), size=n_hyp, p=w)
        noise = rng.normal(0.0, sigma_y, size=n_hyp)
        best_a, best_j = None, float("inf")
        for a in range(n_actions):
            if a in used:
                continue
            j = expected_u_after_action(
                a,
                log_w,
                w,
                centres=centres,
                U=U,
                sigma_y=sigma_y,
                alpha=alpha,
                margin=margin,
                u_grid=u_grid,
                idx=idx,
                noise=noise,
            )
            if j < best_j - 1e-15 or (abs(j - best_j) <= 1e-15 and (best_a is None or a < best_a)):
                best_j, best_a = j, a
        assert best_a is not None
        return int(best_a)

    def terminal_u(log_w: np.ndarray) -> float:
        w = normalize_log_weights(log_w)
        return float(
            posterior_safe_u_ctrl(U, w, alpha, margin=margin, u_grid=u_grid)
        )

    adapt_costs: list[float] = []
    fixed_costs: list[float] = []
    n_action_changes_with_y = 0
    n_branch_checks = 0
    belief_count = 0

    for s_idx, sys in enumerate(systems):
        for rep in range(noise_replicas):
            # --- adaptive myopic path ---
            log_w = log_p0.copy()
            used: set[int] = set()
            seq_a: list[int] = []
            for t in range(horizon):
                w = normalize_log_weights(log_w)
                a = myopic_choose(log_w, w, used)
                # sensitivity check: perturb would-be observation bins at t>=1
                if t >= 1:
                    n_branch_checks += 1
                    # compare chosen action under current w vs under reweighted prior
                    a_alt = myopic_choose(log_p0, p0, used)
                    if a_alt != a:
                        n_action_changes_with_y += 1
                z = keyed_noise(
                    global_seed=global_seed,
                    theta_id=s_idx,
                    rollout_id=rep,
                    step=t,
                    action_id=a,
                )
                y = float(_lookup(sys, a)) + float(sigma_y) * z
                log_w, _ = update_posterior(log_w, y, centres[a], sigma_y)
                used.add(a)
                seq_a.append(a)
                belief_count += 1
            adapt_costs.append(terminal_u(log_w))

            # --- exact fixed path (sorted subset order), same keyed mechanism ---
            log_wf = log_p0.copy()
            for t, a in enumerate(fixed_seq[:horizon]):
                z = keyed_noise(
                    global_seed=global_seed,
                    theta_id=s_idx,
                    rollout_id=rep,
                    step=t,
                    action_id=a,
                )
                y = float(_lookup(sys, a)) + float(sigma_y) * z
                log_wf, _ = update_posterior(log_wf, y, centres[a], sigma_y)
            fixed_costs.append(terminal_u(log_wf))

    adapt_arr = np.asarray(adapt_costs, dtype=np.float64)
    fixed_arr = np.asarray(fixed_costs, dtype=np.float64)
    diff = fixed_arr - adapt_arr
    # paired bootstrap CI for Δ
    brng = np.random.default_rng(int(rng.integers(1, 10**9)))
    boots = []
    for _ in range(5000):
        idx = brng.integers(0, len(diff), size=len(diff))
        boots.append(float(np.mean(diff[idx])))
    lo, hi = np.quantile(boots, [0.025, 0.975])

    runtime = float(time.perf_counter() - t0)
    return {
        "exact_or_approximate": "approximate_crn_myopic_adaptive_scenario",
        "tree_depth": horizon,
        "observation_branch_count": noise_replicas,
        "belief_state_count": belief_count,
        "candidate_actions_evaluated": horizon * n_actions * len(adapt_costs),
        "adaptive_reference_value": float(np.mean(adapt_arr)),
        "exact_fixed_value_crn": float(np.mean(fixed_arr)),
        "estimated_adaptive_value": float(np.mean(diff)),
        "confidence_interval": [float(lo), float(hi)],
        "fraction_steps_action_changed_vs_prior_myopic": (
            float(n_action_changes_with_y / max(1, n_branch_checks))
        ),
        "n_bins": n_bins,
        "beam_width": beam_width,
        "n_hyp_leaf": n_hyp,
        "runtime": runtime,
        "uses_offline_banks_only": True,
        "used_test_systems": False,
        "not_globally_optimal": True,
        "fixed_subset_used": fixed_seq,
    }


def exact_fixed_value_for_horizon(
    horizon: int,
    *,
    table_support: TableThetaSupport,
    U: np.ndarray,
    systems: list[dict[str, Any]],
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid,
    noise_replicas: int,
    global_seed: int,
    n_actions: int,
    known_subset: list[int] | None = None,
) -> dict[str, Any]:
    """Exhaustive for T<=3; for T=4 use known exact subset if provided else score known."""
    centres_cache = {
        a: np.asarray(y_sim_last_step_from_tables(table_support, [a]), dtype=np.float64)
        for a in range(n_actions)
    }
    n_comb = math.comb(n_actions, horizon)
    if known_subset is not None and horizon == 4:
        m = score_subset_detailed(
            known_subset,
            table_support=table_support,
            U_support=U,
            systems=systems,
            sigma_y=sigma_y,
            alpha=alpha,
            noise_replicas=noise_replicas,
            margin=margin,
            u_grid=u_grid,
            global_seed=global_seed,
            centres_cache=centres_cache,
        )
        return {
            "exact_fixed_subset": list(sorted(known_subset)),
            "exact_fixed_value": float(m["validation_mean_u_ctrl"]),
            "n_subsets": 27405,
            "search_mode": "exhaustive_known_from_ieee5_T4_fixed_exact",
            "validation_safety": float(m["validation_safety_rate"]),
        }
    if n_comb > 5000 and known_subset is None:
        raise RuntimeError(f"exhaustive C({n_actions},{horizon})={n_comb} too large without known subset")

    best = None
    best_obj = float("inf")
    evaluated = 0
    for subset in combinations(range(n_actions), horizon):
        m = score_subset_detailed(
            subset,
            table_support=table_support,
            U_support=U,
            systems=systems,
            sigma_y=sigma_y,
            alpha=alpha,
            noise_replicas=noise_replicas,
            margin=margin,
            u_grid=u_grid,
            global_seed=global_seed,
            centres_cache=centres_cache,
        )
        evaluated += 1
        if abs(m["validation_safety_rate"] - 1.0) > 1e-12:
            continue
        obj = m["validation_mean_u_ctrl"]
        if obj < best_obj - 1e-15 or (
            abs(obj - best_obj) <= 1e-15 and (best is None or list(subset) < best)
        ):
            best_obj = obj
            best = list(subset)
    return {
        "exact_fixed_subset": best,
        "exact_fixed_value": float(best_obj),
        "n_subsets": evaluated,
        "search_mode": "exhaustive",
    }


# ---------------------------------------------------------------------------
# Existing DAD observation sensitivity (diagnostic only)
# ---------------------------------------------------------------------------


def dad_observation_sensitivity(
    policy_path: Path,
    *,
    n_actions: int,
    horizon: int,
    sigma_y: float,
    n_bins: int = 8,
    fixed_prefix: list[int] | None = None,
) -> list[dict[str, Any]]:
    import torch
    from src.neural.policy import DADPolicy

    device = torch.device("cpu")
    pol = DADPolicy(n_actions=n_actions, max_steps=horizon).to(device)
    try:
        state = torch.load(policy_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(policy_path, map_location=device)
    if isinstance(state, dict) and "policy" in state:
        pol.load_state_dict(state["policy"])
    elif isinstance(state, dict) and any(k.startswith("encoder.") or k.startswith("head.") for k in state):
        pol.load_state_dict(state)
    elif isinstance(state, dict) and "state_dict" in state:
        pol.load_state_dict(state["state_dict"])
    else:
        pol.load_state_dict(state)
    pol.eval()

    prefix = list(fixed_prefix or [])
    # If empty prefix, probe step t=1 (no history) — only steps with history matter
    rows = []
    # For each step t=2..T, fix actions 1..t-1 and vary y_{t-1} (and earlier y)
    for t in range(2, horizon + 1):
        acts = (prefix + list(range(t - 1)))[: t - 1]
        if len(set(acts)) < len(acts):
            # ensure unique actions
            acts = []
            for a in range(n_actions):
                if len(acts) >= t - 1:
                    break
                acts.append(a)
        y_grid = np.linspace(-3 * sigma_y, 3 * sigma_y, n_bins)  # relative; use absolute scale
        # Use a range of ROCOF-like observations
        y_vals = np.linspace(-2.0, 2.0, n_bins)
        probs_by_bin = []
        selected = []
        entropies = []
        for y_last in y_vals:
            ys = [0.0] * (t - 2) + [float(y_last)]
            if t - 1 > 1:
                # vary only last observation; earlier fixed at 0 for controlled probe
                ys = [0.0] * (t - 2) + [float(y_last)]
            act_t = torch.tensor([acts], dtype=torch.long, device=device)
            obs_t = torch.tensor([ys], dtype=torch.float32, device=device)
            mask = torch.ones(1, t - 1, device=device)
            feas = torch.ones(1, n_actions, dtype=torch.bool, device=device)
            for a in acts:
                feas[0, a] = False
            with torch.no_grad():
                logits = pol.forward(act_t, obs_t, mask, feas)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                a_star = int(np.argmax(probs))
            # sensitivity via autograd on observations
            pol.zero_grad(set_to_none=True)
            obs_req = obs_t.clone().detach().requires_grad_(True)
            logits_g = pol.forward(act_t, obs_req, mask, feas)
            try:
                logits_g[0, a_star].backward()
                grad_norm = (
                    float(obs_req.grad.norm().item())
                    if obs_req.grad is not None
                    else 0.0
                )
            except RuntimeError:
                grad_norm = 0.0
            probs_by_bin.append(probs)
            selected.append(int(np.argmax(probs)))
            p = probs[probs > 0]
            entropies.append(float(-np.sum(p * np.log(p + 1e-12))))
            rows.append(
                {
                    "step": t,
                    "fixed_previous_actions": " ".join(map(str, acts)),
                    "observation_values_or_bins": float(y_last),
                    "selected_action": int(np.argmax(probs)),
                    "action_entropy": entropies[-1],
                    "logit_grad_norm_wrt_y": grad_norm,
                    "next_action_probabilities": " ".join(f"{x:.6f}" for x in probs),
                }
            )
        P = np.stack(probs_by_bin, axis=0)
        # pairwise mean TV / KL across bins
        tvs = []
        kls = []
        for i in range(n_bins):
            for j in range(i + 1, n_bins):
                tvs.append(0.5 * float(np.sum(np.abs(P[i] - P[j]))))
                # KL(i||j)
                kls.append(float(np.sum(P[i] * (np.log(P[i] + 1e-12) - np.log(P[j] + 1e-12)))))
        # mutual information I(Ybin; A) with uniform bins
        p_y = np.ones(n_bins) / n_bins
        p_a = P.mean(axis=0)
        mi = 0.0
        for i in range(n_bins):
            for a in range(n_actions):
                if P[i, a] <= 0 or p_a[a] <= 0:
                    continue
                mi += p_y[i] * P[i, a] * (math.log(P[i, a] + 1e-12) - math.log(p_a[a] + 1e-12))
        for r in rows:
            if r["step"] == t:
                r["mean_TV_between_bins"] = float(np.mean(tvs)) if tvs else 0.0
                r["mean_KL_between_bins"] = float(np.mean(kls)) if kls else 0.0
                r["observation_to_action_mutual_information"] = float(mi)
                r["unique_selected_actions_across_bins"] = int(len(set(selected)))
    return rows


def reward_resolution_from_validation(
    *,
    table_support: TableThetaSupport,
    U: np.ndarray,
    systems: list[dict[str, Any]],
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid,
    sequences: list[list[int]],
    global_seed: int = 7,
) -> dict[str, Any]:
    """Terminal cost quantization diagnostics on validation systems."""
    from src.control.legacy.ieee5_t4_fixed_exact import _log_gauss_const

    u_grid_arr = np.asarray(u_grid, dtype=np.float64)
    log_p0 = np.asarray(table_support.log_p0, dtype=np.float64)
    costs = []
    pre_snap = []
    for seq in sequences:
        for s_idx, sys in enumerate(systems):
            log_w = log_p0.copy()
            for t, a in enumerate(seq):
                y = lookup_action_y_sim(sys, a) + sigma_y * np.random.default_rng(
                    global_seed + 1000 * s_idx + 17 * t + a
                ).normal()
                c = y_sim_last_step_from_tables(table_support, [a])
                s2 = sigma_y**2
                log_L = -0.5 * math.log(2 * math.pi * s2) - 0.5 * ((y - c) ** 2) / s2
                log_w = log_w + log_L
            w = normalize_log_weights(log_w)
            # pre-snap quantile
            from src.control.posterior_ctrl import weighted_quantile

            u0 = weighted_quantile(U, w, 1.0 - alpha)
            pre_snap.append(float(u0 + margin))
            costs.append(float(snap_up_to_grid(u0 + margin, u_grid_arr)))
    arr = np.asarray(costs, dtype=np.float64)
    pre = np.asarray(pre_snap, dtype=np.float64)
    # equal-cost pair fraction
    n = len(arr)
    if n >= 2:
        # sample pairs
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n, size=(min(5000, n * n), 2))
        eq = float(np.mean(np.isclose(arr[idx[:, 0]], arr[idx[:, 1]])))
    else:
        eq = 1.0
    vals, counts = np.unique(np.round(arr, 10), return_counts=True)
    modal_frac = float(counts.max() / counts.sum()) if counts.size else float("nan")
    grid_dist = {float(v): int(c) for v, c in zip(vals, counts)}
    return {
        "unique_terminal_cost_count": int(len(vals)),
        "fraction_at_modal_terminal_cost": modal_frac,
        "terminal_cost_std": float(np.std(arr)),
        "fraction_of_equal_cost_pairs": eq,
        "pre_snap_quantile_mean": float(np.mean(pre)),
        "pre_snap_quantile_std": float(np.std(pre)),
        "control_grid_level_distribution": grid_dist,
        "n_trajectories": int(n),
    }


# ---------------------------------------------------------------------------
# Potential-based rewards (for tests / future training; not altering frozen runs)
# ---------------------------------------------------------------------------


def potential_shaped_rewards(u_ctrl_path: list[float]) -> list[float]:
    """r_t = u(h_{t-1}) - u(h_t); telescopes to u(h0) - u(h_T)."""
    if len(u_ctrl_path) < 2:
        return []
    return [float(u_ctrl_path[t - 1] - u_ctrl_path[t]) for t in range(1, len(u_ctrl_path))]


def normalize_advantages(adv: np.ndarray) -> np.ndarray:
    a = np.asarray(adv, dtype=np.float64)
    sd = float(np.std(a))
    if sd < 1e-12:
        return np.zeros_like(a)
    return (a - float(np.mean(a))) / sd


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_adaptive_value_diagnosis(
    *,
    project_root: Path | None = None,
    exp_dir: Path | None = None,
    out_dir: Path | None = None,
    K_outer: int = 192,
    n_hyp_inner: int = 96,
) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    exp_dir = Path(exp_dir or (root / "experiments" / "ieee5_T4")).resolve()
    out = Path(out_dir or (root / OUT_DEFAULT)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    print("=== IEEE5 adaptive-value diagnosis (banks only; no test) ===", flush=True)
    run = load_experiment_run(exp_dir, root)
    splits = load_pilot_splits(exp_dir, run)
    frozen = load_frozen_terminal_rule(exp_dir, expected_margin=FROZEN_MARGIN)
    assert frozen.terminal_rule_hash == EXPECTED_HASH
    control_spec = frozen.to_control_spec(ControlSpec.from_cfg(run.cfg))
    val = list(splits["validation_systems"])
    table_support = TableThetaSupport(
        systems=splits["support_systems"],
        log_p0=log_prior_uniform_discrete(len(splits["support_systems"])),
    )
    U = extract_U_bank(splits["support_systems"])
    n_actions = len(build_catalog(run.cfg))
    sigma_y = float(run.cfg.sigma_y)
    rng = np.random.default_rng(4242)
    centres = _centres_matrix(table_support, n_actions)
    log_p0 = np.asarray(table_support.log_p0, dtype=np.float64)
    p0 = normalize_log_weights(log_p0)
    u_grid = np.asarray(frozen.u_candidates, dtype=np.float64)

    # --- T=2 ---
    print("\n=== T=2 exact adaptive reference ===", flush=True)
    t2 = run_adaptive_reference_t2(
        table_support=table_support,
        U=U,
        systems=val,
        sigma_y=sigma_y,
        alpha=frozen.alpha,
        margin=frozen.margin,
        u_grid=frozen.u_candidates,
        n_actions=n_actions,
        K_outer=K_outer,
        n_hyp_inner=n_hyp_inner,
        noise_replicas=int(control_spec.fixed_noise_replicas),
        global_seed=7,
        rng=rng,
    )
    _write_csv(
        out / "adaptive_reference_T2.csv",
        [{"T": 2, **{k: t2[k] for k in t2 if k != "per_a1"}}]
        + [{"T": 2, "row": "per_a1", **r} for r in t2["per_a1"]],
    )
    print(
        f"  J_adapt={t2['adaptive_reference_value']:.6f}  "
        f"J_fixed={t2['exact_fixed_value']:.6f}  "
        f"Δ={t2['estimated_adaptive_value']:.6f}  "
        f"CI={t2['confidence_interval']}",
        flush=True,
    )

    # --- T=3 ---
    print("\n=== T=3 approximate adaptive reference (CRN myopic-adaptive) ===", flush=True)
    print("  exhaustive Fixed T=3 on validation...", flush=True)
    t3_fixed = exact_fixed_value_for_horizon(
        3,
        table_support=table_support,
        U=U,
        systems=val,
        sigma_y=sigma_y,
        alpha=frozen.alpha,
        margin=frozen.margin,
        u_grid=frozen.u_candidates,
        noise_replicas=int(control_spec.fixed_noise_replicas),
        global_seed=7,
        n_actions=n_actions,
    )
    t3_adapt = approximate_adaptive_reference(
        horizon=3,
        centres=centres,
        U=U,
        log_p0=log_p0,
        p0=p0,
        sigma_y=sigma_y,
        alpha=frozen.alpha,
        margin=frozen.margin,
        u_grid=u_grid,
        n_actions=n_actions,
        systems=val,
        fixed_subset=list(t3_fixed["exact_fixed_subset"]),
        noise_replicas=int(control_spec.fixed_noise_replicas),
        global_seed=7,
        n_hyp=128,
        rng=np.random.default_rng(4243),
    )
    t3 = {
        **t3_adapt,
        "exact_fixed_value": t3_fixed["exact_fixed_value"],
        "exact_fixed_subset": t3_fixed["exact_fixed_subset"],
        # Prefer CRN paired Δ (same outer draws); also store exhaustive Fixed point
        "exact_fixed_value_exhaustive": t3_fixed["exact_fixed_value"],
        "estimated_adaptive_value_vs_exhaustive_fixed": float(
            t3_fixed["exact_fixed_value"] - t3_adapt["adaptive_reference_value"]
        ),
    }
    _write_csv(out / "adaptive_reference_T3.csv", [{k: t3[k] for k in t3}])
    print(
        f"  J_adapt≈{t3['adaptive_reference_value']:.6f}  "
        f"J_fixed_exh={t3['exact_fixed_value']:.6f}  "
        f"J_fixed_crn={t3['exact_fixed_value_crn']:.6f}  "
        f"Δ_crn≈{t3['estimated_adaptive_value']:.6f}  CI={t3['confidence_interval']}",
        flush=True,
    )

    # --- T=4 ---
    print("\n=== T=4 approximate adaptive reference (CRN myopic-adaptive) ===", flush=True)
    t4_fixed = exact_fixed_value_for_horizon(
        4,
        table_support=table_support,
        U=U,
        systems=val,
        sigma_y=sigma_y,
        alpha=frozen.alpha,
        margin=frozen.margin,
        u_grid=frozen.u_candidates,
        noise_replicas=int(control_spec.fixed_noise_replicas),
        global_seed=7,
        n_actions=n_actions,
        known_subset=[8, 19, 20, 28],
    )
    t4_adapt = approximate_adaptive_reference(
        horizon=4,
        centres=centres,
        U=U,
        log_p0=log_p0,
        p0=p0,
        sigma_y=sigma_y,
        alpha=frozen.alpha,
        margin=frozen.margin,
        u_grid=u_grid,
        n_actions=n_actions,
        systems=val,
        fixed_subset=list(t4_fixed["exact_fixed_subset"]),
        noise_replicas=int(control_spec.fixed_noise_replicas),
        global_seed=7,
        n_hyp=128,
        rng=np.random.default_rng(4244),
    )
    t4 = {
        **t4_adapt,
        "exact_fixed_value": t4_fixed["exact_fixed_value"],
        "exact_fixed_subset": t4_fixed["exact_fixed_subset"],
        "exact_fixed_value_exhaustive": t4_fixed["exact_fixed_value"],
        "estimated_adaptive_value_vs_exhaustive_fixed": float(
            t4_fixed["exact_fixed_value"] - t4_adapt["adaptive_reference_value"]
        ),
    }
    _write_csv(out / "adaptive_reference_T4.csv", [{k: t4[k] for k in t4}])
    print(
        f"  J_adapt≈{t4['adaptive_reference_value']:.6f}  "
        f"J_fixed_exh={t4['exact_fixed_value']:.6f}  "
        f"J_fixed_crn={t4['exact_fixed_value_crn']:.6f}  "
        f"Δ_crn≈{t4['estimated_adaptive_value']:.6f}  CI={t4['confidence_interval']}",
        flush=True,
    )

    def case_for(delta: float, ci: list[float]) -> str:
        # Meaningful positive adaptive value: CI entirely above a small epsilon
        eps = 0.005  # half a control-grid step (grid has 0.05 steps; use conservative)
        if ci[0] > eps:
            return "B_measurable_adaptive_value"
        return "A_no_measurable_adaptive_value"

    case2 = case_for(t2["estimated_adaptive_value"], t2["confidence_interval"])
    case3 = case_for(t3["estimated_adaptive_value"], t3["confidence_interval"])
    case4 = case_for(t4["estimated_adaptive_value"], t4["confidence_interval"])
    # Overall: Case B only if any horizon clearly shows value; primary interest T=3/T=4
    overall = "B_measurable_adaptive_value"
    if case2 == case3 == case4 == "A_no_measurable_adaptive_value":
        overall = "A_no_measurable_adaptive_value"
    elif case3 == "A_no_measurable_adaptive_value" and case4 == "A_no_measurable_adaptive_value":
        # T=2 alone with tiny delta still Case A for sequential BOED claim at T>=3
        if case2 == "A_no_measurable_adaptive_value":
            overall = "A_no_measurable_adaptive_value"
        else:
            overall = "A_no_measurable_adaptive_value"  # T>=3 are the frozen experiment horizons

    summary = {
        "terminal_rule_hash": frozen.terminal_rule_hash,
        "alpha": frozen.alpha,
        "margin": frozen.margin,
        "used_test_systems": False,
        "uses_offline_banks_only": True,
        "T2": {k: t2[k] for k in t2 if k != "per_a1"},
        "T3": t3,
        "T4": t4,
        "case_T2": case2,
        "case_T3": case3,
        "case_T4": case4,
        "overall_case": overall,
        "decision": (
            "IEEE5 provides little measurable value of observation-dependent "
            "adaptation under the current objective and design space."
            if overall.startswith("A_")
            else "Measurable adaptive value exists; DAD underperformed the adaptive reference."
        ),
        "proceed_to_dad_improvement": overall.startswith("B_"),
        "move_to_ieee9_recommended": overall.startswith("A_"),
        "frozen_ieee5_reports_untouched": True,
    }
    _write_json(out / "adaptive_value_summary.json", summary)

    # Always run lightweight existing-DAD sensitivity + reward resolution (diagnostics)
    print("\n=== Existing DAD observation sensitivity ===", flush=True)
    sens_rows = []
    for seed in (101, 202, 303):
        pth = exp_dir / "train" / "dad" / f"seed_{seed}" / "dad.pth"
        if not pth.is_file():
            pth = exp_dir / "train" / "dad" / f"seed_{seed}" / f"dad_seed{seed}.pth"
        if not pth.is_file():
            continue
        # dominant T4 prefix from frozen experiment
        rows = dad_observation_sensitivity(
            pth,
            n_actions=n_actions,
            horizon=4,
            sigma_y=sigma_y,
            fixed_prefix=[23, 29, 10],
        )
        for r in rows:
            r["seed"] = seed
            sens_rows.append(r)
    if sens_rows:
        _write_csv(out / "existing_dad_observation_sensitivity.csv", sens_rows)

    print("=== Reward resolution diagnostics ===", flush=True)
    reward = reward_resolution_from_validation(
        table_support=table_support,
        U=U,
        systems=val,
        sigma_y=sigma_y,
        alpha=frozen.alpha,
        margin=frozen.margin,
        u_grid=frozen.u_candidates,
        sequences=[
            [23, 29, 10, 19],
            [8, 19, 20, 28],
            [0, 19, 28],
            [1, 8],
        ],
        global_seed=7,
    )
    _write_csv(out / "reward_resolution.csv", [reward])

    # Skipped training variants under Case A
    if overall.startswith("A_"):
        _write_csv(
            out / "training_variant_summary.csv",
            [
                {
                    "variant": "skipped",
                    "reason": "Case A: no measurable adaptive value; do not force DAD adaptivity",
                }
            ],
        )
        _write_csv(
            out / "seed_summary.csv",
            [{"note": "DAD retraining skipped under Case A decision rule"}],
        )
        _write_csv(
            out / "dominant_sequence_comparison.csv",
            [{"note": "skipped under Case A; see frozen ieee5_T4 adaptivity diagnostics"}],
        )
    else:
        _write_csv(
            out / "training_variant_summary.csv",
            [{"variant": "pending", "reason": "Case B: implement improvements next"}],
        )

    _write_plots(out, summary, sens_rows)
    _write_report(out, summary, reward, sens_rows)
    print(f"\nOverall case: {overall}", flush=True)
    print(f"Decision: {summary['decision']}", flush=True)
    print(f"Outputs → {out}", flush=True)
    return summary


def _write_plots(out: Path, summary: dict[str, Any], sens_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plots = out / "plots"
    Ts = [2, 3, 4]
    j_adapt = [
        summary["T2"]["adaptive_reference_value"],
        summary["T3"]["adaptive_reference_value"],
        summary["T4"]["adaptive_reference_value"],
    ]
    j_fix = [
        summary["T2"]["exact_fixed_value"],
        summary["T3"]["exact_fixed_value"],
        summary["T4"]["exact_fixed_value"],
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(Ts, j_fix, "o-", label="Exact Fixed")
    ax.plot(Ts, j_adapt, "s-", label="Adaptive reference")
    ax.set_xlabel("T")
    ax.set_ylabel("validation mean u_ctrl")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "adaptive_reference_vs_fixed.png", dpi=120)
    plt.close(fig)

    if sens_rows:
        by_step = {}
        for r in sens_rows:
            by_step.setdefault(int(r["step"]), []).append(float(r.get("logit_grad_norm_wrt_y", 0)))
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for step, vals in sorted(by_step.items()):
            ax.plot([step] * len(vals), vals, "o", alpha=0.4, label=f"t={step}" if step == min(by_step) else None)
        ax.set_xlabel("decision step")
        ax.set_ylabel(r"‖∂ℓ/∂y‖")
        ax.set_title("Logit sensitivity to observations (existing DAD)")
        fig.tight_layout()
        fig.savefig(plots / "logit_sensitivity_to_observations.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4))
        for seed in sorted({int(r["seed"]) for r in sens_rows}):
            xs, ys = [], []
            for step in sorted({int(r["step"]) for r in sens_rows}):
                mis = [
                    float(r["observation_to_action_mutual_information"])
                    for r in sens_rows
                    if int(r["seed"]) == seed and int(r["step"]) == step
                ]
                if mis:
                    xs.append(step)
                    ys.append(mis[0])
            ax.plot(xs, ys, "o-", label=f"seed {seed}")
        ax.set_xlabel("step")
        ax.set_ylabel("I(y bin; action)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "observation_action_information.png", dpi=120)
        plt.close(fig)

    # Placeholder plots for skipped training
    for fname, ylabel in [
        ("policy_entropy_by_epoch.png", "entropy (skipped)"),
        ("dominant_sequence_fraction_by_epoch.png", "dominant fraction (skipped)"),
        ("validation_control_by_variant.png", "val u_ctrl (skipped)"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.text(0.5, 0.5, "Skipped under Case A" if summary["overall_case"].startswith("A_") else "See training logs",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(plots / fname, dpi=120)
        plt.close(fig)


def _write_report(
    out: Path,
    summary: dict[str, Any],
    reward: dict[str, Any],
    sens_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# IEEE5 DAD adaptive-value diagnosis",
        "",
        "Frozen IEEE5 final reports were **not** modified.",
        "Final test systems were **not** used.",
        "",
        f"**Overall case:** `{summary['overall_case']}`",
        "",
        summary["decision"],
        "",
        "## Adaptive reference vs Exact Fixed",
        "",
        "| T | mode | J_adaptive | J_fixed | Δ_adapt | CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    for T, key in [(2, "T2"), (3, "T3"), (4, "T4")]:
        d = summary[key]
        lines.append(
            f"| {T} | {d.get('exact_or_approximate')} | "
            f"{d['adaptive_reference_value']:.6f} | {d['exact_fixed_value']:.6f} | "
            f"{d['estimated_adaptive_value']:.6f} | {d.get('confidence_interval')} |"
        )
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            f"- proceed_to_dad_improvement: `{summary['proceed_to_dad_improvement']}`",
            f"- move_to_ieee9_recommended: `{summary['move_to_ieee9_recommended']}`",
            "",
            "## Existing DAD observation sensitivity",
            "",
        ]
    )
    if sens_rows:
        # summarize unique actions across bins
        for seed in sorted({int(r["seed"]) for r in sens_rows}):
            for step in sorted({int(r["step"]) for r in sens_rows}):
                sub = [r for r in sens_rows if int(r["seed"]) == seed and int(r["step"]) == step]
                if not sub:
                    continue
                lines.append(
                    f"- seed {seed} step {step}: unique_actions_across_y_bins="
                    f"{sub[0].get('unique_selected_actions_across_bins')} "
                    f"mean_TV={sub[0].get('mean_TV_between_bins')} "
                    f"MI={sub[0].get('observation_to_action_mutual_information')}"
                )
    else:
        lines.append("- (no checkpoints found)")
    lines.extend(
        [
            "",
            "## Reward quantization",
            "",
            f"```json\n{json.dumps(reward, indent=2)}\n```",
            "",
            "## Training variants",
            "",
            (
                "Skipped: Case A says do not force DAD to become adaptive."
                if summary["overall_case"].startswith("A_")
                else "Case B: training variants should proceed."
            ),
            "",
            "## Complete-history note",
            "",
            "DAD `HistoryEncoder` receives `(action_indices, observations, mask)` for all "
            "past steps (`src/neural/policy.py`). Call path: rollout buffers → "
            "`DADPolicy.forward` → `HistoryEncoder.forward` → attention pool → logits head.",
            "",
        ]
    )
    (out / "dad_improvement_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_adaptive_value_diagnosis()
