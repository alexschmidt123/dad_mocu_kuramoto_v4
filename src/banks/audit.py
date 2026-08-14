"""Plan-2 bank structure audit: redundancy + non-myopic / adaptive room.

Goal: decide whether the physical bank + (N_obs, sigma, T) setting leaves enough
structure for amortized adaptive policies (DAD / RL-sBOED) to beat Myopic and
Fixed — before investing in MoE-sBOED innovation.

This is a read-only structural screen (no neural training). Lower J = better
(expected posterior-safe u_ctrl).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import SBOEDConfig, repo_root
from src.banks.power_grid import (
    load_bank_from_path,
    resolve_dataset_dir,
    system_name_from_cfg,
)
from src.control.posterior_ctrl import normalize_log_weights, posterior_safe_u_ctrl
from src.observations.compress import build_centres_bank
from src.observations.likelihood import vector_gaussian_loglik


DEFAULT_NEAR_DUP_CORR = 0.98
DEFAULT_ADVANTAGE_EPS = 1e-4
# Solution-1 tuning gate: planning must beat Fixed by at least this much (J units).
# Tiny gaps (~1e-3) previously "passed" but Fixed still won objective sweeps.
DEFAULT_MIN_FIXED_ADVANTAGE = 0.01
# History-contingent branching must also move terminal u_ctrl (not flat near-ties).
DEFAULT_MIN_MEAN_BRANCH_VALUE = 0.01
# Mode second-action mass above this ⇒ effectively open-loop.
DEFAULT_MAX_MODE_SECOND_ACTION_PROB = 0.75
# Multi-T: each extra probe should deepen planner advantage vs Fixed.
DEFAULT_MIN_GAP_IMPROVE_PER_HORIZON = 0.005
DEFAULT_STRUCTURE_AUDIT_HORIZONS = (2, 3, 4)
DEFAULT_MAX_FIXED_SUBSETS = 220


def _update(
    log_w: np.ndarray, y: np.ndarray, centres: np.ndarray, sigma: float
) -> np.ndarray:
    return log_w + vector_gaussian_loglik(y, centres, sigma)


def _terminal_u(
    U: np.ndarray,
    log_w: np.ndarray,
    *,
    alpha: float,
    margin: float,
    grid: np.ndarray,
) -> float:
    return float(
        posterior_safe_u_ctrl(
            U,
            normalize_log_weights(log_w),
            alpha,
            margin=margin,
            u_grid=grid,
            snap_up=True,
        )
    )


def _posterior_indices(log_w: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(normalize_log_weights(log_w))
    return np.searchsorted(cdf, np.minimum(uniforms, 1.0 - 1e-12), side="left")


def _one_step_scores(
    Y: np.ndarray,
    U: np.ndarray,
    log_w: np.ndarray,
    used: set[int],
    *,
    sigma: float,
    alpha: float,
    margin: float,
    grid: np.ndarray,
    uniforms: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    scores = np.full(Y.shape[1], np.inf, dtype=np.float64)
    idx = _posterior_indices(log_w, uniforms)
    for a in range(Y.shape[1]):
        if a in used:
            continue
        vals: list[float] = []
        centres = Y[:, a, :]
        for j, n in enumerate(idx):
            y = centres[int(n)] + noise[j]
            vals.append(
                _terminal_u(
                    U,
                    _update(log_w, y, centres, sigma),
                    alpha=alpha,
                    margin=margin,
                    grid=grid,
                )
            )
        scores[a] = float(np.mean(vals))
    return scores


def _parse_design_meta(
    catalog_designs: list[Any] | None, n_actions: int
) -> tuple[list[int | None], list[float | None]]:
    buses: list[int | None] = [None] * n_actions
    amps: list[float | None] = [None] * n_actions
    if not catalog_designs:
        return buses, amps
    for i, d in enumerate(catalog_designs[:n_actions]):
        if isinstance(d, dict):
            buses[i] = int(d.get("bus", d.get("bus_location", -1)))
            amps[i] = float(d.get("amp", d.get("amplitude", float("nan"))))
        elif isinstance(d, (list, tuple)) and len(d) >= 2:
            amps[i] = float(d[0])
            buses[i] = int(d[1])
    return buses, amps


def analyze_action_redundancy(
    *,
    max_rocof: np.ndarray,
    catalog_designs: list[Any] | None,
    near_dup_corr: float = DEFAULT_NEAR_DUP_CORR,
) -> dict[str, Any]:
    """Pairwise action similarity on θ→max-|ROCOF| fingerprints.

    Also detects amplitude scale-redundancy: if same-bus designs remain
    |corr|≈1 after dividing ROCOF by amp, multiple amps add no ranking info.
    """
    R = np.asarray(max_rocof, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] < 2:
        return {
            "n_actions": int(R.shape[1]) if R.ndim == 2 else 0,
            "near_duplicate_frac": 0.0,
            "mean_abs_corr": 0.0,
            "max_abs_corr": 0.0,
            "same_bus_near_dup_frac": 0.0,
            "amp_scale_redundant": False,
            "n_unique_amps": 0,
            "n_unique_buses": 0,
            "top_near_duplicates": [],
        }
    n_theta, n_actions = R.shape
    buses, amps = _parse_design_meta(catalog_designs, n_actions)
    amp_vals = [a for a in amps if a is not None and np.isfinite(a)]
    bus_vals = [b for b in buses if b is not None]
    n_unique_amps = len({round(float(a), 8) for a in amp_vals})
    n_unique_buses = len(set(bus_vals))

    # Standardize per action across θ; corr of fingerprints.
    Z = R - R.mean(axis=0, keepdims=True)
    std = R.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    Z = Z / std
    corr = (Z.T @ Z) / max(n_theta - 1, 1)
    np.fill_diagonal(corr, 0.0)
    abs_corr = np.abs(corr)
    iu = np.triu_indices(n_actions, k=1)
    pair_vals = abs_corr[iu]
    near = pair_vals >= float(near_dup_corr)
    near_frac = float(np.mean(near)) if pair_vals.size else 0.0

    same_bus_near = 0
    same_bus_pairs = 0
    amp_norm_same_dur: list[float] = []
    durs: list[float | None] = [None] * n_actions
    if catalog_designs is not None:
        for i, d in enumerate(catalog_designs[:n_actions]):
            if isinstance(d, dict):
                durs[i] = float(d.get("duration", d.get("duration_s", float("nan"))))
            elif isinstance(d, (list, tuple)) and len(d) >= 3:
                durs[i] = float(d[2])
    for i in range(n_actions):
        for j in range(i + 1, n_actions):
            if buses[i] is None or buses[j] is None:
                continue
            if buses[i] != buses[j]:
                continue
            same_bus_pairs += 1
            if abs_corr[i, j] >= float(near_dup_corr):
                same_bus_near += 1
            # Amp scale-redundancy only among same-bus *same-duration* pairs.
            # Different durations are allowed waveform diversity, not pure scales.
            same_dur = (
                durs[i] is not None
                and durs[j] is not None
                and np.isfinite(durs[i])
                and np.isfinite(durs[j])
                and abs(float(durs[i]) - float(durs[j])) <= 1e-12
            )
            if (
                same_dur
                and amps[i] is not None
                and amps[j] is not None
                and float(amps[i]) > 1e-12
                and float(amps[j]) > 1e-12
            ):
                fi = R[:, i] / float(amps[i])
                fj = R[:, j] / float(amps[j])
                if float(np.std(fi)) > 1e-12 and float(np.std(fj)) > 1e-12:
                    amp_norm_same_dur.append(abs(float(np.corrcoef(fi, fj)[0, 1])))
    same_bus_frac = (
        float(same_bus_near / same_bus_pairs) if same_bus_pairs else 0.0
    )
    amp_norm_mean = (
        float(np.mean(amp_norm_same_dur)) if amp_norm_same_dur else 0.0
    )
    # Pure scale copies at fixed duration: multi-amp adds no ranking info.
    # If durations also vary, keep the flag informational; hard reject is optional.
    amp_scale_redundant = bool(
        n_unique_amps >= 2
        and len(amp_norm_same_dur) > 0
        and amp_norm_mean >= 0.99
    )

    order = np.argsort(-pair_vals)[:8]
    top: list[dict[str, Any]] = []
    for k in order:
        if pair_vals[k] < float(near_dup_corr) and len(top) >= 3:
            break
        i, j = int(iu[0][k]), int(iu[1][k])
        top.append(
            {
                "i": i,
                "j": j,
                "abs_corr": float(pair_vals[k]),
                "bus_i": buses[i],
                "bus_j": buses[j],
                "amp_i": amps[i],
                "amp_j": amps[j],
            }
        )

    return {
        "n_actions": int(n_actions),
        "n_theta": int(n_theta),
        "n_unique_amps": int(n_unique_amps),
        "n_unique_buses": int(n_unique_buses),
        "near_duplicate_threshold": float(near_dup_corr),
        "near_duplicate_frac": near_frac,
        "n_near_duplicate_pairs": int(near.sum()),
        "n_pairs": int(pair_vals.size),
        "mean_abs_corr": float(pair_vals.mean()) if pair_vals.size else 0.0,
        "max_abs_corr": float(pair_vals.max()) if pair_vals.size else 0.0,
        "same_bus_pair_count": int(same_bus_pairs),
        "same_bus_near_dup_frac": same_bus_frac,
        "same_bus_amp_normalized_corr_mean": amp_norm_mean,
        "amp_scale_redundant": amp_scale_redundant,
        "top_near_duplicates": top,
    }


def screen_t2_adaptive_room(
    *,
    Y: np.ndarray,
    U: np.ndarray,
    sigma: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_outer: int = 24,
    n_inner: int = 16,
    top_k: int = 12,
    seed: int = 20260808,
) -> dict[str, Any]:
    """Approximate T=2 Myopic / Fixed / adaptive-planning screen (vector obs)."""
    Y = np.asarray(Y, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64).reshape(-1)
    grid = np.asarray(u_grid, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    n_obs = int(Y.shape[2]) if Y.ndim == 3 else 1
    if Y.ndim == 2:
        Y = Y[:, :, None]
        n_obs = 1
    log_w0 = np.full(len(U), -np.log(len(U)), dtype=np.float64)

    u0 = rng.random(n_inner)
    z0 = rng.normal(size=(n_inner, n_obs)) * float(sigma)
    prior_scores = _one_step_scores(
        Y,
        U,
        log_w0,
        set(),
        sigma=float(sigma),
        alpha=float(alpha),
        margin=float(margin),
        grid=grid,
        uniforms=u0,
        noise=z0,
    )
    myopic_first = int(np.argmin(prior_scores))
    candidates = np.argsort(prior_scores)[: min(int(top_k), Y.shape[1])].tolist()
    if myopic_first not in candidates:
        candidates.append(myopic_first)

    outer_idx = rng.integers(0, len(U), size=int(n_outer))
    outer_noise = rng.normal(size=(int(n_outer), n_obs)) * float(sigma)
    inner_uniforms = rng.random((int(n_outer), int(n_inner)))
    inner_noise = rng.normal(size=(int(n_outer), int(n_inner), n_obs)) * float(sigma)

    planning_scores: dict[int, float] = {}
    branch_actions: dict[int, list[int]] = {}
    # Per-history one-step scores after ξ1 (for branch-value vs Fixed ξ2).
    branch_score_rows: dict[int, list[np.ndarray]] = {}
    for a1 in candidates:
        vals: list[float] = []
        seconds: list[int] = []
        score_rows: list[np.ndarray] = []
        centres1 = Y[:, a1, :]
        for r, n in enumerate(outer_idx):
            y1 = centres1[int(n)] + outer_noise[r]
            lw1 = _update(log_w0, y1, centres1, float(sigma))
            s2 = _one_step_scores(
                Y,
                U,
                lw1,
                {int(a1)},
                sigma=float(sigma),
                alpha=float(alpha),
                margin=float(margin),
                grid=grid,
                uniforms=inner_uniforms[r],
                noise=inner_noise[r],
            )
            a2 = int(np.argmin(s2))
            seconds.append(a2)
            vals.append(float(s2[a2]))
            score_rows.append(s2)
        planning_scores[int(a1)] = float(np.mean(vals))
        branch_actions[int(a1)] = seconds
        branch_score_rows[int(a1)] = score_rows

    best_first = min(planning_scores, key=planning_scores.get)
    plan_support = float(planning_scores[best_first])
    myopic_support = float(planning_scores[myopic_first])

    fixed_best = np.inf
    fixed_pair: list[int] | None = None
    fixed_scores: list[tuple[int, int, float]] = []
    for a1 in candidates:
        for a2 in range(Y.shape[1]):
            if a2 == a1:
                continue
            vals = []
            for r, n in enumerate(outer_idx):
                lw = _update(
                    log_w0, Y[int(n), a1] + outer_noise[r], Y[:, a1, :], float(sigma)
                )
                y2 = Y[int(n), a2] + inner_noise[r, 0]
                lw = _update(lw, y2, Y[:, a2, :], float(sigma))
                vals.append(
                    _terminal_u(
                        U, lw, alpha=float(alpha), margin=float(margin), grid=grid
                    )
                )
            score = float(np.mean(vals))
            fixed_scores.append((int(a1), int(a2), score))
            if score < fixed_best:
                fixed_best = score
                fixed_pair = [int(a1), int(a2)]

    seconds = branch_actions[best_first]
    uniq, counts = np.unique(seconds, return_counts=True)
    order = np.argsort(-counts)
    uniq, counts = uniq[order], counts[order]
    p = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    branch_mass = {str(int(a)): float(c) for a, c in zip(uniq, p)}
    ent = float(-np.sum(p * np.log(p + 1e-12)))
    mode_prob = float(p[0]) if p.size else 1.0
    effective_n = float(1.0 / np.sum(p * p)) if p.size else 1.0

    # Branch value at planning-optimal ξ1 vs Fixed-pair complementary second.
    # Always force the Fixed-pair partner (open-loop complementary action), not
    # the best open-loop a2 for an arbitrary a1.
    forced_a2 = None
    if fixed_pair is not None:
        fa, fb = int(fixed_pair[0]), int(fixed_pair[1])
        bf = int(best_first)
        if bf == fa:
            forced_a2 = fb
        elif bf == fb:
            forced_a2 = fa
        else:
            forced_a2 = fb if fb != bf else fa
    branch_deltas: list[float] = []
    for s2 in branch_score_rows[int(best_first)]:
        a2_star = int(np.argmin(s2))
        u_ad = float(s2[a2_star])
        if forced_a2 is not None and forced_a2 != int(best_first):
            u_fx = float(s2[forced_a2])
        else:
            u_fx = u_ad
        branch_deltas.append(u_fx - u_ad)
    branch_deltas_arr = np.asarray(branch_deltas, dtype=np.float64)
    mean_branch_value = (
        float(np.mean(branch_deltas_arr)) if branch_deltas_arr.size else 0.0
    )

    return {
        "N_obs": int(n_obs),
        "sigma": float(sigma),
        "support_size": int(len(U)),
        "n_actions": int(Y.shape[1]),
        "alpha": float(alpha),
        "margin": float(margin),
        "myopic_first_design": myopic_first,
        "planning_first_design": int(best_first),
        "approx_fixed_pair": fixed_pair,
        "J_myopic_T2": myopic_support,
        "J_planning_T2": plan_support,
        "J_fixed_T2_approx": float(fixed_best),
        "planning_minus_myopic": float(plan_support - myopic_support),
        "planning_minus_fixed": float(plan_support - float(fixed_best)),
        "n_distinct_second_actions": int(len(uniq)),
        "second_action_entropy": ent,
        "second_action_mass": branch_mass,
        "most_common_second_action_prob": mode_prob,
        "second_action_effective_n": effective_n,
        "forced_fixed_second_action": forced_a2,
        "mean_branch_value": mean_branch_value,
        "median_branch_value": (
            float(np.median(branch_deltas_arr)) if branch_deltas_arr.size else 0.0
        ),
        "frac_branch_value_gt0": (
            float(np.mean(branch_deltas_arr > 1e-12)) if branch_deltas_arr.size else 0.0
        ),
        "first_candidates": [int(x) for x in candidates],
        "search_note": (
            "Approximate Monte Carlo T=2 screen: top-k first actions, "
            "full second-action catalog. Not a trained-policy result. "
            "mean_branch_value = E[u|Fixed ξ2] − E[u|a2★] at planning ξ1."
        ),
    }


def _score_fixed_subset_mc(
    subset: list[int] | tuple[int, ...],
    *,
    Y: np.ndarray,
    U: np.ndarray,
    log_w0: np.ndarray,
    outer_idx: np.ndarray,
    outer_noise: np.ndarray,
    sigma: float,
    alpha: float,
    margin: float,
    grid: np.ndarray,
) -> float:
    """Mean posterior-safe u for an unordered size-T probe set (MC outer)."""
    acts = tuple(sorted(int(a) for a in subset))
    vals: list[float] = []
    for r, n in enumerate(outer_idx):
        lw = log_w0.copy()
        for t, a in enumerate(acts):
            noise = outer_noise[r, min(t, outer_noise.shape[1] - 1)]
            y = Y[int(n), int(a)] + noise
            lw = _update(lw, y, Y[:, int(a), :], float(sigma))
        vals.append(
            _terminal_u(U, lw, alpha=float(alpha), margin=float(margin), grid=grid)
        )
    return float(np.mean(vals))


def _candidate_fixed_subsets(
    *,
    n_actions: int,
    horizon: int,
    top_actions: list[int],
    rng: np.random.Generator,
    max_subsets: int = DEFAULT_MAX_FIXED_SUBSETS,
) -> list[tuple[int, ...]]:
    """Build a Fixed size-T pool: top-k combinations + random complements."""
    import itertools

    T = int(horizon)
    pool: set[tuple[int, ...]] = set()
    tops = [int(a) for a in top_actions if 0 <= int(a) < n_actions]
    if len(tops) < T:
        tops = list(dict.fromkeys(tops + list(range(n_actions))))[: max(T, len(tops))]
    # Prefer combinations among one-step-strong actions.
    if len(tops) >= T:
        combos = list(itertools.combinations(tops, T))
        if len(combos) > max_subsets:
            pick = rng.choice(len(combos), size=int(max_subsets), replace=False)
            combos = [combos[int(i)] for i in pick]
        for c in combos:
            pool.add(tuple(sorted(int(a) for a in c)))
    # Random complements anchored on each top action.
    remain_all = [a for a in range(n_actions)]
    n_extra = max(0, int(max_subsets) - len(pool))
    for _ in range(n_extra):
        if not tops:
            break
        a0 = int(rng.choice(tops))
        rest = [a for a in remain_all if a != a0]
        if len(rest) < T - 1:
            break
        comp = rng.choice(rest, size=T - 1, replace=False).tolist()
        pool.add(tuple(sorted([a0] + [int(x) for x in comp])))
        if len(pool) >= int(max_subsets):
            break
    return list(pool)


def screen_horizon_adaptive_room(
    *,
    Y: np.ndarray,
    U: np.ndarray,
    sigma: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    horizon: int,
    n_outer: int = 24,
    n_inner: int = 16,
    top_k: int = 12,
    max_fixed_subsets: int = DEFAULT_MAX_FIXED_SUBSETS,
    seed: int = 20260808,
) -> dict[str, Any]:
    """Approximate adaptive vs Fixed screen at horizon T.

    Adaptive: non-myopic prefix of length ``min(2, T)`` among a reduced
    candidate set, then one-step-greedy remainder (history-contingent).
    Fixed: best unordered size-T subset among a top-k / random candidate pool.
    """
    Y = np.asarray(Y, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64).reshape(-1)
    grid = np.asarray(u_grid, dtype=np.float64)
    T = int(horizon)
    if T < 1:
        raise ValueError(f"horizon must be >=1, got {T}")
    rng = np.random.default_rng(int(seed) + 31 * T)
    n_obs = int(Y.shape[2]) if Y.ndim == 3 else 1
    if Y.ndim == 2:
        Y = Y[:, :, None]
        n_obs = 1
    n_actions = int(Y.shape[1])
    if T > n_actions:
        raise ValueError(f"horizon T={T} exceeds n_actions={n_actions}")
    log_w0 = np.full(len(U), -np.log(len(U)), dtype=np.float64)

    u0 = rng.random(n_inner)
    z0 = rng.normal(size=(n_inner, n_obs)) * float(sigma)
    prior_scores = _one_step_scores(
        Y,
        U,
        log_w0,
        set(),
        sigma=float(sigma),
        alpha=float(alpha),
        margin=float(margin),
        grid=grid,
        uniforms=u0,
        noise=z0,
    )
    top_actions = np.argsort(prior_scores)[: min(int(top_k), n_actions)].tolist()
    myopic_first = int(np.argmin(prior_scores))
    candidates = list(dict.fromkeys([int(a) for a in top_actions] + [myopic_first]))
    # Smaller prefix set keeps T≥3 pairwise search tractable.
    prefix_cand = candidates[: min(8, len(candidates))]

    outer_idx = rng.integers(0, len(U), size=int(n_outer))
    outer_noise = rng.normal(size=(int(n_outer), T, n_obs)) * float(sigma)
    rest_uniforms = rng.random((int(n_outer), T, int(n_inner)))
    rest_noise = rng.normal(size=(int(n_outer), T, int(n_inner), n_obs)) * float(
        sigma
    )

    def _rollout_prefix(prefix: list[int]) -> float:
        vals: list[float] = []
        for r, n in enumerate(outer_idx):
            lw = log_w0.copy()
            used: set[int] = set()
            ok = True
            for step, a in enumerate(prefix):
                if int(a) in used:
                    ok = False
                    break
                y = Y[int(n), int(a)] + outer_noise[r, step]
                lw = _update(lw, y, Y[:, int(a), :], float(sigma))
                used.add(int(a))
            if not ok:
                vals.append(float("inf"))
                continue
            for step in range(len(prefix), T):
                s = _one_step_scores(
                    Y,
                    U,
                    lw,
                    used,
                    sigma=float(sigma),
                    alpha=float(alpha),
                    margin=float(margin),
                    grid=grid,
                    uniforms=rest_uniforms[r, step],
                    noise=rest_noise[r, step],
                )
                a = int(np.argmin(s))
                y = Y[int(n), a] + outer_noise[r, step]
                lw = _update(lw, y, Y[:, a, :], float(sigma))
                used.add(a)
            vals.append(
                _terminal_u(U, lw, alpha=float(alpha), margin=float(margin), grid=grid)
            )
        return float(np.mean(vals))

    # Non-myopic prefix length 1 (always) and 2 (when T≥2).
    planning_scores: dict[tuple[int, ...], float] = {}
    for a1 in prefix_cand:
        planning_scores[(int(a1),)] = _rollout_prefix([int(a1)])
    if T >= 2:
        for a1 in prefix_cand:
            # Second-prefix candidates: other prefix + a few prior-strong actions.
            second_pool = list(
                dict.fromkeys(
                    [int(a) for a in prefix_cand if int(a) != int(a1)]
                    + [int(a) for a in candidates if int(a) != int(a1)][:6]
                )
            )[:10]
            for a2 in second_pool:
                planning_scores[(int(a1), int(a2))] = _rollout_prefix(
                    [int(a1), int(a2)]
                )
    best_prefix = min(planning_scores, key=planning_scores.get)
    j_plan = float(planning_scores[best_prefix])

    subsets = _candidate_fixed_subsets(
        n_actions=n_actions,
        horizon=T,
        top_actions=[int(a) for a in candidates],
        rng=rng,
        max_subsets=int(max_fixed_subsets),
    )
    fixed_best = float("inf")
    fixed_subset: list[int] | None = None
    for subset in subsets:
        score = _score_fixed_subset_mc(
            subset,
            Y=Y,
            U=U,
            log_w0=log_w0,
            outer_idx=outer_idx,
            outer_noise=outer_noise,
            sigma=float(sigma),
            alpha=float(alpha),
            margin=float(margin),
            grid=grid,
        )
        if score < fixed_best:
            fixed_best = score
            fixed_subset = list(subset)
    if fixed_subset is None:
        fixed_best = j_plan
        fixed_subset = []

    gap = float(j_plan - float(fixed_best))
    return {
        "horizon": int(T),
        "N_obs": int(n_obs),
        "sigma": float(sigma),
        "support_size": int(len(U)),
        "n_actions": int(n_actions),
        "J_plan": j_plan,
        "J_fixed": float(fixed_best),
        "gap": gap,
        "planning_minus_fixed": gap,
        "planning_prefix": [int(a) for a in best_prefix],
        "myopic_first_design": int(myopic_first),
        "approx_fixed_subset": fixed_subset,
        "n_fixed_subsets_scored": int(len(subsets)),
        "n_prefixes_scored": int(len(planning_scores)),
        "top_actions": [int(a) for a in top_actions],
        "first_candidates": [int(a) for a in candidates],
        "search_note": (
            "Approximate MC multi-T screen: non-myopic prefix (len≤2) then "
            "one-step-greedy remainder vs best Fixed size-T subset. "
            "Not a trained policy."
        ),
    }


def screen_multi_horizon_adaptive_room(
    *,
    Y: np.ndarray,
    U: np.ndarray,
    sigma: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    horizons: tuple[int, ...] | list[int] = DEFAULT_STRUCTURE_AUDIT_HORIZONS,
    n_outer: int = 24,
    n_inner: int = 16,
    top_k: int = 12,
    max_fixed_subsets: int = DEFAULT_MAX_FIXED_SUBSETS,
    min_gap_improve: float = DEFAULT_MIN_GAP_IMPROVE_PER_HORIZON,
    min_fixed_advantage: float = DEFAULT_MIN_FIXED_ADVANTAGE,
    seed: int = 20260808,
) -> dict[str, Any]:
    """Run horizon screens and test monotone growth of adaptive room vs Fixed."""
    hs = [int(t) for t in horizons]
    by_T: dict[str, Any] = {}
    for T in hs:
        by_T[str(T)] = screen_horizon_adaptive_room(
            Y=Y,
            U=U,
            sigma=float(sigma),
            alpha=float(alpha),
            margin=float(margin),
            u_grid=u_grid,
            horizon=int(T),
            n_outer=n_outer,
            n_inner=n_inner,
            top_k=top_k,
            max_fixed_subsets=max_fixed_subsets,
            seed=seed,
        )
    gaps = {int(T): float(by_T[str(T)]["gap"]) for T in hs}
    mono_fails: list[str] = []
    min_adv = float(min_fixed_advantage)
    min_imp = float(min_gap_improve)
    if 2 in gaps and not (gaps[2] <= -min_adv):
        mono_fails.append(
            f"gap(2)={gaps[2]:.6f} (need ≤ -{min_adv:g})"
        )
    for a, b in zip(hs, hs[1:]):
        # Larger T ⇒ more negative gap (planner pulls further ahead of Fixed).
        need = float(gaps[a]) - min_imp
        if not (gaps[b] <= need):
            mono_fails.append(
                f"gap({b})={gaps[b]:.6f} not ≤ gap({a})-{min_imp:g}="
                f"{need:.6f}"
            )
    return {
        "horizons": hs,
        "by_horizon": by_T,
        "gaps": {str(k): float(v) for k, v in gaps.items()},
        "min_fixed_advantage": min_adv,
        "min_gap_improve_per_horizon": min_imp,
        "monotone_ok": len(mono_fails) == 0,
        "monotone_failures": mono_fails,
    }


def _u_heterogeneity(U: np.ndarray) -> dict[str, Any]:
    U = np.asarray(U, dtype=np.float64).reshape(-1)
    if U.size == 0:
        return {"n": 0}
    uniq = np.unique(np.round(U, 8))
    return {
        "n": int(U.size),
        "mean": float(U.mean()),
        "std": float(U.std()),
        "q05": float(np.quantile(U, 0.05)),
        "q95": float(np.quantile(U, 0.95)),
        "headroom_q95_minus_mean": float(np.quantile(U, 0.95) - U.mean()),
        "positive_frac": float(np.mean(U > 1e-12)),
        "n_unique_rounded": int(uniq.size),
    }


def analyze_myopic_trap(
    *,
    max_rocof: np.ndarray,
    t2: dict[str, Any],
    catalog_designs: list[Any] | None = None,
) -> dict[str, Any]:
    """Detect the classic Myopic trap.

    Trap structure (user definition):
      - ξ1 is best one-step (largest immediate expected u / MOCU reduction);
      - ξ1 overlaps other useful designs (high fingerprint correlation);
      - optimal T≥2 combo prefers a different first design / excludes ξ1;
      - so greedy always starts with ξ1 and is suboptimal for T>1.
    """
    R = np.asarray(max_rocof, dtype=np.float64)
    myopic_first = int(t2.get("myopic_first_design", -1))
    planning_first = int(t2.get("planning_first_design", -1))
    fixed_pair = list(t2.get("approx_fixed_pair") or [])
    d_my = float(t2.get("planning_minus_myopic", 0.0))
    n_actions = int(R.shape[1]) if R.ndim == 2 else 0

    buses, amps = _parse_design_meta(catalog_designs, n_actions)
    durs: list[float | None] = [None] * n_actions
    if catalog_designs is not None:
        for i, d in enumerate(catalog_designs[:n_actions]):
            if isinstance(d, dict):
                durs[i] = float(d.get("duration", d.get("duration_s", float("nan"))))
            elif isinstance(d, (list, tuple)) and len(d) >= 3:
                durs[i] = float(d[2])

    overlap = float("nan")
    if 0 <= myopic_first < n_actions and n_actions >= 2:
        Z = R - R.mean(axis=0, keepdims=True)
        std = R.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        Z = Z / std
        corr = (Z.T @ Z) / max(R.shape[0] - 1, 1)
        others = [j for j in range(n_actions) if j != myopic_first]
        overlap = float(np.mean(np.abs(corr[myopic_first, others])))

    first_differs = myopic_first != planning_first and myopic_first >= 0
    excluded_from_fixed = bool(fixed_pair) and (myopic_first not in fixed_pair)
    nonmyopic_gap = d_my < -DEFAULT_ADVANTAGE_EPS
    # Strong trap: greedy first is not the planning-optimal first, with value gap.
    # Overlay evidence: mean |corr| of ξ1 with other actions is not tiny.
    trap_present = bool(first_differs and nonmyopic_gap)
    strong_trap = bool(
        trap_present and (excluded_from_fixed or (np.isfinite(overlap) and overlap > 0.15))
    )

    return {
        "myopic_first_design": myopic_first,
        "planning_first_design": planning_first,
        "approx_fixed_pair": fixed_pair,
        "myopic_first_equals_planning_first": not first_differs,
        "myopic_first_in_fixed_pair": (
            (myopic_first in fixed_pair) if fixed_pair else None
        ),
        "planning_minus_myopic": d_my,
        "xi1_mean_abs_corr_with_others": overlap,
        "trap_present": trap_present,
        "strong_trap": strong_trap,
        "myopic_first_meta": {
            "bus": buses[myopic_first] if 0 <= myopic_first < n_actions else None,
            "amp": amps[myopic_first] if 0 <= myopic_first < n_actions else None,
            "duration": durs[myopic_first] if 0 <= myopic_first < n_actions else None,
        },
        "planning_first_meta": {
            "bus": buses[planning_first] if 0 <= planning_first < n_actions else None,
            "amp": amps[planning_first] if 0 <= planning_first < n_actions else None,
            "duration": durs[planning_first] if 0 <= planning_first < n_actions else None,
        },
        "interpretation": (
            "MYOPIC_TRAP: ξ1 is one-step best but a different first design is "
            "better for T≥2 (information overlay / option-value)."
            if trap_present
            else "NO_MYOPIC_TRAP: one-step greedy first matches (or is as good as) "
            "non-myopic first — Myopic is hard to beat; add complementary "
            "overlap structure (e.g. multi-duration waveforms), not just more amps."
        ),
    }


def _recommendations(
    *,
    redundancy: dict[str, Any],
    t2: dict[str, Any],
    u_stats: dict[str, Any],
    near_dup_limit: float,
    trap: dict[str, Any] | None = None,
) -> list[str]:
    tips: list[str] = []
    trap = trap or {}
    if bool(redundancy.get("amp_scale_redundant")):
        tips.append(
            "Probe amplitudes are scale-redundant (same-bus amp-normalized "
            "|corr|≈1): use a single probe_amplitude. Multi-amp does not create "
            "a Myopic trap — it only duplicates ξ."
        )
    elif float(redundancy.get("near_duplicate_frac", 0.0)) > near_dup_limit:
        tips.append(
            "High cross-action redundancy: drop near-duplicate buses/amps and "
            "regenerate into a new dataset_dir."
        )
    if not bool(trap.get("trap_present")):
        tips.append(
            "No Myopic trap yet: greedy ξ1 is also (near) optimal for T≥2. "
            "Need a one-step-best design that overlays other useful probes so the "
            "optimal T-set excludes ξ1. Try probe_durations=[short,mid,long] with "
            "one amp (short pulse = high-ROCOF bait; long pulse = complementary)."
        )
    if (
        not bool(redundancy.get("amp_scale_redundant"))
        and float(redundancy.get("same_bus_near_dup_frac", 0.0)) > 0.5
    ):
        tips.append(
            "Same-bus designs remain near-duplicates: keep one amplitude per bus."
        )
    if float(t2.get("planning_minus_myopic", 0.0)) > -DEFAULT_ADVANTAGE_EPS:
        tips.append(
            "Little non-myopic gap vs Myopic: enlarge overlay trap (durations) or "
            "use moderate noise with vector N_obs>=100 and T>=3."
        )
    if float(t2.get("planning_minus_fixed", 0.0)) > -DEFAULT_ADVANTAGE_EPS:
        tips.append(
            "Adaptive planning does not beat Fixed on this screen: increase U "
            "heterogeneity (stronger contingency / lower nadir) or reduce open-loop "
            "sufficiency by making probes more informative but belief-dependent."
        )
    if int(t2.get("n_distinct_second_actions", 1)) <= 1:
        tips.append(
            "No observation-dependent second action: branching is flat. Prefer "
            "vector Δf (N_obs>0) over scalar max-ROCOF and avoid near-duplicate designs."
        )
    if float(u_stats.get("headroom_q95_minus_mean", 0.0)) < 0.05:
        tips.append(
            "Weak U headroom: tighten control.contingency magnitude and regenerate."
        )
    if bool(trap.get("trap_present")) and not tips:
        tips.append(
            "Myopic trap detected: train DAD/RL-sBOED (not MoE yet) — they should "
            "learn to avoid the one-step bait ξ1."
        )
    if not tips:
        tips.append(
            "Structure looks usable: train DAD and RL-sBOED (not MoE yet) on this "
            "cell and compare against Fixed/Myopic with paired CIs."
        )
    return tips


def _verdict(
    *,
    redundancy: dict[str, Any],
    t2: dict[str, Any],
    near_dup_limit: float,
    trap: dict[str, Any] | None = None,
) -> str:
    red = float(redundancy.get("near_duplicate_frac", 0.0))
    d_my = float(t2.get("planning_minus_myopic", 0.0))
    d_fx = float(t2.get("planning_minus_fixed", 0.0))
    branch = int(t2.get("n_distinct_second_actions", 0))
    room_my = d_my < -DEFAULT_ADVANTAGE_EPS
    room_fx = d_fx < -DEFAULT_ADVANTAGE_EPS
    trap = trap or {}
    if bool(redundancy.get("amp_scale_redundant")) and not (room_my and room_fx):
        return "AMP_SCALE_REDUNDANT"
    if red > near_dup_limit and not (room_my or room_fx):
        return "REDUNDANT_LITTLE_ADAPTIVE_ROOM"
    if bool(trap.get("strong_trap")):
        return "MYOPIC_TRAP_READY_FOR_DAD_RL"
    if bool(trap.get("trap_present")) and room_fx:
        return "MYOPIC_TRAP_PRESENT"
    if room_my and room_fx and branch >= 2:
        return "ROOM_FOR_DAD_RL"
    if room_my or room_fx:
        return "PARTIAL_ADAPTIVE_ROOM"
    if branch <= 1:
        return "NO_BRANCHING_FIXED_OR_MYOPIC_CEILING"
    return "LITTLE_ADAPTIVE_ROOM_MYOPIC_HARD"


def run_bank_structure_audit(
    cfg: SBOEDConfig,
    *,
    n_obs: int = 200,
    noise_sigma: float = 0.01,
    support_size: int = 96,
    n_outer: int = 24,
    n_inner: int = 16,
    top_k: int = 12,
    seed: int = 20260808,
    near_dup_corr: float = DEFAULT_NEAR_DUP_CORR,
    near_dup_frac_limit: float = 0.25,
    min_fixed_advantage: float = DEFAULT_MIN_FIXED_ADVANTAGE,
    min_distinct_second_actions: int = 2,
    min_mean_branch_value: float = DEFAULT_MIN_MEAN_BRANCH_VALUE,
    max_mode_second_action_prob: float = DEFAULT_MAX_MODE_SECOND_ACTION_PROB,
    structure_audit_horizons: tuple[int, ...] | list[int] | None = None,
    min_gap_improve_per_horizon: float = DEFAULT_MIN_GAP_IMPROVE_PER_HORIZON,
    max_fixed_subsets: int = DEFAULT_MAX_FIXED_SUBSETS,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Full Plan-2 audit for one config / observation setting."""
    root = project_root or repo_root()
    data_dir = resolve_dataset_dir(cfg, root)
    system = system_name_from_cfg(cfg)
    bank = load_bank_from_path(data_dir, cfg=cfg, skip_quality_check=True)
    train_df = np.asarray(bank["full_train"])
    train_rocof = bank.get("max_rocof_train")
    if train_rocof is None:
        raise RuntimeError(
            f"Bank at {data_dir} missing train/max_rocof.npy; required for structure audit"
        )
    train_rocof = np.asarray(train_rocof)
    train_U = np.asarray(bank["U_train"], dtype=np.float64).reshape(-1)

    catalog = {}
    cat_path = data_dir / "meta" / "catalog.json"
    if cat_path.is_file():
        catalog = json.loads(cat_path.read_text(encoding="utf-8"))
    designs = list(catalog.get("designs") or [])

    redundancy = analyze_action_redundancy(
        max_rocof=train_rocof,
        catalog_designs=designs,
        near_dup_corr=near_dup_corr,
    )
    u_stats = _u_heterogeneity(train_U)

    control = dict(cfg.raw.get("control") or {})
    alpha = float(control.get("alpha", 0.01))
    margin = float(control.get("safety_margin", 0.0))
    u_grid = np.asarray(control.get("u_candidates"), dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    n_pick = int(min(max(support_size, 8), len(train_U)))
    pick = np.sort(rng.choice(len(train_U), size=n_pick, replace=False))
    centres_a_n_d, indices, mode = build_centres_bank(
        train_df[pick], train_rocof[pick], int(n_obs)
    )
    # centres_a_n_d: (n_actions, n_theta, obs_dim) → Y (n_theta, n_actions, obs_dim)
    Y = np.transpose(np.asarray(centres_a_n_d, dtype=np.float64), (1, 0, 2))
    U = train_U[pick]

    t2 = screen_t2_adaptive_room(
        Y=Y,
        U=U,
        sigma=float(noise_sigma),
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        n_outer=n_outer,
        n_inner=n_inner,
        top_k=top_k,
        seed=seed + 17,
    )
    t2["observation_mode"] = str(mode)
    t2["obs_indices_head"] = np.asarray(indices).reshape(-1)[:5].tolist()

    horizons = (
        list(structure_audit_horizons)
        if structure_audit_horizons is not None
        else list(DEFAULT_STRUCTURE_AUDIT_HORIZONS)
    )
    multi_t = screen_multi_horizon_adaptive_room(
        Y=Y,
        U=U,
        sigma=float(noise_sigma),
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        horizons=horizons,
        n_outer=n_outer,
        n_inner=n_inner,
        top_k=top_k,
        max_fixed_subsets=max_fixed_subsets,
        min_gap_improve=float(min_gap_improve_per_horizon),
        min_fixed_advantage=float(min_fixed_advantage),
        seed=seed + 41,
    )

    trap = analyze_myopic_trap(
        max_rocof=train_rocof, t2=t2, catalog_designs=designs
    )
    verdict = _verdict(
        redundancy=redundancy,
        t2=t2,
        near_dup_limit=float(near_dup_frac_limit),
        trap=trap,
    )
    tips = _recommendations(
        redundancy=redundancy,
        t2=t2,
        u_stats=u_stats,
        near_dup_limit=float(near_dup_frac_limit),
        trap=trap,
    )
    if not bool(multi_t.get("monotone_ok")):
        tips.append(
            "Multi-T adaptive room is not monotone: enlarge nested duration "
            "complementarity so gap(T)=J_plan−J_fixed deepens at T=3,4 "
            "(history-contingent residual U-tail), then regenerate with --force."
        )

    dad_rl_ready = verdict in (
        "MYOPIC_TRAP_READY_FOR_DAD_RL",
        "MYOPIC_TRAP_PRESENT",
        "ROOM_FOR_DAD_RL",
        "PARTIAL_ADAPTIVE_ROOM",
    )
    myopic_beatable = bool(trap.get("trap_present"))
    # Solution-1 / v3 gate: Fixed gap + meaningful history-contingent branching.
    d_fx = float(t2.get("planning_minus_fixed", 0.0))
    n_branch = int(t2.get("n_distinct_second_actions", 0))
    mean_bv = float(t2.get("mean_branch_value", 0.0))
    mode_p = float(t2.get("most_common_second_action_prob", 1.0))
    min_adv = float(min_fixed_advantage)
    min_branch = int(min_distinct_second_actions)
    min_bv = float(min_mean_branch_value)
    max_mode = float(max_mode_second_action_prob)
    fixed_beatable = bool(d_fx <= -min_adv)
    branching_ok = bool(
        n_branch >= min_branch and mean_bv >= min_bv and mode_p <= max_mode
    )
    adaptive_room = bool(fixed_beatable and branching_ok)
    monotone_adaptive_room = bool(multi_t.get("monotone_ok"))
    report = {
        "purpose": (
            "Plan-2 structure audit: Myopic trap + Fixed-beatable adaptive room "
            "+ multi-T monotone gap(T) (Solution-1 bank gate). Pass/fail only — "
            "never filter θ/designs."
        ),
        "system": system,
        "data_dir": str(data_dir.resolve()),
        "config_path": str(getattr(cfg, "config_path", "") or ""),
        "N_obs": int(n_obs),
        "noise_sigma": float(noise_sigma),
        "verdict": verdict,
        "dad_rl_ready": bool(dad_rl_ready),
        "myopic_beatable": bool(myopic_beatable),
        "fixed_beatable": bool(fixed_beatable),
        "branching_ok": bool(branching_ok),
        "adaptive_room": bool(adaptive_room),
        "monotone_adaptive_room": bool(monotone_adaptive_room),
        "moe_deferred": True,
        "u_heterogeneity": u_stats,
        "action_redundancy": redundancy,
        "myopic_trap": trap,
        "t2_adaptive_screen": t2,
        "multi_horizon_adaptive_screen": multi_t,
        "thresholds": {
            "near_duplicate_corr": float(near_dup_corr),
            "near_duplicate_frac_limit": float(near_dup_frac_limit),
            "advantage_eps": float(DEFAULT_ADVANTAGE_EPS),
            "min_fixed_advantage": float(min_adv),
            "min_distinct_second_actions": int(min_branch),
            "min_mean_branch_value": float(min_bv),
            "max_mode_second_action_prob": float(max_mode),
            "structure_audit_horizons": [int(t) for t in horizons],
            "min_gap_improve_per_horizon": float(min_gap_improve_per_horizon),
        },
        "recommendations": tips,
        "next_steps": (
            [
                "Solution-1 PASS: freeze dataset_dir + YAML.",
                "Train DAD and RL-sBOED first; confirm they beat Fixed/Myopic.",
                "Only then enable MoE-sBOED for learner ranking.",
            ]
            if myopic_beatable and adaptive_room and monotone_adaptive_room
            else [
                "Solution-1 FAIL: retune YAML (contingency / probe_durations), "
                "regenerate FULL bank with --force (no θ/design filtering).",
                "Re-run --bank-structure-audit at every sweep σ until "
                "myopic_trap, adaptive_room, and monotone_adaptive_room pass.",
            ]
        ),
    }
    return report


def write_audit_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "bank_structure_audit.json"
    md_path = out_dir / "bank_structure_audit.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    red = report["action_redundancy"]
    t2 = report["t2_adaptive_screen"]
    u = report["u_heterogeneity"]
    trap = report.get("myopic_trap") or {}
    multi = report.get("multi_horizon_adaptive_screen") or {}
    lines = [
        "# Plan-2 bank structure audit\n",
        f"- **system:** `{report['system']}`",
        f"- **data_dir:** `{report['data_dir']}`",
        f"- **N_obs / sigma:** `{report['N_obs']}` / `{report['noise_sigma']}`",
        f"- **verdict:** `{report['verdict']}`",
        f"- **Myopic beatable (trap):** `{report.get('myopic_beatable')}`",
        f"- **Fixed beatable (planning−fixed):** `{report.get('fixed_beatable')}`",
        f"- **Branching (distinct ξ₂≥2):** `{report.get('branching_ok')}`",
        f"- **Adaptive room (Fixed-beatable ∧ branching):** "
        f"`{report.get('adaptive_room')}` "
        f"(need plan−Fixed ≤ −{(report.get('thresholds') or {}).get('min_fixed_advantage', 0.01)}, "
        f"ξ₂≥{(report.get('thresholds') or {}).get('min_distinct_second_actions', 2)})",
        f"- **Monotone adaptive room (multi-T gap):** "
        f"`{report.get('monotone_adaptive_room')}`",
        f"- **DAD/RL ready:** `{report['dad_rl_ready']}` "
        f"(MoE deferred: `{report['moe_deferred']}`)\n",
        "## Myopic trap\n",
        f"- trap_present=`{trap.get('trap_present')}`, "
        f"strong_trap=`{trap.get('strong_trap')}`",
        f"- myopic_first=`{trap.get('myopic_first_design')}` "
        f"{trap.get('myopic_first_meta')}",
        f"- planning_first=`{trap.get('planning_first_design')}` "
        f"{trap.get('planning_first_meta')}",
        f"- fixed_pair=`{trap.get('approx_fixed_pair')}`, "
        f"ξ1 in fixed=`{trap.get('myopic_first_in_fixed_pair')}`",
        f"- ξ1 mean |corr| with others="
        f"`{trap.get('xi1_mean_abs_corr_with_others')}`",
        f"- {trap.get('interpretation')}\n",
        "## U heterogeneity\n",
        f"- mean={u.get('mean'):.4f}, std={u.get('std'):.4f}, "
        f"Q95={u.get('q95'):.4f}, headroom={u.get('headroom_q95_minus_mean'):.4f}, "
        f"U>0 frac={u.get('positive_frac'):.3f}, unique≈{u.get('n_unique_rounded')}\n",
        "## Action redundancy (max-|ROCOF| fingerprints)\n",
        f"- n_actions={red.get('n_actions')} "
        f"(amps={red.get('n_unique_amps')}, buses={red.get('n_unique_buses')})",
        f"- near-dup frac={red.get('near_duplicate_frac'):.3f} "
        f"(thr |corr|≥{red.get('near_duplicate_threshold')})",
        f"- mean |corr|={red.get('mean_abs_corr'):.3f}, "
        f"max |corr|={red.get('max_abs_corr'):.3f}",
        f"- amp_scale_redundant={red.get('amp_scale_redundant')}, "
        f"same-bus near-dup frac={red.get('same_bus_near_dup_frac'):.3f}\n",
        "## T=2 adaptive screen (lower J better)\n",
        f"- J_myopic={t2.get('J_myopic_T2'):.6f}",
        f"- J_planning={t2.get('J_planning_T2'):.6f}",
        f"- J_fixed≈{t2.get('J_fixed_T2_approx'):.6f}",
        f"- planning−myopic={t2.get('planning_minus_myopic'):.6f}",
        f"- planning−fixed={t2.get('planning_minus_fixed'):.6f}",
        f"- distinct second actions={t2.get('n_distinct_second_actions')}, "
        f"entropy={t2.get('second_action_entropy'):.3f}",
        f"- mode ξ2 prob={t2.get('most_common_second_action_prob')}, "
        f"eff_n={t2.get('second_action_effective_n')}",
        f"- mean_branch_value={t2.get('mean_branch_value')}\n",
        "## Multi-T adaptive room (gap = J_plan − J_fixed)\n",
        f"- gaps=`{multi.get('gaps')}`",
        f"- monotone_ok=`{multi.get('monotone_ok')}`",
        f"- monotone_failures=`{multi.get('monotone_failures')}`",
        f"- min_gap_improve=`{multi.get('min_gap_improve_per_horizon')}`\n",
        "## Recommendations\n",
    ]
    for tip in report["recommendations"]:
        lines.append(f"- {tip}")
    lines.append("\n## Next steps\n")
    for step in report["next_steps"]:
        lines.append(f"- {step}")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def structure_gate_failures(
    redundancy: dict[str, Any],
    *,
    max_near_duplicate_frac: float | None,
    max_same_bus_near_dup_frac: float | None,
    reject_amp_scale_redundant: bool = False,
) -> list[str]:
    """Optional hard gates used by bank_quality (Plan-2 configs)."""
    fails: list[str] = []
    if reject_amp_scale_redundant and bool(redundancy.get("amp_scale_redundant")):
        fails.append(
            "amp_scale_redundant=true (multi-amp catalog is pure ROCOF scaling; "
            "set probe_amplitudes to a single value)"
        )
    if max_near_duplicate_frac is not None:
        val = float(redundancy.get("near_duplicate_frac", 0.0))
        if val > float(max_near_duplicate_frac):
            fails.append(
                f"near_duplicate_frac={val:.3f} > max={float(max_near_duplicate_frac):.3f}"
            )
    # Only enforce same-bus near-dup when amps are not pure scales (otherwise
    # amp_scale_redundant already covers the failure mode).
    if (
        max_same_bus_near_dup_frac is not None
        and not bool(redundancy.get("amp_scale_redundant"))
    ):
        val = float(redundancy.get("same_bus_near_dup_frac", 0.0))
        if val > float(max_same_bus_near_dup_frac):
            fails.append(
                f"same_bus_near_dup_frac={val:.3f} > "
                f"max={float(max_same_bus_near_dup_frac):.3f}"
            )
    return fails


def adaptive_structure_gate_failures(
    report: dict[str, Any],
    *,
    require_myopic_trap: bool,
    require_adaptive_room: bool,
    require_monotone_adaptive_room: bool = False,
    min_distinct_second_actions: int = 2,
    min_fixed_advantage: float = DEFAULT_MIN_FIXED_ADVANTAGE,
    min_mean_branch_value: float = DEFAULT_MIN_MEAN_BRANCH_VALUE,
    max_mode_second_action_prob: float = DEFAULT_MAX_MODE_SECOND_ACTION_PROB,
    min_gap_improve_per_horizon: float = DEFAULT_MIN_GAP_IMPROVE_PER_HORIZON,
) -> list[str]:
    """Pass/fail only — never filters θ or designs. Tune YAML and regenerate."""
    fails: list[str] = []
    trap = report.get("myopic_trap") or {}
    t2 = report.get("t2_adaptive_screen") or {}
    multi = report.get("multi_horizon_adaptive_screen") or {}
    sigma = report.get("noise_sigma")
    n_obs = report.get("N_obs")
    tag = f"N_obs={n_obs} sigma={sigma}"
    # Prefer thresholds recorded on the audit report when present.
    thr = report.get("thresholds") or {}
    min_adv = float(thr.get("min_fixed_advantage", min_fixed_advantage))
    min_branch = int(
        thr.get("min_distinct_second_actions", min_distinct_second_actions)
    )
    min_bv = float(thr.get("min_mean_branch_value", min_mean_branch_value))
    max_mode = float(
        thr.get("max_mode_second_action_prob", max_mode_second_action_prob)
    )
    min_imp = float(
        thr.get("min_gap_improve_per_horizon", min_gap_improve_per_horizon)
    )

    if require_myopic_trap and not bool(trap.get("trap_present")):
        fails.append(
            f"myopic_trap missing ({tag}): trap_present=false — adjust "
            "probe_durations / catalog so one-step-best ξ1 is excluded from "
            "optimal T≥2 sets, then regenerate with --force"
        )

    if require_adaptive_room:
        d_fx = float(t2.get("planning_minus_fixed", 0.0))
        n_branch = int(t2.get("n_distinct_second_actions", 0))
        mean_bv = float(t2.get("mean_branch_value", 0.0))
        mode_p = float(t2.get("most_common_second_action_prob", 1.0))
        if not (d_fx <= -min_adv):
            fails.append(
                f"adaptive_room Fixed gap missing ({tag}): "
                f"planning_minus_fixed={d_fx:.6f} (need ≤ -{min_adv:g}) — "
                "increase U heterogeneity / belief-dependent probes via YAML, "
                "then regenerate with --force"
            )
        if n_branch < min_branch:
            fails.append(
                f"adaptive_room branching missing ({tag}): "
                f"n_distinct_second_actions={n_branch} (need ≥{min_branch}) — "
                "make probes less redundant / more informative under vector obs, "
                "then regenerate with --force"
            )
        if mean_bv < min_bv:
            fails.append(
                f"adaptive_room branch value missing ({tag}): "
                f"mean_branch_value={mean_bv:.6f} (need ≥ {min_bv:g}) — "
                "strengthen history-contingent control-tail ambiguity "
                "(contingency / complementary durations), then --force"
            )
        if mode_p > max_mode:
            fails.append(
                f"adaptive_room ξ2 collapse ({tag}): "
                f"most_common_second_action_prob={mode_p:.3f} "
                f"(need ≤ {max_mode:g}) — branching is effectively open-loop; "
                "retune probes so posterior histories change optimal ξ2"
            )

    if require_monotone_adaptive_room:
        if not bool(multi.get("monotone_ok", report.get("monotone_adaptive_room"))):
            detail = multi.get("monotone_failures") or multi.get("gaps") or "unknown"
            fails.append(
                f"monotone_adaptive_room missing ({tag}): {detail} "
                f"(need gap(2)≤−{min_adv:g} and each next T improves by "
                f"≥{min_imp:g}); retune nested durations and --force"
            )
    return fails
