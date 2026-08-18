"""Objective-optimized nonadaptive fixed design: unordered size-T subsets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from src.control.posterior_ctrl import normalize_log_weights, posterior_safe_u_ctrl
from src.observations.likelihood import log_gaussian_observation_density
from src.banks.tables import TableThetaSupport, y_sim_last_step_from_tables


@dataclass
class FixedSearchResult:
    subset: list[int]
    objective: float
    n_candidates_evaluated: int
    search_mode: str
    seed: int
    metadata: dict[str, Any]


def n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def estimate_fixed_subset_objective(
    subset: tuple[int, ...] | list[int],
    *,
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    calibration_systems: list[dict[str, Any]],
    sigma_y: float,
    alpha: float,
    noise_replicas: int,
    rng: np.random.Generator,
    margin: float = 0.0,
    u_grid=None,
) -> float:
    """
    Monte Carlo estimate of E[u_ctrl | X] for a fixed unordered probe set.

    For each calibration θ*, draw noise replicas on banked y_sim centres, update
    posterior from the full set, then compute posterior-safe u_ctrl.
    """
    subset = tuple(sorted(int(a) for a in subset))
    if not subset:
        w0 = normalize_log_weights(table_support.log_p0)
        return posterior_safe_u_ctrl(
            U_support, w0, alpha, margin=margin, u_grid=u_grid
        )

    centres = {a: y_sim_last_step_from_tables(table_support, [a]) for a in subset}
    vals: list[float] = []
    n_rep = max(1, int(noise_replicas))
    for sys in calibration_systems:
        # Prefer banked clean centres for the true θ* when available via lookup.
        from src.banks.tables import lookup_action_y_sim

        y_clean = np.asarray(
            [lookup_action_y_sim(sys, a) for a in subset], dtype=np.float64
        )
        for _ in range(n_rep):
            y_obs = y_clean + rng.normal(0.0, sigma_y, size=len(subset))
            log_w = np.asarray(table_support.log_p0, dtype=np.float64).copy()
            for a, y in zip(subset, y_obs):
                log_L = log_gaussian_observation_density(float(y), centres[a], sigma_y)
                log_w = log_w + log_L
            w = normalize_log_weights(log_w)
            vals.append(
                posterior_safe_u_ctrl(
                    U_support, w, alpha, margin=margin, u_grid=u_grid
                )
            )
    return float(np.mean(vals)) if vals else float("inf")


def search_fixed_subset(
    *,
    n_actions: int,
    horizon: int,
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    calibration_systems: list[dict[str, Any]],
    sigma_y: float,
    alpha: float,
    rng: np.random.Generator,
    exhaustive_threshold: int = 5000,
    noise_replicas: int = 4,
    greedy_restarts: int = 8,
    seed: int = 0,
    margin: float = 0.0,
    u_grid=None,
) -> FixedSearchResult:
    """
    Minimize E[u_ctrl] over unordered size-T subsets.

    Exhaustive search when C(N_ξ, T) ≤ threshold; otherwise multi-start greedy
    forward selection (configured approximate algorithm).
    """
    T = int(horizon)
    n_cand = n_choose_k(n_actions, T)
    meta: dict[str, Any] = {
        "n_actions": n_actions,
        "horizon": T,
        "n_combinations": n_cand,
        "exhaustive_threshold": exhaustive_threshold,
    }

    if n_cand <= exhaustive_threshold:
        best_subset: tuple[int, ...] | None = None
        best_obj = float("inf")
        evaluated = 0
        for subset in combinations(range(n_actions), T):
            obj = estimate_fixed_subset_objective(
                subset,
                table_support=table_support,
                U_support=U_support,
                calibration_systems=calibration_systems,
                sigma_y=sigma_y,
                alpha=alpha,
                noise_replicas=noise_replicas,
                rng=rng,
                margin=margin,
                u_grid=u_grid,
            )
            evaluated += 1
            if obj < best_obj:
                best_obj = obj
                best_subset = tuple(subset)
        assert best_subset is not None
        return FixedSearchResult(
            subset=list(best_subset),
            objective=float(best_obj),
            n_candidates_evaluated=evaluated,
            search_mode="exhaustive",
            seed=int(seed),
            metadata=meta,
        )

    # Approximate: multi-start greedy forward selection.
    best_subset_g: list[int] | None = None
    best_obj_g = float("inf")
    evaluated_g = 0
    for restart in range(max(1, greedy_restarts)):
        chosen: list[int] = []
        remaining = list(range(n_actions))
        rng.shuffle(remaining)
        # Seed with a random first action for diversity.
        if remaining:
            first = int(remaining.pop(0))
            chosen.append(first)
        while len(chosen) < T and remaining:
            cand_best = None
            cand_obj = float("inf")
            for a in remaining:
                trial = tuple(sorted(chosen + [a]))
                obj = estimate_fixed_subset_objective(
                    trial,
                    table_support=table_support,
                    U_support=U_support,
                    calibration_systems=calibration_systems,
                    sigma_y=sigma_y,
                    alpha=alpha,
                    noise_replicas=noise_replicas,
                    rng=rng,
                    margin=margin,
                    u_grid=u_grid,
                )
                evaluated_g += 1
                if obj < cand_obj or (abs(obj - cand_obj) < 1e-15 and (cand_best is None or a < cand_best)):
                    cand_obj = obj
                    cand_best = a
            assert cand_best is not None
            chosen.append(int(cand_best))
            remaining.remove(cand_best)
        chosen_sorted = sorted(chosen)
        obj_final = estimate_fixed_subset_objective(
            chosen_sorted,
            table_support=table_support,
            U_support=U_support,
            calibration_systems=calibration_systems,
            sigma_y=sigma_y,
            alpha=alpha,
            noise_replicas=noise_replicas,
            rng=rng,
            margin=margin,
            u_grid=u_grid,
        )
        evaluated_g += 1
        if obj_final < best_obj_g:
            best_obj_g = obj_final
            best_subset_g = chosen_sorted
        meta[f"restart_{restart}"] = {"subset": chosen_sorted, "objective": obj_final}

    assert best_subset_g is not None
    return FixedSearchResult(
        subset=list(best_subset_g),
        objective=float(best_obj_g),
        n_candidates_evaluated=evaluated_g,
        search_mode="greedy_multistart",
        seed=int(seed),
        metadata=meta,
    )


def save_fixed_search(result: FixedSearchResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "subset": result.subset,
        "objective": result.objective,
        "n_candidates_evaluated": result.n_candidates_evaluated,
        "search_mode": result.search_mode,
        "seed": result.seed,
        "metadata": result.metadata,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_fixed_search(path: Path) -> FixedSearchResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FixedSearchResult(
        subset=list(payload["subset"]),
        objective=float(payload["objective"]),
        n_candidates_evaluated=int(payload["n_candidates_evaluated"]),
        search_mode=str(payload["search_mode"]),
        seed=int(payload["seed"]),
        metadata=dict(payload.get("metadata") or {}),
    )
