"""Read-only N_obs=120 structural check for objective-based sBOED banks.

This intentionally does not train a neural policy.  It asks whether vector
observations create observation-dependent second actions and whether a
two-step adaptive lookahead can improve on a one-step Myopic policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import load_config_for_run
from src.control.posterior_ctrl import normalize_log_weights, posterior_safe_u_ctrl
from src.observations.compress import build_centres_bank
from src.observations.likelihood import vector_gaussian_loglik


def update(log_w: np.ndarray, y: np.ndarray, centres: np.ndarray, sigma: float) -> np.ndarray:
    return log_w + vector_gaussian_loglik(y, centres, sigma)


def terminal_u(U: np.ndarray, log_w: np.ndarray, alpha: float, grid: np.ndarray) -> float:
    return float(
        posterior_safe_u_ctrl(
            U,
            normalize_log_weights(log_w),
            alpha,
            margin=0.0,
            u_grid=grid,
            snap_up=True,
        )
    )


def posterior_indices(log_w: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(normalize_log_weights(log_w))
    return np.searchsorted(cdf, np.minimum(uniforms, 1.0 - 1e-12), side="left")


def one_step_scores(
    Y: np.ndarray,
    U: np.ndarray,
    log_w: np.ndarray,
    used: set[int],
    sigma: float,
    alpha: float,
    grid: np.ndarray,
    uniforms: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    scores = np.full(Y.shape[1], np.inf, dtype=np.float64)
    idx = posterior_indices(log_w, uniforms)
    for a in range(Y.shape[1]):
        if a in used:
            continue
        vals = []
        centres = Y[:, a, :]
        for j, n in enumerate(idx):
            y = centres[n] + noise[j]
            vals.append(terminal_u(U, update(log_w, y, centres, sigma), alpha, grid))
        scores[a] = float(np.mean(vals))
    return scores


def check_system(
    root: Path,
    system: str,
    n_obs: int,
    sigma: float,
    support_size: int,
    n_outer: int,
    n_inner: int,
    top_k: int,
    seed: int,
) -> dict:
    cfg = load_config_for_run(str(root / "configs" / f"{system}.yaml"), root)
    data = root / "data" / system
    full = np.load(data / "train" / "delta_f.npy", mmap_mode="r")
    U_all = np.load(data / "train" / "U.npy").astype(np.float64)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(U_all), size=min(support_size, len(U_all)), replace=False))
    centres_a_n_d, indices, _ = build_centres_bank(full[pick], None, n_obs)
    Y = np.transpose(centres_a_n_d, (1, 0, 2))
    U = U_all[pick]
    control = dict(cfg.raw.get("control") or {})
    alpha = float(control.get("alpha", 0.05))
    grid = np.asarray(control.get("u_candidates"), dtype=np.float64)
    log_w0 = np.full(len(U), -np.log(len(U)), dtype=np.float64)

    # Shared standard-normal draws; sigma is applied here so comparisons are CRN.
    u0 = rng.random(n_inner)
    z0 = rng.normal(size=(n_inner, n_obs)) * sigma
    prior_scores = one_step_scores(
        Y, U, log_w0, set(), sigma, alpha, grid, u0, z0
    )
    myopic_first = int(np.argmin(prior_scores))
    candidates = np.argsort(prior_scores)[: min(top_k, Y.shape[1])].tolist()
    if myopic_first not in candidates:
        candidates.append(myopic_first)

    outer_idx = rng.integers(0, len(U), size=n_outer)
    outer_noise = rng.normal(size=(n_outer, n_obs)) * sigma
    inner_uniforms = rng.random((n_outer, n_inner))
    inner_noise = rng.normal(size=(n_outer, n_inner, n_obs)) * sigma

    planning_scores: dict[int, float] = {}
    branch_actions: dict[int, list[int]] = {}
    for a1 in candidates:
        vals = []
        seconds = []
        centres1 = Y[:, a1, :]
        for r, n in enumerate(outer_idx):
            y1 = centres1[n] + outer_noise[r]
            lw1 = update(log_w0, y1, centres1, sigma)
            s2 = one_step_scores(
                Y,
                U,
                lw1,
                {a1},
                sigma,
                alpha,
                grid,
                inner_uniforms[r],
                inner_noise[r],
            )
            a2 = int(np.argmin(s2))
            seconds.append(a2)
            vals.append(float(s2[a2]))
        planning_scores[int(a1)] = float(np.mean(vals))
        branch_actions[int(a1)] = seconds

    best_first = min(planning_scores, key=planning_scores.get)
    plan_support = float(planning_scores[best_first])

    # Myopic T=2 support estimate: prior-myopic first, posterior-myopic second.
    myopic_support = float(planning_scores[myopic_first])

    # Approximate Fixed T=2 over the screened first candidates and full seconds.
    fixed_best = np.inf
    fixed_pair = None
    for a1 in candidates:
        for a2 in range(Y.shape[1]):
            if a2 == a1:
                continue
            vals = []
            for r, n in enumerate(outer_idx):
                lw = update(log_w0, Y[n, a1] + outer_noise[r], Y[:, a1, :], sigma)
                # Independent second noise, shared across candidate pairs.
                y2 = Y[n, a2] + inner_noise[r, 0]
                lw = update(lw, y2, Y[:, a2, :], sigma)
                vals.append(terminal_u(U, lw, alpha, grid))
            score = float(np.mean(vals))
            if score < fixed_best:
                fixed_best = score
                fixed_pair = [int(a1), int(a2)]

    seconds = branch_actions[best_first]
    uniq, counts = np.unique(seconds, return_counts=True)
    branch_mass = {str(int(a)): float(c / len(seconds)) for a, c in zip(uniq, counts)}
    return {
        "system": system,
        "N_obs": n_obs,
        "sigma_frequency_hz": sigma,
        "support_size": len(U),
        "n_actions": int(Y.shape[1]),
        "obs_indices_head": indices[:5].tolist(),
        "obs_indices_tail": indices[-5:].tolist(),
        "margin_for_structure_check": 0.0,
        "myopic_first_design": myopic_first,
        "planning_first_design": int(best_first),
        "approx_fixed_pair": fixed_pair,
        "J_myopic_T2_support": myopic_support,
        "J_planning_T2_support": plan_support,
        "J_fixed_T2_support_approx": float(fixed_best),
        "planning_minus_myopic": plan_support - myopic_support,
        "planning_minus_fixed": plan_support - float(fixed_best),
        "n_distinct_second_actions": int(len(uniq)),
        "second_action_mass": branch_mass,
        "first_candidates": [int(x) for x in candidates],
        "search_note": (
            "Approximate Monte Carlo structural screen: top-k first actions, "
            "full second-action catalog, margin=0. Not a trained-policy result."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--n-obs", type=int, default=120)
    p.add_argument("--sigmas", default="0.001,0.005,0.01,0.02,0.2")
    p.add_argument("--systems", default="ieee5,ieee9,ieee14")
    p.add_argument("--support-size", type=int, default=96)
    p.add_argument("--n-outer", type=int, default=24)
    p.add_argument("--n-inner", type=int, default=16)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260723)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    rows = []
    systems = tuple(x.strip() for x in args.systems.split(",") if x.strip())
    for si, system in enumerate(systems):
        for sigma in (float(x) for x in args.sigmas.split(",")):
            row = check_system(
                args.root,
                system,
                args.n_obs,
                sigma,
                args.support_size,
                args.n_outer,
                args.n_inner,
                args.top_k,
                # Same support particles and standardized random stream across
                # sigma values for a valid noise-sensitivity comparison.
                args.seed + 1000 * si,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    payload = {
        "purpose": "N_obs vector-observation structural advantage screen",
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
