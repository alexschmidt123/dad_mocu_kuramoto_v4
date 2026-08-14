#!/usr/bin/env python3
"""Plan-2 T=2 Fixed-vs-adaptive diagnostic (ChatGPT brief §§11–14, H).

Uses the FROZEN bank ``data/ieee5_plan2_trap_v2`` and existing likelihood /
posterior-safe u_ctrl / MOCU machinery. Does not filter θ or redesign physics.

Outputs under ``tools/plan2_fixed_vs_adaptive/results/``.

Usage (repo root, conda env with deps)::

    python3 tools/plan2_fixed_vs_adaptive/run_t2_fixed_vs_adaptive.py \\
        --noise_sigma 0.01 --seed 101
    python3 tools/plan2_fixed_vs_adaptive/run_t2_fixed_vs_adaptive.py \\
        --noise_sigma 0.005 --seed 101
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config_for_run, repo_root
from src.banks.audit import (
    _one_step_scores,
    _terminal_u,
    _update,
)
from src.banks.power_grid import load_bank_from_path, resolve_dataset_dir
from src.observations.compress import build_centres_bank
from src.domains.swing.design import build_catalog

OUT_DIR = Path(__file__).resolve().parent / "results"


def _entropy(counts: np.ndarray) -> float:
    p = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def _effective_n(counts: np.ndarray) -> float:
    p = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    return float(1.0 / np.sum(p * p))


def diagnose_t2(
    *,
    Y: np.ndarray,
    U: np.ndarray,
    sigma: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    catalog: list[Any],
    n_outer: int,
    n_inner: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    """Branching frequencies + branch value vs Fixed second + adaptive planner."""
    Y = np.asarray(Y, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64).reshape(-1)
    grid = np.asarray(u_grid, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    n_obs = int(Y.shape[2])
    n_act = int(Y.shape[1])
    log_w0 = np.full(len(U), -np.log(len(U)), dtype=np.float64)

    # Prior one-step scores → myopic first + candidate set
    u0 = rng.random(n_inner)
    z0 = rng.normal(size=(n_inner, n_obs)) * float(sigma)
    prior_scores = _one_step_scores(
        Y, U, log_w0, set(),
        sigma=float(sigma), alpha=float(alpha), margin=float(margin),
        grid=grid, uniforms=u0, noise=z0,
    )
    myopic_first = int(np.argmin(prior_scores))
    candidates = np.argsort(prior_scores)[: min(int(top_k), n_act)].tolist()
    if myopic_first not in candidates:
        candidates.append(myopic_first)

    outer_idx = rng.integers(0, len(U), size=int(n_outer))
    outer_noise = rng.normal(size=(int(n_outer), n_obs)) * float(sigma)
    inner_uniforms = rng.random((int(n_outer), int(n_inner)))
    inner_noise = rng.normal(size=(int(n_outer), int(n_inner), n_obs)) * float(sigma)

    # Exhaustive Fixed among candidate×all (same MC outer/inner as screen)
    fixed_best = np.inf
    fixed_pair: list[int] | None = None
    fixed_scores: list[dict[str, Any]] = []
    for a1 in candidates:
        for a2 in range(n_act):
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
                    _terminal_u(U, lw, alpha=float(alpha), margin=float(margin), grid=grid)
                )
            score = float(np.mean(vals))
            fixed_scores.append({"a1": int(a1), "a2": int(a2), "J": score})
            if score < fixed_best:
                fixed_best = score
                fixed_pair = [int(a1), int(a2)]
    fixed_scores.sort(key=lambda d: d["J"])
    near_best = fixed_scores[:10]

    # Per first-action: adaptive second + branch value vs Fixed's second
    per_first: list[dict[str, Any]] = []
    planning_scores: dict[int, float] = {}
    for a1 in candidates:
        centres1 = Y[:, a1, :]
        a2_star_list: list[int] = []
        u_adapt: list[float] = []
        u_force_fixed: list[float] = []
        u_changed: list[int] = []
        # Fixed second for this a1 if Fixed pair starts with a1, else best Fixed pair's a2
        # Diagnostic C: force the second probe of the globally best Fixed pair when a1 matches,
        # else force the best open-loop a2 for this a1 from fixed_scores.
        best_for_a1 = min(
            (d for d in fixed_scores if d["a1"] == a1),
            key=lambda d: d["J"],
            default=None,
        )
        fixed_a2_for_a1 = (
            int(fixed_pair[1])
            if fixed_pair and fixed_pair[0] == a1
            else (int(best_for_a1["a2"]) if best_for_a1 else None)
        )

        for r, n in enumerate(outer_idx):
            y1 = centres1[int(n)] + outer_noise[r]
            lw1 = _update(log_w0, y1, centres1, float(sigma))
            s2 = _one_step_scores(
                Y, U, lw1, {int(a1)},
                sigma=float(sigma), alpha=float(alpha), margin=float(margin),
                grid=grid, uniforms=inner_uniforms[r], noise=inner_noise[r],
            )
            a2_star = int(np.argmin(s2))
            a2_star_list.append(a2_star)
            u_ad = float(s2[a2_star])
            u_adapt.append(u_ad)
            if fixed_a2_for_a1 is not None and fixed_a2_for_a1 != a1:
                u_fx = float(s2[fixed_a2_for_a1])
            else:
                u_fx = u_ad
            u_force_fixed.append(u_fx)
            # Does discrete u change? Compare terminal u on this outer sample
            # using one inner draw for second obs (same as Fixed path style).
            y2_ad = Y[int(n), a2_star] + inner_noise[r, 0]
            lw_ad = _update(lw1, y2_ad, Y[:, a2_star, :], float(sigma))
            u_term_ad = _terminal_u(
                U, lw_ad, alpha=float(alpha), margin=float(margin), grid=grid
            )
            if fixed_a2_for_a1 is not None and fixed_a2_for_a1 != a1:
                y2_fx = Y[int(n), fixed_a2_for_a1] + inner_noise[r, 0]
                lw_fx = _update(lw1, y2_fx, Y[:, fixed_a2_for_a1, :], float(sigma))
                u_term_fx = _terminal_u(
                    U, lw_fx, alpha=float(alpha), margin=float(margin), grid=grid
                )
            else:
                u_term_fx = u_term_ad
            u_changed.append(int(abs(u_term_ad - u_term_fx) > 1e-12))

        deltas = np.asarray(u_force_fixed) - np.asarray(u_adapt)  # >0 ⇒ branching helps
        uniq, counts = np.unique(a2_star_list, return_counts=True)
        order = np.argsort(-counts)
        uniq, counts = uniq[order], counts[order]
        mass = {str(int(a)): float(c / len(a2_star_list)) for a, c in zip(uniq, counts)}
        d = catalog[int(a1)]
        meta = {
            "amp": float(d.amplitude),
            "duration": float(d.duration),
            "bus": int(d.bus),
        }
        row = {
            "a1": int(a1),
            "a1_meta": meta,
            "J_adaptive": float(np.mean(u_adapt)),
            "fixed_a2_forced": fixed_a2_for_a1,
            "n_distinct_a2": int(len(uniq)),
            "most_common_a2": int(uniq[0]),
            "most_common_a2_prob": float(counts[0] / len(a2_star_list)),
            "a2_entropy": _entropy(counts),
            "a2_effective_n": _effective_n(counts),
            "a2_mass": mass,
            "branch_value_mean": float(np.mean(deltas)),
            "branch_value_median": float(np.median(deltas)),
            "branch_value_std": float(np.std(deltas)),
            "branch_value_q90": float(np.quantile(deltas, 0.90)),
            "branch_value_max": float(np.max(deltas)),
            "frac_branch_value_gt0": float(np.mean(deltas > 1e-12)),
            "frac_branch_value_gt_0p005": float(np.mean(deltas > 0.005)),
            "frac_terminal_u_changes": float(np.mean(u_changed)),
            "is_myopic_first": bool(a1 == myopic_first),
        }
        per_first.append(row)
        planning_scores[int(a1)] = row["J_adaptive"]

    best_first = min(planning_scores, key=planning_scores.get)
    plan_J = float(planning_scores[best_first])
    myopic_J = float(planning_scores[myopic_first])

    return {
        "sigma": float(sigma),
        "N_obs": int(n_obs),
        "n_actions": n_act,
        "n_outer": int(n_outer),
        "n_inner": int(n_inner),
        "support_size": int(len(U)),
        "myopic_first": myopic_first,
        "planning_first": int(best_first),
        "J_myopic": myopic_J,
        "J_planning": plan_J,
        "J_fixed_approx": float(fixed_best),
        "planning_minus_fixed": float(plan_J - float(fixed_best)),
        "planning_minus_myopic": float(plan_J - myopic_J),
        "approx_fixed_pair": fixed_pair,
        "fixed_near_best": near_best,
        "fixed_unique_dominant": bool(
            len(near_best) >= 2
            and (near_best[1]["J"] - near_best[0]["J"]) > 0.005
        ),
        "per_first": per_first,
        "catalog_note": (
            "Frozen Plan-2 v2 catalog from config (amp×duration×bus). "
            "ChatGPT brief listed D={0.05..0.20}; frozen YAML uses "
            "D={0.03,0.06,0.10,0.15,0.22,0.30}."
        ),
    }


def analyze_rollouts(exp_dir: Path) -> dict[str, Any]:
    """Learned-policy collapse diagnostics from existing eval rollouts.csv."""
    path = exp_dir / "eval" / "rollouts.csv"
    if not path.is_file():
        return {"error": f"missing {path}"}
    rows = list(csv.DictReader(path.open()))
    by: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)

    fixed_seqs = [r["sequence"] for r in by.get("Fixed", [])]
    fixed_mode = Counter(fixed_seqs).most_common(1)[0][0] if fixed_seqs else None
    out: dict[str, Any] = {"exp_dir": str(exp_dir), "fixed_mode_sequence": fixed_mode}
    methods = {}
    for m, rs in by.items():
        seqs = [r["sequence"] for r in rs]
        ctr = Counter(seqs)
        gaps = [float(r["control_gap"]) for r in rs]
        match = (
            sum(1 for s in seqs if s == fixed_mode) / len(seqs) if fixed_mode else 0.0
        )
        # second actions
        a2 = []
        for s in seqs:
            parts = s.split()
            if len(parts) >= 2:
                a2.append(parts[1])
        a2c = Counter(a2)
        methods[m] = {
            "n": len(rs),
            "mean_gap": float(np.mean(gaps)),
            "mean_u_ctrl": float(np.mean([float(r["u_ctrl"]) for r in rs])),
            "n_unique_sequences": len(ctr),
            "sequence_entropy": _entropy(np.array(list(ctr.values()))),
            "top_sequences": ctr.most_common(5),
            "frac_match_fixed_mode": float(match),
            "n_unique_a2": len(a2c),
            "a2_entropy": _entropy(np.array(list(a2c.values()))) if a2c else 0.0,
            "most_common_a2_prob": (
                float(a2c.most_common(1)[0][1] / len(a2)) if a2 else 0.0
            ),
        }
    out["methods"] = methods
    # primary table from eval_meta if present
    meta_path = exp_dir / "eval" / "eval_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        out["summaries"] = meta.get("summaries")
    return out


def _write_md(report: dict[str, Any], path: Path) -> None:
    t2 = report["t2"]
    lines = [
        "# Plan-2 T=2 Fixed vs adaptive diagnostic\n",
        f"- config: `{report['config']}`",
        f"- bank: `{report['data_dir']}`",
        f"- sigma: `{t2['sigma']}`, N_obs=`{t2['N_obs']}`, seed=`{report['seed']}`",
        f"- catalog note: {t2['catalog_note']}\n",
        "## Table A — screen objectives (expected terminal u_ctrl, lower better)\n",
        "| Method | J (approx) | notes |",
        "|--------|------------|-------|",
        f"| Myopic (greedy ξ1 + adaptive ξ2) | {t2['J_myopic']:.6f} | |",
        f"| T=2 adaptive planner | {t2['J_planning']:.6f} | best ξ1 under branch-optimal ξ2 |",
        f"| Optimized Fixed (approx on candidates) | {t2['J_fixed_approx']:.6f} | pair={t2['approx_fixed_pair']} |",
        f"| planning − Fixed | {t2['planning_minus_fixed']:.6f} | <0 ⇒ adaptive room |",
        f"| planning − Myopic | {t2['planning_minus_myopic']:.6f} | Myopic trap if <0 |\n",
        "## Fixed near-best subsets\n",
    ]
    for i, d in enumerate(t2["fixed_near_best"][:8]):
        lines.append(f"- #{i+1}: pair=({d['a1']},{d['a2']}) J={d['J']:.6f}")
    lines.append("\n## Table B — per-ξ1 branching structure\n")
    lines.append(
        "| a1 | bus | dur | J_adapt | n_a2 | P(mode a2) | H(a2) | "
        "eff_n | Δu_branch mean | frac Δu>0 | frac u changes |"
    )
    lines.append(
        "|----|-----|-----|---------|------|------------|-------|"
        "------|----------------|-----------|----------------|"
    )
    for r in sorted(t2["per_first"], key=lambda x: x["J_adaptive"]):
        m = r["a1_meta"]
        lines.append(
            f"| {r['a1']} | {m['bus']} | {m['duration']} | {r['J_adaptive']:.5f} | "
            f"{r['n_distinct_a2']} | {r['most_common_a2_prob']:.2f} | "
            f"{r['a2_entropy']:.3f} | {r['a2_effective_n']:.2f} | "
            f"{r['branch_value_mean']:.5f} | {r['frac_branch_value_gt0']:.2f} | "
            f"{r['frac_terminal_u_changes']:.2f} |"
        )

    # Aggregate for best planning first
    best = next(r for r in t2["per_first"] if r["a1"] == t2["planning_first"])
    lines += [
        "\n## Branching at planning-optimal ξ1\n",
        f"- a1={best['a1']} meta={best['a1_meta']}",
        f"- a2 mass={best['a2_mass']}",
        f"- most_common_prob={best['most_common_a2_prob']:.3f} "
        f"(≈open-loop if ≳0.9)",
        f"- mean branch value Δu={best['branch_value_mean']:.5f} "
        f"(Fixed-forced − adapt)",
        f"- frac histories with Δu>0: {best['frac_branch_value_gt0']:.3f}",
        f"- frac terminal discrete u changes: {best['frac_terminal_u_changes']:.3f}\n",
    ]

    if report.get("rollouts"):
        ro = report["rollouts"]
        lines.append("## Table H — learned policies on held-out rollouts\n")
        lines.append(
            "| method | mean_gap | uniq | H_seq | frac≡Fixed mode | uniq a2 | P(mode a2) |"
        )
        lines.append(
            "|--------|----------|------|-------|-----------------|---------|------------|"
        )
        for m, d in sorted(
            (ro.get("methods") or {}).items(), key=lambda kv: kv[1]["mean_gap"]
        ):
            lines.append(
                f"| {m} | {d['mean_gap']:.5f} | {d['n_unique_sequences']} | "
                f"{d['sequence_entropy']:.3f} | {d['frac_match_fixed_mode']:.2f} | "
                f"{d['n_unique_a2']} | {d['most_common_a2_prob']:.2f} |"
            )
        lines.append(f"\nFixed mode sequence: `{ro.get('fixed_mode_sequence')}`\n")

    # Category
    gap = t2["planning_minus_fixed"]
    mode_p = best["most_common_a2_prob"]
    bv = best["branch_value_mean"]
    if gap > -0.005 and mode_p >= 0.85 and bv < 0.005:
        cat = (
            "Category 1 — Fixed wins because adaptive room is intrinsically weak "
            "(planner ≈ Fixed; imbalanced ξ2; small branch value)."
        )
    elif gap <= -0.01 and (mode_p < 0.7 or bv >= 0.005):
        cat = (
            "Category 2 — Adaptive room exists (planner beats Fixed / meaningful "
            "branching); if learned methods still lose, training/collapse is suspect."
        )
    else:
        cat = (
            "Category 4 / mixed — planner edge and branching evidence are both "
            "modest; interpret with rollout Table H."
        )
    lines += ["## Preliminary category (T=2 screen)\n", cat, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/ieee5_plan2_trap.yaml")
    p.add_argument("--noise_sigma", type=float, default=0.01)
    p.add_argument("--N_obs", type=int, default=200)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--n_outer", type=int, default=48)
    p.add_argument("--n_inner", type=int, default=24)
    p.add_argument("--top_k", type=int, default=12)
    p.add_argument("--support_size", type=int, default=128)
    p.add_argument(
        "--exp-dir",
        default="",
        help="Optional completed T=2 experiment dir for rollout collapse table",
    )
    args = p.parse_args()

    root = repo_root()
    cfg = load_config_for_run(args.config, root, step_number=2)
    data_dir = resolve_dataset_dir(cfg, root)
    catalog = build_catalog(cfg)
    bank = load_bank_from_path(data_dir, cfg=cfg, skip_quality_check=True)
    train_df = np.asarray(bank["full_train"])
    train_rocof = np.asarray(bank["max_rocof_train"])
    train_U = np.asarray(bank["U_train"], dtype=np.float64).reshape(-1)

    control = dict(cfg.raw.get("control") or {})
    alpha = float(control.get("alpha", 0.01))
    margin = float(control.get("safety_margin", 0.0))
    u_grid = np.asarray(control.get("u_candidates"), dtype=np.float64)

    rng = np.random.default_rng(int(args.seed))
    n_pick = int(min(max(args.support_size, 8), len(train_U)))
    pick = np.sort(rng.choice(len(train_U), size=n_pick, replace=False))
    centres, _, mode = build_centres_bank(
        train_df[pick], train_rocof[pick], int(args.N_obs)
    )
    Y = np.transpose(np.asarray(centres, dtype=np.float64), (1, 0, 2))
    U = train_U[pick]

    print(f"bank={data_dir} n_actions={len(catalog)} support={n_pick} obs_mode={mode}")
    print(f"designs head={[d.as_tuple() for d in catalog[:3]]} ...")

    t2 = diagnose_t2(
        Y=Y, U=U, sigma=float(args.noise_sigma), alpha=alpha, margin=margin,
        u_grid=u_grid, catalog=catalog,
        n_outer=int(args.n_outer), n_inner=int(args.n_inner),
        top_k=int(args.top_k), seed=int(args.seed) + 17,
    )
    print(
        f"σ={args.noise_sigma}: J_plan={t2['J_planning']:.5f} "
        f"J_fixed={t2['J_fixed_approx']:.5f} "
        f"plan−fixed={t2['planning_minus_fixed']:.5f} "
        f"fixed_pair={t2['approx_fixed_pair']}"
    )
    best = next(r for r in t2["per_first"] if r["a1"] == t2["planning_first"])
    print(
        f"  at plan ξ1={t2['planning_first']}: "
        f"P(mode a2)={best['most_common_a2_prob']:.3f} "
        f"H={best['a2_entropy']:.3f} "
        f"Δu_branch={best['branch_value_mean']:.5f} "
        f"frac_u_change={best['frac_terminal_u_changes']:.3f}"
    )

    rollouts = None
    if args.exp_dir:
        rollouts = analyze_rollouts(Path(args.exp_dir))
    else:
        # auto-find newest T=2 cell for this sigma
        tag = str(args.noise_sigma).replace(".", "p")
        cands = sorted(
            (ROOT / "experiments").glob(
                f"*ieee5_plan2_trap_Uctrl_T2_Nobs{args.N_obs}_sigma{tag}"
            ),
            key=lambda p: p.stat().st_mtime,
        )
        # prefer v2 sweep stamps >= 145803
        cands = [c for c in cands if "08112026_1" in c.name or "08112026_2" in c.name]
        if cands:
            rollouts = analyze_rollouts(cands[-1])
            print(f"rollouts from {cands[-1].name}")

    report = {
        "config": args.config,
        "data_dir": str(data_dir),
        "seed": int(args.seed),
        "t2": t2,
        "rollouts": rollouts,
        "fixed_search_note": (
            f"C(30,2)=435; this screen scores top_k×(29) pairs with MC, "
            f"not full exhaustive Fixed used in evaluate. Full Fixed lives in "
            f"context._exhaustive_fixed_sequence when C(n,T) under threshold."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sigma_tag = str(args.noise_sigma).replace(".", "p")
    json_path = OUT_DIR / f"t2_diag_sigma{sigma_tag}.json"
    md_path = OUT_DIR / f"t2_diag_sigma{sigma_tag}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_md(report, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
