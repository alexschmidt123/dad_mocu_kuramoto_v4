"""Validation-only Myopic n_hypothetical convergence and production freeze."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.control.banks import extract_U_bank
from src.control.myopic import MyopicControlSelector, score_all_actions_myopic
from src.control.pilot import load_pilot_splits
from src.control.posterior_ctrl import normalize_log_weights
from src.control.terminal_rule import (
    load_frozen_terminal_rule,
    observe_with_keyed_noise,
    posterior_to_u_ctrl,
)
from src.control.u_req import ControlSpec
from src.contrastive.spce import log_prior_uniform_discrete
from src.rollout import update_log_weights
from src.run_context import load_experiment_run
from src.swing_equation_ode.design import build_catalog
from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables


DEFAULT_GRID = [16, 32, 64, 128, 256, 512, 1024]
DEFAULT_SEEDS = [11, 22, 33, 44, 55]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _thresholds_from_cfg(raw: dict[str, Any]) -> dict[str, float]:
    sel = dict((raw.get("myopic") or {}).get("selection") or {})
    return {
        "agreement_min": float(sel.get("agreement_min", 0.95)),
        "rank_corr_min": float(sel.get("rank_corr_min", 0.95)),
        "mean_u_diff_max": float(sel.get("mean_u_diff_max", 0.005)),
        "seed_std_max": float(sel.get("seed_std_max", 0.005)),
    }


def _score_vector(
    scores: dict[int, float], n_actions: int
) -> np.ndarray:
    v = np.full(n_actions, np.nan, dtype=np.float64)
    for a, j in scores.items():
        v[int(a)] = float(j)
    return v


def _run_validation_rollouts(
    *,
    n_h: int,
    systems: list[dict[str, Any]],
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    frozen,
    n_actions: int,
    sigma_y: float,
    horizon: int,
    n_rollouts: int,
    global_seed: int,
    method_seed: int,
) -> dict[str, Any]:
    """Posterior terminal u_ctrl + action traces on validation systems (no test)."""
    rng = np.random.default_rng(method_seed)
    selector = MyopicControlSelector(
        table_support=table_support,
        U_support=U_support,
        n_actions=n_actions,
        sigma_y=sigma_y,
        alpha=frozen.alpha,
        n_hypothetical=n_h,
        safety_margin=frozen.margin,
        u_candidates=frozen.u_candidates,
    )
    u_vals: list[float] = []
    seqs: list[list[int]] = []
    gaps: list[float] = []
    exact_ties: list[int] = []
    near_ties: list[int] = []
    decision_runtimes: list[float] = []
    rollout_runtimes: list[float] = []
    score_snapshots: list[dict[str, Any]] = []

    for rid in range(n_rollouts):
        t_roll = time.perf_counter()
        sys = systems[int(rid % len(systems))]
        log_w = np.asarray(table_support.log_p0, dtype=np.float64).copy()
        used: set[int] = set()
        seq: list[int] = []
        step_scores: list[dict[int, float]] = []
        for step in range(horizon):
            weights = normalize_log_weights(log_w)
            t0 = time.perf_counter()
            a = int(
                selector.select(
                    used=used, log_weights=log_w, weights=weights, rng=rng
                )
            )
            decision_runtimes.append(time.perf_counter() - t0)
            scores = dict(getattr(selector, "last_scores", {}))
            step_scores.append(scores)
            ordered = sorted(scores, key=lambda x: (scores[x], x))
            best = scores[ordered[0]]
            gap = (
                float(scores[ordered[1]] - best)
                if len(ordered) > 1
                else float("nan")
            )
            gaps.append(gap)
            exact_ties.append(
                sum(1 for j in scores.values() if abs(j - best) <= 1e-12) - 1
            )
            near_ties.append(
                sum(1 for j in scores.values() if abs(j - best) <= 1e-3) - 1
            )
            y = observe_with_keyed_noise(
                sys,
                a,
                sigma_y=sigma_y,
                global_seed=global_seed,
                theta_id=int(rid % len(systems)),
                rollout_id=rid,
                step=step,
            )
            centres = y_sim_last_step_from_tables(table_support, [a])
            log_w = update_log_weights(log_w, y, centres, sigma_y)
            seq.append(a)
            used.add(a)
        w = normalize_log_weights(log_w)
        u_vals.append(float(posterior_to_u_ctrl(w, U_support, frozen)))
        seqs.append(seq)
        score_snapshots.append(
            {
                "step0": _score_vector(step_scores[0], n_actions).tolist()
                if step_scores
                else [],
                "step1": _score_vector(step_scores[1], n_actions).tolist()
                if len(step_scores) > 1
                else [],
            }
        )
        rollout_runtimes.append(time.perf_counter() - t_roll)

    # Empirical SE of first-step scores via extra CRN replicates on first context
    se_vals: list[float] = []
    log_w0 = np.asarray(table_support.log_p0, dtype=np.float64).copy()
    w0 = normalize_log_weights(log_w0)
    score_mat = []
    for r in range(min(24, max(8, n_rollouts // 4))):
        sc = score_all_actions_myopic(
            MyopicControlSelector(
                table_support=table_support,
                U_support=U_support,
                n_actions=n_actions,
                sigma_y=sigma_y,
                alpha=frozen.alpha,
                n_hypothetical=n_h,
                safety_margin=frozen.margin,
                u_candidates=frozen.u_candidates,
            ),
            used=set(),
            log_weights=log_w0,
            weights=w0,
            rng=np.random.default_rng(method_seed + 10_000 + r),
        )
        score_mat.append(_score_vector(sc, n_actions))
    score_mat_a = np.stack(score_mat, axis=0)
    mean_se = float(np.nanmean(np.nanstd(score_mat_a, axis=0, ddof=1)))

    return {
        "mean_u_ctrl": float(np.mean(u_vals)),
        "std_u_ctrl": float(np.std(u_vals)),
        "sequences": seqs,
        "mean_best_second_score_gap": float(np.nanmean(gaps)),
        "mean_score_standard_error": mean_se,
        "exact_tie_frequency": float(np.mean(np.asarray(exact_ties) > 0)),
        "near_tie_frequency": float(np.mean(np.asarray(near_ties) > 0)),
        "mean_exact_tie_count": float(np.mean(exact_ties)),
        "mean_near_tie_count": float(np.mean(near_ties)),
        "runtime_per_decision": float(np.mean(decision_runtimes)),
        "runtime_per_rollout": float(np.mean(rollout_runtimes)),
        "score_snapshots": score_snapshots,
    }


def _agreement(seqs_a: list[list[int]], seqs_b: list[list[int]]) -> dict[str, float]:
    n = min(len(seqs_a), len(seqs_b))
    if n == 0:
        return {
            "selected_action_agreement_with_1024": float("nan"),
            "step_0_action_agreement": float("nan"),
            "step_1_action_agreement": float("nan"),
        }
    full = []
    s0 = []
    s1 = []
    for i in range(n):
        a, b = seqs_a[i], seqs_b[i]
        full.append(a == b)
        if a and b:
            s0.append(a[0] == b[0])
        if len(a) > 1 and len(b) > 1:
            s1.append(a[1] == b[1])
    return {
        "selected_action_agreement_with_1024": float(np.mean(full)),
        "step_0_action_agreement": float(np.mean(s0)) if s0 else float("nan"),
        "step_1_action_agreement": float(np.mean(s1)) if s1 else float("nan"),
    }


def _rank_corr(
    snaps_a: list[dict[str, Any]], snaps_b: list[dict[str, Any]]
) -> float:
    corrs: list[float] = []
    for sa, sb in zip(snaps_a, snaps_b):
        for key in ("step0", "step1"):
            va = np.asarray(sa.get(key) or [], dtype=np.float64)
            vb = np.asarray(sb.get(key) or [], dtype=np.float64)
            if va.size == 0 or vb.size == 0 or va.size != vb.size:
                continue
            mask = np.isfinite(va) & np.isfinite(vb)
            if mask.sum() < 3:
                continue
            rho, _ = spearmanr(va[mask], vb[mask])
            if np.isfinite(rho):
                corrs.append(float(rho))
    return float(np.mean(corrs)) if corrs else float("nan")


def run_myopic_convergence(
    exp_dir: Path | str,
    *,
    project_root: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Select smallest n_hypothetical meeting validation stability thresholds.

    Uses validation systems only. Never touches the final test set for selection.
    """
    from src.config import repo_root

    root = project_root or repo_root()
    run = load_experiment_run(Path(exp_dir), root)
    exp_dir = run.exp_dir
    frozen = load_frozen_terminal_rule(exp_dir)
    splits = load_pilot_splits(exp_dir, run)
    val_systems = splits["validation_systems"]
    if not val_systems:
        raise RuntimeError("No validation systems available for Myopic convergence.")

    raw = run.cfg.raw
    myopic_cfg = dict(raw.get("myopic") or {})
    sel_cfg = dict(myopic_cfg.get("selection") or {})
    grid = [int(x) for x in sel_cfg.get("n_hypothetical_grid", DEFAULT_GRID)]
    seeds = [int(x) for x in sel_cfg.get("estimator_seeds", DEFAULT_SEEDS)]
    ref_n = int(myopic_cfg.get("reference_n_hypothetical", 1024))
    if ref_n not in grid:
        grid = sorted(set(grid + [ref_n]))
    n_rollouts = int(sel_cfg.get("n_validation_rollouts", 64))
    thresholds = _thresholds_from_cfg(raw)

    catalog = build_catalog(run.cfg)
    n_actions = len(catalog)
    horizon = int(run.cfg.step_number)
    sigma_y = float(run.cfg.sigma_y)
    table_support = TableThetaSupport(
        systems=splits["support_systems"],
        log_p0=log_prior_uniform_discrete(len(splits["support_systems"])),
    )
    U_support = extract_U_bank(splits["support_systems"])
    global_seed = int((raw.get("pilot") or {}).get("global_seed", 1234))

    out_dir = Path(
        out_dir
        or (exp_dir / "diagnostics" / "myopic_convergence")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"=== Myopic convergence (validation only) ===\n"
        f"  grid={grid}  ref={ref_n}  seeds={seeds}  n_rollouts={n_rollouts}\n"
        f"  thresholds={thresholds}"
    )

    # Reference results per seed at ref_n
    ref_by_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        print(f"  reference n_h={ref_n} seed={seed} ...")
        ref_by_seed[seed] = _run_validation_rollouts(
            n_h=ref_n,
            systems=val_systems,
            table_support=table_support,
            U_support=U_support,
            frozen=frozen,
            n_actions=n_actions,
            sigma_y=sigma_y,
            horizon=horizon,
            n_rollouts=n_rollouts,
            global_seed=global_seed,
            method_seed=seed,
        )

    rows: list[dict[str, Any]] = []
    by_nh: dict[str, Any] = {}
    for n_h in grid:
        seed_means: list[float] = []
        seed_rows: list[dict[str, Any]] = []
        agreements = []
        step0_agreements = []
        step1_agreements = []
        rank_corrs = []
        for seed in seeds:
            print(f"  n_h={n_h} seed={seed} ...")
            if n_h == ref_n:
                res = ref_by_seed[seed]
            else:
                res = _run_validation_rollouts(
                    n_h=n_h,
                    systems=val_systems,
                    table_support=table_support,
                    U_support=U_support,
                    frozen=frozen,
                    n_actions=n_actions,
                    sigma_y=sigma_y,
                    horizon=horizon,
                    n_rollouts=n_rollouts,
                    global_seed=global_seed,
                    method_seed=seed,
                )
            ref = ref_by_seed[seed]
            agr = _agreement(res["sequences"], ref["sequences"])
            rho = _rank_corr(res["score_snapshots"], ref["score_snapshots"])
            seed_means.append(res["mean_u_ctrl"])
            agreements.append(agr["selected_action_agreement_with_1024"])
            step0_agreements.append(agr["step_0_action_agreement"])
            step1_agreements.append(agr["step_1_action_agreement"])
            rank_corrs.append(rho)
            seed_rows.append(
                {
                    "seed": seed,
                    "validation_mean_u_ctrl": res["mean_u_ctrl"],
                    "validation_std_u_ctrl": res["std_u_ctrl"],
                    **agr,
                    "candidate_rank_correlation_with_1024": rho,
                    "mean_best_second_score_gap": res["mean_best_second_score_gap"],
                    "mean_score_standard_error": res["mean_score_standard_error"],
                    "exact_tie_frequency": res["exact_tie_frequency"],
                    "near_tie_frequency": res["near_tie_frequency"],
                    "runtime_per_decision": res["runtime_per_decision"],
                    "runtime_per_rollout": res["runtime_per_rollout"],
                }
            )

        mean_u = float(np.mean(seed_means))
        std_across_seeds = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
        ref_mean = float(np.mean([ref_by_seed[s]["mean_u_ctrl"] for s in seeds]))
        mean_u_diff = abs(mean_u - ref_mean)
        summary = {
            "n_hypothetical": n_h,
            "validation_mean_u_ctrl": mean_u,
            "validation_std_u_ctrl": float(
                np.mean([r["validation_std_u_ctrl"] for r in seed_rows])
            ),
            "repeated_seed_std_of_validation_mean_u_ctrl": std_across_seeds,
            "validation_mean_u_ctrl_difference_from_1024": mean_u_diff,
            "selected_action_agreement_with_1024": float(np.mean(agreements)),
            "step_0_action_agreement": float(np.nanmean(step0_agreements)),
            "step_1_action_agreement": float(np.nanmean(step1_agreements)),
            "candidate_rank_correlation_with_1024": float(np.nanmean(rank_corrs)),
            "mean_best_second_score_gap": float(
                np.mean([r["mean_best_second_score_gap"] for r in seed_rows])
            ),
            "mean_score_standard_error": float(
                np.mean([r["mean_score_standard_error"] for r in seed_rows])
            ),
            "exact_tie_frequency": float(
                np.mean([r["exact_tie_frequency"] for r in seed_rows])
            ),
            "near_tie_frequency": float(
                np.mean([r["near_tie_frequency"] for r in seed_rows])
            ),
            "runtime_per_decision": float(
                np.mean([r["runtime_per_decision"] for r in seed_rows])
            ),
            "runtime_per_rollout": float(
                np.mean([r["runtime_per_rollout"] for r in seed_rows])
            ),
            "per_seed": seed_rows,
        }
        summary["passes_thresholds"] = bool(
            summary["selected_action_agreement_with_1024"] >= thresholds["agreement_min"]
            and summary["candidate_rank_correlation_with_1024"]
            >= thresholds["rank_corr_min"]
            and summary["validation_mean_u_ctrl_difference_from_1024"]
            <= thresholds["mean_u_diff_max"]
            and summary["repeated_seed_std_of_validation_mean_u_ctrl"]
            <= thresholds["seed_std_max"]
        )
        by_nh[str(n_h)] = summary
        rows.append(summary)
        print(
            f"  → n_h={n_h}: agree={summary['selected_action_agreement_with_1024']:.3f} "
            f"rho={summary['candidate_rank_correlation_with_1024']:.3f} "
            f"|Δu|={mean_u_diff:.4f} seed_std={std_across_seeds:.4f} "
            f"pass={summary['passes_thresholds']}"
        )

    # Smallest passing count below ref; else ref
    selected = ref_n
    for n_h in sorted(x for x in grid if x < ref_n):
        if by_nh[str(n_h)]["passes_thresholds"]:
            selected = n_h
            break
    if selected == ref_n and by_nh[str(ref_n)]["passes_thresholds"]:
        # Prefer smallest that passes including checking if any below passed
        pass
    elif selected == ref_n:
        # force 1024 if nothing smaller passed
        selected = ref_n

    # If something smaller passed we already set it; if none, use 1024
    smaller_pass = [
        n for n in sorted(x for x in grid if x < ref_n) if by_nh[str(n)]["passes_thresholds"]
    ]
    selected = int(smaller_pass[0]) if smaller_pass else int(ref_n)

    report = {
        "selection_source": "validation_convergence",
        "reference_n_hypothetical": ref_n,
        "selected_n_hypothetical": selected,
        "thresholds": thresholds,
        "n_validation_systems": len(val_systems),
        "n_validation_rollouts": n_rollouts,
        "estimator_seeds": seeds,
        "grid": grid,
        "terminal_rule_hash": frozen.terminal_rule_hash,
        "used_test_systems": False,
        "by_n_hypothetical": by_nh,
        "rows": rows,
    }
    _write_json(out_dir / "convergence_report.json", report)
    _write_markdown(out_dir / "convergence_report.md", report)

    # Freeze into experiment run_config and return payload for parent config update
    run_cfg_path = exp_dir / "run_config.yaml"
    if run_cfg_path.is_file():
        import yaml

        cfg_data = yaml.safe_load(run_cfg_path.read_text()) or {}
        cfg_data.setdefault("control", {})["myopic_hypothetical"] = selected
        cfg_data["myopic"] = {
            "n_hypothetical": selected,
            "selection_source": "validation_convergence",
            "reference_n_hypothetical": ref_n,
            "selection": {
                **thresholds,
                "n_hypothetical_grid": grid,
                "estimator_seeds": seeds,
                "n_validation_rollouts": n_rollouts,
            },
        }
        run_cfg_path.write_text(yaml.safe_dump(cfg_data, sort_keys=False), encoding="utf-8")

    print(f"\nSelected production Myopic n_hypothetical = {selected}")
    print(f"Report → {out_dir}")
    return report


def freeze_into_source_config(
    config_path: Path, selected: int, report: dict[str, Any]
) -> None:
    import yaml

    data = yaml.safe_load(config_path.read_text()) or {}
    data.setdefault("control", {})["myopic_hypothetical"] = int(selected)
    data["myopic"] = {
        "n_hypothetical": int(selected),
        "selection_source": "validation_convergence",
        "reference_n_hypothetical": int(report["reference_n_hypothetical"]),
        "selection": report["thresholds"]
        | {
            "n_hypothetical_grid": report["grid"],
            "estimator_seeds": report["estimator_seeds"],
            "n_validation_rollouts": report["n_validation_rollouts"],
        },
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Myopic n_hypothetical validation convergence",
        "",
        f"**Selected:** `{report['selected_n_hypothetical']}`",
        f"**Reference:** `{report['reference_n_hypothetical']}`",
        f"**Selection source:** validation only (test unused)",
        f"**Terminal rule hash:** `{report['terminal_rule_hash']}`",
        "",
        "## Thresholds",
        "",
        "```",
        json.dumps(report["thresholds"], indent=2),
        "```",
        "",
        "## Results",
        "",
        "| n_h | agree@1024 | rank ρ | |Δmean u| | seed std | pass | runtime/roll |",
        "|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['n_hypothetical']} | "
            f"{row['selected_action_agreement_with_1024']:.3f} | "
            f"{row['candidate_rank_correlation_with_1024']:.3f} | "
            f"{row['validation_mean_u_ctrl_difference_from_1024']:.4f} | "
            f"{row['repeated_seed_std_of_validation_mean_u_ctrl']:.4f} | "
            f"{row['passes_thresholds']} | "
            f"{row['runtime_per_rollout']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
