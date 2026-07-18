"""Amplitude adaptive-value diagnostic under continuous vs snapped u_ctrl."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.control.banks import extract_U_bank
from src.control.continuous_uctrl_amplitude import OUT, ROOT
from src.control.continuous_uctrl_amplitude.audit import (
    freeze_continuous_terminal_rule,
    write_u_bank_audit,
)
from src.control.pilot import load_pilot_splits
from src.control.posterior_batch import centres_matrix, expected_u_after_action
from src.control.posterior_ctrl import normalize_log_weights, posterior_control_decision
from src.control.terminal_rule import FrozenTerminalRule, load_frozen_terminal_rule
from src.contrastive.spce import log_prior_uniform_discrete
from src.experiment_layout import (
    RunMetadata,
    ensure_standard_layout,
    git_commit_hash,
    utc_now_stamp,
    write_run_metadata,
    write_study_run_config,
)
from src.run_context import load_experiment_run
from src.swing_equation_ode.design import build_catalog
from src.table_scoring import TableThetaSupport


NEAR_TIE_TOL = 1e-4


@dataclass
class StudyBundle:
    system: str
    horizon: int
    centres: np.ndarray
    U: np.ndarray
    log_p0: np.ndarray
    sigma_y: float
    alpha: float
    margin: float
    u_grid: np.ndarray
    catalog: list[Any]
    amplitudes: list[float]
    buses: list[int]
    n_actions: int
    continuous_rule: FrozenTerminalRule
    snapped_rule: FrozenTerminalRule
    train_val_systems: list[dict[str, Any]]
    fixed_sequence: list[int]
    probe_duration: float


def load_bundle(system: str) -> StudyBundle:
    exp = ROOT / "experiments" / f"{system}_T3"
    run = load_experiment_run(exp, ROOT)
    splits = load_pilot_splits(exp, run)
    snapped = load_frozen_terminal_rule(exp)
    continuous = freeze_continuous_terminal_rule(system)
    support_systems = list(splits["support_systems"])
    support = TableThetaSupport(
        systems=support_systems,
        log_p0=log_prior_uniform_discrete(len(support_systems)),
    )
    U = np.asarray(extract_U_bank(support.systems), dtype=np.float64).reshape(-1)
    catalog = build_catalog(run.cfg)
    amps = sorted({float(d.amplitude) for d in catalog})
    buses = sorted({int(d.bus) for d in catalog})
    fixed_path = exp / "eval" / "fixed" / "subset_meta.json"
    fixed_seq = (
        [int(x) for x in json.loads(fixed_path.read_text())["selected_action_ids"]]
        if fixed_path.is_file()
        else []
    )
    train_val = list(splits["support_systems"]) + list(splits["validation_systems"])
    return StudyBundle(
        system=system,
        horizon=3,
        centres=centres_matrix(support, len(catalog)),
        U=U,
        log_p0=np.asarray(support.log_p0, dtype=np.float64),
        sigma_y=float(run.cfg.sigma_y),
        alpha=float(continuous.alpha),
        margin=float(continuous.margin),
        u_grid=np.asarray(continuous.u_candidates, dtype=np.float64),
        catalog=catalog,
        amplitudes=amps,
        buses=buses,
        n_actions=len(catalog),
        continuous_rule=continuous,
        snapped_rule=snapped,
        train_val_systems=train_val,
        fixed_sequence=fixed_seq,
        probe_duration=float(run.cfg.probe_duration),
    )


def _action_id(bundle: StudyBundle, bus: int, amp: float) -> int:
    for i, d in enumerate(bundle.catalog):
        if int(d.bus) == int(bus) and abs(float(d.amplitude) - float(amp)) < 1e-12:
            return i
    raise KeyError(f"no design bus={bus} amp={amp}")


def _design_of(bundle: StudyBundle, action: int) -> tuple[int, float]:
    d = bundle.catalog[int(action)]
    return int(d.bus), float(d.amplitude)


def load_h1_histories(
    bundle: StudyBundle,
    *,
    max_histories: int,
    smoke: bool,
) -> list[dict[str, Any]]:
    """Reuse objective_adaptive_value first-step histories (train/val only)."""
    path = (
        ROOT
        / "experiments"
        / "objective_adaptive_value"
        / f"{bundle.system}_T3"
        / "first_history_results.csv"
    )
    rows: list[dict[str, Any]] = []
    # h0 prior history
    rows.append(
        {
            "history_id": -1,
            "history_step": 0,
            "previous_actions": [],
            "previous_observations": [],
            "xi1": None,
            "y1": None,
            "theta_id": None,
            "log_w": bundle.log_p0.copy(),
            "source": "prior_h0",
        }
    )
    if not path.is_file():
        return rows[: 1 + (8 if smoke else max_histories)]
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if smoke and i >= 12:
                break
            if not smoke and i >= max_histories:
                break
            xi1 = int(row["xi1"])
            y1 = float(row["y1"])
            s2 = float(bundle.sigma_y) ** 2
            centre = bundle.centres[xi1]
            log_L = (
                -0.5 * math.log(2.0 * math.pi * s2)
                - 0.5 * ((y1 - centre) ** 2) / s2
            )
            log_w = bundle.log_p0 + log_L
            rows.append(
                {
                    "history_id": int(row["history_id"]),
                    "history_step": 1,
                    "previous_actions": [xi1],
                    "previous_observations": [y1],
                    "xi1": xi1,
                    "y1": y1,
                    "theta_id": int(row["theta_id"]) if row.get("theta_id") not in (None, "") else None,
                    "log_w": log_w,
                    "source": "objective_adaptive_value/first_history_results.csv",
                }
            )
    return rows


def score_history_designs(
    bundle: StudyBundle,
    history: dict[str, Any],
    *,
    n_hyp: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    log_w = np.asarray(history["log_w"], dtype=np.float64)
    w = normalize_log_weights(log_w)
    used = set(int(a) for a in history["previous_actions"])
    idx = rng.choice(len(w), size=n_hyp, p=w)
    noise = rng.normal(0.0, bundle.sigma_y, size=n_hyp)
    out: list[dict[str, Any]] = []
    for action in range(bundle.n_actions):
        if action in used:
            continue
        bus, amp = _design_of(bundle, action)
        j_c = expected_u_after_action(
            action,
            log_w,
            w,
            centres=bundle.centres,
            U=bundle.U,
            sigma_y=bundle.sigma_y,
            alpha=bundle.alpha,
            margin=bundle.margin,
            u_grid=bundle.u_grid,
            idx=idx,
            noise=noise,
            snap_up=False,
        )
        j_s = expected_u_after_action(
            action,
            log_w,
            w,
            centres=bundle.centres,
            U=bundle.U,
            sigma_y=bundle.sigma_y,
            alpha=bundle.alpha,
            margin=bundle.margin,
            u_grid=bundle.u_grid,
            idx=idx,
            noise=noise,
            snap_up=True,
        )
        out.append(
            {
                "action": action,
                "bus": bus,
                "amplitude": amp,
                "J_continuous": float(j_c),
                "J_snapped": float(j_s),
            }
        )
    return out


def _entropy(counts: Counter) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * math.log(p)
    return float(h)


def analyze_system(
    system: str,
    *,
    max_histories: int = 200,
    n_hyp: int = 64,
    smoke: bool = False,
) -> dict[str, Any]:
    write_u_bank_audit()
    bundle = load_bundle(system)
    if smoke:
        max_histories = min(max_histories, 12)
        n_hyp = min(n_hyp, 24)

    exp_dir = OUT / f"{system}_T3"
    for sub in ("config", "results", "plots", "logs", "summary", "eval", "diagnostics"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
    ensure_standard_layout(exp_dir)
    write_study_run_config(
        exp_dir,
        study_name="continuous_uctrl_amplitude_adaptive_value",
        system=system,
        horizon=3,
        methods=["diagnostic_amplitude"],
        data_dir=ROOT / "data" / system,
        source_config=ROOT / "config" / f"{system}_config.yaml",
        terminal_rule_hash=bundle.continuous_rule.terminal_rule_hash,
        extra={
            "u_ctrl_mode": "continuous_Q_plus_margin_no_snap",
            "u_ctrl_is_approximation": True,
            "probe_duration_sec": bundle.probe_duration,
            "n_amplitudes": len(bundle.amplitudes),
            "n_buses": len(bundle.buses),
            "n_designs": bundle.n_actions,
            "amplitudes": bundle.amplitudes,
            "buses": bundle.buses,
        },
    )
    write_run_metadata(
        exp_dir,
        RunMetadata(
            experiment_name=f"continuous_uctrl_amplitude_adaptive_value/{system}_T3",
            entry_point="run.sh",
            timestamp_utc=utc_now_stamp(),
            system=system,
            horizon=3,
            method="amplitude_diagnostic",
            git_commit=git_commit_hash(ROOT),
            terminal_rule_hash=bundle.continuous_rule.terminal_rule_hash,
            extra={
                "snapped_parent_hash": bundle.snapped_rule.terminal_rule_hash,
                "continuous_physically_validated": False,
            },
        ),
    )

    histories = load_h1_histories(bundle, max_histories=max_histories, smoke=smoke)
    # Focus primary analysis on h1 (+ keep h0 separately).
    h1 = [h for h in histories if int(h["history_step"]) == 1]
    h0 = [h for h in histories if int(h["history_step"]) == 0]

    rng = np.random.default_rng(17001 + (0 if system == "ieee5" else 9))
    detail_rows: list[dict[str, Any]] = []
    joint_opt: list[dict[str, Any]] = []
    bus_amp_opt: list[dict[str, Any]] = []

    # MC stability: compare n_hyp vs 2*n_hyp on a few histories.
    stability = []
    for hist in h1[:3]:
        a = score_history_designs(bundle, hist, n_hyp=n_hyp, rng=rng)
        b = score_history_designs(bundle, hist, n_hyp=2 * n_hyp, rng=rng)
        a_best = min(a, key=lambda r: r["J_continuous"])["action"]
        b_best = min(b, key=lambda r: r["J_continuous"])["action"]
        stability.append(
            {
                "history_id": hist["history_id"],
                "best_action_n": a_best,
                "best_action_2n": b_best,
                "agree": int(a_best == b_best),
                "n_hyp": n_hyp,
            }
        )

    all_scored_h1: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for hist in h0 + h1:
        scored = score_history_designs(bundle, hist, n_hyp=n_hyp, rng=rng)
        if int(hist["history_step"]) == 1:
            all_scored_h1.append((hist, scored))
        if not scored:
            continue
        best_c = min(r["J_continuous"] for r in scored)
        # joint optimum
        joint = min(scored, key=lambda r: (r["J_continuous"], r["action"]))
        second = sorted(scored, key=lambda r: (r["J_continuous"], r["action"]))
        second_best = second[1] if len(second) > 1 else joint
        for r in scored:
            detail_rows.append(
                {
                    "system": system,
                    "history_id": hist["history_id"],
                    "history_step": hist["history_step"],
                    "previous_actions": " ".join(map(str, hist["previous_actions"])),
                    "previous_observations": " ".join(
                        f"{x:.8f}" for x in hist["previous_observations"]
                    ),
                    "candidate_bus": r["bus"],
                    "candidate_amplitude": r["amplitude"],
                    "candidate_action": r["action"],
                    "expected_continuous_u_ctrl": r["J_continuous"],
                    "expected_snapped_u_ctrl": r["J_snapped"],
                    "is_joint_optimal": int(
                        r["action"] == joint["action"]
                        and abs(r["J_continuous"] - joint["J_continuous"]) < 1e-15
                    ),
                    "objective_gap_from_best": r["J_continuous"] - best_c,
                    "history_source": hist["source"],
                    "theta_id": hist.get("theta_id"),
                }
            )
        joint_opt.append(
            {
                "system": system,
                "history_id": hist["history_id"],
                "history_step": hist["history_step"],
                "b_star": joint["bus"],
                "A_star": joint["amplitude"],
                "action_star": joint["action"],
                "J_star_continuous": joint["J_continuous"],
                "J_star_snapped": joint["J_snapped"],
                "second_action": second_best["action"],
                "second_bus": second_best["bus"],
                "second_amplitude": second_best["amplitude"],
                "gap_best_second_continuous": second_best["J_continuous"]
                - joint["J_continuous"],
                "gap_best_second_snapped": second_best["J_snapped"] - joint["J_snapped"],
            }
        )
        # Conditional on bus
        for bus in bundle.buses:
            subset = [r for r in scored if r["bus"] == bus]
            if not subset:
                continue
            best = min(subset, key=lambda r: (r["J_continuous"], r["amplitude"]))
            ordered = sorted(subset, key=lambda r: (r["J_continuous"], r["amplitude"]))
            second = ordered[1] if len(ordered) > 1 else best
            for r in subset:
                # mark amplitude-optimal given bus in detail via later pass
                pass
            bus_amp_opt.append(
                {
                    "system": system,
                    "history_id": hist["history_id"],
                    "history_step": hist["history_step"],
                    "bus": bus,
                    "A_star": best["amplitude"],
                    "J_star": best["J_continuous"],
                    "second_amplitude": second["amplitude"],
                    "gap_best_second": second["J_continuous"] - best["J_continuous"],
                }
            )

    # Mark is_amplitude_optimal_given_bus on detail rows
    bus_best = {
        (int(r["history_id"]), int(r["history_step"]), int(r["bus"])): float(r["A_star"])
        for r in bus_amp_opt
    }
    for row in detail_rows:
        key = (int(row["history_id"]), int(row["history_step"]), int(row["candidate_bus"]))
        row["is_amplitude_optimal_given_bus"] = int(
            abs(float(row["candidate_amplitude"]) - bus_best.get(key, 1e9)) < 1e-12
        )

    # Amplitude regrets on h1 joint analysis
    h1_joint = [r for r in joint_opt if int(r["history_step"]) == 1]
    A_stars = [float(r["A_star"]) for r in h1_joint]
    amp_counts = Counter(A_stars)
    dominant_amp, dom_n = amp_counts.most_common(1)[0] if amp_counts else (float("nan"), 0)

    # Build per-history min_b J(h,A) for continuous
    hist_amp_J: dict[int, dict[float, float]] = {}
    for hist, scored in all_scored_h1:
        hid = int(hist["history_id"])
        hist_amp_J[hid] = {}
        for amp in bundle.amplitudes:
            vals = [r["J_continuous"] for r in scored if abs(r["amplitude"] - amp) < 1e-12]
            if vals:
                hist_amp_J[hid][amp] = float(min(vals))

    regret_rows = []
    wrong_amp_regrets = []
    dominant_regrets = []
    fixed_amp = None
    if bundle.fixed_sequence:
        fixed_amp = float(bundle.catalog[bundle.fixed_sequence[1]].amplitude) if len(bundle.fixed_sequence) > 1 else float(bundle.catalog[bundle.fixed_sequence[0]].amplitude)
    fixed_regrets = []
    cross_regrets = []
    hids = list(hist_amp_J.keys())
    for hid in hids:
        amap = hist_amp_J[hid]
        if not amap:
            continue
        A_i = min(amap, key=amap.get)
        J_i = amap[A_i]
        # wrong amplitude = second best
        ordered = sorted(amap.items(), key=lambda kv: kv[1])
        if len(ordered) > 1:
            wrong_amp_regrets.append(ordered[1][1] - J_i)
        if dominant_amp in amap:
            dominant_regrets.append(amap[dominant_amp] - J_i)
        if fixed_amp is not None and fixed_amp in amap:
            fixed_regrets.append(amap[fixed_amp] - J_i)
        # cross-history: sample other histories
        for hid_j in hids:
            if hid_j == hid:
                continue
            A_j = min(hist_amp_J[hid_j], key=hist_amp_J[hid_j].get)
            if A_j in amap:
                reg = amap[A_j] - J_i
                cross_regrets.append(reg)
                regret_rows.append(
                    {
                        "system": system,
                        "history_i": hid,
                        "history_j": hid_j,
                        "A_star_i": A_i,
                        "A_star_j": A_j,
                        "amplitude_regret": reg,
                    }
                )

    # Design regret for joint (b*,A*)
    design_regrets = []
    for i, ri in enumerate(h1_joint):
        for j, rj in enumerate(h1_joint):
            if i == j:
                continue
            # find J of using (b_j, A_j) on history i from detail
            pass

    # Use scored tables for design regret
    scored_by_hid = {int(h["history_id"]): scored for h, scored in all_scored_h1}
    for hi, scored_i in scored_by_hid.items():
        best_i = min(scored_i, key=lambda r: r["J_continuous"])
        for hj, scored_j in scored_by_hid.items():
            if hi == hj:
                continue
            best_j = min(scored_j, key=lambda r: r["J_continuous"])
            match = [
                r
                for r in scored_i
                if r["bus"] == best_j["bus"] and abs(r["amplitude"] - best_j["amplitude"]) < 1e-12
            ]
            if match:
                design_regrets.append(match[0]["J_continuous"] - best_i["J_continuous"])

    def _stats(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {
                "mean": float("nan"),
                "median": float("nan"),
                "p95": float("nan"),
                "max": float("nan"),
            }
        arr = np.asarray(xs, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.quantile(arr, 0.95)),
            "max": float(arr.max()),
        }

    # Continuous vs snapped diversity
    cont_vals = [r["J_star_continuous"] for r in h1_joint]
    snap_vals = [r["J_star_snapped"] for r in h1_joint]
    gaps_c = [r["gap_best_second_continuous"] for r in h1_joint]
    gaps_s = [r["gap_best_second_snapped"] for r in h1_joint]
    unique_amp_c = len(set(A_stars))
    unique_bus_c = len({int(r["b_star"]) for r in h1_joint})
    unique_pair_c = len({(int(r["b_star"]), float(r["A_star"])) for r in h1_joint})
    A_stars_snap = []
    for hist, scored in all_scored_h1:
        best = min(scored, key=lambda r: (r["J_snapped"], r["action"]))
        A_stars_snap.append(best["amplitude"])
    unique_amp_s = len(set(A_stars_snap))

    # Case label (Part XIII)
    # "Meaningful" = wrong-amplitude regret that is not a near-tie relative to
    # typical terminal u_ctrl (~0.8–1.0). Median≈0 with tiny mean ⇒ Case B.
    frac_non_dom = 1.0 - (dom_n / max(len(A_stars), 1))
    mean_wrong = float(np.mean(wrong_amp_regrets)) if wrong_amp_regrets else 0.0
    median_wrong = float(np.median(wrong_amp_regrets)) if wrong_amp_regrets else 0.0
    p95_wrong = float(np.quantile(wrong_amp_regrets, 0.95)) if wrong_amp_regrets else 0.0
    mean_cross = float(np.mean(cross_regrets)) if cross_regrets else 0.0
    meaningful = (median_wrong > 5e-4) or (mean_wrong > 2e-3) or (p95_wrong > 5e-3 and frac_non_dom > 0.15)
    amp_changes = unique_amp_c >= 2 and frac_non_dom > 0.05
    snap_amp_changes = unique_amp_s >= 2
    mean_gap_s = float(np.mean(gaps_s)) if gaps_s else 0.0
    mean_gap_c = float(np.mean(gaps_c)) if gaps_c else 0.0

    if amp_changes and meaningful:
        case = "C"
        case_note = "preferred_amplitude_changes_with_meaningful_regret"
    elif amp_changes and not meaningful:
        case = "B"
        case_note = (
            "nominal_amplitude_branching_near_zero_regret; "
            "continuous still low practical value => also Case E "
            "(snap not the main amplitude bottleneck)"
        )
    else:
        case = "A"
        case_note = (
            "same_amplitude_preferred_almost_always; "
            "continuous still low => also Case E"
        )

    if (
        case == "C"
        and (not snap_amp_changes or mean_gap_s < 0.5 * max(mean_gap_c, 1e-12))
    ):
        case_note = (
            "Case C under continuous u_ctrl; snapped landscape flatter "
            "(partial Case D: snap_up suppressed objective gaps)"
        )
    elif case == "B":
        c_std = float(np.std(cont_vals)) if cont_vals else 0.0
        s_std = float(np.std(snap_vals)) if snap_vals else 0.0
        if c_std > 1.2 * max(s_std, 1e-12):
            case_note = (
                case_note
                + "; continuous u_ctrl more variable than snapped, but amplitude "
                "regret still near-zero (not Case D)"
            )

    summary = {
        "system": system,
        "number_of_histories": len(h1),
        "number_of_amplitudes": len(bundle.amplitudes),
        "number_of_valid_buses": len(bundle.buses),
        "n_designs": bundle.n_actions,
        "amplitudes": bundle.amplitudes,
        "buses": bundle.buses,
        "probe_duration_sec": bundle.probe_duration,
        "dominant_amplitude": dominant_amp,
        "dominant_amplitude_fraction": float(dom_n / max(len(A_stars), 1)),
        "number_of_unique_optimal_amplitudes": unique_amp_c,
        "number_of_unique_optimal_buses": unique_bus_c,
        "number_of_unique_optimal_pairs": unique_pair_c,
        "fraction_histories_with_non_dominant_amplitude": float(frac_non_dom),
        "entropy_optimal_amplitude": _entropy(amp_counts),
        "fraction_history_pairs_different_amplitude": float(
            np.mean(
                [
                    int(A_stars[i] != A_stars[j])
                    for i in range(len(A_stars))
                    for j in range(i + 1, len(A_stars))
                ]
            )
        )
        if len(A_stars) > 1
        else 0.0,
        "mean_best_second_gap": float(np.mean(gaps_c)) if gaps_c else float("nan"),
        "median_best_second_gap": float(np.median(gaps_c)) if gaps_c else float("nan"),
        "mean_best_second_gap_snapped": float(np.mean(gaps_s)) if gaps_s else float("nan"),
        "wrong_amplitude_regret": _stats(wrong_amp_regrets),
        "cross_history_amplitude_regret": _stats(cross_regrets),
        "dominant_amplitude_regret": _stats(dominant_regrets),
        "fixed_amplitude_regret": _stats(fixed_regrets),
        "design_regret": _stats(design_regrets),
        "continuous_J_std": float(np.std(cont_vals)) if cont_vals else float("nan"),
        "snapped_J_std": float(np.std(snap_vals)) if snap_vals else float("nan"),
        "continuous_unique_J": len({round(x, 8) for x in cont_vals}),
        "snapped_unique_J": len({round(x, 8) for x in snap_vals}),
        "unique_optimal_amplitudes_snapped": unique_amp_s,
        "case": case,
        "case_note": case_note,
        "continuous_terminal_rule_hash": bundle.continuous_rule.terminal_rule_hash,
        "snapped_terminal_rule_hash": bundle.snapped_rule.terminal_rule_hash,
        "u_ctrl_physically_validated": False,
        "u_ctrl_is_approximation": True,
        "n_hyp": n_hyp,
        "mc_stability": stability,
        "fixed_sequence": bundle.fixed_sequence,
        "fixed_plan_amplitude": fixed_amp,
        "used_confirmation_split": False,
    }

    # Write CSVs
    _write_csv(exp_dir / "results" / "history_design_scores.csv", detail_rows)
    _write_csv(exp_dir / "results" / "joint_optima.csv", joint_opt)
    _write_csv(exp_dir / "results" / "bus_conditional_amplitude.csv", bus_amp_opt)
    _write_csv(exp_dir / "results" / "amplitude_regret.csv", regret_rows)
    (exp_dir / "summary" / "system_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_csv(
        exp_dir / "summary" / "summary_table.csv",
        [
            {
                "system": system,
                "number_of_histories": summary["number_of_histories"],
                "number_of_amplitudes": summary["number_of_amplitudes"],
                "number_of_valid_buses": summary["number_of_valid_buses"],
                "dominant_amplitude": summary["dominant_amplitude"],
                "dominant_amplitude_fraction": summary["dominant_amplitude_fraction"],
                "number_of_unique_optimal_amplitudes": summary[
                    "number_of_unique_optimal_amplitudes"
                ],
                "fraction_histories_with_non_dominant_amplitude": summary[
                    "fraction_histories_with_non_dominant_amplitude"
                ],
                "mean_best_second_gap": summary["mean_best_second_gap"],
                "median_best_second_gap": summary["median_best_second_gap"],
                "mean_wrong_amplitude_regret": summary["wrong_amplitude_regret"]["mean"],
                "p95_wrong_amplitude_regret": summary["wrong_amplitude_regret"]["p95"],
                "max_wrong_amplitude_regret": summary["wrong_amplitude_regret"]["max"],
                "case": case,
            }
        ],
    )

    _make_plots(bundle, hist_amp_J, h1_joint, A_stars, summary, exp_dir / "plots")
    return summary


def _make_plots(
    bundle: StudyBundle,
    hist_amp_J: dict[int, dict[float, float]],
    h1_joint: list[dict[str, Any]],
    A_stars: list[float],
    summary: dict[str, Any],
    plots: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plots.mkdir(parents=True, exist_ok=True)
    amps = bundle.amplitudes
    hids = sorted(hist_amp_J.keys())[:80]
    if hids:
        mat = np.full((len(hids), len(amps)), np.nan)
        for i, hid in enumerate(hids):
            for j, a in enumerate(amps):
                if a in hist_amp_J[hid]:
                    mat[i, j] = hist_amp_J[hid][a]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mat, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(amps)))
        ax.set_xticklabels([str(a) for a in amps])
        ax.set_xlabel("amplitude")
        ax.set_ylabel("history index (subset)")
        ax.set_title(f"{bundle.system}: min_b J_continuous(h, A)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(plots / "history_amplitude_heatmap.png", dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = Counter(A_stars)
    ax.bar([str(a) for a in amps], [counts.get(a, 0) for a in amps])
    ax.set_xlabel("A*(h)")
    ax.set_ylabel("count")
    ax.set_title(f"{bundle.system}: optimal amplitude histogram")
    fig.tight_layout()
    fig.savefig(plots / "optimal_amplitude_hist.png", dpi=140)
    plt.close(fig)

    # gap comparison continuous vs snapped
    if h1_joint:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            [r["gap_best_second_snapped"] for r in h1_joint],
            [r["gap_best_second_continuous"] for r in h1_joint],
            s=12,
            alpha=0.7,
        )
        lim = max(
            max(r["gap_best_second_snapped"] for r in h1_joint),
            max(r["gap_best_second_continuous"] for r in h1_joint),
            1e-6,
        )
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlabel("best−second gap (snapped)")
        ax.set_ylabel("best−second gap (continuous)")
        ax.set_title(f"{bundle.system}: continuous vs snapped gaps")
        fig.tight_layout()
        fig.savefig(plots / "continuous_vs_snapped_gaps.png", dpi=140)
        plt.close(fig)

    # amplitude vs J for a few histories on bus 0
    for hist_id in hids[:4]:
        fig, ax = plt.subplots(figsize=(6, 4))
        amap = hist_amp_J.get(hist_id, {})
        ax.plot(list(amap.keys()), list(amap.values()), marker="o")
        ax.set_xlabel("amplitude")
        ax.set_ylabel("min_b expected continuous u_ctrl")
        ax.set_title(f"{bundle.system} history {hist_id}")
        fig.tight_layout()
        fig.savefig(plots / f"amp_vs_J_hist_{hist_id}.png", dpi=120)
        plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
