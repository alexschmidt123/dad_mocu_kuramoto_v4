"""Bus adaptivity + four-way joint bus-amplitude decomposition."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.control.bus_joint_adaptive_value import OUT, ROOT
from src.control.continuous_uctrl_amplitude.audit import U_BANK_AUDIT
from src.control.continuous_uctrl_amplitude.diagnostic import (
    StudyBundle,
    _design_of,
    _write_csv,
    load_bundle,
    load_h1_histories,
    score_history_designs,
)
from src.control.posterior_batch import expected_u_after_action
from src.control.posterior_ctrl import normalize_log_weights
from src.experiment_layout import (
    RunMetadata,
    ensure_standard_layout,
    git_commit_hash,
    utc_now_stamp,
    write_run_metadata,
    write_study_run_config,
)


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


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


def _bus_level(
    scored: list[dict[str, Any]],
    *,
    key: str,
) -> dict[int, dict[str, float]]:
    """Per bus: min_A J and corresponding A*."""
    by_bus: dict[int, list[dict[str, Any]]] = {}
    for row in scored:
        by_bus.setdefault(int(row["bus"]), []).append(row)
    out: dict[int, dict[str, float]] = {}
    for bus, rows in by_bus.items():
        best = min(rows, key=lambda r: (float(r[key]), float(r["amplitude"])))
        ordered = sorted(rows, key=lambda r: (float(r[key]), float(r["amplitude"])))
        second = ordered[1] if len(ordered) > 1 else best
        out[bus] = {
            "J": float(best[key]),
            "A_star": float(best["amplitude"]),
            "action": int(best["action"]),
            "gap_amp": float(second[key]) - float(best[key]),
        }
    return out


def _freeze_rule_copy(bundle: StudyBundle, exp_dir: Path) -> None:
    cfg = exp_dir / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    payload = {
        "rule": {
            "alpha": bundle.continuous_rule.alpha,
            "margin": bundle.continuous_rule.margin,
            "u_candidates": list(bundle.continuous_rule.u_candidates),
            "snap_up": False,
            "formula": "Q_{1-alpha}(U|w) + margin",
            "role": "diagnostic_high_resolution_objective",
        },
        **bundle.continuous_rule.metadata(),
        "u_bank_audit": U_BANK_AUDIT,
        "snapped_parent_hash": bundle.snapped_rule.terminal_rule_hash,
        "continuous_physically_validated": False,
    }
    (cfg / "continuous_terminal_rule.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (cfg / "terminal_rule_hash.txt").write_text(
        bundle.continuous_rule.terminal_rule_hash + "\n", encoding="utf-8"
    )
    (cfg / "u_bank_audit.json").write_text(
        json.dumps(U_BANK_AUDIT, indent=2), encoding="utf-8"
    )


def analyze_system(
    system: str,
    *,
    max_histories: int = 200,
    n_hyp: int = 64,
    smoke: bool = False,
    n_decomp_rollouts: int = 128,
) -> dict[str, Any]:
    bundle = load_bundle(system)
    # Point continuous rule freeze into this study's tree (not the amplitude study).
    if smoke:
        max_histories = min(max_histories, 12)
        n_hyp = min(n_hyp, 24)
        n_decomp_rollouts = min(n_decomp_rollouts, 32)

    exp_dir = OUT / f"{system}_T3"
    for sub in ("config", "results", "plots", "logs", "summary", "eval", "diagnostics"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
    ensure_standard_layout(exp_dir)
    _freeze_rule_copy(bundle, exp_dir)
    write_study_run_config(
        exp_dir,
        study_name="bus_joint_adaptive_value",
        system=system,
        horizon=3,
        methods=["bus_diagnostic", "joint_decomposition"],
        data_dir=ROOT / "data" / system,
        source_config=ROOT / "config" / f"{system}_config.yaml",
        terminal_rule_hash=bundle.continuous_rule.terminal_rule_hash,
        extra={
            "u_cont_role": "diagnostic_high_resolution",
            "u_ctrl_snapped_role": "historical_comparison",
            "probe_duration_sec": bundle.probe_duration,
            "n_amplitudes": len(bundle.amplitudes),
            "n_buses": len(bundle.buses),
            "n_designs": bundle.n_actions,
            "amplitudes": bundle.amplitudes,
            "buses": bundle.buses,
            "prior_amplitude_case": "B",
            "no_dad_rl_retraining": True,
        },
    )
    write_run_metadata(
        exp_dir,
        RunMetadata(
            experiment_name=f"bus_joint_adaptive_value/{system}_T3",
            entry_point="run.sh",
            timestamp_utc=utc_now_stamp(),
            system=system,
            horizon=3,
            method="bus_joint_diagnostic",
            git_commit=git_commit_hash(ROOT),
            terminal_rule_hash=bundle.continuous_rule.terminal_rule_hash,
            extra={"used_confirmation_split": False},
        ),
    )

    histories = load_h1_histories(bundle, max_histories=max_histories, smoke=smoke)
    h1 = [h for h in histories if int(h["history_step"]) == 1]
    h0 = [h for h in histories if int(h["history_step"]) == 0]
    rng = np.random.default_rng(19001 + (0 if system == "ieee5" else 9))

    # MC stability
    stability = []
    for hist in h1[:3]:
        a = score_history_designs(bundle, hist, n_hyp=n_hyp, rng=rng)
        b = score_history_designs(bundle, hist, n_hyp=2 * n_hyp, rng=rng)
        ba = _bus_level(a, key="J_continuous")
        bb = _bus_level(b, key="J_continuous")
        b_star_a = min(ba, key=lambda k: ba[k]["J"])
        b_star_b = min(bb, key=lambda k: bb[k]["J"])
        stability.append(
            {
                "history_id": hist["history_id"],
                "b_star_n": int(b_star_a),
                "b_star_2n": int(b_star_b),
                "agree": int(b_star_a == b_star_b),
                "n_hyp": n_hyp,
            }
        )

    detail_rows: list[dict[str, Any]] = []
    bus_opt_rows: list[dict[str, Any]] = []
    scored_h1: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    for hist in h0 + h1:
        scored = score_history_designs(bundle, hist, n_hyp=n_hyp, rng=rng)
        if int(hist["history_step"]) == 1:
            scored_h1.append((hist, scored))
        if not scored:
            continue
        bus_cont = _bus_level(scored, key="J_continuous")
        bus_snap = _bus_level(scored, key="J_snapped")
        b_star_c = min(bus_cont, key=lambda k: (bus_cont[k]["J"], k))
        b_star_s = min(bus_snap, key=lambda k: (bus_snap[k]["J"], k))
        ordered_c = sorted(bus_cont.items(), key=lambda kv: (kv[1]["J"], kv[0]))
        second_c = ordered_c[1] if len(ordered_c) > 1 else ordered_c[0]
        ordered_s = sorted(bus_snap.items(), key=lambda kv: (kv[1]["J"], kv[0]))
        second_s = ordered_s[1] if len(ordered_s) > 1 else ordered_s[0]
        joint_c = min(scored, key=lambda r: (r["J_continuous"], r["action"]))
        joint_s = min(scored, key=lambda r: (r["J_snapped"], r["action"]))

        bus_opt_rows.append(
            {
                "system": system,
                "history_id": hist["history_id"],
                "history_step": hist["history_step"],
                "b_star_cont": int(b_star_c),
                "A_star_on_b_star_cont": bus_cont[b_star_c]["A_star"],
                "J_bus_star_cont": bus_cont[b_star_c]["J"],
                "second_bus_cont": int(second_c[0]),
                "gap_best_second_bus_cont": second_c[1]["J"] - bus_cont[b_star_c]["J"],
                "b_star_snap": int(b_star_s),
                "A_star_on_b_star_snap": bus_snap[b_star_s]["A_star"],
                "J_bus_star_snap": bus_snap[b_star_s]["J"],
                "second_bus_snap": int(second_s[0]),
                "gap_best_second_bus_snap": second_s[1]["J"] - bus_snap[b_star_s]["J"],
                "joint_bus_cont": int(joint_c["bus"]),
                "joint_amp_cont": float(joint_c["amplitude"]),
                "bus_ranking_cont": " ".join(str(b) for b, _ in ordered_c),
            }
        )

        for r in scored:
            bus = int(r["bus"])
            is_bus_opt = int(bus == b_star_c)
            is_amp_opt = int(
                abs(float(r["amplitude"]) - bus_cont[bus]["A_star"]) < 1e-12
            )
            detail_rows.append(
                {
                    "system": system,
                    "history_id": hist["history_id"],
                    "history_step": hist["history_step"],
                    "previous_actions": " ".join(map(str, hist["previous_actions"])),
                    "previous_observations": " ".join(
                        f"{x:.8f}" for x in hist["previous_observations"]
                    ),
                    "candidate_bus": bus,
                    "candidate_amplitude": r["amplitude"],
                    "expected_continuous_objective": r["J_continuous"],
                    "expected_snapped_objective": r["J_snapped"],
                    "is_joint_optimal": int(r["action"] == joint_c["action"]),
                    "is_bus_optimal": is_bus_opt,
                    "is_amplitude_optimal_given_bus": is_amp_opt,
                    "bus_objective_gap": bus_cont[bus]["J"] - bus_cont[b_star_c]["J"],
                    "history_source": hist["source"],
                }
            )

    # Wrong-bus regrets on h1
    h1_opt = [r for r in bus_opt_rows if int(r["history_step"]) == 1]
    b_stars = [int(r["b_star_cont"]) for r in h1_opt]
    bus_counts = Counter(b_stars)
    dominant_bus, dom_n = bus_counts.most_common(1)[0] if bus_counts else (0, 0)

    # Map history -> bus J table
    hist_bus_J_c: dict[int, dict[int, float]] = {}
    hist_bus_J_s: dict[int, dict[int, float]] = {}
    for hist, scored in scored_h1:
        hid = int(hist["history_id"])
        bc = _bus_level(scored, key="J_continuous")
        bs = _bus_level(scored, key="J_snapped")
        hist_bus_J_c[hid] = {b: v["J"] for b, v in bc.items()}
        hist_bus_J_s[hid] = {b: v["J"] for b, v in bs.items()}

    fixed_bus = None
    fixed_amp = None
    if len(bundle.fixed_sequence) >= 2:
        fixed_bus, fixed_amp = _design_of(bundle, bundle.fixed_sequence[1])
    elif bundle.fixed_sequence:
        fixed_bus, fixed_amp = _design_of(bundle, bundle.fixed_sequence[0])

    cross_c, cross_s = [], []
    dom_c, dom_s = [], []
    fix_c, fix_s = [], []
    gaps_c = [float(r["gap_best_second_bus_cont"]) for r in h1_opt]
    gaps_s = [float(r["gap_best_second_bus_snap"]) for r in h1_opt]
    regret_pair_rows = []
    hids = list(hist_bus_J_c.keys())
    for hid in hids:
        jc = hist_bus_J_c[hid]
        js = hist_bus_J_s[hid]
        b_i = min(jc, key=jc.get)
        J_i = jc[b_i]
        if dominant_bus in jc:
            dom_c.append(jc[dominant_bus] - J_i)
        if dominant_bus in js:
            b_i_s = min(js, key=js.get)
            dom_s.append(js[dominant_bus] - js[b_i_s])
        if fixed_bus is not None and fixed_bus in jc:
            fix_c.append(jc[fixed_bus] - J_i)
        if fixed_bus is not None and fixed_bus in js:
            b_i_s = min(js, key=js.get)
            fix_s.append(js[fixed_bus] - js[b_i_s])
        for hid_j in hids:
            if hid_j == hid:
                continue
            b_j = min(hist_bus_J_c[hid_j], key=hist_bus_J_c[hid_j].get)
            if b_j in jc:
                reg = jc[b_j] - J_i
                cross_c.append(reg)
                regret_pair_rows.append(
                    {
                        "system": system,
                        "history_i": hid,
                        "history_j": hid_j,
                        "b_star_i": b_i,
                        "b_star_j": b_j,
                        "wrong_bus_regret_cont": reg,
                        "wrong_bus_regret_snap": (
                            hist_bus_J_s[hid][b_j] - min(hist_bus_J_s[hid].values())
                            if b_j in hist_bus_J_s[hid]
                            else float("nan")
                        ),
                    }
                )
            b_j_s = min(hist_bus_J_s[hid_j], key=hist_bus_J_s[hid_j].get)
            if b_j_s in js:
                cross_s.append(js[b_j_s] - min(js.values()))

    # Amplitude regret from prior study (for comparison plot)
    amp_summary_path = (
        ROOT
        / "experiments"
        / "continuous_uctrl_amplitude_adaptive_value"
        / f"{system}_T3"
        / "summary"
        / "system_summary.json"
    )
    amp_wrong_mean = float("nan")
    if amp_summary_path.is_file():
        amp_wrong_mean = float(
            json.loads(amp_summary_path.read_text())["wrong_amplitude_regret"]["mean"]
        )

    # Four-way decomposition on train/validation systems (not test)
    decomp = _four_way_decomposition(
        bundle,
        n_rollouts=n_decomp_rollouts,
        n_hyp=n_hyp,
        rng=rng,
        fixed_bus=fixed_bus,
        fixed_amp=fixed_amp,
    )

    # Case label (bus). Prefer best−second bus gap + decomposition over
    # heavy-tailed cross-history mean regret (often median 0).
    frac_non_dom = 1.0 - (dom_n / max(len(b_stars), 1))
    mean_wrong = float(np.mean(cross_c)) if cross_c else 0.0
    median_wrong = float(np.median(cross_c)) if cross_c else 0.0
    p95_wrong = float(np.quantile(cross_c, 0.95)) if cross_c else 0.0
    mean_gap_c = float(np.mean(gaps_c)) if gaps_c else 0.0
    median_gap_c = float(np.median(gaps_c)) if gaps_c else 0.0
    mean_gap_s = float(np.mean(gaps_s)) if gaps_s else 0.0
    mean_wrong_s = float(np.mean(cross_s)) if cross_s else 0.0
    bus_changes = len(set(b_stars)) >= 2 and frac_non_dom > 0.05
    gap_meaningful = (median_gap_c > 5e-3) or (mean_gap_c > 1e-2 and frac_non_dom > 0.1)
    regret_meaningful = (median_wrong > 1e-3) or (
        p95_wrong > 2e-2 and mean_wrong > 5e-3 and frac_non_dom > 0.15
    )
    meaningful = gap_meaningful or regret_meaningful

    decomp_ci_excludes_zero = False
    for p in (decomp.get("paired") or []):
        if p["comparison"] in (
            "adaptive_bus_fixed_amp - fully_fixed",
            "adaptive_bus_adaptive_amp - fully_fixed",
        ):
            if float(p["ci95_high"]) < 0:
                decomp_ci_excludes_zero = True

    if bus_changes and meaningful and decomp_ci_excludes_zero:
        case = "BUS-C"
        case_note = (
            "bus preference changes with meaningful one-step gaps AND adaptive-bus "
            "structures beat Fully Fixed on terminal objective"
        )
    elif bus_changes and meaningful and not decomp_ci_excludes_zero:
        case = "BUS-B"
        case_note = (
            "one-step bus gaps/branching exist, but four-way terminal decomposition "
            "does not significantly beat Fully Fixed (practical terminal bus adaptive "
            "value still low; also BUS-E for policy training)"
        )
    elif bus_changes and not meaningful:
        case = "BUS-B"
        case_note = "nominal bus branching with near-zero gaps/regret; also BUS-E"
    else:
        case = "BUS-A"
        case_note = "same_bus_preferred_almost_always; also BUS-E"

    if case == "BUS-C" and (
        mean_wrong_s < 0.5 * max(mean_wrong, 1e-12)
        or mean_gap_s < 0.5 * max(mean_gap_c, 1e-12)
    ):
        case_note = case_note + "; snapped landscape flatter (partial BUS-D)"
    elif (
        case in ("BUS-A", "BUS-B")
        and regret_meaningful
        and mean_wrong_s < 5e-4
        and decomp_ci_excludes_zero
    ):
        case = "BUS-D"
        case_note = "snap_up_hid_bus_structure_revealed_by_continuous_and_decomp"

    summary = {
        "system": system,
        "number_of_histories": len(h1),
        "number_of_valid_buses": len(bundle.buses),
        "number_of_amplitudes": len(bundle.amplitudes),
        "n_designs": bundle.n_actions,
        "amplitudes": bundle.amplitudes,
        "buses": bundle.buses,
        "probe_duration_sec": bundle.probe_duration,
        "dominant_bus": int(dominant_bus),
        "dominant_bus_fraction": float(dom_n / max(len(b_stars), 1)),
        "number_of_unique_optimal_buses": len(set(b_stars)),
        "fraction_non_dominant_bus": float(frac_non_dom),
        "entropy_optimal_bus": _entropy(bus_counts),
        "fraction_history_pairs_different_bus": float(
            np.mean(
                [
                    int(b_stars[i] != b_stars[j])
                    for i in range(len(b_stars))
                    for j in range(i + 1, len(b_stars))
                ]
            )
        )
        if len(b_stars) > 1
        else 0.0,
        "mean_best_second_bus_gap": mean_gap_c,
        "median_best_second_bus_gap": float(np.median(gaps_c)) if gaps_c else float("nan"),
        "p95_best_second_bus_gap": float(np.quantile(gaps_c, 0.95)) if gaps_c else float("nan"),
        "mean_best_second_bus_gap_snapped": mean_gap_s,
        "wrong_bus_regret_cont": _stats(cross_c),
        "wrong_bus_regret_snap": _stats(cross_s),
        "mean_wrong_bus_regret": mean_wrong,
        "p95_wrong_bus_regret": p95_wrong,
        "max_wrong_bus_regret": float(max(cross_c)) if cross_c else float("nan"),
        "mean_dominant_bus_regret": float(np.mean(dom_c)) if dom_c else float("nan"),
        "mean_fixed_bus_regret": float(np.mean(fix_c)) if fix_c else float("nan"),
        "dominant_bus_regret_cont": _stats(dom_c),
        "fixed_bus_regret_cont": _stats(fix_c),
        "prior_wrong_amplitude_regret_mean": amp_wrong_mean,
        "case": case,
        "case_note": case_note,
        "continuous_terminal_rule_hash": bundle.continuous_rule.terminal_rule_hash,
        "snapped_terminal_rule_hash": bundle.snapped_rule.terminal_rule_hash,
        "u_cont_physically_validated": False,
        "n_hyp": n_hyp,
        "mc_stability": stability,
        "fixed_sequence": bundle.fixed_sequence,
        "fixed_bus": fixed_bus,
        "fixed_amplitude": fixed_amp,
        "decomposition": decomp,
        "used_confirmation_split": False,
    }

    _write_csv(exp_dir / "results" / "history_design_scores.csv", detail_rows)
    _write_csv(exp_dir / "results" / "bus_optima.csv", bus_opt_rows)
    _write_csv(exp_dir / "results" / "wrong_bus_regret_pairs.csv", regret_pair_rows)
    (exp_dir / "summary" / "system_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_csv(
        exp_dir / "summary" / "summary_table.csv",
        [
            {
                "system": system,
                "number_of_histories": summary["number_of_histories"],
                "number_of_valid_buses": summary["number_of_valid_buses"],
                "number_of_amplitudes": summary["number_of_amplitudes"],
                "dominant_bus": summary["dominant_bus"],
                "dominant_bus_fraction": summary["dominant_bus_fraction"],
                "number_of_unique_optimal_buses": summary["number_of_unique_optimal_buses"],
                "fraction_non_dominant_bus": summary["fraction_non_dominant_bus"],
                "mean_best_second_bus_gap": summary["mean_best_second_bus_gap"],
                "median_best_second_bus_gap": summary["median_best_second_bus_gap"],
                "mean_wrong_bus_regret": summary["mean_wrong_bus_regret"],
                "p95_wrong_bus_regret": summary["p95_wrong_bus_regret"],
                "max_wrong_bus_regret": summary["max_wrong_bus_regret"],
                "mean_dominant_bus_regret": summary["mean_dominant_bus_regret"],
                "mean_fixed_bus_regret": summary["mean_fixed_bus_regret"],
                "case": case,
            }
        ],
    )
    _write_csv(exp_dir / "results" / "joint_decomposition.csv", decomp["rows"])
    _make_plots(
        bundle,
        hist_bus_J_c,
        h1_opt,
        b_stars,
        cross_c,
        amp_wrong_mean,
        decomp,
        summary,
        exp_dir / "plots",
    )
    return summary


def _action_for(bundle: StudyBundle, bus: int, amp: float) -> int:
    for i, d in enumerate(bundle.catalog):
        if int(d.bus) == int(bus) and abs(float(d.amplitude) - float(amp)) < 1e-12:
            return i
    raise KeyError(f"missing design bus={bus} amp={amp}")


def _four_way_decomposition(
    bundle: StudyBundle,
    *,
    n_rollouts: int,
    n_hyp: int,
    rng: np.random.Generator,
    fixed_bus: int | None,
    fixed_amp: float | None,
) -> dict[str, Any]:
    """Experimental decomposition on train/validation systems only."""
    systems = bundle.train_val_systems
    if not systems or fixed_bus is None or fixed_amp is None:
        return {"rows": [], "paired": [], "note": "missing fixed plan or systems"}

    # Global best validation amplitude (train/val): most common Fixed amp is fine;
    # also compute argmin mean J under prior for amp on fixed bus.
    log_w0 = bundle.log_p0.copy()
    w0 = normalize_log_weights(log_w0)
    idx0 = rng.choice(len(w0), size=n_hyp, p=w0)
    noise0 = rng.normal(0.0, bundle.sigma_y, size=n_hyp)
    amp_scores = {}
    for amp in bundle.amplitudes:
        a = _action_for(bundle, fixed_bus, amp)
        amp_scores[amp] = expected_u_after_action(
            a,
            log_w0,
            w0,
            centres=bundle.centres,
            U=bundle.U,
            sigma_y=bundle.sigma_y,
            alpha=bundle.alpha,
            margin=bundle.margin,
            u_grid=bundle.u_grid,
            idx=idx0,
            noise=noise0,
            snap_up=False,
        )
    global_best_amp = min(amp_scores, key=amp_scores.get)

    modes = (
        "fully_fixed",
        "fixed_bus_adaptive_amp",
        "adaptive_bus_fixed_amp",
        "adaptive_bus_adaptive_amp",
    )
    terminal_cont = {m: [] for m in modes}
    terminal_snap = {m: [] for m in modes}

    for rid in range(n_rollouts):
        tid = rid % len(systems)
        system_row = systems[tid]
        # Shared first probe from Fixed plan step 0 for fair history generation
        # then branch according to mode for remaining steps — for T=3 diagnostic
        # of next-step value we evaluate one-step adaptive choice after h1 built
        # from Fixed xi1 (matches prior studies' h1 focus).
        from src.control.terminal_rule import observe_with_keyed_noise

        # Shared first probe from Fixed plan step 0 for fair history generation
        # then branch according to mode for remaining steps — for T=3 diagnostic
        # of next-step value we evaluate one-step adaptive choice after h1 built
        # from Fixed xi1 (matches prior studies' h1 focus).
        xi1 = int(bundle.fixed_sequence[0])
        y1 = float(
            observe_with_keyed_noise(
                system_row,
                xi1,
                sigma_y=bundle.sigma_y,
                global_seed=44011,
                theta_id=tid,
                rollout_id=rid,
                step=0,
            )
        )
        s2 = float(bundle.sigma_y) ** 2
        centre = bundle.centres[xi1]
        log_w = bundle.log_p0 + (
            -0.5 * math.log(2.0 * math.pi * s2) - 0.5 * ((y1 - centre) ** 2) / s2
        )
        w = normalize_log_weights(log_w)
        idx = rng.choice(len(w), size=n_hyp, p=w)
        noise = rng.normal(0.0, bundle.sigma_y, size=n_hyp)
        used = {xi1}

        # Score all unused designs once (shared CRN)
        scored = []
        for action in range(bundle.n_actions):
            if action in used:
                continue
            bus, amp = _design_of(bundle, action)
            jc = expected_u_after_action(
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
            js = expected_u_after_action(
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
            scored.append(
                {"action": action, "bus": bus, "amplitude": amp, "J_continuous": jc, "J_snapped": js}
            )

        bus_c = _bus_level(scored, key="J_continuous")
        del bus_c  # available if needed for debugging

        def pick(mode: str) -> dict[str, Any]:
            if mode == "fully_fixed":
                subset = [
                    r
                    for r in scored
                    if r["bus"] == fixed_bus and abs(r["amplitude"] - fixed_amp) < 1e-12
                ]
                if not subset:
                    raise RuntimeError("Fixed design not available after xi1")
                return subset[0]
            if mode == "fixed_bus_adaptive_amp":
                subset = [r for r in scored if r["bus"] == fixed_bus]
                return min(subset, key=lambda r: (r["J_continuous"], r["amplitude"]))
            if mode == "adaptive_bus_fixed_amp":
                amp_use = fixed_amp
                subset = [r for r in scored if abs(r["amplitude"] - amp_use) < 1e-12]
                if not subset:
                    amp_use = global_best_amp
                    subset = [r for r in scored if abs(r["amplitude"] - amp_use) < 1e-12]
                return min(subset, key=lambda r: (r["J_continuous"], r["bus"]))
            return min(scored, key=lambda r: (r["J_continuous"], r["action"]))

        for mode in modes:
            choice = pick(mode)
            # Roll remaining steps with chosen xi2 then Fixed xi3 or adaptive continuation
            # For one-step diagnostic of second action, evaluate expected terminal after
            # choosing xi2 then completing with Fixed xi3 if available.
            a2 = int(choice["action"])
            y2 = float(
                observe_with_keyed_noise(
                    system_row,
                    a2,
                    sigma_y=bundle.sigma_y,
                    global_seed=44011,
                    theta_id=tid,
                    rollout_id=rid,
                    step=1,
                )
            )
            centre2 = bundle.centres[a2]
            log_w2 = log_w + (
                -0.5 * math.log(2.0 * math.pi * s2)
                - 0.5 * ((y2 - centre2) ** 2) / s2
            )
            # Third step: Fixed plan step 2 if unused else best remaining under mode
            used2 = {xi1, a2}
            if len(bundle.fixed_sequence) >= 3 and int(bundle.fixed_sequence[2]) not in used2:
                a3 = int(bundle.fixed_sequence[2])
            else:
                # pick any unused with same mode rule loosely
                rem = [i for i in range(bundle.n_actions) if i not in used2]
                a3 = rem[0]
            y3 = float(
                observe_with_keyed_noise(
                    system_row,
                    a3,
                    sigma_y=bundle.sigma_y,
                    global_seed=44011,
                    theta_id=tid,
                    rollout_id=rid,
                    step=2,
                )
            )
            centre3 = bundle.centres[a3]
            log_w3 = log_w2 + (
                -0.5 * math.log(2.0 * math.pi * s2)
                - 0.5 * ((y3 - centre3) ** 2) / s2
            )
            w3 = normalize_log_weights(log_w3)
            from src.control.posterior_ctrl import posterior_control_decision

            d_c = posterior_control_decision(
                bundle.U,
                w3,
                bundle.alpha,
                margin=bundle.margin,
                u_grid=bundle.u_grid,
                snap_up=False,
            )
            d_s = posterior_control_decision(
                bundle.U,
                w3,
                bundle.alpha,
                margin=bundle.margin,
                u_grid=bundle.u_grid,
                snap_up=True,
            )
            terminal_cont[mode].append(float(d_c.u_ctrl))
            terminal_snap[mode].append(float(d_s.u_ctrl_snapped))

    rows = []
    for mode in modes:
        arr_c = np.asarray(terminal_cont[mode], dtype=np.float64)
        arr_s = np.asarray(terminal_snap[mode], dtype=np.float64)
        rows.append(
            {
                "system": bundle.system,
                "structure": mode,
                "mean_u_cont": float(arr_c.mean()),
                "std_u_cont": float(arr_c.std()),
                "mean_u_snap": float(arr_s.mean()),
                "std_u_snap": float(arr_s.std()),
                "n_rollouts": len(arr_c),
                "fixed_bus": fixed_bus,
                "fixed_amplitude": fixed_amp,
                "global_best_amp_on_fixed_bus": global_best_amp,
            }
        )

    # Paired bootstrap vs fully_fixed
    paired = []
    base_c = np.asarray(terminal_cont["fully_fixed"], dtype=np.float64)
    for mode in modes:
        if mode == "fully_fixed":
            continue
        diff = np.asarray(terminal_cont[mode], dtype=np.float64) - base_c
        paired.append({"comparison": f"{mode} - fully_fixed", **_bootstrap(diff)})
    # adaptive_bus_fixed vs adaptive both
    diff2 = (
        np.asarray(terminal_cont["adaptive_bus_adaptive_amp"])
        - np.asarray(terminal_cont["adaptive_bus_fixed_amp"])
    )
    paired.append(
        {"comparison": "adaptive_bus_adaptive_amp - adaptive_bus_fixed_amp", **_bootstrap(diff2)}
    )
    return {
        "rows": rows,
        "paired": paired,
        "global_best_amp_on_fixed_bus": global_best_amp,
        "n_rollouts": n_rollouts,
    }


def _bootstrap(diff: np.ndarray, n_boot: int = 10000, seed: int = 123) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return {"mean_diff": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(diff[idx].mean()))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean_diff": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def _make_plots(
    bundle: StudyBundle,
    hist_bus_J: dict[int, dict[int, float]],
    h1_opt: list[dict[str, Any]],
    b_stars: list[int],
    cross_c: list[float],
    amp_wrong_mean: float,
    decomp: dict[str, Any],
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
    buses = bundle.buses
    hids = sorted(hist_bus_J.keys())[:80]
    if hids:
        mat = np.full((len(hids), len(buses)), np.nan)
        for i, hid in enumerate(hids):
            for j, b in enumerate(buses):
                if b in hist_bus_J[hid]:
                    mat[i, j] = hist_bus_J[hid][b]
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mat, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(buses)))
        ax.set_xticklabels([str(b) for b in buses])
        ax.set_xlabel("bus")
        ax.set_ylabel("history index (subset)")
        ax.set_title(f"{bundle.system}: min_A J_cont(h,b)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(plots / "history_bus_heatmap.png", dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = Counter(b_stars)
    ax.bar([str(b) for b in buses], [counts.get(b, 0) for b in buses])
    ax.set_xlabel("b*(h)")
    ax.set_ylabel("count")
    ax.set_title(f"{bundle.system}: optimal bus histogram")
    fig.tight_layout()
    fig.savefig(plots / "optimal_bus_hist.png", dpi=140)
    plt.close(fig)

    for hid in hids[:4]:
        fig, ax = plt.subplots(figsize=(6, 4))
        vals = [hist_bus_J[hid].get(b, np.nan) for b in buses]
        ax.plot(buses, vals, marker="o")
        ax.set_xlabel("bus")
        ax.set_ylabel("J_bus continuous")
        ax.set_title(f"{bundle.system} history {hid}")
        fig.tight_layout()
        fig.savefig(plots / f"bus_vs_J_hist_{hid}.png", dpi=120)
        plt.close(fig)

    if cross_c:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(cross_c, bins=40, color="steelblue", alpha=0.85)
        ax.set_xlabel("wrong-bus regret (continuous)")
        ax.set_ylabel("count")
        ax.set_title(f"{bundle.system}: wrong-bus regret")
        fig.tight_layout()
        fig.savefig(plots / "wrong_bus_regret_hist.png", dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["wrong-amp\n(prior)", "wrong-bus\n(this study)"],
        [
            0.0 if not np.isfinite(amp_wrong_mean) else amp_wrong_mean,
            summary["mean_wrong_bus_regret"],
        ],
    )
    ax.set_ylabel("mean regret")
    ax.set_title(f"{bundle.system}: amplitude vs bus regret")
    fig.tight_layout()
    fig.savefig(plots / "amp_vs_bus_regret.png", dpi=140)
    plt.close(fig)

    if decomp.get("rows"):
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [r["structure"].replace("_", "\n") for r in decomp["rows"]]
        means = [r["mean_u_cont"] for r in decomp["rows"]]
        ax.bar(labels, means, color="teal", alpha=0.85)
        ax.set_ylabel("mean terminal u_cont")
        ax.set_title(f"{bundle.system}: four-way decomposition")
        fig.tight_layout()
        fig.savefig(plots / "four_way_decomposition.png", dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        [summary["wrong_bus_regret_snap"]["mean"]],
        [summary["wrong_bus_regret_cont"]["mean"]],
        s=60,
    )
    ax.set_xlabel("mean wrong-bus regret (snapped)")
    ax.set_ylabel("mean wrong-bus regret (continuous)")
    ax.set_title(f"{bundle.system}: continuous vs snapped bus regret")
    fig.tight_layout()
    fig.savefig(plots / "continuous_vs_snapped_bus_regret.png", dpi=140)
    plt.close(fig)
