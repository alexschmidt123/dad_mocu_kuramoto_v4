"""Orchestrate particle-posterior-adequacy convergence analysis."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.control.particle_posterior_adequacy import (
    NESTED_SUPPORT_SIZES,
    OUT,
    ROOT,
)
from src.control.particle_posterior_adequacy.adaptive_value import (
    summarize_adaptive_for_grid,
)
from src.control.particle_posterior_adequacy.diagnostics import (
    analyze_history_support,
    error_summary,
)
from src.control.particle_posterior_adequacy.histories import (
    assert_histories_independent_of_support,
    build_multistep_histories,
    load_stratified_h1_rows,
)
from src.control.particle_posterior_adequacy.supports import (
    SUPPORT_SEEDS,
    assert_scientific_invariants,
    build_support,
    load_master_arrays,
)


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _conv(x: Any) -> Any:
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        if isinstance(x, dict):
            return {k: _conv(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_conv(v) for v in x]
        return x

    path.write_text(json.dumps(_conv(payload), indent=2), encoding="utf-8")


def analyze_system(
    system: str,
    *,
    smoke: bool = False,
    max_histories: int | None = None,
    n_hyp: int | None = None,
    support_seeds: tuple[int, ...] | None = None,
    particle_counts: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    master = load_master_arrays(system)
    assert_scientific_invariants(master)

    seeds = support_seeds or ((101, 202) if smoke else SUPPORT_SEEDS)
    counts = particle_counts or (
        (128, 256, 512) if smoke else NESTED_SUPPORT_SIZES
    )
    # Stratify across designs: ~4 histories/xi1 for IEEE5 (30 actions) ⇒ 120.
    n_hist = max_histories or (12 if smoke else 120)
    n_hyp_mc = n_hyp or (16 if smoke else 32)
    history_steps = (0, 1, 2, 3)

    out_dir = OUT / f"{system}_T3"
    for sub in ("config", "results", "plots", "logs", "summary"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    h1_rows = load_stratified_h1_rows(
        system, max_histories=n_hist, smoke=smoke, project_root=ROOT
    )
    histories = build_multistep_histories(
        master, h1_rows, horizon=3, project_root=ROOT
    )

    particle_rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []

    # Cache reference (largest N) optima / scores per (seed, history_id, step)
    reference_n = max(counts)
    ref_cache: dict[tuple[int, int, int], dict[str, Any]] = {}

    for seed_i, seed in enumerate(seeds):
        print(
            f"  [{system}] support_seed={seed} ({seed_i + 1}/{len(seeds)}) "
            f"histories={len(histories)} counts={list(counts)}"
        )
        # Build supports largest-first so reference exists
        ordered_counts = sorted(counts, reverse=True)
        support_by_n = {
            n: build_support(master, n, seed) for n in ordered_counts
        }
        assert_histories_independent_of_support(
            histories, support_by_n[reference_n].indices
        )

        # Per (history, step) results across N for this seed
        hist_scores_by_n: dict[int, list[dict[str, Any]]] = {n: [] for n in counts}

        for hist in histories:
            for step in history_steps:
                if step not in hist["steps"]:
                    continue
                # Reference first
                rng_ref = np.random.default_rng(
                    10_000 * seed
                    + 97 * int(hist["history_id"])
                    + 13 * step
                    + 7
                )
                do_design = step <= 1
                ref = analyze_history_support(
                    master,
                    support_by_n[reference_n],
                    hist,
                    step,
                    n_hyp=n_hyp_mc,
                    rng=rng_ref,
                    score_designs_flag=do_design,
                )
                key = (seed, int(hist["history_id"]), step)
                ref_cache[key] = {
                    "opt": ref["opt"],
                    "scores_snapped": ref["scores_snapped"],
                    "u_cont": ref["particle_row"]["u_cont"],
                    "u_ctrl": ref["particle_row"]["u_ctrl"],
                }
                particle_rows.append(ref["particle_row"])
                if do_design:
                    design_rows.append(ref["design_row"])
                if do_design and step == 1 and hist.get("history_id", -1) >= 0:
                    hist_scores_by_n[reference_n].append(
                        {
                            "history_id": hist["history_id"],
                            "history_step": 1,
                            "xi1": hist.get("xi1"),
                            "optimal_design": ref["opt"]["optimal_design"],
                            "J_star_snapped": ref["opt"]["J_star_snapped"],
                            "scores_snapped": ref["scores_snapped"],
                        }
                    )

                for n in ordered_counts:
                    if n == reference_n:
                        continue
                    # Identical CRN stream for fair design scoring across N
                    rng = np.random.default_rng(
                        10_000 * seed
                        + 97 * int(hist["history_id"])
                        + 13 * step
                        + 7
                    )
                    res = analyze_history_support(
                        master,
                        support_by_n[n],
                        hist,
                        step,
                        n_hyp=n_hyp_mc,
                        rng=rng,
                        reference_scores=ref_cache[key]["scores_snapped"] if do_design else None,
                        reference_optimal=ref_cache[key]["opt"] if do_design else None,
                        score_designs_flag=do_design,
                    )
                    particle_rows.append(res["particle_row"])
                    if do_design:
                        design_rows.append(res["design_row"])
                    if do_design and step == 1 and hist.get("history_id", -1) >= 0:
                        hist_scores_by_n[n].append(
                            {
                                "history_id": hist["history_id"],
                                "history_step": 1,
                                "xi1": hist.get("xi1"),
                                "optimal_design": res["opt"]["optimal_design"],
                                "J_star_snapped": res["opt"]["J_star_snapped"],
                                "scores_snapped": res["scores_snapped"],
                            }
                        )

        # Fixed objective: mean terminal u_ctrl under Fixed sequence at h3 (reference N)
        fixed_u = []
        for hist in histories:
            if hist["history_id"] == -1:
                continue
            if 3 not in hist["steps"]:
                continue
            key = (seed, int(hist["history_id"]), 3)
            if key in ref_cache:
                fixed_u.append(ref_cache[key]["u_ctrl"])
        fixed_obj = float(np.mean(fixed_u)) if fixed_u else None

        for n in counts:
            adaptive_rows.append(
                summarize_adaptive_for_grid(
                    system=system,
                    particle_count=n,
                    support_seed=seed,
                    history_scores=hist_scores_by_n[n],
                    design_rows=[
                        r
                        for r in design_rows
                        if r["support_seed"] == seed and r["particle_count"] == n
                    ],
                    fixed_objective=fixed_obj,
                )
            )

    # Convergence aggregates vs reference N
    conv_rows = _convergence_tables(particle_rows, design_rows, reference_n)

    _write_csv(out_dir / "results" / "posterior_particle_diagnostics.csv", particle_rows)
    _write_csv(out_dir / "results" / "design_stability.csv", design_rows)
    _write_csv(out_dir / "results" / "adaptive_value.csv", adaptive_rows)
    _write_csv(out_dir / "results" / "uctrl_convergence.csv", conv_rows["uctrl"])
    _write_csv(out_dir / "results" / "design_regret_summary.csv", conv_rows["regret"])

    summary = {
        "system": system,
        "latent_dimension": master.latent_dim,
        "n_buses": master.n_buses,
        "n_actions": master.n_actions,
        "amplitudes": master.amplitudes,
        "buses": master.buses,
        "probe_duration": master.probe_duration,
        "dataset": str(master.data_path),
        "particle_counts": list(counts),
        "support_seeds": list(seeds),
        "n_histories_including_h0": len(histories),
        "n_diagnostic_histories": len(h1_rows),
        "n_hyp": n_hyp_mc,
        "reference_particle_count": reference_n,
        "production_train_theta_count": master.train_theta_count_production,
        "production_test_theta_count": master.test_theta_count_production,
        "smoke": smoke,
        "official_metric": "u_ctrl = snap_up(Q_{1-alpha}(U|w)+margin)",
        "diagnostic_metric": "u_cont = Q_{1-alpha}(U|w)+margin",
        "adaptive_rows": adaptive_rows,
        "convergence": conv_rows["summary"],
    }
    _write_json(out_dir / "summary" / "system_summary.json", summary)
    return summary


def _convergence_tables(
    particle_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    reference_n: int,
) -> dict[str, Any]:
    # Index reference controls
    ref_u: dict[tuple[Any, ...], dict[str, float]] = {}
    for r in particle_rows:
        if int(r["particle_count"]) != reference_n:
            continue
        key = (r["system"], r["support_seed"], r["history_id"], r["history_step"])
        ref_u[key] = {"u_cont": float(r["u_cont"]), "u_ctrl": float(r["u_ctrl"])}

    uctrl_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    systems = sorted({r["system"] for r in particle_rows})
    for system in systems:
        for n in sorted({int(r["particle_count"]) for r in particle_rows}):
            err_c: list[float] = []
            err_s: list[float] = []
            changed = 0
            total = 0
            for r in particle_rows:
                if r["system"] != system or int(r["particle_count"]) != n:
                    continue
                key = (r["system"], r["support_seed"], r["history_id"], r["history_step"])
                if key not in ref_u:
                    continue
                total += 1
                err_c.append(abs(float(r["u_cont"]) - ref_u[key]["u_cont"]))
                err_s.append(abs(float(r["u_ctrl"]) - ref_u[key]["u_ctrl"]))
                if abs(float(r["u_ctrl"]) - ref_u[key]["u_ctrl"]) > 1e-12:
                    changed += 1
            if n == reference_n:
                err_c = [0.0] * max(total, 0)
                err_s = [0.0] * max(total, 0)
                changed = 0
            row = {
                "system": system,
                "particle_count": n,
                "reference_particle_count": reference_n,
                "frac_u_ctrl_changed": changed / max(total, 1),
                **{f"u_cont_{k}": v for k, v in error_summary(err_c).items()},
                **{f"u_ctrl_{k}": v for k, v in error_summary(err_s).items()},
            }
            uctrl_rows.append(row)

    regret_rows: list[dict[str, Any]] = []
    for system in systems:
        for n in sorted({int(r["particle_count"]) for r in design_rows}):
            regs = [
                float(r["reference_regret"])
                for r in design_rows
                if r["system"] == system
                and int(r["particle_count"]) == n
                and r.get("reference_regret") is not None
                and r["reference_regret"] == r["reference_regret"]
            ]
            agrees = [
                bool(r["design_agreement"])
                for r in design_rows
                if r["system"] == system
                and int(r["particle_count"]) == n
                and r.get("design_agreement") is not None
            ]
            bus_ag = [
                bool(r["bus_agreement"])
                for r in design_rows
                if r["system"] == system
                and int(r["particle_count"]) == n
                and r.get("bus_agreement") is not None
            ]
            amp_ag = [
                bool(r["amplitude_agreement"])
                for r in design_rows
                if r["system"] == system
                and int(r["particle_count"]) == n
                and r.get("amplitude_agreement") is not None
            ]
            regret_rows.append(
                {
                    "system": system,
                    "particle_count": n,
                    "frac_design_agreement": float(np.mean(agrees)) if agrees else float("nan"),
                    "frac_bus_agreement": float(np.mean(bus_ag)) if bus_ag else float("nan"),
                    "frac_amplitude_agreement": float(np.mean(amp_ag)) if amp_ag else float("nan"),
                    **{f"regret_{k}": v for k, v in error_summary(regs).items()},
                }
            )
    summary = {"uctrl": uctrl_rows, "regret": regret_rows}
    return {"uctrl": uctrl_rows, "regret": regret_rows, "summary": summary}


def write_comparison(summaries: list[dict[str, Any]]) -> None:
    comp = OUT / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    mapping = {
        "posterior_particle_diagnostics.csv": "posterior_ess.csv",
        "design_stability.csv": "design_stability.csv",
        "adaptive_value.csv": "adaptive_value_convergence.csv",
        "uctrl_convergence.csv": "uctrl_convergence.csv",
        "design_regret_summary.csv": "design_regret.csv",
    }
    for src_name, dst_name in mapping.items():
        rows: list[dict[str, Any]] = []
        for s in summaries:
            path = OUT / f"{s['system']}_T3" / "results" / src_name
            if not path.is_file() or path.stat().st_size == 0:
                continue
            with path.open(encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
        _write_csv(comp / dst_name, rows)
