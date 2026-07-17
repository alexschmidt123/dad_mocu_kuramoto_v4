"""Terminal u_ctrl sensitivity audit (train/validation only)."""

from __future__ import annotations

import csv
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.control.objective_rl_sboed import OUT, ROOT
from src.control.objective_rl_sboed.context import (
    StudyContext,
    control_from_log_weights,
    load_study_context,
    observe_bank,
    update_log_weights,
)
from src.control.posterior_ctrl import normalize_log_weights
from src.swing_equation_ode.design import build_catalog
from src.run_context import load_experiment_run


GLOBAL_SEED = 55101
NEAR_TIE_TOL = 1e-4


def _enumerate_sequences(n_actions: int, horizon: int, max_sequences: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    if n_actions <= 12 and horizon <= 3:
        all_seq = list(itertools.permutations(range(n_actions), horizon))
        if len(all_seq) <= max_sequences:
            return all_seq
        idx = rng.choice(len(all_seq), size=max_sequences, replace=False)
        return [all_seq[i] for i in idx]
    # Sample distinct random sequences without replacement of actions within a sequence.
    out: set[tuple[int, ...]] = set()
    attempts = 0
    while len(out) < max_sequences and attempts < max_sequences * 50:
        attempts += 1
        seq = tuple(int(x) for x in rng.choice(n_actions, size=horizon, replace=False))
        out.add(seq)
    return list(out)


def evaluate_sequence_expectation(
    ctx: StudyContext,
    sequence: tuple[int, ...],
    systems: list[dict[str, Any]],
    *,
    n_noise: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    u_ctrl_vals: list[float] = []
    u_raw_vals: list[float] = []
    for si, system in enumerate(systems):
        for rep in range(n_noise):
            log_w = ctx.log_p0.copy()
            for step, action in enumerate(sequence):
                y = observe_bank(
                    system,
                    int(action),
                    sigma_y=ctx.sigma_y,
                    global_seed=GLOBAL_SEED,
                    theta_id=si,
                    step=step,
                    rollout_id=rep,
                )
                log_w = update_log_weights(ctx, log_w, int(action), y)
            decision = control_from_log_weights(ctx, log_w)
            u_ctrl_vals.append(decision.u_ctrl)
            u_raw_vals.append(decision.u_raw)
    u_ctrl = np.asarray(u_ctrl_vals, dtype=np.float64)
    u_raw = np.asarray(u_raw_vals, dtype=np.float64)
    return {
        "mean_u_ctrl": float(u_ctrl.mean()),
        "std_u_ctrl": float(u_ctrl.std()),
        "mean_u_raw": float(u_raw.mean()),
        "std_u_raw": float(u_raw.std()),
        "n_eval": float(u_ctrl.size),
    }


def run_sensitivity_audit(
    system: str,
    *,
    max_sequences: int = 400,
    n_systems: int = 40,
    n_noise: int = 2,
    smoke: bool = False,
) -> dict[str, Any]:
    ctx = load_study_context(system)
    rng = np.random.default_rng(GLOBAL_SEED + (0 if system == "ieee5" else 9))
    systems = list(ctx.train_systems) + list(ctx.validation_systems)
    if smoke:
        max_sequences = min(max_sequences, 40)
        n_systems = min(n_systems, 12)
        n_noise = 1
    systems = systems[:n_systems]
    sequences = _enumerate_sequences(ctx.n_actions, ctx.horizon, max_sequences, rng)

    rows: list[dict[str, Any]] = []
    for seq in sequences:
        stats = evaluate_sequence_expectation(
            ctx, seq, systems, n_noise=n_noise, rng=rng
        )
        rows.append(
            {
                "system": system,
                "sequence": " ".join(map(str, seq)),
                "xi1": seq[0],
                "xi2": seq[1],
                "xi3": seq[2],
                **stats,
            }
        )

    means = np.asarray([r["mean_u_ctrl"] for r in rows], dtype=np.float64)
    raw_means = np.asarray([r["mean_u_raw"] for r in rows], dtype=np.float64)
    unique_ctrl = len({round(float(x), 10) for x in means})
    unique_raw = len({round(float(x), 10) for x in raw_means})
    order = np.argsort(means)
    best = float(means[order[0]])
    second = float(means[order[1]]) if len(order) > 1 else best
    gaps = np.diff(np.sort(means))
    near_tie_rate = float(np.mean(np.abs(gaps) <= NEAR_TIE_TOL)) if gaps.size else 0.0
    modal = Counter(np.round(means, 6)).most_common(1)[0]
    identical_frac = float(modal[1] / len(means))

    # Probe-order sensitivity: compare a sequence to its reverse when distinct.
    order_gaps: list[float] = []
    by_seq = {tuple(int(x) for x in r["sequence"].split()): float(r["mean_u_ctrl"]) for r in rows}
    for seq, val in by_seq.items():
        rev = tuple(reversed(seq))
        if rev != seq and rev in by_seq:
            order_gaps.append(abs(val - by_seq[rev]))

    # Single-probe replacement sensitivity relative to best sequence.
    best_seq = tuple(int(x) for x in rows[int(order[0])]["sequence"].split())
    replace_gaps: list[float] = []
    for pos in range(ctx.horizon):
        for a in range(ctx.n_actions):
            if a == best_seq[pos]:
                continue
            cand = list(best_seq)
            cand[pos] = a
            key = tuple(cand)
            if key in by_seq:
                replace_gaps.append(abs(by_seq[key] - best))

    catalog = build_catalog(load_experiment_run(ctx.exp_dir, ROOT).cfg)
    amp_vals = {
        i: float(catalog[i].amplitude) for i in range(ctx.n_actions)
    }
    bus_vals = {i: int(catalog[i].bus) for i in range(ctx.n_actions)}

    # Amplitude / bus sensitivity around best sequence.
    amp_gaps: list[float] = []
    bus_gaps: list[float] = []
    for pos in range(ctx.horizon):
        base_amp = amp_vals[best_seq[pos]]
        base_bus = bus_vals[best_seq[pos]]
        for a in range(ctx.n_actions):
            cand = list(best_seq)
            cand[pos] = a
            key = tuple(cand)
            if key not in by_seq:
                continue
            if amp_vals[a] != base_amp and bus_vals[a] == base_bus:
                amp_gaps.append(abs(by_seq[key] - best))
            if bus_vals[a] != base_bus and amp_vals[a] == base_amp:
                bus_gaps.append(abs(by_seq[key] - best))

    snap_gap = float(np.mean(np.abs(raw_means - means)))
    summary = {
        "system": system,
        "horizon": ctx.horizon,
        "n_sequences": len(rows),
        "n_systems": len(systems),
        "n_noise": n_noise,
        "unique_terminal_u_ctrl": unique_ctrl,
        "unique_terminal_u_raw": unique_raw,
        "terminal_u_ctrl_std": float(means.std()),
        "terminal_u_raw_std": float(raw_means.std()),
        "fraction_identical_modal_u_ctrl": identical_frac,
        "near_tie_rate_adjacent_sorted": near_tie_rate,
        "best_mean_u_ctrl": best,
        "second_best_mean_u_ctrl": second,
        "best_minus_second_gap": second - best,
        "mean_order_gap": float(np.mean(order_gaps)) if order_gaps else float("nan"),
        "mean_single_probe_replace_gap": float(np.mean(replace_gaps)) if replace_gaps else float("nan"),
        "mean_amplitude_replace_gap": float(np.mean(amp_gaps)) if amp_gaps else float("nan"),
        "mean_bus_replace_gap": float(np.mean(bus_gaps)) if bus_gaps else float("nan"),
        "mean_abs_u_raw_minus_u_ctrl": snap_gap,
        "u_grid_size": int(ctx.u_grid.size),
        "margin": float(ctx.margin),
        "terminal_rule_hash": ctx.terminal_rule_hash,
        "used_confirmation_split": False,
        "likely_sensitivity_drivers": _classify_drivers(
            unique_ctrl=unique_ctrl,
            unique_raw=unique_raw,
            ctrl_std=float(means.std()),
            raw_std=float(raw_means.std()),
            snap_gap=snap_gap,
            replace_gap=float(np.mean(replace_gaps)) if replace_gaps else 0.0,
        ),
    }

    out_dir = OUT / "diagnostics" / "sensitivity_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{system}_T3_sensitivity.csv"
    _write_csv(csv_path, rows)
    (out_dir / f"{system}_T3_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _classify_drivers(
    *,
    unique_ctrl: int,
    unique_raw: int,
    ctrl_std: float,
    raw_std: float,
    snap_gap: float,
    replace_gap: float,
) -> list[str]:
    drivers: list[str] = []
    if unique_raw > unique_ctrl * 1.5 and snap_gap > 1e-4:
        drivers.append("C_snap_up_quantization")
    if unique_raw <= max(3, unique_ctrl) and raw_std < 1e-3:
        drivers.append("A_true_similarity_in_control_requirements")
    if replace_gap < 1e-4:
        drivers.append("E_probes_produce_similar_posterior_changes")
    if ctrl_std < 1e-3 and raw_std >= ctrl_std:
        drivers.append("B_coarse_control_candidate_grid")
    if not drivers:
        drivers.append("F_weak_probe_or_observation_sensitivity")
    return drivers


def write_sensitivity_report(summaries: list[dict[str, Any]]) -> Path:
    out_dir = OUT / "diagnostics" / "sensitivity_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Terminal u_ctrl sensitivity audit",
        "",
        "Train/validation systems only. Confirmation/test unused.",
        "Primary metric: snapped terminal `u_ctrl`. Diagnostic: continuous `u_raw`.",
        "",
    ]
    for s in summaries:
        lines += [
            f"## {s['system']} T={s['horizon']}",
            "",
            f"- Sequences evaluated: {s['n_sequences']}",
            f"- Unique terminal u_ctrl: {s['unique_terminal_u_ctrl']}",
            f"- Unique terminal u_raw: {s['unique_terminal_u_raw']}",
            f"- std(u_ctrl): {s['terminal_u_ctrl_std']:.6f}",
            f"- std(u_raw): {s['terminal_u_raw_std']:.6f}",
            f"- Modal identical fraction: {s['fraction_identical_modal_u_ctrl']:.3f}",
            f"- Best−second gap: {s['best_minus_second_gap']:.6f}",
            f"- Mean |u_raw−u_ctrl|: {s['mean_abs_u_raw_minus_u_ctrl']:.6f}",
            f"- Mean order gap: {s.get('mean_order_gap', float('nan'))}",
            f"- Mean single-probe replace gap: {s.get('mean_single_probe_replace_gap', float('nan'))}",
            f"- Likely drivers: {', '.join(s['likely_sensitivity_drivers'])}",
            "",
        ]
    lines += [
        "## Decision (Part III)",
        "",
        "**No experiment-design or control-grid modification in this study version.**",
        "",
        "Reasons:",
        "",
        "1. Objective differences across sequences exist but are small (std ~0.01–0.02), "
        "consistent with the completed Case-B adaptive-value study.",
        "2. `u_raw` shows somewhat finer variation than snapped `u_ctrl`, indicating snap_up "
        "quantization contributes — changing the grid/safety rule would create a new experiment "
        "version and must not be mixed into the primary DAD vs RL-sBOED contrast.",
        "3. The controlled comparison proceeds under the **frozen** terminal rules already used "
        "by IEEE5/IEEE9 T=3 authoritative experiments.",
        "",
        "Optional later ablation (not primary): RL-sBOED-raw-reward diagnostic using `u_raw` "
        "stepwise differences, still evaluated by snapped terminal `u_ctrl`.",
        "",
    ]
    path = out_dir / "sensitivity_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    from src.control.objective_rl_sboed.diagnostics import ensure_sensitivity_audit_alias

    ensure_sensitivity_audit_alias()
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
