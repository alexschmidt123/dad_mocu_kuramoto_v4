"""
Step-1 design EIG (ΔH) heatmap and sBOED suitability check.

Uses stored noisy observations from the test table (physical rollout data), not resampled noise.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.config import SBOEDConfig
from src.contrastive.spce import (
    clamp_info_gain,
    log_gaussian_observation_density,
    normalize_log_weights,
    posterior_entropy,
)
from src.data import lookup_action_y, save_json
from src.swing_equation_ode.design import build_catalog
from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables

DEFAULT_MIN_RELATIVE_SPREAD = 0.05


def score_step1_delta_h(
    y_obs: float,
    action: int,
    log_unnorm: np.ndarray,
    table_support: TableThetaSupport,
    sigma_y: float,
) -> float:
    """One Bayesian update from prior; return non-negative ΔH = H_before − H_after."""
    p_before = normalize_log_weights(log_unnorm)
    H_before = posterior_entropy(p_before)
    f_vals = y_sim_last_step_from_tables(table_support, [int(action)])
    log_L = log_gaussian_observation_density(float(y_obs), f_vals, sigma_y)
    p_after = normalize_log_weights(log_unnorm + log_L)
    H_after = posterior_entropy(p_after)
    return clamp_info_gain(H_before - H_after)


def compute_step1_design_eig(
    cfg: SBOEDConfig,
    test_systems: list[dict[str, Any]],
    catalog,
    table_support: TableThetaSupport,
) -> dict[str, Any]:
    """
    Mean Step-1 ΔH for each design ξ=(bus, amplitude), averaged over test θ*.

    Observation y comes from each test system's pre-generated one-step bank.
    """
    n_actions = len(catalog)
    per_action_delta_h: list[list[float]] = [[] for _ in range(n_actions)]
    log_p0 = np.array(table_support.log_p0, dtype=np.float64)

    for system in test_systems:
        for action in range(n_actions):
            y_t = float(lookup_action_y(system, action))
            dh = score_step1_delta_h(
                y_t, action, log_p0.copy(), table_support, cfg.sigma_y,
            )
            per_action_delta_h[action].append(float(dh))

    rows: list[dict[str, Any]] = []
    matrix: dict[tuple[int, float], float] = {}
    buses = sorted({int(d.bus) for d in catalog})
    amps = sorted({float(d.amplitude) for d in catalog})

    for action, design in enumerate(catalog):
        vals = np.asarray(per_action_delta_h[action], dtype=np.float64)
        mean_dh = float(np.mean(vals)) if vals.size else 0.0
        std_dh = float(np.std(vals)) if vals.size > 1 else 0.0
        sem = std_dh / np.sqrt(max(vals.size, 1))
        row = {
            "action_index": action,
            "bus": int(design.bus),
            "amplitude": float(design.amplitude),
            "duration": float(design.duration),
            "mean_step1_delta_h": mean_dh,
            "std_step1_delta_h": std_dh,
            "sem_step1_delta_h": float(sem),
            "ci95_low": mean_dh - 1.96 * sem,
            "ci95_high": mean_dh + 1.96 * sem,
            "n_rollouts": int(vals.size),
        }
        rows.append(row)
        matrix[(int(design.bus), float(design.amplitude))] = mean_dh

    grid = np.full((len(buses), len(amps)), np.nan, dtype=np.float64)
    bus_index = {b: i for i, b in enumerate(buses)}
    amp_index = {a: j for j, a in enumerate(amps)}
    for (bus, amp), val in matrix.items():
        grid[bus_index[bus], amp_index[amp]] = val

    return {
        "rows": rows,
        "buses": buses,
        "amplitudes": amps,
        "grid": grid.tolist(),
        "matrix": {f"{b},{a}": v for (b, a), v in matrix.items()},
    }


def assess_sboed_suitability(
    heatmap: dict[str, Any],
    *,
    min_relative_spread: float = DEFAULT_MIN_RELATIVE_SPREAD,
) -> dict[str, Any]:
    """Flag whether Step-1 design EIG varies enough to justify sBOED on this system."""
    means = [float(r["mean_step1_delta_h"]) for r in heatmap["rows"]]
    if not means:
        return {
            "suitable_for_sboed": False,
            "reason": "no designs",
            "min_relative_spread": min_relative_spread,
        }

    mn = float(min(means))
    mx = float(max(means))
    spread = mx - mn
    rel_spread = spread / max(mx, 1e-12)
    best = max(heatmap["rows"], key=lambda r: r["mean_step1_delta_h"])
    worst = min(heatmap["rows"], key=lambda r: r["mean_step1_delta_h"])
    suitable = rel_spread >= float(min_relative_spread)

    reason = (
        f"Step-1 ΔH spread {spread:.4f} nats ({rel_spread:.1%} relative) "
        f"≥ {min_relative_spread:.0%} threshold — designs are distinguishable."
        if suitable
        else f"Step-1 ΔH nearly flat: spread {spread:.4f} nats ({rel_spread:.1%} relative) "
        f"< {min_relative_spread:.0%} — poor sBOED benchmark under current probes/prior."
    )

    return {
        "suitable_for_sboed": suitable,
        "reason": reason,
        "min_relative_spread": float(min_relative_spread),
        "mean_step1_delta_h_min": mn,
        "mean_step1_delta_h_max": mx,
        "mean_step1_delta_h_spread": spread,
        "relative_spread": rel_spread,
        "best_design": {
            "bus": best["bus"],
            "amplitude": best["amplitude"],
            "action_index": best["action_index"],
            "mean_step1_delta_h": best["mean_step1_delta_h"],
        },
        "worst_design": {
            "bus": worst["bus"],
            "amplitude": worst["amplitude"],
            "action_index": worst["action_index"],
            "mean_step1_delta_h": worst["mean_step1_delta_h"],
        },
        "n_designs": len(means),
        "n_test_rollouts": int(heatmap["rows"][0].get("n_rollouts", 0)) if heatmap["rows"] else 0,
    }


def write_step1_design_report(
    eval_dir: Path,
    cfg: SBOEDConfig,
    test_systems: list[dict[str, Any]],
    table_support: TableThetaSupport,
    *,
    min_relative_spread: float = DEFAULT_MIN_RELATIVE_SPREAD,
) -> dict[str, Any]:
    """Write CSV, JSON suitability, and heatmap PNG under ``eval/``."""
    catalog = build_catalog(cfg)
    heatmap = compute_step1_design_eig(cfg, test_systems, catalog, table_support)
    suitability = assess_sboed_suitability(heatmap, min_relative_spread=min_relative_spread)

    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "step1_design_eig.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bus",
                "amplitude",
                "action_index",
                "mean_step1_delta_h",
                "std_step1_delta_h",
                "sem_step1_delta_h",
                "ci95_low",
                "ci95_high",
                "n_rollouts",
            ],
        )
        writer.writeheader()
        for row in heatmap["rows"]:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    json_path = eval_dir / "sboed_suitability.json"
    payload = {
        "system_label": cfg.system_label,
        "run_slug": cfg.run_slug,
        "experiment_horizon_T": cfg.step_number,
        "metric": "Step-1 ΔH = E[H₀−H₁ | design ξ=(bus, amplitude)] using stored test observations",
        "heatmap": heatmap,
        "suitability": suitability,
    }
    save_json(payload, json_path)

    md_path = eval_dir / "step1_design_eig.md"
    md_lines = [
        f"# Step-1 design EIG — {cfg.system_label}",
        "",
        suitability["reason"],
        "",
        f"- Best design: bus {suitability['best_design']['bus'] + 1}, "
        f"amplitude {suitability['best_design']['amplitude']} "
        f"(ΔH={suitability['best_design']['mean_step1_delta_h']:.4f} nats)",
        f"- Worst design: bus {suitability['worst_design']['bus'] + 1}, "
        f"amplitude {suitability['worst_design']['amplitude']} "
        f"(ΔH={suitability['worst_design']['mean_step1_delta_h']:.4f} nats)",
        f"- Relative spread: {suitability['relative_spread']:.1%} "
        f"(threshold {min_relative_spread:.0%})",
        f"- **sBOED suitable:** {'yes' if suitability['suitable_for_sboed'] else 'no'}",
        "",
        "See `step1_design_eig.csv` and `step1_design_eig.png`.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    png_path = eval_dir / "step1_design_eig.png"
    from src.stepwise_eig.plots import plot_step1_heatmap

    plot_step1_heatmap(
        {
            "buses": heatmap["buses"],
            "amplitudes": heatmap["amplitudes"],
            "grid": heatmap["grid"],
        },
        out_path=png_path,
        system_label=cfg.system_label,
    )

    return {
        "heatmap": heatmap,
        "suitability": suitability,
        "csv_path": str(csv_path.resolve()),
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
        "png_path": str(png_path.resolve()),
    }
