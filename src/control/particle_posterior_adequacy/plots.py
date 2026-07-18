"""Plots for particle-posterior-adequacy study."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.control.particle_posterior_adequacy import OUT


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, Any], key: str) -> float:
    v = row.get(key)
    if v in (None, ""):
        return float("nan")
    return float(v)


def plot_system(system: str) -> list[Path]:
    base = OUT / f"{system}_T3"
    results = base / "results"
    plots = base / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    particle = _read(results / "posterior_particle_diagnostics.csv")
    uctrl = _read(results / "uctrl_convergence.csv")
    regret = _read(results / "design_regret_summary.csv")
    adaptive = _read(results / "adaptive_value.csv")
    written: list[Path] = []

    def save(fig, name: str) -> None:
        path = plots / name
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    # 1) median normalized ESS vs N
    by_n: dict[int, list[float]] = defaultdict(list)
    for r in particle:
        by_n[int(r["particle_count"])].append(_f(r, "normalized_ESS"))
    if by_n:
        ns = sorted(by_n)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns, [float(np.nanmedian(by_n[n])) for n in ns], marker="o")
        ax.set_xlabel("N_particle")
        ax.set_ylabel("median normalized ESS")
        ax.set_title(f"{system}: normalized ESS vs particle count")
        ax.set_xscale("log", base=2)
        save(fig, "norm_ess_vs_N.png")

    # 2) ESS by history step
    by_step: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in particle:
        by_step[int(r["history_step"])][int(r["particle_count"])].append(_f(r, "ESS"))
    if by_step:
        fig, ax = plt.subplots(figsize=(7, 4))
        for step in sorted(by_step):
            ns = sorted(by_step[step])
            ax.plot(
                ns,
                [float(np.nanmedian(by_step[step][n])) for n in ns],
                marker="o",
                label=f"h{step}",
            )
        ax.set_xlabel("N_particle")
        ax.set_ylabel("median ESS")
        ax.set_title(f"{system}: ESS by history step")
        ax.set_xscale("log", base=2)
        ax.legend()
        save(fig, "ess_by_history_step.png")

    # 3–4) u_cont / u_ctrl errors
    if uctrl:
        ns = [int(r["particle_count"]) for r in uctrl]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns, [_f(r, "u_cont_median_abs_error") for r in uctrl], marker="o", label="u_cont")
        ax.plot(ns, [_f(r, "u_ctrl_median_abs_error") for r in uctrl], marker="s", label="u_ctrl")
        ax.set_xlabel("N_particle")
        ax.set_ylabel("median |error| vs reference")
        ax.set_title(f"{system}: control error vs particle count")
        ax.set_xscale("log", base=2)
        ax.legend()
        save(fig, "uctrl_error_vs_N.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns, [_f(r, "frac_u_ctrl_changed") for r in uctrl], marker="o")
        ax.set_xlabel("N_particle")
        ax.set_ylabel("fraction snapped u_ctrl changed")
        ax.set_title(f"{system}: u_ctrl change fraction")
        ax.set_xscale("log", base=2)
        save(fig, "uctrl_change_fraction.png")

    # 5–6) design agreement / regret
    if regret:
        ns = [int(r["particle_count"]) for r in regret]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns, [_f(r, "frac_design_agreement") for r in regret], marker="o", label="design")
        ax.plot(ns, [_f(r, "frac_bus_agreement") for r in regret], marker="s", label="bus")
        ax.plot(ns, [_f(r, "frac_amplitude_agreement") for r in regret], marker="^", label="amp")
        ax.set_xlabel("N_particle")
        ax.set_ylabel("agreement with reference")
        ax.set_title(f"{system}: optimal-design agreement")
        ax.set_xscale("log", base=2)
        ax.legend()
        save(fig, "design_agreement_vs_N.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns, [_f(r, "regret_median_abs_error") for r in regret], marker="o", label="median")
        ax.plot(ns, [_f(r, "regret_p95_abs_error") for r in regret], marker="s", label="p95")
        ax.set_xlabel("N_particle")
        ax.set_ylabel("reference regret")
        ax.set_title(f"{system}: design regret vs particle count")
        ax.set_xscale("log", base=2)
        ax.legend()
        save(fig, "design_regret_vs_N.png")

    # 7–8) Delta_adaptive / Fixed gap
    if adaptive:
        by_n_d: dict[int, list[float]] = defaultdict(list)
        by_n_gap: dict[int, list[float]] = defaultdict(list)
        for r in adaptive:
            n = int(r["particle_count"])
            by_n_d[n].append(_f(r, "Delta_adaptive"))
            fixed = _f(r, "Fixed_objective")
            jad = _f(r, "J_adaptive")
            if fixed == fixed and jad == jad:
                by_n_gap[n].append(fixed - jad)
        ns = sorted(by_n_d)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns, [float(np.nanmean(by_n_d[n])) for n in ns], marker="o")
        ax.axhline(0.0, color="gray", lw=0.8)
        ax.set_xlabel("N_particle")
        ax.set_ylabel("mean Δ_adaptive")
        ax.set_title(f"{system}: Δ_adaptive vs particle count")
        ax.set_xscale("log", base=2)
        save(fig, "delta_adaptive_vs_N.png")

        if by_n_gap:
            fig, ax = plt.subplots(figsize=(6, 4))
            ns2 = sorted(by_n_gap)
            ax.plot(ns2, [float(np.nanmean(by_n_gap[n])) for n in ns2], marker="o")
            ax.set_xlabel("N_particle")
            ax.set_ylabel("Fixed − Adaptive (u_ctrl)")
            ax.set_title(f"{system}: Fixed vs adaptive gap")
            ax.set_xscale("log", base=2)
            save(fig, "fixed_adaptive_gap_vs_N.png")

    # 9) max weight distribution
    if particle:
        ns = sorted({int(r["particle_count"]) for r in particle})
        data = [
            [_f(r, "max_weight") for r in particle if int(r["particle_count"]) == n]
            for n in ns
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(data, labels=[str(n) for n in ns], showfliers=False)
        ax.set_xlabel("N_particle")
        ax.set_ylabel("max posterior weight")
        ax.set_title(f"{system}: max weight by particle count")
        save(fig, "max_weight_boxplot.png")

    return written


def plot_comparison(systems: tuple[str, ...] = ("ieee5", "ieee9")) -> Path | None:
    comp = OUT / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, system in zip(axes, systems):
        path = OUT / f"{system}_T3" / "results" / "uctrl_convergence.csv"
        rows = _read(path)
        if not rows:
            continue
        ns = [int(r["particle_count"]) for r in rows]
        ax.plot(ns, [_f(r, "u_ctrl_median_abs_error") for r in rows], marker="o", label="u_ctrl")
        ax.plot(ns, [_f(r, "u_cont_median_abs_error") for r in rows], marker="s", label="u_cont")
        ax.set_xscale("log", base=2)
        ax.set_title(system)
        ax.set_xlabel("N_particle")
        ax.legend()
    axes[0].set_ylabel("median |error| vs reference")
    fig.suptitle("IEEE5 vs IEEE9 control convergence")
    out = comp / "ieee5_ieee9_convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
