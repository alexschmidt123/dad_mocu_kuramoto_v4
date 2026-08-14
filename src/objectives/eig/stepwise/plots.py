"""Figures for stepwise EIG evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.objectives.eig.stepwise.evaluate import METHOD_LABELS, STEPWISE_METHODS

METHOD_COLORS: dict[str, str] = {
    "dad_spce": "#1f77b4",
    "dad_delta_h": "#d62728",
    "dad_eig": "#1f77b4",
    "moe_sboed": "#9467bd",
    "rl_sboed_eig": "#ff7f0e",
    "myopic_delta_h": "#2ca02c",
    "random": "#7f7f7f",
    "fixed_open_loop": "#bcbd22",
}


def plot_step1_heatmap(
    heatmap: dict[str, Any],
    *,
    out_path: Path,
    system_label: str,
) -> None:
    buses = heatmap["buses"]
    amps = heatmap["amplitudes"]
    grid = np.asarray(heatmap["grid"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(buses) + 1.5)))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(amps)))
    ax.set_xticklabels([f"{a:g}" for a in amps], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(buses)))
    ax.set_yticklabels([f"Bus {b + 1}" for b in buses])
    ax.set_xlabel("Probe amplitude")
    ax.set_ylabel("Probe bus")
    ax.set_title(f"{system_label} — Step-1 EIG E[H₀−H₁ | ξ=(bus, amplitude)]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean ΔH (nats)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="white", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_stepwise_eig(
    method_results: dict[str, Any],
    *,
    out_path: Path,
    system_label: str,
    horizon: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    steps = np.arange(1, horizon + 1)
    for method in STEPWISE_METHODS:
        if method not in method_results:
            continue
        payload = method_results[method]
        mean = np.asarray(payload["mean_eig_by_step"], dtype=np.float64)
        sem = np.asarray(
            [row["sem_eig"] for row in payload["step_summary"]],
            dtype=np.float64,
        )
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, None)
        ax.errorbar(
            steps,
            mean,
            yerr=1.96 * sem,
            marker="o",
            capsize=3,
            label=label,
            color=color,
            linewidth=1.8,
        )
    ax.set_xlabel("Step t")
    ax.set_ylabel("EIG_t = mean ΔH_t (nats)")
    ax.set_title(f"{system_label} — Stepwise EIG (T={horizon})")
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_terminal_eig_vs_T(
    terminal_rows: list[dict[str, Any]],
    *,
    out_path: Path,
    system_label: str,
) -> None:
    if not terminal_rows:
        return
    by_method: dict[str, list[tuple[int, float, float]]] = {}
    for row in terminal_rows:
        method = str(row["method"])
        by_method.setdefault(method, []).append(
            (int(row["T"]), float(row["terminal_eig_mean"]), float(row.get("terminal_eig_sem", 0.0)))
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in STEPWISE_METHODS:
        if method not in by_method:
            continue
        pts = sorted(by_method[method], key=lambda x: x[0])
        T_vals = np.array([p[0] for p in pts], dtype=np.float64)
        mean = np.array([p[1] for p in pts], dtype=np.float64)
        sem = np.array([p[2] for p in pts], dtype=np.float64)
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, None)
        ax.errorbar(
            T_vals,
            mean,
            yerr=1.96 * sem,
            marker="o",
            capsize=3,
            label=label,
            color=color,
            linewidth=1.8,
        )
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Terminal EIG = Σ_t EIG_t (nats)")
    ax.set_title(f"{system_label} — Terminal EIG vs horizon")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
