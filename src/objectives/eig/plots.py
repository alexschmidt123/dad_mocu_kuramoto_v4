"""Plots from eval/summary.csv and eval/*.json: metrics, training curves, cross-system overviews."""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_LABELS: dict[str, str] = {
    "random": "Random",
    "fixed_open_loop": "Fixed open-loop",
    "myopic_delta_h": "Myopic ΔH",
    "dad_spce": "DAD sPCE",
    "dad_delta_h": "DAD ΔH",
    "dad_eig": "DAD-EIG",
    "rl_sboed_eig": "RL-sBOED-EIG",
    "moe_sboed": "MoE-sBOED",
}

METHOD_ORDER = [
    "random",
    "fixed_open_loop",
    "myopic_delta_h",
    "rl_sboed_eig",
    "moe_sboed",
    "dad_eig",
]

DAD_METHODS = ("dad_eig",)

METHOD_COLORS: dict[str, str] = {
    "random": "#7f7f7f",
    "fixed_open_loop": "#bcbd22",
    "moe_sboed": "#9467bd",
    "rl_sboed_eig": "#ff7f0e",
    "myopic_delta_h": "#2ca02c",
    "dad_spce": "#1f77b4",
    "dad_delta_h": "#d62728",
    "dad_eig": "#1f77b4",
}

SYSTEM_PREFIXES = ("ieee5", "ieee9", "ieee14")


def parse_bracket_list(raw: str) -> list[float]:
    text = str(raw).strip()
    if not text:
        return []
    return [float(x) for x in ast.literal_eval(text)]


def parse_time_seconds(raw: str) -> float | None:
    text = str(raw).strip()
    if not text or text in {"-", "—", "NA", "N/A"}:
        return None
    return float(text)


def load_summary_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cumulative_delta_h(delta_h_by_step: list[float]) -> list[float]:
    return list(np.cumsum(delta_h_by_step))


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {r["Method"]: r for r in rows}
    ordered = [by_name[m] for m in METHOD_ORDER if m in by_name]
    extra = [r for r in rows if r["Method"] not in METHOD_ORDER]
    return ordered + extra


def extract_horizon_from_exp_dir(exp_dir: Path) -> int | None:
    match = re.search(r"_T(\d+)(?:_|$)", exp_dir.name)
    return int(match.group(1)) if match else None


def extract_run_prefix_from_exp_dir(exp_dir: Path) -> str | None:
    match = re.search(r"(ieee\d+)_T\d+", exp_dir.name)
    return match.group(1) if match else None


def find_latest_experiment_dirs(
    project_root: Path,
    *,
    run_prefix: str = "ieee14",
) -> dict[int, Path]:
    """Latest experiment folder per horizon T."""
    experiments = project_root / "experiments"
    if not experiments.is_dir():
        return {}

    patterns = [
        f"*{run_prefix}_T*",
        f"{run_prefix}_T*",
    ]
    if not run_prefix.endswith("_config"):
        patterns.extend(
            [
                f"*{run_prefix}_config_T*",
                f"{run_prefix}_config_T*",
            ]
        )

    by_T: dict[int, Path] = {}
    for pattern in patterns:
        for path in experiments.glob(pattern):
            if not path.is_dir():
                continue
            T = extract_horizon_from_exp_dir(path)
            if T is None:
                continue
            prev = by_T.get(T)
            if prev is None or path.stat().st_mtime > prev.stat().st_mtime:
                by_T[T] = path
    return dict(sorted(by_T.items()))


def load_method_eval(exp_dir: Path, method: str) -> dict[str, Any]:
    path = exp_dir / "eval" / f"{method}.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_training_metrics(exp_dir: Path, method: str) -> dict[str, Any]:
    path = exp_dir / "model" / f"{method}_training_metrics.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _saturation_line(ax: plt.Axes, metric: str, *, spce_L: int = 512) -> None:
    if metric == "delta_h":
        ax.axhline(math.log(256), color="gray", linestyle="--", linewidth=1, alpha=0.55, label="log N")
    elif metric == "spce":
        from src.inference.spce import spce_info_ceiling

        ceiling = spce_info_ceiling(spce_L)
        ax.axhline(ceiling, color="gray", linestyle="--", linewidth=1, alpha=0.55, label=f"log(L+1), L={spce_L}")


def plot_experiment_detailed(
    exp_dir: Path,
    *,
    out_path: Path,
) -> None:
    """Six-panel figure from eval JSON: per-step and cumulative ΔH / sPCE, MSE breakdown."""
    T = extract_horizon_from_exp_dir(exp_dir)
    if T is None:
        raise ValueError(f"Cannot parse horizon from {exp_dir.name}")

    methods_present = [
        m for m in METHOD_ORDER if (exp_dir / "eval" / f"{m}.json").is_file()
    ]
    if not methods_present:
        raise FileNotFoundError(f"No eval JSON under {exp_dir / 'eval'}")

    run_prefix = extract_run_prefix_from_exp_dir(exp_dir) or exp_dir.name
    steps = np.arange(1, T + 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    title_bits = [run_prefix, f"T={T}", exp_dir.name]
    fig.suptitle("  ".join(title_bits), fontsize=13, y=1.01)

    mse_theta: list[float] = []
    mse_M: list[float] = []
    mse_K: list[float] = []
    mse_labels: list[str] = []

    for method in methods_present:
        doc = load_method_eval(exp_dir, method)
        summary = doc.get("summary") or {}
        label = _method_label(method)
        color = METHOD_COLORS.get(method)

        dh = summary.get("delta_h_by_step") or []
        spce = summary.get("spce_by_step") or []
        if len(dh) == T:
            axes[0, 0].plot(steps, dh, marker="o", label=label, color=color, linewidth=1.8)
            axes[0, 1].plot(
                steps, cumulative_delta_h(dh), marker="o", label=label, color=color, linewidth=1.8
            )
        if len(spce) == T:
            axes[0, 2].plot(steps, spce, marker="s", label=label, color=color, linewidth=1.8)
            axes[1, 0].plot(
                steps, cumulative_delta_h(spce), marker="s", label=label, color=color, linewidth=1.8
            )

        if summary.get("mse_theta") is not None:
            mse_theta.append(float(summary["mse_theta"]))
            mse_M.append(float(summary.get("mse_M", 0)))
            mse_K.append(float(summary.get("mse_K", 0)))
            mse_labels.append(label)

    for ax, metric_key, subtitle, cumulative in [
        (axes[0, 0], "dh", "Per-step ΔH", False),
        (axes[0, 1], "dh", "Cumulative ΔH", True),
        (axes[0, 2], "spce", "Per-step sPCE", False),
        (axes[1, 0], "spce", "Cumulative sPCE", True),
    ]:
        ax.set_xlabel("Probe step")
        ax.set_ylabel(subtitle.split(" ", 1)[1])
        ax.set_title(subtitle)
        ax.set_xticks(steps)
        ax.grid(True, alpha=0.3)
        if metric_key == "dh":
            _saturation_line(ax, "delta_h")
        else:
            _saturation_line(ax, "spce")
        ax.legend(fontsize=7, loc="best")

    ax_mse = axes[1, 1]
    if mse_labels:
        x = np.arange(len(mse_labels))
        w = 0.25
        ax_mse.bar(x - w, mse_M, w, label="MSE_M", color="#8c564b")
        ax_mse.bar(x, mse_K, w, label="MSE_K", color="#9467bd")
        ax_mse.bar(x + w, mse_theta, w, label="MSE_θ", color="#17becf")
        ax_mse.set_xticks(x)
        ax_mse.set_xticklabels(mse_labels, rotation=25, ha="right")
        ax_mse.set_ylabel("MSE")
        ax_mse.set_title("MSE breakdown (M, K, θ)")
        ax_mse.legend(fontsize=7)
        ax_mse.grid(True, axis="y", alpha=0.3)

    ax_ctx = axes[1, 2]
    ax_ctx.axis("off")
    lines = [f"Experiment: {exp_dir.name}", f"Horizon T = {T}"]
    cfg_path = exp_dir / "run_config.yaml"
    if cfg_path.is_file():
        text = cfg_path.read_text(encoding="utf-8")
        for key in ("sigma:", "support_size:", "mc_samples:", "L:"):
            for line in text.splitlines():
                if line.strip().startswith(key):
                    lines.append(line.strip())
                    break
    first = load_method_eval(exp_dir, methods_present[0]).get("summary") or {}
    lines.append(f"test systems: {first.get('theta_sample_size', '?')}")
    ax_ctx.text(
        0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9,
        family="monospace", transform=ax_ctx.transAxes,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    exp_dir: Path,
    *,
    out_path: Path,
) -> list[str]:
    """Training loss and mean reward for DAD methods (one subplot per method)."""
    methods = [
        m for m in DAD_METHODS
        if (exp_dir / "model" / f"{m}_training_metrics.json").is_file()
    ]
    if not methods:
        return []

    T = extract_horizon_from_exp_dir(exp_dir)
    fig, axes = plt.subplots(len(methods), 2, figsize=(12, 4 * len(methods)), squeeze=False)
    run_prefix = extract_run_prefix_from_exp_dir(exp_dir) or exp_dir.name
    fig.suptitle(
        f"{run_prefix}  T={T}  training curves  ({exp_dir.name})",
        fontsize=13,
        y=1.01,
    )

    for row, method in enumerate(methods):
        metrics = load_training_metrics(exp_dir, method)
        losses = metrics.get("epoch_losses") or []
        rewards = metrics.get("epoch_mean_reward") or []
        epochs = np.arange(1, len(losses) + 1)
        label = _method_label(method)

        ax_loss = axes[row, 0]
        if losses:
            ax_loss.plot(epochs, losses, color=METHOD_COLORS.get(method), linewidth=1.2)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"{label} — epoch loss")
        ax_loss.grid(True, alpha=0.3)

        ax_reward = axes[row, 1]
        if rewards:
            ax_reward.plot(epochs, rewards, color=METHOD_COLORS.get(method), linewidth=1.2)
        ax_reward.set_xlabel("Epoch")
        ax_reward.set_ylabel("Mean reward")
        ax_reward.set_title(f"{label} — epoch mean reward")
        ax_reward.grid(True, alpha=0.3)
        if metrics.get("elapsed_seconds") is not None:
            ax_reward.text(
                0.02, 0.02,
                f"elapsed {metrics['elapsed_seconds']:.1f}s",
                transform=ax_reward.transAxes,
                fontsize=8,
                va="bottom",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return methods


def plot_training_grid(
    exp_dirs_by_T: dict[int, Path],
    *,
    method: str,
    title: str,
    out_path: Path,
) -> None:
    """Reward curves for one DAD method across all horizons (small multiples)."""
    Ts = sorted(exp_dirs_by_T)
    n = len(Ts)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), squeeze=False)
    fig.suptitle(title, fontsize=13, y=1.02)

    for col, T in enumerate(Ts):
        ax = axes[0, col]
        metrics = load_training_metrics(exp_dirs_by_T[T], method)
        rewards = metrics.get("epoch_mean_reward") or []
        if rewards:
            ax.plot(np.arange(1, len(rewards) + 1), rewards, color=METHOD_COLORS.get(method), linewidth=1.2)
        ax.set_title(f"T={T}")
        ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel("Mean reward")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_systems_overview(
    project_root: Path,
    *,
    out_dir: Path,
) -> Path:
    """3×3 grid: ΔH, MSE_θ, Tot.sPCE vs T for ieee5/9/14."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex="col")
    metric_specs = [
        ("ΔH", "Total ΔH", "delta_h"),
        ("MSE_θ", "MSE_θ", None),
        ("Tot.sPCE", "Total sPCE", "spce"),
    ]

    for row, run_prefix in enumerate(SYSTEM_PREFIXES):
        summaries = find_latest_summaries(project_root, run_prefix=run_prefix)
        if not summaries:
            continue
        summaries_by_T = {T: load_summary_csv(p) for T, p in summaries.items()}
        Ts = sorted(summaries_by_T)

        for col, (col_key, ylabel, sat) in enumerate(metric_specs):
            ax = axes[row, col]
            for method in METHOD_ORDER:
                xs: list[int] = []
                ys: list[float] = []
                for T in Ts:
                    rows = {r["Method"]: r for r in summaries_by_T[T]}
                    if method not in rows:
                        continue
                    val_raw = rows[method].get(col_key, "")
                    val = float(val_raw)
                    xs.append(T)
                    ys.append(val)
                if not xs:
                    continue
                ax.plot(
                    xs, ys, marker="o", linewidth=1.8, markersize=5,
                    label=_method_label(method), color=METHOD_COLORS.get(method),
                )
            if sat == "delta_h":
                _saturation_line(ax, "delta_h")
            elif sat == "spce":
                _saturation_line(ax, "spce")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(ylabel)
            if row == 2:
                ax.set_xlabel("Horizon T")
            if col == 2 and row == 0:
                ax.legend(fontsize=7, loc="best")
            if col == 0:
                ax.text(-0.28, 0.5, run_prefix, transform=ax.transAxes,
                        fontsize=11, va="center", rotation=90)

    fig.suptitle("All IEEE systems — metrics vs horizon (σ=0.08)", fontsize=14, y=1.01)
    fig.tight_layout()
    out_path = out_dir / "all_systems_metrics_vs_T.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_method_win_counts(
    project_root: Path,
    *,
    out_dir: Path,
) -> Path:
    """Bar chart of which method wins per (system, T) for ΔH and MSE_θ."""
    metrics = ("ΔH", "MSE_θ")
    wins: dict[str, dict[str, int]] = {m: {method: 0 for method in METHOD_ORDER} for m in metrics}

    for run_prefix in SYSTEM_PREFIXES:
        summaries = find_latest_summaries(project_root, run_prefix=run_prefix)
        for csv_path in summaries.values():
            rows = _ordered_rows(load_summary_csv(csv_path))
            for metric in metrics:
                vals = [(r["Method"], float(r[metric])) for r in rows]
                if metric == "ΔH":
                    winner = max(vals, key=lambda x: x[1])[0]
                else:
                    winner = min(vals, key=lambda x: x[1])[0]
                wins[metric][winner] += 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, metrics):
        methods = METHOD_ORDER
        counts = [wins[metric][m] for m in methods]
        x = np.arange(len(methods))
        ax.bar(x, counts, color=[METHOD_COLORS.get(m) for m in methods], edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([_method_label(m) for m in methods], rotation=20, ha="right")
        ax.set_ylabel("Wins (system × T cells)")
        ax.set_title(f"Best {metric} wins")
        ax.grid(True, axis="y", alpha=0.3)
        for i, c in enumerate(counts):
            ax.text(i, c, str(c), ha="center", va="bottom", fontsize=9)

    fig.suptitle("Method win counts across all horizons and systems", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "method_win_counts.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def find_latest_summaries(
    project_root: Path,
    *,
    run_prefix: str = "ieee14",
) -> dict[int, Path]:
    """Latest ``eval/summary.csv`` per horizon T under ``experiments/``."""
    experiments = project_root / "experiments"
    if not experiments.is_dir():
        return {}

    patterns = [
        f"*{run_prefix}_T*/eval/summary.csv",
        f"{run_prefix}_T*/eval/summary.csv",
    ]
    if not run_prefix.endswith("_config"):
        patterns.extend(
            [
                f"*{run_prefix}_config_T*/eval/summary.csv",
                f"{run_prefix}_config_T*/eval/summary.csv",
            ]
        )

    by_T: dict[int, Path] = {}
    for pattern in patterns:
        for path in experiments.glob(pattern):
            T = extract_horizon_from_exp_dir(path.parent.parent)
            if T is None:
                continue
            prev = by_T.get(T)
            if prev is None or path.stat().st_mtime > prev.stat().st_mtime:
                by_T[T] = path
    return dict(sorted(by_T.items()))


def plot_single_summary(
    rows: list[dict[str, Any]],
    *,
    horizon: int,
    title: str,
    out_path: Path,
) -> None:
    rows = _ordered_rows(rows)
    methods = [r["Method"] for r in rows]
    labels = [_method_label(m) for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(title, fontsize=13, y=1.02)

    # 1) Cumulative ΔH curves
    ax_h = axes[0]
    steps = np.arange(1, horizon + 1)
    for row in rows:
        method = row["Method"]
        dh_steps = parse_bracket_list(row["ΔH_1..T"])
        if len(dh_steps) != horizon:
            raise ValueError(
                f"{method}: expected {horizon} ΔH steps, got {len(dh_steps)}"
            )
        cum_h = cumulative_delta_h(dh_steps)
        ax_h.plot(
            steps,
            cum_h,
            marker="o",
            linewidth=2,
            markersize=5,
            label=_method_label(method),
            color=METHOD_COLORS.get(method),
        )
    ax_h.set_xlabel("Probe step")
    ax_h.set_ylabel("Cumulative ΔH")
    ax_h.set_title("Information gain (cumulative ΔH)")
    ax_h.set_xticks(steps)
    ax_h.grid(True, alpha=0.3)
    ax_h.legend(fontsize=8, loc="best")

    # 2) MSE bar chart
    ax_mse = axes[1]
    mse_vals = [float(row["MSE_θ"]) for row in rows]
    x = np.arange(len(methods))
    bars = ax_mse.bar(
        x,
        mse_vals,
        color=[METHOD_COLORS.get(m, "#888888") for m in methods],
        edgecolor="black",
        linewidth=0.5,
    )
    ax_mse.set_xticks(x)
    ax_mse.set_xticklabels(labels, rotation=25, ha="right")
    ax_mse.set_ylabel("MSE_θ")
    ax_mse.set_title("Parameter MSE")
    ax_mse.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, mse_vals):
        ax_mse.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # 3) Train / test time bars (log y)
    ax_time = axes[2]
    train_vals = [parse_time_seconds(row["train_s"]) for row in rows]
    test_vals = [parse_time_seconds(row["test_s"]) for row in rows]
    width = 0.36
    train_heights = [v if v is not None else np.nan for v in train_vals]
    test_heights = [v if v is not None else np.nan for v in test_vals]

    train_bars = ax_time.bar(
        x - width / 2,
        train_heights,
        width,
        label="train_s (DAD training)",
        color="#4c72b0",
        edgecolor="black",
        linewidth=0.5,
    )
    test_bars = ax_time.bar(
        x + width / 2,
        test_heights,
        width,
        label="test_s (eval rollout + scoring)",
        color="#dd8452",
        edgecolor="black",
        linewidth=0.5,
    )
    ax_time.set_yscale("log")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels, rotation=25, ha="right")
    ax_time.set_ylabel("Time (seconds, log scale)")
    ax_time.set_title("Compute time by phase")
    ax_time.legend(fontsize=8, loc="best")
    ax_time.grid(True, which="both", axis="y", alpha=0.3)

    for bar, val in zip(train_bars, train_vals):
        if val is None:
            continue
        ax_time.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"train\n{val:.2g}s",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    for bar, val in zip(test_bars, test_vals):
        if val is None:
            continue
        ax_time.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"test\n{val:.2g}s",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_h_grid(
    summaries_by_T: dict[int, list[dict[str, Any]]],
    *,
    title: str,
    out_path: Path,
) -> None:
    Ts = sorted(summaries_by_T)
    n = len(Ts)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    fig.suptitle(title, fontsize=13, y=1.02)

    for col, T in enumerate(Ts):
        ax = axes[0, col]
        rows = _ordered_rows(summaries_by_T[T])
        steps = np.arange(1, T + 1)
        for row in rows:
            method = row["Method"]
            dh_steps = parse_bracket_list(row["ΔH_1..T"])
            cum_h = cumulative_delta_h(dh_steps)
            ax.plot(
                steps,
                cum_h,
                marker="o",
                linewidth=2,
                markersize=4,
                label=_method_label(method),
                color=METHOD_COLORS.get(method),
            )
        ax.set_title(f"T = {T}")
        ax.set_xlabel("Probe step")
        ax.set_ylabel("Cumulative ΔH")
        ax.set_xticks(steps)
        ax.grid(True, alpha=0.3)
        if col == n - 1:
            ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stepwise_delta_h_grid(
    summaries_by_T: dict[int, list[dict[str, Any]]],
    *,
    title: str,
    out_path: Path,
    nrows: int = 3,
    ncols: int = 3,
    horizons: list[int] | None = None,
) -> None:
    """Grid of per-step ΔH curves (all methods) — one panel per horizon T."""
    if horizons is None:
        Ts = sorted(summaries_by_T)[: nrows * ncols]
    else:
        Ts = [T for T in horizons if T in summaries_by_T]
    if not Ts:
        raise ValueError("No summaries available for requested horizons")

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    fig.suptitle(title, fontsize=13, y=1.01)

    ymax = 0.0
    for T in Ts:
        for row in _ordered_rows(summaries_by_T[T]):
            dh_steps = parse_bracket_list(row["ΔH_1..T"])
            if dh_steps:
                ymax = max(ymax, max(dh_steps))

    legend_ax = axes_flat[len(Ts) - 1]
    for idx, ax in enumerate(axes_flat):
        if idx >= len(Ts):
            ax.axis("off")
            continue

        T = Ts[idx]
        rows = _ordered_rows(summaries_by_T[T])
        steps = np.arange(1, T + 1)
        for row in rows:
            method = row["Method"]
            dh_steps = parse_bracket_list(row["ΔH_1..T"])
            if len(dh_steps) != T:
                raise ValueError(
                    f"T={T}, {method}: expected {T} ΔH steps, got {len(dh_steps)}"
                )
            ax.plot(
                steps,
                dh_steps,
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=_method_label(method),
                color=METHOD_COLORS.get(method),
            )
        ax.set_title(f"T = {T}")
        ax.set_xlabel("Probe step")
        ax.set_ylabel("Per-step ΔH")
        ax.set_xticks(steps)
        ax.set_ylim(0.0, ymax * 1.08 if ymax > 0 else 1.0)
        ax.grid(True, alpha=0.3)

    handles, labels = legend_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(METHOD_ORDER),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_slug_from_rows(rows: list[dict[str, Any]], *, fallback: str) -> str:
    if not rows:
        return fallback
    run_name = str(rows[0].get("Run") or "").strip()
    if run_name:
        return run_name.replace(" ", "")
    system = str(rows[0].get("System") or "").strip()
    if system:
        return system.replace(" ", "").replace("-", "")
    return fallback


def _summary_title(rows: list[dict[str, Any]], *, horizon: int, exp_name: str, fallback: str) -> str:
    if rows and rows[0].get("System"):
        system = rows[0]["System"]
        run_name = rows[0].get("Run", fallback)
        return f"{system}  {run_name}  T={horizon}  ({exp_name})"
    return f"{fallback}  T={horizon}  ({exp_name})"


def _metric_vs_T_on_axes(
    ax: plt.Axes,
    summaries_by_T: dict[int, list[dict[str, Any]]],
    *,
    col_key: str,
    ylabel: str,
    saturation: str | None = None,
    spce_L: int = 512,
    show_legend: bool = True,
) -> None:
    """Line plot: one metric vs horizon T for each method."""
    Ts = sorted(summaries_by_T)
    for method in METHOD_ORDER:
        xs: list[int] = []
        ys: list[float] = []
        for T in Ts:
            rows = {r["Method"]: r for r in summaries_by_T[T]}
            if method not in rows:
                continue
            xs.append(T)
            ys.append(float(rows[method][col_key]))
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            markersize=6,
            label=_method_label(method),
            color=METHOD_COLORS.get(method),
        )

    ax.set_xlabel("Horizon T")
    ax.set_ylabel(ylabel)
    ax.set_xticks(Ts)
    ax.grid(True, alpha=0.3)
    if saturation == "delta_h":
        _saturation_line(ax, "delta_h")
    elif saturation == "spce":
        _saturation_line(ax, "spce", spce_L=spce_L)
    if show_legend:
        ax.legend(fontsize=9, loc="best")


def _terminal_metric_by_method(
    summaries_by_T: dict[int, list[dict[str, Any]]],
    col_key: str,
) -> tuple[int, list[tuple[str, str, float]]]:
    """Values at the largest horizon T, one row per method in METHOD_ORDER."""
    T_max = max(summaries_by_T)
    rows = {r["Method"]: r for r in summaries_by_T[T_max]}
    entries: list[tuple[str, str, float]] = []
    for method in METHOD_ORDER:
        if method not in rows:
            continue
        entries.append((method, _method_label(method), float(rows[method][col_key])))
    return T_max, entries


def _draw_terminal_metric_table(
    ax_table: plt.Axes,
    entries: list[tuple[str, str, float]],
    *,
    col_key: str,
    T_max: int,
    fontsize: float = 9,
) -> None:
    ax_table.axis("off")
    if not entries:
        return

    metric_label = "ΔH" if col_key == "ΔH" else "Tot.sPCE"
    col_labels = ["Method", f"{metric_label}\n(T={T_max})"]
    cell_text = [[label, f"{val:.4f}"] for _, label, val in entries]
    max_val = max(val for _, _, val in entries)

    table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.35)

    n_rows = len(entries)
    for row_idx, (method, _, val) in enumerate(entries):
        row = row_idx + 1
        color = METHOD_COLORS.get(method, "#cccccc")
        table[(row, 0)].set_facecolor(color)
        table[(row, 0)].set_alpha(0.35)
        if val == max_val:
            table[(row, 1)].set_text_props(weight="bold")

    for col in range(2):
        table[(0, col)].set_facecolor("#f0f0f0")
        table[(0, col)].set_text_props(weight="bold")

    ax_table.set_title("Terminal", fontsize=fontsize + 1, pad=6)


def _metric_vs_T_panel_with_table(
    ax_plot: plt.Axes,
    ax_table: plt.Axes,
    summaries_by_T: dict[int, list[dict[str, Any]]],
    *,
    col_key: str,
    ylabel: str,
    saturation: str | None = None,
    spce_L: int = 512,
    show_legend: bool = True,
    table_fontsize: float = 9,
) -> None:
    _metric_vs_T_on_axes(
        ax_plot,
        summaries_by_T,
        col_key=col_key,
        ylabel=ylabel,
        saturation=saturation,
        spce_L=spce_L,
        show_legend=show_legend,
    )
    T_max, entries = _terminal_metric_by_method(summaries_by_T, col_key)
    _draw_terminal_metric_table(
        ax_table,
        entries,
        col_key=col_key,
        T_max=T_max,
        fontsize=table_fontsize,
    )


def _save_metric_vs_T_with_table(
    summaries_by_T: dict[int, list[dict[str, Any]]],
    *,
    col_key: str,
    ylabel: str,
    saturation: str | None,
    title: str,
    out_path: Path,
    spce_L: int = 512,
    figsize: tuple[float, float] = (10.5, 5.0),
) -> None:
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.0], wspace=0.25)
    ax_plot = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])
    _metric_vs_T_panel_with_table(
        ax_plot,
        ax_table,
        summaries_by_T,
        col_key=col_key,
        ylabel=ylabel,
        saturation=saturation,
        spce_L=spce_L,
        show_legend=True,
        table_fontsize=9,
    )
    ax_plot.set_title(title)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, wspace=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delta_h_vs_T(
    summaries_by_T: dict[int, list[dict[str, Any]]],
    *,
    title: str,
    out_path: Path,
) -> None:
    """Single panel: total ΔH vs horizon T for each method."""
    _save_metric_vs_T_with_table(
        summaries_by_T,
        col_key="ΔH",
        ylabel="Total ΔH",
        saturation="delta_h",
        title=title,
        out_path=out_path,
    )


def plot_spce_vs_T(
    summaries_by_T: dict[int, list[dict[str, Any]]],
    *,
    title: str,
    out_path: Path,
    spce_L: int = 512,
) -> None:
    """Single panel: total sPCE vs horizon T for each method."""
    _save_metric_vs_T_with_table(
        summaries_by_T,
        col_key="Tot.sPCE",
        ylabel="Total sPCE",
        saturation="spce",
        title=title,
        out_path=out_path,
        spce_L=spce_L,
    )


def plot_system_sweep_metrics(
    summaries_by_T: dict[int, list[dict[str, Any]]],
    *,
    system: str,
    out_path: Path,
    spce_L: int = 512,
) -> None:
    """One figure per system: ΔH (top) and sPCE (bottom), each with a terminal table."""
    fig = plt.figure(figsize=(10.5, 9.0))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[3.2, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.38,
        wspace=0.28,
    )

    ax_dh = fig.add_subplot(gs[0, 0])
    ax_dh_tbl = fig.add_subplot(gs[0, 1])
    _metric_vs_T_panel_with_table(
        ax_dh,
        ax_dh_tbl,
        summaries_by_T,
        col_key="ΔH",
        ylabel="Total ΔH",
        saturation="delta_h",
        show_legend=True,
        table_fontsize=9,
    )
    ax_dh.set_title("Total ΔH vs horizon T")

    ax_spce = fig.add_subplot(gs[1, 0])
    ax_spce_tbl = fig.add_subplot(gs[1, 1])
    _metric_vs_T_panel_with_table(
        ax_spce,
        ax_spce_tbl,
        summaries_by_T,
        col_key="Tot.sPCE",
        ylabel="Total sPCE",
        saturation="spce",
        spce_L=spce_L,
        show_legend=False,
        table_fontsize=9,
    )
    ax_spce.set_title("Total sPCE vs horizon T")

    fig.suptitle(system, fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.06)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_six_metrics(
    project_root: Path,
    *,
    out_dir: Path | None = None,
    run_prefixes: tuple[str, ...] = SYSTEM_PREFIXES,
    spce_L: int = 512,
) -> list[Path]:
    """Three figures: one per system with ΔH (top) and sPCE (bottom), each with terminal table."""
    if out_dir is None:
        out_dir = project_root / "documents" / "plots" / "sweep_six"

    written: list[Path] = []
    for run_prefix in run_prefixes:
        paths = find_latest_summaries(project_root, run_prefix=run_prefix)
        if not paths:
            print(f"  (skip {run_prefix}: no summaries)")
            continue

        summaries_by_T = {T: load_summary_csv(p) for T, p in paths.items()}
        first_rows = summaries_by_T[min(summaries_by_T)]
        system = first_rows[0].get("System", run_prefix) if first_rows else run_prefix

        out_path = out_dir / f"{run_prefix}_metrics_vs_T.png"
        plot_system_sweep_metrics(
            summaries_by_T,
            system=system,
            out_path=out_path,
            spce_L=spce_L,
        )
        written.append(out_path)
        print(f"  → {out_path}")

    return written


def plot_metrics_vs_T(
    summaries_by_T: dict[int, list[dict[str, Any]]],
    *,
    title: str,
    out_path: Path,
) -> None:
    """Line plots: ΔH, Tot.sPCE, MSE_θ, train/test time vs horizon T (all methods)."""
    Ts = sorted(summaries_by_T)
    panels: list[tuple[str, str, str, bool]] = [
        ("ΔH", "ΔH", "Total information gain", False),
        ("Tot.sPCE", "Tot.sPCE", "Total sPCE", False),
        ("MSE_θ", "MSE_θ", "Parameter MSE", False),
        ("train_s", "Training time (s)", "DAD training wall time", True),
        ("test_s", "Test time (s)", "Eval rollout + scoring", True),
    ]

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8))
    fig.suptitle(title, fontsize=13, y=1.01)

    for ax, (col, ylabel, subtitle, log_y) in zip(axes.flat, panels):
        for method in METHOD_ORDER:
            xs: list[int] = []
            ys: list[float] = []
            for T in Ts:
                rows = {r["Method"]: r for r in summaries_by_T[T]}
                if method not in rows:
                    continue
                val_raw = rows[method].get(col, "")
                if col in {"train_s", "test_s"}:
                    val = parse_time_seconds(val_raw)
                    if val is None:
                        continue
                else:
                    val = float(val_raw)
                xs.append(T)
                ys.append(val)
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                markersize=6,
                label=_method_label(method),
                color=METHOD_COLORS.get(method),
            )
        ax.set_xlabel("Horizon T")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.set_xticks(Ts)
        ax.grid(True, alpha=0.3)
        if col == "ΔH":
            _saturation_line(ax, "delta_h")
        elif col == "Tot.sPCE":
            _saturation_line(ax, "spce")
        if log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=7, loc="best")

    if len(panels) < nrows * ncols:
        axes.flat[-1].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_summaries(
    project_root: Path,
    *,
    out_dir: Path | None = None,
    run_prefix: str = "ieee14",
) -> list[Path]:
    """Build per-T summary figures and an all-T cumulative-ΔH grid."""
    paths = find_latest_summaries(project_root, run_prefix=run_prefix)
    if not paths:
        raise FileNotFoundError(
            f"No eval/summary.csv found under experiments/{run_prefix}_T*/ "
            f"(or legacy {run_prefix}_config_T*/)"
        )

    if out_dir is None:
        out_dir = project_root / "documents" / "plots"

    written: list[Path] = []
    summaries_by_T: dict[int, list[dict[str, Any]]] = {}
    plot_slug = run_prefix

    for T, csv_path in paths.items():
        rows = load_summary_csv(csv_path)
        summaries_by_T[T] = rows
        exp_name = csv_path.parent.parent.name
        plot_slug = _plot_slug_from_rows(rows, fallback=run_prefix)
        out_path = out_dir / f"{plot_slug}_T{T}_summary.png"
        plot_single_summary(
            rows,
            horizon=T,
            title=_summary_title(rows, horizon=T, exp_name=exp_name, fallback=run_prefix),
            out_path=out_path,
        )
        written.append(out_path)
        print(f"  → {out_path}")

    grid_path = out_dir / f"{plot_slug}_all_T_cumulative_delta_h.png"
    first_rows = summaries_by_T[min(summaries_by_T)]
    grid_title = (
        f"{first_rows[0]['System']} — cumulative ΔH across horizons"
        if first_rows and first_rows[0].get("System")
        else f"{run_prefix} — cumulative ΔH across horizons"
    )
    plot_cumulative_h_grid(
        summaries_by_T,
        title=grid_title,
        out_path=grid_path,
    )
    written.append(grid_path)
    print(f"  → {grid_path}")

    stepwise_grid_path = out_dir / f"{plot_slug}_stepwise_delta_h_3x3.png"
    stepwise_ts = sorted(summaries_by_T)[:9]
    stepwise_title = (
        f"{first_rows[0]['System']} — per-step ΔH by method (T={stepwise_ts[0]}…{stepwise_ts[-1]})"
        if first_rows and first_rows[0].get("System")
        else f"{run_prefix} — per-step ΔH by method (T={stepwise_ts[0]}…{stepwise_ts[-1]})"
    )
    plot_stepwise_delta_h_grid(
        summaries_by_T,
        title=stepwise_title,
        out_path=stepwise_grid_path,
        nrows=3,
        ncols=3,
    )
    written.append(stepwise_grid_path)
    print(f"  → {stepwise_grid_path}")

    if max(summaries_by_T) >= 10:
        stepwise_t10_path = out_dir / f"{plot_slug}_stepwise_delta_h_T10.png"
        plot_stepwise_delta_h_grid(
            summaries_by_T,
            title=(
                f"{first_rows[0]['System']} — per-step ΔH by method (T=10)"
                if first_rows and first_rows[0].get("System")
                else f"{run_prefix} — per-step ΔH by method (T=10)"
            ),
            out_path=stepwise_t10_path,
            nrows=1,
            ncols=1,
            horizons=[10],
        )
        written.append(stepwise_t10_path)
        print(f"  → {stepwise_t10_path}")

    delta_h_path = out_dir / f"{plot_slug}_delta_h_vs_T.png"
    delta_h_title = (
        f"{first_rows[0]['System']} — total ΔH vs horizon T"
        if first_rows and first_rows[0].get("System")
        else f"{run_prefix} — total ΔH vs horizon T"
    )
    plot_delta_h_vs_T(
        summaries_by_T,
        title=delta_h_title,
        out_path=delta_h_path,
    )
    written.append(delta_h_path)
    print(f"  → {delta_h_path}")

    metrics_path = out_dir / f"{plot_slug}_all_T_metrics.png"
    metrics_title = (
        f"{first_rows[0]['System']} — {plot_slug}: ΔH, MSE, train/test time vs T"
        if first_rows and first_rows[0].get("System")
        else f"{run_prefix} — metrics vs horizon T"
    )
    plot_metrics_vs_T(
        summaries_by_T,
        title=metrics_title,
        out_path=metrics_path,
    )
    written.append(metrics_path)
    print(f"  → {metrics_path}")
    return written


def plot_all_detailed(
    project_root: Path,
    *,
    out_dir: Path | None = None,
    run_prefixes: tuple[str, ...] | None = None,
) -> list[Path]:
    """Full visualization bundle: summaries, per-T detail, training curves, cross-system overview."""
    if out_dir is None:
        out_dir = project_root / "documents" / "plots"
    prefixes = run_prefixes or SYSTEM_PREFIXES

    written: list[Path] = []

    for run_prefix in prefixes:
        try:
            written.extend(plot_all_summaries(project_root, out_dir=out_dir / run_prefix, run_prefix=run_prefix))
        except FileNotFoundError:
            print(f"  (skip {run_prefix}: no summaries)")

        exp_dirs = find_latest_experiment_dirs(project_root, run_prefix=run_prefix)
        if not exp_dirs:
            continue

        detail_dir = out_dir / run_prefix / "detailed"
        train_dir = out_dir / run_prefix / "training"

        for T, exp_dir in exp_dirs.items():
            detail_path = detail_dir / f"T{T}_detailed.png"
            plot_experiment_detailed(exp_dir, out_path=detail_path)
            written.append(detail_path)
            print(f"  → {detail_path}")

            train_path = train_dir / f"T{T}_training.png"
            if plot_training_curves(exp_dir, out_path=train_path):
                written.append(train_path)
                print(f"  → {train_path}")

        for method in DAD_METHODS:
            grid_path = train_dir / f"all_T_{method}_reward.png"
            plot_training_grid(
                exp_dirs,
                method=method,
                title=f"{run_prefix} — {_method_label(method)} mean reward vs epoch",
                out_path=grid_path,
            )
            written.append(grid_path)
            print(f"  → {grid_path}")

    overview = plot_all_systems_overview(project_root, out_dir=out_dir)
    written.append(overview)
    print(f"  → {overview}")

    wins_path = plot_method_win_counts(project_root, out_dir=out_dir)
    written.append(wins_path)
    print(f"  → {wins_path}")

    return written
