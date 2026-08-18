"""Method-comparison tables (CSV) written under an experiment ``eval/`` folder."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

COMPARISON_COLUMNS = [
    "System",
    "Run",
    "T",
    "N_b",
    "Method",
    "mean_u_ctrl",
    "median_u_ctrl",
    "std_u_ctrl",
    "safety_rate",
    "mean_excess",
    "train_s",
    "test_s",
]

PER_ROLLOUT_COLUMNS = [
    "theta_test_id",
    "u_req_true",
    "u_ctrl",
    "excess_control",
    "max_rocof",
    "frequency_nadir",
    "rocof_safe",
    "nadir_safe",
    "safe_total",
]


def aggregate_control_metrics(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    u = np.asarray([float(r["u_ctrl"]) for r in rollouts], dtype=np.float64)
    excess = np.asarray(
        [
            float(r["excess_control"])
            if r.get("excess_control") is not None
            else np.nan
            for r in rollouts
        ],
        dtype=np.float64,
    )
    safe = [
        1.0 if r.get("safe_total") else 0.0
        for r in rollouts
        if r.get("safe_total") is not None
    ]
    return {
        "n": len(rollouts),
        "mean_u_ctrl": float(np.mean(u)) if u.size else float("nan"),
        "median_u_ctrl": float(np.median(u)) if u.size else float("nan"),
        "std_u_ctrl": float(np.std(u)) if u.size else float("nan"),
        "u_ctrl_values": u.tolist(),
        "mean_excess": float(np.nanmean(excess)) if excess.size else float("nan"),
        "safety_rate": float(np.mean(safe)) if safe else float("nan"),
        "mean_weight_sum": float(
            np.mean([float(r.get("weights_sum", 1.0)) for r in rollouts])
        )
        if rollouts
        else float("nan"),
    }


def build_control_table_rows(
    summaries: dict[str, dict[str, Any]],
    timing: dict[str, Any],
    *,
    methods: list[str],
    run_labels: dict[str, Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    train_s = timing.get("training_seconds") or {}
    test_s = timing.get("test_seconds") or {}
    for m in methods:
        s = summaries.get(m) or {}
        t_train = train_s.get(m)
        t_test = (test_s.get(m) or {}).get("test_total_seconds")
        rows.append(
            {
                "System": str(run_labels.get("system_label", "")),
                "Run": str(run_labels.get("run_name", "run")),
                "T": str(run_labels.get("step_number", "")),
                "N_b": str(run_labels.get("n_buses", "")),
                "Method": m,
                "mean_u_ctrl": f"{s.get('mean_u_ctrl', float('nan')):.4f}",
                "median_u_ctrl": f"{s.get('median_u_ctrl', float('nan')):.4f}",
                "std_u_ctrl": f"{s.get('std_u_ctrl', float('nan')):.4f}",
                "safety_rate": f"{s.get('safety_rate', float('nan')):.3f}",
                "mean_excess": f"{s.get('mean_excess', float('nan')):.4f}",
                "train_s": "-" if t_train is None else f"{float(t_train):.1f}",
                "test_s": "-" if t_test is None else f"{float(t_test):.4f}",
            }
        )
    return rows


def save_control_comparison_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPARISON_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in COMPARISON_COLUMNS})


def save_per_rollout_csv(rollouts: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_ROLLOUT_COLUMNS)
        w.writeheader()
        for r in rollouts:
            w.writerow({k: r.get(k, "") for k in PER_ROLLOUT_COLUMNS})


def print_control_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    headers = COMPARISON_COLUMNS
    widths = [max(len(h), max(len(r.get(h, "")) for r in rows)) for h in headers]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print("\n" + line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(r.get(h, "").ljust(w) for h, w in zip(headers, widths)))
