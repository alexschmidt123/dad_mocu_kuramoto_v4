"""CSV / JSON writers for stepwise EIG evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.stepwise_eig.evaluate import STEPWISE_METHODS


def write_rollouts_csv(path: Path, method_results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "rollout_index",
        "step",
        "action_index",
        "bus",
        "amplitude",
        "duration",
        "y_clean",
        "y_noisy",
        "H_t",
        "delta_H_t",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, payload in method_results.items():
            for rec in payload["per_rollout"]:
                ridx = rec["rollout_index"]
                steps = rec["design_selection"].get("steps") or []
                for t, dh in enumerate(rec["delta_h_by_step"]):
                    step_info = steps[t] if t < len(steps) else {}
                    writer.writerow({
                        "method": method,
                        "rollout_index": ridx,
                        "step": t + 1,
                        "action_index": step_info.get("action_index", rec["sequence"][t]),
                        "bus": step_info.get("bus", ""),
                        "amplitude": step_info.get("amplitude", ""),
                        "duration": step_info.get("duration", ""),
                        "y_clean": rec["y_clean"][t],
                        "y_noisy": rec["y_noisy"][t],
                        "H_t": rec["entropy_trace"][t + 1],
                        "delta_H_t": dh,
                    })


def write_entropy_traces_csv(path: Path, method_results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "rollout_index", "t", "H_t"])
        for method, payload in method_results.items():
            for rec in payload["per_rollout"]:
                for t, H in enumerate(rec["entropy_trace"]):
                    writer.writerow([method, rec["rollout_index"], t, H])


def write_stepwise_summary_csv(path: Path, method_results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "method_label",
        "step",
        "mean_eig",
        "std_eig",
        "sem_eig",
        "ci95_low",
        "ci95_high",
        "n_rollouts",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, payload in method_results.items():
            for row in payload["step_summary"]:
                writer.writerow({
                    "method": method,
                    "method_label": payload["method_label"],
                    "n_rollouts": payload["n_rollouts"],
                    **row,
                })


def write_terminal_summary_csv(
    path: Path,
    method_results: dict[str, Any],
    *,
    horizon: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "method_label",
        "T",
        "terminal_eig_mean",
        "terminal_eig_std",
        "terminal_eig_sem",
        "ci95_low",
        "ci95_high",
        "terminal_from_entropy_mean",
        "sum_stepwise_eig",
        "consistency_max_abs_diff",
        "consistency_mean_abs_diff",
        "n_rollouts",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, payload in method_results.items():
            sum_steps = float(sum(payload["mean_eig_by_step"]))
            writer.writerow({
                "method": method,
                "method_label": payload["method_label"],
                "T": horizon,
                "terminal_eig_mean": payload["terminal_eig_mean"],
                "terminal_eig_std": payload["terminal_eig_std"],
                "terminal_eig_sem": payload["terminal_eig_sem"],
                "ci95_low": payload["terminal_eig_ci95_low"],
                "ci95_high": payload["terminal_eig_ci95_high"],
                "terminal_from_entropy_mean": payload["terminal_from_entropy_mean"],
                "sum_stepwise_eig": sum_steps,
                "consistency_max_abs_diff": payload["terminal_consistency_max_abs_diff"],
                "consistency_mean_abs_diff": payload["terminal_consistency_mean_abs_diff"],
                "n_rollouts": payload["n_rollouts"],
            })


def write_step1_heatmap_csv(path: Path, heatmap: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bus",
        "amplitude",
        "action_index",
        "mean_eig_step1",
        "std_eig_step1",
        "sem_eig_step1",
        "ci95_low",
        "ci95_high",
        "n_rollouts",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in heatmap["rows"]:
            writer.writerow({k: row[k] for k in fieldnames})


def write_terminal_vs_T_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "method_label",
        "T",
        "terminal_eig_mean",
        "terminal_eig_sem",
        "ci95_low",
        "ci95_high",
        "experiment_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_consistency_csv(path: Path, method_results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "rollout_index",
        "terminal_from_steps",
        "terminal_from_entropy",
        "abs_diff",
        "H0",
        "HT",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, payload in method_results.items():
            for rec in payload["per_rollout"]:
                writer.writerow({
                    "method": method,
                    "rollout_index": rec["rollout_index"],
                    "terminal_from_steps": rec["terminal_eig_from_steps"],
                    "terminal_from_entropy": rec["terminal_eig_from_entropy"],
                    "abs_diff": rec["terminal_eig_abs_diff"],
                    "H0": rec["entropy_trace"][0],
                    "HT": rec["entropy_trace"][-1],
                })


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize(payload), f, indent=2)


def write_report_md(
    path: Path,
    *,
    system_label: str,
    run_prefix: str,
    horizon: int,
    method_results: dict[str, Any],
    heatmap: dict[str, Any],
    terminal_vs_T: list[dict[str, Any]],
    exp_dir: Path,
) -> None:
    lines = [
        f"# Stepwise EIG report — {system_label}",
        "",
        f"- Run prefix: `{run_prefix}`",
        f"- Primary horizon: T={horizon}",
        f"- Experiment: `{exp_dir}`",
        f"- Metric: realized entropy reduction ΔH_t = H_{{t-1}} - H_t (nats), fresh noise on banked y_sim",
        "",
        "## Step-1 EIG heatmap",
        "",
        "Each cell is mean Step-1 ΔH for design ξ=(bus, amplitude), averaged over test θ*.",
        "",
        "## Stepwise EIG by method",
        "",
    ]
    header_cols = " | ".join(f"EIG_{t}" for t in range(1, horizon + 1))
    header = f"| Method | {header_cols} | Terminal EIG |"
    sep = "|---|" + "|".join(["---:"] * horizon) + "|---:|"
    lines.extend([header, sep])
    for method in method_results:
        payload = method_results[method]
        steps = payload["mean_eig_by_step"]
        cells = " | ".join(f"{v:.4f}" for v in steps)
        lines.append(
            f"| {payload['method_label']} | {cells} | {payload['terminal_eig_mean']:.4f} |"
        )

    lines.extend(["", "## DAD vs Myopic ΔH (terminal EIG)", ""])
    labels = {m: method_results[m]["method_label"] for m in method_results if m in method_results}
    for key in ("dad_spce", "dad_delta_h", "myopic_delta_h"):
        if key not in method_results:
            continue
        p = method_results[key]
        lines.append(
            f"- **{labels[key]}**: terminal EIG = {p['terminal_eig_mean']:.4f} nats "
            f"(Step-1 = {p['mean_eig_by_step'][0]:.4f})"
        )

    if {"dad_spce", "myopic_delta_h"}.issubset(method_results):
        dad = method_results["dad_spce"]
        myopic = method_results["myopic_delta_h"]
        s1_dad = dad["mean_eig_by_step"][0]
        s1_my = myopic["mean_eig_by_step"][0]
        term_dad = dad["terminal_eig_mean"]
        term_my = myopic["terminal_eig_mean"]
        lines.extend([
            "",
            "### Interpretation",
            "",
        ])
        if s1_dad < s1_my and term_dad > term_my:
            lines.append(
                "DAD-sPCE accepts a **smaller Step-1 EIG** than Myopic ΔH but achieves "
                f"**larger terminal EIG** ({term_dad:.4f} vs {term_my:.4f} nats)."
            )
        elif term_dad > term_my:
            lines.append(
                f"DAD-sPCE terminal EIG ({term_dad:.4f}) exceeds Myopic ΔH ({term_my:.4f})."
            )
        else:
            lines.append(
                f"Myopic ΔH terminal EIG ({term_my:.4f}) meets or exceeds DAD-sPCE ({term_dad:.4f})."
            )
        if len(dad["mean_eig_by_step"]) > 1:
            later_dad = float(np.mean(dad["mean_eig_by_step"][1:]))
            later_my = float(np.mean(myopic["mean_eig_by_step"][1:]))
            lines.append(
                f"Mean later-step EIG (t≥2): DAD-sPCE={later_dad:.4f}, Myopic ΔH={later_my:.4f}."
            )

    if terminal_vs_T:
        lines.extend(["", "## Terminal EIG vs horizon", ""])
        lines.append("| T | " + " | ".join(labels.get(m, m) for m in STEPWISE_METHODS if m in labels) + " |")
        lines.append("|---:|" + "|".join(["---:"] * sum(1 for m in STEPWISE_METHODS if m in labels)) + "|")
        by_T: dict[int, dict[str, float]] = {}
        for row in terminal_vs_T:
            by_T.setdefault(int(row["T"]), {})[row["method"]] = float(row["terminal_eig_mean"])
        for T in sorted(by_T):
            vals = []
            for m in STEPWISE_METHODS:
                if m not in labels:
                    continue
                vals.append(f"{by_T[T].get(m, float('nan')):.4f}")
            lines.append(f"| {T} | " + " | ".join(vals) + " |")

    lines.extend(["", "## Terminal consistency checks", ""])
    for method, payload in method_results.items():
        lines.append(
            f"- {payload['method_label']}: max |Σ_t ΔH_t − (H0−HT)| = "
            f"{payload['terminal_consistency_max_abs_diff']:.2e}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
