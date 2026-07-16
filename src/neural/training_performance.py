"""Training performance validation (writes to experiments/result/)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import repo_root


def result_dir(project_root: Path | None = None) -> Path:
    root = project_root or repo_root()
    out = root / "experiments" / "result"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _method_delta_h(summaries: dict[str, Any], method: str) -> float | None:
    s = summaries.get(method)
    if not isinstance(s, dict):
        return None
    if "delta_h" in s:
        return float(s["delta_h"])
    return None


def training_performance_verdict(summaries: dict[str, Any]) -> dict[str, Any]:
    """Compare DAD methods against myopic on mean terminal ΔH."""
    myopic = _method_delta_h(summaries, "myopic_delta_h")
    dad_spce = _method_delta_h(summaries, "dad_spce")
    dad_dh = _method_delta_h(summaries, "dad_delta_h")
    best_dad = max(v for v in (dad_spce, dad_dh) if v is not None) if any(
        v is not None for v in (dad_spce, dad_dh)
    ) else None

    passed = False
    if myopic is not None and best_dad is not None:
        passed = best_dad >= myopic - 1e-6

    return {
        "myopic_delta_h": myopic,
        "dad_spce_delta_h": dad_spce,
        "dad_delta_h_delta_h": dad_dh,
        "best_dad_delta_h": best_dad,
        "passed": passed,
        "verdict": (
            "PASS: best DAD method matches or exceeds myopic on mean ΔH."
            if passed
            else "FAIL: DAD did not reach myopic mean ΔH on the evaluation split."
        ),
    }


def write_training_performance_training(
    exp_dir: Path,
    method: str,
    metrics: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> Path:
    """Save post-training metrics under experiments/result/."""
    root = project_root or repo_root()
    run_name = exp_dir.name
    out = result_dir(root) / f"{_stamp()}_{run_name}_{method}_training_performance.json"
    payload = {
        "check": "training_performance",
        "phase": "training",
        "method": method,
        "experiment_dir": str(exp_dir.resolve()),
        "metrics": metrics,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def write_training_performance_evaluation(
    exp_dir: Path,
    summaries: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    project_root: Path | None = None,
) -> Path:
    """Save evaluation comparison and training_performance verdict under experiments/result/."""
    root = project_root or repo_root()
    run_name = exp_dir.name
    verdict = training_performance_verdict(summaries)
    out = result_dir(root) / f"{_stamp()}_{run_name}_training_performance.json"
    payload = {
        "check": "training_performance",
        "phase": "evaluation",
        "experiment_dir": str(exp_dir.resolve()),
        "comparison_rows": rows,
        "summaries": summaries,
        "training_performance": verdict,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
