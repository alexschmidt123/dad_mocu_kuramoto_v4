"""Diagnose terminal-control bank feasibility before regenerating physical data.

This script reuses the stored theta particles and sweeps control profile shape,
duration, and sign through the same CUDA engine used by U-bank generation.  It
does not modify data/<system>.  Results are written under this test directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import load_config_for_run, repo_root
from src.control.cuda_control import CudaControlEngine
from src.control.u_req import ControlSpec
from src.domains.swing.design import build_simulator


HERE = Path(__file__).resolve().parents[2]
RESULTS = HERE / "reports" / "control_bank_generation" / Path(__file__).stem


def count_safe_intervals(row: np.ndarray) -> int:
    x = np.asarray(row, dtype=bool)
    return int(np.sum(x & np.r_[True, ~x[:-1]]))


def summarize_variant(
    *,
    system: str,
    split: str,
    shape: str,
    duration: float,
    sign: int,
    candidates: np.ndarray,
    rocof: np.ndarray,
    nadir: np.ndarray,
    spec: ControlSpec,
) -> dict[str, object]:
    safe = (rocof <= spec.rocof_limit_hz_s) & (nadir >= spec.delta_f_nadir_hz)
    has_safe = safe.any(axis=1)
    first_idx = np.argmax(safe, axis=1)
    U = np.where(has_safe, candidates[first_idx], np.nan)
    nonmonotonic = np.any(safe[:, :-1] & ~safe[:, 1:], axis=1)
    intervals = np.asarray([count_safe_intervals(x) for x in safe])
    return {
        "system": system,
        "split": split,
        "shape": shape,
        "duration": float(duration),
        "sign": int(sign),
        "n_particles": int(safe.shape[0]),
        "feasible_rate": float(has_safe.mean()),
        "u_max_safety_rate": float(safe[:, -1].mean()),
        "monotonic_rate": float((~nonmonotonic).mean()),
        "n_nonmonotonic": int(nonmonotonic.sum()),
        "mean_safe_intervals": float(intervals.mean()),
        "max_safe_intervals": int(intervals.max()),
        "mean_u_req_feasible": float(np.nanmean(U)) if has_safe.any() else None,
        "std_u_req_feasible": float(np.nanstd(U)) if has_safe.any() else None,
        "n_unique_u_req_feasible": int(np.unique(U[has_safe]).size),
        "rocof_u0_p95": float(np.percentile(rocof[:, 0], 95)),
        "rocof_umax_p95": float(np.percentile(rocof[:, -1], 95)),
        "nadir_u0_p05": float(np.percentile(nadir[:, 0], 5)),
        "nadir_umax_p05": float(np.percentile(nadir[:, -1], 5)),
        "passes_documented_assumptions": bool(has_safe.all() and safe[:, -1].all() and not nonmonotonic.any()),
    }


def run_system(system: str, split: str, limit: int | None, batch_size: int) -> list[dict[str, object]]:
    root = repo_root()
    cfg = load_config_for_run(root / "configs" / f"{system}.yaml", root)
    base = ControlSpec.from_cfg(cfg)
    M = np.load(root / "data" / system / split / "theta_M.npy")
    K = np.load(root / "data" / system / split / "theta_K.npy")
    if limit is not None:
        M, K = M[:limit], K[:limit]

    durations = (0.2, 0.5, 1.0, 2.0, 5.0, float(base.T_obs_sec))
    rows: list[dict[str, object]] = []
    for shape in ("step", "hann", "ramp"):
        for duration in durations:
            for sign in (1, -1):
                candidates = np.asarray(base.u_candidates, dtype=np.float64) * sign
                spec = replace(base, profile=replace(base.profile, shape=shape, duration=duration))
                sim = build_simulator(cfg)
                sim.T_obs_sec = spec.T_obs_sec
                sim.ode_dt = spec.ode_dt
                sim.fs_hz = spec.fs_hz
                engine = CudaControlEngine(sim, spec)
                n, n_u = len(M), len(candidates)
                rocof, nadir = engine.simulate_metrics_batch(
                    np.repeat(M, n_u, axis=0),
                    np.repeat(K, n_u, axis=0),
                    np.tile(candidates, n),
                    batch_size=batch_size,
                )
                row = summarize_variant(
                    system=system,
                    split=split,
                    shape=shape,
                    duration=duration,
                    sign=sign,
                    candidates=np.abs(candidates),
                    rocof=rocof.reshape(n, n_u),
                    nadir=nadir.reshape(n, n_u),
                    spec=spec,
                )
                rows.append(row)
                print(
                    f"{system} {shape:4s} duration={duration:4g} sign={sign:+d} "
                    f"feasible={row['feasible_rate']:.3f} umax={row['u_max_safety_rate']:.3f} "
                    f"monotonic={row['monotonic_rate']:.3f}"
                )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", default="ieee5,ieee9,ieee14")
    p.add_argument("--split", default="test", choices=("train", "test"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    rows: list[dict[str, object]] = []
    for system in args.systems.split(","):
        rows.extend(run_system(system.strip(), args.split, args.limit, args.batch_size))
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "terminal_control_profile_sweep.json"
    csv_path = RESULTS / "terminal_control_profile_sweep.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
