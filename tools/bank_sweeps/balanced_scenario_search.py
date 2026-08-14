"""Search valid terminal scenarios after enforcing the initial equilibrium.

Every candidate uses a restorative step at the contingency bus, with its maximum
magnitude equal to the disturbance magnitude.  This guarantees an interpretable
endpoint: u_max exactly cancels the contingency.  The search reports, but does
not install, scenarios with feasible/monotone and heterogeneous U-banks.
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


def equilibrium_injections(B: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.asarray([np.sum(B[i] * np.sin(theta[i] - theta)) for i in range(len(theta))])


def run_system(system: str, split: str, batch_size: int) -> list[dict[str, object]]:
    root = repo_root()
    cfg = load_config_for_run(root / "configs" / f"{system}.yaml", root)
    base = ControlSpec.from_cfg(cfg)
    M = np.load(root / "data" / system / split / "theta_M.npy")
    K = np.load(root / "data" / system / split / "theta_K.npy")
    magnitudes = (0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0)
    nadir_limits = (-0.05, -0.1, -0.2, -0.3, -0.5, -0.75)
    rocof_limits = (5.0, 10.0, 22.0)
    rows: list[dict[str, object]] = []

    for magnitude in magnitudes:
        grid = np.linspace(0.0, magnitude, 21)
        for nadir_limit in nadir_limits:
            for rocof_limit in rocof_limits:
                sim = build_simulator(cfg)
                sim.P_m = equilibrium_injections(sim.B, sim.theta0)
                spec = replace(
                    base,
                    contingency=replace(base.contingency, magnitude=-magnitude),
                    profile=replace(
                        base.profile,
                        bus=base.contingency.bus,
                        t_start=0.0,
                        duration=base.T_obs_sec,
                        shape="step",
                    ),
                    u_candidates=tuple(float(x) for x in grid),
                    delta_f_nadir_hz=nadir_limit,
                    rocof_limit_hz_s=rocof_limit,
                )
                sim.T_obs_sec = spec.T_obs_sec
                sim.ode_dt = spec.ode_dt
                sim.fs_hz = spec.fs_hz
                engine = CudaControlEngine(sim, spec)
                n, n_u = len(M), len(grid)
                rocof, nadir = engine.simulate_metrics_batch(
                    np.repeat(M, n_u, axis=0), np.repeat(K, n_u, axis=0),
                    np.tile(grid, n), batch_size=batch_size,
                )
                rocof = rocof.reshape(n, n_u)
                nadir = nadir.reshape(n, n_u)
                safe = (rocof <= rocof_limit) & (nadir >= nadir_limit)
                has_safe = safe.any(axis=1)
                nonmono = np.any(safe[:, :-1] & ~safe[:, 1:], axis=1)
                first = np.argmax(safe, axis=1)
                U = np.where(has_safe, grid[first], np.nan)
                valid = bool(has_safe.all() and safe[:, -1].all() and not nonmono.any())
                q95 = float(np.nanquantile(U, 0.95))
                row = {
                    "system": system,
                    "split": split,
                    "contingency_magnitude": -magnitude,
                    "nadir_limit": nadir_limit,
                    "rocof_limit": rocof_limit,
                    "n_particles": n,
                    "feasible_rate": float(has_safe.mean()),
                    "u_max_safety_rate": float(safe[:, -1].mean()),
                    "monotonic_rate": float((~nonmono).mean()),
                    "u_positive_rate": float(np.mean(U > 0)),
                    "u_mean": float(np.nanmean(U)),
                    "u_std": float(np.nanstd(U)),
                    "u_q95": q95,
                    "u_headroom_q95_minus_mean": q95 - float(np.nanmean(U)),
                    "n_unique_u": int(np.unique(U[has_safe]).size),
                    "valid": valid,
                    "quality_candidate": bool(
                        valid and np.mean(U > 0) >= 0.5 and np.nanstd(U) > 0.01
                        and np.unique(U[has_safe]).size >= 4
                    ),
                }
                rows.append(row)
    candidates = [r for r in rows if r["quality_candidate"]]
    candidates.sort(key=lambda r: (-float(r["n_unique_u"]), -float(r["u_std"])))
    print(f"{system}: {len(candidates)} valid heterogeneous candidates")
    for row in candidates[:10]:
        print(
            f"  cont={row['contingency_magnitude']} nadir={row['nadir_limit']} "
            f"rocof={row['rocof_limit']} positive={row['u_positive_rate']:.3f} "
            f"std={row['u_std']:.4f} unique={row['n_unique_u']}"
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", default="ieee5,ieee9,ieee14")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=1024)
    args = p.parse_args()
    rows = []
    for system in args.systems.split(","):
        rows.extend(run_system(system.strip(), args.split, args.batch_size))
    RESULTS.mkdir(parents=True, exist_ok=True)
    jp = RESULTS / "balanced_scenario_search.json"
    cp = RESULTS / "balanced_scenario_search.csv"
    jp.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with cp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {jp}"); print(f"Wrote {cp}")


if __name__ == "__main__":
    main()
