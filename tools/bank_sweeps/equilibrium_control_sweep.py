"""Test initial-equilibrium and restorative-control corrections on stored theta.

Results are diagnostic only; this script never modifies data/<system>.
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
    return np.asarray(
        [np.sum(B[i] * np.sin(theta[i] - theta)) for i in range(len(theta))],
        dtype=np.float64,
    )


def evaluate(system: str, split: str, batch_size: int) -> list[dict[str, object]]:
    root = repo_root()
    cfg = load_config_for_run(root / "configs" / f"{system}.yaml", root)
    base = ControlSpec.from_cfg(cfg)
    M = np.load(root / "data" / system / split / "theta_M.npy")
    K = np.load(root / "data" / system / split / "theta_K.npy")
    rows: list[dict[str, object]] = []

    for balance_initial in (False, True):
        for cap_at_contingency in (False, True):
            sim = build_simulator(cfg)
            equilibrium_p = equilibrium_injections(sim.B, sim.theta0)
            residual_before = sim.P_m - equilibrium_p
            if balance_initial:
                sim.P_m = equilibrium_p

            cands = np.asarray(base.u_candidates, dtype=np.float64)
            if cap_at_contingency:
                cap = abs(float(base.contingency.magnitude))
                cands = np.unique(np.r_[cands[cands <= cap + 1e-12], cap])
            spec = replace(
                base,
                u_candidates=tuple(float(x) for x in cands),
                profile=replace(base.profile, shape="step", duration=base.T_obs_sec),
            )
            sim.T_obs_sec = spec.T_obs_sec
            sim.ode_dt = spec.ode_dt
            sim.fs_hz = spec.fs_hz
            engine = CudaControlEngine(sim, spec)
            n, n_u = len(M), len(cands)
            rocof, nadir = engine.simulate_metrics_batch(
                np.repeat(M, n_u, axis=0),
                np.repeat(K, n_u, axis=0),
                np.tile(cands, n),
                batch_size=batch_size,
            )
            rocof = rocof.reshape(n, n_u)
            nadir = nadir.reshape(n, n_u)
            safe = (rocof <= spec.rocof_limit_hz_s) & (nadir >= spec.delta_f_nadir_hz)
            has_safe = safe.any(axis=1)
            nonmono = np.any(safe[:, :-1] & ~safe[:, 1:], axis=1) if n_u > 1 else np.zeros(n, bool)
            first = np.argmax(safe, axis=1)
            U = np.where(has_safe, cands[first], np.nan)
            row = {
                "system": system,
                "split": split,
                "balance_initial": balance_initial,
                "cap_at_contingency": cap_at_contingency,
                "n_particles": n,
                "n_candidates": n_u,
                "u_max": float(cands[-1]),
                "configured_net_injection": float(np.sum(sim.P_m if not balance_initial else equilibrium_p)),
                "original_equilibrium_residual_linf": float(np.max(np.abs(residual_before))),
                "original_equilibrium_residual_l2": float(np.linalg.norm(residual_before)),
                "feasible_rate": float(has_safe.mean()),
                "u_max_safety_rate": float(safe[:, -1].mean()),
                "monotonic_rate": float((~nonmono).mean()),
                "mean_u_req_feasible": float(np.nanmean(U)) if has_safe.any() else None,
                "std_u_req_feasible": float(np.nanstd(U)) if has_safe.any() else None,
                "n_unique_u_req_feasible": int(np.unique(U[has_safe]).size),
                "passes_documented_assumptions": bool(has_safe.all() and safe[:, -1].all() and not nonmono.any()),
            }
            rows.append(row)
            print(
                f"{system} balance={balance_initial} cap={cap_at_contingency} "
                f"feasible={row['feasible_rate']:.3f} umax={row['u_max_safety_rate']:.3f} "
                f"monotonic={row['monotonic_rate']:.3f} unique_U={row['n_unique_u_req_feasible']}"
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", default="ieee5,ieee9,ieee14")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()
    rows = []
    for system in args.systems.split(","):
        rows.extend(evaluate(system.strip(), args.split, args.batch_size))
    RESULTS.mkdir(parents=True, exist_ok=True)
    jp = RESULTS / "equilibrium_control_sweep.json"
    cp = RESULTS / "equilibrium_control_sweep.csv"
    jp.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with cp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {jp}")
    print(f"Wrote {cp}")


if __name__ == "__main__":
    main()
