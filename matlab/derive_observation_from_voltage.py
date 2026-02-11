"""
Derive the same observation (ROCOF_max, f_min) from MATLAB ScopeBus voltage CSVs.

This lets you compare observation *type* with Python: both sides output ROCOF_max (Hz/s)
and f_min (Hz, absolute). Definitions match Python's extract_frequency_features():
  - Phase φ from 3-phase voltage (Clarke); f = (1/2π) dφ/dt; Δf = f - f_nominal
  - ROCOF_max = max |d(Δf)/dt| over window; f_min = f_nominal + min(Δf)
  - Downsampled to fs = 12 Hz for ROCOF (PMU-like).

Run from repo root:
  python matlab/derive_observation_from_voltage.py

Requires: matlab/results/fourteen_bus_dynamic/ScopeBus1.csv ... ScopeBus14.csv
Output: prints ROCOF_max and f_min per bus (and overall); optionally writes
        matlab/results/fourteen_bus_dynamic/observation_from_voltage.csv

Note: MATLAB .mdl has different dynamics (no swing ODE, no probe), so numerical
values will NOT match Python; this script only provides the same *observation
definition* for qualitative comparison.
"""

import csv
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MATLAB_DYNAMIC = REPO / "matlab" / "results" / "fourteen_bus_dynamic"
F_NOMINAL = 50.0  # Hz, match Python
FS = 12.0  # Hz, observation sampling (match Python)


def load_scope_bus(path: Path):
    """Load ScopeBus CSV: first column = time, next 3 = 3-phase voltage (Va, Vb, Vc)."""
    with open(path) as f:
        rows = list(csv.reader(f))
    data = np.array([[float(x) for x in row] for row in rows])
    t = data[:, 0]
    va, vb, vc = data[:, 1], data[:, 2], data[:, 3]
    return t, va, vb, vc


def phase_from_three_phase(va, vb, vc):
    """Clarke transform: (Va,Vb,Vc) -> (V_alpha, V_beta); phase phi = atan2(V_beta, V_alpha)."""
    v_alpha = (2.0 / 3.0) * (va - 0.5 * vb - 0.5 * vc)
    v_beta = (1.0 / np.sqrt(3.0)) * (vb - vc)
    phi = np.arctan2(v_beta, v_alpha)
    return np.unwrap(phi)  # unwrap for continuous phase


def derive_rocof_fmin(t, phi, f_nominal=F_NOMINAL, fs=FS):
    """
    From time and phase (rad), compute ROCOF_max and f_min using same definition as Python.
    - f(t) = (1/2π) dφ/dt (Hz)
    - Δf = f - f_nominal
    - Downsample to fs Hz, then ROCOF = d(Δf)/dt, ROCOF_max = max|ROCOF|, f_min = f_nominal + min(Δf).
    """
    if len(t) < 2:
        return np.nan, np.nan
    dt = np.gradient(t)
    # Instantaneous frequency (Hz)
    f_inst = np.gradient(phi, t) / (2.0 * np.pi)
    delta_f = f_inst - f_nominal

    # Downsample to ~fs Hz for consistency with Python observation
    t_min, t_max = t.min(), t.max()
    n_obs = max(2, int((t_max - t_min) * fs))
    t_obs = np.linspace(t_min, t_max, n_obs)
    delta_f_obs = np.interp(t_obs, t, delta_f)
    h_obs = (t_max - t_min) / (n_obs - 1) if n_obs > 1 else (t_max - t_min)

    rocof = np.gradient(delta_f_obs, h_obs)
    rocof_max = float(np.max(np.abs(rocof)))
    f_min = float(f_nominal + np.min(delta_f_obs))
    return rocof_max, f_min


def main():
    out_dir = MATLAB_DYNAMIC
    if not out_dir.exists():
        print("matlab/results/fourteen_bus_dynamic/ not found. Run run_fourteen_bus_dynamic_save.m first.")
        return

    results = []
    for bus in range(1, 15):
        path = out_dir / f"ScopeBus{bus}.csv"
        if not path.exists():
            continue
        t, va, vb, vc = load_scope_bus(path)
        phi = phase_from_three_phase(va, vb, vc)
        rocof_max, f_min = derive_rocof_fmin(t, phi)
        results.append({"bus": bus, "ROCOF_max": rocof_max, "f_min": f_min})
        print(f"  Bus {bus:2d}: ROCOF_max = {rocof_max:.6f} Hz/s, f_min = {f_min:.6f} Hz")

    if not results:
        print("No ScopeBus*.csv found.")
        return

    rocof_vals = [r["ROCOF_max"] for r in results if not np.isnan(r["ROCOF_max"])]
    fmin_vals = [r["f_min"] for r in results if not np.isnan(r["f_min"])]
    print("\nDerived from MATLAB voltage (same observation definition as Python):")
    print(f"  ROCOF_max: min={min(rocof_vals):.6f}, max={max(rocof_vals):.6f} Hz/s")
    print(f"  f_min:     min={min(fmin_vals):.6f}, max={max(fmin_vals):.6f} Hz")
    print("\nCompare with tests/output/design_comparison_table.csv (Python).")
    print("Numerical match not expected: MATLAB = electrical dynamics, no probe; Python = swing ODE with probe.")

    # Write CSV for side-by-side comparison
    out_csv = out_dir / "observation_from_voltage.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bus", "ROCOF_max", "f_min"])
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
