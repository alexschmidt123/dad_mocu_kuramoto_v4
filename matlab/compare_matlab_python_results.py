"""
Load MATLAB Simulink results and Python test outputs; print summary and side-by-side comparison.

Run from repo root:
  python matlab/compare_matlab_python_results.py

Expects:
  - matlab/results/fourteen_bus_dynamic/ScopeBus1.csv .. ScopeBus14.csv
  - matlab/results/fourteen_bus_dynamic/observation_from_voltage.csv (run derive_observation_from_voltage.py first)
  - matlab/results/fourteen_bus/summary.txt, tout.csv (if present)
  - tests/output/design_comparison_table.csv (from pytest design pipeline tests)
"""

import csv
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MATLAB_DYNAMIC = REPO / "matlab" / "results" / "fourteen_bus_dynamic"
OBS_FROM_VOLTAGE = MATLAB_DYNAMIC / "observation_from_voltage.csv"
MATLAB_STEADY = REPO / "matlab" / "results" / "fourteen_bus"
PYTHON_OUTPUT = REPO / "tests" / "output" / "design_comparison_table.csv"


def load_matlab_dynamic():
    """Load ScopeBus1..14 CSVs; return list of (t, signals) per bus, and summary stats."""
    out = []
    for i in range(1, 15):
        p = MATLAB_DYNAMIC / f"ScopeBus{i}.csv"
        if not p.exists():
            continue
        with open(p) as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        data = [[float(x) for x in row] for row in rows]
        t = [row[0] for row in data]
        # columns 1.. are signals (e.g. 3-phase or complex)
        signals = [row[1:] for row in data]
        out.append((i, t, signals))
    return out


def load_python_table():
    """Load design_comparison_table.csv; return rows as list of dicts."""
    if not PYTHON_OUTPUT.exists():
        return None
    with open(PYTHON_OUTPUT) as f:
        r = csv.DictReader(f)
        rows = list(r)
    return rows


def load_observation_from_voltage():
    """Load observation_from_voltage.csv (Simulink-derived ROCOF_max, f_min per bus)."""
    if not OBS_FROM_VOLTAGE.exists():
        return None
    with open(OBS_FROM_VOLTAGE) as f:
        r = csv.DictReader(f)
        rows = list(r)
    return rows


def main():
    print("=" * 60)
    print("MATLAB results (Simulink IEEE 14-bus)")
    print("=" * 60)

    # --- MATLAB dynamic ---
    if not MATLAB_DYNAMIC.exists():
        print("matlab/results/fourteen_bus_dynamic/ not found. Run run_fourteen_bus_dynamic_save.m first.")
    else:
        dynamic = load_matlab_dynamic()
        if not dynamic:
            print("No ScopeBus*.csv found in fourteen_bus_dynamic/.")
        else:
            print(f"\nFourteen_bus_dynamic: {len(dynamic)} buses (ScopeBus1..14)")
            t0, t1 = dynamic[0][1][0], dynamic[0][1][-1]
            npts = len(dynamic[0][1])
            ncols = len(dynamic[0][2][0]) if dynamic[0][2] else 0
            print(f"  Time range: [{t0:.4f}, {t1:.4f}] s")
            print(f"  Points per bus: {npts}")
            print(f"  Signal columns per bus: {ncols} (e.g. 3-phase voltage)")
            # signal stats over first bus
            import numpy as np
            sigs = np.array(dynamic[0][2], dtype=float)
            print(f"  Bus 1 signal range: min={sigs.min():.4f}, max={sigs.max():.4f}")

    # --- MATLAB steady-state ---
    if MATLAB_STEADY.exists():
        tout_path = MATLAB_STEADY / "tout.csv"
        if tout_path.exists():
            with open(tout_path) as f:
                t = [float(row[0]) for row in csv.reader(f) if row]
            print(f"\nFourteen_bus (steady-state 0.12 s): tout {len(t)} points, range [{min(t):.4f}, {max(t):.4f}] s")
        summary = MATLAB_STEADY / "summary.txt"
        if summary.exists():
            print(f"  Summary: {summary.read_text().strip()[:200]}...")
    else:
        print("\nmatlab/results/fourteen_bus/ not found.")

    print("\n" + "=" * 60)
    print("Python test results (swing ODE, design pipeline)")
    print("=" * 60)

    rows = load_python_table()
    if not rows:
        print("tests/output/design_comparison_table.csv not found. Run pytest tests/test_experiment_design_pipeline.py first.")
    else:
        import numpy as np
        rocof = [float(r["ROCOF_max"]) for r in rows]
        fmin = [float(r["f_min"]) for r in rows]
        print(f"\ndesign_comparison_table.csv: {len(rows)} rows (14 buses x 10 amplitudes)")
        print(f"  ROCOF_max (Hz/s): min={min(rocof):.4f}, max={max(rocof):.4f}, mean={np.mean(rocof):.4f}")
        print(f"  f_min (Hz):       min={min(fmin):.4f}, max={max(fmin):.4f}, mean={np.mean(fmin):.4f}")
        print(f"  (Python uses f_nominal = 50 Hz; f_min < 50 during probe; aligned with MATLAB .mdl.)")

    print("\n" + "=" * 60)
    print("Side-by-side comparison (same observation: ROCOF_max, f_min)")
    print("=" * 60)

    import numpy as np
    sim = load_observation_from_voltage()
    py_rows = load_python_table()
    if sim and py_rows:
        rocof_sim = [float(r["ROCOF_max"]) for r in sim]
        fmin_sim = [float(r["f_min"]) for r in sim]
        rocof_py = [float(r["ROCOF_max"]) for r in py_rows]
        fmin_py = [float(r["f_min"]) for r in py_rows]
        print("\n  Source                          ROCOF_max (Hz/s)        f_min (Hz)")
        print("  " + "-" * 58)
        print(f"  Simulink (from voltage, 14 buses)  [{min(rocof_sim):.4f}, {max(rocof_sim):.4f}]   [{min(fmin_sim):.4f}, {max(fmin_sim):.4f}]")
        print(f"  Python (swing ODE, 140 designs)    [{min(rocof_py):.4f}, {max(rocof_py):.4f}]   [{min(fmin_py):.4f}, {max(fmin_py):.4f}]")
        print("\n  Difference (why they differ):")
        print("  - Simulink: no probe; small frequency deviation → ROCOF ~0.01–0.03 Hz/s, f_min ≈ 50 Hz.")
        print("  - Python:   Hann probe at one bus → large ROCOF (several Hz/s) and f_min drops (e.g. 49.1–49.5 Hz).")
        print("  - Same observation definition (12 Hz, f_nominal 50 Hz); different dynamics ⇒ different values.")
    elif not sim and py_rows:
        print("\n  Run first: python matlab/derive_observation_from_voltage.py")
        print("  to get Simulink-derived ROCOF_max, f_min for side-by-side comparison.")
    else:
        print("\n  Need both observation_from_voltage.csv and design_comparison_table.csv for side-by-side.")

    print("\n" + "=" * 60)
    print("Summary (Simulink vs Python)")
    print("=" * 60)
    print("""
  - Models: Simulink = detailed electrical (Simscape); Python = reduced swing ODE (θ, ω) with probe.
  - Outputs: Simulink = voltage time series → we derive ROCOF_max, f_min; Python = ROCOF_max, f_min per (bus, amplitude).
  - Time: both 5 s; nominal frequency: both 50 Hz; topology: same IEEE 14.
  - Numerical match not expected; use Simulink for reference (electrical), Python for design (ROCOF/posterior).
""")


if __name__ == "__main__":
    main()
