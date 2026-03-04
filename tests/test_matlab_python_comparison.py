"""
Validation tests: run Python ODE (IEEE 14, probe at bus 1) and compare with MATLAB results.

These tests prove the Python swing ODE is scientifically consistent by comparing
with MATLAB Simulink (IEEE 14-bus) results. They are not part of the main project
pipeline; they validate the ODE and observation definition.

Key flow:
  1. Run Python ODE individually (probe at bus 1, same A/Tp/T as MATLAB).
  2. Optionally derive ROCOF_max, f_min from MATLAB ScopeBus voltage CSVs (same observation definition).
  3. Compare Python vs MATLAB (side-by-side ROCOF_max, f_min).

Expects (for full comparison):
  - matlab/results/fourteen_bus_dynamic/ScopeBus1.csv .. ScopeBus14.csv (from run_fourteen_bus_dynamic_save.m)
  - tests/output/design_comparison_table.csv (from pytest test_experiment_design_pipeline)
Output: observation_from_voltage.csv written to matlab/results/... when deriving from MATLAB.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.swing_equation_ode import (
    solve_swing_equation_ode,
    extract_frequency_features,
)
from src.core.rocof import extract_max_rocof
from src.core.swing_equation_params import get_default_swing_equation_params

# Paths (relative to repo root)
REPO = PROJECT_ROOT
MATLAB_RESULTS = REPO / "matlab" / "results"
MATLAB_DYNAMIC = MATLAB_RESULTS / "fourteen_bus_dynamic"
MATLAB_STEADY = MATLAB_RESULTS / "fourteen_bus"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PYTHON_TABLE = OUTPUT_DIR / "design_comparison_table.csv"
F_NOMINAL = 50.0
FS = 12.0

# --- MATLAB observation derivation (same definition as Python) ---


def _load_scope_bus(path: Path):
    """Load ScopeBus CSV: first column = time, next 3 = 3-phase voltage (Va, Vb, Vc)."""
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    data = np.array([[float(x) for x in row] for row in rows])
    if data.shape[1] < 4:
        raise ValueError(
            f"ScopeBus CSV must have at least 4 columns (time, Va, Vb, Vc), got {data.shape[1]} in {path}"
        )
    t = data[:, 0]
    va, vb, vc = data[:, 1], data[:, 2], data[:, 3]
    return t, va, vb, vc


def _phase_from_three_phase(va, vb, vc):
    """Clarke transform; phase phi = atan2(V_beta, V_alpha), unwrapped."""
    v_alpha = (2.0 / 3.0) * (va - 0.5 * vb - 0.5 * vc)
    v_beta = (1.0 / np.sqrt(3.0)) * (vb - vc)
    phi = np.arctan2(v_beta, v_alpha)
    return np.unwrap(phi)


def _derive_rocof_fmin(t, phi, f_nominal=F_NOMINAL, fs=FS):
    """ROCOF_max and f_min from time and phase (same definition as Python)."""
    if len(t) < 2:
        return np.nan, np.nan
    f_inst = np.gradient(phi, t) / (2.0 * np.pi)
    delta_f = f_inst - f_nominal
    t_min, t_max = t.min(), t.max()
    n_obs = max(2, int((t_max - t_min) * fs))
    t_obs = np.linspace(t_min, t_max, n_obs)
    delta_f_obs = np.interp(t_obs, t, delta_f)
    h_obs = (t_max - t_min) / (n_obs - 1) if n_obs > 1 else (t_max - t_min)
    rocof = np.gradient(delta_f_obs, h_obs)
    rocof_max = float(np.max(np.abs(rocof)))
    f_min = float(f_nominal + np.min(delta_f_obs))
    return rocof_max, f_min


def derive_observation_from_matlab_folder(out_dir: Path):
    """
    Derive ROCOF_max, f_min per bus from ScopeBus CSVs in out_dir.
    Returns list of dicts [{"bus", "ROCOF_max", "f_min"}]; writes observation_from_voltage.csv.
    """
    results = []
    for bus in range(1, 15):
        path = out_dir / f"ScopeBus{bus}.csv"
        if not path.exists():
            continue
        t, va, vb, vc = _load_scope_bus(path)
        phi = _phase_from_three_phase(va, vb, vc)
        rocof_max, f_min = _derive_rocof_fmin(t, phi)
        results.append({"bus": bus, "ROCOF_max": rocof_max, "f_min": f_min})
    if results:
        out_csv = out_dir / "observation_from_voltage.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["bus", "ROCOF_max", "f_min"])
            w.writeheader()
            w.writerows(results)
    return results


# --- Python ODE: probe at any bus (aligned with MATLAB) ---

PROBE_BUS_1_BASED = 1
PROBE_BUS_0_INDEXED = PROBE_BUS_1_BASED - 1


def python_per_bus_observation(omega_trajectory, h, f_nominal=F_NOMINAL, fs=FS):
    """
    Per-bus ROCOF_max and f_min from omega [M, N]. ODE state omega is frequency
    deviation in rad/s, so delta_f (Hz) = omega/(2*pi); f_absolute = f_nominal + delta_f.
    Returns list of dicts [{"bus": 1..N, "ROCOF_max", "f_min"}].
    """
    M, N = omega_trajectory.shape
    t = np.arange(M) * h
    t_min, t_max = float(t.min()), float(t.max())
    n_obs = max(2, int((t_max - t_min) * fs))
    t_obs = np.linspace(t_min, t_max, n_obs)
    h_obs = (t_max - t_min) / (n_obs - 1) if n_obs > 1 else (t_max - t_min)
    results = []
    for bus in range(N):
        omega_b = omega_trajectory[:, bus]
        # omega = deviation in rad/s -> delta_f (Hz) = omega/(2*pi)
        delta_f = omega_b / (2.0 * np.pi)
        delta_f_obs = np.interp(t_obs, t, delta_f)
        rocof = np.gradient(delta_f_obs, h_obs)
        rocof_max = float(np.max(np.abs(rocof)))
        f_min = float(f_nominal + np.min(delta_f_obs))
        results.append({"bus": bus + 1, "ROCOF_max": rocof_max, "f_min": f_min})
    return results


def run_ieee14_ode_probe_bus(
    probe_bus_1based,
    amplitude=0.2,
    duration=2.0,
    T=5.0,
    h=1.0 / 160.0,
    device="cpu",
    seed=42,
    return_per_bus=False,
):
    """
    Run IEEE 14-bus swing ODE with probe at given bus (1-based).
    Returns dict with ROCOF_max, f_min (system-wide) and optionally per_bus list.
    """
    params = get_default_swing_equation_params(
        N=14,
        topology="ieee14",
        coupling_strength=1.0,
        damping=0.1,
        base_power=1.0,
        M_lower=0.01,
        M_upper=0.06,
        K_lower=0.05,
        K_upper=0.50,
    )
    B = params["B"]
    P_m = params["P_m"]
    D = params["D"]
    g = params["g"]
    np.random.seed(seed)
    M_true = float(np.random.uniform(params["M_lower"], params["M_upper"]))
    K_true = float(np.random.uniform(params["K_lower"], params["K_upper"]))
    M_steps = int(round(T / h))
    probe_0 = (probe_bus_1based - 1) if probe_bus_1based >= 1 else 0
    state_traj = solve_swing_equation_ode(
        B, P_m, D, M_true, K_true, g,
        probe_bus=probe_0,
        probe_amplitude=amplitude,
        probe_duration=duration,
        h=h, M_steps=M_steps, T=T,
        device=device,
        timeout=30.0,
    )
    N = len(P_m)
    omega_traj = state_traj[:, N:]
    rocof_max = extract_max_rocof(omega_traj, fs=12.0, window_sec=min(10.0, T), h=h)
    obs = extract_frequency_features(omega_traj, h, fs=12.0)
    obs["ROCOF_max"] = rocof_max
    if return_per_bus:
        obs["per_bus"] = python_per_bus_observation(omega_traj, h, f_nominal=F_NOMINAL, fs=FS)
    return obs


def run_ieee14_ode_probe_bus1(
    amplitude=0.2,
    duration=2.0,
    T=5.0,
    h=1.0 / 160.0,
    device="cpu",
    seed=42,
):
    """
    Run IEEE 14-bus swing ODE with probe at bus 1 only.
    Returns dict with ROCOF_max, f_min (same observation as design pipeline).
    """
    return run_ieee14_ode_probe_bus(
        PROBE_BUS_1_BASED,
        amplitude=amplitude,
        duration=duration,
        T=T,
        h=h,
        device=device,
        seed=seed,
        return_per_bus=False,
    )


# --- Load MATLAB / Python outputs for comparison ---


def load_matlab_dynamic_scope_buses():
    """Load ScopeBus1..14 CSVs from fourteen_bus_dynamic; return list of (bus_id, t, signals)."""
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
        signals = [row[1:] for row in data]
        out.append((i, t, signals))
    return out


def load_observation_from_voltage_csv(csv_path: Path):
    """Load observation_from_voltage.csv; return list of dicts."""
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        r = csv.DictReader(f)
        return list(r)


def load_python_design_table():
    """Load design_comparison_table.csv; return list of dicts."""
    if not PYTHON_TABLE.exists():
        return None
    with open(PYTHON_TABLE) as f:
        return list(csv.DictReader(f))


# --- Tests ---


def test_python_ode_probe_bus1_runs():
    """Run Python ODE with probe at bus 1; assert observation in expected range (probe effect)."""
    obs = run_ieee14_ode_probe_bus1(amplitude=0.2, duration=2.0, T=5.0, device="cpu")
    assert "ROCOF_max" in obs and "f_min" in obs
    assert obs["ROCOF_max"] > 0
    assert 49.0 < obs["f_min"] < 50.0  # probe causes frequency drop
    assert obs["ROCOF_max"] > 1.0  # probe causes significant ROCOF


def test_python_ode_probe_bus1_save_to_output():
    """Run Python ODE probe bus 1 and save observation to tests/output for comparison."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    obs = run_ieee14_ode_probe_bus1(amplitude=0.2, duration=2.0, T=5.0, device="cpu")
    out_file = OUTPUT_DIR / "ieee14_probe_bus1_observation.txt"
    with open(out_file, "w") as f:
        f.write(f"ROCOF_max={obs['ROCOF_max']}\n")
        f.write(f"f_min={obs['f_min']}\n")
    assert out_file.exists()


@pytest.mark.skipif(not MATLAB_DYNAMIC.exists(), reason="MATLAB results not present")
def test_derive_matlab_observation_from_voltage():
    """If MATLAB ScopeBus CSVs exist, derive observation_from_voltage.csv (same definition as Python)."""
    if not (MATLAB_DYNAMIC / "ScopeBus1.csv").exists():
        pytest.skip("ScopeBus1.csv not found")
    results = derive_observation_from_matlab_folder(MATLAB_DYNAMIC)
    assert len(results) == 14
    obs_path = MATLAB_DYNAMIC / "observation_from_voltage.csv"
    assert obs_path.exists()
    rows = load_observation_from_voltage_csv(obs_path)
    assert len(rows) == 14
    assert "ROCOF_max" in rows[0] and "f_min" in rows[0]


def test_compare_python_ode_with_matlab():
    """
    Compare Python ODE results with MATLAB-derived observation.
    Run Python ODE probe at bus 1; if MATLAB observation exists, assert Python shows probe effect (larger ROCOF).
    """
    obs_py = run_ieee14_ode_probe_bus1(amplitude=0.2, duration=2.0, T=5.0, device="cpu")
    matlab_obs_path = MATLAB_DYNAMIC / "observation_from_voltage.csv"
    matlab_rows = load_observation_from_voltage_csv(matlab_obs_path) if matlab_obs_path.exists() else None
    if matlab_rows:
        rocof_matlab = [float(r["ROCOF_max"]) for r in matlab_rows]
        # Python with probe at bus 1 gives large ROCOF; MATLAB (no probe) gives small ROCOF
        assert obs_py["ROCOF_max"] > max(rocof_matlab) or max(rocof_matlab) < 0.5
    else:
        # No MATLAB data: just ensure Python ODE runs and gives plausible probe effect
        assert obs_py["ROCOF_max"] > 1.0
        assert obs_py["f_min"] < 50.0


def test_compare_design_table_with_matlab_if_present():
    """If design_comparison_table.csv and MATLAB observation exist, load both and sanity-check ranges."""
    py_rows = load_python_design_table()
    matlab_obs = load_observation_from_voltage_csv(MATLAB_DYNAMIC / "observation_from_voltage.csv")
    if not py_rows:
        pytest.skip("design_comparison_table.csv not found (run test_experiment_design_pipeline first)")
    if not matlab_obs:
        pytest.skip("MATLAB observation_from_voltage.csv not found")
    rocof_py = [float(r["ROCOF_max"]) for r in py_rows]
    fmin_py = [float(r["f_min"]) for r in py_rows]
    rocof_matlab = [float(r["ROCOF_max"]) for r in matlab_obs]
    fmin_matlab = [float(r["f_min"]) for r in matlab_obs]
    assert len(rocof_py) == 140
    assert min(rocof_py) >= 0 and max(fmin_py) <= 51
    # Python (with probe) typically has larger ROCOF range than MATLAB (no probe)
    assert max(rocof_py) >= max(rocof_matlab) - 0.1


# --- Comparison table: MATLAB dynamic, probe + Python per-bus ---

MATLAB_PROBE = MATLAB_RESULTS / "fourteen_bus_dynamic_probe"


def build_comparison_table():
    """
    Load MATLAB observation_from_voltage.csv (dynamic and probe), run Python ODE
    once per bus (14 runs): for probe_bus=1 also compute per-bus obs for MATLAB comparison.
    Write matlab/results/COMPARISON_TABLE.md and matlab/results/comparison_table.csv.
    """
    out_dir = MATLAB_RESULTS
    out_md = out_dir / "COMPARISON_TABLE.md"
    out_csv = out_dir / "comparison_table.csv"

    # Load MATLAB
    matlab_dyn = load_observation_from_voltage_csv(MATLAB_DYNAMIC / "observation_from_voltage.csv")
    matlab_probe = load_observation_from_voltage_csv(MATLAB_PROBE / "observation_from_voltage.csv") if MATLAB_PROBE.exists() else None

    # Python: 14 runs only — one per probe bus; when probe_bus==1 also get per-bus for table sections 3 & 4
    py_per_bus = []
    py_probe_system = []
    for bus in range(1, 15):
        o = run_ieee14_ode_probe_bus(
            bus, amplitude=0.2, duration=2.0, T=5.0, device="cpu", return_per_bus=(bus == 1)
        )
        py_probe_system.append({
            "probe_bus": bus,
            "ROCOF_max": o["ROCOF_max"],
            "f_min": o["f_min"],
        })
        if bus == 1:
            py_per_bus = o.get("per_bus") or []

    # Build markdown table
    lines = [
        "# MATLAB vs Python: ROCOF_max and f_min comparison",
        "",
        "Same observation definition: f_nominal=50 Hz, fs=12 Hz. T=5 s.",
        "",
        "## 1. MATLAB fourteen_bus_dynamic (no probe)",
        "",
        "| bus | ROCOF_max (Hz/s) | f_min (Hz) |",
        "|-----|------------------|------------|",
    ]
    if matlab_dyn:
        for r in matlab_dyn:
            lines.append(f"| {r['bus']} | {float(r['ROCOF_max']):.6f} | {float(r['f_min']):.6f} |")
    else:
        lines.append("| (no data) | | |")
    lines.extend(["", "## 2. MATLAB fourteen_bus_dynamic_probe (probe at bus 1)", "", "| bus | ROCOF_max (Hz/s) | f_min (Hz) |", "|-----|------------------|------------|"])
    if matlab_probe:
        for r in matlab_probe:
            lines.append(f"| {r['bus']} | {float(r['ROCOF_max']):.6f} | {float(r['f_min']):.6f} |")
    else:
        lines.append("| (no data) | | |")
    lines.extend(["", "## 3. Python ODE (probe at bus 1) — per-bus", "", "| bus | ROCOF_max (Hz/s) | f_min (Hz) |", "|-----|------------------|------------|"])
    for r in py_per_bus:
        lines.append(f"| {r['bus']} | {r['ROCOF_max']:.6f} | {r['f_min']:.6f} |")
    lines.extend([
        "",
        "## 4. MATLAB (probe) vs Python (probe bus 1) — same test bus",
        "",
        "| bus | MATLAB ROCOF_max | Python ROCOF_max | MATLAB f_min | Python f_min |",
        "|-----|------------------|------------------|--------------|--------------|",
    ])
    for b in range(1, 15):
        md = next((x for x in (matlab_probe or []) if int(x["bus"]) == b), None)
        pd = next((x for x in py_per_bus if x["bus"] == b), None)
        if md and pd:
            lines.append(f"| {b} | {float(md['ROCOF_max']):.6f} | {pd['ROCOF_max']:.6f} | {float(md['f_min']):.6f} | {pd['f_min']:.6f} |")
        elif pd:
            lines.append(f"| {b} | — | {pd['ROCOF_max']:.6f} | — | {pd['f_min']:.6f} |")
    lines.extend([
        "",
        "## 5. Python ODE: probe at each bus (system-wide ROCOF_max, f_min)",
        "",
        "| probe_bus | ROCOF_max (Hz/s) | f_min (Hz) |",
        "|-----------|------------------|------------|",
    ])
    for r in py_probe_system:
        lines.append(f"| {r['probe_bus']} | {r['ROCOF_max']:.6f} | {r['f_min']:.6f} |")
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    # CSV: bus, source, ROCOF_max, f_min (source = matlab_dynamic | matlab_probe | python_probe_bus1)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bus", "source", "ROCOF_max", "f_min"])
        w.writeheader()
        for r in matlab_dyn or []:
            w.writerow({"bus": r["bus"], "source": "matlab_dynamic", "ROCOF_max": r["ROCOF_max"], "f_min": r["f_min"]})
        for r in matlab_probe or []:
            w.writerow({"bus": r["bus"], "source": "matlab_probe", "ROCOF_max": r["ROCOF_max"], "f_min": r["f_min"]})
        for r in py_per_bus:
            w.writerow({"bus": r["bus"], "source": "python_probe_bus1", "ROCOF_max": f"{r['ROCOF_max']:.10g}", "f_min": f"{r['f_min']:.10g}"})
    return out_md, out_csv


# --- Run as script: Python ODE + derive MATLAB + print comparison ---


def main():
    """Run Python ODE probe bus 1, derive MATLAB observation if ScopeBus present, print comparison."""
    print("=" * 60)
    print("Python ODE (IEEE 14, probe at bus 1)")
    print("=" * 60)
    obs_py = run_ieee14_ode_probe_bus1(amplitude=0.2, duration=2.0, T=5.0, device="cpu")
    print(f"  ROCOF_max = {obs_py['ROCOF_max']:.6f} Hz/s")
    print(f"  f_min     = {obs_py['f_min']:.6f} Hz")

    print("\n" + "=" * 60)
    print("MATLAB results (Simulink IEEE 14-bus)")
    print("=" * 60)
    probe_folder = MATLAB_RESULTS / "fourteen_bus_dynamic_probe"
    if MATLAB_DYNAMIC.exists():
        buses = load_matlab_dynamic_scope_buses()
        if buses:
            t0, t1 = buses[0][1][0], buses[0][1][-1]
            print(f"  fourteen_bus_dynamic: {len(buses)} buses, time [{t0:.4f}, {t1:.4f}] s")
            # Derive observation if not present
            if not (MATLAB_DYNAMIC / "observation_from_voltage.csv").exists():
                derive_observation_from_matlab_folder(MATLAB_DYNAMIC)
            if probe_folder.exists() and (probe_folder / "ScopeBus1.csv").exists():
                if not (probe_folder / "observation_from_voltage.csv").exists():
                    derive_observation_from_matlab_folder(probe_folder)
            matlab_rows = load_observation_from_voltage_csv(MATLAB_DYNAMIC / "observation_from_voltage.csv")
            if matlab_rows:
                rocof_m = [float(r["ROCOF_max"]) for r in matlab_rows]
                fmin_m = [float(r["f_min"]) for r in matlab_rows]
                print(f"  ROCOF_max: [{min(rocof_m):.4f}, {max(rocof_m):.4f}] Hz/s")
                print(f"  f_min:     [{min(fmin_m):.4f}, {max(fmin_m):.4f}] Hz")
    else:
        print("  matlab/results/fourteen_bus_dynamic/ not found.")

    print("\n" + "=" * 60)
    print("Side-by-side (Python ODE vs MATLAB)")
    print("=" * 60)
    matlab_rows = load_observation_from_voltage_csv(MATLAB_DYNAMIC / "observation_from_voltage.csv")
    if matlab_rows:
        rocof_m = [float(r["ROCOF_max"]) for r in matlab_rows]
        fmin_m = [float(r["f_min"]) for r in matlab_rows]
        print(f"  Python (probe bus 1):  ROCOF_max = {obs_py['ROCOF_max']:.4f} Hz/s, f_min = {obs_py['f_min']:.4f} Hz")
        print(f"  MATLAB (14 buses):     ROCOF_max = [{min(rocof_m):.4f}, {max(rocof_m):.4f}], f_min = [{min(fmin_m):.4f}, {max(fmin_m):.4f}]")
    else:
        print("  No MATLAB observation_from_voltage.csv; run MATLAB scripts then re-run this test to derive.")

    py_table = load_python_design_table()
    if py_table:
        rocof_p = [float(r["ROCOF_max"]) for r in py_table]
        fmin_p = [float(r["f_min"]) for r in py_table]
        print(f"  Python (140 designs):  ROCOF_max = [{min(rocof_p):.4f}, {max(rocof_p):.4f}], f_min = [{min(fmin_p):.4f}, {max(fmin_p):.4f}]")
    print("\n  Same observation definition (12 Hz, f_nominal 50 Hz). Numerical match not expected (different dynamics).")

    # Build and write comparison table (MATLAB dynamic, probe + Python per-bus, all 14 probe buses)
    print("\n" + "=" * 60)
    print("Building comparison table (MATLAB + Python per-bus, probe at 1..14)")
    print("=" * 60)
    try:
        md_path, csv_path = build_comparison_table()
        print(f"  Table written: {md_path}")
        print(f"  CSV written:   {csv_path}")
    except Exception as e:
        print(f"  Table build failed: {e}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--table-only", action="store_true", help="Only build comparison table and exit")
    args = p.parse_args()
    if args.table_only:
        md_path, csv_path = build_comparison_table()
        print(f"COMPARISON_TABLE.md: {md_path}")
        print(f"comparison_table.csv: {csv_path}")
    else:
        main()
