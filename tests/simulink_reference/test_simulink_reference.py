"""
**Experiment:** Python swing ODE vs Simulink/MATLAB reference (IEEE-14, probe); optional ``matlab/results`` CSVs.

Helpers: ``tests/simulink_reference/ode_validation.py``. Run: ``python -m tests.simulink_reference.ode_validation``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.simulink_reference.ode_validation import (
    MATLAB_DYNAMIC,
    OUTPUT_DIR,
    derive_observation_from_matlab_folder,
    load_observation_from_voltage_csv,
    load_python_design_table,
    run_ieee14_ode_probe_bus1,
)


def test_python_ode_probe_bus1_runs():
    obs = run_ieee14_ode_probe_bus1(amplitude=0.2, duration=2.0, T=5.0, device="cpu")
    assert "ROCOF_max" in obs and "f_min" in obs
    assert obs["ROCOF_max"] > 0
    assert 49.0 < obs["f_min"] < 50.0
    assert obs["ROCOF_max"] > 1.0


def test_python_ode_probe_bus1_save_to_output():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    obs = run_ieee14_ode_probe_bus1(amplitude=0.2, duration=2.0, T=5.0, device="cpu")
    out_file = OUTPUT_DIR / "ieee14_probe_bus1_observation.txt"
    with open(out_file, "w") as f:
        f.write(f"ROCOF_max={obs['ROCOF_max']}\n")
        f.write(f"f_min={obs['f_min']}\n")
    assert out_file.exists()


@pytest.mark.skipif(not MATLAB_DYNAMIC.exists(), reason="MATLAB results not present")
def test_derive_matlab_observation_from_voltage():
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
    obs_py = run_ieee14_ode_probe_bus1(amplitude=0.2, duration=2.0, T=5.0, device="cpu")
    matlab_obs_path = MATLAB_DYNAMIC / "observation_from_voltage.csv"
    matlab_rows = load_observation_from_voltage_csv(matlab_obs_path) if matlab_obs_path.exists() else None
    if matlab_rows:
        rocof_matlab = [float(r["ROCOF_max"]) for r in matlab_rows]
        assert obs_py["ROCOF_max"] > max(rocof_matlab) or max(rocof_matlab) < 0.5
    else:
        assert obs_py["ROCOF_max"] > 1.0
        assert obs_py["f_min"] < 50.0


def test_compare_design_table_with_matlab_if_present():
    py_rows = load_python_design_table()
    matlab_obs = load_observation_from_voltage_csv(MATLAB_DYNAMIC / "observation_from_voltage.csv")
    if not py_rows:
        pytest.skip(
            "design_comparison_table.csv not found (run pytest tests/posterior_inference/ -k experiment_design -v first)"
        )
    if not matlab_obs:
        pytest.skip("MATLAB observation_from_voltage.csv not found")
    rocof_py = [float(r["ROCOF_max"]) for r in py_rows]
    fmin_py = [float(r["f_min"]) for r in py_rows]
    rocof_matlab = [float(r["ROCOF_max"]) for r in matlab_obs]
    fmin_matlab = [float(r["f_min"]) for r in matlab_obs]
    assert len(rocof_py) == 140
    assert min(rocof_py) >= 0 and max(fmin_py) <= 51
    assert max(rocof_py) >= max(rocof_matlab) - 0.1
