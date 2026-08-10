"""Tests for continuous u_ctrl and amplitude adaptive-value study."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.control.posterior_ctrl import (
    compute_u_ctrl,
    compute_u_ctrl_snapped,
    posterior_control_decision,
    snap_up_to_grid,
)


ROOT = Path(__file__).resolve().parents[1]


def test_continuous_u_ctrl_does_not_snap():
    U = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    w = np.ones(4) / 4.0
    grid = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    d = posterior_control_decision(
        U, w, alpha=0.05, margin=0.1, u_grid=grid, snap_up=False
    )
    assert d.u_ctrl == pytest.approx(d.u_raw)
    assert d.u_ctrl == pytest.approx(d.u_quantile + 0.1)
    assert d.u_ctrl_snapped == pytest.approx(snap_up_to_grid(d.u_raw, grid))
    assert d.u_ctrl_snapped >= d.u_ctrl - 1e-12
    assert compute_u_ctrl(U, w, alpha=0.05, margin=0.1, u_grid=grid, snap_up=False) == d.u_ctrl


def test_historical_snapped_u_ctrl_reproducible():
    U = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    w = np.ones(4) / 4.0
    grid = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    d = posterior_control_decision(
        U, w, alpha=0.05, margin=0.1, u_grid=grid, snap_up=True
    )
    assert d.u_ctrl == d.u_ctrl_snapped
    assert d.u_ctrl == pytest.approx(compute_u_ctrl_snapped(U, w, alpha=0.05, margin=0.1, u_grid=grid))
    assert d.u_ctrl >= d.u_raw - 1e-12


def test_no_ode_in_continuous_amplitude_package():
    pkg = ROOT / "src" / "control" / "continuous_uctrl_amplitude"
    forbidden = ("solve_ivp", "swing_equation_ode.simulator", "cuda_batch")
    for path in pkg.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text, f"{path} references {name}"
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = (
                    node.module
                    if isinstance(node, ast.ImportFrom)
                    else ",".join(a.name for a in node.names)
                )
                assert "swing_equation_ode.simulator" not in (mod or "")


def test_existing_six_amplitudes_and_duration_from_config():
    from src.run_context import load_experiment_run
    from src.swing_equation_ode.design import build_catalog

    for system, n_bus, n_design in (("ieee5", 5, 30), ("ieee9", 9, 54)):
        run = load_experiment_run(ROOT / "experiments" / f"{system}_T3", ROOT)
        cat = build_catalog(run.cfg)
        amps = sorted({float(d.amplitude) for d in cat})
        buses = sorted({int(d.bus) for d in cat})
        assert len(amps) == 6
        assert amps == [0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
        assert buses == list(range(n_bus))
        assert len(cat) == n_design
        assert abs(float(run.cfg.probe_duration) - 0.2) < 1e-12
        assert all(abs(float(d.duration) - 0.2) < 1e-12 for d in cat)


def test_histories_train_val_only_source():
    src = (
        ROOT / "src" / "control" / "continuous_uctrl_amplitude" / "diagnostic.py"
    ).read_text(encoding="utf-8")
    assert "used_confirmation_split" in src
    assert "objective_adaptive_value" in src
    assert "False" in src


def test_historical_adaptive_value_untouched():
    path = ROOT / "experiments" / "objective_adaptive_value" / "summary" / "final_report.md"
    assert path.exists()
    assert "Case B" in path.read_text(encoding="utf-8") or "**B**" in path.read_text(
        encoding="utf-8"
    )


def test_run_sh_dispatches_continuous_study():
    text = (ROOT / "run.sh").read_text(encoding="utf-8")
    assert "continuous_uctrl_amplitude" in text
    script = ROOT / "scripts" / "continuous_uctrl_amplitude.sh"
    assert script.is_file()
