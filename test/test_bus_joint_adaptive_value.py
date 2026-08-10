"""Tests for bus + joint adaptive-value diagnostic."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_no_ode_in_bus_joint_package():
    pkg = ROOT / "src" / "control" / "bus_joint_adaptive_value"
    forbidden = ("solve_ivp", "swing_equation_ode.simulator", "cuda_batch")
    for path in pkg.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = (
                    node.module
                    if isinstance(node, ast.ImportFrom)
                    else ",".join(a.name for a in node.names)
                )
                assert "swing_equation_ode.simulator" not in (mod or "")


def test_duration_fixed_and_six_amplitudes():
    from src.run_context import load_experiment_run
    from src.swing_equation_ode.design import build_catalog

    for system, n_bus, n_des in (("ieee5", 5, 30), ("ieee9", 9, 54)):
        run = load_experiment_run(ROOT / "experiments" / f"{system}_T3", ROOT)
        cat = build_catalog(run.cfg)
        assert abs(float(run.cfg.probe_duration) - 0.2) < 1e-12
        amps = sorted({float(d.amplitude) for d in cat})
        assert len(amps) == 6
        assert len({int(d.bus) for d in cat}) == n_bus
        assert len(cat) == n_des


def test_decomposition_mode_invariants_in_source():
    src = (
        ROOT / "src" / "control" / "bus_joint_adaptive_value" / "diagnostic.py"
    ).read_text(encoding="utf-8")
    assert "fixed_bus_adaptive_amp" in src
    assert "adaptive_bus_fixed_amp" in src
    assert "adaptive_bus_adaptive_amp" in src
    assert "fully_fixed" in src
    assert "used_confirmation_split\": False" in src or "used_confirmation_split\": False" in src.replace(
        " ", ""
    )


def test_run_sh_dispatches_bus_study():
    text = (ROOT / "run.sh").read_text(encoding="utf-8")
    assert "bus_joint_adaptive_value" in text
    assert (ROOT / "scripts" / "bus_joint_adaptive_value.sh").is_file()


def test_historical_studies_untouched():
    amp = (
        ROOT
        / "experiments"
        / "continuous_uctrl_amplitude_adaptive_value"
        / "summary"
        / "final_report.md"
    )
    av = ROOT / "experiments" / "objective_adaptive_value" / "summary" / "final_report.md"
    assert amp.is_file()
    assert av.is_file()
    assert "Case B" in amp.read_text(encoding="utf-8") or "nominal" in amp.read_text(
        encoding="utf-8"
    )
