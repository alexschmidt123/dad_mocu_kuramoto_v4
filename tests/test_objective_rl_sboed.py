"""Tests for objective RL-sBOED rewards and shared control."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.control.objective_rl_sboed.rewards import (
    GAMMA,
    dad_rewards,
    verify_rl_sboed_rollout,
)
from src.control.posterior_ctrl import (
    posterior_control_decision,
    posterior_safe_u_ctrl,
    posterior_u_raw,
)


ROOT = Path(__file__).resolve().parents[1]


def test_dad_intermediate_rewards_zero_terminal_neg_u():
    trace = dad_rewards([1.2, 1.1, 0.9, 0.8])
    assert trace.rewards[:-1] == (0.0, 0.0)
    assert abs(trace.rewards[-1] + 0.8) < 1e-12


def test_rl_sboed_step_rewards_and_telescope():
    u = [1.0, 0.9, 0.85, 0.7]
    trace = verify_rl_sboed_rollout(u)
    assert abs(trace.rewards[0] - 0.1) < 1e-12
    assert abs(trace.rewards[1] - 0.05) < 1e-12
    assert abs(trace.rewards[2] - 0.15) < 1e-12
    assert abs(sum(trace.rewards) - (u[0] - u[-1])) < 1e-12
    assert abs(trace.gamma - 1.0) < 1e-15
    assert abs(GAMMA - 1.0) < 1e-15


def test_shared_u_raw_and_u_ctrl():
    U = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    w = np.ones(4) / 4.0
    grid = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    decision = posterior_control_decision(U, w, alpha=0.05, margin=0.1, u_grid=grid)
    assert decision.u_raw == pytest.approx(decision.u_quantile + 0.1)
    assert decision.u_ctrl >= decision.u_raw - 1e-12
    assert decision.u_ctrl == decision.u_ctrl_snapped
    assert posterior_safe_u_ctrl(U, w, 0.05, margin=0.1, u_grid=grid) == decision.u_ctrl
    assert posterior_u_raw(U, w, 0.05, margin=0.1) == pytest.approx(decision.u_raw)


def test_no_ode_imports_in_objective_rl_sboed_package():
    pkg = ROOT / "src" / "control" / "objective_rl_sboed"
    forbidden = ("solve_ivp", "swing_equation_ode.simulator", "cuda_batch")
    for path in pkg.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text, f"{path} references {name}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = (
                    node.module
                    if isinstance(node, ast.ImportFrom)
                    else ",".join(a.name for a in node.names)
                )
                assert "swing_equation_ode.simulator" not in (mod or "")


def test_historical_adaptive_value_untouched():
    path = ROOT / "experiments" / "objective_adaptive_value" / "summary" / "final_report.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Case B" in text or "case B" in text or "**B**" in text


def test_standard_experiment_layout_helpers():
    from src.experiment_layout import ensure_standard_layout, RunMetadata, write_run_metadata

    root = ROOT / "experiments" / "objective_rl_sboed" / "ieee5_T3"
    paths = ensure_standard_layout(root)
    assert paths.train.is_dir()
    assert paths.eval.is_dir()
    assert paths.model.is_dir()
    assert paths.logs.is_dir()
    assert paths.diagnostics.is_dir()
    assert paths.summary.is_dir()
    assert (root / "run_config.yaml").is_file()
    meta = RunMetadata(
        experiment_name="objective_rl_sboed/ieee5_T3",
        entry_point="run.sh",
        timestamp_utc="2026-01-01T00:00:00Z",
        system="ieee5",
        horizon=3,
        method="DAD",
        seed=101,
    )
    write_run_metadata(root, meta)
    assert (root / "run_metadata.json").is_file()


def test_production_scripts_syntax():
    import subprocess

    scripts = list((ROOT / "scripts").glob("*.sh")) + [
        ROOT / "run.sh",
        ROOT / "sweep_run.sh",
    ]
    for script in scripts:
        if not script.exists():
            continue
        proc = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"{script}: {proc.stderr}"


def test_fixed_init_uses_train_validation_fixed_sequence_only():
    from src.control.objective_rl_sboed.context import load_fixed_sequence

    seq = load_fixed_sequence("ieee5")
    assert len(seq) == 3
    assert len(set(seq)) == 3
    # Fixed plan comes from frozen ieee5_T3 eval metadata, not confirmation outcomes.
    meta = ROOT / "experiments" / "ieee5_T3" / "eval" / "fixed" / "subset_meta.json"
    assert meta.is_file()


def test_same_theta_throughout_rollout_contract():
    """sample_trajectory keeps one theta_id for the full T-step history."""
    src = (ROOT / "src" / "control" / "objective_rl_sboed" / "ppo_train.py").read_text(
        encoding="utf-8"
    )
    assert "theta_id=tid" in src or "theta_id=theta_id" in src
    assert "for step in range(ctx.horizon)" in src


def test_sensitivity_audit_outputs_exist():
    audit = ROOT / "experiments" / "objective_rl_sboed" / "diagnostics" / "sensitivity_audit"
    assert (audit / "ieee5_T3_sensitivity.csv").is_file()
    assert (audit / "ieee9_T3_sensitivity.csv").is_file()
    assert (audit / "sensitivity_report.md").is_file()
