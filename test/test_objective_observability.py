"""Tests for objective-observability gate (no named-method training)."""

from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path

import numpy as np

from src.control.observability import (
    ObservabilityGateConfig,
    evaluate_gate,
    run_diagnostic_rollout,
    spearman_corr,
)
from src.control.posterior_ctrl import normalize_log_weights, posterior_safe_u_ctrl
from src.rollout import RandomSelector, update_log_weights
from src.table_scoring import TableThetaSupport


class _FakeSupport:
    def __init__(self, n: int, y_sim_by_action: dict[int, np.ndarray]):
        self.systems = [{"u_req": float(i)} for i in range(n)]
        self.log_p0 = np.zeros(n, dtype=np.float64)
        self._y = y_sim_by_action

    def __len__(self) -> int:
        return len(self.systems)


def test_prior_terminal_control_uniform_weights():
    U = np.array([0.1, 0.2, 0.4, 0.8])
    w = np.ones(4) / 4.0
    # 1-α = 0.95 → need cumulative ≥ 0.95 → last value 0.8
    assert abs(posterior_safe_u_ctrl(U, w, alpha=0.05) - 0.8) < 1e-12
    # α=0.5 → q=0.5 → second mass point at cum=0.5 → 0.2
    assert abs(posterior_safe_u_ctrl(U, w, alpha=0.5) - 0.2) < 1e-12


def test_posterior_uses_all_probe_observation_pairs(monkeypatch):
    """Likelihood increments must accumulate over every (action, y) pair."""
    n = 4
    U = np.array([0.1, 0.2, 0.3, 0.9])
    # Distinct centres so each observation moves weights.
    centres = {
        0: np.array([0.0, 1.0, 2.0, 3.0]),
        1: np.array([0.0, 0.5, 1.0, 1.5]),
    }

    class Support(TableThetaSupport):
        pass

    support = Support(systems=[{"u_req": float(u)} for u in U], log_p0=np.zeros(n))

    def fake_y_sim(table_support, sequence):
        a = int(sequence[-1])
        return centres[a]

    monkeypatch.setattr(
        "src.control.observability.y_sim_last_step_from_tables", fake_y_sim
    )
    monkeypatch.setattr(
        "src.control.observability.lookup_action_y",
        lambda system, a: {0: 0.0, 1: 0.5}[int(a)],
    )

    system = {"u_req": 0.2}
    out = run_diagnostic_rollout(
        system=system,
        table_support=support,
        U_support=U,
        horizon=2,
        n_actions=4,
        sigma_y=0.5,
        alpha=0.05,
        rng=np.random.default_rng(0),
        update_posterior=True,
        forced_sequence=[0, 1],
        forced_observations=[0.0, 0.5],
    )
    # Manual two-step update
    log_w = np.zeros(n)
    log_w = update_log_weights(log_w, 0.0, centres[0], 0.5)
    log_w = update_log_weights(log_w, 0.5, centres[1], 0.5)
    w = normalize_log_weights(log_w)
    assert np.allclose(out["weights"], w)
    assert abs(out["u_ctrl_final"] - posterior_safe_u_ctrl(U, w, 0.05)) < 1e-12


def test_no_update_retains_prior_control(monkeypatch):
    U = np.array([0.05, 0.2, 0.4, 0.7])
    support = TableThetaSupport(
        systems=[{"u_req": float(u)} for u in U], log_p0=np.zeros(4)
    )
    monkeypatch.setattr(
        "src.control.observability.y_sim_last_step_from_tables",
        lambda *_a, **_k: np.zeros(4),
    )
    monkeypatch.setattr(
        "src.control.observability.lookup_action_y", lambda *_a, **_k: 1.0
    )
    out = run_diagnostic_rollout(
        system={"u_req": 0.2},
        table_support=support,
        U_support=U,
        horizon=3,
        n_actions=5,
        sigma_y=0.1,
        alpha=0.05,
        rng=np.random.default_rng(1),
        update_posterior=False,
        forced_sequence=[0, 2, 4],
        forced_observations=[0.1, 0.2, 0.3],
    )
    assert abs(out["u_ctrl_final"] - out["u_ctrl_prior"]) < 1e-12
    assert all(abs(u - out["u_ctrl_prior"]) < 1e-12 for u in out["u_ctrl_path"])


def test_diagnostic_probes_without_replacement():
    sel = RandomSelector(n_actions=6)
    rng = np.random.default_rng(42)
    used: set[int] = set()
    for _ in range(6):
        a = sel.select(used=used, rng=rng)
        assert a not in used
        used.add(a)
    assert used == set(range(6))


def test_constant_terminal_controls_fail_gate():
    summary = {
        "unique_final_u_ctrl_count": 1,
        "final_u_ctrl_std": 0.0,
        "fraction_changed_from_prior": 0.0,
        "true_safety_rate": 1.0,
        "real_spearman": 0.5,
        "shuffled_spearman": 0.1,
    }
    gate = ObservabilityGateConfig()
    result = evaluate_gate(summary, gate)
    assert not result["passed"]
    assert "unique_final_u_ctrl_count" in result["failed_checks"]
    assert "final_u_ctrl_std" in result["failed_checks"]
    assert "fraction_changed_from_prior" in result["failed_checks"]


def test_synthetic_informative_case_passes_gate():
    summary = {
        "unique_final_u_ctrl_count": 5,
        "final_u_ctrl_std": 0.12,
        "fraction_changed_from_prior": 0.4,
        "true_safety_rate": 1.0,
        "real_spearman": 0.55,
        "shuffled_spearman": 0.05,
    }
    result = evaluate_gate(summary, ObservabilityGateConfig())
    assert result["passed"]
    assert result["failed_checks"] == []


def test_safety_below_one_fails_gate():
    summary = {
        "unique_final_u_ctrl_count": 5,
        "final_u_ctrl_std": 0.12,
        "fraction_changed_from_prior": 0.4,
        "true_safety_rate": 0.99,
        "real_spearman": 0.55,
        "shuffled_spearman": 0.05,
    }
    result = evaluate_gate(summary, ObservabilityGateConfig())
    assert not result["passed"]
    assert "true_safety_rate" in result["failed_checks"]


def test_run_sh_blocks_training_after_gate_failure(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    run_sh = (tmp_path / "run.sh").resolve()
    # Minimal stub of run.sh phase order with a failing gate.
    gate = tmp_path / "gate.sh"
    train = tmp_path / "train.sh"
    train.write_text("#!/bin/bash\necho TRAINED > \"%s\"\n" % (tmp_path / "trained.flag"), encoding="utf-8")
    gate.write_text("#!/bin/bash\necho GATE_FAIL\nexit 1\n", encoding="utf-8")
    run_sh.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            set -euo pipefail
            ./gate.sh
            ./train.sh
            """
        ),
        encoding="utf-8",
    )
    # Actually test the real run.sh contains the gate before training, and a
    # failing observability exit stops training via set -e.
    real = (root / "run.sh").read_text(encoding="utf-8")
    assert "objective_observability.sh" in real
    assert real.index("objective_observability.sh") < real.index("dad_training.sh")

    os.chmod(gate, os.stat(gate).st_mode | stat.S_IEXEC)
    os.chmod(train, os.stat(train).st_mode | stat.S_IEXEC)
    os.chmod(run_sh, os.stat(run_sh).st_mode | stat.S_IEXEC)
    # Run stub from tmp_path
    proc = subprocess.run(["bash", str(run_sh)], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode != 0
    assert not (tmp_path / "trained.flag").exists()


def test_run_sh_proceeds_after_gate_success(tmp_path: Path):
    gate = tmp_path / "gate.sh"
    train = tmp_path / "train.sh"
    run_sh = tmp_path / "run.sh"
    gate.write_text("#!/bin/bash\necho GATE_OK\nexit 0\n", encoding="utf-8")
    train.write_text(
        f"#!/bin/bash\necho TRAINED > '{tmp_path / 'trained.flag'}'\n",
        encoding="utf-8",
    )
    run_sh.write_text(
        "#!/bin/bash\nset -euo pipefail\n./gate.sh\n./train.sh\n",
        encoding="utf-8",
    )
    for p in (gate, train, run_sh):
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    proc = subprocess.run(["bash", str(run_sh)], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0
    assert (tmp_path / "trained.flag").is_file()


def test_sweep_run_preserves_file_lock():
    root = Path(__file__).resolve().parents[1]
    text = (root / "sweep_run.sh").read_text(encoding="utf-8")
    assert ".sweep.lock" in text
    assert "flock -n 200" in text
    assert "./run.sh" in text
    # Gate must not be duplicated in sweep — only via run.sh
    assert "objective_observability" not in text


def test_diagnostic_does_not_train_or_evaluate_named_methods():
    root = Path(__file__).resolve().parents[1]
    src = (root / "src" / "control" / "observability.py").read_text(encoding="utf-8")
    assert "train_dad_policy" not in src
    assert "run_myopic" not in src
    assert "run_fixed" not in src
    assert "run_random" not in src
    assert "run_dad" not in src
    # Explicitly documents empty method lists in summary path via string markers
    assert "methods_trained" in src
    assert "methods_evaluated" in src


def test_spearman_basic():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    assert abs(spearman_corr(x, y) - 1.0) < 1e-9
