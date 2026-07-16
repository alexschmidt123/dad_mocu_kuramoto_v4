"""Tests for frozen Myopic selection and IEEE5 sweep invariants."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.control.ieee5_horizon_sweep import check_t1_myopic_fixed_equivalence
from src.control.terminal_rule import keyed_noise, load_frozen_terminal_rule


ROOT = Path(__file__).resolve().parents[1]


def test_myopic_n_h_selected_on_validation_only():
    report = ROOT / "experiments/ieee5_horizon_sweep/myopic_convergence/convergence_report.json"
    assert report.is_file(), "run select-myopic-n-hypothetical first"
    import json

    data = json.loads(report.read_text())
    assert data["used_test_systems"] is False
    assert data["selection_source"] == "validation_convergence"
    assert int(data["selected_n_hypothetical"]) == 1024  # no smaller count passed thresholds


def test_ieee5_config_frozen_myopic_count():
    cfg = yaml.safe_load((ROOT / "config/ieee5_config.yaml").read_text())
    assert int(cfg["myopic"]["n_hypothetical"]) == int(cfg["control"]["myopic_hypothetical"])
    assert cfg["myopic"]["selection_source"] == "validation_convergence"


def test_terminal_rule_hash_unchanged():
    exp = ROOT / "experiments/07132026_220727_ieee5_T2"
    rule = load_frozen_terminal_rule(exp)
    assert rule.terminal_rule_hash == "dc0dc35332b394b7"
    assert abs(rule.alpha - 0.05) < 1e-12
    assert abs(rule.margin - 0.40) < 1e-12


def test_keyed_noise_deterministic():
    a = keyed_noise(global_seed=1, theta_id=2, rollout_id=3, step=0, action_id=4)
    b = keyed_noise(global_seed=1, theta_id=2, rollout_id=3, step=0, action_id=4)
    assert a == b


def test_t1_equivalence_helper():
    report = {
        "summaries": {
            "myopic": {"mean_u_ctrl": 0.80},
            "fixed": {"mean_u_ctrl": 0.80},
        },
        "paired_differences": {
            "myopic_minus_fixed": {"ci95_low": -0.01, "ci95_high": 0.01}
        },
    }
    assert check_t1_myopic_fixed_equivalence(report)["passed"]


def test_no_eig_in_horizon_sweep_module():
    text = (ROOT / "src/control/ieee5_horizon_sweep.py").read_text().lower()
    assert "eig_score" not in text
    assert "delta_h" not in text
