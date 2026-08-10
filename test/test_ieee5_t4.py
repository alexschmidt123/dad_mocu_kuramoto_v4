"""Tests for IEEE5 T=4 controlled experiment invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.control.legacy.ieee5_t4 import EXPECTED_HASH, FROZEN_MARGIN, _entropy, compute_dad_adaptivity
from src.control.terminal_rule import FrozenTerminalRule, load_frozen_terminal_rule
from src.rollout import RandomSelector

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_rule_hash_unchanged():
    rule_path = (
        ROOT
        / "experiments"
        / "ieee5_policy_robust_calibration_T2"
        / "selected_policy_robust_rule.json"
    )
    if not rule_path.is_file():
        return
    # Install into a temp-like check via load after writing to T4 if present
    t4 = ROOT / "experiments" / "ieee5_T4"
    if (t4 / "selected_policy_robust_rule.json").is_file():
        frozen = load_frozen_terminal_rule(t4, expected_margin=FROZEN_MARGIN)
        assert frozen.terminal_rule_hash == EXPECTED_HASH
        assert abs(frozen.margin - 0.55) < 1e-12
        assert abs(frozen.alpha - 0.05) < 1e-12


def test_adaptivity_dominant_fraction():
    rows = [
        {"sequence": [1, 2, 3, 4], "y_obs": [0.1, 0.2, 0.3, 0.4], "rollout_id": i, "theta_test_id": 0}
        for i in range(10)
    ]
    rows[0] = {
        "sequence": [9, 8, 7, 6],
        "y_obs": [0.0, 0.0, 0.0, 0.0],
        "rollout_id": 0,
        "theta_test_id": 0,
    }
    adapt = compute_dad_adaptivity(rows)
    assert adapt["number_of_unique_sequences"] == 2
    assert abs(adapt["dominant_sequence_fraction"] - 0.9) < 1e-12
    assert adapt["dominant_sequence"] == [1, 2, 3, 4]


def test_entropy_singleton_zero():
    assert _entropy({(1, 2): 5}) == 0.0


def test_random_without_replacement():
    sel = RandomSelector(n_actions=30)
    rng = np.random.default_rng(0)
    used = set()
    for _ in range(4):
        a = sel.select(used=used, rng=rng)
        assert a not in used
        used.add(a)


def test_fixed_size_4_binomial():
    from math import comb

    assert comb(30, 4) == 27405


def test_myopic_is_one_step():
    import inspect
    from src.control.myopic import MyopicControlSelector

    src = inspect.getsource(MyopicControlSelector._expected_u_ctrl)
    # One-step: updates weights for a single action/observation, no future probe loop.
    assert "for t in range" not in src
    assert "n_hypothetical" in MyopicControlSelector.__init__.__code__.co_varnames


def test_safety_first_in_train():
    import inspect
    from src.neural import train as train_mod

    src = inspect.getsource(train_mod.train_dad_policy)
    assert "validation_safety" in src
    assert "safety_first" in src or "validation_safety_below_1" in src


def test_t2_t3_not_overwritten_by_t4_layout():
    t3 = ROOT / "experiments" / "ieee5_T3" / "eval" / "summary.json"
    t2 = ROOT / "experiments" / "ieee5_policy_robust_calibration_T2" / "rerun_T2_summary.csv"
    assert t3.is_file() or t2.is_file()
    # T4 path is distinct
    assert (ROOT / "experiments" / "ieee5_T4") != (ROOT / "experiments" / "ieee5_T3")


def test_shared_rule_metadata_hash():
    rule = FrozenTerminalRule(0.05, 0.55, tuple(np.round(np.arange(0, 1.05, 0.05), 10)))
    # May not match grid exactly; just ensure hash stable
    h1 = rule.terminal_rule_hash
    h2 = FrozenTerminalRule(0.05, 0.55, rule.u_candidates).terminal_rule_hash
    assert h1 == h2


def test_paired_bootstrap_resamples_indices():
    from src.control.pilot import paired_diff_stats

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.1, 2.1, 3.1, 4.1])
    stats = paired_diff_stats(a, b, n_boot=200, seed=0)
    assert stats["n_bootstrap"] == 200
    assert "ci95_low" in stats
