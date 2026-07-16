"""Tests for frozen terminal rule and pilot invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.control.posterior_ctrl import normalize_log_weights
from src.control.terminal_rule import (
    FrozenTerminalRule,
    assert_shared_rule_metadata,
    keyed_noise,
    load_frozen_terminal_rule,
    posterior_to_u_ctrl,
)
from src.neural.policy import DADPolicy
from src.rollout import RandomSelector


def test_frozen_rule_shared_hash():
    rule = FrozenTerminalRule(0.05, 0.4, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    meta = {
        "dad": rule.metadata(),
        "myopic": rule.metadata(),
        "fixed": rule.metadata(),
        "random": rule.metadata(),
    }
    shared = assert_shared_rule_metadata(meta)
    assert shared["quantile_level"] == 0.95
    assert abs(shared["additive_margin"] - 0.4) < 1e-12


def test_posterior_to_u_ctrl_snap_up():
    rule = FrozenTerminalRule(0.05, 0.4, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    U = np.array([0.1, 0.2, 0.3, 0.5])
    w = np.ones(4) / 4
    u = posterior_to_u_ctrl(w, U, rule)
    # Q_0.95 = 0.5, +0.4 = 0.9 → snap to 1.0
    assert abs(u - 1.0) < 1e-12


def test_load_frozen_rule_from_exp():
    root = Path(__file__).resolve().parents[1]
    exp = root / "experiments" / "07132026_220727_ieee5_T2"
    rule = load_frozen_terminal_rule(exp)
    assert abs(rule.alpha - 0.05) < 1e-12
    assert abs(rule.margin - 0.40) < 1e-12


def test_dad_complete_history_encoder_differs():
    policy = DADPolicy(n_actions=8, hidden=32, max_steps=4)
    policy.eval()
    with torch.no_grad():
        h_a = policy.encoder(
            torch.tensor([[0, 1]]), torch.tensor([[1.0, 2.0]]), torch.ones(1, 2)
        )
        h_b = policy.encoder(
            torch.tensor([[3, 1]]), torch.tensor([[9.0, 2.0]]), torch.ones(1, 2)
        )
    assert not torch.allclose(h_a, h_b, atol=1e-5)


def test_reinforce_sign_lower_cost_positive_advantage():
    baseline = 0.5
    cost_low = 0.2
    cost_high = 0.8
    adv_low = baseline - cost_low
    adv_high = baseline - cost_high
    assert adv_low > 0
    assert adv_high < 0
    # L = -adv * log_prob  ⇒  ∂L/∂log_prob = -adv
    # Low cost (adv>0): gradient negative → increase log-prob (more likely).
    # High cost (adv<0): gradient positive → decrease log-prob (less likely).
    assert -adv_low < 0
    assert -adv_high > 0


def test_random_without_replacement():
    sel = RandomSelector(n_actions=10)
    rng = np.random.default_rng(0)
    used = set()
    for _ in range(10):
        a = sel.select(used=used, rng=rng)
        assert a not in used
        used.add(a)


def test_keyed_noise_deterministic():
    a = keyed_noise(global_seed=1, theta_id=2, rollout_id=3, step=0, action_id=4)
    b = keyed_noise(global_seed=1, theta_id=2, rollout_id=3, step=0, action_id=4)
    c = keyed_noise(global_seed=1, theta_id=2, rollout_id=3, step=0, action_id=5)
    assert a == b
    assert a != c


def test_metadata_mismatch_raises():
    rule = FrozenTerminalRule(0.05, 0.4, (0.0, 1.0))
    bad = dict(rule.metadata())
    bad["additive_margin"] = 0.1
    try:
        assert_shared_rule_metadata({"dad": rule.metadata(), "myopic": bad})
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_no_eig_in_pilot_module():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "control" / "pilot.py").read_text(encoding="utf-8")
    banned = ["delta_h", "expected information gain", "eig_score", "recalibrat"]
    low = text.lower()
    for b in banned:
        assert b not in low, b
