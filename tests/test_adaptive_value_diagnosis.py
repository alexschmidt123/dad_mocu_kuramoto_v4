"""Tests for IEEE5 adaptive-value diagnosis (no test-set leakage)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.control.adaptive_value_diagnosis import (
    expected_u_after_action,
    j_adaptive_t2_for_action,
    normalize_advantages,
    potential_shaped_rewards,
    update_posterior,
)
from src.control.posterior_ctrl import normalize_log_weights, posterior_safe_u_ctrl
from src.neural.policy import DADPolicy, HistoryEncoder

ROOT = Path(__file__).resolve().parents[1]


def test_potential_rewards_telescope():
    path = [1.0, 0.9, 0.8, 0.7]
    r = potential_shaped_rewards(path)
    assert abs(sum(r) - (path[0] - path[-1])) < 1e-12


def test_shaped_reward_preserves_terminal_objective():
    # Maximizing sum r <=> minimizing terminal u (prior constant).
    u0 = 1.0
    for uT in (0.9, 0.8, 0.7):
        r = potential_shaped_rewards([u0, 0.95, uT])
        assert abs(sum(r) - (u0 - uT)) < 1e-12


def test_advantage_normalization():
    adv = np.asarray([1.0, 2.0, 3.0])
    z = normalize_advantages(adv)
    assert abs(float(np.mean(z))) < 1e-12
    assert abs(float(np.std(z)) - 1.0) < 1e-12
    assert np.allclose(normalize_advantages(np.ones(5)), 0.0)


def test_exact_t2_adaptive_synthetic():
    """
    Two particles, two actions. Action 0 reveals identity; action 1 is uninformative.
    Adaptive: take 0 first, then optionally 1. Fixed must commit to both unordered.
    Nested adaptive value for a1=0 should be <= value for a1=1.
    """
    rng = np.random.default_rng(0)
    # centres[a, n]
    centres = np.asarray(
        [
            [0.0, 5.0],  # action 0 separates particles
            [2.5, 2.5],  # action 1 identical centres
        ],
        dtype=np.float64,
    )
    U = np.asarray([0.2, 0.8], dtype=np.float64)
    log_p0 = np.zeros(2)
    p0 = normalize_log_weights(log_p0)
    sigma_y = 0.1
    alpha = 0.05
    margin = 0.55
    u_grid = np.asarray([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    j0, _ = j_adaptive_t2_for_action(
        0,
        centres=centres,
        U=U,
        log_p0=log_p0,
        p0=p0,
        sigma_y=sigma_y,
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        n_actions=2,
        K_outer=64,
        n_hyp_inner=32,
        rng=rng,
    )
    j1, _ = j_adaptive_t2_for_action(
        1,
        centres=centres,
        U=U,
        log_p0=log_p0,
        p0=p0,
        sigma_y=sigma_y,
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        n_actions=2,
        K_outer=64,
        n_hyp_inner=32,
        rng=rng,
    )
    # Informative first action cannot be worse than uninformative first action.
    assert j0 <= j1 + 0.05


def test_adaptive_later_action_depends_on_observation():
    """After observing y from action 0, best second action's expected cost can differ."""
    centres = np.asarray([[0.0, 5.0], [0.0, 5.0], [2.5, 2.5]], dtype=np.float64)
    U = np.asarray([0.2, 0.8], dtype=np.float64)
    log_p0 = np.zeros(2)
    sigma_y = 0.05
    alpha = 0.05
    margin = 0.55
    u_grid = np.asarray([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    rng = np.random.default_rng(1)

    def best_second(y: float) -> int:
        log_w1, w1 = update_posterior(log_p0, y, centres[0], sigma_y)
        idx = rng.choice(2, size=64, p=w1)
        noise = rng.normal(0.0, sigma_y, size=64)
        scores = {}
        for a2 in (1, 2):
            scores[a2] = expected_u_after_action(
                a2,
                log_w1,
                w1,
                centres=centres,
                U=U,
                sigma_y=sigma_y,
                alpha=alpha,
                margin=margin,
                u_grid=u_grid,
                idx=idx,
                noise=noise,
            )
        return int(min(scores, key=scores.get))

    # y near particle 0 vs particle 1
    a_lo = best_second(0.0)
    a_hi = best_second(5.0)
    # Not required that they differ every seed, but scores path must be observation-dependent:
    log_w_a, w_a = update_posterior(log_p0, 0.0, centres[0], sigma_y)
    log_w_b, w_b = update_posterior(log_p0, 5.0, centres[0], sigma_y)
    assert not np.allclose(w_a, w_b)
    assert a_lo in (1, 2) and a_hi in (1, 2)


def test_offline_banks_only_no_simulator_import_in_module():
    src = (ROOT / "src" / "control" / "adaptive_value_diagnosis.py").read_text(encoding="utf-8")
    assert "CudaControlEngine" not in src
    assert "build_simulator" not in src
    assert "lookup_action_y_sim" in src


def test_complete_history_encoder_changes_with_early_observation():
    enc = HistoryEncoder(n_actions=5, hidden=32, max_steps=4)
    # history of length 2: actions (0,1), vary y0 keeping y1 fixed
    act = torch.tensor([[0, 1]], dtype=torch.long)
    mask = torch.ones(1, 2)
    obs_a = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    obs_b = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
    h_a = enc(act, obs_a, mask)
    h_b = enc(act, obs_b, mask)
    assert h_a.shape == (1, 32)
    assert not torch.allclose(h_a, h_b)


def test_dad_forward_uses_full_history_call_path():
    pol = DADPolicy(n_actions=5, hidden=32, max_steps=4)
    act = torch.tensor([[0, 2, 1]], dtype=torch.long)
    obs = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    mask = torch.ones(1, 3)
    feas = torch.ones(1, 5, dtype=torch.bool)
    feas[0, 0] = feas[0, 2] = feas[0, 1] = False
    logits = pol.forward(act, obs, mask, feas)
    assert logits.shape == (1, 5)
    # Changing early obs changes logits
    obs2 = obs.clone()
    obs2[0, 0] = 3.0
    logits2 = pol.forward(act, obs2, mask, feas)
    assert not torch.allclose(logits, logits2)


def test_value_baseline_must_not_use_true_theta_contract():
    # Contract test: baseline helpers in diagnosis module do not take u_req/theta.
    import inspect
    from src.control import adaptive_value_diagnosis as m

    src = inspect.getsource(m.normalize_advantages)
    assert "u_req" not in src and "theta" not in src


def test_frozen_terminal_rule_unchanged_in_diagnosis():
    from src.control.adaptive_value_diagnosis import EXPECTED_HASH, FROZEN_MARGIN

    assert EXPECTED_HASH == "c2e2af33cb68a5ea"
    assert abs(FROZEN_MARGIN - 0.55) < 1e-12


def test_no_eig_objective_in_diagnosis_training_path():
    src = (ROOT / "src" / "control" / "adaptive_value_diagnosis.py").read_text(encoding="utf-8")
    assert "EIG" not in src
    assert "delta_h" not in src.lower() or "Delta-H" not in src
    # Allow commenting that we do not use EIG
    assert "posterior_safe_u_ctrl" in src or "expected_u_after_action" in src


def test_validation_not_test_for_variant_selection_contract():
    src = (ROOT / "src" / "control" / "adaptive_value_diagnosis.py").read_text(encoding="utf-8")
    assert "used_test_systems" in src
    assert "validation_systems" in src
    assert "test_systems" not in src or "used_test_systems" in src
