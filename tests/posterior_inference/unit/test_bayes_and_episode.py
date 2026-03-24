"""
Unit tests: ``src.core.discrete_bayes`` and episode helpers (``episode_helpers``).

Run: ``pytest tests/posterior_inference/unit/ -v``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.discrete_bayes import (
    log_gaussian_observation_density,
    log_prior_uniform_discrete,
    mocu_gamma_star,
    normalize_log_weights,
    posterior_after_sequential_gaussian_observations,
    sequential_posterior_from_log_likelihoods,
    single_step_discrete_bayes_report,
    weighted_median,
)
from tests.posterior_inference.episode_helpers import (
    DEFAULT_NUM_PROBE_STEPS,
    ODE_HORIZON_QUICK,
    QUICK_PHYSICS_GRID,
    QUICK_PHYSICS_TIMEOUT,
    main_body_single_step_simulation,
    run_physics_episode,
    run_single_step_physics_episode,
    run_synthetic_episode,
)


def test_normalize_log_weights_uniform():
    n = 5
    p = normalize_log_weights(np.zeros(n))
    assert np.allclose(p, 1.0 / n)
    assert abs(float(np.sum(p)) - 1.0) < 1e-10


def test_log_prior_uniform_matches_normalize():
    n = 7
    log_p0 = log_prior_uniform_discrete(n)
    p0 = normalize_log_weights(log_p0)
    assert np.allclose(p0, 1.0 / n)


def test_weighted_median_pseudocode():
    v = np.array([1.0, 2.0, 3.0])
    w = np.array([0.2, 0.3, 0.5])
    assert weighted_median(v, w) == 2.0


def test_mocu_gamma_star_uniform_three_points():
    g = np.array([10.0, 20.0, 30.0])
    p = np.ones(3) / 3.0
    m, gh = mocu_gamma_star(p, g)
    assert gh == 20.0
    assert abs(m - (10.0 / 3.0 + 10.0 / 3.0)) < 1e-9


def test_sequential_log_likelihoods_one_step():
    N = 4
    mu = np.array([0.0, 0.5, 1.0, 1.5])
    log_L = log_gaussian_observation_density(1.0, mu, 0.1)
    log_L_steps = log_L.reshape(1, N)
    _, p_trace = sequential_posterior_from_log_likelihoods(log_L_steps)
    assert len(p_trace) == 2
    assert np.allclose(np.sum(p_trace[-1]), 1.0)


def test_synthetic_episode_default_steps():
    r = run_synthetic_episode(num_probe_steps=DEFAULT_NUM_PROBE_STEPS, seed=0)
    assert r["num_probe_steps"] == DEFAULT_NUM_PROBE_STEPS
    assert len(r["mocu_trace"]) == DEFAULT_NUM_PROBE_STEPS + 1
    assert len(r["p_trace"]) == DEFAULT_NUM_PROBE_STEPS + 1
    assert r["grid"].shape[0] == r["n"]


def test_synthetic_single_step_is_special_case():
    """One probe--observe round: ``p_trace`` has prior + posterior (length 2)."""
    r = run_synthetic_episode(num_probe_steps=1, seed=0)
    assert r["num_probe_steps"] == 1
    assert len(r["mocu_trace"]) == 2
    assert len(r["p_trace"]) == 2


def test_single_step_discrete_bayes_report_matches_sequential_posterior():
    """
    For T=1, ``single_step_discrete_bayes_report`` must agree with
    ``posterior_after_sequential_gaussian_observations`` on the same ``(mu, y, log_p0)``.

    Uses the same prior in both the synthetic episode and the report (explicit ``log_p0``).
    """
    grid_side = 3
    n = grid_side * grid_side
    log_p0 = log_prior_uniform_discrete(n)

    r = run_synthetic_episode(
        num_probe_steps=1,
        seed=0,
        grid_side=grid_side,
        log_p0=log_p0,
    )
    mu = r["mu_predictions"][0]
    y = float(r["y_steps"][0])
    gamma_n = r["gamma_star_support"]

    rep = single_step_discrete_bayes_report(
        y, mu, r["sigma_feat"], gamma_n, log_p0=log_p0
    )

    # Sequential path: p_trace[0]=prior, p_trace[1]=posterior after one y
    assert np.allclose(rep["p0"], r["p_trace"][0], rtol=1e-12, atol=1e-14)
    assert np.allclose(rep["p1"], r["p_trace"][1], rtol=1e-6, atol=1e-10)

    # Normalizer: Z = sum_n tilde p^n (report field Z)
    assert abs(rep["Z"] - float(np.sum(rep["tilde_p"]))) < 1e-8

    # MOCU(p1) and γ̂ from report match direct mocu_gamma_star on p1
    m_direct, gh_direct = mocu_gamma_star(rep["p1"], gamma_n)
    assert abs(rep["mocu_post"] - m_direct) < 1e-12
    assert abs(rep["gamma_hat_post"] - gh_direct) < 1e-12


def test_posterior_gaussian_sequence_matches_manual_one_step():
    n = 3
    mu = np.linspace(0.0, 1.0, n)
    mu_steps = mu.reshape(1, n)
    y = np.array([0.3])
    sigma = 0.05
    _, p_trace = posterior_after_sequential_gaussian_observations(mu_steps, y, sigma)
    log_p0 = log_prior_uniform_discrete(n)
    log_L = log_gaussian_observation_density(float(y[0]), mu, sigma)
    p_manual = normalize_log_weights(log_p0 + log_L)
    assert np.allclose(p_trace[-1], p_manual)


@pytest.mark.slow
def test_physics_episode_smoke():
    pytest.importorskip("torch")
    pytest.importorskip("torchdiffeq")
    run_physics_episode(
        num_probe_steps=1,
        seed=123,
        grid_side=QUICK_PHYSICS_GRID,
        device="cpu",
        ode_horizon_sec=ODE_HORIZON_QUICK,
        ode_timeout=QUICK_PHYSICS_TIMEOUT,
    )


@pytest.mark.slow
def test_main_body_single_step_stages_match_raw():
    """Staged pipeline dict agrees with run_single_step_physics_episode."""
    pytest.importorskip("torch")
    pytest.importorskip("torchdiffeq")
    out = main_body_single_step_simulation(
        seed=7,
        grid_side=QUICK_PHYSICS_GRID,
        device="cpu",
        ode_horizon_sec=ODE_HORIZON_QUICK,
        ode_timeout=QUICK_PHYSICS_TIMEOUT,
    )
    raw = out["raw"]
    rep = out["posterior"]["single_step_report"]
    assert out["mocu"]["mocu_post"] == rep["mocu_post"]
    assert out["map_likelihood"]["y"] == raw["single_step_report"]["y"]
    assert out["evaluation_control"]["gamma_star_true"] == raw["gamma_star_true"]
    assert set(out.keys()) >= {
        "xi",
        "gamma_star_support",
        "map_likelihood",
        "posterior",
        "evaluation_control",
        "mocu",
        "raw",
    }


@pytest.mark.slow
def test_single_step_physics_episode_matches_physics_one_step_and_includes_u_ctrl():
    """Full γ* bisection + single_step_discrete_bayes_report + u_ctrl sample at θ_true."""
    pytest.importorskip("torch")
    pytest.importorskip("torchdiffeq")
    r = run_single_step_physics_episode(
        seed=123,
        grid_side=QUICK_PHYSICS_GRID,
        device="cpu",
        ode_horizon_sec=ODE_HORIZON_QUICK,
        ode_timeout=QUICK_PHYSICS_TIMEOUT,
    )
    assert r["num_probe_steps"] == 1
    rep = r["single_step_report"]
    assert np.allclose(rep["p0"], r["p_trace"][0])
    assert np.allclose(rep["p1"], r["p_trace"][1])
    assert np.isfinite(r["gamma_star_true"])
    assert r["u_ctrl_trajectory"].ndim == 2
    assert r["u_ctrl_trajectory"].shape[1] == 14
    assert r["u_ctrl_max_abs"] >= 0.0
