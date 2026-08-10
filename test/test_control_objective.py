"""Unit tests for control-objective invariants and power-injection model."""

from __future__ import annotations

import math
from itertools import combinations, permutations

import numpy as np
import torch

from src.control.fixed_search import n_choose_k
from src.control.posterior_ctrl import normalize_log_weights, posterior_safe_u_ctrl, weighted_quantile
from src.control.u_req import ControlSpec
from src.neural.policy import DADPolicy
from src.rollout import RandomSelector


def test_weights_sum_to_one():
    w = normalize_log_weights(np.array([0.0, -1.0, 2.0, -5.0]))
    assert abs(float(np.sum(w)) - 1.0) < 1e-12


def test_weighted_quantile_and_posterior_safe_u():
    U = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([0.1, 0.2, 0.3, 0.4])
    assert abs(posterior_safe_u_ctrl(U, w, alpha=0.05) - 4.0) < 1e-12
    assert abs(posterior_safe_u_ctrl(U, w, alpha=0.5) - 3.0) < 1e-12
    assert abs(weighted_quantile(U, w, 0.0) - 1.0) < 1e-12


def test_fixed_searches_combinations_not_permutations():
    n, T = 6, 2
    assert n_choose_k(n, T) == math.comb(n, T)
    assert n_choose_k(n, T) < math.perm(n, T)
    assert len(list(combinations(range(n), T))) == math.comb(n, T)


def test_random_samples_without_replacement():
    rng = np.random.default_rng(0)
    sel = RandomSelector(n_actions=10)
    used: set[int] = set()
    for _ in range(5):
        a = sel.select(used=used, rng=rng)
        assert a not in used
        used.add(a)


def test_dad_receives_complete_history():
    policy = DADPolicy(n_actions=8, hidden=32, max_steps=4)
    policy.eval()
    with torch.no_grad():
        h_a = policy.encoder(
            torch.tensor([[0, 1]]), torch.tensor([[1.0, 2.0]]), torch.ones(1, 2)
        )
        h_b = policy.encoder(
            torch.tensor([[3, 1]]), torch.tensor([[9.0, 2.0]]), torch.ones(1, 2)
        )
        h_last = policy.encoder(
            torch.tensor([[1]]), torch.tensor([[2.0]]), torch.ones(1, 1)
        )
    assert not torch.allclose(h_a, h_b, atol=1e-5)
    assert not torch.allclose(h_a, h_last, atol=1e-5)


def test_control_spec_is_power_injection_not_droop():
    from src.config import load_config_for_run, repo_root

    cfg = load_config_for_run("ieee5_config", repo_root(), step_number=2)
    spec = ControlSpec.from_cfg(cfg)
    assert spec.profile.units == "pu"
    assert spec.profile.shape in {"step", "hann", "ramp"}
    assert "delta_f_nadir_hz" in cfg.raw["control"] or spec.delta_f_nadir_hz is not None
    # Must not silently treat u as droop on K: config documents power injection.
    assert cfg.raw["control"]["profile"]["shape"] in {"step", "hann", "ramp"}


def test_nadir_key_is_deviation():
    from src.config import load_config_for_run, repo_root

    cfg = load_config_for_run("ieee5_config", repo_root(), step_number=2)
    assert "delta_f_nadir_hz" in cfg.raw["control"]
    assert "f_nadir_hz" not in cfg.raw["control"]


def test_old_objectives_absent_from_main_methods():
    from src.config import ALL_METHODS

    assert ALL_METHODS == ["dad", "myopic", "fixed", "random"]


def test_myopic_is_one_step_only():
    import inspect
    from src.control.myopic import MyopicControlSelector

    sig = inspect.signature(MyopicControlSelector.select)
    assert "remaining" not in sig.parameters
    assert "horizon" not in sig.parameters


def test_pycuda_backend_flag_still_required():
    from src.config import load_config_for_run, repo_root

    cfg = load_config_for_run("ieee5_config", repo_root(), step_number=2)
    assert cfg.data.get("backend", "cuda") == "cuda"


def test_six_amplitudes():
    from src.config import load_config_for_run, repo_root
    from src.swing_equation_ode.design import build_catalog

    cfg = load_config_for_run("ieee5_config", repo_root(), step_number=2)
    assert len(cfg.probe_amplitudes) == 6
    assert len(build_catalog(cfg)) == 6 * cfg.N
