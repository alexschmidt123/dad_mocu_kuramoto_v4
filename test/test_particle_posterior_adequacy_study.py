"""Tests for particle-posterior-adequacy convergence study (no GPU required)."""

from __future__ import annotations

import numpy as np

from src.control.particle_posterior_adequacy.adaptive_value import (
    adaptive_gain_from_scores,
    classify_bus_case_from_designs,
)
from src.control.particle_posterior_adequacy.diagnostics import (
    degeneracy_flag,
    posterior_weight_stats,
)
from src.control.particle_posterior_adequacy.supports import (
    MASTER_N,
    nested_indices,
)
from src.control.legacy.objective_adaptive_value import classify_case


def test_nested_indices_are_prefixes_of_same_permutation():
    seed = 101
    idx2048 = nested_indices(MASTER_N, 2048, seed)
    assert len(idx2048) == 2048
    assert len(set(idx2048.tolist())) == 2048
    for n in (128, 256, 512, 1024):
        idx = nested_indices(MASTER_N, n, seed)
        assert np.array_equal(idx, idx2048[:n])


def test_different_seeds_permute_differently():
    a = nested_indices(MASTER_N, 256, 101)
    b = nested_indices(MASTER_N, 256, 202)
    assert not np.array_equal(a, b)


def test_posterior_weight_stats_and_flags():
    w = np.ones(100) / 100.0
    stats = posterior_weight_stats(w)
    assert stats["normalized_ESS"] > 0.9
    assert stats["degeneracy_flag"] == "stable_support"
    assert degeneracy_flag(0.01, 0.8) == "severe_degeneracy"
    assert degeneracy_flag(0.1, 0.3) == "moderate_degeneracy"


def test_case_classification_b_when_branching_without_significant_delta():
    gain_rows = [
        {
            "Delta_adaptive": 1e-6,
            "ci95_low": -0.01,
            "ci95_high": 0.01,
            "n_unique_xi2_star": 4,
        }
        for _ in range(5)
    ]
    assert classify_case(gain_rows) == "B"


def test_adaptive_gain_and_bus_case_helpers():
    scored = [
        {
            "history_id": i,
            "xi1": 0,
            "optimal_design": 1 if i % 2 == 0 else 2,
            "J_star_snapped": 0.8,
            "scores_snapped": {1: 0.8, 2: 0.81, 3: 0.9},
        }
        for i in range(8)
    ]
    gain = adaptive_gain_from_scores(scored, xi1=0)
    assert gain["n_histories"] == 8
    assert gain["n_unique_xi2_star"] == 2
    bus = classify_bus_case_from_designs(
        [
            {
                "optimal_bus": i % 3,
                "optimal_amplitude": 0.1,
                "reference_regret": 0.0,
                "history_step": 1,
            }
            for i in range(20)
        ]
    )
    assert bus["bus_case_classification"] in {"BUS-A", "BUS-B", "BUS-C"}


def test_duration_and_amplitude_invariants_on_configs():
    from src.config import load_config_for_run, repo_root
    from src.control.particle_posterior_adequacy import SYSTEM_CONFIGS

    root = repo_root()
    for system, cfg_name in SYSTEM_CONFIGS.items():
        cfg = load_config_for_run(cfg_name, root, step_number=3)
        assert abs(cfg.probe_duration - 0.2) < 1e-12
        assert len(cfg.probe_amplitudes) == 6
        assert cfg.theta_sample_size("train") == MASTER_N
        assert 2 * cfg.N in (10, 18)
        del system
