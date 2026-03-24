"""
Shared pytest fixtures for ``tests/posterior_inference/`` (commercial layout: tests only here).

Used by integration tests under ``integration/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.swing_equation_params import get_default_swing_equation_params


@pytest.fixture(scope="session")
def ieee14_params():
    """IEEE-14 bus parameters (B, P_m, D, g) and M/K bounds."""
    return get_default_swing_equation_params(
        N=14,
        topology="ieee14",
        coupling_strength=1.0,
        damping=0.1,
        base_power=1.0,
        M_lower=0.01,
        M_upper=0.06,
        K_lower=0.05,
        K_upper=0.50,
    )


@pytest.fixture(scope="session")
def prior_bounds(ieee14_params):
    """Prior bounds and fixed true (M, K) for reproducible integration tests."""
    M_lower = float(ieee14_params["M_lower"])
    M_upper = float(ieee14_params["M_upper"])
    K_lower = float(ieee14_params["K_lower"])
    K_upper = float(ieee14_params["K_upper"])
    np.random.seed(42)
    M_true = np.random.uniform(M_lower, M_upper)
    K_true = np.random.uniform(K_lower, K_upper)
    return {
        "M_lower": M_lower,
        "M_upper": M_upper,
        "K_lower": K_lower,
        "K_upper": K_upper,
        "M_true": M_true,
        "K_true": K_true,
    }


@pytest.fixture(scope="session")
def simulation_settings():
    """ODE / observation settings for design-pipeline integration tests."""
    return {
        "h": 1.0 / 160.0,
        "T": 5.0,
        "probe_duration": 2.0,
        "fs": 12.0,
        "device": "cpu",
        "timeout": 10.0,
    }


@pytest.fixture(scope="session")
def design_candidates():
    """140 designs: buses 1..14 × 10 amplitudes."""
    buses = list(range(1, 15))
    amplitudes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    return [(b, A) for b in buses for A in amplitudes]
