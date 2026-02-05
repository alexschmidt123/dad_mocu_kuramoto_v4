"""
Pytest fixtures for experimental design pipeline tests.

Provides IEEE-14 bus system parameters, prior uncertainty bounds, and simulation settings.
No MOCU calculation is performed in these fixtures.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.swing_equation_params import get_default_swing_equation_params


@pytest.fixture(scope="session")
def ieee14_params():
    """Load IEEE-14 bus system parameters (B, P_m, D, g) and M/K bounds."""
    params = get_default_swing_equation_params(
        N=14,
        topology="ieee14",
        coupling_strength=1.0,
        damping=0.1,
        base_power=1.0,
        M_lower=0.3,
        M_upper=2.0,
        K_lower=0.05,
        K_upper=0.50,
    )
    return params


@pytest.fixture(scope="session")
def prior_bounds(ieee14_params):
    """Prior uncertainty bounds (M and K) and a fixed true parameter for testing."""
    M_lower = float(ieee14_params["M_lower"])
    M_upper = float(ieee14_params["M_upper"])
    K_lower = float(ieee14_params["K_lower"])
    K_upper = float(ieee14_params["K_upper"])
    # Fixed true parameters (inside prior) for reproducible tests
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
    """ODE and observation settings (no MOCU)."""
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
    """Candidate experimental designs: B in {1..14}, A in 10 values → 140 rows."""
    buses = list(range(1, 15))
    amplitudes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    return [(b, A) for b in buses for A in amplitudes]
