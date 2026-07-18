"""Posterior particle adequacy / convergence study (IEEE5 & IEEE9 master banks)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "particle_posterior_adequacy"

NESTED_SUPPORT_SIZES = (128, 256, 512, 1024, 2048)
MASTER_N = 2048

SYSTEM_CONFIGS = {
    "ieee5": "ieee5_particle_adequacy_master_2048_config",
    "ieee9": "ieee9_particle_adequacy_master_2048_config",
}

HISTORICAL_DATA_SLUGS = {
    "ieee5": "ieee5",
    "ieee9": "ieee9",
}

SUPPORT_SEEDS = (101, 202, 303, 404, 505)

__all__ = [
    "ROOT",
    "OUT",
    "NESTED_SUPPORT_SIZES",
    "MASTER_N",
    "SYSTEM_CONFIGS",
    "HISTORICAL_DATA_SLUGS",
    "SUPPORT_SEEDS",
]
