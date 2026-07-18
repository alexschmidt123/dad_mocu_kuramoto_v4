"""Unit tests for particle-adequacy master naming / nested supports (no GPU)."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.config import load_config_for_run, repo_root
from src.control.particle_posterior_adequacy import (
    HISTORICAL_DATA_SLUGS,
    MASTER_N,
    NESTED_SUPPORT_SIZES,
    SYSTEM_CONFIGS,
)
from src.control.particle_posterior_adequacy.master_bank import (
    assert_not_historical,
    dataset_slug,
    write_nested_subsets,
)
from src.data import data_dir


def test_dataset_slugs_and_configs_resolve():
    root = repo_root()
    for system, cfg_name in SYSTEM_CONFIGS.items():
        assert dataset_slug(system) == f"{system}_particle_adequacy_master_{MASTER_N}"
        cfg = load_config_for_run(cfg_name, root, step_number=3)
        assert cfg.theta_sample_size("train") == MASTER_N
        assert str(cfg.data.get("backend")).lower() == "cuda"
        assert abs(float(cfg.probe_duration) - 0.2) < 1e-12
        assert list(cfg.probe_amplitudes) == [0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
        d = data_dir(root, cfg)
        assert d.name == dataset_slug(system)
        assert d.name != HISTORICAL_DATA_SLUGS[system]


def test_assert_not_historical(tmp_path: Path):
    hist = tmp_path / "ieee5"
    hist.mkdir()
    try:
        assert_not_historical(hist, "ieee5")
        raised = False
    except RuntimeError:
        raised = True
    assert raised

    ok = tmp_path / "ieee5_particle_adequacy_master_2048"
    ok.mkdir()
    assert_not_historical(ok, "ieee5")


def test_nested_supports_are_prefixes(tmp_path: Path):
    nested = write_nested_subsets(tmp_path, MASTER_N)
    assert nested["ordering"] == "prefix_of_master_train"
    for n in NESTED_SUPPORT_SIZES:
        support = nested["supports"][str(n)]
        assert support["particle_indices"] == list(range(n))
        assert support["index_end_exclusive"] == n
    loaded = yaml.safe_load((tmp_path / "nested_particle_supports.yaml").read_text())
    assert loaded["n_master"] == MASTER_N


def test_ieee14_not_in_system_configs():
    assert "ieee14" not in SYSTEM_CONFIGS
