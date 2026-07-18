"""Generate and annotate particle-adequacy master banks (PyCUDA only)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.config import load_config_for_run, repo_root
from src.control.banks import extract_U_bank
from src.control.generate import generate_control_bank
from src.control.particle_posterior_adequacy import (
    HISTORICAL_DATA_SLUGS,
    MASTER_N,
    NESTED_SUPPORT_SIZES,
    OUT,
    SYSTEM_CONFIGS,
)
from src.data import (
    data_dir,
    get_systems,
    load_manifest,
    load_tables,
)
from src.experiment import generate_tables
from src.swing_equation_ode.design import build_catalog
from src.swing_equation_ode.simulator import system_mk


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_slug(system: str) -> str:
    return f"{system}_particle_adequacy_master_{MASTER_N}"


def config_name_for(system: str) -> str:
    if system not in SYSTEM_CONFIGS:
        raise ValueError(f"unsupported system {system!r}; expected ieee5|ieee9")
    return SYSTEM_CONFIGS[system]


def assert_cuda_only(cfg) -> None:
    backend = str(cfg.data.get("backend", "")).lower()
    if backend != "cuda":
        raise RuntimeError(
            f"particle-adequacy master banks require data_generation.backend=cuda "
            f"(got {backend!r}). Python/CPU ODE fallback is not allowed."
        )


def assert_not_historical(data_path: Path, system: str) -> None:
    hist = HISTORICAL_DATA_SLUGS[system]
    if data_path.resolve().name == hist:
        raise RuntimeError(
            f"Refusing to write into historical production bank {hist!r}. "
            f"Use slug {dataset_slug(system)!r}."
        )
    # Never nest under historical folders either.
    if hist in data_path.resolve().parts and data_path.resolve().name == hist:
        raise RuntimeError(f"Refusing to overwrite historical path {data_path}")


def extract_master_arrays(
    systems: list[dict[str, Any]],
    *,
    n_buses: int,
    n_actions: int,
) -> dict[str, np.ndarray]:
    n = len(systems)
    M = np.zeros((n, n_buses), dtype=np.float64)
    K = np.zeros((n, n_buses), dtype=np.float64)
    Y_sim = np.full((n, n_actions), np.nan, dtype=np.float64)
    Y = np.full((n, n_actions), np.nan, dtype=np.float64)
    for i, sys in enumerate(systems):
        m, k = system_mk(sys, n_buses)
        M[i] = m
        K[i] = k
        trajs = sys.get("trajectories") or []
        by_a = {int((t.get("sequence") or [-1])[0]): t for t in trajs}
        for a in range(n_actions):
            t = by_a.get(a)
            if t is None:
                continue
            ys = t.get("y_sim") or [np.nan]
            yo = t.get("y") or [np.nan]
            Y_sim[i, a] = float(ys[0])
            Y[i, a] = float(yo[0])
    U = extract_U_bank(systems).astype(np.float64).reshape(-1)
    if U.shape[0] != n:
        raise ValueError(f"U_bank length {U.shape[0]} != n_theta {n}")
    return {"M": M, "K": K, "Y_sim": Y_sim, "Y": Y, "U": U}


def write_nested_subsets(data_path: Path, n_master: int) -> dict[str, Any]:
    sizes = [n for n in NESTED_SUPPORT_SIZES if n <= n_master]
    if MASTER_N not in sizes and n_master >= MASTER_N:
        sizes.append(MASTER_N)
    if n_master not in sizes:
        sizes.append(n_master)
    sizes = sorted(set(sizes))
    doc: dict[str, Any] = {
        "ordering": "prefix_of_master_train",
        "description": (
            "Convergence-study particle supports are nested prefixes of the "
            "ordered master train bank. Particle index n is the same θ in "
            "theta/M/K, Y-bank rows, and U-bank rows."
        ),
        "n_master": int(n_master),
        "supports": {},
    }
    for n in sizes:
        indices = list(range(n))
        doc["supports"][str(n)] = {
            "n_particles": n,
            "index_start": 0,
            "index_end_exclusive": n,
            "particle_indices": indices,
            "selection_rule": f"theta_master[:{n}]",
        }
    out = data_path / "nested_particle_supports.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
    return doc


def write_compact_arrays(data_path: Path, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    paths = {
        "theta_M": "theta_M.npy",
        "theta_K": "theta_K.npy",
        "Y_bank_sim": "Y_bank_sim.npy",
        "Y_bank": "Y_bank.npy",
        "U_bank": "U_bank.npy",
    }
    mapping = {
        "theta_M": arrays["M"],
        "theta_K": arrays["K"],
        "Y_bank_sim": arrays["Y_sim"],
        "Y_bank": arrays["Y"],
        "U_bank": arrays["U"],
    }
    written: dict[str, str] = {}
    for key, name in paths.items():
        p = data_path / name
        np.save(p, mapping[key])
        written[key] = str(p.resolve())
    # Combined theta samples for convenience.
    theta = np.concatenate([arrays["M"], arrays["K"]], axis=1)
    theta_path = data_path / "theta_samples.npy"
    np.save(theta_path, theta)
    written["theta_samples"] = str(theta_path.resolve())
    return written


def build_metadata(
    *,
    system: str,
    cfg,
    data_path: Path,
    arrays: dict[str, np.ndarray],
    nested: dict[str, Any],
    git_commit: str | None,
) -> dict[str, Any]:
    catalog = build_catalog(cfg)
    n_buses = int(cfg.N)
    n_actions = len(catalog)
    n_master = int(arrays["M"].shape[0])
    buses = list(range(n_buses))
    amplitudes = list(cfg.probe_amplitudes)
    duration = float(cfg.probe_duration)
    designs = [list(d.as_tuple()) for d in catalog]
    created = datetime.now(timezone.utc).isoformat()
    train_hash = _file_sha256(data_path / "train.json")
    u_hash = _file_sha256(data_path / "train_control_bank.json")
    return {
        "dataset_name": dataset_slug(system),
        "purpose": "posterior particle adequacy / convergence study",
        "intended_experiment": "experiments/particle_posterior_adequacy/",
        "system": system,
        "system_label": cfg.system_label,
        "topology": cfg.topology,
        "latent_definition": "theta = (M_1,...,M_N, K_1,...,K_N)",
        "latent_dimension": 2 * n_buses,
        "prior": {
            "M_i": "Uniform[0.01, 0.06]",
            "K_i": "Uniform[0.05, 0.5]",
            "sampling": "independent per bus via sample_mk_prior",
        },
        "number_of_theta_samples_master_train": n_master,
        "number_of_theta_samples_test_holdout": int(cfg.theta_sample_size("test")),
        "observation": "max absolute ROCOF (Hz/s) over the observation window",
        "design": {
            "xi": "{A, B, d}",
            "amplitude_set": amplitudes,
            "buses": buses,
            "duration_s": duration,
            "n_actions": n_actions,
            "catalog": designs,
            "ordering": "outer amplitude, inner bus 0..N-1",
        },
        "simulator": {
            "backend": "cuda",
            "stack": "offline PyCUDA through C++/CUDA",
            "cpu_fallback_allowed": False,
            "trajectory_mode": "one_step_bank",
        },
        "Y_bank_shape": list(arrays["Y"].shape),
        "Y_bank_sim_shape": list(arrays["Y_sim"].shape),
        "U_bank_shape": list(arrays["U"].shape),
        "particle_ordering": (
            "index n is the same theta^(n) across theta_samples, "
            "Y_bank rows, and U_bank rows"
        ),
        "nested_particle_supports": nested,
        "random_seeds": {
            "train_seed": int(cfg.data.get("train_seed", 0)),
            "test_seed": int(cfg.data.get("test_seed", 1)),
            "mc_support_seed": int(cfg.prior.get("mc_support_seed", 1)),
        },
        "creation_date_utc": created,
        "config_path": str(cfg.config_path.resolve()),
        "dataset_path": str(data_path.resolve()),
        "git_commit": git_commit,
        "dataset_hashes": {
            "train_json_sha256": train_hash,
            "train_control_bank_sha256": u_hash,
        },
        "historical_banks_preserved": {
            "ieee5_production": "data/ieee5 (do not overwrite)",
            "ieee9_production": "data/ieee9 (do not overwrite)",
        },
        "artifacts": {
            "train_json": "train.json",
            "test_json": "test.json",
            "train_control_bank": "train_control_bank.json",
            "test_control_bank": "test_control_bank.json",
            "manifest": "manifest.yaml",
            "theta_samples": "theta_samples.npy",
            "theta_M": "theta_M.npy",
            "theta_K": "theta_K.npy",
            "Y_bank": "Y_bank.npy",
            "Y_bank_sim": "Y_bank_sim.npy",
            "U_bank": "U_bank.npy",
            "nested_supports": "nested_particle_supports.yaml",
            "metadata": "metadata.yaml",
            "readme": "README.md",
        },
    }


def write_readme(data_path: Path, meta: dict[str, Any]) -> Path:
    amps = meta["design"]["amplitude_set"]
    buses = meta["design"]["buses"]
    y_shape = meta["Y_bank_shape"]
    u_shape = meta["U_bank_shape"]
    nested_ns = sorted(int(k) for k in meta["nested_particle_supports"]["supports"])
    text = f"""# {meta['dataset_name']}

## Purpose

Posterior particle adequacy / convergence study master bank.

Intended experiment: `{meta['intended_experiment']}`

## System

- Identifier: `{meta['system']}` ({meta['system_label']})
- Topology: `{meta['topology']}`

## Latent definition

`theta = (M_1,...,M_N, K_1,...,K_N)`

- Latent dimension: **{meta['latent_dimension']}**
- Prior:
  - `M_i ~ Uniform[0.01, 0.06]`
  - `K_i ~ Uniform[0.05, 0.5]`

## Master bank size

- Train (master particles): **{meta['number_of_theta_samples_master_train']}**
- Test holdout (independent): **{meta['number_of_theta_samples_test_holdout']}**

## Observation

{meta['observation']}

## Design

`xi = {{A, B, d}}`

- Duration: **{meta['design']['duration_s']} s**
- Amplitudes: `{amps}`
- Buses: `{buses}`
- Actions: **{meta['design']['n_actions']}** (`len(amplitudes) * n_buses`)

## Simulator

Offline **PyCUDA through C++/CUDA** only. CPU ODE fallback is not allowed.

## Bank shapes

- Y-bank (noisy): `{y_shape}` → `(n_theta, n_actions)`
- U-bank: `{u_shape}` → `(n_theta,)`

Particle index `n` is consistent across theta samples, Y-bank rows, and U-bank rows.

## Nested particle supports

Physical data generated once at `N_master = {meta['number_of_theta_samples_master_train']}`.

Convergence supports (prefixes of the ordered master train bank):

{', '.join(str(n) for n in nested_ns)}

See `nested_particle_supports.yaml`.

## Seeds

- train_seed: `{meta['random_seeds']['train_seed']}`
- test_seed: `{meta['random_seeds']['test_seed']}`

## Provenance

- Created (UTC): `{meta['creation_date_utc']}`
- Config: `{meta['config_path']}`
- Dataset path: `{meta['dataset_path']}`
- Git commit: `{meta['git_commit']}`
- train.json sha256: `{meta['dataset_hashes']['train_json_sha256']}`
- train_control_bank.json sha256: `{meta['dataset_hashes']['train_control_bank_sha256']}`

## Historical data

This dataset is separate from production banks:

- `data/ieee5`
- `data/ieee9`

Do not overwrite those folders.
"""
    path = data_path / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_experiment_trace(
    *,
    system: str,
    cfg,
    data_path: Path,
    meta: dict[str, Any],
) -> Path:
    exp_sys = OUT / f"{system}_T3"
    exp_sys.mkdir(parents=True, exist_ok=True)
    (exp_sys / "config").mkdir(exist_ok=True)
    (exp_sys / "logs").mkdir(exist_ok=True)
    (exp_sys / "summary").mkdir(exist_ok=True)

    # Link experiment to the master dataset (traceability).
    run_cfg = {
        "study": "particle_posterior_adequacy",
        "system": system,
        "step_number": int(cfg.step_number),
        "dataset_name": meta["dataset_name"],
        "data_dir": str(data_path.resolve()),
        "dataset_path": str(data_path.resolve()),
        "dataset_version": {
            "git_commit": meta.get("git_commit"),
            "train_json_sha256": meta["dataset_hashes"]["train_json_sha256"],
            "train_control_bank_sha256": meta["dataset_hashes"]["train_control_bank_sha256"],
            "creation_date_utc": meta["creation_date_utc"],
        },
        "master_n_particles": meta["number_of_theta_samples_master_train"],
        "nested_support_sizes": sorted(
            int(k) for k in meta["nested_particle_supports"]["supports"]
        ),
        "support_selection": "prefix_of_master_train",
        "support_seed": meta["random_seeds"]["train_seed"],
        "selected_particle_indices_rule": "range(0, N) for each nested N",
        "config_path": str(cfg.config_path.resolve()),
        "historical_data_not_used": f"data/{HISTORICAL_DATA_SLUGS[system]}",
    }
    run_path = exp_sys / "run_config.yaml"
    with run_path.open("w", encoding="utf-8") as f:
        yaml.dump(run_cfg, f, default_flow_style=False, sort_keys=False)

    # Per-subset index files for exact traceability.
    subset_dir = exp_sys / "particle_subsets"
    subset_dir.mkdir(exist_ok=True)
    for key, support in meta["nested_particle_supports"]["supports"].items():
        n = int(key)
        subset_doc = {
            "dataset_name": meta["dataset_name"],
            "dataset_path": str(data_path.resolve()),
            "particle_subset_size": n,
            "support_seed": meta["random_seeds"]["train_seed"],
            "selected_particle_indices": support["particle_indices"],
            "selection_rule": support["selection_rule"],
            "dataset_hashes": meta["dataset_hashes"],
        }
        with (subset_dir / f"support_N{n}.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(subset_doc, f, default_flow_style=False, sort_keys=False)

    # Copy config used for generation.
    cfg_copy = exp_sys / "config" / Path(cfg.config_path).name
    cfg_copy.write_text(Path(cfg.config_path).read_text(encoding="utf-8"), encoding="utf-8")

    meta_path = exp_sys / "dataset_trace.json"
    meta_path.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    return run_path


def generate_master_for_system(
    system: str,
    *,
    project_root: Path | None = None,
    skip_if_complete: bool = True,
) -> dict[str, Any]:
    """
    Generate Y-bank + U-bank for one system via PyCUDA, then write study metadata.

    Physical data lands under ``data/<system>_particle_adequacy_master_2048/``.
    """
    if system == "ieee14":
        raise ValueError("IEEE14 master data is out of scope for this study.")
    root = project_root or repo_root()
    cfg_name = config_name_for(system)
    cfg = load_config_for_run(cfg_name, root, step_number=3)
    assert_cuda_only(cfg)

    data_path = data_dir(root, cfg)
    assert_not_historical(data_path, system)
    expected = dataset_slug(system)
    if data_path.name != expected:
        raise RuntimeError(
            f"Config run_slug produced data path {data_path.name!r}; "
            f"expected {expected!r}"
        )

    n_train = int(cfg.theta_sample_size("train"))
    if n_train != MASTER_N:
        raise RuntimeError(
            f"Config train θ count is {n_train}, expected master size {MASTER_N}"
        )

    print(f"\n=== particle_posterior_adequacy master bank [{system}] ===")
    print(f"  dataset: {expected}")
    print(f"  data_dir: {data_path}")
    print(f"  historical preserved: data/{HISTORICAL_DATA_SLUGS[system]}")
    print(f"  backend: cuda (PyCUDA / C++ CUDA) — no CPU fallback")

    # Phase A: Y-bank (probe observations)
    exp_link = OUT / f"{system}_T3"
    exp_link.mkdir(parents=True, exist_ok=True)
    generate_tables(cfg, root, exp_dir=exp_link)

    # Phase B: U-bank
    print(f"\n=== generate-control-bank [{system}] ===")
    generate_control_bank(cfg_name, project_root=root, splits=("train", "test"))

    # Phase C: compact arrays + nested supports + metadata
    train_payload = load_tables(data_path / "train.json")
    systems = list(get_systems(train_payload))
    catalog = build_catalog(cfg)
    arrays = extract_master_arrays(
        systems, n_buses=int(cfg.N), n_actions=len(catalog),
    )
    if arrays["M"].shape[0] != MASTER_N:
        raise RuntimeError(
            f"master train size {arrays['M'].shape[0]} != {MASTER_N}"
        )
    written = write_compact_arrays(data_path, arrays)
    nested = write_nested_subsets(data_path, MASTER_N)
    git_commit = _git_commit(root)
    meta = build_metadata(
        system=system,
        cfg=cfg,
        data_path=data_path,
        arrays=arrays,
        nested=nested,
        git_commit=git_commit,
    )
    meta["compact_array_paths"] = written
    # Preserve any generation timing from manifest.
    man = load_manifest(data_path)
    meta["generation_duration"] = man.get("generation_duration")
    meta["generation_seconds"] = man.get("generation_seconds")

    meta_path = data_path / "metadata.yaml"
    with meta_path.open("w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
    write_readme(data_path, meta)
    run_path = write_experiment_trace(
        system=system, cfg=cfg, data_path=data_path, meta=meta,
    )

    summary = {
        "system": system,
        "dataset_name": expected,
        "data_dir": str(data_path.resolve()),
        "n_master": MASTER_N,
        "n_actions": len(catalog),
        "Y_bank_shape": list(arrays["Y"].shape),
        "U_bank_shape": list(arrays["U"].shape),
        "metadata": str(meta_path.resolve()),
        "experiment_run_config": str(run_path.resolve()),
        "git_commit": git_commit,
        "skipped_regeneration": False,
    }
    # Touch skip flag only for callers that want it; generation always refreshes metadata.
    _ = skip_if_complete
    print(json.dumps(summary, indent=2))
    return summary


def generate_masters(
    systems: tuple[str, ...] = ("ieee5", "ieee9"),
    *,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for system in systems:
        out.append(generate_master_for_system(system, project_root=project_root))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison").mkdir(exist_ok=True)
    (OUT / "summary").mkdir(exist_ok=True)
    overview = {
        "study": "particle_posterior_adequacy",
        "systems": out,
        "note": "Physical banks live under data/; results under experiments/particle_posterior_adequacy/",
        "ieee14": "not generated (out of scope)",
    }
    overview_path = OUT / "summary" / "master_bank_generation.json"
    overview_path.write_text(json.dumps(overview, indent=2), encoding="utf-8")
    return out
