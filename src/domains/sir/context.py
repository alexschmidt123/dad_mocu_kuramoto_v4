"""Build an ExperimentContext for SIR ODE vector-EIG training/evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import SBOEDConfig, repo_root
from src.inference.spce import log_prior_uniform_discrete
from src.banks.power_grid import resolve_dataset_dir, system_name_from_cfg
from src.objectives.mocu.context import (
    ExperimentContext,
    config_sha256,
    resolve_oracle_tolerance,
    resolve_sigma_y,
)
from src.domains.sir.banks import load_sir_bank, sir_bank_is_complete


def is_sir_config(cfg: SBOEDConfig) -> bool:
    name = system_name_from_cfg(cfg).strip().lower().replace("-", "_")
    if name in {"sir_sde", "sir_sde_eig"}:
        raise ValueError(
            "SIR SDE is out of scope (implicit path likelihood). "
            "Use system.name=sir_ode and config key sir_ode."
        )
    if name in {"sir_ode", "sir"}:
        return True
    if cfg.raw.get("sir_sde"):
        raise ValueError(
            "Config key 'sir_sde' is removed. Use 'sir_ode' "
            "(deterministic ODE + explicit Gaussian likelihood)."
        )
    return bool(cfg.raw.get("sir_ode") or cfg.raw.get("sir"))


def build_sir_context(
    cfg: SBOEDConfig,
    *,
    project_root: Path | None = None,
    ensure_bank: bool = True,
    smoke: bool = False,
    out_dir: Path | None = None,
    experiment_type: str = "eig_based",
) -> ExperimentContext:
    """Assemble a vector-EIG-compatible context with dummy control fields."""
    if str(experiment_type).lower().replace("-", "_") != "eig_based":
        raise ValueError(
            "SIR ODE currently supports experiment_type=eig_based only "
            "(no MOCU/control track)."
        )
    root = project_root or repo_root()
    system = system_name_from_cfg(cfg)
    data_dir = resolve_dataset_dir(cfg, root)

    if ensure_bank and not sir_bank_is_complete(data_dir):
        raise FileNotFoundError(
            f"SIR databank missing or incomplete at {data_dir}. "
            "SIR training/evaluation is databank-only and will not simulate "
            "trajectories on the fly. Restore the complete databank before "
            "running the experiment."
        )

    bank = load_sir_bank(data_dir)
    centres_train = np.asarray(bank["centres_train"], dtype=np.float64)  # Tθ,A,D
    centres_test = np.asarray(bank["centres_test"], dtype=np.float64)
    beta_train = np.asarray(bank["beta_train"], dtype=np.float64)
    gamma_train = np.asarray(bank["gamma_train"], dtype=np.float64)
    beta_test = np.asarray(bank["beta_test"], dtype=np.float64)
    gamma_test = np.asarray(bank["gamma_test"], dtype=np.float64)

    n_train = int(centres_train.shape[0])
    n_val = max(1, n_train // 4) if n_train >= 4 else max(1, n_train // 2)
    n_fit = max(1, n_train - n_val)
    requested_mc = int((cfg.raw.get("prior") or {}).get("mc_samples", n_fit))
    mc_target = min(requested_mc, n_fit)
    mc_seed = int((cfg.raw.get("prior") or {}).get("mc_support_seed", 1))
    fit_indices = np.arange(n_fit)
    if mc_target < n_fit:
        pick = np.random.default_rng(mc_seed).choice(
            fit_indices, size=mc_target, replace=False
        )
        pick = np.sort(pick)
        print(
            f"[prior] SIR using mc_samples={mc_target}/{n_fit} fit particles "
            f"(seed={mc_seed})"
        )
    else:
        pick = fit_indices

    centres_support = np.transpose(centres_train[pick], (1, 0, 2)).astype(np.float64)
    beta_support = beta_train[pick]
    gamma_support = gamma_train[pick]
    u_support = np.zeros(pick.shape[0], dtype=np.float64)
    log_p0 = log_prior_uniform_discrete(pick.shape[0])

    def _systems(beta, gamma, centres_nt) -> list[dict[str, Any]]:
        out = []
        for i in range(beta.shape[0]):
            out.append(
                {
                    "M": [float(beta[i])],
                    "K": [float(gamma[i])],
                    "beta": float(beta[i]),
                    "gamma": float(gamma[i]),
                    "u_req": 0.0,
                    "obs_clean": centres_nt[i],
                }
            )
        return out

    train_systems_full = _systems(beta_train, gamma_train, centres_train)
    test_systems = _systems(beta_test, gamma_test, centres_test)
    validation_systems = train_systems_full[n_fit:]
    train_fit = train_systems_full[:n_fit]

    raw_particles = np.column_stack(
        [beta_support, gamma_support, u_support]
    ).astype(np.float64)
    p_mean = raw_particles.mean(axis=0)
    p_std = np.maximum(raw_particles.std(axis=0), 1e-8)
    particles = ((raw_particles - p_mean) / p_std).astype(np.float32)

    obs_flat = centres_train[pick].reshape(-1)
    obs_mean = float(obs_flat.mean())
    obs_std = float(max(obs_flat.std(), 1e-8))
    obs_dim = int(centres_support.shape[-1])
    n_actions = int(centres_support.shape[0])

    if out_dir is None:
        from src.objectives.mocu.context import experiment_out_dir

        out_dir = experiment_out_dir(
            cfg, root, experiment_type="eig_based", create_new=False
        )
    else:
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    from src.reporting.run_context import ensure_result_layout

    ensure_result_layout(out_dir)

    u_grid = np.asarray([0.0], dtype=np.float64)
    alpha = 0.01
    margin = 0.0
    fixed_seq = list(range(int(cfg.step_number)))

    obs = dict(cfg.raw.get("observation") or {})
    obs["N_obs"] = int(obs_dim)
    cfg.raw["observation"] = obs

    return ExperimentContext(
        system=system,
        cfg=cfg,
        horizon=int(cfg.step_number),
        n_actions=n_actions,
        n_obs=obs_dim,
        n_sim=obs_dim,
        obs_dim=obs_dim,
        obs_indices=np.arange(obs_dim, dtype=np.int64),
        observation_mode="sir_infected_count",
        sigma_y=resolve_sigma_y(cfg),
        alpha=alpha,
        margin=margin,
        robust_rule="quantile",
        u_grid=u_grid,
        centres_support=centres_support,
        U_support=u_support,
        log_p0=log_p0,
        M_support=beta_support,
        K_support=gamma_support,
        particle_features=particles,
        obs_mean=obs_mean,
        obs_std=obs_std,
        test_systems=test_systems,
        train_systems=train_fit,
        validation_systems=validation_systems,
        U_test=np.zeros(beta_test.shape[0], dtype=np.float64),
        M_test=beta_test.reshape(-1, 1),
        K_test=gamma_test.reshape(-1, 1),
        data_dir=Path(bank["path"]),
        out_dir=out_dir,
        oracle_tolerance=resolve_oracle_tolerance(cfg),
        fixed_sequence=fixed_seq,
        terminal_rule_hash="sir_ode_no_terminal_rule",
        config_hash=config_sha256(cfg),
    )
