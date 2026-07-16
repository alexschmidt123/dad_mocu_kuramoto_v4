"""Realized T-step rollouts for certification (virtual noise from y_sim)."""

from __future__ import annotations

import numpy as np

from src.contrastive.spce import (
    clamp_info_gain,
    log_gaussian_observation_density,
    normalize_log_weights,
    posterior_entropy,
)
from src.data import lookup_action_y_sim
from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables
from tests.data_check.oracle_eig import mc_particle_eig


def sample_virtual_observation(
    system: dict,
    action: int,
    sigma_y: float,
    rng: np.random.Generator,
) -> float:
    """y = y_sim(theta*, a) + epsilon,  epsilon ~ N(0, sigma_y^2)."""
    y_clean = float(lookup_action_y_sim(system, int(action)))
    if sigma_y <= 0:
        return y_clean
    return float(y_clean + rng.normal(0.0, sigma_y))


def _feasible_actions(used: set[int], n_actions: int) -> list[int]:
    return [a for a in range(n_actions) if a not in used]


def _select_myopic_action(
    log_unnorm: np.ndarray,
    table_support: TableThetaSupport,
    seq: list[int],
    used: set[int],
    sigma_y: float,
    n_actions: int,
) -> int:
    p_before = normalize_log_weights(log_unnorm)
    h_before = posterior_entropy(p_before)
    best_a, best_dh = _feasible_actions(used, n_actions)[0], -np.inf
    for cand in _feasible_actions(used, n_actions):
        m_vals = y_sim_last_step_from_tables(table_support, seq + [cand])
        y_hat = float(np.sum(p_before * m_vals))
        log_L = log_gaussian_observation_density(y_hat, m_vals, sigma_y)
        p_after = normalize_log_weights(log_unnorm + log_L)
        dh = clamp_info_gain(h_before - posterior_entropy(p_after))
        if dh > best_dh:
            best_dh, best_a = dh, cand
    return int(best_a)


def _select_oracle_second_action(
    log_unnorm: np.ndarray,
    table_support: TableThetaSupport,
    seq: list[int],
    used: set[int],
    sigma_y: float,
    n_actions: int,
    rng: np.random.Generator,
    mc_k: int,
) -> int:
    best_a, best_eig = _feasible_actions(used, n_actions)[0], -np.inf
    for b in _feasible_actions(used, n_actions):
        res = mc_particle_eig(
            log_unnorm,
            y_sim_last_step_from_tables(table_support, seq + [b]),
            sigma_y,
            mc_k,
            rng,
        )
        if res.eig > best_eig:
            best_eig, best_a = res.eig, b
    return int(best_a)


def realized_terminal_delta_h(
    log_p0: np.ndarray,
    table_support: TableThetaSupport,
    system: dict,
    horizon: int,
    sigma_y: float,
    n_actions: int,
    rng: np.random.Generator,
    *,
    policy: str,
    oracle_first_action: int | None = None,
    fixed_sequence: tuple[int, ...] | None = None,
    mc_k: int = 128,
) -> float:
    """Single-pass realized Delta H_{1:T} with fresh virtual noise each step."""
    log_unnorm = np.array(log_p0, dtype=np.float64)
    h0 = posterior_entropy(normalize_log_weights(log_unnorm))
    used: set[int] = set()
    seq: list[int] = []

    if policy == "fixed":
        if fixed_sequence is None:
            raise ValueError("fixed_sequence required for policy='fixed'")
        action_plan = list(fixed_sequence)
    else:
        action_plan = None

    for t in range(horizon):
        if policy == "fixed":
            a = int(action_plan[t])
        elif policy == "oracle":
            if t == 0:
                if oracle_first_action is None:
                    raise ValueError("oracle_first_action required at t=0")
                a = int(oracle_first_action)
            else:
                a = _select_oracle_second_action(
                    log_unnorm, table_support, seq, used, sigma_y, n_actions, rng, mc_k,
                )
        elif policy == "myopic":
            a = _select_myopic_action(log_unnorm, table_support, seq, used, sigma_y, n_actions)
        else:
            raise ValueError(f"unknown policy {policy!r}")

        seq.append(a)
        y = sample_virtual_observation(system, a, sigma_y, rng)
        m_obs = y_sim_last_step_from_tables(table_support, seq)
        log_unnorm = log_unnorm + log_gaussian_observation_density(y, m_obs, sigma_y)
        used.add(a)

    hT = posterior_entropy(normalize_log_weights(log_unnorm))
    return clamp_info_gain(h0 - hT)


def evaluate_policy_matrix(
    log_p0: np.ndarray,
    table_support: TableThetaSupport,
    systems: list[dict],
    horizon: int,
    sigma_y: float,
    n_actions: int,
    rng: np.random.Generator,
    *,
    policy: str,
    noise_replicas: int,
    oracle_first_action: int | None = None,
    fixed_sequence: tuple[int, ...] | None = None,
    mc_k: int = 128,
) -> np.ndarray:
    """Shape (M, R): realized terminal Delta H for each system and noise replica."""
    m = len(systems)
    r = int(noise_replicas)
    out = np.empty((m, r), dtype=np.float64)
    for i, sys in enumerate(systems):
        for rep in range(r):
            out[i, rep] = realized_terminal_delta_h(
                log_p0,
                table_support,
                sys,
                horizon,
                sigma_y,
                n_actions,
                rng,
                policy=policy,
                oracle_first_action=oracle_first_action,
                fixed_sequence=fixed_sequence,
                mc_k=mc_k,
            )
    return out
