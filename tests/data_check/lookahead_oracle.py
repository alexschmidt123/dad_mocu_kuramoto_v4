"""Planning diagnostics and certification protocol (information redundancy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.contrastive.spce import (
    clamp_info_gain,
    log_gaussian_observation_density,
    normalize_log_weights,
    posterior_entropy,
)
from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables
from tests.data_check.oracle_eig import mc_particle_eig
from tests.data_check.rollout import evaluate_policy_matrix


@dataclass(frozen=True)
class CertificationResult:
    horizon: int
    noise_replicas: int
    n_certification_systems: int
    # Search-split planning diagnostics (NOT used for G_full / G_adapt)
    q_values: dict[int, float]
    oracle_first_action: int
    myopic_first_action: int
    v_star_planning: float
    best_fixed_sequence: tuple[int, ...]
    branch_report: dict[str, Any]
    # Certification-split paired realized values (virtual noise from y_sim)
    v_oracle: float
    v_myopic: float
    v_fixed: float
    g_full: float
    g_adapt: float
    per_system_oracle_mean: np.ndarray
    per_system_myopic_mean: np.ndarray
    per_system_fixed_mean: np.ndarray
    per_system_g_full: np.ndarray
    per_system_g_adapt: np.ndarray


def _feasible_actions(used: set[int], n_actions: int) -> list[int]:
    return [a for a in range(n_actions) if a not in used]


def _centres_for_action(table_support: TableThetaSupport, action: int) -> np.ndarray:
    return y_sim_last_step_from_tables(table_support, [action])


def _delta_h_from_update(log_w: np.ndarray, y: float, m: np.ndarray, sigma_y: float) -> tuple[float, np.ndarray]:
    h_before = posterior_entropy(normalize_log_weights(log_w))
    log_L = log_gaussian_observation_density(y, m, sigma_y)
    log_w1 = log_w + log_L
    h_after = posterior_entropy(normalize_log_weights(log_w1))
    return clamp_info_gain(h_before - h_after), log_w1


def estimate_q2(
    log_p0: np.ndarray,
    table_support: TableThetaSupport,
    action: int,
    sigma_y: float,
    K: int,
    rng: np.random.Generator,
    n_actions: int,
) -> float:
    """Q_2(a) = E_{y_1}[ DH_1 + max_{b!=a} E_{y_2|h_1,b}[ DH_2 ] ]  (planning diagnostic)."""
    m_a = _centres_for_action(table_support, action)
    p0 = normalize_log_weights(log_p0)
    totals = np.empty(K, dtype=np.float64)
    inner_k = max(16, K // 4)

    for k in range(K):
        n_k = int(rng.choice(len(p0), p=p0))
        y1 = float(rng.normal(loc=m_a[n_k], scale=sigma_y))
        dh1, log_w1 = _delta_h_from_update(log_p0, y1, m_a, sigma_y)
        best_second = 0.0
        for b in _feasible_actions({action}, n_actions):
            m_b = _centres_for_action(table_support, b)
            res = mc_particle_eig(log_w1, m_b, sigma_y, inner_k, rng)
            best_second = max(best_second, res.eig)
        totals[k] = dh1 + best_second
    return float(np.mean(totals))


def best_fixed_expected_value_t2(
    log_p0: np.ndarray,
    table_support: TableThetaSupport,
    sigma_y: float,
    K: int,
    rng: np.random.Generator,
    n_actions: int,
) -> tuple[tuple[int, int], float]:
    """Select best open-loop pair (a,b) on search split via MC expected DH."""
    best_seq = (0, 1)
    best_val = -np.inf
    for a in range(n_actions):
        for b in range(n_actions):
            if a == b:
                continue
            val = _expected_fixed_sequence_value(log_p0, table_support, (a, b), sigma_y, K, rng)
            if val > best_val:
                best_val, best_seq = val, (a, b)
    return best_seq, float(best_val)


def _expected_fixed_sequence_value(
    log_p0: np.ndarray,
    table_support: TableThetaSupport,
    seq: tuple[int, ...],
    sigma_y: float,
    K: int,
    rng: np.random.Generator,
) -> float:
    totals = np.empty(K, dtype=np.float64)
    for k in range(K):
        log_w = np.array(log_p0, dtype=np.float64)
        total = 0.0
        p = normalize_log_weights(log_w)
        for a in seq:
            m = _centres_for_action(table_support, a)
            n_k = int(rng.choice(len(p), p=p))
            y = float(rng.normal(loc=m[n_k], scale=sigma_y))
            dh, log_w = _delta_h_from_update(log_w, y, m, sigma_y)
            total += dh
            p = normalize_log_weights(log_w)
        totals[k] = total
    return float(np.mean(totals))


def oracle_adaptive_branches_t2(
    log_p0: np.ndarray,
    table_support: TableThetaSupport,
    first_action: int,
    sigma_y: float,
    K: int,
    rng: np.random.Generator,
    n_actions: int,
    n_bins: int = 5,
) -> dict[str, Any]:
    m_a = _centres_for_action(table_support, first_action)
    p0 = normalize_log_weights(log_p0)
    y_samples = np.empty(K, dtype=np.float64)
    second_actions = np.empty(K, dtype=int)
    inner_k = max(16, K // 4)
    for k in range(K):
        n_k = int(rng.choice(len(p0), p=p0))
        y1 = float(rng.normal(loc=m_a[n_k], scale=sigma_y))
        y_samples[k] = y1
        _, log_w1 = _delta_h_from_update(log_p0, y1, m_a, sigma_y)
        best_b, best_eig = _feasible_actions({first_action}, n_actions)[0], -np.inf
        for b in _feasible_actions({first_action}, n_actions):
            res = mc_particle_eig(log_w1, _centres_for_action(table_support, b), sigma_y, inner_k, rng)
            if res.eig > best_eig:
                best_eig, best_b = res.eig, b
        second_actions[k] = best_b

    quantiles = np.quantile(y_samples, np.linspace(0, 1, n_bins + 1))
    bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = quantiles[i], quantiles[i + 1]
        mask = (y_samples >= lo) & ((y_samples < hi) if i < n_bins - 1 else (y_samples <= hi))
        if not np.any(mask):
            continue
        actions, counts = np.unique(second_actions[mask], return_counts=True)
        dominant = int(actions[int(np.argmax(counts))])
        bins.append({
            "bin": i,
            "y_lo": float(lo),
            "y_hi": float(hi),
            "n_samples": int(np.sum(mask)),
            "dominant_second_action": dominant,
            "action_counts": {int(a): int(c) for a, c in zip(actions, counts)},
        })
    return {
        "first_action": int(first_action),
        "n_unique_second_actions": int(len(np.unique(second_actions))),
        "quantile_bins": bins,
    }


def _myopic_first_action(
    log_p0: np.ndarray,
    table_support: TableThetaSupport,
    sigma_y: float,
    n_actions: int,
) -> tuple[int, dict[int, float]]:
    log_unnorm = np.array(log_p0, dtype=np.float64)
    p_before = normalize_log_weights(log_unnorm)
    h_before = posterior_entropy(p_before)
    best_a, best_dh = 0, -np.inf
    scores: dict[int, float] = {}
    for a in range(n_actions):
        m_vals = y_sim_last_step_from_tables(table_support, [a])
        y_hat = float(np.sum(p_before * m_vals))
        log_L = log_gaussian_observation_density(y_hat, m_vals, sigma_y)
        p_after = normalize_log_weights(log_unnorm + log_L)
        dh = clamp_info_gain(h_before - posterior_entropy(p_after))
        scores[a] = dh
        if dh > best_dh:
            best_dh, best_a = dh, a
    return int(best_a), scores


def run_certification_protocol_t2(
    log_p0: np.ndarray,
    table_support: TableThetaSupport,
    certification_systems: list[dict],
    sigma_y: float,
    K: int,
    rng: np.random.Generator,
    n_actions: int,
    *,
    noise_replicas: int = 8,
    action_subset: list[int] | None = None,
) -> CertificationResult:
    """
    Search-split planning + independent certification evaluation.

    G_full and G_adapt use paired realized terminal Delta H on the certification
    split with fresh virtual noise replicas (y_sim + N(0, sigma_y^2)).
    """
    actions = action_subset if action_subset is not None else list(range(n_actions))
    q_vals = {a: estimate_q2(log_p0, table_support, a, sigma_y, K, rng, n_actions) for a in actions}
    oracle_first = int(max(q_vals, key=q_vals.get))
    v_star_planning = float(q_vals[oracle_first])
    myopic_first, _ = _myopic_first_action(log_p0, table_support, sigma_y, n_actions)
    best_fixed, _ = best_fixed_expected_value_t2(log_p0, table_support, sigma_y, K, rng, n_actions)
    branches = oracle_adaptive_branches_t2(
        log_p0, table_support, oracle_first, sigma_y, K, rng, n_actions,
    )

    oracle_mat = evaluate_policy_matrix(
        log_p0, table_support, certification_systems, 2, sigma_y, n_actions, rng,
        policy="oracle", noise_replicas=noise_replicas,
        oracle_first_action=oracle_first, mc_k=K,
    )
    myopic_mat = evaluate_policy_matrix(
        log_p0, table_support, certification_systems, 2, sigma_y, n_actions, rng,
        policy="myopic", noise_replicas=noise_replicas, mc_k=K,
    )
    fixed_mat = evaluate_policy_matrix(
        log_p0, table_support, certification_systems, 2, sigma_y, n_actions, rng,
        policy="fixed", noise_replicas=noise_replicas,
        fixed_sequence=best_fixed, mc_k=K,
    )

    oracle_mean = np.mean(oracle_mat, axis=1)
    myopic_mean = np.mean(myopic_mat, axis=1)
    fixed_mean = np.mean(fixed_mat, axis=1)
    g_full_per = oracle_mean - myopic_mean
    g_adapt_per = oracle_mean - fixed_mean

    v_oracle = float(np.mean(oracle_mat))
    v_myopic = float(np.mean(myopic_mat))
    v_fixed = float(np.mean(fixed_mat))

    return CertificationResult(
        horizon=2,
        noise_replicas=int(noise_replicas),
        n_certification_systems=len(certification_systems),
        q_values=q_vals,
        oracle_first_action=oracle_first,
        myopic_first_action=int(myopic_first),
        v_star_planning=v_star_planning,
        best_fixed_sequence=best_fixed,
        branch_report=branches,
        v_oracle=v_oracle,
        v_myopic=v_myopic,
        v_fixed=v_fixed,
        g_full=v_oracle - v_myopic,
        g_adapt=v_oracle - v_fixed,
        per_system_oracle_mean=oracle_mean,
        per_system_myopic_mean=myopic_mean,
        per_system_fixed_mean=fixed_mean,
        per_system_g_full=g_full_per,
        per_system_g_adapt=g_adapt_per,
    )


# Backward-compatible alias
run_lookahead_oracle_t2 = run_certification_protocol_t2
LookaheadOracleResult = CertificationResult
