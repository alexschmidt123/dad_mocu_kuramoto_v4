"""Monte Carlo particle-EIG oracle (information redundancy — greedy myopic baseline)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.contrastive.spce import (
    log_gaussian_observation_density,
    normalize_log_weights,
    posterior_entropy,
)


@dataclass(frozen=True)
class ParticleEIGResult:
    eig: float
    std_error: float
    h_before: float
    h_after_mean: float
    h_after_samples: np.ndarray


def log_weights_from_posterior(p: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.log(np.clip(p, eps, 1.0))


def mc_particle_eig(
    log_weights: np.ndarray,
    m_vals: np.ndarray,
    sigma_y: float,
    K: int,
    rng: np.random.Generator,
) -> ParticleEIGResult:
    """
    MC estimate of conditional particle EIG for one action.

    EIG(a|h) = H_N(h) - (1/K) sum_k H_N(h, a, y^(k)),
    with n_k ~ Categorical(w), y^(k) ~ N(m_{n_k}(a), sigma_y^2).
    """
    log_w = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    m = np.asarray(m_vals, dtype=np.float64).reshape(-1)
    if log_w.shape != m.shape:
        raise ValueError("log_weights and m_vals must have the same length")
    if K <= 0:
        raise ValueError("K must be positive")

    p = normalize_log_weights(log_w)
    h_before = posterior_entropy(p)
    h_after = np.empty(K, dtype=np.float64)
    for k in range(K):
        n_k = int(rng.choice(len(p), p=p))
        y_k = float(rng.normal(loc=m[n_k], scale=sigma_y))
        log_L = log_gaussian_observation_density(y_k, m, sigma_y)
        p_after = normalize_log_weights(log_w + log_L)
        h_after[k] = posterior_entropy(p_after)

    h_after_mean = float(np.mean(h_after))
    eig = h_before - h_after_mean
    std_error = float(np.std(h_after, ddof=1) / np.sqrt(K)) if K > 1 else 0.0
    return ParticleEIGResult(
        eig=float(eig),
        std_error=std_error,
        h_before=float(h_before),
        h_after_mean=h_after_mean,
        h_after_samples=h_after.copy(),
    )


def mc_expected_delta_h(
    log_weights: np.ndarray,
    m_vals: np.ndarray,
    sigma_y: float,
    K: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """E[ΔH] = H(h) - E[H(h,y)] for a Gaussian update at centres ``m_vals``."""
    res = mc_particle_eig(log_weights, m_vals, sigma_y, K, rng)
    return res.eig, res.std_error


def greedy_mc_eig_action(
    log_weights: np.ndarray,
    action_centres: dict[int, np.ndarray],
    sigma_y: float,
    K: int,
    rng: np.random.Generator,
    *,
    feasible: np.ndarray | None = None,
) -> tuple[int, dict[int, ParticleEIGResult]]:
    """Return argmax_a EIG(a|h) and per-action MC results."""
    if feasible is None:
        actions = sorted(action_centres.keys())
    else:
        actions = [int(a) for a in np.asarray(feasible, dtype=int).reshape(-1)]
    if not actions:
        raise ValueError("no feasible actions for greedy_mc_eig_action")

    results: dict[int, ParticleEIGResult] = {}
    best_a, best_eig = actions[0], -np.inf
    for a in actions:
        res = mc_particle_eig(log_weights, action_centres[a], sigma_y, K, rng)
        results[a] = res
        if res.eig > best_eig:
            best_a, best_eig = a, res.eig
    return int(best_a), results


def mc_conditional_eig_given_action(
    log_weights: np.ndarray,
    m_vals_a: np.ndarray,
    m_vals_b: np.ndarray,
    sigma_y: float,
    K: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    E_{y_a}[ EIG(b | h, a, y_a) ] for directional redundancy.

    First sample y_a from the predictive of action a, update belief, then
    estimate EIG(b) at the updated belief.
    """
    log_w = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    m_a = np.asarray(m_vals_a, dtype=np.float64).reshape(-1)
    m_b = np.asarray(m_vals_b, dtype=np.float64).reshape(-1)
    p = normalize_log_weights(log_w)

    eigs = np.empty(K, dtype=np.float64)
    for k in range(K):
        n_k = int(rng.choice(len(p), p=p))
        y_a = float(rng.normal(loc=m_a[n_k], scale=sigma_y))
        log_L = log_gaussian_observation_density(y_a, m_a, sigma_y)
        log_w1 = log_w + log_L
        res_b = mc_particle_eig(log_w1, m_b, sigma_y, max(16, K // 4), rng)
        eigs[k] = res_b.eig
    return float(np.mean(eigs)), float(np.std(eigs, ddof=1) / np.sqrt(K)) if K > 1 else 0.0
