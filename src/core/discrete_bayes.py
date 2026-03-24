"""
Discrete support Bayesian updates and MOCU — aligned with documents/pseudocode.tex.

**Single-step vs multi-step:** one round of the sequential update is the same object as
``sequential_posterior_from_log_likelihoods`` with ``T=1``, or equivalently
:func:`single_step_discrete_bayes_report` for that round (same likelihood and normalizer).

References (pseudocode.tex):
  - §Likelihood: Gaussian p(y_t | θ_n, ξ_t) on μ_n^(t) = Map(θ_n, ξ_t)
  - §Posterior (e): log-sum-exp normalization
  - eq:mocu_gamma: MOCU(p_t) = Σ_n p_t^n |γ*(θ_n) − γ̂_t|,  γ̂_t ∈ wmed({γ*_n}, {p_t^n})
"""

from __future__ import annotations

from typing import Any

import numpy as np


def log_gaussian_observation_density(
    y: float,
    mu: np.ndarray,
    sigma_feat: float,
) -> np.ndarray:
    """
    log p(y | θ_n, ξ) for each support point n, with y ~ N(μ_n, σ_feat²).

    Pseudocode §Likelihood, §Posterior (e):
      log L_n = -½log(2πσ²) - (y − μ_n)²/(2σ²)
    """
    s2 = sigma_feat**2
    return -0.5 * np.log(2.0 * np.pi * s2) - 0.5 * (y - mu) ** 2 / s2


def normalize_log_weights(log_unnormalized: np.ndarray) -> np.ndarray:
    """
    p^n ∝ exp(log_unnormalized^n); return normalized p with Σ_n p^n = 1.

    Pseudocode §Posterior (e): c = max_n a^n, log Z = c + log Σ exp(a^n − c),
    p^n = exp(a^n − log Z).
    """
    c = float(np.max(log_unnormalized))
    w = np.exp(log_unnormalized - c)
    s = float(np.sum(w))
    if s <= 0:
        raise RuntimeError("Posterior weights degenerate (sum exp = 0).")
    return w / s


def log_prior_uniform_discrete(N: int) -> np.ndarray:
    """log p_0^n for uniform prior on N support points: p_0^n = 1/N."""
    return np.full(N, -np.log(N))


def sequential_posterior_from_log_likelihoods(
    log_L_steps: np.ndarray,
    log_p0: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Sequential Bayes: p ∝ p_0 * Π_t L_t on the discrete support.

    Args:
        log_L_steps: shape (T, N), log L_{t,n} = log p(y_t | θ_n, ξ_t)
        log_p0: shape (N,) log p_0^n; default uniform 1/N

    Returns:
        p_final, p_trace where p_trace = [p after 0 obs, …, p after T obs]
    """
    T, N = log_L_steps.shape
    if log_p0 is None:
        log_p0 = log_prior_uniform_discrete(N)
    log_unnorm = np.array(log_p0, dtype=np.float64)
    p_trace: list[np.ndarray] = [normalize_log_weights(log_unnorm)]
    for t in range(T):
        log_unnorm = log_unnorm + log_L_steps[t]
        p_trace.append(normalize_log_weights(log_unnorm))
    return p_trace[-1], p_trace


def posterior_after_sequential_gaussian_observations(
    mu_steps: np.ndarray,
    y_steps: np.ndarray,
    sigma_feat: float,
    log_p0: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Belief after T Gaussian observations (same σ_feat), sequential Bayes.

    p(θ | y_{1:T}) ∝ p_0(θ) Π_{t=1}^T p(y_t | θ, ξ_t).

    Args:
        mu_steps: shape (T, N), μ_{t,n} = Map(θ_n, ξ_t)
        y_steps: shape (T,)
        sigma_feat: σ_feature (pseudocode σ_feat)
        log_p0: optional log p_0^n; default uniform on N points

    Returns:
        p_final: shape (N,) posterior after all updates
        p_trace: [p_0, p_1, …, p_T] each shape (N,)
    """
    T, N = mu_steps.shape
    if len(y_steps) != T:
        raise ValueError("y_steps length must match mu_steps.shape[0]")
    log_L_steps = np.zeros((T, N), dtype=np.float64)
    for t in range(T):
        log_L_steps[t] = log_gaussian_observation_density(float(y_steps[t]), mu_steps[t], sigma_feat)
    return sequential_posterior_from_log_likelihoods(log_L_steps, log_p0)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted median (any minimizer in the wmed set; pseudocode: Bayes action under L1).

    Smallest value whose cumulative weight is ≥ 1/2 (after sorting by value).
    """
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, 0.5, side="left"))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def single_step_discrete_bayes_report(
    y: float,
    mu: np.ndarray,
    sigma_feat: float,
    gamma_star_n: np.ndarray,
    log_p0: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    One Bayesian update on a discrete grid (pseudocode.tex §Likelihood, §Posterior (c)–(e), §decision eq:mocu_gamma).

    This is the **T=1** slice of :func:`sequential_posterior_from_log_likelihoods` (one
    likelihood vector ``log_L`` applied to ``log_p0``). Multi-step runs repeat the same update.

    **Inputs**
      - ``y``: observed scalar feature (e.g. ROCOF_max).
      - ``mu``: μ_n = Map(θ_n, ξ), shape (N,) — same indexing as support.
      - ``sigma_feat``: σ_feat in Gaussian likelihood.
      - ``gamma_star_n``: γ*(θ_n) on the support (evaluation model).
      - ``log_p0``: log p_{t-1}^n; default uniform log(1/N).

    **Outputs (all aligned with pseudocode)**
      - ``p0``, ``log_p0``: prior weights.
      - ``log_L``, ``L``: log and density ℒ_n = p(y | θ_n, ξ).
      - ``log_unnormalized``, ``tilde_p``: tilde p^n = p_{t-1}^n ℒ_n (unnormalized).
      - ``log_Z``, ``Z``: marginal likelihood / normalizer Z_t = Σ_n tilde p^n.
      - ``p1``: normalized posterior p_t^n.
      - ``mocu_prior``, ``gamma_hat_prior``: MOCU(p_{t-1}), γ̂ = wmed(γ*, p) before update.
      - ``mocu_post``, ``gamma_hat_post``: MOCU(p_t), γ̂ after update.

    Uses: ``log_gaussian_observation_density``, ``normalize_log_weights``, ``mocu_gamma_star``.
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    gamma_star_n = np.asarray(gamma_star_n, dtype=np.float64).reshape(-1)
    N = len(mu)
    if len(gamma_star_n) != N:
        raise ValueError("gamma_star_n must have same length as mu")
    if log_p0 is None:
        log_p0 = log_prior_uniform_discrete(N)
    log_p0 = np.asarray(log_p0, dtype=np.float64).reshape(-1)
    if log_p0.shape[0] != N:
        raise ValueError("log_p0 must have length N")

    p0 = normalize_log_weights(log_p0)
    log_L = log_gaussian_observation_density(float(y), mu, sigma_feat)
    L = np.exp(log_L)
    log_unnorm = log_p0 + log_L
    c = float(np.max(log_unnorm))
    log_Z = c + np.log(np.sum(np.exp(log_unnorm - c)))
    p1 = np.exp(log_unnorm - log_Z)
    tilde_p = np.exp(log_unnorm)

    m0, g0 = mocu_gamma_star(p0, gamma_star_n)
    m1, g1 = mocu_gamma_star(p1, gamma_star_n)

    return {
        "y": float(y),
        "sigma_feat": float(sigma_feat),
        "mu": mu.copy(),
        "log_p0": log_p0.copy(),
        "p0": p0.copy(),
        "log_L": log_L.copy(),
        "L": L.copy(),
        "log_unnormalized": log_unnorm.copy(),
        "tilde_p": tilde_p.copy(),
        "log_Z": float(log_Z),
        "Z": float(np.exp(log_Z)),
        "p1": p1.copy(),
        "gamma_star_n": gamma_star_n.copy(),
        "mocu_prior": float(m0),
        "gamma_hat_prior": float(g0),
        "mocu_post": float(m1),
        "gamma_hat_post": float(g1),
    }


def mocu_gamma_star(p: np.ndarray, gamma_star: np.ndarray) -> tuple[float, float]:
    """
    MOCU(p) and Bayes estimate γ̂ under L1 (absolute error).

    pseudocode.tex eq:mocu_gamma:
      γ̂ ∈ wmed({γ*(θ_n)}, {p^n}),
      MOCU(p) = Σ_n p^n |γ*(θ_n) − γ̂|
    """
    p = np.asarray(p, dtype=np.float64)
    gamma_star = np.asarray(gamma_star, dtype=np.float64)
    valid = np.isfinite(gamma_star) & (p > 0)
    if not np.any(valid):
        return float("nan"), float("nan")
    g = gamma_star[valid]
    w = p[valid]
    w = w / np.sum(w)
    gh = weighted_median(g, w)
    mocu = float(np.sum(w * np.abs(g - gh)))
    return mocu, gh
