"""
Bayesian inference, prior grid, and sPCE / PCE estimators.

DAD reference (``ae-foster/dad/contrastive/mi.py``, ``PriorContrastiveEstimation``):
fix the observed history ``h_T`` from a rollout at ``θ_0``, resample contrastive
``θ_1,…,θ_L ~ p_0(θ)``, then

    log p(h_T | θ_0, π) − log( (1/(L+1)) Σ_{ℓ=0}^L p(h_T | θ_ℓ, π) ).

Training minimizes the negative (plus policy log-probs). ``mi.py`` implements this
via Pyro traces; here we use explicit Gaussian ROCOF likelihoods.
"""

from __future__ import annotations

import numpy as np


def log_gaussian_observation_density(
    y: float,
    f_vals: np.ndarray,
    sigma_y: float,
) -> np.ndarray:
    """log p(y | θ_n, ξ) = log N(y; F(θ_n, ξ), σ_y²) on a discrete θ support."""
    s2 = sigma_y**2
    return -0.5 * np.log(2.0 * np.pi * s2) - 0.5 * (y - f_vals) ** 2 / s2


def normalize_log_weights(log_unnormalized: np.ndarray) -> np.ndarray:
    c = float(np.max(log_unnormalized))
    w = np.exp(log_unnormalized - c)
    s = float(np.sum(w))
    if s <= 0:
        raise RuntimeError("Posterior weights degenerate.")
    return w / s


def log_prior_uniform_discrete(n: int) -> np.ndarray:
    return np.full(n, -np.log(n))


def posterior_entropy(p: np.ndarray, eps: float = 1e-300) -> float:
    """Shannon entropy H[p] in nats."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def sequential_posterior_from_log_likelihoods(
    log_L_steps: np.ndarray,
    log_p0: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Sequential Bayes on discrete support; returns final posterior and trace."""
    T, N = log_L_steps.shape
    if log_p0 is None:
        log_p0 = log_prior_uniform_discrete(N)
    log_unnorm = np.array(log_p0, dtype=np.float64)
    p_trace: list[np.ndarray] = [normalize_log_weights(log_unnorm)]
    for t in range(T):
        log_unnorm = log_unnorm + log_L_steps[t]
        p_trace.append(normalize_log_weights(log_unnorm))
    return p_trace[-1], p_trace


def posterior_after_gaussian_observations(
    f_steps: np.ndarray,
    y_steps: np.ndarray,
    sigma_y: float,
    log_p0: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Belief after T Gaussian observations."""
    T, N = f_steps.shape
    log_L_steps = np.zeros((T, N), dtype=np.float64)
    for t in range(T):
        log_L_steps[t] = log_gaussian_observation_density(float(y_steps[t]), f_steps[t], sigma_y)
    return sequential_posterior_from_log_likelihoods(log_L_steps, log_p0)


def single_step_bayes_report(
    y: float,
    f_vals: np.ndarray,
    sigma_y: float,
    log_p0: np.ndarray | None = None,
) -> dict:
    """
    One Bayesian update (T=1) with entropy diagnostics.

    Returns prior/posterior weights, log-likelihood, and ΔH_θ.
    """
    f_vals = np.asarray(f_vals, dtype=np.float64).reshape(-1)
    N = len(f_vals)
    if log_p0 is None:
        log_p0 = log_prior_uniform_discrete(N)
    log_p0 = np.asarray(log_p0, dtype=np.float64).reshape(-1)

    p0 = normalize_log_weights(log_p0)
    H0 = posterior_entropy(p0)
    log_L = log_gaussian_observation_density(float(y), f_vals, sigma_y)
    log_unnorm = log_p0 + log_L
    p1 = normalize_log_weights(log_unnorm)
    H1 = posterior_entropy(p1)

    return {
        "y": float(y),
        "sigma_y": float(sigma_y),
        "f": f_vals.copy(),
        "p0": p0.copy(),
        "p1": p1.copy(),
        "log_L": log_L.copy(),
        "H_prior": H0,
        "H_posterior": H1,
        "delta_H": H0 - H1,
    }


def posterior_mean_mk(p: np.ndarray, M_grid: np.ndarray, K_grid: np.ndarray) -> tuple[float, float]:
    """Posterior mean estimates (M̂, K̂) on scalar discrete support (legacy 2-D grid)."""
    p = np.asarray(p, dtype=np.float64)
    M_hat = float(np.sum(p * M_grid))
    K_hat = float(np.sum(p * K_grid))
    return M_hat, K_hat


def posterior_mean_mk_vectors(
    p: np.ndarray,
    M_support: np.ndarray,
    K_support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean per-bus vectors; ``M_support``, ``K_support`` shape ``(N, n_buses)``."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    M_support = np.asarray(M_support, dtype=np.float64)
    K_support = np.asarray(K_support, dtype=np.float64)
    M_hat = np.sum(p[:, None] * M_support, axis=0)
    K_hat = np.sum(p[:, None] * K_support, axis=0)
    return M_hat, K_hat


def build_mk_grid(
    M_lower: float,
    M_upper: float,
    K_lower: float,
    K_upper: float,
    grid_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tensor-product uniform grid on [M_lower,M_upper]×[K_lower,K_upper].

    Returns:
        grid: (N, 2) array of (M, K) pairs
        M_grid: (N,) M values
        K_grid: (N,) K values
    """
    Ms = np.linspace(M_lower, M_upper, grid_side)
    Ks = np.linspace(K_lower, K_upper, grid_side)
    M_mesh, K_mesh = np.meshgrid(Ms, Ks, indexing="ij")
    M_grid = M_mesh.ravel()
    K_grid = K_mesh.ravel()
    grid = np.column_stack([M_grid, K_grid])
    return grid, M_grid, K_grid


def sample_mk_prior(
    M_lower: float,
    M_upper: float,
    K_lower: float,
    K_upper: float,
    n: int,
    rng: np.random.Generator,
    *,
    n_buses: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample θ=(M,K) from independent uniform priors.

    Returns arrays of shape ``(n, n_buses)``: each row is one system with
    **joint** per-bus samples (not separate M-only / K-only loops).
    """
    M = rng.uniform(M_lower, M_upper, size=(n, n_buses))
    K = rng.uniform(K_lower, K_upper, size=(n, n_buses))
    return M, K


def log_rollout_likelihood(
    y_seq: np.ndarray,
    f_seq_per_theta: np.ndarray,
    sigma_y: float,
) -> float:
    """
    log p(h_T | θ) = Σ_t log p(y_t | θ, ξ_t) on discrete θ index.

    Args:
        y_seq: (T,) observations
        f_seq_per_theta: (T,) noiseless F(θ, ξ_t) for one θ
    """
    total = 0.0
    for t, y in enumerate(y_seq):
        log_L = log_gaussian_observation_density(
            float(y), np.array([f_seq_per_theta[t]]), sigma_y,
        )
        total += float(log_L[0])
    return total


def log_likelihood_sequence(
    y_seq: np.ndarray,
    f_seq: np.ndarray,
    sigma_y: float,
) -> float:
    """log p(h_T | θ, ξ_{1:T}) = Σ_t log N(y_t | F(θ, ξ_t), σ_y)."""
    return _log_likelihood_seq(y_seq, f_seq, sigma_y)


def spce_total_from_log_likelihoods(
    log_p_positive: float,
    log_p_contrastive: np.ndarray,
) -> float:
    """
  Total sPCE / PCE lower bound (same as DAD ``PriorContrastiveEstimation.loss``).

  ``log_p_positive`` = log p(h_T | θ_0, π);
  ``log_p_contrastive`` = length-L vector of log p(h_T | θ_ℓ, π) for ℓ=1…L.
    """
    log_denom = _log_mean_exp(np.concatenate([[log_p_positive], np.asarray(log_p_contrastive).ravel()]))
    return float(log_p_positive - log_denom)


def spce_total_from_f_tensor(
    y_seq: np.ndarray,
    f_tensor: np.ndarray,
    sigma_y: float,
    positive_idx: int = 0,
) -> float:
    """
    Total sPCE from noiseless F(θ_ℓ, ξ) and fixed observations ``y_seq``.

    Args:
        y_seq: (T,) realized observations from the θ_0 rollout (fixed across contrastives)
        f_tensor: (L+1, T) noiseless F values; row ``positive_idx`` is θ_0
    """
    log_terms = [
        log_likelihood_sequence(y_seq, f_tensor[ell], sigma_y)
        for ell in range(f_tensor.shape[0])
    ]
    return spce_total_from_log_likelihoods(log_terms[positive_idx], np.delete(np.array(log_terms), positive_idx))


def spce_step_from_f(
    y: float,
    f_positive: np.ndarray,
    f_contrastive: np.ndarray,
    sigma_y: float,
) -> float:
    """
    One-step sPCE conditional EIG on discrete θ support.

    Args:
        y: observed scalar
        f_positive: (N,) F(θ_n, ξ) under true θ_0
        f_contrastive: (L, N) F under contrastive θ_ℓ on same support grid
    """
    log_L0 = log_gaussian_observation_density(y, f_positive, sigma_y)
    log_terms = [float(_log_sum_exp(log_L0))]
    for ell in range(f_contrastive.shape[0]):
        log_L = log_gaussian_observation_density(y, f_contrastive[ell], sigma_y)
        log_terms.append(float(_log_sum_exp(log_L)))
    log_denom = _log_mean_exp(np.array(log_terms))
    return log_terms[0] - log_denom


def spce_step_scalar_likelihoods(
    log_L_positive: float,
    log_L_contrastive: np.ndarray,
) -> float:
    """
    One-step sPCE: log p(y|θ_0,ξ) − log((1/(L+1)) Σ_ℓ p(y|θ_ℓ,ξ)).

    Each argument is a **scalar** log-likelihood at one θ_ℓ (not max over a grid).
    """
    return spce_total_from_log_likelihoods(log_L_positive, log_L_contrastive)


def log_likelihood_scalar(y: float, f_t: float, sigma_y: float) -> float:
    return float(log_gaussian_observation_density(float(y), np.array([float(f_t)]), sigma_y)[0])


def predict_f_sequence(
    sim,
    M: np.ndarray,
    K: np.ndarray,
    catalog,
    sequence: list[int],
    *,
    state0=None,
) -> np.ndarray:
    """Noiseless F(θ, ξ_t) along a design sequence (state carried between steps)."""
    state = state0
    f_seq: list[float] = []
    for a in sequence:
        y, state = sim.simulate_step(M, K, catalog[int(a)], state, add_noise=None)
        f_seq.append(float(y))
    return np.asarray(f_seq, dtype=np.float64)


def sample_contrastive_mk(
    cfg,
    L: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample L contrastive θ=(M,K) rows from the same uniform prior as data generation."""
    sw = cfg.swing
    n_buses = cfg.N
    M, K = sample_mk_prior(
        float(sw["M_lower"]), float(sw["M_upper"]),
        float(sw["K_lower"]), float(sw["K_upper"]),
        L, rng, n_buses=n_buses,
    )
    return M, K


def _log_likelihood_seq(y_seq: np.ndarray, f_seq: np.ndarray, sigma_y: float) -> float:
    total = 0.0
    for y, f_t in zip(y_seq, f_seq):
        total += float(log_gaussian_observation_density(float(y), np.array([float(f_t)]), sigma_y)[0])
    return total


def _log_mean_exp(log_vals: np.ndarray) -> float:
    c = float(np.max(log_vals))
    return c + float(np.log(np.mean(np.exp(log_vals - c))))


def _log_sum_exp(log_vals: np.ndarray) -> float:
    c = float(np.max(log_vals))
    return c + float(np.log(np.sum(np.exp(log_vals - c))))

