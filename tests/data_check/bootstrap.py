"""Bootstrap confidence intervals for information_redundancy acceptance criteria."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lower: float
    upper: float
    confidence_level: float
    n_replicates: int

    @property
    def passes_threshold(self) -> bool:
        return self.lower > 0.0


def bootstrap_mean_ci(
    samples: np.ndarray,
    *,
    n_replicates: int = 1000,
    confidence_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Percentile bootstrap CI for the sample mean."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("bootstrap_mean_ci requires at least one sample")
    if rng is None:
        rng = np.random.default_rng(0)
    n = x.size
    means = np.empty(n_replicates, dtype=np.float64)
    for i in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(x[idx]))
    alpha = 1.0 - confidence_level
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(
        mean=float(np.mean(x)),
        lower=float(lo),
        upper=float(hi),
        confidence_level=float(confidence_level),
        n_replicates=int(n_replicates),
    )


def bootstrap_mean_ci_threshold(
    samples: np.ndarray,
    threshold: float,
    *,
    n_replicates: int = 1000,
    confidence_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[BootstrapCI, bool]:
    """Return CI and whether the lower bound exceeds ``threshold``."""
    ci = bootstrap_mean_ci(
        samples,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        rng=rng,
    )
    return ci, bool(ci.lower > threshold)
