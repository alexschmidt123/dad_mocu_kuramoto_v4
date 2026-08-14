"""Statistics helpers for adaptive-advantage diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def bootstrap_mean_ci(
    x: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = rng.integers(0, x.size, size=x.size)
        means[b] = float(np.mean(x[idx]))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "mean": float(np.mean(x)),
        "se": float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0,
        "ci_low": lo,
        "ci_high": hi,
    }


def paired_delta_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Delta = mean(a - b); positive means a worse (higher u) than b if u is cost."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("paired arrays must match")
    d = a - b
    stats = bootstrap_mean_ci(d, n_boot=n_boot, seed=seed)
    better = float(np.mean(d > 0.0))
    equal = float(np.mean(np.isclose(d, 0.0)))
    worse = float(np.mean(d < 0.0))
    # For Delta = J_myopic - J_planning: PASS if CI low > 0
    lo, hi = stats["ci_low"], stats["ci_high"]
    if lo > 0:
        verdict = "PASS"
    elif hi < 0:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"
    return {
        **stats,
        "paired_mean_gap": stats["mean"],
        "frac_a_higher": better,
        "frac_equal": equal,
        "frac_a_lower": worse,
        "ci_verdict": verdict,
    }


def discrete_entropy(values: np.ndarray) -> float:
    v = np.asarray(values).reshape(-1)
    _, counts = np.unique(v, return_counts=True)
    p = counts.astype(np.float64) / float(counts.sum())
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def mutual_information_binned(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_bins: int,
    y_bins: int,
) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 4:
        return float("nan")
    # Digitize with unique-aware bin edges.
    def _edges(arr: np.ndarray, k: int) -> np.ndarray:
        qs = np.linspace(0.0, 1.0, int(k) + 1)
        e = np.unique(np.quantile(arr, qs))
        if e.size < 2:
            e = np.array([arr.min() - 1e-12, arr.max() + 1e-12])
        return e

    xe = _edges(x, x_bins)
    ye = _edges(y, y_bins)
    xi = np.clip(np.digitize(x, xe[1:-1], right=True), 0, len(xe) - 2)
    yi = np.clip(np.digitize(y, ye[1:-1], right=True), 0, len(ye) - 2)
    n_x = int(xe.size - 1)
    n_y = int(ye.size - 1)
    joint = np.zeros((n_x, n_y), dtype=np.float64)
    for i, j in zip(xi, yi):
        joint[i, j] += 1.0
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mi = 0.0
    for i in range(n_x):
        for j in range(n_y):
            if joint[i, j] <= 0 or px[i, 0] <= 0 or py[0, j] <= 0:
                continue
            mi += joint[i, j] * np.log(joint[i, j] / (px[i, 0] * py[0, j]))
    return float(mi)


def interpret_delta(name: str, ci_verdict: str) -> str:
    return f"{name}:{ci_verdict}"
