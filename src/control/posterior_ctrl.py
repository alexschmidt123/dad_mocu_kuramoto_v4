"""Posterior → terminal control decision u_ctrl(h_T)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    """
    Weighted quantile of ``values`` under ``weights``.

    Uses the inverse of the weighted empirical CDF: smallest ``v`` with
    cumulative weight ≥ ``q``.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError("empty values for weighted quantile")
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    w = np.clip(w, 0.0, None)
    s = float(np.sum(w))
    if s <= 0.0:
        raise ValueError("weights must sum to a positive value")
    w = w / s
    q = float(np.clip(q, 0.0, 1.0))
    order = np.argsort(v, kind="mergesort")
    v_sorted = v[order]
    cdf = np.cumsum(w[order])
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = min(max(idx, 0), v_sorted.size - 1)
    return float(v_sorted[idx])


def snap_up_to_grid(u: float, u_grid: Sequence[float] | np.ndarray) -> float:
    """Smallest grid point ≥ u; if none, return max(grid)."""
    g = np.asarray(u_grid, dtype=np.float64).reshape(-1)
    if g.size == 0:
        return float(u)
    ok = g[g >= float(u) - 1e-15]
    if ok.size:
        return float(ok[0])
    return float(g[-1])


@dataclass(frozen=True)
class TerminalControlRule:
    """
    Common terminal rule for all methods:

        u_ctrl = snap_up( Q_{1-α}(U_bank | w) + margin )
    """

    alpha: float = 0.05
    margin: float = 0.0
    u_candidates: tuple[float, ...] = ()

    @property
    def quantile_level(self) -> float:
        return 1.0 - float(self.alpha)

    def apply(self, U_bank: np.ndarray, weights: np.ndarray) -> float:
        return posterior_safe_u_ctrl(
            U_bank,
            weights,
            self.alpha,
            margin=self.margin,
            u_grid=self.u_candidates if self.u_candidates else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "margin": float(self.margin),
            "quantile_level": float(self.quantile_level),
            "u_candidates": list(self.u_candidates),
            "rule": "snap_up(Q_{1-alpha}(U|w) + margin)",
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TerminalControlRule:
        cands = tuple(float(x) for x in (raw.get("u_candidates") or []))
        return cls(
            alpha=float(raw.get("alpha", 0.05)),
            margin=float(raw.get("margin", 0.0)),
            u_candidates=cands,
        )


@dataclass(frozen=True)
class ControlDecision:
    """Shared posterior → control mapping used by all methods.

    ``u_quantile`` is Q_{1-α}(U|w).
    ``u_raw`` is the continuous pre-snap command ``u_quantile + margin``.
    ``u_ctrl`` is the operational snapped command (primary evaluation metric).
    """

    u_quantile: float
    u_raw: float
    u_ctrl: float


def posterior_control_decision(
    U_bank: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    *,
    margin: float = 0.0,
    u_grid: Sequence[float] | np.ndarray | None = None,
) -> ControlDecision:
    """Compute both continuous ``u_raw`` and operational ``u_ctrl``."""
    q = 1.0 - float(alpha)
    u_quantile = float(weighted_quantile(U_bank, weights, q))
    u_raw = u_quantile + float(margin)
    if u_grid is not None and len(list(u_grid)) > 0:
        u_ctrl = snap_up_to_grid(u_raw, u_grid)
    else:
        u_ctrl = float(u_raw)
    return ControlDecision(u_quantile=u_quantile, u_raw=float(u_raw), u_ctrl=float(u_ctrl))


def posterior_safe_u_ctrl(
    U_bank: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    *,
    margin: float = 0.0,
    u_grid: Sequence[float] | np.ndarray | None = None,
) -> float:
    """
    Posterior-safe control:

        u0 = min{ u : Σ_n w_n 1{U_n ≤ u} ≥ 1 − α }
        u_ctrl = snap_up(u0 + margin)   if u_grid given, else u0 + margin
    """
    return posterior_control_decision(
        U_bank, weights, alpha, margin=margin, u_grid=u_grid
    ).u_ctrl


def posterior_u_raw(
    U_bank: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    *,
    margin: float = 0.0,
) -> float:
    """Continuous pre-snap control ``Q_{1-α}(U|w) + margin`` (training diagnostic)."""
    return posterior_control_decision(
        U_bank, weights, alpha, margin=margin, u_grid=None
    ).u_raw


def normalize_log_weights(log_w: np.ndarray) -> np.ndarray:
    """Stable softmax of log-weights; returns probabilities summing to 1."""
    x = np.asarray(log_w, dtype=np.float64).reshape(-1)
    c = float(np.max(x))
    w = np.exp(x - c)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        raise RuntimeError("Posterior weights degenerate.")
    return w / s


def posterior_ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.clip(w, 0.0, None)
    s = float(np.sum(w))
    if s <= 0.0:
        return 0.0
    w = w / s
    return float(1.0 / np.sum(w * w))


def weighted_cdf_at(values: np.ndarray, weights: np.ndarray, u: float) -> float:
    """Σ w_n 1{U_n ≤ u}."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.clip(w, 0.0, None)
    s = float(np.sum(w))
    if s <= 0.0:
        return float("nan")
    w = w / s
    return float(np.sum(w[v <= float(u)]))
