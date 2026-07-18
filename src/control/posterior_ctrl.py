"""Posterior → terminal control decision u_ctrl(h_T).

Primary scientific mapping for *new* continuous-control studies:

    u_ctrl = Q_{1-α}(U | w) + margin

Historical / snapped mapping (default for frozen legacy rules):

    u_ctrl = snap_up(Q_{1-α}(U | w) + margin)

``u_ctrl_snapped`` is always available as a diagnostic. ``u_raw`` remains an
alias of the continuous pre-snap quantity for backward compatibility.
"""

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
    Common terminal rule for all methods.

    With ``snap_up=True`` (historical):

        u_ctrl = snap_up( Q_{1-α}(U_bank | w) + margin )

    With ``snap_up=False`` (continuous-control studies):

        u_ctrl = Q_{1-α}(U_bank | w) + margin
    """

    alpha: float = 0.05
    margin: float = 0.0
    u_candidates: tuple[float, ...] = ()
    snap_up: bool = True

    @property
    def quantile_level(self) -> float:
        return 1.0 - float(self.alpha)

    def apply(self, U_bank: np.ndarray, weights: np.ndarray) -> float:
        return compute_u_ctrl(
            U_bank,
            weights,
            alpha=self.alpha,
            margin=self.margin,
            u_grid=self.u_candidates if self.u_candidates else None,
            snap_up=self.snap_up,
        )

    def to_dict(self) -> dict[str, Any]:
        formula = (
            "snap_up(Q_{1-alpha}(U|w) + margin)"
            if self.snap_up
            else "Q_{1-alpha}(U|w) + margin"
        )
        return {
            "alpha": float(self.alpha),
            "margin": float(self.margin),
            "quantile_level": float(self.quantile_level),
            "u_candidates": list(self.u_candidates),
            "snap_up": bool(self.snap_up),
            "rule": formula,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TerminalControlRule:
        cands = tuple(float(x) for x in (raw.get("u_candidates") or []))
        return cls(
            alpha=float(raw.get("alpha", 0.05)),
            margin=float(raw.get("margin", 0.0)),
            u_candidates=cands,
            snap_up=bool(raw.get("snap_up", True)),
        )


@dataclass(frozen=True)
class ControlDecision:
    """Shared posterior → control mapping used by all methods.

    ``u_quantile`` is Q_{1-α}(U|w).
    ``u_raw`` is the continuous quantity ``u_quantile + margin`` (legacy name).
    ``u_ctrl`` is the primary operational command (continuous or snapped).
    ``u_ctrl_snapped`` is always the historical snap_up diagnostic.
    """

    u_quantile: float
    u_raw: float
    u_ctrl: float
    u_ctrl_snapped: float


def posterior_control_decision(
    U_bank: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    *,
    margin: float = 0.0,
    u_grid: Sequence[float] | np.ndarray | None = None,
    snap_up: bool = True,
) -> ControlDecision:
    """Compute continuous and snapped control; primary ``u_ctrl`` follows ``snap_up``."""
    q = 1.0 - float(alpha)
    u_quantile = float(weighted_quantile(U_bank, weights, q))
    u_continuous = u_quantile + float(margin)
    if u_grid is not None and len(list(u_grid)) > 0:
        u_snapped = snap_up_to_grid(u_continuous, u_grid)
    else:
        u_snapped = float(u_continuous)
    u_ctrl = float(u_snapped if snap_up else u_continuous)
    return ControlDecision(
        u_quantile=u_quantile,
        u_raw=float(u_continuous),
        u_ctrl=u_ctrl,
        u_ctrl_snapped=float(u_snapped),
    )


def compute_u_ctrl(
    U_bank: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    margin: float = 0.0,
    u_grid: Sequence[float] | np.ndarray | None = None,
    snap_up: bool = True,
) -> float:
    """Shared primary terminal control used by all objective-based methods."""
    return posterior_control_decision(
        U_bank,
        weights,
        alpha,
        margin=margin,
        u_grid=u_grid,
        snap_up=snap_up,
    ).u_ctrl


def compute_u_ctrl_snapped(
    U_bank: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    margin: float = 0.0,
    u_grid: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Historical snap_up diagnostic only (not the primary objective)."""
    return posterior_control_decision(
        U_bank,
        weights,
        alpha,
        margin=margin,
        u_grid=u_grid,
        snap_up=True,
    ).u_ctrl_snapped


def posterior_safe_u_ctrl(
    U_bank: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    *,
    margin: float = 0.0,
    u_grid: Sequence[float] | np.ndarray | None = None,
    snap_up: bool = True,
) -> float:
    """
    Posterior-safe control:

        continuous: u_ctrl = Q_{1-α}(U|w) + margin
        snapped:    u_ctrl = snap_up(Q_{1-α}(U|w) + margin)
    """
    return compute_u_ctrl(
        U_bank,
        weights,
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        snap_up=snap_up,
    )


def posterior_u_raw(
    U_bank: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    *,
    margin: float = 0.0,
) -> float:
    """Continuous ``Q_{1-α}(U|w) + margin`` (legacy name; equals continuous u_ctrl)."""
    return posterior_control_decision(
        U_bank, weights, alpha, margin=margin, u_grid=None, snap_up=False
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
