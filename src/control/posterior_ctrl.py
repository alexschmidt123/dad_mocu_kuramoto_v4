"""Posterior → Yoon IBR terminal operator ψ*(h_T).

Yoon terminology (TSP 2013):
  ψ_θ*  = model-specific optimal operator  (= min safe support for θ)
  ψ*    = IBR robust operator
  MOCU  = E[C_θ(ψ*) - C_θ(ψ_θ*)]

Hard-safety IBR reduction used here:
  ψ*(w) = max { ψ_{θ_n}* : w_n > 0 }

On disk the bank of ψ_θ* values is ``psi_star.npy`` (legacy ``U.npy``
migrated automatically). In memory, arrays may still be named ``U_support``
as a temporary alias for ψ_n*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

RobustRule = Literal["ibr_max", "quantile"]


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


def ibr_max_u_ctrl(
    U_bank: np.ndarray,
    weights: np.ndarray,
    *,
    weight_eps: float = 1e-12,
) -> float:
    """Yoon IBR under hard safety: max U_n on positive-weight posterior support."""
    v = np.asarray(U_bank, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError("empty U_bank for IBR max")
    if v.shape != w.shape:
        raise ValueError("U_bank and weights must have the same shape")
    w = np.clip(w, 0.0, None)
    s = float(np.sum(w))
    if s <= 0.0:
        raise ValueError("weights must sum to a positive value")
    w = w / s
    mask = w > float(weight_eps)
    if not np.any(mask):
        mask = w > 0.0
    if not np.any(mask):
        return float(np.max(v))
    return float(np.max(v[mask]))


@dataclass(frozen=True)
class TerminalControlRule:
    """
    Common terminal rule for all objective-based methods.

    ``robust_rule="ibr_max"`` (Yoon IBR / primary MOCU definition):

        u_ctrl = max { U_n : w_n > 0 }

    ``robust_rule="quantile"`` (chance-constrained / legacy):

        u_ctrl = Q_{1-α}(U|w) + margin   (optionally snap_up)
    """

    alpha: float = 0.05
    margin: float = 0.0
    u_candidates: tuple[float, ...] = ()
    snap_up: bool = True
    robust_rule: RobustRule = "quantile"

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
            robust_rule=self.robust_rule,
        )

    def to_dict(self) -> dict[str, Any]:
        if self.robust_rule == "ibr_max":
            formula = "max{U_n : w_n > 0}  (Yoon IBR / hard safety)"
        elif self.snap_up:
            formula = "snap_up(Q_{1-alpha}(U|w) + margin)"
        else:
            formula = "Q_{1-alpha}(U|w) + margin"
        return {
            "alpha": float(self.alpha),
            "margin": float(self.margin),
            "quantile_level": float(self.quantile_level),
            "u_candidates": list(self.u_candidates),
            "snap_up": bool(self.snap_up),
            "robust_rule": str(self.robust_rule),
            "rule": formula,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TerminalControlRule:
        cands = tuple(float(x) for x in (raw.get("u_candidates") or []))
        rule = str(raw.get("robust_rule", "quantile")).strip().lower()
        if rule in {"ibr", "ibr_max", "max", "yoon_ibr"}:
            robust: RobustRule = "ibr_max"
        else:
            robust = "quantile"
        return cls(
            alpha=float(raw.get("alpha", 0.05)),
            margin=float(raw.get("margin", 0.0)),
            u_candidates=cands,
            snap_up=bool(raw.get("snap_up", True)),
            robust_rule=robust,
        )


@dataclass(frozen=True)
class ControlDecision:
    """Shared posterior → control mapping used by all methods.

    ``u_quantile`` is Q_{1-α}(U|w) (diagnostic; equals u_ctrl under quantile rule).
    ``u_raw`` is the continuous pre-snap quantity.
    ``u_ctrl`` is the primary operational command.
    ``u_ctrl_snapped`` is always the historical snap_up diagnostic.
    """

    u_quantile: float
    u_raw: float
    u_ctrl: float
    u_ctrl_snapped: float
    robust_rule: str = "quantile"


def posterior_control_decision(
    U_bank: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    *,
    margin: float = 0.0,
    u_grid: Sequence[float] | np.ndarray | None = None,
    snap_up: bool = True,
    robust_rule: str = "quantile",
) -> ControlDecision:
    """Compute terminal control; primary ``u_ctrl`` follows ``robust_rule``."""
    rule = str(robust_rule or "quantile").strip().lower()
    use_ibr = rule in {"ibr", "ibr_max", "max", "yoon_ibr"}

    if use_ibr:
        u_ibr = ibr_max_u_ctrl(U_bank, weights)
        if u_grid is not None and len(list(u_grid)) > 0:
            u_snapped = snap_up_to_grid(u_ibr, u_grid)
        else:
            u_snapped = float(u_ibr)
        # IBR values already lie on the U-bank / candidate set; snap is diagnostic.
        u_ctrl = float(u_ibr)
        return ControlDecision(
            u_quantile=float(u_ibr),
            u_raw=float(u_ibr),
            u_ctrl=u_ctrl,
            u_ctrl_snapped=float(u_snapped),
            robust_rule="ibr_max",
        )

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
        robust_rule="quantile",
    )


def compute_u_ctrl(
    U_bank: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    margin: float = 0.0,
    u_grid: Sequence[float] | np.ndarray | None = None,
    snap_up: bool = True,
    robust_rule: str = "quantile",
) -> float:
    """Shared primary terminal control used by all objective-based methods."""
    return posterior_control_decision(
        U_bank,
        weights,
        alpha,
        margin=margin,
        u_grid=u_grid,
        snap_up=snap_up,
        robust_rule=robust_rule,
    ).u_ctrl


def compute_u_ctrl_snapped(
    U_bank: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    margin: float = 0.0,
    u_grid: Sequence[float] | np.ndarray | None = None,
    robust_rule: str = "quantile",
) -> float:
    """Historical snap_up diagnostic only (not the primary objective)."""
    return posterior_control_decision(
        U_bank,
        weights,
        alpha,
        margin=margin,
        u_grid=u_grid,
        snap_up=True,
        robust_rule=robust_rule,
    ).u_ctrl_snapped


def posterior_safe_u_ctrl(
    U_bank: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    *,
    margin: float = 0.0,
    u_grid: Sequence[float] | np.ndarray | None = None,
    snap_up: bool = True,
    robust_rule: str = "quantile",
) -> float:
    """
    Posterior terminal control:

        ibr_max:  u_ctrl = max { U_n : w_n > 0 }
        quantile: u_ctrl = Q_{1-α}(U|w) + margin  (optionally snapped)
    """
    return compute_u_ctrl(
        U_bank,
        weights,
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        snap_up=snap_up,
        robust_rule=robust_rule,
    )


def posterior_u_raw(
    U_bank: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    *,
    margin: float = 0.0,
    robust_rule: str = "quantile",
) -> float:
    """Continuous pre-snap control (legacy name)."""
    return posterior_control_decision(
        U_bank,
        weights,
        alpha,
        margin=margin,
        u_grid=None,
        snap_up=False,
        robust_rule=robust_rule,
    ).u_raw


def ocu(
    psi_theta_star: float | np.ndarray,
    psi_star: float,
) -> float | np.ndarray:
    """Yoon OCU: C(ψ*) - C(ψ_θ*) under hard-safety cost (= ψ* - ψ_θ*)."""
    return np.asarray(psi_star, dtype=np.float64) - np.asarray(
        psi_theta_star, dtype=np.float64
    )


def belief_mocu(
    psi_star_bank: np.ndarray,
    weights: np.ndarray,
    psi_star: float,
) -> float:
    """Yoon belief MOCU = E_w[OCU] = E_w[ψ* - ψ_θ*] for fixed robust operator ψ*."""
    v = np.asarray(psi_star_bank, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.clip(w, 0.0, None)
    s = float(np.sum(w))
    if s <= 0.0:
        raise ValueError("weights must sum to a positive value")
    w = w / s
    return float(np.sum(w * ocu(v, float(psi_star))))


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
