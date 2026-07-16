"""Shared keyed-noise rollout path for observability and method evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.control.posterior_ctrl import (
    normalize_log_weights,
    posterior_ess,
    posterior_safe_u_ctrl,
    weighted_quantile,
)
from src.control.terminal_rule import FrozenTerminalRule, observe_with_keyed_noise
from src.rollout import update_log_weights
from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables


def run_keyed_history(
    *,
    system: dict[str, Any],
    theta_id: int,
    rollout_id: int,
    selector: Any,
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    frozen: FrozenTerminalRule,
    horizon: int,
    sigma_y: float,
    global_seed: int,
    rng: np.random.Generator,
    margin_override: float | None = None,
) -> dict[str, Any]:
    """
    One complete T-step history with keyed observation noise.

    Used by observability Random, pilot Random, and policy-robust calibration
    so identical seeds yield identical histories.
    """
    log_w = np.asarray(table_support.log_p0, dtype=np.float64).copy()
    used: set[int] = set()
    seq: list[int] = []
    y_list: list[float] = []
    for step in range(horizon):
        weights = normalize_log_weights(log_w)
        a = int(
            selector.select(
                step=step,
                history_actions=list(seq),
                history_obs=list(y_list),
                used=set(used),
                log_weights=log_w,
                weights=weights,
                rng=rng,
            )
        )
        if a in used:
            raise RuntimeError(f"repeat action {a}")
        y = observe_with_keyed_noise(
            system,
            a,
            sigma_y=sigma_y,
            global_seed=global_seed,
            theta_id=theta_id,
            rollout_id=rollout_id,
            step=step,
        )
        centres = y_sim_last_step_from_tables(table_support, [a])
        log_w = update_log_weights(log_w, y, centres, sigma_y)
        seq.append(a)
        y_list.append(y)
        used.add(a)

    weights = normalize_log_weights(log_w)
    q95 = float(weighted_quantile(U_support, weights, 1.0 - float(frozen.alpha)))
    margin = float(frozen.margin if margin_override is None else margin_override)
    u_ctrl = float(
        posterior_safe_u_ctrl(
            U_support,
            weights,
            frozen.alpha,
            margin=margin,
            u_grid=frozen.u_candidates,
        )
    )
    u_req = float(system["u_req"])
    residual_raw = max(0.0, u_req - q95)
    under = u_ctrl - u_req
    mean_U = float(np.sum(weights * U_support))
    return {
        "sequence": seq,
        "y_obs": y_list,
        "weights": weights,
        "posterior_ess": float(posterior_ess(weights)),
        "max_posterior_weight": float(np.max(weights)),
        "posterior_mean_U": mean_U,
        "posterior_std_U": float(
            np.sqrt(max(np.sum(weights * (U_support - mean_U) ** 2), 0.0))
        ),
        "posterior_quantile": q95,
        "true_u_req": u_req,
        "selected_u_ctrl": u_ctrl,
        "under_control_residual": float(under),
        "raw_residual_r": float(residual_raw),
        "proxy_safe": bool(u_ctrl + 1e-12 >= u_req),
    }
