"""Shared sequential rollout engine for dad / myopic / fixed / random."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from src.control.cuda_control import CudaControlEngine
from src.control.posterior_ctrl import normalize_log_weights, posterior_safe_u_ctrl
from src.control.u_req import ControlSpec
from src.observations.likelihood import log_gaussian_observation_density
from src.banks.tables import TableThetaSupport, lookup_action_y, y_sim_last_step_from_tables
from src.domains.swing.simulator import system_mk


class ProbeSelector(Protocol):
    def select(
        self,
        *,
        step: int,
        history_actions: list[int],
        history_obs: list[float],
        used: set[int],
        log_weights: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> int: ...


@dataclass
class RolloutResult:
    theta_test_id: int
    sequence: list[int]
    y_obs: list[float]
    log_weights: np.ndarray
    weights: np.ndarray
    u_ctrl: float
    u_req_true: float | None
    excess_control: float | None
    max_rocof: float | None
    frequency_nadir: float | None
    rocof_safe: bool | None
    nadir_safe: bool | None
    safe_total: bool | None
    control_metrics: dict[str, float] = field(default_factory=dict)
    M: Any = None
    K: Any = None


def update_log_weights(
    log_w: np.ndarray,
    y_obs: float,
    centres: np.ndarray,
    sigma_y: float,
) -> np.ndarray:
    log_L = log_gaussian_observation_density(float(y_obs), centres, float(sigma_y))
    return np.asarray(log_w, dtype=np.float64) + log_L


def run_shared_rollout(
    *,
    system: dict[str, Any],
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    selector: ProbeSelector,
    horizon: int,
    n_actions: int,
    sigma_y: float,
    control_spec: ControlSpec,
    rng: np.random.Generator,
    control_engine: CudaControlEngine,
    theta_test_id: int = -1,
    verify_true_safety: bool = True,
) -> RolloutResult:
    """
    Common T-step probe loop. Only the selector differs across methods.

    Probe observations from the offline probe bank (reset-based). Terminal control
    is a separate control-only simulation (probe_amplitude = 0) using the same
    CudaControlEngine as U-bank generation.
    """
    log_w = np.asarray(table_support.log_p0, dtype=np.float64).copy()
    used: set[int] = set()
    seq: list[int] = []
    y_list: list[float] = []

    for t in range(horizon):
        weights = normalize_log_weights(log_w)
        a = int(
            selector.select(
                step=t,
                history_actions=list(seq),
                history_obs=list(y_list),
                used=set(used),
                log_weights=log_w,
                weights=weights,
                rng=rng,
            )
        )
        if a < 0 or a >= n_actions:
            raise ValueError(f"selector returned invalid action {a}")
        if a in used:
            raise ValueError(f"selector returned already-used action {a}")

        y = float(lookup_action_y(system, a))
        centres = y_sim_last_step_from_tables(table_support, [a])
        log_w = update_log_weights(log_w, y, centres, sigma_y)

        seq.append(a)
        y_list.append(y)
        used.add(a)

    weights = normalize_log_weights(log_w)
    u_ctrl = float(
        posterior_safe_u_ctrl(
            U_support,
            weights,
            control_spec.alpha,
            margin=float(getattr(control_spec, "safety_margin", 0.0)),
            u_grid=control_spec.u_candidates,
        )
    )

    u_req_true = float(system["u_req"]) if "u_req" in system else None
    excess = None if u_req_true is None else float(u_ctrl - u_req_true)

    max_rocof = frequency_nadir = None
    rocof_safe = nadir_safe = safe_total = None
    metrics: dict[str, float] = {}
    if verify_true_safety:
        M, K = system_mk(system, control_engine.N)
        metrics = control_engine.evaluate_one(M, K, float(u_ctrl))
        max_rocof = float(metrics["rocof_max"])
        frequency_nadir = float(metrics["delta_f_nadir"])
        rocof_safe = bool(metrics["rocof_safe"] >= 0.5)
        nadir_safe = bool(metrics["nadir_safe"] >= 0.5)
        safe_total = bool(metrics["safe_total"] >= 0.5)

    return RolloutResult(
        theta_test_id=int(theta_test_id),
        sequence=seq,
        y_obs=y_list,
        log_weights=log_w,
        weights=weights,
        u_ctrl=float(u_ctrl),
        u_req_true=u_req_true,
        excess_control=excess,
        max_rocof=max_rocof,
        frequency_nadir=frequency_nadir,
        rocof_safe=rocof_safe,
        nadir_safe=nadir_safe,
        safe_total=safe_total,
        control_metrics=metrics,
        M=system.get("M"),
        K=system.get("K"),
    )


@dataclass
class RandomSelector:
    n_actions: int

    def select(self, *, used: set[int], rng: np.random.Generator, **_kwargs) -> int:
        feasible = [i for i in range(self.n_actions) if i not in used]
        if not feasible:
            raise RuntimeError("no feasible actions remain")
        return int(rng.choice(feasible))


@dataclass
class FixedSelector:
    sequence: list[int]
    _pos: int = 0

    def select(self, *, used: set[int], **_kwargs) -> int:
        if self._pos >= len(self.sequence):
            raise RuntimeError("fixed sequence exhausted")
        a = int(self.sequence[self._pos])
        self._pos += 1
        if a in used:
            raise RuntimeError(f"fixed action {a} already used")
        return a


@dataclass
class FunctionSelector:
    fn: Callable[..., int]

    def select(self, **kwargs) -> int:
        return int(self.fn(**kwargs))
