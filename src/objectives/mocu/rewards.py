"""Reward formulations for DAD and RL-sBOED."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GAMMA = 1.0
TELESCOPE_TOL = 1e-8


@dataclass(frozen=True)
class RewardTrace:
    method: str
    u_path: tuple[float, ...]  # u_0 .. u_T
    rewards: tuple[float, ...]  # length T
    gamma: float = GAMMA

    @property
    def return_sum(self) -> float:
        return float(sum(self.rewards))

    @property
    def terminal_u_ctrl(self) -> float:
        return float(self.u_path[-1])


def safety_aware_control_cost(
    u_ctrl: float,
    u_req: float | None,
    *,
    undercontrol_penalty: float = 0.0,
    violation_penalty: float = 0.0,
) -> float:
    """Energy plus a training-only penalty for unsafe under-control."""
    u = float(u_ctrl)
    if u_req is None:
        return u
    shortfall = max(float(u_req) - u, 0.0)
    violated = float(shortfall > 0.0)
    return (
        u
        + float(undercontrol_penalty) * shortfall
        + float(violation_penalty) * violated
    )


def _cost_path(
    u_path: list[float] | np.ndarray,
    u_req: float | None,
    undercontrol_penalty: float,
    violation_penalty: float,
) -> list[float]:
    return [
        safety_aware_control_cost(
            u,
            u_req,
            undercontrol_penalty=undercontrol_penalty,
            violation_penalty=violation_penalty,
        )
        for u in u_path
    ]


def dad_rewards(
    u_path: list[float] | np.ndarray,
    *,
    u_req: float | None = None,
    undercontrol_penalty: float = 0.0,
    violation_penalty: float = 0.0,
) -> RewardTrace:
    """Terminal goal-oriented DAD reward: energy plus safety penalty."""
    u = [float(x) for x in u_path]
    if len(u) < 2:
        raise ValueError("u_path must include u_0 and at least u_1")
    costs = _cost_path(u, u_req, undercontrol_penalty, violation_penalty)
    t_horizon = len(u) - 1
    rewards = [0.0] * (t_horizon - 1) + [-costs[-1]]
    return RewardTrace(method="DAD", u_path=tuple(u), rewards=tuple(rewards), gamma=GAMMA)


def rl_sboed_rewards(
    u_path: list[float] | np.ndarray,
    *,
    u_req: float | None = None,
    undercontrol_penalty: float = 0.0,
    violation_penalty: float = 0.0,
) -> RewardTrace:
    """Dense potential reward: reduction in safety-aware control cost."""
    u = [float(x) for x in u_path]
    if len(u) < 2:
        raise ValueError("u_path must include u_0 and at least u_1")
    costs = _cost_path(u, u_req, undercontrol_penalty, violation_penalty)
    rewards = [costs[t - 1] - costs[t] for t in range(1, len(costs))]
    return RewardTrace(
        method="RL-sBOED", u_path=tuple(u), rewards=tuple(rewards), gamma=GAMMA
    )


def assert_telescoping(trace: RewardTrace, tol: float = TELESCOPE_TOL) -> None:
    expected = float(trace.u_path[0] - trace.u_path[-1])
    got = float(sum(trace.rewards))
    if abs(got - expected) >= tol:
        raise AssertionError(
            f"RL-sBOED telescope failed: sum(r)={got} vs u0-uT={expected}"
        )


def verify_rl_sboed_rollout(
    u_path: list[float] | np.ndarray,
    tol: float = TELESCOPE_TOL,
    *,
    u_req: float | None = None,
    undercontrol_penalty: float = 0.0,
    violation_penalty: float = 0.0,
) -> RewardTrace:
    trace = rl_sboed_rewards(
        u_path,
        u_req=u_req,
        undercontrol_penalty=undercontrol_penalty,
        violation_penalty=violation_penalty,
    )
    costs = _cost_path(
        u_path, u_req, undercontrol_penalty, violation_penalty
    )
    got = float(sum(trace.rewards))
    expected = float(costs[0] - costs[-1])
    if abs(got - expected) >= tol:
        raise AssertionError(
            f"RL-sBOED cost telescope failed: sum(r)={got} vs C0-CT={expected}"
        )
    if abs(trace.gamma - 1.0) > 0.0:
        raise AssertionError("RL-sBOED requires gamma=1")
    return trace
