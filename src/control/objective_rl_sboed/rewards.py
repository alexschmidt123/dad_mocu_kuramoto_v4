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


def dad_rewards(u_path: list[float] | np.ndarray) -> RewardTrace:
    """Terminal-only DAD rewards: r_t=0 for t<T, r_T=-u_T."""
    u = [float(x) for x in u_path]
    if len(u) < 2:
        raise ValueError("u_path must include u_0 and at least u_1")
    t_horizon = len(u) - 1
    rewards = [0.0] * (t_horizon - 1) + [-u[-1]]
    return RewardTrace(method="DAD", u_path=tuple(u), rewards=tuple(rewards), gamma=GAMMA)


def rl_sboed_rewards(u_path: list[float] | np.ndarray) -> RewardTrace:
    """Dense stepwise rewards r_t = u_{t-1} - u_t with gamma=1."""
    u = [float(x) for x in u_path]
    if len(u) < 2:
        raise ValueError("u_path must include u_0 and at least u_1")
    rewards = [u[t - 1] - u[t] for t in range(1, len(u))]
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


def verify_rl_sboed_rollout(u_path: list[float] | np.ndarray, tol: float = TELESCOPE_TOL) -> RewardTrace:
    trace = rl_sboed_rewards(u_path)
    assert_telescoping(trace, tol=tol)
    if abs(trace.gamma - 1.0) > 0.0:
        raise AssertionError("RL-sBOED requires gamma=1")
    return trace
