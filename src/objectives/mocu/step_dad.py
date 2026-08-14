"""Step-DAD (semi-amortized refinement) entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.objectives.mocu.context import ExperimentContext


@dataclass
class StepDADConfig:
    refinement_steps: int = 4
    fantasy_rollouts: int = 16


def step_dad_report(
    ctx: ExperimentContext,
    *,
    n_rollouts: int,
    seed: int,
    device: str,
    config: StepDADConfig | None = None,
) -> dict[str, Any]:
    """Minimal stub so CLI imports resolve after layout recovery."""
    cfg = config or StepDADConfig()
    return {
        "status": "not_implemented_in_recovered_layout",
        "n_rollouts": int(n_rollouts),
        "seed": int(seed),
        "device": str(device),
        "refinement_steps": int(cfg.refinement_steps),
        "fantasy_rollouts": int(cfg.fantasy_rollouts),
        "horizon": int(ctx.horizon),
        "note": "Step-DAD source was lost in layout recovery; use DAD/RL/MoE.",
    }
