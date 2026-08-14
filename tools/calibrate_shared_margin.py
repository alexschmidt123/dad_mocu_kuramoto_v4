"""Calibrate one fixed objective-control margin across several horizons.

This diagnostic uses strictly off-support validation systems and the banked
true required control ``u_req``.  It fixes alpha=0.10, rolls out no-repeat
random probe sequences once up to max(T), and chooses the smallest single
margin whose empirical safety is at least 0.90 at every requested horizon.

Results are written under reports/ and can then be copied into the system
YAML before a production sweep.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from src.config import SBOEDConfig, load_config, with_step_number
from src.objectives.mocu.context import (
    build_context_from_config,
    observe_compressed,
    update_posterior_vector,
)
from src.control.posterior_ctrl import normalize_log_weights, posterior_safe_u_ctrl


def _config_for_calibration(path: str, max_horizon: int) -> SBOEDConfig:
    base = with_step_number(load_config(path), max_horizon)
    raw = copy.deepcopy(base.raw)
    raw.setdefault("control", {})["alpha"] = 0.10
    raw["control"]["safety_margin"] = 0.0
    raw["control_safety_calibration"] = {
        **dict(raw.get("control_safety_calibration") or {}),
        "enabled": False,
        "mode": "config",
    }
    return SBOEDConfig(raw=raw, config_path=base.config_path)


def calibrate(
    config_path: str,
    horizons: tuple[int, ...],
    *,
    n_rollouts: int,
    seed: int,
    n_obs: int,
    noise_sigma: float,
) -> dict:
    cfg = _config_for_calibration(config_path, max(horizons))
    cfg.raw.setdefault("observation", {})["N_obs"] = int(n_obs)
    cfg.raw["observation"]["noise_sigma"] = float(noise_sigma)
    out_dir = (
        Path("reports/control_horizon_consistency")
        / f"_context_{Path(config_path).stem}"
    )
    ctx = build_context_from_config(
        cfg, ensure_bank=True, smoke=True, out_dir=out_dir
    )
    systems = ctx.validation_systems
    if not systems:
        raise RuntimeError("No off-support validation systems")

    margin_step = float(np.min(np.diff(ctx.u_grid)))
    margin_grid = np.arange(0.0, float(ctx.u_grid[-1]) + margin_step / 2, margin_step)
    safe = {t: np.zeros(len(margin_grid), dtype=np.int64) for t in horizons}
    total = {t: 0 for t in horizons}
    rng = np.random.default_rng(seed)

    for rid in range(n_rollouts):
        sid = rid % len(systems)
        system = systems[sid]
        log_w = ctx.log_p0.copy()
        available = list(range(ctx.n_actions))
        rng.shuffle(available)
        for step, action in enumerate(available[: max(horizons)], start=1):
            y = observe_compressed(
                system,
                action,
                sigma_y=ctx.sigma_y,
                n_obs=ctx.n_obs,
                global_seed=seed,
                theta_id=sid,
                rollout_id=rid,
                step=step - 1,
            )
            log_w = update_posterior_vector(ctx, log_w, action, y)
            if step not in safe:
                continue
            weights = normalize_log_weights(log_w)
            for j, margin in enumerate(margin_grid):
                u = posterior_safe_u_ctrl(
                    ctx.U_support,
                    weights,
                    0.10,
                    margin=float(margin),
                    u_grid=ctx.u_grid,
                    snap_up=True,
                )
                safe[step][j] += int(u + 1e-12 >= float(system["u_req"]))
            total[step] += 1

    rates = {t: safe[t] / max(total[t], 1) for t in horizons}
    valid = np.ones(len(margin_grid), dtype=bool)
    for t in horizons:
        valid &= rates[t] >= 0.90
    if not np.any(valid):
        raise RuntimeError("No margin reaches 90% safety at every horizon")
    index = int(np.flatnonzero(valid)[0])
    chosen = float(margin_grid[index])
    return {
        "system": ctx.system,
        "alpha": 0.10,
        "margin": chosen,
        "minimum_safety": 0.90,
        "horizons": list(horizons),
        "n_rollouts": n_rollouts,
        "N_obs": n_obs,
        "noise_sigma": noise_sigma,
        "safety_by_horizon": {str(t): float(rates[t][index]) for t in horizons},
        "margin_grid": margin_grid.tolist(),
        "safety_grid": {
            str(t): [float(x) for x in rates[t]] for t in horizons
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--T", default="2,3,4,5")
    p.add_argument("--N_obs", type=int, default=120)
    p.add_argument("--noise_sigma", type=float, default=0.005)
    p.add_argument("--rollouts", type=int, default=512)
    p.add_argument("--seed", type=int, default=2468)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    horizons = tuple(sorted({int(x) for x in args.T.split(",")}))
    result = calibrate(
        args.config,
        horizons,
        n_rollouts=args.rollouts,
        seed=args.seed,
        n_obs=args.N_obs,
        noise_sigma=args.noise_sigma,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
