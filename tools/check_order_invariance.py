"""Check fixed-design order invariance for objective SBOED banks."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

from src.config import SBOEDConfig, load_config, with_step_number
from src.objectives.mocu.context import (
    GLOBAL_SEED,
    build_context_from_config,
    observe_compressed,
    terminal_u_ctrl,
    update_posterior_vector,
)
from src.control.posterior_ctrl import normalize_log_weights


def make_context(config: str, max_t: int, n_obs: int, sigma: float, out: Path):
    cfg0 = with_step_number(load_config(config), max_t)
    raw = copy.deepcopy(cfg0.raw)
    raw.setdefault("observation", {})["N_obs"] = n_obs
    raw["observation"]["noise_sigma"] = sigma
    raw.setdefault("control_safety_calibration", {})["mode"] = "config"
    raw["control_safety_calibration"]["enabled"] = False
    cfg = SBOEDConfig(raw=raw, config_path=cfg0.config_path)
    return build_context_from_config(
        cfg, ensure_bank=True, smoke=True, out_dir=out / "_context"
    )


def run(ctx, horizons: range, n_cases: int, seed: int):
    rows = []
    systems = ctx.validation_systems
    for T in horizons:
        max_w_diff = 0.0
        max_u_diff = 0.0
        step_noise_u_diffs = []
        for case in range(n_cases):
            sid = case % len(systems)
            system = systems[sid]
            seq_rng = np.random.default_rng(seed + case * 1009)
            seq = seq_rng.choice(ctx.n_actions, size=T, replace=False).tolist()

            # Couple one observation to each action, then only change order.
            action_y = {}
            for action in seq:
                action_y[action] = observe_compressed(
                    system,
                    action,
                    sigma_y=ctx.sigma_y,
                    n_obs=ctx.n_obs,
                    global_seed=GLOBAL_SEED,
                    theta_id=sid,
                    rollout_id=case,
                    step=0,
                )
            log_f = ctx.log_p0.copy()
            for action in seq:
                log_f = update_posterior_vector(
                    ctx, log_f, action, action_y[action]
                )
            log_r = ctx.log_p0.copy()
            for action in reversed(seq):
                log_r = update_posterior_vector(
                    ctx, log_r, action, action_y[action]
                )
            w_diff = float(
                np.max(
                    np.abs(
                        normalize_log_weights(log_f)
                        - normalize_log_weights(log_r)
                    )
                )
            )
            u_diff = abs(terminal_u_ctrl(ctx, log_f) - terminal_u_ctrl(ctx, log_r))
            max_w_diff = max(max_w_diff, w_diff)
            max_u_diff = max(max_u_diff, u_diff)

            # Current code attaches noise to (step, action). Reversing the order
            # therefore changes the observations, not just their multiplication.
            log_sf = ctx.log_p0.copy()
            for step, action in enumerate(seq):
                y = observe_compressed(
                    system,
                    action,
                    sigma_y=ctx.sigma_y,
                    n_obs=ctx.n_obs,
                    global_seed=GLOBAL_SEED,
                    theta_id=sid,
                    rollout_id=case,
                    step=step,
                )
                log_sf = update_posterior_vector(ctx, log_sf, action, y)
            log_sr = ctx.log_p0.copy()
            for step, action in enumerate(reversed(seq)):
                y = observe_compressed(
                    system,
                    action,
                    sigma_y=ctx.sigma_y,
                    n_obs=ctx.n_obs,
                    global_seed=GLOBAL_SEED,
                    theta_id=sid,
                    rollout_id=case,
                    step=step,
                )
                log_sr = update_posterior_vector(ctx, log_sr, action, y)
            step_noise_u_diffs.append(
                abs(terminal_u_ctrl(ctx, log_sf) - terminal_u_ctrl(ctx, log_sr))
            )
        diffs = np.asarray(step_noise_u_diffs)
        rows.append(
            {
                "system": ctx.system,
                "T": T,
                "n_cases": n_cases,
                "same_pairs_max_posterior_diff": max_w_diff,
                "same_pairs_max_u_ctrl_diff": max_u_diff,
                "step_noise_mean_abs_u_ctrl_diff": float(diffs.mean()),
                "step_noise_max_abs_u_ctrl_diff": float(diffs.max()),
                "step_noise_fraction_u_ctrl_different": float(np.mean(diffs > 1e-12)),
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--N_obs", type=int, default=120)
    p.add_argument("--noise_sigma", type=float, default=0.005)
    p.add_argument("--cases", type=int, default=64)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = make_context(args.config, 10, args.N_obs, args.noise_sigma, out)
    rows = run(ctx, range(2, 11), args.cases, 9173)
    csv_path = out / f"{ctx.system}_order_invariance.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / f"{ctx.system}_order_invariance.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
