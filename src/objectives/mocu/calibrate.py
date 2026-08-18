"""Calibrate (α, margin) on physical banks; write frozen rule beside the data."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import SBOEDConfig, repo_root
from src.objectives.mocu.context import (
    build_context_from_config,
    control_engine_for,
    observe_compressed,
    update_posterior_vector,
)
from src.control.posterior_ctrl import normalize_log_weights, posterior_safe_u_ctrl
from src.control.u_req import ControlSpec


def _cal_cfg(cfg: SBOEDConfig) -> dict[str, Any]:
    return dict(cfg.raw.get("control_safety_calibration") or {})


def frozen_rule_output_path(cfg: SBOEDConfig, root: Path | None = None) -> Path:
    root = root or repo_root()
    raw = str(_cal_cfg(cfg).get("frozen_rule_path", "") or "").strip()
    if not raw:
        from src.banks.power_grid import resolve_dataset_dir

        return resolve_dataset_dir(cfg, root) / "selected_policy_robust_rule.json"
    p = Path(raw)
    return p if p.is_absolute() else root / p


def _rollout_u_ctrl(
    ctx,
    system: dict[str, Any],
    *,
    theta_id: int,
    rollout_id: int,
    alpha: float,
    margin: float,
    rng: np.random.Generator,
) -> float:
    log_w = ctx.log_p0.copy()
    used: set[int] = set()
    for step in range(ctx.horizon):
        avail = [a for a in range(ctx.n_actions) if a not in used]
        a = int(rng.choice(avail))
        used.add(a)
        y = observe_compressed(
            system,
            a,
            sigma_y=ctx.sigma_y,
            n_obs=ctx.n_obs,
            global_seed=int(rng.integers(0, 2**31 - 1)),
            theta_id=theta_id,
            rollout_id=rollout_id,
            step=step,
        )
        log_w = update_posterior_vector(ctx, log_w, a, y)
    w = normalize_log_weights(log_w)
    return float(
        posterior_safe_u_ctrl(
            ctx.U_support, w, alpha, margin=margin, u_grid=ctx.u_grid
        )
    )


def calibrate_terminal_rule(
    cfg: SBOEDConfig,
    *,
    project_root: Path | None = None,
    smoke: bool = False,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Grid-search (α, margin) for empirical true-θ safety on train systems.

    Writes ``selected_policy_robust_rule.json`` to ``frozen_rule_path``.
    """
    root = project_root or repo_root()
    cal = _cal_cfg(cfg)
    out_path = frozen_rule_output_path(cfg, root)

    raw = copy.deepcopy(cfg.raw)
    raw["control_safety_calibration"] = {**cal, "mode": "config"}
    cfg_cal = SBOEDConfig(raw=raw, config_path=cfg.config_path)

    # Rule is written under data/<system>/; ctx.out_dir is only for ExperimentContext.
    # Prefer the caller's stamped experiment folder; never experiments/_*.
    from src.banks.power_grid import resolve_dataset_dir

    ctx_out = Path(out_dir).resolve() if out_dir is not None else resolve_dataset_dir(
        cfg, root
    )
    ctx = build_context_from_config(
        cfg_cal,
        project_root=root,
        ensure_bank=True,
        smoke=smoke,
        out_dir=ctx_out,
        experiment_type="objective_based",
    )
    engine, _spec = control_engine_for(ctx)
    systems = list(ctx.train_systems)
    n_sys = len(systems)
    if n_sys < 4:
        raise RuntimeError("Need at least 4 train systems for safety calibration")

    n_cal = max(2, int(0.7 * n_sys))
    cal_systems = systems[:n_cal]
    val_systems = systems[n_cal:] or systems[-max(1, n_sys // 5) :]

    alpha_grid = tuple(float(x) for x in cal.get("alpha_grid", (0.05, 0.02, 0.01)))
    margin_grid = tuple(
        float(x) for x in cal.get("margin_grid", (0.0, 0.1, 0.2, 0.3))
    )
    n_roll = 32 if smoke else int(cal.get("num_rollouts", 512))
    n_roll = max(n_roll, len(cal_systems))
    seed = int(cal.get("seed", 2468))
    min_cal = float(cal.get("min_calibration_safety_rate", 0.95))
    min_val = float(cal.get("min_validation_safety_rate", 0.95))
    rng = np.random.default_rng(seed)

    def _eval_split(
        split_systems: list[dict[str, Any]], alpha: float, margin: float
    ) -> tuple[float, float]:
        safes: list[float] = []
        us: list[float] = []
        for i in range(n_roll):
            tid = int(i % len(split_systems))
            sys = split_systems[tid]
            u = _rollout_u_ctrl(
                ctx,
                sys,
                theta_id=tid,
                rollout_id=i,
                alpha=alpha,
                margin=margin,
                rng=rng,
            )
            us.append(u)
            M = np.asarray(sys["M"], dtype=np.float64)
            K = np.asarray(sys["K"], dtype=np.float64)
            m = engine.evaluate_one(M, K, float(u))
            safes.append(1.0 if m["safe_total"] >= 0.5 else 0.0)
        return float(np.mean(safes)), float(np.mean(us))

    print(
        f"[calibrate] searching α={list(alpha_grid)} margin={list(margin_grid)} "
        f"rollouts={n_roll} cal={len(cal_systems)} val={len(val_systems)}"
    )
    t0 = time.time()
    candidates: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        for margin in margin_grid:
            cal_safe, cal_u = _eval_split(cal_systems, alpha, margin)
            val_safe, val_u = _eval_split(val_systems, alpha, margin)
            row = {
                "alpha": alpha,
                "margin": margin,
                "cal_safety_rate": cal_safe,
                "val_safety_rate": val_safe,
                "cal_mean_u_ctrl": cal_u,
                "val_mean_u_ctrl": val_u,
            }
            candidates.append(row)
            print(
                f"  α={alpha:.3f} m={margin:.2f}  "
                f"cal_safe={cal_safe:.3f} val_safe={val_safe:.3f}  "
                f"cal_u={cal_u:.3f}"
            )

    feasible = [
        c
        for c in candidates
        if c["cal_safety_rate"] + 1e-12 >= min_cal
        and c["val_safety_rate"] + 1e-12 >= min_val
    ]
    if not feasible:
        print(
            "[calibrate] WARNING: no (α,margin) met safety thresholds; "
            "selecting best available"
        )
        pool = candidates
    else:
        pool = feasible
    best = min(
        pool,
        key=lambda c: (
            -min(c["cal_safety_rate"], c["val_safety_rate"]),
            c["cal_mean_u_ctrl"],
            c["margin"],
        ),
    )
    spec = ControlSpec.from_cfg(cfg)
    rule = {
        "alpha": float(best["alpha"]),
        "margin": float(best["margin"]),
        "u_candidates": list(spec.u_candidates),
        "snap_up": True,
    }
    payload = {
        "rule": rule,
        "mode": "frozen",
        "source": "full_delta_f.calibrate_terminal_rule",
        "calibration": {
            "best": best,
            "elapsed_seconds": time.time() - t0,
            "n_rollouts": n_roll,
            "candidates": candidates,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[calibrate] wrote {out_path}  α={rule['alpha']} margin={rule['margin']} "
        f"({time.time() - t0:.1f}s)"
    )
    return payload


def ensure_calibrated_rule(
    cfg: SBOEDConfig,
    *,
    project_root: Path | None = None,
    smoke: bool = False,
    out_dir: Path | None = None,
) -> Path:
    """Load existing frozen rule or run calibration."""
    root = project_root or repo_root()
    path = frozen_rule_output_path(cfg, root)
    if path.is_file():
        print(f"[calibrate] using existing rule → {path}")
        return path
    calibrate_terminal_rule(cfg, project_root=root, smoke=smoke, out_dir=out_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration did not write {path}")
    return path


def main(argv: list[str] | None = None) -> None:
    import argparse

    from src.config import load_config

    p = argparse.ArgumentParser(description="Calibrate frozen terminal-control rule")
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true", help="Re-run even if rule exists")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    path = frozen_rule_output_path(cfg)
    if args.force and path.is_file():
        path.unlink()
    ensure_calibrated_rule(cfg, smoke=args.smoke)
    print(f"RULE_PATH={path}")


if __name__ == "__main__":
    main()
