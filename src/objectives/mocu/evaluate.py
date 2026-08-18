"""Evaluate DAD, RL-sBOED, Myopic, Fixed, Random + true-θ oracle comparison."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.objectives.mocu.context import (
    GLOBAL_SEED,
    ExperimentContext,
    context_report_meta,
    control_engine_for,
    method_display_name,
    observe_compressed,
    posterior_ess,
    terminal_u_ctrl,
    update_posterior_vector,
)
from src.objectives.mocu.diagnostics import select_myopic_action
from src.objectives.mocu.train import (
    load_trained_policy,
    sample_trajectory,
    sequence_diversity_stats,
)
from src.control.oracle_u_ctrl import (
    check_oracle_consistency,
    load_or_compute_oracle_cache,
)

MIN_VALID_SAFETY_RATE = 0.95


def _synchronize_cuda() -> None:
    """Make wall-clock measurements include queued CUDA work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _evaluation_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fixed_preparation_seconds(ctx: ExperimentContext) -> float:
    """Read the actual offline Fixed search cost recorded by context creation."""
    path = ctx.out_dir / "model" / f"fixed_subset_T{ctx.horizon}.json"
    if not path.is_file():
        return 0.0
    try:
        return float(json.loads(path.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0.0


def _rollout_baseline(
    ctx: ExperimentContext,
    system: dict[str, Any],
    *,
    theta_id: int,
    rollout_id: int,
    choose_action: Callable[[np.ndarray, list[int], int], int],
    eval_seed: int,
) -> dict[str, Any]:
    from src.observations.carry_state import (
        make_carry_observer,
        use_carry_state_observation,
    )

    log_w = ctx.log_p0.copy()
    actions: list[int] = []
    observations: list[list[float]] = []
    ess_path = [posterior_ess(log_w)]
    carry = (
        make_carry_observer(ctx, system)
        if use_carry_state_observation(ctx, for_training=False)
        else None
    )
    for step in range(ctx.horizon):
        action = int(choose_action(log_w, actions, step))
        if carry is not None:
            y = carry.observe_noisy(
                action,
                sigma_y=ctx.sigma_y,
                global_seed=int(eval_seed),
                theta_id=theta_id,
                rollout_id=rollout_id,
                step=step,
                n_obs=ctx.n_obs,
            )
        else:
            y = observe_compressed(
                system,
                action,
                sigma_y=ctx.sigma_y,
                n_obs=ctx.n_obs,
                global_seed=int(eval_seed),
                theta_id=theta_id,
                rollout_id=rollout_id,
                step=step,
            )
        actions.append(action)
        observations.append(y.tolist())
        log_w = update_posterior_vector(ctx, log_w, action, y)
        ess_path.append(posterior_ess(log_w))
    u_ctrl = terminal_u_ctrl(ctx, log_w)
    return {
        "sequence": actions,
        "y_obs": observations,
        "ess_by_step": ess_path,
        "u_ctrl": u_ctrl,
        "theta_id": theta_id,
        "rollout_id": rollout_id,
    }


def evaluate_fixed(
    ctx: ExperimentContext, n_rollouts: int, *, eval_seed: int
) -> list[dict[str, Any]]:
    seq = list(ctx.fixed_sequence)
    if len(seq) != int(ctx.horizon):
        raise RuntimeError(
            f"Fixed sequence length {len(seq)} != horizon T={ctx.horizon}"
        )
    if len(set(seq)) != len(seq):
        raise RuntimeError("Fixed sequence has repeats; no-repeat designs required")
    rows = []
    for rid in range(n_rollouts):
        tid = rid % len(ctx.test_systems)

        def choose(log_w, used, step, seq=seq):
            a = int(seq[step])
            if a in used:
                raise RuntimeError(f"Fixed action {a} already used at step {step}")
            return a

        out = _rollout_baseline(
            ctx,
            ctx.test_systems[tid],
            theta_id=tid,
            rollout_id=rid,
            choose_action=choose,
            eval_seed=eval_seed,
        )
        rows.append({"method": "Fixed", **_flat(out)})
    return rows


def evaluate_random(
    ctx: ExperimentContext,
    n_rollouts: int,
    seed: int | None = None,
    *,
    replicates_per_system: int = 1,
    eval_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Random baseline with independent design seeds within each test system."""
    noise_seed = int(GLOBAL_SEED if eval_seed is None else eval_seed)
    action_seed = int(noise_seed if seed is None else seed)
    rows = []
    for tid in range(n_rollouts):
        for replicate in range(int(replicates_per_system)):
            rid = tid * int(replicates_per_system) + replicate
            rng = np.random.default_rng(action_seed + rid * 17)
            chrono = False

            def choose(log_w, used, step, rng=rng, chrono=chrono):
                if chrono:
                    from src.domains.sir.design import chronological_feasible

                    remaining = max(int(ctx.horizon) - int(step), 1)
                    feasible = chronological_feasible(
                        ctx.n_actions, list(used), remaining_steps=remaining
                    )
                    if feasible.size == 0:
                        raise RuntimeError("No chronological actions remain")
                    return int(rng.choice(feasible))
                available = [a for a in range(ctx.n_actions) if a not in used]
                return int(rng.choice(available))

            out = _rollout_baseline(
                ctx,
                ctx.test_systems[tid % len(ctx.test_systems)],
                theta_id=tid % len(ctx.test_systems),
                rollout_id=rid,
                choose_action=choose,
                eval_seed=noise_seed,
            )
            flat = _flat(out)
            flat["random_replicate"] = replicate
            rows.append({"method": "Random", **flat})
    return rows


def evaluate_myopic(
    ctx: ExperimentContext, n_rollouts: int, *, eval_seed: int
) -> list[dict[str, Any]]:
    rows = []
    for rid in range(n_rollouts):
        tid = rid % len(ctx.test_systems)

        def choose(log_w, used, step, rid=rid):
            return select_myopic_action(
                ctx,
                log_w,
                used,
                rollout_id=rid,
                step=step,
                seed=int(eval_seed),
            )

        out = _rollout_baseline(
            ctx,
            ctx.test_systems[tid],
            theta_id=tid,
            rollout_id=rid,
            choose_action=choose,
            eval_seed=eval_seed,
        )
        rows.append({"method": "Myopic", **_flat(out)})
    return rows


def evaluate_moe_sboed(
    ctx: ExperimentContext, n_rollouts: int, *, eval_seed: int
) -> list[dict[str, Any]]:
    return evaluate_policy_method(
        ctx,
        "MoE-sBOED",
        n_rollouts,
        seed=int(eval_seed),
        deterministic=True,
        method_label="MoE-sBOED",
    )


def evaluate_policy_method(
    ctx: ExperimentContext,
    method: str,
    n_rollouts: int,
    seed: int = 101,
    *,
    deterministic: bool = True,
    method_label: str | None = None,
) -> list[dict[str, Any]]:
    """
    Roll out a trained policy.

    ``deterministic=True`` (argmax) is the primary adaptivity metric: many unique
    sequences across θ means the policy conditions on belief. Stochastic sampling
    is reported separately and must not be used alone to claim adaptivity.
    """
    eval_seed = int(seed)
    device = _evaluation_device()
    policy = load_trained_policy(ctx, method, device=device)
    reward_mode = "dad_terminal" if method == "DAD" else "rl_sboed_stepwise"
    label = method_label or method
    eval_mode = "deterministic" if deterministic else "stochastic"
    rows = []
    for rid in range(n_rollouts):
        tid = rid % len(ctx.test_systems)
        traj = sample_trajectory(
            ctx,
            policy,
            ctx.test_systems[tid],
            theta_id=tid,
            rollout_id=rid,
            global_seed=eval_seed,
            reward_mode=reward_mode,
            device=device,
            deterministic=deterministic,
        )
        # Recompute ESS path for reporting
        log_w = ctx.log_p0.copy()
        ess = [posterior_ess(log_w)]
        y_list = []
        for step, (a, y) in enumerate(zip(traj["actions"], traj["observations"])):
            log_w = update_posterior_vector(ctx, log_w, a, y)
            ess.append(posterior_ess(log_w))
            y_list.append(np.asarray(y).tolist())
        rows.append(
            {
                "method": label,
                "base_method": method,
                "eval_mode": eval_mode,
                "rollout_id": rid,
                "theta_id": tid,
                "sequence": " ".join(map(str, traj["actions"])),
                "y_obs": json.dumps(y_list),
                "ess_by_step": " ".join(f"{x:.4f}" for x in ess),
                "u_ctrl": float(traj["terminal_u_ctrl"]),
            }
        )
    return rows


def _flat(out: dict[str, Any]) -> dict[str, Any]:
    return {
        "rollout_id": out["rollout_id"],
        "theta_id": out["theta_id"],
        "sequence": " ".join(map(str, out["sequence"])),
        "y_obs": json.dumps(out["y_obs"]),
        "ess_by_step": " ".join(f"{x:.4f}" for x in out["ess_by_step"]),
        "u_ctrl": float(out["u_ctrl"]),
        "eval_mode": "baseline",
        "base_method": None,
    }


def attach_oracle(
    ctx: ExperimentContext,
    method_rows: list[dict[str, Any]],
    *,
    skip_cuda_safety: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Add oracle control, realized OCU, and safety to each held-out rollout."""
    # Oracle cache lives under eval/ with other evaluation artifacts.
    oracle_dir = ctx.out_dir / "eval" / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    cache_path = oracle_dir / "u_ctrl_opt_cache.json"

    engine, spec = control_engine_for(ctx)
    # Build oracle systems with true M/K only (not used during design selection).
    oracle_systems = [
        {"M": ctx.M_test[i].tolist(), "K": ctx.K_test[i].tolist()}
        for i in range(len(ctx.test_systems))
    ]
    oracle_rows = load_or_compute_oracle_cache(
        cache_path,
        oracle_systems,
        engine,
        spec,
        tolerance=ctx.oracle_tolerance,
    )
    opt_by_tid = {int(r["theta_id"]): r for r in oracle_rows}

    enriched = []
    errors: list[str] = []
    for row in method_rows:
        tid = int(row["theta_id"])
        opt = opt_by_tid[tid]
        u_opt = float(opt["u_ctrl_opt"])
        u_ctrl = float(row["u_ctrl"])
        gap = u_ctrl - u_opt
        method_safe = False
        if not skip_cuda_safety:
            M = np.asarray(ctx.M_test[tid], dtype=np.float64)
            K = np.asarray(ctx.K_test[tid], dtype=np.float64)
            m = engine.evaluate_one(M, K, u_ctrl)
            method_safe = bool(m["safe_total"] >= 0.5)
        else:
            # Smoke without re-checking: treat u_ctrl >= u_opt as safe proxy.
            method_safe = u_ctrl + 1e-12 >= u_opt
        err = check_oracle_consistency(
            u_ctrl,
            u_opt,
            method_safe,
            tolerance=ctx.oracle_tolerance,
            oracle_feasible=bool(opt.get("feasible", True)),
        )
        if err:
            errors.append(f"{row['method']} theta={tid}: {err}")
        enriched.append(
            {
                **row,
                "u_ctrl_opt": u_opt,
                "control_gap": gap,
                # Realized objective cost of uncertainty for this true model.
                "ocu": gap,
                "method_safe": int(method_safe),
                "oracle_feasible": int(bool(opt.get("feasible", True))),
                "oracle_message": opt.get("message", ""),
                "oracle_consistency_error": err or "",
            }
        )
    return enriched, errors


def summarize_rows(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    sub = [r for r in rows if r["method"] == method]
    if not sub:
        return {"method": method, "n": 0}
    u = np.asarray([float(r["u_ctrl"]) for r in sub], dtype=np.float64)
    gaps = np.asarray([float(r["control_gap"]) for r in sub], dtype=np.float64)
    excess = np.maximum(gaps, 0.0)
    safe = np.asarray([int(r["method_safe"]) for r in sub], dtype=np.float64)
    opts = np.asarray([float(r["u_ctrl_opt"]) for r in sub], dtype=np.float64)
    under = gaps < -1e-12
    div = sequence_diversity_stats([str(r["sequence"]) for r in sub])
    eval_mode = sub[0].get("eval_mode", "")
    base = sub[0].get("base_method") or method
    safety_rate = float(safe.mean())
    # Cluster repeated stochastic-design draws by physical system.  Treating
    # every Random seed as an independent test system would understate the CI.
    gaps_by_theta: dict[int, list[float]] = {}
    for row in sub:
        gaps_by_theta.setdefault(int(row["theta_id"]), []).append(
            float(row["control_gap"])
        )
    clustered_gaps = np.asarray(
        [np.mean(values) for values in gaps_by_theta.values()], dtype=np.float64
    )
    mocu_ci = paired_bootstrap_ci(clustered_gaps)
    return {
        "method": method,
        "base_method": base,
        "eval_mode": eval_mode,
        "n": len(gaps_by_theta),
        "n_design_replicates": len(sub),
        "mean_u_ctrl": float(u.mean()),
        "median_u_ctrl": float(np.median(u)),
        "mean_u_ctrl_opt": float(opts.mean()),
        "median_u_ctrl_opt": float(np.median(opts)),
        # Raw mean(u_ctrl - u_opt); negative ⇒ under-control (often unsafe).
        "mean_gap": float(clustered_gaps.mean()),
        "median_gap": float(np.median(clustered_gaps)),
        # Final objective metric: mean realized OCU on common held-out models.
        "mean_mocu": float(clustered_gaps.mean()),
        "median_mocu": float(np.median(clustered_gaps)),
        "mocu_ci95_low": float(mocu_ci["ci95_low"]),
        "mocu_ci95_high": float(mocu_ci["ci95_high"]),
        # Overshoot only; under-control does not improve this score.
        "mean_excess": float(excess.mean()),
        "under_control_rate": float(under.mean()),
        "gap_ci95_low": float(np.percentile(gaps, 2.5)),
        "gap_ci95_high": float(np.percentile(gaps, 97.5)),
        "safety_rate": safety_rate,
        "valid": int(safety_rate >= MIN_VALID_SAFETY_RATE),
        "validity_threshold": MIN_VALID_SAFETY_RATE,
        "n_unique_sequences": int(div["n_unique_sequences"]),
        "sequence_entropy": float(div["sequence_entropy"]),
        "unique_frac": float(div["unique_frac"]),
        "n_oracle_consistency_errors": int(
            sum(1 for r in sub if r.get("oracle_consistency_error"))
        ),
    }


def oracle_summary_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """One summary line for true-θ optimal control (shared across methods)."""
    opts = [
        float(r["u_ctrl_opt"])
        for r in rows
        if r.get("u_ctrl_opt") is not None
    ]
    if not opts:
        return None
    # Unique by theta_id when rollouts repeat θ
    by_theta: dict[int, float] = {}
    for r in rows:
        if r.get("u_ctrl_opt") is None:
            continue
        by_theta[int(r["theta_id"])] = float(r["u_ctrl_opt"])
    vals = np.asarray(list(by_theta.values()) if by_theta else opts, dtype=np.float64)
    return {
        "method": "Oracle",
        "n": int(vals.size),
        "mean_u_ctrl": float(vals.mean()),
        "median_u_ctrl": float(np.median(vals)),
        "mean_u_ctrl_opt": float(vals.mean()),
        "median_u_ctrl_opt": float(np.median(vals)),
        "mean_gap": 0.0,
        "median_gap": 0.0,
        "mean_mocu": 0.0,
        "median_mocu": 0.0,
        "mocu_ci95_low": 0.0,
        "mocu_ci95_high": 0.0,
        "mean_excess": 0.0,
        "under_control_rate": 0.0,
        "gap_ci95_low": 0.0,
        "gap_ci95_high": 0.0,
        "safety_rate": 1.0,
        "n_oracle_consistency_errors": 0,
        "rank_by_mean_gap": 0,
    }


def paired_bootstrap_ci(
    diff: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=np.float64)
    if d.size == 0:
        return {"mean_diff": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, d.size, size=d.size)
        boots.append(float(d[idx].mean()))
    boots_a = np.asarray(boots)
    return {
        "mean_diff": float(d.mean()),
        "ci95_low": float(np.percentile(boots_a, 2.5)),
        "ci95_high": float(np.percentile(boots_a, 97.5)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_full_evaluation(
    ctx: ExperimentContext,
    *,
    methods: list[str] | None = None,
    smoke: bool = False,
    skip_cuda_safety: bool = False,
    eval_seed: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate selected methods + true-θ oracle.

    ``methods``: canonical keys ``dad|rl_sboed|myopic|fixed|random``.
    Default: all five.
    """
    from src.objectives.mocu.context import ALL_METHOD_KEYS, normalize_method_key
    from src.layout import (
        load_run_config_doc,
        method_checkpoint_available,
        resolve_eval_seed,
    )

    if eval_seed is None:
        eval_seed = resolve_eval_seed(ctx.out_dir)
    eval_seed = int(eval_seed)

    keys = (
        [normalize_method_key(m) for m in methods]
        if methods is not None
        else list(ALL_METHOD_KEYS)
    )
    kept_keys = []
    for key in keys:
        if method_checkpoint_available(ctx.out_dir, key):
            kept_keys.append(key)
        else:
            print(
                f"[evaluate] skip {key}: missing checkpoint under {ctx.out_dir}/model"
            )
    keys = kept_keys
    display = [method_display_name(k) for k in keys]

    run_doc = load_run_config_doc(ctx.out_dir)
    training_results = dict(run_doc.get("training_results") or {})
    offline_seconds = {
        "DAD": float((training_results.get("dad") or {}).get("elapsed_seconds", 0.0)),
        "RL-sBOED": float(
            (training_results.get("rl_sboed") or {}).get("elapsed_seconds", 0.0)
        ),
        "Myopic": 0.0,
        "Fixed": _fixed_preparation_seconds(ctx),
        "Random": 0.0,
    }
    offline_seconds["MoE-sBOED"] = float(
        (training_results.get("moe_sboed") or {}).get("elapsed_seconds", 0.0)
    )
    n = 4 if smoke else min(64, max(16, len(ctx.test_systems) * 2))
    rows: list[dict[str, Any]] = []
    summary_methods: list[str] = []
    runtime_by_method: dict[str, dict[str, Any]] = {}
    for key, name in zip(keys, display):
        print(f"[{ctx.system}] evaluating {name} n={n} eval_seed={eval_seed}")
        _synchronize_cuda()
        started = time.perf_counter()
        method_rows: list[dict[str, Any]]
        if key == "fixed":
            method_rows = evaluate_fixed(ctx, n, eval_seed=eval_seed)
        elif key == "random":
            method_rows = evaluate_random(
                ctx,
                n,
                seed=eval_seed,
                eval_seed=eval_seed,
                replicates_per_system=8 if smoke else 32,
            )
        elif key == "myopic":
            method_rows = evaluate_myopic(ctx, n, eval_seed=eval_seed)
        elif key == "moe_sboed":
            method_rows = evaluate_moe_sboed(ctx, n, eval_seed=eval_seed)
        elif key in ("dad", "rl_sboed", "matched_dense"):
            method_rows = evaluate_policy_method(
                ctx,
                name,
                n,
                seed=eval_seed,
                deterministic=True,
                method_label=name,
            )
        else:
            raise ValueError(f"unsupported method key {key}")
        _synchronize_cuda()
        elapsed = time.perf_counter() - started
        per_rollout = elapsed / max(len(method_rows), 1)
        for row in method_rows:
            row["method_eval_total_seconds"] = float(elapsed)
            row["method_eval_seconds_per_rollout"] = float(per_rollout)
        rows.extend(method_rows)
        summary_methods.append(name)
        runtime_by_method[name] = {
            "offline_training_or_calibration_seconds": offline_seconds.get(name, 0.0),
            "n_rollouts": len(method_rows),
            "total_seconds": float(elapsed),
            "seconds_per_rollout": float(per_rollout),
            "includes": "action selection + observation + posterior update + terminal decision",
        }
        print(
            f"[{ctx.system}] {name} runtime={elapsed:.3f}s "
            f"({per_rollout:.6f}s/rollout)"
        )

    print(f"[{ctx.system}] oracle + safety")
    enriched, errors = attach_oracle(
        ctx, rows, skip_cuda_safety=skip_cuda_safety
    )
    from src.layout import ensure_result_layout

    _, eval_root = ensure_result_layout(ctx.out_dir)
    summaries = [summarize_rows(enriched, m) for m in summary_methods]
    for summary in summaries:
        name = str(summary["method"])
        summary["offline_training_or_calibration_seconds"] = offline_seconds.get(
            name, 0.0
        )
        # Short, publication-table-friendly alias.  For Fixed this is offline
        # design search; for learned methods it is end-to-end policy training.
        summary["training_time_seconds"] = offline_seconds.get(name, 0.0)
        summary["online_total_seconds"] = runtime_by_method[name]["total_seconds"]
        summary["online_seconds_per_rollout"] = runtime_by_method[name][
            "seconds_per_rollout"
        ]
        summary["eval_seed"] = int(eval_seed)
        summary["timing_scope"] = (
            "offline=method-specific preparation; online=warm action selection + "
            "observation lookup + posterior update + terminal decision; shared physical "
            "bank generation excluded and must be reported separately"
        )
    # Hard validity constraint: methods below 95% safety are not ranked.
    # Among valid methods, lower terminal MOCU wins.
    ranked = sorted(
        [
            s
            for s in summaries
            if s.get("n", 0) > 0
            and float(s.get("safety_rate", 0.0)) >= MIN_VALID_SAFETY_RATE
        ],
        key=lambda s: float(s.get("mean_mocu", float("inf"))),
    )
    for s in summaries:
        s["valid"] = int(
            float(s.get("safety_rate", 0.0)) >= MIN_VALID_SAFETY_RATE
        )
        s["validity_threshold"] = MIN_VALID_SAFETY_RATE
        s["rank_by_mean_gap"] = ""
    for i, s in enumerate(ranked):
        s["rank_by_mean_gap"] = i + 1  # Legacy column name; rank is by mean_mocu.

    oracle_row = oracle_summary_row(enriched)
    if oracle_row is not None:
        oracle_row["eval_seed"] = int(eval_seed)
    summary_for_csv = ([oracle_row] if oracle_row else []) + summaries

    # Required: one JSON per method + one summary.csv for all methods.
    summary_by_display = {str(s["method"]): s for s in summaries}
    for key, name in zip(keys, display):
        method_rows = [r for r in enriched if r.get("method") == name]
        payload = {
            "method": key,
            "display_name": name,
            "summary": summary_by_display.get(name, {"method": name, "n": 0}),
            "rollouts": method_rows,
            "n_rollouts": len(method_rows),
        }
        if key in ("dad", "rl_sboed", "moe_sboed"):
            payload["adaptivity_note"] = (
                "Claim adaptivity from n_unique_sequences / sequence_entropy "
                "across θ under argmax (deterministic) rollouts."
            )
        (eval_root / f"{key}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
    write_csv(eval_root / "summary.csv", summary_for_csv)

    # Optional extras (still under eval/)
    write_csv(eval_root / "rollouts.csv", enriched)
    pair_rows = []
    grouped: dict[str, dict[int, list[float]]] = {}
    for r in enriched:
        grouped.setdefault(r["method"], {})
        tid = int(r["theta_id"])
        grouped[r["method"]].setdefault(tid, []).append(float(r["control_gap"]))
    by_method = {
        method: {
            tid: float(np.mean(values)) for tid, values in per_theta.items()
        }
        for method, per_theta in grouped.items()
    }
    pairs = [
        ("DAD", "Random"),
        ("RL-sBOED", "Random"),
        ("RL-sBOED", "DAD"),
        ("Myopic", "Fixed"),
        ("DAD", "Myopic"),
        ("MoE-sBOED", "Myopic"),
        ("MoE-sBOED", "DAD"),
    ]
    for left, right in pairs:
        if left not in by_method or right not in by_method:
            continue
        tids = sorted(set(by_method[left]) & set(by_method[right]))
        if not tids:
            continue
        diff = np.asarray(
            [by_method[left][t] - by_method[right][t] for t in tids], dtype=np.float64
        )
        pair_rows.append({"comparison": f"{left} - {right}", **paired_bootstrap_ci(diff)})
    write_csv(eval_root / "paired_gap.csv", pair_rows)

    (eval_root / "oracle_consistency_errors.json").write_text(
        json.dumps({"errors": errors, "n": len(errors)}, indent=2), encoding="utf-8"
    )
    meta = {
        **context_report_meta(ctx),
        "methods": summary_methods,
        "primary_methods": display,
        "n_rollouts_per_method": n,
        "n_oracle_consistency_errors": len(errors),
        "mean_oracle_u_ctrl": float(
            np.mean([float(r["u_ctrl_opt"]) for r in enriched])
        )
        if enriched
        else None,
        "method_ranking": [s["method"] for s in ranked],
        "minimum_valid_safety_rate": MIN_VALID_SAFETY_RATE,
        "invalid_methods": [
            s["method"] for s in summaries if not bool(s.get("valid", 0))
        ],
        "primary_metric": "mean_mocu",
        "runtime_by_method": runtime_by_method,
        "runtime_scope": (
            "Per-method online evaluation only; training, fixed-search preprocessing, "
            "hybrid calibration, and oracle/safety evaluation are reported separately."
        ),
        "mocu_definition": "mean over held-out systems of (u_ctrl - u_ctrl_opt)",
        "ranking_rule": (
            "safety_rate >= 0.95 required; valid methods ranked by mean_mocu asc"
        ),
        "eval_seed": int(eval_seed),
        "summaries": summaries,
        "adaptivity_eval": (
            "Policy methods use argmax rollouts; claim adaptivity from "
            "n_unique_sequences / sequence_entropy across θ."
        ),
    }
    (eval_root / "eval_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
