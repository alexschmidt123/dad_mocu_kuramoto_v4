"""Orchestrate stepwise EIG evaluation for one or all IEEE benchmark systems."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import repo_root
from src.experiment_layout import model_dir
from src.legacy.plot_summary import find_latest_experiment_dirs
from src.run_context import load_experiment_run
from src.swing_equation_ode.design import build_catalog
from src.table_scoring import TableThetaSupport

from src.stepwise_eig.evaluate import (
    STEPWISE_METHODS,
    _noise_seed,
    aggregate_method_stepwise,
    collect_method_rollouts,
    compute_step1_heatmap,
    score_rollout_stepwise_eig,
)
from src.stepwise_eig.io import (
    write_consistency_csv,
    write_entropy_traces_csv,
    write_json,
    write_report_md,
    write_rollouts_csv,
    write_step1_heatmap_csv,
    write_stepwise_summary_csv,
    write_terminal_summary_csv,
    write_terminal_vs_T_csv,
)
from src.stepwise_eig.plots import (
    plot_step1_heatmap,
    plot_stepwise_eig,
    plot_terminal_eig_vs_T,
)

SYSTEM_PREFIXES = ("ieee5", "ieee9", "ieee14")


def _default_noise_seed(cfg) -> int:
    return int(cfg.data.get("test_seed", 1))


def _default_support_seed(cfg) -> int:
    return int(cfg.prior.get("mc_support_seed", 1))


def _default_rollout_seed(cfg) -> int:
    return int(cfg.data.get("test_seed", 1)) + 1000


def _method_rollout_seed(base_seed: int, method: str) -> int:
    tag = sum(ord(c) for c in method) & 0xFFFF
    return _noise_seed(base_seed, tag, 999)


def _validate_experiment(exp_dir: Path, project_root: Path) -> None:
    run = load_experiment_run(exp_dir, project_root)
    for method in ("dad_spce", "dad_delta_h"):
        policy = model_dir(exp_dir) / f"{method}.pth"
        if not policy.is_file():
            raise FileNotFoundError(
                f"Missing trained policy for {method} under {exp_dir}.\n"
                "Run: ./scripts/dad_training.sh -exp-dir <experiment>"
            )
    if not run.test_systems:
        raise ValueError(f"No test systems loaded for {exp_dir}")


def evaluate_experiment_stepwise_eig(
    exp_dir: Path,
    project_root: Path | None = None,
    *,
    methods: tuple[str, ...] = STEPWISE_METHODS,
    noise_seed: int | None = None,
    support_seed: int | None = None,
    rollout_seed: int | None = None,
) -> dict[str, Any]:
    """Run stepwise EIG for one experiment folder (fixed horizon T)."""
    root = project_root or repo_root()
    exp_dir = exp_dir.resolve()
    _validate_experiment(exp_dir, root)
    run = load_experiment_run(exp_dir, root)
    cfg = run.cfg
    catalog = build_catalog(cfg)

    ns = noise_seed if noise_seed is not None else _default_noise_seed(cfg)
    ss = support_seed if support_seed is not None else _default_support_seed(cfg)
    rs = rollout_seed if rollout_seed is not None else _default_rollout_seed(cfg)

    support_rng = np.random.default_rng(ss)
    table_support = TableThetaSupport.from_train(run.train_systems, cfg, support_rng)

    heatmap = compute_step1_heatmap(
        cfg, run.test_systems, catalog, table_support, noise_seed=ns,
    )

    method_results: dict[str, Any] = {}
    for method in methods:
        method_rng = np.random.default_rng(_method_rollout_seed(rs, method))
        rollouts = collect_method_rollouts(
            method,
            cfg,
            exp_dir,
            run.train_systems,
            run.test_systems,
            method_rng,
            catalog,
            table_support,
        )
        scored = [
            score_rollout_stepwise_eig(
                cfg,
                sys,
                r["sequence"],
                table_support,
                rollout_idx=i,
                noise_seed=ns,
            )
            for i, (sys, r) in enumerate(zip(run.test_systems, rollouts))
        ]
        method_results[method] = aggregate_method_stepwise(
            scored, method=method, catalog=catalog,
        )

    return {
        "system_label": cfg.system_label,
        "run_prefix": cfg.run_slug,
        "horizon": cfg.step_number,
        "exp_dir": str(exp_dir),
        "data_dir": str(run.data_path),
        "noise_seed": ns,
        "support_seed": ss,
        "rollout_seed": rs,
        "n_test_rollouts": len(run.test_systems),
        "heatmap": heatmap,
        "methods": method_results,
    }


def write_system_outputs(
    out_dir: Path,
    payload: dict[str, Any],
    *,
    terminal_vs_T: list[dict[str, Any]] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = payload["methods"]
    horizon = int(payload["horizon"])

    write_rollouts_csv(out_dir / "rollouts.csv", methods)
    write_entropy_traces_csv(out_dir / "entropy_traces.csv", methods)
    write_stepwise_summary_csv(out_dir / "stepwise_eig_summary.csv", methods)
    write_terminal_summary_csv(out_dir / "terminal_eig_summary.csv", methods, horizon=horizon)
    write_step1_heatmap_csv(out_dir / "step1_heatmap.csv", payload["heatmap"])
    write_consistency_csv(out_dir / "terminal_consistency_checks.csv", methods)
    write_json(out_dir / "stepwise_eig_results.json", payload)

    if terminal_vs_T:
        write_terminal_vs_T_csv(out_dir / "terminal_eig_vs_T.csv", terminal_vs_T)

    plot_step1_heatmap(
        payload["heatmap"],
        out_path=out_dir / "step1_heatmap.png",
        system_label=str(payload["system_label"]),
    )
    plot_stepwise_eig(
        methods,
        out_path=out_dir / "stepwise_eig.png",
        system_label=str(payload["system_label"]),
        horizon=horizon,
    )
    if terminal_vs_T:
        plot_terminal_eig_vs_T(
            terminal_vs_T,
            out_path=out_dir / "terminal_eig_vs_T.png",
            system_label=str(payload["system_label"]),
        )

    write_report_md(
        out_dir / "report.md",
        system_label=str(payload["system_label"]),
        run_prefix=str(payload["run_prefix"]),
        horizon=horizon,
        method_results=methods,
        heatmap=payload["heatmap"],
        terminal_vs_T=terminal_vs_T or [],
        exp_dir=Path(payload["exp_dir"]),
    )


def _terminal_rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method, block in payload["methods"].items():
        rows.append({
            "method": method,
            "method_label": block["method_label"],
            "T": int(payload["horizon"]),
            "terminal_eig_mean": block["terminal_eig_mean"],
            "terminal_eig_sem": block["terminal_eig_sem"],
            "ci95_low": block["terminal_eig_ci95_low"],
            "ci95_high": block["terminal_eig_ci95_high"],
            "experiment_dir": payload["exp_dir"],
        })
    return rows


def run_system_stepwise_eig(
    run_prefix: str,
    project_root: Path | None = None,
    *,
    exp_dir: Path | None = None,
    t_max: int | None = None,
    out_dir: Path | None = None,
    noise_seed: int | None = None,
    support_seed: int | None = None,
    rollout_seed: int | None = None,
) -> Path:
    root = project_root or repo_root()
    by_T = find_latest_experiment_dirs(root, run_prefix=run_prefix)
    if not by_T:
        raise FileNotFoundError(
            f"No experiment folders found for prefix '{run_prefix}' under {root / 'experiments'}"
        )

    if exp_dir is not None:
        primary = exp_dir.resolve()
    else:
        target_T = t_max if t_max is not None else max(by_T)
        if target_T not in by_T:
            raise FileNotFoundError(
                f"No experiment for {run_prefix} at T={target_T}. Available: {sorted(by_T)}"
            )
        primary = by_T[target_T]

    primary_payload = evaluate_experiment_stepwise_eig(
        primary,
        root,
        noise_seed=noise_seed,
        support_seed=support_seed,
        rollout_seed=rollout_seed,
    )

    terminal_vs_T: list[dict[str, Any]] = []
    for T, path in sorted(by_T.items()):
        try:
            if path.resolve() == primary.resolve():
                payload_T = primary_payload
            else:
                payload_T = evaluate_experiment_stepwise_eig(
                    path,
                    root,
                    noise_seed=noise_seed,
                    support_seed=support_seed,
                    rollout_seed=rollout_seed,
                )
            terminal_vs_T.extend(_terminal_rows_from_payload(payload_T))
        except (FileNotFoundError, ValueError) as exc:
            print(f"  skip T={T} ({path.name}): {exc}")

    system_out = out_dir or (root / "experiments" / "stepwise_eig" / run_prefix)
    write_system_outputs(system_out, primary_payload, terminal_vs_T=terminal_vs_T)
    return system_out


def run_all_systems(
    project_root: Path | None = None,
    *,
    systems: tuple[str, ...] = SYSTEM_PREFIXES,
    t_max: int | None = None,
    noise_seed: int | None = None,
    support_seed: int | None = None,
    rollout_seed: int | None = None,
) -> dict[str, Path]:
    root = project_root or repo_root()
    outputs: dict[str, Path] = {}
    for prefix in systems:
        print(f"=== Stepwise EIG: {prefix} ===")
        try:
            out = run_system_stepwise_eig(
                prefix,
                root,
                t_max=t_max,
                noise_seed=noise_seed,
                support_seed=support_seed,
                rollout_seed=rollout_seed,
            )
            outputs[prefix] = out
            print(f"  → {out}")
        except FileNotFoundError as exc:
            print(f"  skipped: {exc}")
    return outputs
