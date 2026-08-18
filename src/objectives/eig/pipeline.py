"""
EIG-based experiment pipeline (``experiment_type=eig_based``).

Table-backed train/eval: ``sequence``, noisy ``y``, ``y_sim`` (ODE before noise).
π uses ``y`` only; sPCE / myopic use ``y_sim`` as likelihood centres.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:  # pragma: no cover - cosmetic progress helper
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)
    tqdm.write = print

from src.config import ALL_METHODS, SBOEDConfig, repo_root
from src.banks.tables import (
    TableThetaSupport,
    ensure_data,
    load_split_systems,
    resolve_data_dir,
    save_json,
    validate_trajectory_y_sim,
    y_sim_last_step_from_tables,
    y_sim_sequence_from_table,
    y_sim_steps_from_tables,
)
from src.layout import (
    ExperimentRun,
    eval_dir,
    eval_method_path,
    eval_summary_path,
    load_experiment_run,
    load_run_config_doc,
    make_experiment_dir_name,
    model_dir,
    reset_model_dir,
    write_run_config,
)
from src.domains.swing.simulator import system_mk
from src.control.posterior_ctrl import (
    clamp_info_gain,
    normalize_log_weights,
    posterior_after_gaussian_observations,
    posterior_entropy,
    posterior_mean_mk_vectors,
)
from src.observations.likelihood import log_gaussian_observation_density
from src.policies.dad import DADPolicy
from src.objectives.eig.dad_rollout import policy_rollout
from src.domains.swing.design import build_catalog


# --- Metrics ---------------------------------------------------------------


def _log_mean_exp(log_vals: np.ndarray) -> float:
    c = float(np.max(log_vals))
    return c + float(np.log(np.mean(np.exp(log_vals - c))))


def _foster_pce_from_log_likelihoods(
    log_p_positive: float,
    log_p_contrastive: np.ndarray,
) -> float:
    """Foster prior-contrastive EIG bound (DAD ``PriorContrastiveEstimation``)."""
    log_denom = _log_mean_exp(
        np.concatenate([[log_p_positive], np.asarray(log_p_contrastive).ravel()])
    )
    return clamp_info_gain(float(log_p_positive - log_denom))


def _foster_pce_from_f_tensor(
    y_seq: np.ndarray,
    f_tensor: np.ndarray,
    sigma_y: float,
    positive_idx: int = 0,
) -> float:
    y = np.asarray(y_seq, dtype=np.float64).reshape(-1)
    f = np.asarray(f_tensor, dtype=np.float64)
    if f.ndim != 2 or f.shape[1] != y.shape[0]:
        raise ValueError(
            f"f_tensor shape {f.shape} incompatible with y_seq length {y.shape[0]}"
        )
    s2 = float(sigma_y) ** 2
    log_terms = (
        -0.5 * y.shape[0] * np.log(2.0 * np.pi * s2)
        - 0.5 * np.sum((y[None, :] - f) ** 2, axis=1) / s2
    )
    pos = int(positive_idx)
    contrastive = np.delete(log_terms, pos)
    return _foster_pce_from_log_likelihoods(float(log_terms[pos]), contrastive)


def spce_eig_from_rollout(
    cfg: SBOEDConfig,
    sequence: list[int],
    y_obs: list[float] | np.ndarray,
    theta0_system: dict[str, Any],
    support: TableThetaSupport,
    rng: np.random.Generator,
    L: int | None = None,
) -> tuple[list[float], float, float]:
    """Table-path Foster PCE-EIG: fixed noisy ``y_obs``; centres from banked ``y_sim``."""
    if L is None:
        L = int(cfg.spce.get("L", 4))
    y_seq = np.asarray(y_obs, dtype=np.float64)
    seq = [int(a) for a in sequence]
    pool = list(support.systems)
    others = [s for s in pool if s is not theta0_system] or pool
    if not others:
        raise ValueError("need at least one contrastive latent θ for Foster PCE")
    n_pick = min(int(L), len(others))
    pick = rng.choice(len(others), size=n_pick, replace=False)
    T = len(seq)
    centres = np.empty((n_pick + 1, T), dtype=np.float64)
    centres[0] = y_sim_sequence_from_table(theta0_system, seq)
    for row_i, idx in enumerate(pick, start=1):
        centres[row_i] = y_sim_sequence_from_table(others[int(idx)], seq)
    s2 = float(cfg.sigma_y) ** 2
    log_L_all = (
        -0.5 * np.log(2.0 * np.pi * s2)
        - 0.5 * (y_seq[None, :] - centres) ** 2 / s2
    )
    step_eigs = [
        _foster_pce_from_log_likelihoods(float(log_L_all[0, t]), log_L_all[1:, t])
        for t in range(len(y_seq))
    ]
    total = float(_foster_pce_from_f_tensor(y_seq, centres, cfg.sigma_y, positive_idx=0))
    mean_step = float(np.mean(step_eigs)) if step_eigs else 0.0
    return step_eigs, mean_step, total


def evaluate_rollout(
    cfg: SBOEDConfig,
    system: dict[str, Any],
    sequence: list[int],
    y_seq: list[float] | np.ndarray,
    catalog,
    table_support: TableThetaSupport,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Score rollout: test noisy ``y``; sPCE / ΔH use banked train ``y_sim`` centres."""
    del catalog
    y_arr = np.asarray(y_seq, dtype=np.float64)
    seq = [int(a) for a in sequence]
    centre_steps = y_sim_steps_from_tables(table_support, seq)

    p_final, p_trace = posterior_after_gaussian_observations(
        centre_steps, y_arr, cfg.sigma_y, table_support.log_p0,
    )
    H0 = posterior_entropy(p_trace[0])
    H1 = posterior_entropy(p_final)
    M_rows = np.stack([system_mk(s, cfg.N)[0] for s in table_support.systems])
    K_rows = np.stack([system_mk(s, cfg.N)[1] for s in table_support.systems])
    M_hat, K_hat = posterior_mean_mk_vectors(p_final, M_rows, K_rows)

    step_spce_list, _, total_spce = spce_eig_from_rollout(
        cfg, seq, y_arr, system, table_support, rng,
    )
    entropy_trace = [float(posterior_entropy(p)) for p in p_trace]
    step_entropy = entropy_trace[1:]
    step_delta_h = [
        clamp_info_gain(entropy_trace[t] - entropy_trace[t + 1])
        for t in range(len(y_arr))
    ]
    step_spce_list = [clamp_info_gain(float(x)) for x in step_spce_list]
    total_spce = clamp_info_gain(float(total_spce))
    terminal_delta_h = clamp_info_gain(float(np.sum(step_delta_h)))

    M_arr = np.asarray(system["M"], dtype=np.float64).reshape(-1)
    K_arr = np.asarray(system["K"], dtype=np.float64).reshape(-1)

    mse_M = float(np.mean((M_hat - M_arr) ** 2))
    mse_K = float(np.mean((K_hat - K_arr) ** 2))
    mse_theta = float(np.sum((M_hat - M_arr) ** 2) + np.sum((K_hat - K_arr) ** 2))

    eig = eig_metrics(step_spce_list, step_delta_h, total_spce, terminal_delta_h)
    return {
        "sequence": sequence,
        "y": y_arr.tolist(),
        "M_true": M_arr.tolist(),
        "K_true": K_arr.tolist(),
        "M_hat": M_hat.tolist(),
        "K_hat": K_hat.tolist(),
        "H_prior": H0,
        "H_posterior": H1,
        "entropy_trace": entropy_trace,
        "step_entropy": step_entropy,
        "mse_M": mse_M,
        "mse_K": mse_K,
        "mse_theta": mse_theta,
        **eig,
    }


def eig_metrics(
    spce_by_step: list[float] | np.ndarray,
    delta_h_by_step: list[float] | np.ndarray,
    total_spce: float,
    delta_h: float,
) -> dict[str, Any]:
    """Canonical EIG result block: two lists + two scalars (all non-negative)."""
    spce_steps = [clamp_info_gain(float(x)) for x in spce_by_step]
    dh_steps = [clamp_info_gain(float(x)) for x in delta_h_by_step]
    return {
        "spce_by_step": spce_steps,
        "delta_h_by_step": dh_steps,
        "total_spce": clamp_info_gain(float(total_spce)),
        "delta_h": clamp_info_gain(float(delta_h)),
    }


def read_eig_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Read canonical EIG fields; fall back to legacy eval JSON keys."""
    if all(k in record for k in ("spce_by_step", "delta_h_by_step", "total_spce", "delta_h")):
        return eig_metrics(
            record["spce_by_step"],
            record["delta_h_by_step"],
            record["total_spce"],
            record["delta_h"],
        )

    spce_raw = record.get("mean_spce_eig_by_step") or record.get("step_spce_eig")
    if spce_raw is None:
        tot = record.get("mean_total_spce_eig") or record.get("total_spce_eig")
        spce_by_step = [float(tot)] if tot is not None else []
    else:
        spce_by_step = [float(x) for x in spce_raw]

    dh_raw = record.get("mean_delta_h_by_step") or record.get("step_delta_h")
    if dh_raw is None:
        tot_dh = record.get("mean_delta_H") or record.get("delta_H")
        delta_h_by_step = [float(tot_dh)] if tot_dh is not None else []
    else:
        delta_h_by_step = [float(x) for x in dh_raw]

    total_spce = record.get("mean_total_spce_eig") or record.get("total_spce_eig")
    if total_spce is None and spce_by_step:
        total_spce = spce_by_step[0]

    delta_h = record.get("mean_delta_H") or record.get("delta_H")
    return eig_metrics(
        spce_by_step,
        delta_h_by_step,
        float(total_spce or 0.0),
        float(delta_h or 0.0),
    )


def format_eig_list(vals: list[float], *, prec: int = 4) -> str:
    return "[" + ", ".join(f"{v:.{prec}f}" for v in vals) + "]"


def format_eig_line(eig: dict[str, Any], *, prec: int = 4) -> str:
    return (
        f"spce_by_step={format_eig_list(eig['spce_by_step'], prec=prec)}  "
        f"delta_h_by_step={format_eig_list(eig['delta_h_by_step'], prec=prec)}  "
        f"total_spce={eig['total_spce']:.{prec}f}  "
        f"delta_h={eig['delta_h']:.{prec}f}"
    )


def slim_method_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Metrics-only block (full detail stays in ``eval/<method>.json``)."""
    keys = (
        "mean_u_ctrl",
        "median_u_ctrl",
        "std_u_ctrl",
        "safety_rate",
        "mean_excess",
        "u_ctrl_values",
        "mean_weight_sum",
        "n",
        "test_rollout_seconds_total",
        "test_rollout_seconds_per_system",
        "test_total_seconds",
        "test_total_seconds_per_system",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k in summary:
            out[k] = summary[k]
    return out


def _method_train_seconds_raw(method: str, timing: dict[str, Any] | None) -> float | None:
    train = (timing or {}).get("training_seconds") or {}
    if method in train:
        return float(train[method])
    return None


def _method_test_seconds_raw(
    method: str,
    timing: dict[str, Any] | None,
    summary: dict[str, Any],
) -> float | None:
    test = (timing or {}).get("test_seconds") or {}
    block = test.get(method)
    if isinstance(block, dict):
        val = block.get("test_total_seconds_per_system")
        if val is None:
            val = block.get("test_total_seconds")
        if val is not None:
            return float(val)
    for key in ("test_total_seconds_per_system", "test_total_seconds"):
        if summary.get(key) is not None:
            return float(summary[key])
    return None


def _method_train_seconds(method: str, timing: dict[str, Any] | None) -> str:
    val = _method_train_seconds_raw(method, timing)
    return f"{val:.1f}" if val is not None else "-"


def _method_test_seconds(method: str, timing: dict[str, Any] | None, summary: dict[str, Any]) -> str:
    val = _method_test_seconds_raw(method, timing, summary)
    return f"{val:.4f}" if val is not None else "-"


COMPARISON_TABLE_COLUMNS = [
    "System",
    "Run",
    "T",
    "N_b",
    "Method",
    "sPCE_1..T",
    "Tot.sPCE",
    "ΔH_1..T",
    "ΔH",
    "MSE_θ",
    "train_s",
    "test_s",
]


def _method_display_order(
    summaries: dict[str, Any],
    methods: list[str] | None = None,
) -> list[str]:
    if methods:
        ordered = [m for m in methods if m in summaries]
        extra = sorted(m for m in summaries if m not in ordered)
        return ordered + extra
    return list(summaries.keys())


def build_print_table_rows(
    summaries: dict[str, Any],
    timing: dict[str, Any] | None = None,
    methods: list[str] | None = None,
    *,
    run_labels: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One row per method; values match the terminal comparison table."""
    labels = run_labels or {}
    rows: list[dict[str, Any]] = []
    for method in _method_display_order(summaries, methods):
        summary = summaries[method]
        eig = read_eig_metrics(summary)
        rows.append({
            "System": str(labels.get("system_label", "")),
            "Run": str(labels.get("run_name", "")),
            "T": str(labels.get("step_number", "")),
            "N_b": str(labels.get("n_buses", "")),
            "Method": method,
            "sPCE_1..T": format_eig_list(eig["spce_by_step"]),
            "Tot.sPCE": f"{eig['total_spce']:.4f}",
            "ΔH_1..T": format_eig_list(eig["delta_h_by_step"]),
            "ΔH": f"{eig['delta_h']:.4f}",
            "MSE_θ": f"{float(summary.get('mse_theta') or summary.get('mean_mse_theta') or 0.0):.6f}",
            "train_s": _method_train_seconds(method, timing),
            "test_s": _method_test_seconds(method, timing, summary),
        })
    return rows


def save_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    legacy_json = path.parent / "summary.json"
    if legacy_json.is_file():
        legacy_json.unlink()


def load_eval_aggregates(exp_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Per-method metrics and timing from ``eval/<method>.json``."""
    summaries: dict[str, Any] = {}
    test_timing: dict[str, dict[str, float]] = {}
    root = eval_dir(exp_dir)
    if root.is_dir():
        for path in sorted(root.glob("*.json")):
            if path.name in ("summary.json",):
                continue
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)
            method = path.stem
            summaries[method] = slim_method_summary(payload.get("summary", {}))
            if isinstance(payload.get("timing"), dict):
                test_timing[method] = payload["timing"]
    timing_block = {
        "training_seconds": load_training_timing(exp_dir),
        "test_seconds": test_timing,
    }
    return summaries, timing_block


def aggregate_metrics(per_system: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_system:
        return {}
    out: dict[str, Any] = {"theta_sample_size": len(per_system)}

    spce_rows = [read_eig_metrics(r)["spce_by_step"] for r in per_system]
    dh_rows = [read_eig_metrics(r)["delta_h_by_step"] for r in per_system]
    if spce_rows and all(len(x) == len(spce_rows[0]) for x in spce_rows):
        out.update(
            eig_metrics(
                [float(x) for x in np.mean(np.array(spce_rows, dtype=np.float64), axis=0)],
                [float(x) for x in np.mean(np.array(dh_rows, dtype=np.float64), axis=0)],
                float(np.mean([read_eig_metrics(r)["total_spce"] for r in per_system])),
                float(np.mean([read_eig_metrics(r)["delta_h"] for r in per_system])),
            )
        )

    for k in ("mse_theta", "mse_M", "mse_K"):
        vals = [float(r[k]) for r in per_system if k in r]
        if vals:
            out[k] = float(np.mean(vals))
            out[f"std_{k}"] = float(np.std(vals))
    return out


def design_selection_detail(catalog, sequence: list[int]) -> dict[str, Any]:
    """Decode action indices to human-readable per-step design labels."""
    seq = [int(a) for a in sequence]
    steps: list[dict[str, Any]] = []
    for t, a in enumerate(seq):
        d = catalog[a]
        steps.append({
            "step": t + 1,
            "action_index": a,
            "design_id": a + 1,
            "amplitude": float(d.amplitude),
            "bus": int(d.bus),
            "duration": float(d.duration),
        })
    design_ids = [s["design_id"] for s in steps]
    return {
        "action_indices": seq,
        "design_ids": design_ids,
        "design_label": ", ".join(f"design{did}" for did in design_ids),
        "steps": steps,
    }


def aggregate_design_selections(per_system: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize probe sequences chosen per test latent θ."""
    if not per_system:
        return {}

    rows = []
    for i, r in enumerate(per_system):
        sel = r.get("design_selection") or {}
        rows.append({
            "test_index": int(r.get("test_index", i)),
            "action_indices": list(sel.get("action_indices", r.get("sequence", []))),
            "design_ids": list(sel.get("design_ids", [])),
            "design_label": str(sel.get("design_label", "")),
            "steps": list(sel.get("steps", [])),
        })

    keys = [tuple(x["action_indices"]) for x in rows]
    unique_keys = list(dict.fromkeys(keys))
    counts: dict[tuple[int, ...], int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1

    top_key = max(counts, key=counts.get)
    top_row = next(x for x in rows if tuple(x["action_indices"]) == top_key)
    all_same = len(unique_keys) == 1

    return {
        "n_test_systems": len(rows),
        "n_unique_sequences": len(unique_keys),
        "all_test_systems_same_sequence": all_same,
        "shared_sequence": top_row if all_same else None,
        "most_common_sequence": {
            "count": counts[top_key],
            "fraction": float(counts[top_key] / len(rows)),
            **top_row,
        },
        "per_test_selections": rows,
    }


def _print_design_selection_summary(method_name: str, design_summary: dict[str, Any]) -> None:
    n = int(design_summary.get("n_test_systems", 0))
    n_unique = int(design_summary.get("n_unique_sequences", 0))
    if n == 0:
        return
    if design_summary.get("all_test_systems_same_sequence"):
        shared = design_summary["shared_sequence"]
        print(
            f"  {method_name} designs: {shared['design_label']} "
            f"(same for all {n} test θ)",
            flush=True,
        )
        return
    common = design_summary.get("most_common_sequence", {})
    print(
        f"  {method_name} designs: {n_unique} unique sequence(s) / {n} test θ; "
        f"most common ({common.get('count', 0)}/{n}): {common.get('design_label', '')}",
        flush=True,
    )


# --- Experiment dirs / orchestration ---------------------------------------

def make_experiment_dir(
    project_root: Path,
    run_name: str,
    step_number: int,
    *,
    experiment_type: str = "eig_based",
) -> Path:
    exp_dir = project_root / "experiments" / make_experiment_dir_name(
        run_name, experiment_type, step_number
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def setup_experiment_dir(
    cfg: SBOEDConfig,
    project_root: Path,
    exp_dir: Path | None = None,
    *,
    data_path: Path | None = None,
    experiment_type: str = "eig_based",
) -> Path:
    from src.layout import assert_experiments_result_dir

    if exp_dir is None:
        exp_dir = make_experiment_dir(
            project_root,
            cfg.run_slug,
            cfg.step_number,
            experiment_type=experiment_type,
        )
        reset_model_dir(exp_dir)
    else:
        exp_dir = assert_experiments_result_dir(
            exp_dir.resolve(), project_root=project_root
        )
        exp_dir.mkdir(parents=True, exist_ok=True)
        model_dir(exp_dir).mkdir(parents=True, exist_ok=True)
    eval_dir(exp_dir).mkdir(parents=True, exist_ok=True)

    if data_path is None:
        from src.banks.tables import data_dir, is_present, validate_data_bundle

        d = data_dir(project_root, cfg)
        if is_present(d):
            validate_data_bundle(cfg, d)
            data_path = d

    if data_path is None:
        data_path = resolve_data_dir(exp_dir, project_root)

    write_run_config(exp_dir, cfg, data_path, experiment_type=experiment_type)
    return exp_dir


def load_experiment_systems(exp_dir: Path, project_root: Path) -> tuple[list[dict], list[dict]]:
    run = load_experiment_run(exp_dir, project_root)
    return run.train_systems, run.test_systems


def generate_tables(
    cfg: SBOEDConfig,
    project_root: Path,
    exp_dir: Path | None = None,
    *,
    splits: tuple[str, ...] = ("train", "test"),
    theta_ranges: dict[str, tuple[int, int | None]] | None = None,
    experiment_type: str = "eig_based",
) -> tuple[Path, Path, list[dict], list[dict]]:
    data_path = ensure_data(
        project_root, cfg, splits=splits, theta_ranges=theta_ranges,
    )
    train_systems, test_systems = load_split_systems(data_path)
    linked_exp = setup_experiment_dir(
        cfg,
        project_root,
        exp_dir,
        data_path=data_path,
        experiment_type=experiment_type,
    )
    return linked_exp, data_path, train_systems, test_systems


def run_method(
    method_name: str,
    cfg: SBOEDConfig,
    exp_dir: Path,
    train_systems: list[dict],
    test_systems: list[dict],
    rng: np.random.Generator,
    *,
    catalog,
    table_support: TableThetaSupport,
) -> dict[str, Any]:
    """Dispatch one of {dad, myopic, fixed, random} through the shared rollout engine."""
    from src.control.cuda_control import CudaControlEngine
    from src.results.tables import aggregate_control_metrics, save_per_rollout_csv
    from src.control.u_req import ControlSpec
    from src.objectives.eig.methods import (
        ensure_fixed_subset,
        run_dad,
        run_fixed,
        run_myopic,
        run_random,
        support_U_bank,
    )
    from src.domains.swing.design import build_simulator

    if method_name not in {"dad", "myopic", "fixed", "random"}:
        raise ValueError(
            f"Unknown method '{method_name}'. Final methods: dad, myopic, fixed, random."
        )

    control_spec = ControlSpec.from_cfg(cfg)
    U_support = support_U_bank(table_support)
    sim = build_simulator(cfg)
    sim.T_obs_sec = float(control_spec.T_obs_sec)
    sim.ode_dt = float(control_spec.ode_dt)
    sim.fs_hz = float(control_spec.fs_hz)
    control_engine = CudaControlEngine(sim, control_spec)
    t_rollout_0 = time.perf_counter()

    if method_name == "random":
        rollouts = run_random(
            cfg=cfg,
            test_systems=test_systems,
            table_support=table_support,
            U_support=U_support,
            control_spec=control_spec,
            control_engine=control_engine,
            rng=rng,
        )
    elif method_name == "fixed":
        subset = ensure_fixed_subset(
            cfg=cfg,
            exp_dir=exp_dir,
            table_support=table_support,
            U_support=U_support,
            calibration_systems=train_systems[: min(32, len(train_systems))],
            control_spec=control_spec,
            seed=int(cfg.data.get("train_seed", 0)),
        )
        rollouts = run_fixed(
            cfg=cfg,
            test_systems=test_systems,
            table_support=table_support,
            U_support=U_support,
            control_spec=control_spec,
            control_engine=control_engine,
            rng=rng,
            subset=subset,
        )
    elif method_name == "myopic":
        rollouts = run_myopic(
            cfg=cfg,
            test_systems=test_systems,
            table_support=table_support,
            U_support=U_support,
            control_spec=control_spec,
            control_engine=control_engine,
            rng=rng,
        )
    else:
        meta = {
            "n_actions": len(catalog),
            "step_number": cfg.step_number,
            "sigma_y": cfg.sigma_y,
            "experiment_dir": str(exp_dir.resolve()),
        }
        rollouts = run_dad(
            cfg=cfg,
            exp_dir=exp_dir,
            test_systems=test_systems,
            table_support=table_support,
            U_support=U_support,
            control_spec=control_spec,
            control_engine=control_engine,
            rng=rng,
            meta=meta,
        )

    rollout_seconds = float(time.perf_counter() - t_rollout_0)
    summary = aggregate_control_metrics(rollouts)
    n_test = max(len(test_systems), 1)
    summary["test_rollout_seconds_total"] = rollout_seconds
    summary["test_rollout_seconds_per_system"] = float(rollout_seconds / n_test)
    summary["test_total_seconds"] = rollout_seconds
    summary["test_total_seconds_per_system"] = float(rollout_seconds / n_test)
    save_per_rollout_csv(rollouts, eval_dir(exp_dir) / f"{method_name}_per_rollout.csv")
    print(
        f"  {method_name}: mean u_ctrl={summary['mean_u_ctrl']:.4f}  "
        f"safety={summary['safety_rate']:.3f}  excess={summary['mean_excess']:.4f}"
    )
    return {
        "method": method_name,
        "summary": summary,
        "per_system": rollouts,
        "timing": {
            "test_rollout_seconds_total": rollout_seconds,
            "test_rollout_seconds_per_system": float(rollout_seconds / n_test),
            "test_total_seconds": rollout_seconds,
            "test_total_seconds_per_system": float(rollout_seconds / n_test),
        },
    }


def train_dad_policy(
    run: ExperimentRun,
    method_name: str = "dad",
    *,
    reuse_policy: bool | None = None,
    smoke: bool = False,
) -> Path:
    """Train a separate EIG-DAD policy maximizing terminal entropy reduction."""
    if method_name not in {"dad", "dad_spce", "dad_delta_h"}:
        raise ValueError(f"Unsupported DAD method '{method_name}'")
    exp_dir = run.exp_dir
    out = model_dir(exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy_path = out / "dad_eig.pth"
    if reuse_policy is None:
        reuse_policy = bool(run.cfg.raw.get("training", {}).get("reuse_policy", False))
    if policy_path.exists() and reuse_policy:
        print(f"  Reusing existing policy → {policy_path} (training.reuse_policy=true)")
        return policy_path
    if policy_path.exists():
        print(f"  Training fresh policy (replacing {policy_path.name})")
    print(
        f"  Training EIG-DAD (objective=max terminal entropy reduction, "
        f"T={run.cfg.step_number}, {run.meta.n_actions} actions) → {out}"
    )
    policy_meta = {
        **run.policy_meta,
        "experiment_dir": str(exp_dir.resolve()),
        "method": "dad_eig",
        "objective": "terminal_eig",
    }
    cfg = run.cfg
    validate_trajectory_y_sim(run.train_systems, split="train")
    n_router_val = min(32, max(len(run.train_systems) // 8, 1))
    policy_train_systems = run.train_systems[:-n_router_val] or run.train_systems
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DADPolicy(
        run.meta.n_actions, max_steps=cfg.step_number
    ).to(device)
    training = dict(cfg.raw.get("training") or {})
    epochs = 2 if smoke else int(training.get("eig_epochs", 20))
    batch_size = 4 if smoke else int(training.get("batch_size", 16))
    steps_per_epoch = (
        16 if smoke else int(training.get("eig_steps_per_epoch", len(run.train_systems)))
    )
    lr = float(training.get("learning_rate", 1e-3))
    entropy_coef = float(training.get("entropy_coef", 0.01))
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(int(cfg.data.get("train_seed", 0)))
    support = TableThetaSupport.from_train(
        policy_train_systems,
        cfg,
        np.random.default_rng(int(cfg.prior.get("mc_support_seed", 1))),
    )
    h0 = posterior_entropy(normalize_log_weights(support.log_p0))
    baseline = 0.0
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        gains: list[float] = []
        losses: list[float] = []
        for start in range(0, steps_per_epoch, batch_size):
            batch_losses = []
            for _ in range(min(batch_size, steps_per_epoch - start)):
                system = policy_train_systems[
                    int(rng.integers(len(policy_train_systems)))
                ]
                seq, y_list, log_probs, entropies = policy_rollout(
                    policy,
                    device,
                    system,
                    cfg.step_number,
                    run.meta.n_actions,
                )
                log_w = np.array(support.log_p0, dtype=np.float64)
                for action, y_obs in zip(seq, y_list):
                    centres = y_sim_last_step_from_tables(support, [int(action)])
                    log_w += log_gaussian_observation_density(
                        float(y_obs), centres, cfg.sigma_y
                    )
                gain = float(
                    h0
                    - posterior_entropy(normalize_log_weights(log_w))
                )
                baseline = 0.9 * baseline + 0.1 * gain
                advantage = gain - baseline
                loss = -log_probs.sum() * advantage - entropy_coef * entropies.sum()
                batch_losses.append(loss)
                gains.append(gain)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.stack(batch_losses).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(batch_loss.detach().item()))
        row = {
            "epoch": float(epoch + 1),
            "mean_terminal_eig": float(np.mean(gains)),
            "mean_loss": float(np.mean(losses)),
        }
        history.append(row)
        print(
            f"  EIG-DAD epoch={epoch + 1}/{epochs} "
            f"terminal_eig={row['mean_terminal_eig']:.4f}"
        )
    elapsed = time.perf_counter() - started
    payload = {
        "state_dict": policy.state_dict(),
        "meta": policy_meta,
        "objective": "terminal_eig",
        "elapsed_seconds": elapsed,
        "history": history,
    }
    # Both historical names point to the same generic terminal-EIG DAD policy.
    for name in ("dad_eig.pth", "dad_delta_h.pth", "dad_spce.pth"):
        torch.save(payload, out / name)
    (out / "dad_eig_training_metrics.json").write_text(
        json.dumps(
            {
                "objective": "terminal_eig",
                "device": str(device),
                "elapsed_seconds": elapsed,
                "epochs": epochs,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return policy_path


def train_rl_sboed_eig(
    run: ExperimentRun,
    *,
    smoke: bool = False,
    seed: int = 101,
) -> Path:
    """Train EIG-specific RL-sBOED with stepwise entropy-reduction returns."""
    cfg = run.cfg
    validate_trajectory_y_sim(run.train_systems, split="train")
    n_router_val = min(32, max(len(run.train_systems) // 8, 1))
    policy_train_systems = run.train_systems[:-n_router_val] or run.train_systems
    out = model_dir(run.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy_path = out / "rl_sboed_eig.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DADPolicy(run.meta.n_actions, max_steps=cfg.step_number).to(device)
    training = dict(cfg.raw.get("training") or {})
    epochs = 2 if smoke else int(training.get("eig_rl_epochs", 20))
    batch_size = 4 if smoke else int(training.get("batch_size", 16))
    steps_per_epoch = (
        16
        if smoke
        else int(training.get("eig_rl_steps_per_epoch", len(run.train_systems)))
    )
    lr = float(training.get("learning_rate", 1e-3))
    entropy_coef = float(training.get("entropy_coef", 0.01))
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(int(seed))
    support = TableThetaSupport.from_train(
        policy_train_systems,
        cfg,
        np.random.default_rng(int(cfg.prior.get("mc_support_seed", 1))),
    )
    baseline = np.zeros(cfg.step_number, dtype=np.float64)
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        epoch_rewards: list[np.ndarray] = []
        epoch_losses: list[float] = []
        for start in range(0, steps_per_epoch, batch_size):
            batch_losses = []
            for _ in range(min(batch_size, steps_per_epoch - start)):
                system = policy_train_systems[
                    int(rng.integers(len(policy_train_systems)))
                ]
                seq, y_list, log_probs, entropies = policy_rollout(
                    policy,
                    device,
                    system,
                    cfg.step_number,
                    run.meta.n_actions,
                )
                log_w = np.array(support.log_p0, dtype=np.float64)
                rewards = []
                entropy_before = posterior_entropy(normalize_log_weights(log_w))
                for action, y_obs in zip(seq, y_list):
                    centres = y_sim_last_step_from_tables(support, [int(action)])
                    log_w += log_gaussian_observation_density(
                        float(y_obs), centres, cfg.sigma_y
                    )
                    entropy_after = posterior_entropy(
                        normalize_log_weights(log_w)
                    )
                    rewards.append(float(entropy_before - entropy_after))
                    entropy_before = entropy_after
                rewards_a = np.asarray(rewards, dtype=np.float64)
                returns = np.cumsum(rewards_a[::-1])[::-1].copy()
                baseline = 0.9 * baseline + 0.1 * returns
                advantage = torch.as_tensor(
                    returns - baseline, dtype=torch.float32, device=device
                )
                loss = -torch.sum(log_probs * advantage) - entropy_coef * entropies.sum()
                batch_losses.append(loss)
                epoch_rewards.append(rewards_a)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.stack(batch_losses).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(batch_loss.detach().item()))
        rewards_matrix = np.stack(epoch_rewards)
        row = {
            "epoch": float(epoch + 1),
            "mean_terminal_eig": float(rewards_matrix.sum(axis=1).mean()),
            "mean_step_reward": float(rewards_matrix.mean()),
            "mean_loss": float(np.mean(epoch_losses)),
        }
        history.append(row)
        print(
            f"  RL-sBOED-EIG epoch={epoch + 1}/{epochs} "
            f"terminal_eig={row['mean_terminal_eig']:.4f}"
        )
    elapsed = time.perf_counter() - started
    payload = {
        "state_dict": policy.state_dict(),
        "meta": {
            **run.policy_meta,
            "experiment_dir": str(run.exp_dir.resolve()),
            "method": "rl_sboed_eig",
            "objective": "stepwise_entropy_reduction",
        },
        "objective": "stepwise_entropy_reduction",
        "elapsed_seconds": elapsed,
        "history": history,
    }
    torch.save(payload, policy_path)
    (out / "rl_sboed_eig_training_metrics.json").write_text(
        json.dumps(
            {
                "objective": "stepwise_entropy_reduction",
                "device": str(device),
                "elapsed_seconds": elapsed,
                "epochs": epochs,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return policy_path


def load_training_timing(exp_dir: Path) -> dict[str, float]:
    """Read ``model/dad_training_metrics.json`` elapsed times."""
    out: dict[str, float] = {}
    mdir = model_dir(exp_dir)
    path = mdir / "dad_training_metrics.json"
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            out["dad"] = float((json.load(f) or {}).get("elapsed_seconds", 0.0))
    return out


def run_evaluation(
    run: ExperimentRun,
    methods: list[str] | None = None,
    rng: np.random.Generator | None = None,
    training_timing: dict[str, float] | None = None,
) -> dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng(run.meta.test_seed)

    cfg = run.cfg
    exp_dir = run.exp_dir
    if training_timing is None:
        training_timing = load_training_timing(exp_dir)
    train_systems = run.train_systems
    test_systems = run.test_systems

    validate_trajectory_y_sim(train_systems, split="train")
    validate_trajectory_y_sim(test_systems, split="test")

    from src.banks.generate_control import control_banks_certified

    certified, cert_detail = control_banks_certified(run.data_path)
    if not certified:
        raise RuntimeError(
            "Control-bank safety invariants not certified "
            "(oracle / u_max / U-bank particle safety rates must all be 1.0). "
            "Run: python -m src.objectives.eig.cli generate-control-bank --config <config>\n"
            f"Detail: {cert_detail}"
        )
    print(f"  Control-bank certified → {cert_detail}")

    catalog = build_catalog(cfg)
    mc_seed = int(cfg.prior.get("mc_support_seed", run.meta.test_seed))
    table_support = TableThetaSupport.from_train(
        train_systems, cfg, np.random.default_rng(mc_seed),
    )
    eval_root = eval_dir(exp_dir)
    eval_root.mkdir(parents=True, exist_ok=True)
    print(
        f"  Eval from data {run.data_path.name}: experiment T={cfg.step_number}, "
        f"train-table θ support: {len(table_support)} latent θ × {run.meta.n_buses} buses"
    )

    run_methods = methods or cfg.methods

    summaries, _ = load_eval_aggregates(exp_dir)
    comparison_csv = eval_summary_path(exp_dir)
    test_timing: dict[str, dict[str, float]] = {}
    for m in tqdm(run_methods, desc="methods", unit="method"):
        if m not in ALL_METHODS:
            raise ValueError(f"Unknown method '{m}'. Valid: {ALL_METHODS}")
        out_path = eval_method_path(exp_dir, m)
        if out_path.is_file():
            tqdm.write(f"[{m}] skip (already evaluated → {out_path.name})")
            with out_path.open(encoding="utf-8") as f:
                payload_m = json.load(f)
                summaries[m] = slim_method_summary(payload_m["summary"])
                if isinstance(payload_m.get("timing"), dict):
                    test_timing[m] = payload_m["timing"]
                elif isinstance(payload_m["summary"], dict):
                    s = payload_m["summary"]
                    test_timing[m] = {
                        "test_rollout_seconds_total": float(s.get("test_rollout_seconds_total", 0.0)),
                        "test_rollout_seconds_per_system": float(s.get("test_rollout_seconds_per_system", 0.0)),
                        "test_total_seconds": float(s.get("test_total_seconds", 0.0)),
                        "test_total_seconds_per_system": float(s.get("test_total_seconds_per_system", 0.0)),
                    }
            continue

        tqdm.write(f"[{m}]")
        method_result = run_method(
            m,
            cfg,
            exp_dir,
            train_systems,
            test_systems,
            rng,
            catalog=catalog,
            table_support=table_support,
        )
        save_json(method_result, eval_method_path(exp_dir, m))
        summaries[m] = slim_method_summary(method_result["summary"])
        test_timing[m] = method_result.get("timing", {})

    timing_block = {
        "training_seconds": training_timing or {},
        "test_seconds": test_timing,
    }
    from src.results.tables import (
        build_control_table_rows,
        print_control_table,
        save_control_comparison_csv,
    )

    labels = cfg.run_labels()
    labels["step_number"] = cfg.step_number
    labels["n_buses"] = cfg.N
    rows = build_control_table_rows(
        summaries, timing_block, methods=run_methods, run_labels=labels,
    )
    save_control_comparison_csv(rows, comparison_csv)
    print_control_table(rows)
    print(f"\nComparison table → {comparison_csv}")

    return {
        "comparison_csv": str(comparison_csv.resolve()),
        "rows": rows,
        "summaries": summaries,
    }


def print_print_table(rows: list[dict[str, Any]]) -> None:
    if rows and rows[0].get("System"):
        print(
            f"\nSystem={rows[0]['System']}  Run={rows[0].get('Run', '')}  "
            f"T={rows[0].get('T', '')}  N_b={rows[0].get('N_b', '')}"
        )
    spce_w = max((len(r["sPCE_1..T"]) for r in rows), default=12)
    dh_w = max((len(r["ΔH_1..T"]) for r in rows), default=12)
    spce_w = max(spce_w, len("sPCE_1..T"))
    dh_w = max(dh_w, len("ΔH_1..T"))
    print(
        f"\n{'Method':<18} {'sPCE_1..T':<{spce_w}} {'Tot.sPCE':>9} "
        f"{'ΔH_1..T':<{dh_w}} {'ΔH':>9} {'MSE_θ':>10} {'train_s':>8} {'test_s':>8}"
    )
    print("-" * (18 + spce_w + dh_w + 58))
    for row in rows:
        print(
            f"{row['Method']:<18} {row['sPCE_1..T']:<{spce_w}} "
            f"{row['Tot.sPCE']:>9} "
            f"{row['ΔH_1..T']:<{dh_w}} "
            f"{row['ΔH']:>9} {row['MSE_θ']:>10} "
            f"{row['train_s']:>8} {row['test_s']:>8}"
        )


def print_results_table(
    summaries: dict[str, Any],
    *,
    timing: dict[str, Any] | None = None,
    step_number: int | None = None,
) -> None:
    del step_number
    print_print_table(build_print_table_rows(summaries, timing))


def print_experiment_banner(
    cfg: SBOEDConfig,
    exp_dir: Path,
    data_path: Path,
    train_systems: list[dict],
    test_systems: list[dict],
    methods: list[str],
) -> None:
    n_actions = len(build_catalog(cfg))
    print(f"Experiment: {exp_dir.name}")
    print(f"  dir={exp_dir}")
    print(f"  data={data_path}")
    print(
        f"  system={cfg.system_label}  topology={cfg.topology}  "
        f"preset={cfg.config_preset}  run={cfg.run_slug}"
    )
    print(f"  yaml={cfg.config_path.name}  T={cfg.step_number}  amplitudes={cfg.probe_amplitudes}")
    print(
        f"  actions={n_actions}  one_step_rows_per_system={n_actions}  "
        f"BOED_horizon={cfg.step_number}"
    )
    print(f"  theta_dim={2 * cfg.N}  (per-bus M,K on {cfg.N} buses)")
    print(
        f"  train_theta_sample_size={len(train_systems)}  "
        f"test_theta_sample_size={len(test_systems)}"
    )
    print(f"  methods={methods}")


def run_experiment(
    config_path: Path,
    project_root: Path,
    methods: list[str] | None = None,
    exp_dir: Path | None = None,
    step_number: int | None = None,
) -> Path:
    from src.config import load_config_for_run

    cfg = load_config_for_run(config_path, project_root, step_number=step_number)
    data_path = ensure_data(project_root, cfg)
    exp_dir = setup_experiment_dir(cfg, project_root, exp_dir, data_path=data_path)
    run = load_experiment_run(exp_dir, project_root)
    run_methods = methods or run.cfg.methods
    print_experiment_banner(
        run.cfg, run.exp_dir, run.data_path, run.train_systems, run.test_systems, run_methods,
    )
    if "dad" in run_methods:
        train_dad_policy(run, "dad")
    run_evaluation(run, methods=run_methods, training_timing=load_training_timing(exp_dir))
    print(f"EXP_DIR={exp_dir}")
    print(f"DATA_DIR={data_path}")
    return exp_dir


def refresh_eval_summary(exp_dir: Path) -> list[dict[str, Any]]:
    """Rebuild the printed comparison table and save to ``eval/summary.csv``."""
    summaries, timing_block = load_eval_aggregates(exp_dir)
    method_order: list[str] | None = None
    run_labels: dict[str, Any] | None = None
    try:
        run = load_experiment_run(exp_dir, repo_root())
        method_order = list(run.cfg.methods)
        run_labels = run.cfg.run_labels()
    except Exception:
        doc = load_run_config_doc(exp_dir)
        if doc:
            run_labels = {
                k: doc.get(k)
                for k in (
                    "system_label",
                    "topology",
                    "run_name",
                    "config_name",  # legacy manifest key
                    "preset",
                    "config_preset",  # legacy
                    "n_buses",
                    "step_number",
                )
                if doc.get(k) is not None
            }
            if "run_name" not in run_labels and run_labels.get("config_name"):
                name = str(run_labels["config_name"])
                run_labels["run_name"] = name[: -len("_config")] if name.endswith("_config") else name
            if "preset" not in run_labels and run_labels.get("config_preset"):
                run_labels["preset"] = run_labels["config_preset"]
    rows = None
    try:
        from src.results.tables import (
            build_control_table_rows,
            save_control_comparison_csv,
        )

        labels = run_labels or {}
        rows = build_control_table_rows(
            summaries, timing_block, methods=method_order or list(summaries), run_labels=labels,
        )
        csv_path = eval_summary_path(exp_dir)
        save_control_comparison_csv(rows, csv_path)
        return rows
    except Exception:
        rows = build_print_table_rows(
            summaries, timing_block, methods=method_order, run_labels=run_labels,
        )
        csv_path = eval_summary_path(exp_dir)
        save_comparison_csv(rows, csv_path)
        return rows


def eval_experiment(exp_dir: Path) -> list[dict[str, Any]]:
    rows = refresh_eval_summary(exp_dir)
    from src.results.tables import print_control_table

    if rows and "mean_u_ctrl" in rows[0]:
        print_control_table(rows)
    else:
        print_print_table(rows)
    csv_path = eval_summary_path(exp_dir)
    print(f"\nComparison table → {csv_path}")
    return rows
