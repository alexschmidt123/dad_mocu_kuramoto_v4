"""
Experiment pipeline: Foster DAD + iDAD-style tables.

Train/eval: table ``sequence``, noisy ``y``, ``y_sim`` (ODE before noise). π uses ``y`` only.
sPCE / myopic use ``y_sim`` as likelihood centres (no ODE at train/eval).
"""

from __future__ import annotations

import csv
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ALL_METHODS, SBOEDConfig, repo_root
from src.run_context import ExperimentRun, load_experiment_run
from src.data import (
    clear_trajectory_sim_context,
    ensure_data,
    load_split_systems,
    lookup_prefix_y,
    lookup_sequence_y,
    resolve_data_dir,
    save_json,
    set_trajectory_sim_context,
    system_uses_on_demand,
    trajectory_storage_mode,
)
from src.swing_equation_ode.simulator import system_mk
from src.experiment_layout import (
    eval_dir,
    eval_method_path,
    eval_summary_path,
    make_experiment_dir_name,
    model_dir,
    write_run_config,
)
from src.contrastive.spce import (
    log_gaussian_observation_density,
    normalize_log_weights,
    posterior_after_gaussian_observations,
    posterior_entropy,
    posterior_mean_mk_vectors,
)
from src.neural.train import rollout_dad, train_dad_policy as _train_dad_core
from src.swing_equation_ode.design import (
    build_catalog,
    count_no_repeat_sequences,
    masked_action_indices,
    random_valid_sequence,
)
from src.data import validate_trajectory_y_sim
from src.table_scoring import (
    TableThetaSupport,
    spce_eig_from_rollout,
    y_sim_last_step_from_tables,
    y_sim_steps_from_tables,
)


# --- Baselines -------------------------------------------------------------

class Method(ABC):
    name: str

    @abstractmethod
    def run(self, cfg: SBOEDConfig, test_systems: list[dict], rng: np.random.Generator) -> list[dict]:
        """Return list of rollouts: {M, K, sequence, y}."""


class RandomMethod(Method):
    name = "random"

    def run(self, cfg, test_systems, rng):
        catalog = build_catalog(cfg)
        out = []
        for sys in test_systems:
            seq = random_valid_sequence(catalog, cfg.step_number, rng)
            y = lookup_sequence_y(sys, seq)
            out.append({"M": sys["M"], "K": sys["K"], "sequence": seq, "y": y})
        return out


class FixedOpenLoopMethod(Method):
    name = "fixed_open_loop"

    def __init__(self, fixed_sequence: list[int] | None = None):
        self.fixed_sequence = fixed_sequence

    def run(self, cfg, test_systems, rng):
        catalog = build_catalog(cfg)
        seq = self.fixed_sequence or random_valid_sequence(catalog, cfg.step_number, rng)
        out = []
        for sys in test_systems:
            y = lookup_sequence_y(sys, seq)
            out.append({"M": sys["M"], "K": sys["K"], "sequence": list(seq), "y": y})
        return out


class MyopicDeltaHMethod(Method):
    """Greedy ΔH: banked train ``y`` on support; test ``y`` from test table."""

    name = "myopic_delta_h"

    def __init__(self, catalog, table_support: TableThetaSupport):
        self.catalog = catalog
        self.table_support = table_support

    def run(self, cfg, test_systems, rng):
        out = []
        log_p0 = self.table_support.log_p0
        n_sys = len(test_systems)

        for i_sys, sys in enumerate(test_systems):
            if i_sys == 0 or (i_sys + 1) % 10 == 0 or i_sys + 1 == n_sys:
                print(
                    f"  myopic_delta_h (train-table y_sim): latent θ {i_sys + 1}/{n_sys}",
                    flush=True,
                )
            used: set[int] = set()
            seq, y_hist = [], []
            log_unnorm = np.array(log_p0, dtype=np.float64)

            for _ in range(cfg.step_number):
                p_before = normalize_log_weights(log_unnorm)
                H_before = posterior_entropy(p_before)

                feasible = masked_action_indices(used, self.catalog)
                best_a, best_dh = int(feasible[0]), -np.inf
                for a in feasible:
                    trial = seq + [int(a)]
                    m_vals = y_sim_last_step_from_tables(self.table_support, trial)
                    y_hat = float(np.sum(p_before * m_vals))
                    log_L = log_gaussian_observation_density(y_hat, m_vals, cfg.sigma_y)
                    p_after = normalize_log_weights(log_unnorm + log_L)
                    dh = H_before - posterior_entropy(p_after)
                    if dh > best_dh:
                        best_dh, best_a = dh, int(a)

                seq.append(best_a)
                y_hist = lookup_prefix_y(sys, seq)
                y = float(y_hist[-1])
                m_obs = y_sim_last_step_from_tables(self.table_support, seq)
                log_unnorm = log_unnorm + log_gaussian_observation_density(
                    y, m_obs, cfg.sigma_y,
                )
                used.add(best_a)

            out.append({"M": sys["M"], "K": sys["K"], "sequence": seq, "y": y_hist})
        return out


def get_method(
    name: str,
    train_systems: list[dict] | None = None,
    *,
    catalog=None,
    table_support: TableThetaSupport | None = None,
    **kwargs,
) -> Method:
    if name == "random":
        return RandomMethod()
    if name == "fixed_open_loop":
        return FixedOpenLoopMethod(kwargs.get("fixed_sequence"))
    if name == "myopic_delta_h":
        if catalog is None or table_support is None:
            raise ValueError("myopic_delta_h requires catalog and table_support")
        return MyopicDeltaHMethod(catalog, table_support)
    raise ValueError(f"Unknown method: {name}. Use dad_spce/dad_delta_h via scripts/dad_training.sh.")

# --- Metrics ---------------------------------------------------------------


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
        entropy_trace[t] - entropy_trace[t + 1]
        for t in range(len(y_arr))
    ]

    M_arr = np.asarray(system["M"], dtype=np.float64).reshape(-1)
    K_arr = np.asarray(system["K"], dtype=np.float64).reshape(-1)

    mse_M = float(np.mean((M_hat - M_arr) ** 2))
    mse_K = float(np.mean((K_hat - K_arr) ** 2))
    mse_theta = float(np.sum((M_hat - M_arr) ** 2) + np.sum((K_hat - K_arr) ** 2))

    eig = eig_metrics(step_spce_list, step_delta_h, total_spce, H0 - H1)
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
    """Canonical EIG result block: two lists + two scalars."""
    return {
        "spce_by_step": [float(x) for x in spce_by_step],
        "delta_h_by_step": [float(x) for x in delta_h_by_step],
        "total_spce": float(total_spce),
        "delta_h": float(delta_h),
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
    eig = read_eig_metrics(summary)
    out: dict[str, Any] = dict(eig)
    if summary.get("theta_sample_size") is not None:
        out["theta_sample_size"] = int(summary["theta_sample_size"])
    if summary.get("mse_theta") is not None:
        out["mse_theta"] = float(summary["mse_theta"])
    if summary.get("std_mse_theta") is not None:
        out["std_mse_theta"] = float(summary["std_mse_theta"])
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
) -> list[dict[str, Any]]:
    """One row per method; values match the terminal comparison table."""
    rows: list[dict[str, Any]] = []
    for method in _method_display_order(summaries, methods):
        summary = summaries[method]
        eig = read_eig_metrics(summary)
        rows.append({
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

def make_experiment_dir(project_root: Path, config_name: str, step_number: int) -> Path:
    exp_dir = project_root / "experiments" / make_experiment_dir_name(config_name, step_number)
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def setup_experiment_dir(
    cfg: SBOEDConfig,
    project_root: Path,
    exp_dir: Path | None = None,
    *,
    data_path: Path | None = None,
) -> Path:
    if exp_dir is None:
        exp_dir = make_experiment_dir(project_root, cfg.name, cfg.step_number)
    else:
        exp_dir = exp_dir.resolve()
        exp_dir.mkdir(parents=True, exist_ok=True)

    model_dir(exp_dir).mkdir(parents=True, exist_ok=True)
    eval_dir(exp_dir).mkdir(parents=True, exist_ok=True)

    if data_path is None and (project_root / "data").exists():
        from src.data import data_dir

        d = data_dir(project_root, cfg)
        if d.is_dir():
            data_path = d

    if data_path is None:
        data_path = resolve_data_dir(exp_dir, project_root)

    write_run_config(exp_dir, cfg, data_path)
    return exp_dir


def load_experiment_systems(exp_dir: Path, project_root: Path) -> tuple[list[dict], list[dict]]:
    run = load_experiment_run(exp_dir, project_root)
    return run.train_systems, run.test_systems


def generate_tables(
    cfg: SBOEDConfig,
    project_root: Path,
    exp_dir: Path | None = None,
) -> tuple[Path, Path, list[dict], list[dict]]:
    data_path = ensure_data(project_root, cfg)
    train_systems, test_systems = load_split_systems(data_path)
    linked_exp = setup_experiment_dir(cfg, project_root, exp_dir, data_path=data_path)
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
    t_rollout_0 = time.perf_counter()
    if method_name in {"dad_spce", "dad_delta_h"}:
        meta = {
            "n_actions": len(catalog),
            "step_number": cfg.step_number,
            "sigma_y": cfg.sigma_y,
        }
        out = model_dir(exp_dir)
        out.mkdir(parents=True, exist_ok=True)
        policy_path = out / f"{method_name}.pth"
        if not policy_path.is_file():
            raise FileNotFoundError(
                f"DAD policy missing: {policy_path}\n"
                "Train first: ./scripts/dad_training.sh -exp-dir <run>"
            )
        rollouts = rollout_dad(cfg, test_systems, policy_path, meta, rng)
    else:
        fixed_seq = None
        if method_name == "fixed_open_loop" and train_systems:
            trajs = train_systems[0].get("trajectories") or []
            if trajs:
                fixed_seq = list(trajs[0]["sequence"])
        method = get_method(
            method_name,
            train_systems,
            fixed_sequence=fixed_seq,
            catalog=catalog,
            table_support=table_support,
        )
        rollouts = method.run(cfg, test_systems, rng)
    rollout_seconds = float(time.perf_counter() - t_rollout_0)

    t_score_0 = time.perf_counter()
    per_system = []
    for i_sys, (sys, r) in enumerate(zip(test_systems, rollouts)):
        row = evaluate_rollout(
            cfg, sys, r["sequence"], r["y"], catalog, table_support, rng,
        )
        row["test_index"] = i_sys
        row["design_selection"] = design_selection_detail(catalog, r["sequence"])
        per_system.append(row)
    score_seconds = float(time.perf_counter() - t_score_0)
    summary = aggregate_metrics(per_system)
    design_summary = aggregate_design_selections(per_system)
    summary["design_selection"] = design_summary
    n_test = max(len(test_systems), 1)
    summary["test_rollout_seconds_total"] = rollout_seconds
    summary["test_rollout_seconds_per_system"] = float(rollout_seconds / n_test)
    summary["test_scoring_seconds_total"] = score_seconds
    summary["test_scoring_seconds_per_system"] = float(score_seconds / n_test)
    summary["test_total_seconds"] = float(rollout_seconds + score_seconds)
    summary["test_total_seconds_per_system"] = float((rollout_seconds + score_seconds) / n_test)
    eig = read_eig_metrics(summary)
    print(
        f"  {method_name}: {format_eig_line(eig)}  "
        f"MSE_θ={summary.get('mse_theta', 0):.6f}"
    )
    _print_design_selection_summary(method_name, design_summary)
    return {
        "method": method_name,
        "summary": summary,
        "design_selection": design_summary,
        "per_system": per_system,
        "timing": {
            "test_rollout_seconds_total": rollout_seconds,
            "test_rollout_seconds_per_system": float(rollout_seconds / n_test),
            "test_scoring_seconds_total": score_seconds,
            "test_scoring_seconds_per_system": float(score_seconds / n_test),
            "test_total_seconds": float(rollout_seconds + score_seconds),
            "test_total_seconds_per_system": float((rollout_seconds + score_seconds) / n_test),
        },
    }


def train_dad_policy(run: ExperimentRun, method_name: str = "dad_spce") -> Path:
    """Train selected DAD variant; policy file is ``<method_name>.pth``."""
    if method_name not in {"dad_spce", "dad_delta_h"}:
        raise ValueError(f"Unsupported DAD method '{method_name}'")
    exp_dir = run.exp_dir
    out = model_dir(exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy_path = out / f"{method_name}.pth"
    force = bool(run.cfg.raw.get("training", {}).get("force_retrain", False))
    if policy_path.exists() and not force:
        print(f"  Policy already exists → {policy_path} (set training.force_retrain: true to retrain)")
        return policy_path
    objective = "reinforce" if method_name == "dad_spce" else "delta_h"
    run.cfg.raw.setdefault("training", {})["objective"] = objective
    print(
        f"  Training {method_name} (objective={objective}, data T={run.meta.step_number}, "
        f"{run.meta.n_actions} actions) → {out}"
    )
    return _train_dad_core(
        run.cfg, run.train_systems, run.policy_meta, out, data_dir=run.data_path, run_tag=method_name,
    )


def load_training_timing(exp_dir: Path) -> dict[str, float]:
    """Read ``model/*_training_metrics.json`` elapsed times (used by ``run.sh`` eval phase)."""
    out: dict[str, float] = {}
    mdir = model_dir(exp_dir)
    for method in ("dad_spce", "dad_delta_h"):
        path = mdir / f"{method}_training_metrics.json"
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                out[method] = float((json.load(f) or {}).get("elapsed_seconds", 0.0))
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

    on_demand = system_uses_on_demand(train_systems[0]) if train_systems else False
    if on_demand:
        set_trajectory_sim_context(cfg, int(cfg.data.get("test_seed", 1)))
        print(f"  trajectory_mode=on_demand (PyCUDA sim at lookup; T={cfg.step_number})")

    catalog = build_catalog(cfg)
    mc_seed = int(cfg.prior.get("mc_support_seed", run.meta.test_seed))
    table_support = TableThetaSupport.from_train(
        train_systems, cfg, np.random.default_rng(mc_seed),
    )
    eval_root = eval_dir(exp_dir)
    eval_root.mkdir(parents=True, exist_ok=True)
    print(
        f"  Eval from data {run.data_path.name}: T={run.meta.step_number}, "
        f"train-table θ support: {len(table_support)} latent θ × {run.meta.n_buses} buses"
    )

    run_methods = methods or cfg.methods

    summaries, _ = load_eval_aggregates(exp_dir)
    comparison_csv = eval_summary_path(exp_dir)
    test_timing: dict[str, dict[str, float]] = {}
    for m in run_methods:
        if m not in ALL_METHODS:
            raise ValueError(f"Unknown method '{m}'. Valid: {ALL_METHODS}")
        out_path = eval_method_path(exp_dir, m)
        if out_path.is_file():
            print(f"[{m}] skip (already evaluated → {out_path.name})")
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

        print(f"[{m}]")
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
    rows = build_print_table_rows(summaries, timing_block, methods=run_methods)
    save_comparison_csv(rows, comparison_csv)
    print_print_table(rows)
    print(f"\nComparison table → {comparison_csv}")
    clear_trajectory_sim_context()
    return {"comparison_csv": str(comparison_csv.resolve()), "rows": rows}


def print_print_table(rows: list[dict[str, Any]]) -> None:
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
    storage_mode = trajectory_storage_mode(cfg)
    n_seq = (
        count_no_repeat_sequences(n_actions, cfg.step_number)
        if storage_mode == "full_bank"
        else 0
    )
    print(f"Experiment: {exp_dir.name}")
    print(f"  dir={exp_dir}")
    print(f"  data={data_path}")
    print(f"  config={cfg.config_path.name}  T={cfg.step_number}  amplitudes={cfg.probe_amplitudes}")
    seq_label = (
        f"sequences_per_system={n_seq}"
        if storage_mode == "full_bank"
        else "trajectories=on_demand PyCUDA"
    )
    print(f"  actions={n_actions}  {seq_label}")
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
    if "dad_spce" in run_methods:
        train_dad_policy(run, "dad_spce")
    if "dad_delta_h" in run_methods:
        train_dad_policy(run, "dad_delta_h")
    run_evaluation(run, methods=run_methods, training_timing=load_training_timing(exp_dir))
    print(f"EXP_DIR={exp_dir}")
    print(f"DATA_DIR={data_path}")
    return exp_dir


def refresh_eval_summary(exp_dir: Path) -> list[dict[str, Any]]:
    """Rebuild the printed comparison table and save to ``eval/summary.csv``."""
    summaries, timing_block = load_eval_aggregates(exp_dir)
    method_order: list[str] | None = None
    try:
        run = load_experiment_run(exp_dir, repo_root())
        method_order = list(run.cfg.methods)
    except Exception:
        pass
    rows = build_print_table_rows(summaries, timing_block, methods=method_order)
    csv_path = eval_summary_path(exp_dir)
    save_comparison_csv(rows, csv_path)
    return rows


def eval_experiment(exp_dir: Path) -> list[dict[str, Any]]:
    rows = refresh_eval_summary(exp_dir)
    print_print_table(rows)
    csv_path = eval_summary_path(exp_dir)
    print(f"\nComparison table → {csv_path}")
    return rows
