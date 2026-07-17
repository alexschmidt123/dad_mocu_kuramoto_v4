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

try:  # pragma: no cover - cosmetic progress helper
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)
    tqdm.write = print

from src.config import ALL_METHODS, SBOEDConfig, repo_root
from src.run_context import ExperimentRun, load_experiment_run
from src.data import (
    ensure_data,
    load_split_systems,
    lookup_action_y,
    lookup_sequence_y,
    resolve_data_dir,
    save_json,
)
from src.swing_equation_ode.simulator import system_mk
from src.experiment_layout import (
    eval_dir,
    eval_method_path,
    eval_summary_path,
    load_run_config_doc,
    make_experiment_dir_name,
    model_dir,
    reset_model_dir,
    write_run_config,
)
from src.contrastive.spce import (
    clamp_info_gain,
    log_gaussian_observation_density,
    normalize_log_weights,
    posterior_after_gaussian_observations,
    posterior_entropy,
    posterior_mean_mk_vectors,
)
from src.neural.train import rollout_dad, train_dad_policy as _train_dad_core
from src.swing_equation_ode.design import (
    build_catalog,
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


def default_fixed_sequence(cfg: SBOEDConfig, catalog) -> list[int]:
    """
    Deterministic open-loop baseline: spread probes across buses at the middle
    configured amplitude. This avoids catalog-order bias such as always taking
    the first adjacent bus actions.
    """
    if cfg.step_number <= 0:
        return []
    if cfg.step_number > len(catalog):
        raise ValueError(f"T={cfg.step_number} exceeds action catalog size={len(catalog)}")

    amplitudes = sorted({float(d.amplitude) for d in catalog})
    target_amp = amplitudes[len(amplitudes) // 2]
    if cfg.N == 1:
        target_buses = [0]
    else:
        target_buses = [
            int(round(x))
            for x in np.linspace(0, cfg.N - 1, num=min(cfg.step_number, cfg.N))
        ]

    seq: list[int] = []
    used: set[int] = set()
    for bus in target_buses:
        for i, design in enumerate(catalog):
            if i in used:
                continue
            if int(design.bus) == int(bus) and abs(float(design.amplitude) - target_amp) < 1e-12:
                seq.append(i)
                used.add(i)
                break
        if len(seq) >= cfg.step_number:
            return seq

    for i in range(len(catalog)):
        if i not in used:
            seq.append(i)
            used.add(i)
        if len(seq) >= cfg.step_number:
            break
    return seq


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
        seq = self.fixed_sequence or default_fixed_sequence(cfg, catalog)
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

        for sys in tqdm(test_systems, desc="myopic eval", unit="θ", leave=False):
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
                    dh = clamp_info_gain(H_before - posterior_entropy(p_after))
                    if dh > best_dh:
                        best_dh, best_a = dh, int(a)

                seq.append(best_a)
                y = lookup_action_y(sys, best_a)
                y_hist.append(float(y))
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

def make_experiment_dir(project_root: Path, run_name: str, step_number: int) -> Path:
    exp_dir = project_root / "experiments" / make_experiment_dir_name(run_name, step_number)
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
        exp_dir = make_experiment_dir(project_root, cfg.run_slug, cfg.step_number)
        reset_model_dir(exp_dir)
    else:
        exp_dir = exp_dir.resolve()
        exp_dir.mkdir(parents=True, exist_ok=True)
        model_dir(exp_dir).mkdir(parents=True, exist_ok=True)
    eval_dir(exp_dir).mkdir(parents=True, exist_ok=True)

    if data_path is None:
        from src.data import data_dir, is_present, validate_data_bundle

        d = data_dir(project_root, cfg)
        if is_present(d):
            validate_data_bundle(cfg, d)
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
    *,
    splits: tuple[str, ...] = ("train", "test"),
    theta_ranges: dict[str, tuple[int, int | None]] | None = None,
) -> tuple[Path, Path, list[dict], list[dict]]:
    data_path = ensure_data(
        project_root, cfg, splits=splits, theta_ranges=theta_ranges,
    )
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
    """Dispatch one of {dad, myopic, fixed, random} through the shared rollout engine."""
    from src.control.cuda_control import CudaControlEngine
    from src.control.eval_metrics import aggregate_control_metrics, save_per_rollout_csv
    from src.control.u_req import ControlSpec
    from src.methods import (
        ensure_fixed_subset,
        run_dad,
        run_fixed,
        run_myopic,
        run_random,
        support_U_bank,
    )
    from src.swing_equation_ode.design import build_simulator

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
) -> Path:
    """Train DAD policy minimizing E[u_ctrl]; saves ``model/dad.pth``."""
    if method_name not in {"dad", "dad_spce", "dad_delta_h"}:
        # Accept legacy aliases but always train the control-objective dad.
        raise ValueError(f"Unsupported DAD method '{method_name}'")
    method_name = "dad"
    exp_dir = run.exp_dir
    out = model_dir(exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy_path = out / "dad.pth"
    if reuse_policy is None:
        reuse_policy = bool(run.cfg.raw.get("training", {}).get("reuse_policy", False))
    if policy_path.exists() and reuse_policy:
        print(f"  Reusing existing policy → {policy_path} (training.reuse_policy=true)")
        return policy_path
    if policy_path.exists():
        print(f"  Training fresh policy (replacing {policy_path.name})")
    print(
        f"  Training dad (objective=min E[u_ctrl], experiment T={run.cfg.step_number}, "
        f"{run.meta.n_actions} actions) → {out}"
    )
    policy_meta = {
        **run.policy_meta,
        "experiment_dir": str(exp_dir.resolve()),
        "method": "dad",
    }
    # Hold out a small validation slice from train for checkpoint selection.
    n_val = max(4, min(16, len(run.train_systems) // 8))
    val_systems = run.train_systems[-n_val:]
    train_systems = run.train_systems[:-n_val] if len(run.train_systems) > n_val else run.train_systems
    return _train_dad_core(
        run.cfg,
        train_systems,
        policy_meta,
        out,
        data_dir=run.data_path,
        run_tag="dad",
        validation_systems=val_systems,
    )


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

    from src.control.generate import control_banks_certified

    certified, cert_detail = control_banks_certified(run.data_path)
    if not certified:
        raise RuntimeError(
            "Control-bank safety invariants not certified "
            "(oracle / u_max / U-bank particle safety rates must all be 1.0). "
            "Run: python -m src.cli generate-control-bank --config <config>\n"
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
    from src.control.eval_metrics import (
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
        from src.control.eval_metrics import (
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
    from src.control.eval_metrics import print_control_table

    if rows and "mean_u_ctrl" in rows[0]:
        print_control_table(rows)
    else:
        print_print_table(rows)
    csv_path = eval_summary_path(exp_dir)
    print(f"\nComparison table → {csv_path}")
    return rows
