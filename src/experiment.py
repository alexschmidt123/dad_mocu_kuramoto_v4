"""
Experiment pipeline: Foster DAD + iDAD-style tables.

Train/eval: table ``sequence``, noisy ``y``, ``y_sim`` (ODE before noise). π uses ``y`` only.
sPCE / myopic use ``y_sim`` as likelihood centres (no ODE at train/eval).
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ALL_METHODS, SBOEDConfig, repo_root
from src.run_context import ExperimentRun, load_experiment_run
from src.data import (
    ensure_data,
    load_split_systems,
    lookup_prefix_y,
    lookup_sequence_y,
    resolve_data_dir,
    save_json,
)
from src.swing_equation_ode.simulator import system_mk
from src.experiment_layout import (
    eval_dir,
    eval_method_path,
    eval_summary_path,
    load_eval_summary,
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

    step_spce_list, step_spce_mean, total_spce = spce_eig_from_rollout(
        cfg, seq, y_arr, system, table_support, rng,
    )
    step_delta_h = [
        float(posterior_entropy(p_trace[t]) - posterior_entropy(p_trace[t + 1]))
        for t in range(len(y_arr))
    ]

    M_arr = np.asarray(system["M"], dtype=np.float64).reshape(-1)
    K_arr = np.asarray(system["K"], dtype=np.float64).reshape(-1)

    mse_M = float(np.mean((M_hat - M_arr) ** 2))
    mse_K = float(np.mean((K_hat - K_arr) ** 2))
    mse_theta = float(np.sum((M_hat - M_arr) ** 2) + np.sum((K_hat - K_arr) ** 2))

    return {
        "sequence": sequence,
        "y": y_arr.tolist(),
        "M_true": M_arr.tolist(),
        "K_true": K_arr.tolist(),
        "M_hat": M_hat.tolist(),
        "K_hat": K_hat.tolist(),
        "H_prior": H0,
        "H_posterior": H1,
        "delta_H": H0 - H1,
        "step_spce_eig": step_spce_list,
        "step_delta_h": step_delta_h,
        "stepwise_spce_eig": step_spce_mean,
        "total_spce_eig": total_spce,
        "mse_M": mse_M,
        "mse_K": mse_K,
        "mse_theta": mse_theta,
    }


def aggregate_metrics(per_system: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_system:
        return {}
    keys = [
        "stepwise_spce_eig", "total_spce_eig",
        "H_posterior", "delta_H", "mse_M", "mse_K", "mse_theta",
    ]
    out: dict[str, Any] = {"theta_sample_size": len(per_system)}
    for k in keys:
        vals = [float(r[k]) for r in per_system if k in r]
        out[f"mean_{k}"] = float(np.mean(vals)) if vals else None
        if vals:
            out[f"std_{k}"] = float(np.std(vals))

    for field, out_key in (("step_spce_eig", "mean_spce_eig_by_step"), ("step_delta_h", "mean_delta_h_by_step")):
        rows = [r[field] for r in per_system if field in r and r[field]]
        if not rows:
            continue
        n_steps = len(rows[0])
        if all(len(x) == n_steps for x in rows):
            by_step = np.array(rows, dtype=np.float64)
            out[out_key] = [float(x) for x in by_step.mean(axis=0)]
            out[f"std_{out_key}"] = [float(x) for x in by_step.std(axis=0)]
    return out


# --- Experiment dirs / orchestration ---------------------------------------

def make_experiment_dir(project_root: Path, config_name: str) -> Path:
    stamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    exp_dir = project_root / "experiments" / f"{config_name}_{stamp}"
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
        exp_dir = make_experiment_dir(project_root, cfg.name)
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
            first = train_systems[0]["trajectories"][0]
            fixed_seq = list(first["sequence"])
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
    per_system = [
        evaluate_rollout(
            cfg, sys, r["sequence"], r["y"], catalog, table_support, rng,
        )
        for sys, r in zip(test_systems, rollouts)
    ]
    score_seconds = float(time.perf_counter() - t_score_0)
    summary = aggregate_metrics(per_system)
    n_test = max(len(test_systems), 1)
    summary["test_rollout_seconds_total"] = rollout_seconds
    summary["test_rollout_seconds_per_system"] = float(rollout_seconds / n_test)
    summary["test_scoring_seconds_total"] = score_seconds
    summary["test_scoring_seconds_per_system"] = float(score_seconds / n_test)
    summary["test_total_seconds"] = float(rollout_seconds + score_seconds)
    summary["test_total_seconds_per_system"] = float((rollout_seconds + score_seconds) / n_test)
    step_spce = summary.get("mean_spce_eig_by_step")
    step_dh = summary.get("mean_delta_h_by_step")
    step_str = ""
    if step_spce:
        step_str = "  step-sPCE=" + ",".join(f"{v:.4f}" for v in step_spce)
    if step_dh:
        step_str += "  step-ΔH=" + ",".join(f"{v:.4f}" for v in step_dh)
    print(
        f"  {method_name}: Tot.sPCE={summary.get('mean_total_spce_eig', 0):.4f}  "
        f"ΔH={summary.get('mean_delta_H', 0):.4f}  "
        f"MSE_θ={summary.get('mean_mse_theta', 0):.6f}{step_str}"
    )
    return {
        "method": method_name,
        "summary": summary,
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
    train_systems = run.train_systems
    test_systems = run.test_systems

    validate_trajectory_y_sim(train_systems, split="train")
    validate_trajectory_y_sim(test_systems, split="test")

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

    existing: dict[str, Any] = {}
    summary_file = eval_summary_path(exp_dir)
    if summary_file.is_file():
        with summary_file.open(encoding="utf-8") as f:
            existing = json.load(f).get("methods", {})

    summaries: dict[str, Any] = dict(existing)
    test_timing: dict[str, dict[str, float]] = {}
    for m in run_methods:
        if m not in ALL_METHODS:
            raise ValueError(f"Unknown method '{m}'. Valid: {ALL_METHODS}")
        out_path = eval_method_path(exp_dir, m)
        if out_path.is_file():
            print(f"[{m}] skip (already evaluated → {out_path.name})")
            with out_path.open(encoding="utf-8") as f:
                payload_m = json.load(f)
                summaries[m] = payload_m["summary"]
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
        summaries[m] = method_result["summary"]
        test_timing[m] = method_result.get("timing", {})

    payload = {
        "experiment_dir": str(exp_dir.resolve()),
        "data_dir": str(run.data_path.resolve()),
        "data_slug": run.meta.data_slug,
        "step_number": run.meta.step_number,
        "theta_dim": run.meta.theta_dim,
        "eval_mc_samples": len(table_support),
        "methods": summaries,
        "timing": {
            "training_seconds": training_timing or {},
            "test_seconds": test_timing,
        },
    }
    save_json(payload, summary_file)
    print_results_table(summaries)
    return payload


def print_results_table(summaries: dict[str, Any]) -> None:
    n_steps = 0
    for s in summaries.values():
        by = s.get("mean_spce_eig_by_step") or []
        n_steps = max(n_steps, len(by))
    step_hdr = "".join(f" {'sPCE'+str(t+1):>8}" for t in range(n_steps))
    print(f"\n{'Method':<18}{step_hdr} {'Tot.sPCE':>10} {'ΔH':>10} {'MSE_θ':>12}")
    print("-" * (18 + 8 * n_steps + 34))
    for method, s in summaries.items():
        by = s.get("mean_spce_eig_by_step") or []
        step_cols = "".join(
            f" {(by[t] if t < len(by) else 0):8.4f}" for t in range(n_steps)
        )
        print(
            f"{method:<18}{step_cols} "
            f"{(s.get('mean_total_spce_eig') or 0):10.4f} "
            f"{(s.get('mean_delta_H') or 0):10.4f} "
            f"{(s.get('mean_mse_theta') or 0):12.6f}"
        )


def print_experiment_banner(
    cfg: SBOEDConfig,
    exp_dir: Path,
    data_path: Path,
    train_systems: list[dict],
    test_systems: list[dict],
    methods: list[str],
) -> None:
    n_actions = len(build_catalog(cfg))
    n_seq = count_no_repeat_sequences(n_actions, cfg.step_number)
    print(f"Experiment: {exp_dir.name}")
    print(f"  dir={exp_dir}")
    print(f"  data={data_path}")
    print(f"  config={cfg.config_path.name}  T={cfg.step_number}  amplitudes={cfg.probe_amplitudes}")
    print(f"  actions={n_actions}  sequences_per_system={n_seq}")
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
    training_timing: dict[str, float] = {}
    if "dad_spce" in run_methods:
        train_dad_policy(run, "dad_spce")
    if "dad_delta_h" in run_methods:
        train_dad_policy(run, "dad_delta_h")
    for m in ("dad_spce", "dad_delta_h"):
        if m not in run_methods:
            continue
        metrics_path = model_dir(exp_dir) / f"{m}_training_metrics.json"
        if metrics_path.is_file():
            with metrics_path.open(encoding="utf-8") as f:
                training_timing[m] = float((json.load(f) or {}).get("elapsed_seconds", 0.0))
    run_evaluation(run, methods=run_methods, training_timing=training_timing)
    print(f"EXP_DIR={exp_dir}")
    print(f"DATA_DIR={data_path}")
    return exp_dir


def eval_experiment(exp_dir: Path) -> dict[str, Any]:
    payload = load_eval_summary(exp_dir)
    print_results_table(payload.get("methods", {}))
    return payload
