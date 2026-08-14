"""Stepwise ΔH / EIG scoring with fresh Gaussian noise on banked y_sim centres."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import SBOEDConfig
from src.inference.spce import (
    clamp_info_gain,
    log_gaussian_observation_density,
    normalize_log_weights,
    posterior_entropy,
)
from src.banks.tables import lookup_action_y
from src.banks.tables import lookup_action_y_sim
from src.reporting.run_context import model_dir
from src.objectives.eig.dad_rollout import rollout_dad
from src.domains.swing.design import build_catalog
from src.inference.scoring import TableThetaSupport, y_sim_last_step_from_tables

from src.objectives.eig.pipeline import (
    default_fixed_sequence,
    design_selection_detail,
    get_method,
)

STEPWISE_METHODS: tuple[str, ...] = (
    "dad_eig",
    "rl_sboed_eig",
    "moe_sboed",
    "myopic_delta_h",
    "random",
    "fixed_open_loop",
)

METHOD_LABELS: dict[str, str] = {
    "dad_eig": "DAD-EIG",
    "rl_sboed_eig": "RL-sBOED-EIG",
    "moe_sboed": "MoE-sBOED",
    "myopic_delta_h": "Myopic ΔH",
    "random": "Random",
    "fixed_open_loop": "Fixed design",
}


def _noise_seed(master_seed: int, rollout_idx: int, step_idx: int) -> int:
    return int((hash((int(master_seed), int(rollout_idx), int(step_idx))) & 0xFFFFFFFF))


def fresh_noise(master_seed: int, rollout_idx: int, step_idx: int, sigma_y: float) -> float:
    """Shared ε_{r,t} across methods for matched rollouts."""
    rng = np.random.default_rng(_noise_seed(master_seed, rollout_idx, step_idx))
    return float(rng.normal(0.0, sigma_y))


def collect_method_rollouts(
    method_name: str,
    cfg: SBOEDConfig,
    exp_dir,
    train_systems: list[dict[str, Any]],
    test_systems: list[dict[str, Any]],
    rng: np.random.Generator,
    catalog,
    table_support: TableThetaSupport,
) -> list[dict[str, Any]]:
    """Return ``[{M, K, sequence, y}, ...]`` using existing decision logic only."""
    if method_name == "moe_sboed":
        from src.policies.moe import rollout_eig_moe

        return rollout_eig_moe(
            cfg, Path(exp_dir), test_systems, catalog, table_support
        )
    if method_name in {"dad_eig", "rl_sboed_eig"}:
        policy_path = model_dir(exp_dir) / f"{method_name}.pth"
        if not policy_path.is_file():
            raise FileNotFoundError(
                f"Learned policy missing for stepwise EIG: {policy_path}\n"
                "Train the requested method before evaluation."
            )
        meta = {
            "n_actions": len(catalog),
            "step_number": cfg.step_number,
            "sigma_y": cfg.sigma_y,
            "experiment_dir": str(exp_dir.resolve()),
        }
        return rollout_dad(
            cfg,
            test_systems,
            policy_path,
            meta,
            rng,
            expected_experiment_dir=exp_dir,
        )

    fixed_seq = (
        default_fixed_sequence(cfg, catalog)
        if method_name == "fixed_open_loop"
        else None
    )
    method = get_method(
        method_name,
        train_systems,
        fixed_sequence=fixed_seq,
        catalog=catalog,
        table_support=table_support,
    )
    return method.run(cfg, test_systems, rng)


def score_single_step_delta_h(
    y_obs: float,
    action: int,
    log_unnorm: np.ndarray,
    table_support: TableThetaSupport,
    sigma_y: float,
) -> tuple[float, float, np.ndarray]:
    """One Bayesian update; return (H_before, delta_H, log_unnorm_after)."""
    p_before = normalize_log_weights(log_unnorm)
    H_before = posterior_entropy(p_before)
    f_vals = y_sim_last_step_from_tables(table_support, [int(action)])
    log_L = log_gaussian_observation_density(float(y_obs), f_vals, sigma_y)
    log_after = log_unnorm + log_L
    p_after = normalize_log_weights(log_after)
    H_after = posterior_entropy(p_after)
    return H_before, clamp_info_gain(H_before - H_after), log_after


def score_rollout_stepwise_eig(
    cfg: SBOEDConfig,
    system: dict[str, Any],
    sequence: list[int],
    table_support: TableThetaSupport,
    *,
    rollout_idx: int,
    noise_seed: int,
) -> dict[str, Any]:
    """
    Sequential ΔH with fresh noise on banked y_sim observations.

    Realized step-(t) reduction: δH_t = H_{t-1} - H_t (not clipped).
    """
    seq = [int(a) for a in sequence]
    log_unnorm = np.array(table_support.log_p0, dtype=np.float64)
    entropy_trace: list[float] = [posterior_entropy(normalize_log_weights(log_unnorm))]
    delta_h: list[float] = []
    y_clean: list[float] = []
    y_noisy: list[float] = []
    actions: list[int] = []

    for step_idx, action in enumerate(seq):
        y_c = float(lookup_action_y_sim(system, action))
        eps = fresh_noise(noise_seed, rollout_idx, step_idx, cfg.sigma_y)
        y_t = y_c + eps
        H_before, dh, log_unnorm = score_single_step_delta_h(
            y_t, action, log_unnorm, table_support, cfg.sigma_y,
        )
        del H_before
        entropy_trace.append(entropy_trace[-1] - dh)
        delta_h.append(float(dh))
        y_clean.append(y_c)
        y_noisy.append(float(y_t))
        actions.append(int(action))

    H0 = float(entropy_trace[0])
    HT = float(entropy_trace[-1])
    terminal_from_steps = float(np.sum(delta_h))
    terminal_from_entropy = float(H0 - HT)
    return {
        "sequence": seq,
        "actions": actions,
        "y_clean": y_clean,
        "y_noisy": y_noisy,
        "entropy_trace": [float(x) for x in entropy_trace],
        "delta_h_by_step": delta_h,
        "H0": H0,
        "HT": HT,
        "terminal_eig_from_steps": terminal_from_steps,
        "terminal_eig_from_entropy": terminal_from_entropy,
        "terminal_eig_abs_diff": abs(terminal_from_steps - terminal_from_entropy),
    }


def compute_step1_heatmap(
    cfg: SBOEDConfig,
    test_systems: list[dict[str, Any]],
    catalog,
    table_support: TableThetaSupport,
    *,
    noise_seed: int,
) -> dict[str, Any]:
    """
    EIG_1(ξ) = E[H_0 - H_1 | ξ=(b,a)] averaged over test θ* rollouts.
    """
    n_actions = len(catalog)
    per_action_delta_h: list[list[float]] = [[] for _ in range(n_actions)]

    log_p0 = np.array(table_support.log_p0, dtype=np.float64)
    for rollout_idx, system in enumerate(test_systems):
        for action in range(n_actions):
            y_t = float(lookup_action_y(system, action))
            _, dh, _ = score_single_step_delta_h(
                y_t, action, log_p0.copy(), table_support, cfg.sigma_y,
            )
            per_action_delta_h[action].append(float(dh))

    rows: list[dict[str, Any]] = []
    matrix: dict[tuple[int, float], float] = {}
    buses = sorted({int(d.bus) for d in catalog})
    amps = sorted({float(d.amplitude) for d in catalog})

    for action, design in enumerate(catalog):
        vals = np.asarray(per_action_delta_h[action], dtype=np.float64)
        mean_dh = float(np.mean(vals)) if vals.size else 0.0
        std_dh = float(np.std(vals)) if vals.size > 1 else 0.0
        sem = std_dh / np.sqrt(max(vals.size, 1))
        row = {
            "action_index": action,
            "bus": int(design.bus),
            "amplitude": float(design.amplitude),
            "duration": float(design.duration),
            "mean_eig_step1": mean_dh,
            "std_eig_step1": std_dh,
            "sem_eig_step1": float(sem),
            "ci95_low": mean_dh - 1.96 * sem,
            "ci95_high": mean_dh + 1.96 * sem,
            "n_rollouts": int(vals.size),
        }
        rows.append(row)
        matrix[(int(design.bus), float(design.amplitude))] = mean_dh

    grid = np.full((len(buses), len(amps)), np.nan, dtype=np.float64)
    bus_index = {b: i for i, b in enumerate(buses)}
    amp_index = {a: j for j, a in enumerate(amps)}
    for (bus, amp), val in matrix.items():
        grid[bus_index[bus], amp_index[amp]] = val

    return {
        "rows": rows,
        "buses": buses,
        "amplitudes": amps,
        "grid": grid,
    }


def aggregate_method_stepwise(
    rollout_records: list[dict[str, Any]],
    *,
    method: str,
    catalog,
) -> dict[str, Any]:
    """Mean stepwise EIG, terminal EIG, and consistency checks across rollouts."""
    if not rollout_records:
        raise ValueError(f"No rollouts for method {method}")

    T = len(rollout_records[0]["delta_h_by_step"])
    dh_matrix = np.array([r["delta_h_by_step"] for r in rollout_records], dtype=np.float64)
    entropy_matrix = np.array([r["entropy_trace"] for r in rollout_records], dtype=np.float64)
    terminal_steps = np.array([r["terminal_eig_from_steps"] for r in rollout_records], dtype=np.float64)
    terminal_ent = np.array([r["terminal_eig_from_entropy"] for r in rollout_records], dtype=np.float64)
    terminal_diff = np.array([r["terminal_eig_abs_diff"] for r in rollout_records], dtype=np.float64)

    mean_step = np.mean(dh_matrix, axis=0)
    std_step = np.std(dh_matrix, axis=0)
    sem_step = std_step / np.sqrt(max(dh_matrix.shape[0], 1))

    mean_terminal = float(np.mean(terminal_steps))
    std_terminal = float(np.std(terminal_steps))
    sem_terminal = std_terminal / np.sqrt(max(len(terminal_steps), 1))

    per_rollout = []
    for i, rec in enumerate(rollout_records):
        sel = design_selection_detail(catalog, rec["sequence"])
        per_rollout.append({
            "rollout_index": i,
            "sequence": rec["sequence"],
            "design_selection": sel,
            "entropy_trace": rec["entropy_trace"],
            "delta_h_by_step": rec["delta_h_by_step"],
            "y_clean": rec["y_clean"],
            "y_noisy": rec["y_noisy"],
            "terminal_eig_from_steps": rec["terminal_eig_from_steps"],
            "terminal_eig_from_entropy": rec["terminal_eig_from_entropy"],
            "terminal_eig_abs_diff": rec["terminal_eig_abs_diff"],
        })

    step_summary = []
    for t in range(T):
        step_summary.append({
            "step": t + 1,
            "mean_eig": float(mean_step[t]),
            "std_eig": float(std_step[t]),
            "sem_eig": float(sem_step[t]),
            "ci95_low": float(mean_step[t] - 1.96 * sem_step[t]),
            "ci95_high": float(mean_step[t] + 1.96 * sem_step[t]),
        })

    return {
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "n_rollouts": len(rollout_records),
        "T": T,
        "H0_mean": float(np.mean(entropy_matrix[:, 0])),
        "step_summary": step_summary,
        "mean_eig_by_step": [float(x) for x in mean_step],
        "per_rollout": per_rollout,
        "terminal_eig_mean": mean_terminal,
        "terminal_eig_std": std_terminal,
        "terminal_eig_sem": float(sem_terminal),
        "terminal_eig_ci95_low": mean_terminal - 1.96 * sem_terminal,
        "terminal_eig_ci95_high": mean_terminal + 1.96 * sem_terminal,
        "terminal_from_entropy_mean": float(np.mean(terminal_ent)),
        "terminal_consistency_max_abs_diff": float(np.max(terminal_diff)),
        "terminal_consistency_mean_abs_diff": float(np.mean(terminal_diff)),
    }
