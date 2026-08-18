"""Four control-objective methods: dad, myopic, fixed, random."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.banks.control_u import extract_U_bank
from src.control.cuda_control import CudaControlEngine
from src.control.fixed_search import load_fixed_search, save_fixed_search, search_fixed_subset
from src.control.myopic import MyopicControlSelector
from src.control.u_req import ControlSpec
from src.objectives.eig.dad_rollout import rollout_dad
from src.objectives.eig.rollout import FixedSelector, RandomSelector, RolloutResult, run_shared_rollout
from src.domains.swing.design import build_catalog
from src.banks.tables import TableThetaSupport


METHOD_NAMES = ("dad", "myopic", "fixed", "random")


def _results_to_rollouts(results: list[RolloutResult]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in results:
        out.append(
            {
                "theta_test_id": r.theta_test_id,
                "M": r.M,
                "K": r.K,
                "sequence": list(r.sequence),
                "y": list(r.y_obs),
                "u_ctrl": r.u_ctrl,
                "u_req_true": r.u_req_true,
                "excess_control": r.excess_control,
                "max_rocof": r.max_rocof,
                "frequency_nadir": r.frequency_nadir,
                "rocof_safe": r.rocof_safe,
                "nadir_safe": r.nadir_safe,
                "safe_total": r.safe_total,
                "control_metrics": dict(r.control_metrics),
                "weights_sum": float(np.sum(r.weights)),
            }
        )
    return out


def _roll_all(
    *,
    test_systems: list[dict],
    make_selector,
    cfg,
    table_support,
    U_support,
    control_spec,
    control_engine,
    rng,
) -> list[dict[str, Any]]:
    catalog = build_catalog(cfg)
    results = []
    for i, sys in enumerate(test_systems):
        selector = make_selector(i, sys)
        results.append(
            run_shared_rollout(
                system=sys,
                table_support=table_support,
                U_support=U_support,
                selector=selector,
                horizon=cfg.step_number,
                n_actions=len(catalog),
                sigma_y=cfg.sigma_y,
                control_spec=control_spec,
                rng=rng,
                control_engine=control_engine,
                theta_test_id=i,
            )
        )
    return _results_to_rollouts(results)


def run_random(
    *,
    cfg,
    test_systems: list[dict],
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    control_spec: ControlSpec,
    control_engine: CudaControlEngine,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    catalog = build_catalog(cfg)
    return _roll_all(
        test_systems=test_systems,
        make_selector=lambda _i, _s: RandomSelector(n_actions=len(catalog)),
        cfg=cfg,
        table_support=table_support,
        U_support=U_support,
        control_spec=control_spec,
        control_engine=control_engine,
        rng=rng,
    )


def run_fixed(
    *,
    cfg,
    test_systems: list[dict],
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    control_spec: ControlSpec,
    control_engine: CudaControlEngine,
    rng: np.random.Generator,
    subset: list[int],
) -> list[dict[str, Any]]:
    seq = sorted(int(a) for a in subset)
    if len(seq) != cfg.step_number:
        raise ValueError(f"fixed subset length {len(seq)} != T={cfg.step_number}")
    return _roll_all(
        test_systems=test_systems,
        make_selector=lambda _i, _s: FixedSelector(sequence=list(seq)),
        cfg=cfg,
        table_support=table_support,
        U_support=U_support,
        control_spec=control_spec,
        control_engine=control_engine,
        rng=rng,
    )


def run_myopic(
    *,
    cfg,
    test_systems: list[dict],
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    control_spec: ControlSpec,
    control_engine: CudaControlEngine,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    catalog = build_catalog(cfg)
    n_hyp = int(control_spec.myopic_hypothetical)

    def make_sel(_i, _s):
        return MyopicControlSelector(
            table_support=table_support,
            U_support=U_support,
            n_actions=len(catalog),
            sigma_y=cfg.sigma_y,
            alpha=control_spec.alpha,
            n_hypothetical=n_hyp,
            safety_margin=float(getattr(control_spec, "safety_margin", 0.0)),
            u_candidates=tuple(control_spec.u_candidates),
        )

    return _roll_all(
        test_systems=test_systems,
        make_selector=make_sel,
        cfg=cfg,
        table_support=table_support,
        U_support=U_support,
        control_spec=control_spec,
        control_engine=control_engine,
        rng=rng,
    )


def run_dad(
    *,
    cfg,
    exp_dir: Path,
    test_systems: list[dict],
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    control_spec: ControlSpec,
    control_engine: CudaControlEngine,
    rng: np.random.Generator,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    from src.layout import model_dir

    policy_path = model_dir(exp_dir) / "dad.pth"
    if not policy_path.is_file():
        raise FileNotFoundError(f"Missing DAD policy at {policy_path}")

    dad_rollouts = rollout_dad(
        cfg,
        test_systems,
        policy_path,
        meta,
        rng,
        expected_experiment_dir=exp_dir,
    )

    def make_sel(i, _s):
        seq = [int(a) for a in dad_rollouts[i]["sequence"]]
        return FixedSelector(sequence=seq)

    return _roll_all(
        test_systems=test_systems,
        make_selector=make_sel,
        cfg=cfg,
        table_support=table_support,
        U_support=U_support,
        control_spec=control_spec,
        control_engine=control_engine,
        rng=rng,
    )


def ensure_fixed_subset(
    *,
    cfg,
    exp_dir: Path,
    table_support: TableThetaSupport,
    U_support: np.ndarray,
    calibration_systems: list[dict],
    control_spec: ControlSpec,
    seed: int = 0,
) -> list[int]:
    from src.layout import eval_dir

    out_path = eval_dir(exp_dir) / "fixed_subset_search.json"
    if out_path.is_file():
        return load_fixed_search(out_path).subset

    catalog = build_catalog(cfg)
    rng = np.random.default_rng(seed)
    result = search_fixed_subset(
        n_actions=len(catalog),
        horizon=cfg.step_number,
        table_support=table_support,
        U_support=U_support,
        calibration_systems=calibration_systems,
        sigma_y=cfg.sigma_y,
        alpha=control_spec.alpha,
        rng=rng,
        exhaustive_threshold=int(control_spec.fixed_exhaustive_threshold),
        noise_replicas=int(control_spec.fixed_noise_replicas),
        greedy_restarts=int(control_spec.fixed_greedy_restarts),
        seed=seed,
        margin=float(getattr(control_spec, "safety_margin", 0.0)),
        u_grid=control_spec.u_candidates,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fixed_search(result, out_path)
    return result.subset


def support_U_bank(table_support: TableThetaSupport) -> np.ndarray:
    return extract_U_bank(table_support.systems)
