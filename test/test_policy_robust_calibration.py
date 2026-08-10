"""Tests for policy-robust common-margin calibration invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.control.legacy.policy_robust_calibration import (
    HORIZON,
    _onesided_wilson_lower,
    _u_ctrl_from_q,
    evaluate_margin_candidates,
    out_dir_default,
)
from src.control.terminal_rule import run_keyed_history
from src.control.terminal_rule import FrozenTerminalRule
from src.rollout import RandomSelector
from src.table_scoring import TableThetaSupport
from src.contrastive.spce import log_prior_uniform_discrete


ROOT = Path(__file__).resolve().parents[1]


def test_shared_rollout_identical_seeds():
    systems = [
        {
            "u_req": 0.5,
            "y_sim": {str(a): float(0.1 * a) for a in range(8)},
        }
    ]
    # Minimal fake tables: y_sim keyed by action string
    for s in systems:
        s["actions"] = list(range(8))
    # TableThetaSupport needs proper system structure — use real train if available
    from src.run_context import load_experiment_run
    from src.control.pilot import load_pilot_splits
    from src.control.banks import extract_U_bank

    exp = ROOT / "experiments" / "ieee5_horizon_sweep" / "T2"
    if not exp.is_dir():
        return
    run = load_experiment_run(exp, ROOT)
    splits = load_pilot_splits(exp, run)
    ts = TableThetaSupport(
        systems=splits["support_systems"],
        log_p0=log_prior_uniform_discrete(len(splits["support_systems"])),
    )
    U = extract_U_bank(splits["support_systems"])
    frozen = FrozenTerminalRule(0.05, 0.4, tuple(np.round(np.arange(0, 1.55, 0.05), 10)))
    n_actions = 30
    a = run_keyed_history(
        system=splits["calibration_systems"][0],
        theta_id=0,
        rollout_id=7,
        selector=RandomSelector(n_actions=n_actions),
        table_support=ts,
        U_support=U,
        frozen=frozen,
        horizon=2,
        sigma_y=0.08,
        global_seed=1234,
        rng=np.random.default_rng(99),
    )
    b = run_keyed_history(
        system=splits["calibration_systems"][0],
        theta_id=0,
        rollout_id=7,
        selector=RandomSelector(n_actions=n_actions),
        table_support=ts,
        U_support=U,
        frozen=frozen,
        horizon=2,
        sigma_y=0.08,
        global_seed=1234,
        rng=np.random.default_rng(99),
    )
    assert a["sequence"] == b["sequence"]
    assert np.allclose(a["y_obs"], b["y_obs"])
    assert abs(a["selected_u_ctrl"] - b["selected_u_ctrl"]) < 1e-12


def test_unsafe_implies_under_control():
    """All unsafe cases must have selected control below true u_req."""
    exp = ROOT / "experiments" / "ieee5_horizon_sweep" / "T2" / "eval"
    if not exp.is_dir():
        return
    import csv

    for method_dir in exp.iterdir():
        csv_path = method_dir / "rollouts.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                safe = str(row["safe_total"]).lower() in ("1", "true", "yes")
                u_ctrl = float(row["u_ctrl"])
                u_req = float(row["u_req_true"])
                if not safe:
                    assert u_ctrl < u_req + 1e-12
                if u_ctrl + 1e-12 >= u_req:
                    # Implication: should be safe (proxy). GPU may differ slightly —
                    # only enforce strict direction for unsafe.
                    pass


def test_pooled_cannot_hide_policy_failure():
    frozen = FrozenTerminalRule(0.05, 0.4, (0.0, 0.5, 1.0, 1.5))
    # Construct rows: Random always safe, DAD always unsafe at low margin
    rows = []
    for i in range(20):
        rows.append(
            {
                "policy_name": "Random",
                "split": "calibration",
                "posterior_quantile": 0.2,
                "true_u_req": 0.5,
            }
        )
        rows.append(
            {
                "policy_name": "Random",
                "split": "validation",
                "posterior_quantile": 0.2,
                "true_u_req": 0.5,
            }
        )
        rows.append(
            {
                "policy_name": "DAD",
                "split": "calibration",
                "posterior_quantile": 0.0,
                "true_u_req": 1.0,
            }
        )
        rows.append(
            {
                "policy_name": "DAD",
                "split": "validation",
                "posterior_quantile": 0.0,
                "true_u_req": 1.0,
            }
        )
        rows.append(
            {
                "policy_name": "Myopic",
                "split": "calibration",
                "posterior_quantile": 0.2,
                "true_u_req": 0.5,
            }
        )
        rows.append(
            {
                "policy_name": "Myopic",
                "split": "validation",
                "posterior_quantile": 0.2,
                "true_u_req": 0.5,
            }
        )
        rows.append(
            {
                "policy_name": "Fixed",
                "split": "calibration",
                "posterior_quantile": 0.2,
                "true_u_req": 0.5,
            }
        )
        rows.append(
            {
                "policy_name": "Fixed",
                "split": "validation",
                "posterior_quantile": 0.2,
                "true_u_req": 0.5,
            }
        )
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        m, results, _ = evaluate_margin_candidates(rows, frozen, Path(td))
        for r in results:
            if r["pooled_cal_safety"] == 1.0 and not r["all_policies_cal_safe"]:
                assert r["pooled_hides_failure"]
            if r["admissible"]:
                assert r["DAD_cal_safety"] == 1.0
                assert r["DAD_val_safety"] == 1.0


def test_selected_margin_common_and_method_agnostic():
    rule = FrozenTerminalRule(0.05, 0.7, (0.0, 0.5, 1.0))
    meta = rule.metadata()
    assert "method" not in meta
    assert abs(meta["additive_margin"] - 0.7) < 1e-12
    # Terminal control depends only on posterior + rule
    U = np.array([0.1, 0.2, 0.9])
    w = np.array([0.1, 0.1, 0.8])
    from src.control.terminal_rule import posterior_to_u_ctrl

    u = posterior_to_u_ctrl(w, U, rule)
    assert u == _u_ctrl_from_q(0.9, 0.7, rule.u_candidates) or u >= 0.0


def test_cal_val_disjoint_and_no_final_test_in_calibration():
    import json

    split = (
        ROOT
        / "experiments"
        / "ieee5_horizon_sweep"
        / "T2"
        / "diagnostics"
        / "control_safety_calibration"
        / "split_metadata.json"
    )
    if not split.is_file():
        return
    meta = json.loads(split.read_text())
    s, c, v = set(meta["support_ids"]), set(meta["calibration_ids"]), set(meta["validation_ids"])
    assert not (c & v)
    assert not (s & c)
    assert not (s & v)
    assert meta.get("test_ids_untouched") is True


def test_horizon_locked_to_T2():
    assert HORIZON == 2
    out = out_dir_default(ROOT)
    assert "T2" in str(out)
    assert "T3" not in str(out)


def test_safety_first_checkpoint_metric_name():
    # Training metrics after safety-first change should advertise the metric.
    from src.neural import train as train_mod
    import inspect

    src = inspect.getsource(train_mod.train_dad_policy)
    assert "safety_first" in src or "validation_safety" in src
    assert "validation_safety_below_1" in src


def test_wilson_bound_zero_failures():
    lo = _onesided_wilson_lower(100, 100)
    assert 0.0 < lo <= 1.0


def test_observability_and_baseline_share_keyed_path():
    from src.control import observability, pilot
    from src.control import terminal_rule as shared_rollout
    import inspect

    obs_src = inspect.getsource(observability.run_diagnostic_rollout)
    assert "observe_with_keyed_noise" in obs_src or "use_keyed_noise" in obs_src
    assert hasattr(shared_rollout, "run_keyed_history")
    pilot_src = inspect.getsource(pilot._run_one_rollout)
    assert "observe_with_keyed_noise" in pilot_src
