"""IEEE5 T=3 controlled experiment with frozen policy-robust terminal rule (margin 0.55)."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.control.observability import check_objective_observability, evaluate_gate, ObservabilityGateConfig
from src.control.pilot import dad_adaptation_table, run_pilot
from src.control.safety_calibration import install_frozen_terminal_rule
from src.control.terminal_rule import load_frozen_terminal_rule
from src.run_context import load_experiment_run

FROZEN_MARGIN = 0.55
FROZEN_ALPHA = 0.05
DEFAULT_RULE = (
    "experiments/ieee5_policy_robust_calibration_T2/selected_policy_robust_rule.json"
)
DEFAULT_SPLIT = (
    "experiments/ieee5_horizon_sweep/T2/diagnostics/control_safety_calibration/"
    "split_metadata.json"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def prepare_ieee5_t3(
    *,
    root: Path,
    exp_dir: Path,
    config_name: str = "ieee5_config",
    frozen_rule_path: Path | None = None,
) -> Path:
    """Create experiments/ieee5_T3 with T=3 tables and frozen margin-0.55 rule."""
    from src.config import load_config_for_run
    from src.control.generate import generate_control_bank
    from src.experiment import generate_tables

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config_for_run(config_name, root, step_number=3)
    # Pin frozen calibration mode into the experiment config.
    cfg.raw.setdefault("control_safety_calibration", {})
    cfg.raw["control_safety_calibration"]["mode"] = "frozen"
    cfg.raw["control_safety_calibration"]["frozen_rule_path"] = str(
        frozen_rule_path or (root / DEFAULT_RULE)
    )
    cfg.raw["control_safety_calibration"]["expected_alpha"] = FROZEN_ALPHA
    cfg.raw["control_safety_calibration"]["expected_margin"] = FROZEN_MARGIN
    cfg.raw.setdefault("control", {})
    cfg.raw["control"]["alpha"] = FROZEN_ALPHA
    cfg.raw["control"]["safety_margin"] = FROZEN_MARGIN

    generate_tables(cfg, root, exp_dir)
    generate_control_bank(config_name, splits=("train", "test"))

    rule_src = Path(frozen_rule_path or (root / DEFAULT_RULE))
    if not rule_src.is_absolute():
        rule_src = root / rule_src
    split_src = root / DEFAULT_SPLIT
    meta = install_frozen_terminal_rule(
        exp_dir,
        rule_src,
        expected_alpha=FROZEN_ALPHA,
        expected_margin=FROZEN_MARGIN,
        split_source=split_src if split_src.is_file() else None,
    )
    frozen = load_frozen_terminal_rule(exp_dir, expected_margin=FROZEN_MARGIN)
    if abs(frozen.margin - FROZEN_MARGIN) > 1e-12:
        raise RuntimeError(f"Expected margin {FROZEN_MARGIN}, got {frozen.margin}")

    # Pin run_config.yaml
    rc = exp_dir / "run_config.yaml"
    data = yaml.safe_load(rc.read_text()) or {}
    data["step_number"] = 3
    data.setdefault("control", {})
    data["control"]["alpha"] = FROZEN_ALPHA
    data["control"]["safety_margin"] = FROZEN_MARGIN
    data["control"]["u_candidates"] = list(frozen.u_candidates)
    data.setdefault("control_safety_calibration", {})
    data["control_safety_calibration"]["mode"] = "frozen"
    data["control_safety_calibration"]["frozen_rule_path"] = str(rule_src.resolve())
    data["control_safety_calibration"]["expected_alpha"] = FROZEN_ALPHA
    data["control_safety_calibration"]["expected_margin"] = FROZEN_MARGIN
    if cfg.raw.get("myopic"):
        data["myopic"] = cfg.raw["myopic"]
        if data["myopic"].get("n_hypothetical") is not None:
            data["control"]["myopic_hypothetical"] = int(data["myopic"]["n_hypothetical"])
    if cfg.raw.get("pilot"):
        data["pilot"] = cfg.raw["pilot"]
    rc.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    _write_json(exp_dir / "frozen_rule_install.json", meta)
    return exp_dir


def dad_t3_adaptivity(rows: list[dict[str, Any]], n_bins: int = 5) -> dict[str, Any]:
    """T=3 adaptivity: action/obs bins for steps 1→2→3 and dominant-sequence rate."""
    base = dad_adaptation_table(rows, n_bins=n_bins)
    # Second-obs → third action
    y2_vals = [float(r["y_obs"][1]) for r in rows if len(r.get("y_obs", [])) >= 2]
    third_buckets: dict[tuple[int, int, int], Counter] = {}
    from collections import defaultdict

    third_buckets_d: dict[tuple[int, int], Counter] = defaultdict(Counter)
    if y2_vals:
        edges = np.quantile(y2_vals, np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        for r in rows:
            seq = r["sequence"]
            y = r["y_obs"]
            if len(seq) < 3 or len(y) < 2:
                continue
            b = int(np.searchsorted(edges, float(y[1]), side="right") - 1)
            b = min(max(b, 0), n_bins - 1)
            third_buckets_d[(int(seq[1]), b)][int(seq[2])] += 1
    third_table = []
    for (a1, b), ctr in sorted(third_buckets_d.items()):
        total = sum(ctr.values())
        most = ctr.most_common(1)[0][0]
        third_table.append(
            {
                "second_action": a1,
                "second_observation_bin": b,
                "most_common_third_action": most,
                "third_action_distribution": {str(k): v / total for k, v in ctr.items()},
                "n": total,
            }
        )
    seq_counts = Counter(tuple(r["sequence"]) for r in rows if r.get("sequence"))
    n = len(rows)
    dominant_seq, dominant_n = (seq_counts.most_common(1)[0] if seq_counts else ((), 0))
    return {
        "first_to_second": base,
        "second_to_third": third_table,
        "dominant_sequence": list(dominant_seq),
        "dominant_sequence_fraction": float(dominant_n / n) if n else float("nan"),
        "n_unique_sequences": int(len(seq_counts)),
        "effectively_nonadaptive": bool(dominant_n / n >= 0.90) if n else False,
    }


def write_t3_outputs(exp_dir: Path, pilot_report: dict[str, Any]) -> None:
    """Normalize pilot outputs into the required ieee5_T3 layout."""
    eval_root = exp_dir / "eval"
    frozen = load_frozen_terminal_rule(exp_dir, expected_margin=FROZEN_MARGIN)

    # dad_seed_summary.csv
    seed_rows = []
    for seed, ss in (pilot_report.get("dad_seed_summaries") or {}).items():
        seed_rows.append(
            {
                "seed": seed,
                "mean_u_ctrl": ss.get("mean_u_ctrl"),
                "true_safety_rate": ss.get("true_safety_rate"),
                "mean_excess_control": ss.get("mean_excess_control"),
                "mean_runtime_per_rollout": ss.get("mean_runtime_per_rollout"),
                "terminal_rule_hash": frozen.terminal_rule_hash,
            }
        )
    _write_csv(eval_root / "dad_seed_summary.csv", seed_rows)

    # paired_comparisons.csv
    pair_rows = []
    for name, stats in (pilot_report.get("paired_differences") or {}).items():
        pair_rows.append({"contrast": name, **stats})
    _write_csv(eval_root / "paired_comparisons.csv", pair_rows)

    # Adaptivity (T=3)
    dad_rows_path = eval_root / "dad" / "rollouts.csv"
    dad_rows = []
    if dad_rows_path.is_file():
        with dad_rows_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seq = [int(x) for x in str(r.get("sequence", "")).split() if x != ""]
                # y_obs not in compact csv — load from summary adaptation or rebuild lightly
                dad_rows.append({"sequence": seq, "y_obs": []})
    # Prefer full rows from summary.json if adaptation already written
    adapt_path = eval_root / "dad" / "adaptation.json"
    # Rebuild from paired if needed — run_pilot already wrote adaptation for 1→2.
    # Recompute with y_obs from a richer source: re-read summary doesn't have y.
    # Load from dad seed directory rollouts if they store y — they don't.
    # Keep existing adaptation.json and add sequence dominance from sequences.
    seq_counts = Counter()
    if dad_rows_path.is_file():
        with dad_rows_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seq = tuple(int(x) for x in str(r.get("sequence", "")).split() if x != "")
                if seq:
                    seq_counts[seq] += 1
    n_seq = sum(seq_counts.values())
    dom_seq, dom_n = (seq_counts.most_common(1)[0] if seq_counts else ((), 0))
    adapt_extra = {
        "dominant_sequence": list(dom_seq),
        "dominant_sequence_fraction": float(dom_n / n_seq) if n_seq else float("nan"),
        "n_unique_sequences": int(len(seq_counts)),
        "effectively_nonadaptive": bool(n_seq and dom_n / n_seq >= 0.90),
        "terminal_rule_hash": frozen.terminal_rule_hash,
    }
    if adapt_path.is_file():
        existing = json.loads(adapt_path.read_text())
        if isinstance(existing, list):
            payload = {"first_to_second": existing, **adapt_extra}
        else:
            payload = {**existing, **adapt_extra}
    else:
        payload = adapt_extra
    _write_json(adapt_path, payload)

    # Myopic tie diagnostics if present on summary
    myopic = (pilot_report.get("summaries") or {}).get("myopic") or {}
    tie_diag = myopic.get("tie_diagnostics") or {}
    if tie_diag:
        _write_json(eval_root / "myopic" / "tie_diagnostics.json", tie_diag)

    # T3_report.md
    obs = {}
    obs_path = (
        exp_dir / "diagnostics" / "objective_observability" / "observability_summary.json"
    )
    if obs_path.is_file():
        obs = json.loads(obs_path.read_text())
    metas = json.loads((eval_root / "method_metadata.json").read_text()) if (
        eval_root / "method_metadata.json"
    ).is_file() else {}
    lines = [
        "# IEEE5 T=3 controlled experiment report",
        "",
        f"**Passed: {pilot_report.get('pilot_passed')}**",
        "",
        "## Frozen terminal rule",
        "",
        f"- α = `{frozen.alpha}`",
        f"- additive_margin = `{frozen.margin}`",
        f"- terminal_rule_hash = `{frozen.terminal_rule_hash}`",
        f"- control_grid_hash = `{frozen.control_grid_hash}`",
        "",
        "### Per-method rule hashes",
        "",
    ]
    for name, m in (metas.get("methods") or {}).items():
        lines.append(
            f"- `{name}`: `{m.get('terminal_rule_hash')}` "
            f"(α={m.get('alpha')}, margin={m.get('additive_margin')})"
        )
    lines.extend(
        [
            "",
            "## Objective observability",
            "",
            f"- true_safety_rate: `{obs.get('true_safety_rate')}`",
            f"- unique_final_u_ctrl_count: `{obs.get('unique_final_u_ctrl_count')}`",
            f"- final_u_ctrl_std: `{obs.get('final_u_ctrl_std')}`",
            f"- fraction_changed_from_prior: `{obs.get('fraction_changed_from_prior')}`",
            f"- real Spearman: `{obs.get('real_spearman')}`",
            f"- shuffled Spearman: `{obs.get('shuffled_spearman')}`",
            f"- gate_passed: `{obs.get('gate_passed')}`",
            "",
            "## Per-method results",
            "",
            "| method | mean u_ctrl | safety | mean excess | runtime |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for m in ("dad", "myopic", "fixed", "random"):
        s = (pilot_report.get("summaries") or {}).get(m) or {}
        lines.append(
            f"| {m} | {s.get('mean_u_ctrl', float('nan')):.4f} | "
            f"{s.get('true_safety_rate', float('nan')):.3f} | "
            f"{s.get('mean_excess_control', float('nan')):.4f} | "
            f"{s.get('mean_runtime_per_rollout', float('nan')):.4f}s |"
        )
    lines.extend(["", "## DAD seeds", ""])
    for seed, ss in sorted((pilot_report.get("dad_seed_summaries") or {}).items()):
        lines.append(
            f"- seed {seed}: mean_u={ss.get('mean_u_ctrl'):.4f} "
            f"safety={ss.get('true_safety_rate'):.3f}"
        )
    lines.append(
        f"- selected primary: seed "
        f"`{(pilot_report.get('summaries') or {}).get('dad', {}).get('seed')}` "
        f"(safety-first then min val mean u_ctrl)"
    )
    fixed = pilot_report.get("fixed_subset") or {}
    lines.extend(
        [
            "",
            "## Fixed subset",
            "",
            f"- search_mode: `{fixed.get('search_mode')}`",
            f"- selected_action_ids: `{fixed.get('selected_action_ids')}`",
            f"- subsets_evaluated: `{fixed.get('number_of_subsets_evaluated')}`",
            f"- estimated_mean_u_ctrl: `{fixed.get('estimated_mean_u_ctrl')}`",
            f"- search_runtime: `{fixed.get('search_runtime')}`",
            f"- search_seed: `{fixed.get('search_seed')}`",
            "",
            "## Random uniformity",
            "",
            f"`{json.dumps(pilot_report.get('random_uniformity') or {}, indent=2)}`",
            "",
            "## Paired differences",
            "",
        ]
    )
    for k, v in (pilot_report.get("paired_differences") or {}).items():
        lines.append(
            f"- `{k}`: mean={v.get('mean_paired_diff'):.4f} "
            f"CI95=[{v.get('ci95_low'):.4f},{v.get('ci95_high'):.4f}] "
            f"frac_A_lower={v.get('fraction_a_lower'):.3f} "
            f"frac_B_lower={v.get('fraction_b_lower'):.3f} "
            f"tied={v.get('fraction_tied'):.3f}"
        )
    lines.extend(
        [
            "",
            "## DAD adaptivity",
            "",
            f"- dominant_sequence: `{adapt_extra.get('dominant_sequence')}`",
            f"- dominant_sequence_fraction: `{adapt_extra.get('dominant_sequence_fraction')}`",
            f"- n_unique_sequences: `{adapt_extra.get('n_unique_sequences')}`",
            f"- effectively_nonadaptive: `{adapt_extra.get('effectively_nonadaptive')}`",
            "",
            "## T=4 resume",
            "",
        ]
    )
    safeties = [
        ((pilot_report.get("summaries") or {}).get(m) or {}).get("true_safety_rate")
        for m in ("dad", "myopic", "fixed", "random")
    ]
    can_t4 = bool(
        pilot_report.get("pilot_passed")
        and all(s is not None and abs(float(s) - 1.0) < 1e-12 for s in safeties)
    )
    lines.append(
        f"**{'Yes' if can_t4 else 'No'}** — T=4 may proceed only if all four "
        f"methods have safety 1.0 under the frozen rule (current: {safeties})."
    )
    (eval_root / "T3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Also update pilot_report title file
    (eval_root / "pilot_report.md").write_text(
        (eval_root / "T3_report.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def run_ieee5_t3(
    *,
    project_root: Path | None = None,
    exp_dir: str | Path | None = None,
    frozen_rule_path: str | Path | None = None,
    evaluation_rollouts: int | None = None,
    skip_observability: bool = False,
) -> dict[str, Any]:
    from src.config import repo_root

    root = Path(project_root or repo_root())
    exp_dir = Path(exp_dir or (root / "experiments" / "ieee5_T3"))
    # Never overwrite T=2 policy-robust or legacy pilot trees.
    assert "ieee5_T3" in str(exp_dir) or str(exp_dir).endswith("T3")

    print("=== Prepare IEEE5 T=3 (frozen margin 0.55) ===", flush=True)
    prepare_ieee5_t3(
        root=root,
        exp_dir=exp_dir,
        frozen_rule_path=Path(frozen_rule_path) if frozen_rule_path else None,
    )
    frozen = load_frozen_terminal_rule(exp_dir, expected_margin=FROZEN_MARGIN)
    print(
        f"  Confirmed frozen rule hash={frozen.terminal_rule_hash} "
        f"α={frozen.alpha} margin={frozen.margin}",
        flush=True,
    )

    if not skip_observability:
        print("\n=== Objective observability (T=3) ===", flush=True)
        obs = check_objective_observability(exp_dir, project_root=root)
        run = load_experiment_run(exp_dir, root)
        gate = ObservabilityGateConfig.from_cfg(run.cfg)
        # Prefer on-disk summary
        obs_path = (
            exp_dir / "diagnostics" / "objective_observability" / "observability_summary.json"
        )
        if obs_path.is_file():
            obs = json.loads(obs_path.read_text())
        gate_result = evaluate_gate(obs, gate)
        if not gate_result.get("passed", False):
            stop = {
                "stop_reason": "objective_observability_failed",
                "gate": gate_result,
                "exp_dir": str(exp_dir),
            }
            _write_json(exp_dir / "stop_status.json", stop)
            raise RuntimeError(
                f"T=3 objective-observability gate FAILED: {gate_result.get('failed_checks')}. "
                "Stopping before DAD training."
            )
        print("  Observability gate PASS", flush=True)

    print("\n=== Four-method T=3 experiment (DAD/Myopic/Fixed/Random) ===", flush=True)
    report = run_pilot(
        exp_dir,
        project_root=root,
        debug_one_seed=False,
        n_eval_rollouts=evaluation_rollouts,
    )
    # Stop conditions
    stop_reasons = []
    for m in ("dad", "myopic", "fixed", "random"):
        s = (report.get("summaries") or {}).get(m) or {}
        sr = s.get("true_safety_rate")
        if sr is None or abs(float(sr) - 1.0) > 1e-12:
            stop_reasons.append(f"{m}_safety_{sr}")
    shared = (report.get("shared_rule_assert") or {})
    if not shared:
        stop_reasons.append("terminal_rule_hash_mismatch")
    # Check hashes equal across methods
    meta_path = exp_dir / "eval" / "method_metadata.json"
    if meta_path.is_file():
        metas = json.loads(meta_path.read_text())
        hashes = {
            k: v.get("terminal_rule_hash")
            for k, v in (metas.get("methods") or {}).items()
            if not str(k).startswith("dad_seed") or k == "dad"
        }
        # Include all dad seeds too
        all_hashes = {
            k: v.get("terminal_rule_hash")
            for k, v in (metas.get("methods") or {}).items()
        }
        if len(set(all_hashes.values())) != 1:
            stop_reasons.append("terminal_rule_hashes_differ")

    write_t3_outputs(exp_dir, report)

    status = {
        "exp_dir": str(exp_dir),
        "T": 3,
        "frozen_margin": FROZEN_MARGIN,
        "terminal_rule_hash": frozen.terminal_rule_hash,
        "pilot_passed": bool(report.get("pilot_passed")),
        "stop_reasons": stop_reasons,
        "can_proceed_to_T4": bool(report.get("pilot_passed") and not stop_reasons),
        "summaries": {
            m: {
                "mean_u_ctrl": (report.get("summaries") or {}).get(m, {}).get("mean_u_ctrl"),
                "true_safety_rate": (report.get("summaries") or {}).get(m, {}).get(
                    "true_safety_rate"
                ),
            }
            for m in ("dad", "myopic", "fixed", "random")
        },
    }
    _write_json(exp_dir / "stop_status.json", status)
    if stop_reasons:
        print(f"  STOP: {stop_reasons}", flush=True)
    else:
        print("  T=3 PASS — T=4 may proceed under the same frozen rule.", flush=True)
    return status


if __name__ == "__main__":
    run_ieee5_t3()
