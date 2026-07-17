"""IEEE5 T=4 controlled experiment with frozen margin-0.55 rule and adaptivity diagnostics."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.control.observability import (
    ObservabilityGateConfig,
    check_objective_observability,
    evaluate_gate,
)
from src.control.pilot import (
    evaluate_method_paired,
    load_pilot_splits,
    paired_diff_stats,
    run_pilot,
)
from src.control.safety_calibration import install_frozen_terminal_rule
from src.control.terminal_rule import load_frozen_terminal_rule
from src.contrastive.spce import log_prior_uniform_discrete
from src.control.banks import extract_U_bank
from src.rollout import FixedSelector
from src.run_context import load_experiment_run
from src.swing_equation_ode.design import build_catalog
from src.table_scoring import TableThetaSupport
from src.control.u_req import ControlSpec

FROZEN_MARGIN = 0.55
FROZEN_ALPHA = 0.05
EXPECTED_HASH = "c2e2af33cb68a5ea"
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


def _entropy(counts: Counter | dict) -> float:
    vals = np.asarray(list(counts.values()), dtype=np.float64)
    total = float(vals.sum())
    if total <= 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def prepare_ieee5_t4(
    *,
    root: Path,
    exp_dir: Path,
    config_name: str = "ieee5_config",
    frozen_rule_path: Path | None = None,
) -> Path:
    from src.config import load_config_for_run
    from src.control.generate import generate_control_bank
    from src.experiment import generate_tables

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config_for_run(config_name, root, step_number=4)
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
    if frozen.terminal_rule_hash != EXPECTED_HASH:
        raise RuntimeError(
            f"Frozen rule hash {frozen.terminal_rule_hash} != expected {EXPECTED_HASH}"
        )

    rc = exp_dir / "run_config.yaml"
    data = yaml.safe_load(rc.read_text()) or {}
    data["step_number"] = 4
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


def load_dad_rollout_rows(eval_root: Path) -> list[dict[str, Any]]:
    path = eval_root / "dad" / "rollouts.csv"
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seq = [int(x) for x in str(r.get("sequence", "")).split() if x != ""]
            y = [float(x) for x in str(r.get("y_obs", "")).split() if x != ""]
            rows.append(
                {
                    "theta_test_id": int(r["theta_test_id"]),
                    "rollout_id": int(r["rollout_id"]),
                    "sequence": seq,
                    "y_obs": y,
                    "u_ctrl": float(r["u_ctrl"]),
                    "u_req_true": float(r["u_req_true"]),
                    "safe_total": str(r["safe_total"]).lower() in ("1", "true", "yes"),
                }
            )
    return rows


def compute_dad_adaptivity(rows: list[dict[str, Any]], n_bins: int = 5) -> dict[str, Any]:
    detail_rows = []
    for r in rows:
        seq = r["sequence"]
        y = r["y_obs"]
        detail = {
            "rollout_id": r["rollout_id"],
            "theta_test_id": r["theta_test_id"],
            "complete_action_sequence": " ".join(map(str, seq)),
        }
        for i in range(4):
            detail[f"selected_action_{i+1}"] = seq[i] if i < len(seq) else ""
            detail[f"observation_{i+1}"] = y[i] if i < len(y) else ""
        detail_rows.append(detail)

    seq_counts = Counter(tuple(r["sequence"]) for r in rows if r.get("sequence"))
    n = len(rows)
    dom_seq, dom_n = (seq_counts.most_common(1)[0] if seq_counts else ((), 0))
    unique_at = {}
    for step in (2, 3, 4):
        acts = {r["sequence"][step - 1] for r in rows if len(r["sequence"]) >= step}
        unique_at[f"unique_actions_at_step_{step}"] = int(len(acts))

    # Conditioned next-action tables
    cond_rows = []
    for step in range(1, 4):  # predict action at step+1 from obs at step
        y_vals = [
            float(r["y_obs"][step - 1])
            for r in rows
            if len(r["y_obs"]) >= step and len(r["sequence"]) > step
        ]
        if not y_vals:
            continue
        edges = np.quantile(y_vals, np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        buckets: dict[tuple, Counter] = defaultdict(Counter)
        for r in rows:
            if len(r["sequence"]) <= step or len(r["y_obs"]) < step:
                continue
            hist = tuple(r["sequence"][:step])
            b = int(np.searchsorted(edges, float(r["y_obs"][step - 1]), side="right") - 1)
            b = min(max(b, 0), n_bins - 1)
            buckets[(hist, b)][int(r["sequence"][step])] += 1
        for (hist, b), ctr in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
            total = sum(ctr.values())
            most = ctr.most_common(1)[0][0]
            cond_rows.append(
                {
                    "step": step + 1,
                    "previous_action_history": " ".join(map(str, hist)),
                    "observation_bin": b,
                    "most_common_next_action": most,
                    "next_action_entropy": _entropy(ctr),
                    "next_action_distribution": json.dumps(
                        {str(k): v / total for k, v in ctr.items()}
                    ),
                    "n": total,
                }
            )

    # Interpretation
    frac = float(dom_n / n) if n else float("nan")
    n_unique = int(len(seq_counts))
    # Does next action vary with observation bin given history?
    adaptive_signal = False
    by_hist: dict[tuple, list] = defaultdict(list)
    for row in cond_rows:
        by_hist[tuple(row["previous_action_history"].split())].append(row)
    for hist, group in by_hist.items():
        actions = {g["most_common_next_action"] for g in group}
        if len(actions) > 1 and any(g["next_action_entropy"] > 1e-9 for g in group):
            # Different bins → different modal action
            modal_by_bin = {g["observation_bin"]: g["most_common_next_action"] for g in group}
            if len(set(modal_by_bin.values())) > 1:
                adaptive_signal = True
                break

    if frac >= 0.90 and n_unique <= 2:
        interpretation = "effectively_nonadaptive"
    elif n_unique > 1 and not adaptive_signal:
        interpretation = "stochastic_not_clearly_adaptive"
    elif adaptive_signal:
        interpretation = "adaptive"
    else:
        interpretation = "effectively_nonadaptive"

    summary = {
        "number_of_unique_sequences": n_unique,
        "dominant_sequence": list(dom_seq),
        "dominant_sequence_fraction": frac,
        "sequence_entropy": _entropy(seq_counts),
        **unique_at,
        "interpretation": interpretation,
        "conditioned_tables": cond_rows,
        "detail_rows": detail_rows,
    }
    return summary


def evaluate_dominant_sequence(
    *,
    exp_dir: Path,
    root: Path,
    dominant_sequence: list[int],
    dad_rows: list[dict[str, Any]],
    n_rollouts: int,
) -> dict[str, Any]:
    """Diagnostic: evaluate frozen dominant sequence on same paired rollouts."""
    run = load_experiment_run(exp_dir, root)
    splits = load_pilot_splits(exp_dir, run)
    frozen = load_frozen_terminal_rule(exp_dir, expected_margin=FROZEN_MARGIN)
    control_spec = frozen.to_control_spec(ControlSpec.from_cfg(run.cfg))
    table_support = TableThetaSupport(
        systems=splits["support_systems"],
        log_p0=log_prior_uniform_discrete(len(splits["support_systems"])),
    )
    U_support = extract_U_bank(splits["support_systems"])
    catalog = build_catalog(run.cfg)
    n_actions = len(catalog)
    from src.swing_equation_ode.design import build_simulator
    from src.control.cuda_control import CudaControlEngine

    sim = build_simulator(run.cfg)
    sim.T_obs_sec = control_spec.T_obs_sec
    sim.ode_dt = control_spec.ode_dt
    sim.fs_hz = control_spec.fs_hz
    engine = CudaControlEngine(sim, control_spec)
    pilot_cfg = dict(run.cfg.raw.get("pilot") or {})
    global_seed = int(pilot_cfg.get("global_seed", 1234))
    method_seed = int(list(pilot_cfg.get("method_seeds", [101]))[0])

    rows, summary = evaluate_method_paired(
        method="dad_dominant_sequence_diagnostic",
        selector_factory=lambda: FixedSelector(sequence=list(dominant_sequence)),
        test_systems=splits["test_systems"],
        table_support=table_support,
        U_support=U_support,
        frozen=frozen,
        control_spec=control_spec,
        control_engine=engine,
        horizon=4,
        n_actions=n_actions,
        sigma_y=float(run.cfg.sigma_y),
        n_rollouts=n_rollouts,
        global_seed=global_seed,
        method_seed=method_seed,
    )
    n = min(len(dad_rows), len(rows))
    ua = np.asarray([dad_rows[i]["u_ctrl"] for i in range(n)], dtype=np.float64)
    ub = np.asarray([rows[i]["u_ctrl"] for i in range(n)], dtype=np.float64)
    paired = paired_diff_stats(ua, ub, n_boot=10000, seed=global_seed)
    out = exp_dir / "eval" / "dad_dominant_sequence_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "summary.json", {**summary, "paired_vs_dad": paired})
    cmp_rows = [
        {
            "rollout_id": i,
            "u_dad": float(ua[i]),
            "u_dominant_sequence": float(ub[i]),
            "diff_dad_minus_dominant": float(ua[i] - ub[i]),
        }
        for i in range(n)
    ]
    _write_csv(exp_dir / "eval" / "dad_dominant_sequence_comparison.csv", cmp_rows)
    return {
        "dominant_sequence": dominant_sequence,
        "summary": summary,
        "paired_vs_dad": paired,
        "label": "dad_dominant_sequence_diagnostic",
    }


def write_t4_outputs(
    exp_dir: Path,
    pilot_report: dict[str, Any],
    adapt: dict[str, Any],
    dominant_cmp: dict[str, Any],
) -> None:
    eval_root = exp_dir / "eval"
    frozen = load_frozen_terminal_rule(exp_dir, expected_margin=FROZEN_MARGIN)

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

    pair_rows = [
        {"contrast": name, **stats}
        for name, stats in (pilot_report.get("paired_differences") or {}).items()
    ]
    _write_csv(eval_root / "paired_comparisons.csv", pair_rows)

    _write_csv(eval_root / "dad_adaptivity.csv", adapt.get("detail_rows") or [])
    _write_json(
        eval_root / "dad" / "adaptation.json",
        {
            "number_of_unique_sequences": adapt["number_of_unique_sequences"],
            "dominant_sequence": adapt["dominant_sequence"],
            "dominant_sequence_fraction": adapt["dominant_sequence_fraction"],
            "sequence_entropy": adapt["sequence_entropy"],
            "unique_actions_at_step_2": adapt.get("unique_actions_at_step_2"),
            "unique_actions_at_step_3": adapt.get("unique_actions_at_step_3"),
            "unique_actions_at_step_4": adapt.get("unique_actions_at_step_4"),
            "interpretation": adapt["interpretation"],
            "conditioned_next_action": adapt.get("conditioned_tables"),
            "terminal_rule_hash": frozen.terminal_rule_hash,
        },
    )

    myopic = (pilot_report.get("summaries") or {}).get("myopic") or {}
    if myopic.get("tie_diagnostics"):
        _write_json(eval_root / "myopic" / "tie_diagnostics.json", myopic["tie_diagnostics"])

    fixed = pilot_report.get("fixed_subset") or {}
    search_mode = fixed.get("search_mode", "")
    exhaustive = search_mode == "exhaustive"
    fixed_label = (
        "exhaustive"
        if exhaustive
        else "approximately optimized Fixed baseline"
    )
    fixed["exhaustive_or_approximate"] = fixed_label
    _write_json(eval_root / "fixed" / "subset_meta.json", fixed)

    obs = {}
    obs_path = (
        exp_dir / "diagnostics" / "objective_observability" / "observability_summary.json"
    )
    if obs_path.is_file():
        obs = json.loads(obs_path.read_text())
    stepwise_path = (
        exp_dir / "diagnostics" / "objective_observability" / "stepwise_observability.csv"
    )
    stepwise = []
    if stepwise_path.is_file():
        with stepwise_path.open(encoding="utf-8") as f:
            stepwise = list(csv.DictReader(f))

    metas = {}
    if (eval_root / "method_metadata.json").is_file():
        metas = json.loads((eval_root / "method_metadata.json").read_text())

    paired_dom = dominant_cmp.get("paired_vs_dad") or {}
    lines = [
        "# IEEE5 T=4 controlled experiment report",
        "",
        f"**Passed: {pilot_report.get('pilot_passed')}**",
        "",
        "## Frozen terminal rule",
        "",
        f"- α = `{frozen.alpha}`",
        f"- additive_margin = `{frozen.margin}`",
        f"- terminal_rule_hash = `{frozen.terminal_rule_hash}` (expected `{EXPECTED_HASH}`)",
        "",
        "### Per-method hashes",
        "",
    ]
    for name, m in (metas.get("methods") or {}).items():
        lines.append(f"- `{name}`: `{m.get('terminal_rule_hash')}`")
    lines.extend(["", "## Objective observability", ""])
    lines.append(f"- true_safety_rate: `{obs.get('true_safety_rate')}`")
    lines.append(f"- unique_final_u_ctrl_count: `{obs.get('unique_final_u_ctrl_count')}`")
    lines.append(f"- final_u_ctrl_std: `{obs.get('final_u_ctrl_std')}`")
    lines.append(f"- fraction_changed_from_prior: `{obs.get('fraction_changed_from_prior')}`")
    lines.append(f"- real Spearman: `{obs.get('real_spearman')}` vs shuffled `{obs.get('shuffled_spearman')}`")
    lines.extend(["", "### Stepwise", ""])
    for row in stepwise:
        lines.append(
            f"- t={row.get('step')}: unique={row.get('n_unique')} "
            f"mean={row.get('u_ctrl_mean')} std={row.get('u_ctrl_std')} "
            f"changed_from_prev={row.get('fraction_changed_from_previous_step')} "
            f"ESS={row.get('posterior_ess_mean')}"
        )
    lines.extend(["", "## Per-method results", "",
        "| method | mean u_ctrl | safety | excess | runtime |",
        "|---|---:|---:|---:|---:|"])
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
            f"- seed {seed}: mean_u={ss.get('mean_u_ctrl'):.4f} safety={ss.get('true_safety_rate'):.3f}"
        )
    lines.append(
        f"- primary: seed `{(pilot_report.get('summaries') or {}).get('dad', {}).get('seed')}`"
    )
    lines.extend(
        [
            "",
            "## DAD adaptivity",
            "",
            f"- unique sequences: `{adapt['number_of_unique_sequences']}`",
            f"- dominant sequence: `{adapt['dominant_sequence']}`",
            f"- dominant fraction: `{adapt['dominant_sequence_fraction']}`",
            f"- sequence entropy: `{adapt['sequence_entropy']}`",
            f"- unique actions step 2/3/4: "
            f"`{adapt.get('unique_actions_at_step_2')}` / "
            f"`{adapt.get('unique_actions_at_step_3')}` / "
            f"`{adapt.get('unique_actions_at_step_4')}`",
            f"- interpretation: **{adapt['interpretation']}**",
            "",
            "## DAD vs dominant-sequence diagnostic",
            "",
            f"- label: `dad_dominant_sequence_diagnostic` (not a primary method)",
            f"- mean paired diff (DAD − dominant): `{paired_dom.get('mean_paired_diff')}`",
            f"- CI95: `[{paired_dom.get('ci95_low')}, {paired_dom.get('ci95_high')}]`",
            f"- fraction tied: `{paired_dom.get('fraction_tied')}`",
            "",
            "## Myopic ties",
            "",
            f"`{json.dumps(myopic.get('tie_diagnostics') or {}, indent=2)}`",
            "",
            "## Fixed",
            "",
            f"- exhaustive_or_approximate: `{fixed_label}`",
            f"- search_mode: `{search_mode}`",
            f"- selected: `{fixed.get('selected_action_ids')}`",
            f"- subsets_evaluated: `{fixed.get('number_of_subsets_evaluated')}`",
            f"- validation_mean_u_ctrl: `{fixed.get('estimated_mean_u_ctrl')}`",
            f"- runtime: `{fixed.get('search_runtime')}`",
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
            f"tied={v.get('fraction_tied'):.3f}"
        )
    (eval_root / "T4_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (eval_root / "pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_horizon_summary(root: Path) -> dict[str, Any]:
    """Combine T=2 (policy-robust rerun), T=3, T=4 into ieee5_horizon_summary/."""
    out = root / "experiments" / "ieee5_horizon_summary"
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        2: root / "experiments" / "ieee5_policy_robust_calibration_T2" / "rerun_T2" / "eval",
        3: root / "experiments" / "ieee5_T3" / "eval",
        4: root / "experiments" / "ieee5_T4" / "eval",
    }
    # Fallback T2 summary CSV if eval missing structure
    t2_csv = root / "experiments" / "ieee5_policy_robust_calibration_T2" / "rerun_T2_summary.csv"
    summary_rows = []
    paired_rows = []
    seq_entropy = {}
    myopic_tie = {}

    for T, eval_root in sources.items():
        sj = eval_root / "summary.json"
        if sj.is_file():
            rep = json.loads(sj.read_text())
            for m in ("dad", "myopic", "fixed", "random"):
                s = (rep.get("summaries") or {}).get(m) or {}
                paired = rep.get("paired_differences") or {}
                vs_rand = paired.get(f"{m}_minus_random") or {}
                vs_fix = paired.get(f"{m}_minus_fixed") or paired.get("dad_minus_fixed") if m == "dad" else {}
                if m == "fixed":
                    vs_fix = {"mean_paired_diff": 0.0}
                elif m == "myopic":
                    vs_fix = {
                        "mean_paired_diff": -(
                            (paired.get("myopic_minus_fixed") or {}).get("mean_paired_diff", float("nan"))
                        )
                    }
                elif m == "random":
                    vs_fix = {
                        "mean_paired_diff": -(
                            (paired.get("fixed_minus_random") or {}).get("mean_paired_diff", float("nan"))
                        )
                    }
                vs_my = paired.get(f"{m}_minus_myopic") or {}
                if m == "myopic":
                    vs_my = {"mean_paired_diff": 0.0}
                elif m == "dad":
                    vs_my = paired.get("dad_minus_myopic") or {}
                elif m == "fixed":
                    vs_my = {
                        "mean_paired_diff": -(
                            (paired.get("myopic_minus_fixed") or {}).get("mean_paired_diff", float("nan"))
                        )
                    }
                elif m == "random":
                    vs_my = {
                        "mean_paired_diff": -(
                            (paired.get("myopic_minus_random") or {}).get("mean_paired_diff", float("nan"))
                        )
                    }
                summary_rows.append(
                    {
                        "T": T,
                        "method": m,
                        "mean_u_ctrl": s.get("mean_u_ctrl"),
                        "true_safety_rate": s.get("true_safety_rate"),
                        "mean_excess_control": s.get("mean_excess_control"),
                        "runtime": s.get("mean_runtime_per_rollout"),
                        "paired_difference_vs_random": (vs_rand or {}).get("mean_paired_diff"),
                        "paired_difference_vs_fixed": (vs_fix or {}).get("mean_paired_diff"),
                        "paired_difference_vs_myopic": (vs_my or {}).get("mean_paired_diff"),
                    }
                )
            for name, stats in (rep.get("paired_differences") or {}).items():
                paired_rows.append({"T": T, "contrast": name, **stats})
            adapt_p = eval_root / "dad" / "adaptation.json"
            if adapt_p.is_file():
                ad = json.loads(adapt_p.read_text())
                if isinstance(ad, dict):
                    seq_entropy[T] = ad.get("sequence_entropy", ad.get("dominant_sequence_fraction"))
                    if "sequence_entropy" in ad:
                        seq_entropy[T] = ad["sequence_entropy"]
                    elif ad.get("dominant_sequence_fraction") is not None:
                        # entropy proxy from fraction for T3
                        p = float(ad["dominant_sequence_fraction"])
                        seq_entropy[T] = float(
                            0.0 if p >= 1.0 - 1e-12 else -(p * math.log(p) + (1 - p) * math.log(1 - p + 1e-15))
                        )
            tie_p = eval_root / "myopic" / "tie_diagnostics.json"
            if tie_p.is_file():
                myopic_tie[T] = json.loads(tie_p.read_text()).get("exact_tie_rate")
            elif (rep.get("summaries") or {}).get("myopic", {}).get("tie_diagnostics"):
                myopic_tie[T] = rep["summaries"]["myopic"]["tie_diagnostics"].get(
                    "exact_tie_rate"
                )
        elif T == 2 and t2_csv.is_file():
            with t2_csv.open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    summary_rows.append(
                        {
                            "T": 2,
                            "method": r["method"],
                            "mean_u_ctrl": float(r["mean_u_ctrl"]),
                            "true_safety_rate": float(r["true_safety_rate"]),
                            "mean_excess_control": float(r["mean_excess_control"]),
                            "runtime": "",
                            "paired_difference_vs_random": "",
                            "paired_difference_vs_fixed": "",
                            "paired_difference_vs_myopic": "",
                        }
                    )
            # T3 known nonadaptive
            seq_entropy.setdefault(3, 0.0)

    # Fill T2/T3 sequence entropy from known results if missing
    seq_entropy.setdefault(2, float("nan"))  # unknown without adaptivity dump
    if 3 not in seq_entropy:
        seq_entropy[3] = 0.0  # T3 fraction 1.0 → entropy 0

    _write_csv(out / "ieee5_T2_T3_T4_summary.csv", summary_rows)
    _write_csv(out / "ieee5_T2_T3_T4_paired.csv", paired_rows)

    # Plots
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots = out / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        methods = ["dad", "myopic", "fixed", "random"]
        Ts = [2, 3, 4]

        def series(method, key):
            outv = []
            for T in Ts:
                row = next(
                    (
                        r
                        for r in summary_rows
                        if int(r["T"]) == T and r["method"] == method
                    ),
                    None,
                )
                outv.append(float(row[key]) if row and row.get(key) not in ("", None) else np.nan)
            return outv

        for key, fname, ylabel in [
            ("mean_u_ctrl", "mean_u_ctrl_vs_T.png", "mean u_ctrl"),
            ("mean_excess_control", "excess_control_vs_T.png", "mean excess"),
            ("runtime", "runtime_vs_T.png", "runtime / rollout (s)"),
        ]:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for m in methods:
                ax.plot(Ts, series(m, key), marker="o", label=m)
            ax.set_xlabel("T")
            ax.set_ylabel(ylabel)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots / fname, dpi=120)
            plt.close(fig)

        for contrast, fname in [
            ("dad_minus_myopic", "dad_minus_myopic_vs_T.png"),
            ("dad_minus_fixed", "dad_minus_fixed_vs_T.png"),
            ("myopic_minus_fixed", "myopic_minus_fixed_vs_T.png"),
        ]:
            xs, ys, lo, hi = [], [], [], []
            for T in Ts:
                row = next(
                    (r for r in paired_rows if int(r["T"]) == T and r["contrast"] == contrast),
                    None,
                )
                if not row:
                    continue
                xs.append(T)
                ys.append(float(row["mean_paired_diff"]))
                lo.append(float(row["ci95_low"]))
                hi.append(float(row["ci95_high"]))
            if not xs:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.plot(xs, ys, "o-", color="#2c5f7c")
            ax.fill_between(xs, lo, hi, alpha=0.25, color="#2c5f7c")
            ax.axhline(0.0, color="k", lw=0.8)
            ax.set_xlabel("T")
            ax.set_ylabel(contrast)
            fig.tight_layout()
            fig.savefig(plots / fname, dpi=120)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3.8))
        xs = sorted(seq_entropy)
        ax.plot(xs, [seq_entropy[t] for t in xs], "o-", color="#2c5f7c")
        ax.set_xlabel("T")
        ax.set_ylabel("DAD sequence entropy")
        fig.tight_layout()
        fig.savefig(plots / "dad_sequence_entropy_vs_T.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3.8))
        xs = sorted(myopic_tie)
        if xs:
            ax.plot(xs, [myopic_tie[t] for t in xs], "o-", color="#c44e52")
        ax.set_xlabel("T")
        ax.set_ylabel("Myopic exact-tie rate")
        fig.tight_layout()
        fig.savefig(plots / "myopic_tie_rate_vs_T.png", dpi=120)
        plt.close(fig)
    except Exception:
        pass

    # Answers
    def mean_at(T, method):
        row = next((r for r in summary_rows if int(r["T"]) == T and r["method"] == method), None)
        return float(row["mean_u_ctrl"]) if row else float("nan")

    answers = {
        "1_T4_lower_than_T3": {
            m: mean_at(4, m) < mean_at(3, m) - 1e-12 for m in methods
        },
        "2_dad_tied_fixed_T4": None,  # filled from paired
        "3_dad_adaptive_T4": None,
        "4_myopic_worse_than_fixed_T4": mean_at(4, "myopic") > mean_at(4, "fixed"),
        "5_random_weakest_T4": mean_at(4, "random")
        == max(mean_at(4, m) for m in methods),
        "6_design_benefit_vs_T": {},
        "7_T4_cost_justified": None,
    }
    # paired dad-fixed at T4
    df = next(
        (r for r in paired_rows if int(r["T"]) == 4 and r["contrast"] == "dad_minus_fixed"),
        None,
    )
    if df:
        answers["2_dad_tied_fixed_T4"] = float(df["ci95_low"]) <= 0 <= float(df["ci95_high"])
    adapt4 = sources[4] / "dad" / "adaptation.json"
    if adapt4.is_file():
        answers["3_dad_adaptive_T4"] = (
            json.loads(adapt4.read_text()).get("interpretation") == "adaptive"
        )
    # benefit vs random: more negative dad_minus_random is better
    for T in Ts:
        row = next(
            (r for r in paired_rows if int(r["T"]) == T and r["contrast"] == "dad_minus_random"),
            None,
        )
        answers["6_design_benefit_vs_T"][str(T)] = (
            None if not row else float(row["mean_paired_diff"])
        )
    # cost: compare mean runtime dad T4 vs reduction from T3
    red = mean_at(3, "dad") - mean_at(4, "dad")
    rt4 = mean_at(4, "dad")  # placeholder — runtime in summary
    rt_row = next((r for r in summary_rows if int(r["T"]) == 4 and r["method"] == "dad"), None)
    answers["7_T4_cost_justified"] = {
        "dad_mean_u_reduction_T3_to_T4": red,
        "dad_runtime_T4": None if not rt_row or rt_row.get("runtime") in ("", None) else float(rt_row["runtime"]),
        "note": "Justified if control reduction is material relative to extra probe/runtime cost.",
    }

    report_lines = [
        "# IEEE5 horizon summary (T=2, T=3, T=4)",
        "",
        "Frozen terminal rule: α=0.05, margin=0.55.",
        "",
        "## Mean u_ctrl by T",
        "",
        "| T | dad | myopic | fixed | random |",
        "|---|---:|---:|---:|---:|",
    ]
    for T in Ts:
        vals = {m: mean_at(T, m) for m in methods}
        report_lines.append(
            f"| {T} | {vals['dad']:.4f} | {vals['myopic']:.4f} | "
            f"{vals['fixed']:.4f} | {vals['random']:.4f} |"
        )
    report_lines.extend(["", "## Answers", ""])
    report_lines.append(f"1. T=4 lower than T=3? `{answers['1_T4_lower_than_T3']}`")
    report_lines.append(f"2. DAD tied with Fixed at T=4? `{answers['2_dad_tied_fixed_T4']}`")
    report_lines.append(f"3. DAD adaptive at T=4? `{answers['3_dad_adaptive_T4']}`")
    report_lines.append(f"4. Myopic worse than Fixed at T=4? `{answers['4_myopic_worse_than_fixed_T4']}`")
    report_lines.append(f"5. Random weakest at T=4? `{answers['5_random_weakest_T4']}`")
    report_lines.append(f"6. DAD−Random by T: `{answers['6_design_benefit_vs_T']}`")
    report_lines.append(f"7. T=4 cost justification: `{answers['7_T4_cost_justified']}`")
    (out / "ieee5_horizon_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    _write_json(out / "answers.json", answers)
    return {"summary_rows": summary_rows, "answers": answers, "out_dir": str(out)}


def run_ieee5_t4(
    *,
    project_root: Path | None = None,
    exp_dir: str | Path | None = None,
    frozen_rule_path: str | Path | None = None,
    evaluation_rollouts: int | None = None,
    skip_observability: bool = False,
) -> dict[str, Any]:
    from src.config import repo_root

    root = Path(project_root or repo_root())
    exp_dir = Path(exp_dir or (root / "experiments" / "ieee5_T4"))
    assert "ieee5_T4" in str(exp_dir)

    print("=== Prepare IEEE5 T=4 (frozen margin 0.55) ===", flush=True)
    prepare_ieee5_t4(
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
        print("\n=== Objective observability (T=4) ===", flush=True)
        check_objective_observability(exp_dir, project_root=root)
        run = load_experiment_run(exp_dir, root)
        gate = ObservabilityGateConfig.from_cfg(run.cfg)
        obs_path = (
            exp_dir / "diagnostics" / "objective_observability" / "observability_summary.json"
        )
        obs = json.loads(obs_path.read_text()) if obs_path.is_file() else {}
        gate_result = evaluate_gate(obs, gate)
        if not gate_result.get("passed", False):
            _write_json(
                exp_dir / "stop_status.json",
                {"stop_reason": "objective_observability_failed", "gate": gate_result},
            )
            raise RuntimeError(
                f"T=4 observability FAILED: {gate_result.get('failed_checks')}"
            )
        print("  Observability gate PASS", flush=True)

    print("\n=== Four-method T=4 experiment ===", flush=True)
    report = run_pilot(
        exp_dir,
        project_root=root,
        debug_one_seed=False,
        n_eval_rollouts=evaluation_rollouts,
    )

    # Label Fixed approximate if needed
    fixed = report.get("fixed_subset") or {}
    if fixed.get("search_mode") != "exhaustive":
        print(
            "  Fixed search is approximately optimized (not exhaustive; "
            f"mode={fixed.get('search_mode')})",
            flush=True,
        )

    dad_rows = load_dad_rollout_rows(exp_dir / "eval")
    adapt = compute_dad_adaptivity(dad_rows)
    print(
        f"  DAD adaptivity: unique={adapt['number_of_unique_sequences']} "
        f"dom_frac={adapt['dominant_sequence_fraction']:.3f} "
        f"interp={adapt['interpretation']}",
        flush=True,
    )
    n_eval = int(report.get("eval_rollouts") or len(dad_rows))
    dominant_cmp = evaluate_dominant_sequence(
        exp_dir=exp_dir,
        root=root,
        dominant_sequence=list(adapt["dominant_sequence"]),
        dad_rows=dad_rows,
        n_rollouts=n_eval,
    )
    write_t4_outputs(exp_dir, report, adapt, dominant_cmp)

    stop_reasons = []
    for m in ("dad", "myopic", "fixed", "random"):
        sr = ((report.get("summaries") or {}).get(m) or {}).get("true_safety_rate")
        if sr is None or abs(float(sr) - 1.0) > 1e-12:
            stop_reasons.append(f"{m}_safety_{sr}")
    meta_path = exp_dir / "eval" / "method_metadata.json"
    if meta_path.is_file():
        metas = json.loads(meta_path.read_text())
        hashes = {v.get("terminal_rule_hash") for v in (metas.get("methods") or {}).values()}
        if hashes != {EXPECTED_HASH}:
            stop_reasons.append(f"terminal_rule_hashes_differ:{hashes}")
    if frozen.terminal_rule_hash != EXPECTED_HASH:
        stop_reasons.append("frozen_hash_mismatch")

    print("\n=== Horizon summary T=2/3/4 ===", flush=True)
    horizon = build_horizon_summary(root)

    status = {
        "exp_dir": str(exp_dir),
        "T": 4,
        "frozen_margin": FROZEN_MARGIN,
        "terminal_rule_hash": frozen.terminal_rule_hash,
        "pilot_passed": bool(report.get("pilot_passed")),
        "stop_reasons": stop_reasons,
        "can_freeze_ieee5_before_ieee9": bool(
            report.get("pilot_passed") and not stop_reasons
        ),
        "dad_adaptivity": {
            "unique": adapt["number_of_unique_sequences"],
            "dominant_fraction": adapt["dominant_sequence_fraction"],
            "interpretation": adapt["interpretation"],
        },
        "dominant_vs_dad": dominant_cmp.get("paired_vs_dad"),
        "summaries": {
            m: {
                "mean_u_ctrl": (report.get("summaries") or {}).get(m, {}).get("mean_u_ctrl"),
                "true_safety_rate": (report.get("summaries") or {}).get(m, {}).get(
                    "true_safety_rate"
                ),
            }
            for m in ("dad", "myopic", "fixed", "random")
        },
        "horizon_summary": horizon.get("out_dir"),
    }
    _write_json(exp_dir / "stop_status.json", status)
    if stop_reasons:
        print(f"  STOP: {stop_reasons}", flush=True)
    else:
        print("  T=4 PASS — IEEE5 ready to freeze before IEEE9.", flush=True)
    return status


if __name__ == "__main__":
    run_ieee5_t4()
