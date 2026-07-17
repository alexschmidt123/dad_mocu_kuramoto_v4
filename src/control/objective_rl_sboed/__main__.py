"""CLI entry for the objective RL-sBOED study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.control.objective_rl_sboed import OUT
from src.control.objective_rl_sboed.context import load_study_context
from src.control.objective_rl_sboed.diagnostics import (
    collect_action_regret,
    collect_adaptivity,
    ensure_sensitivity_audit_alias,
    reward_diagnostics,
)
from src.control.objective_rl_sboed.evaluate import (
    evaluate_fixed,
    evaluate_myopic,
    evaluate_random,
    paired_bootstrap_ci,
    summarize_rows,
    write_csv,
)
from src.control.objective_rl_sboed.ppo_train import train_policy
from src.control.objective_rl_sboed.sensitivity import (
    run_sensitivity_audit,
    write_sensitivity_report,
)

SEEDS = (101, 202, 303, 404, 505)
MIN_FULL_UPDATES = 20


def _is_full_result(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    updates = int((payload.get("config") or {}).get("updates", 0))
    return updates >= MIN_FULL_UPDATES


def cmd_sensitivity(args: argparse.Namespace) -> None:
    summaries = []
    systems = ("ieee5", "ieee9") if args.system == "both" else (args.system,)
    for system in systems:
        print(f"[sensitivity] {system}")
        summaries.append(
            run_sensitivity_audit(
                system,
                max_sequences=args.max_sequences,
                n_systems=args.n_systems,
                n_noise=args.n_noise,
                smoke=args.smoke,
            )
        )
    report = write_sensitivity_report(summaries)
    ensure_sensitivity_audit_alias()
    print(json.dumps({"summaries": summaries, "report": str(report)}, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    ctx = load_study_context(args.system)
    from src.control.objective_rl_sboed.layout import training_output_dir

    out = training_output_dir(args.system, args.method, args.init, args.seed)
    if getattr(args, "skip_if_done", False) and _is_full_result(out / "result.json"):
        print(json.dumps({"skipped": True, "path": str(out / "result.json")}, indent=2))
        return
    result = train_policy(
        ctx,
        method=args.method,
        init_mode=args.init,
        seed=args.seed,
        output_dir=out,
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


def cmd_migrate_layout(args: argparse.Namespace) -> None:
    from src.control.objective_rl_sboed.layout import migrate_legacy_objective_rl_sboed_tree

    report = migrate_legacy_objective_rl_sboed_tree()
    ensure_sensitivity_audit_alias()
    print(json.dumps(report, indent=2))


def cmd_baselines(args: argparse.Namespace) -> None:
    ctx = load_study_context(args.system)
    from src.control.objective_rl_sboed.layout import prepare_system_experiment
    from src.experiment_layout import eval_method_dir, ensure_standard_layout

    exp_dir = prepare_system_experiment(
        args.system, terminal_rule_hash=ctx.terminal_rule_hash, entry_point="run.sh"
    )
    ensure_standard_layout(exp_dir)
    systems = ctx.confirmation_systems
    n = args.n_rollouts
    fixed_rows = evaluate_fixed(ctx, systems, n)
    random_rows = evaluate_random(ctx, systems, n, seed=args.seed)
    myopic_rows = evaluate_myopic(ctx, systems, n)
    write_csv(eval_method_dir(exp_dir, "fixed") / "rollouts.csv", fixed_rows)
    write_csv(eval_method_dir(exp_dir, "random") / "rollouts.csv", random_rows)
    write_csv(eval_method_dir(exp_dir, "myopic") / "rollouts.csv", myopic_rows)
    summaries = [
        summarize_rows(fixed_rows, "Fixed"),
        summarize_rows(random_rows, "Random"),
        summarize_rows(myopic_rows, "Myopic"),
    ]
    write_csv(exp_dir / "eval" / "baseline_summary.csv", summaries)
    write_csv(exp_dir / "eval" / "summary.csv", summaries)
    print(json.dumps(summaries, indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    from src.control.objective_rl_sboed.layout import method_key, prepare_system_experiment

    out = prepare_system_experiment(args.system, entry_point="run.sh")
    rows: list[dict[str, Any]] = []
    for method, init in (
        ("DAD", "random"),
        ("DAD", "fixed"),
        ("RL-sBOED", "random"),
        ("RL-sBOED", "fixed"),
    ):
        folder = out / "train" / method_key(method, init)
        for seed in SEEDS:
            result_path = folder / f"seed_{seed}" / "result.json"
            if not result_path.exists():
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "system": args.system,
                    "method": method,
                    "init_mode": init,
                    "seed": seed,
                    "validation_mean_u_ctrl": payload.get("best_validation_mean_u_ctrl"),
                    "confirmation_mean_u_ctrl": payload.get("confirmation_mean_u_ctrl"),
                    "confirmation_std_u_ctrl": payload.get("confirmation_std_u_ctrl"),
                    "confirmation_median_u_ctrl": payload.get("confirmation_median_u_ctrl"),
                    "dominant_fraction": payload.get("confirmation_dominant_fraction"),
                    "unique_sequences": payload.get("confirmation_unique_sequences"),
                    "elapsed_seconds": payload.get("elapsed_seconds"),
                    "is_full_run": int(
                        int((payload.get("config") or {}).get("updates", 0))
                        >= MIN_FULL_UPDATES
                    ),
                }
            )
    write_csv(out / "eval" / "comparison.csv", rows)
    write_csv(out / "comparison.csv", rows)

    selected: dict[str, dict[str, Any]] = {}
    for method in ("DAD", "RL-sBOED"):
        candidates = [
            r for r in rows if r["method"] == method and int(r.get("is_full_run", 0)) == 1
        ]
        if not candidates:
            candidates = [r for r in rows if r["method"] == method]
        if not candidates:
            continue
        by_init: dict[str, list[float]] = {}
        for r in candidates:
            by_init.setdefault(str(r["init_mode"]), []).append(
                float(r["validation_mean_u_ctrl"])
            )
        best_init = min(by_init, key=lambda k: sum(by_init[k]) / len(by_init[k]))
        selected[method] = {
            "method": method,
            "selected_init": best_init,
            "mean_validation_u_ctrl": sum(by_init[best_init]) / len(by_init[best_init]),
        }
    (out / "summary" / "selected_initialization.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )

    pair_rows = []
    for left, right in (
        ("RL-sBOED", "DAD"),
        ("RL-sBOED", "Myopic"),
        ("RL-sBOED", "Fixed"),
        ("RL-sBOED", "Random"),
        ("DAD", "Myopic"),
        ("DAD", "Fixed"),
    ):
        if left in selected and right in selected:
            l_init = selected[left]["selected_init"]
            r_init = selected[right]["selected_init"]
            l_vals = []
            r_vals = []
            for seed in SEEDS:
                lp = out / "train" / method_key(left, l_init) / f"seed_{seed}" / "result.json"
                rp = out / "train" / method_key(right, r_init) / f"seed_{seed}" / "result.json"
                if lp.exists() and rp.exists() and _is_full_result(lp) and _is_full_result(rp):
                    l_vals.append(
                        float(json.loads(lp.read_text())["confirmation_mean_u_ctrl"])
                    )
                    r_vals.append(
                        float(json.loads(rp.read_text())["confirmation_mean_u_ctrl"])
                    )
            if l_vals:
                import numpy as np

                diff = np.asarray(l_vals) - np.asarray(r_vals)
                stats = paired_bootstrap_ci(diff)
                pair_rows.append(
                    {
                        "comparison": f"{left} - {right}",
                        **stats,
                        "n_seeds": len(l_vals),
                    }
                )
    baseline_path = out / "eval" / "baseline_summary.csv"
    if baseline_path.exists() and selected:
        import csv

        import numpy as np

        baselines = {
            r["method"]: float(r["mean_u_ctrl"])
            for r in csv.DictReader(baseline_path.open())
        }
        for method, meta in selected.items():
            init = meta["selected_init"]
            vals = []
            for seed in SEEDS:
                p = out / "train" / method_key(method, init) / f"seed_{seed}" / "result.json"
                if p.exists() and _is_full_result(p):
                    vals.append(float(json.loads(p.read_text())["confirmation_mean_u_ctrl"]))
            if not vals:
                continue
            for base in ("Myopic", "Fixed", "Random"):
                if base not in baselines:
                    continue
                diff = np.asarray(vals) - baselines[base]
                stats = paired_bootstrap_ci(diff)
                pair_rows.append(
                    {
                        "comparison": f"{method} - {base}",
                        **stats,
                        "n_seeds": len(vals),
                        "note": "baseline treated as fixed mean across seeds",
                    }
                )
    write_csv(out / "eval" / "paired_bootstrap.csv", pair_rows)
    write_csv(out / "paired_bootstrap.csv", pair_rows)

    adapt_rows = collect_adaptivity(args.system, SEEDS)
    write_csv(out / "eval" / "adaptivity.csv", adapt_rows)
    write_csv(out / "adaptivity.csv", adapt_rows)

    regret_rows = collect_action_regret(args.system, SEEDS, selected, max_histories=200)
    write_csv(out / "eval" / "action_regret.csv", regret_rows)
    write_csv(out / "action_regret.csv", regret_rows)

    reward_rows = reward_diagnostics(args.system, SEEDS)
    write_csv(out / "summary" / "reward_diagnostics.csv", reward_rows)
    write_csv(OUT / "summary" / "reward_diagnostics.csv", reward_rows)

    print(
        json.dumps(
            {
                "comparison_rows": len(rows),
                "paired_rows": pair_rows,
                "adaptivity_rows": len(adapt_rows),
                "regret_rows": len(regret_rows),
                "selected": selected,
            },
            indent=2,
        )
    )


def _case_label(pair_rows: list[dict[str, Any]], adapt_rows: list[dict[str, Any]]) -> str:
    rl_dad = next((r for r in pair_rows if r.get("comparison") == "RL-sBOED - DAD"), None)
    if rl_dad is None:
        return "incomplete"
    mean = float(rl_dad["mean_diff"])
    lo = float(rl_dad["ci95_low"])
    hi = float(rl_dad["ci95_high"])
    # Lower u_ctrl is better => negative mean_diff favors RL-sBOED.
    if hi < 0:
        return "CASE 1: RL-sBOED better than DAD (CI excludes 0)"
    if lo > 0:
        return "CASE 4: RL-sBOED worse than DAD"
    # Similar terminal; check regret if available.
    regrets = [float(r.get("observation_dependent_rate", 0)) for r in adapt_rows if r.get("method") == "RL-sBOED"]
    dad_reg = [float(r.get("observation_dependent_rate", 0)) for r in adapt_rows if r.get("method") == "DAD"]
    if regrets and dad_reg and (sum(regrets) / len(regrets)) > (sum(dad_reg) / len(dad_reg) + 0.05):
        return "CASE 3: more adaptive branching, similar terminal u_ctrl"
    return "CASE 2: RL-sBOED ≈ DAD (low adaptive value)"


def cmd_report(args: argparse.Namespace) -> None:
    import csv

    summary_dir = OUT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    ensure_sensitivity_audit_alias()

    method_rows: list[dict[str, Any]] = []
    for system in ("ieee5", "ieee9"):
        comp = OUT / f"{system}_T3" / "eval" / "comparison.csv"
        if not comp.exists():
            continue
        with comp.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                method_rows.append(row)
    write_csv(summary_dir / "final_method_comparison.csv", method_rows)

    path = summary_dir / "final_report.md"
    # Preserve the authored Part-XIX report once present; only refresh CSV aggregates.
    if path.is_file() and "Answers to Part XIX" in path.read_text(encoding="utf-8"):
        print(f"kept authored {path} (CSV aggregates refreshed)")
        return

    lines = [
        "# Objective RL-sBOED study — final report",
        "",
        "Scientific methods: **DAD**, **RL-sBOED**, **Myopic**, **Fixed**, **Random**.",
        "",
        "Controlled contrast: terminal-only DAD reward vs dense stepwise RL-sBOED reward",
        "(`r_t = u_{t-1} - u_t`, γ=1), same R2-style policy backbone and PPO trainer.",
        "Primary metric for every method: terminal snapped `u_ctrl(h_T)` (lower is better).",
        "",
        "## Sensitivity audit (Part II)",
        "",
    ]
    sens_report = OUT / "diagnostics" / "sensitivity_audit" / "sensitivity_report.md"
    if sens_report.exists():
        lines.append(sens_report.read_text(encoding="utf-8"))
        lines.append("")
    else:
        lines.append("Sensitivity report missing.")
        lines.append("")

    lines += [
        "## Experiment-design modifications (Part III)",
        "",
        "**None.** Frozen terminal rules and design space retained for this study version.",
        "",
        "## Per-system results",
        "",
    ]

    for system in ("ieee5", "ieee9"):
        exp = OUT / f"{system}_T3"
        lines.append(f"### {system} T=3")
        lines.append("")
        sel_path = exp / "summary" / "selected_initialization.json"
        if sel_path.exists():
            sel = json.loads(sel_path.read_text(encoding="utf-8"))
            lines.append(f"- Selected inits (validation): `{json.dumps(sel)}`")
        comp = exp / "eval" / "comparison.csv"
        lines.append(f"- comparison present: {comp.exists()}")
        pair = exp / "eval" / "paired_bootstrap.csv"
        lines.append(f"- paired bootstrap present: {pair.exists()}")
        base = exp / "eval" / "baseline_summary.csv"
        if base.exists():
            with base.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    lines.append(
                        f"- baseline {row['method']}: mean u_ctrl={float(row['mean_u_ctrl']):.6f}"
                    )
        if pair.exists():
            with pair.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    lines.append(
                        f"- {row['comparison']}: mean_diff={float(row['mean_diff']):+.6f} "
                        f"CI95=[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}]"
                    )
        adapt = exp / "eval" / "adaptivity.csv"
        adapt_rows: list[dict[str, Any]] = []
        if adapt.exists():
            with adapt.open(encoding="utf-8") as handle:
                adapt_rows = list(csv.DictReader(handle))
            for method in ("DAD", "RL-sBOED"):
                subset = [r for r in adapt_rows if r["method"] == method]
                if not subset:
                    continue
                uniq = [int(r["n_unique_sequences"]) for r in subset]
                dom = [float(r["dominant_sequence_fraction"]) for r in subset]
                lines.append(
                    f"- {method} adaptivity: mean unique seq={sum(uniq)/len(uniq):.2f}, "
                    f"mean dominant fraction={sum(dom)/len(dom):.3f}"
                )
        regret = exp / "eval" / "action_regret.csv"
        if regret.exists():
            with regret.open(encoding="utf-8") as handle:
                rrows = list(csv.DictReader(handle))
            for method in ("DAD", "RL-sBOED"):
                subset = [r for r in rrows if r["method"] == method]
                if not subset:
                    continue
                regrets = [float(r["regret"]) for r in subset]
                agree = [int(r["agree"]) for r in subset]
                lines.append(
                    f"- {method} action regret: mean={sum(regrets)/len(regrets):+.6f}, "
                    f"xi2* agreement={sum(agree)/len(agree):.3f}"
                )
        pair_rows: list[dict[str, Any]] = []
        if pair.exists():
            with pair.open(encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
        lines.append(f"- Interpretation: {_case_label(pair_rows, adapt_rows)}")
        lines.append("")

    lines += [
        "## Final questions (Part XIX)",
        "",
        "1. **Is terminal u_ctrl sufficiently sensitive?** Partially — dozens of unique values exist, "
        "but std is small (~0.01–0.02) and best−second gaps are tiny (IEEE5 ≈ 0).",
        "2. **Coarseness cause?** Mix of **C snap_up quantization** and **E similar posterior changes**; "
        "`u_raw` is somewhat more dispersed than snapped `u_ctrl`.",
        "3. **Experiment-design modification?** No (this version).",
        "4–6. **Fixed init / DAD branching:** see per-system selected_init and adaptivity tables.",
        "7–10. **RL-sBOED vs DAD / regret / branching / terminal gains:** see paired bootstrap and "
        "action_regret; interpret via CASE labels above.",
        "11–13. **vs Myopic/Fixed and significance:** see paired bootstrap CIs.",
        "14. **Consistent with prior Case B?** Yes — low intrinsic adaptive value on IEEE5/IEEE9 T=3 "
        "remains the dominant story unless CIs show otherwise.",
        "",
        "## Method definitions (unchanged)",
        "",
        "- **DAD**: terminal-reward full-horizon adaptive policy",
        "- **RL-sBOED**: stepwise-reward full-horizon adaptive policy (same terminal objective)",
        "- **Myopic**: one-step objective-based adaptive optimization",
        "- **Fixed**: optimized nonadaptive complete plan",
        "- **Random**: random valid probes",
        "",
        "Final performance criterion: smallest safe terminal `u_ctrl`.",
        "",
    ]
    path = summary_dir / "final_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="objective_rl_sboed")
    sub = parser.add_subparsers(dest="command", required=True)

    sens = sub.add_parser("sensitivity")
    sens.add_argument("--system", choices=("ieee5", "ieee9", "both"), default="both")
    sens.add_argument("--max-sequences", type=int, default=400)
    sens.add_argument("--n-systems", type=int, default=40)
    sens.add_argument("--n-noise", type=int, default=2)
    sens.add_argument("--smoke", action="store_true")
    sens.set_defaults(func=cmd_sensitivity)

    train = sub.add_parser("train")
    train.add_argument("--system", choices=("ieee5", "ieee9"), required=True)
    train.add_argument("--method", choices=("DAD", "RL-sBOED"), required=True)
    train.add_argument("--init", choices=("random", "fixed"), default="random")
    train.add_argument("--seed", type=int, default=101)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--skip-if-done", action="store_true")
    train.set_defaults(func=cmd_train)

    base = sub.add_parser("baselines")
    base.add_argument("--system", choices=("ieee5", "ieee9"), required=True)
    base.add_argument("--n-rollouts", type=int, default=128)
    base.add_argument("--seed", type=int, default=7)
    base.set_defaults(func=cmd_baselines)

    comp = sub.add_parser("compare")
    comp.add_argument("--system", choices=("ieee5", "ieee9"), required=True)
    comp.set_defaults(func=cmd_compare)

    rep = sub.add_parser("report")
    rep.set_defaults(func=cmd_report)

    mig = sub.add_parser("migrate-layout")
    mig.set_defaults(func=cmd_migrate_layout)

    full = sub.add_parser("run-system")
    full.add_argument("--system", choices=("ieee5", "ieee9"), required=True)
    full.add_argument("--smoke", action="store_true")
    full.add_argument("--skip-sensitivity", action="store_true")
    full.add_argument("--skip-if-done", action="store_true", default=True)
    full.set_defaults(func=cmd_run_system)
    return parser


def cmd_run_system(args: argparse.Namespace) -> None:
    if not args.skip_sensitivity:
        run_sensitivity_audit(args.system, smoke=args.smoke)
        ensure_sensitivity_audit_alias()
    seeds = (101,) if args.smoke else SEEDS
    for method in ("DAD", "RL-sBOED"):
        for init in ("random", "fixed"):
            for seed in seeds:
                print(f"[train] {args.system} {method} init={init} seed={seed}")
                cmd_train(
                    argparse.Namespace(
                        system=args.system,
                        method=method,
                        init=init,
                        seed=seed,
                        smoke=args.smoke,
                        skip_if_done=bool(getattr(args, "skip_if_done", True))
                        and not args.smoke,
                    )
                )
    cmd_baselines(
        argparse.Namespace(
            system=args.system, n_rollouts=32 if args.smoke else 128, seed=7
        )
    )
    cmd_compare(argparse.Namespace(system=args.system))
    cmd_report(argparse.Namespace())


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
