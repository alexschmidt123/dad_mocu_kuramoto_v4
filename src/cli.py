"""CLI for scripts/: generate-data | train | evaluate | summarize | run."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import (
    ALL_METHODS,
    DEFAULT_STEP_NUMBER,
    load_config_for_run,
    repo_root,
    resolve_config_path,
    resolve_exp_dir,
)
from src.experiment import (
    eval_experiment,
    generate_tables,
    print_experiment_banner,
    run_evaluation,
    run_experiment,
    train_dad_policy,
)
from src.plot_summary import plot_all_detailed, plot_all_summaries, plot_sweep_six_metrics
from src.stepwise_eig.runner import run_all_systems, run_system_stepwise_eig
from src.run_context import load_experiment_run


def _add_T(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-T",
        "--step-number",
        type=int,
        default=None,
        metavar="T",
        help=f"Probe horizon (default {DEFAULT_STEP_NUMBER} if omitted)",
    )


def _dad_methods_from_cfg(methods: list[str]) -> list[str]:
    return [m for m in methods if m == "dad"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Swing-equation DAD experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-data", help="Build or reuse data/<run_slug>/ (T-independent bank)")
    gen.add_argument("--config", required=True)
    gen.add_argument("--exp-dir", default=None)
    gen.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="both",
        help="Which split to generate (batched runs)",
    )
    gen.add_argument(
        "--theta-start",
        type=int,
        default=0,
        help="First θ index (inclusive) for the selected split(s)",
    )
    gen.add_argument(
        "--theta-end",
        type=int,
        default=None,
        help="Last θ index (exclusive); default = all in split",
    )
    _add_T(gen)

    gcb = sub.add_parser(
        "generate-control-bank",
        help="Build PyCUDA U-bank (u_req) for existing probe banks; validate safety invariants",
    )
    gcb.add_argument("--config", required=True)
    gcb.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="both",
    )

    obs = sub.add_parser(
        "check-objective-observability",
        help="Gate: verify probe histories change posterior terminal u_ctrl (no method training)",
    )
    obs.add_argument("--exp-dir", required=True)
    obs.add_argument("--num-rollouts", type=int, default=None)
    obs.add_argument("--seed", type=int, default=None)

    csc = sub.add_parser(
        "calibrate-control-safety",
        help="Calibrate posterior terminal rule (alpha, margin) on train-only splits",
    )
    csc.add_argument("--exp-dir", required=True)
    csc.add_argument("--num-rollouts", type=int, default=None)
    csc.add_argument("--seed", type=int, default=None)
    csc.add_argument(
        "--skip-diagnosis",
        action="store_true",
        help="Skip diagnosis of existing observability rollouts",
    )

    pilot = sub.add_parser(
        "run-pilot",
        help="IEEE5 T=2 four-method pilot (frozen terminal rule; no full sweep)",
    )
    pilot.add_argument("--exp-dir", required=True)
    pilot.add_argument(
        "--debug-one-seed",
        action="store_true",
        help="Train only the first DAD seed with fewer epochs",
    )
    pilot.add_argument("--num-rollouts", type=int, default=None, help="Evaluation rollouts")

    smh = sub.add_parser(
        "select-myopic-n-hypothetical",
        help="Validation-only Myopic n_hypothetical convergence; freeze production count",
    )
    smh.add_argument("--exp-dir", required=True)
    smh.add_argument(
        "--freeze-config",
        default=None,
        help="Optional source config path to write selected n_hypothetical into",
    )
    smh.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory (default: <exp>/diagnostics/myopic_convergence)",
    )

    dmf = sub.add_parser(
        "diagnose-myopic-fixed",
        help="Diagnose Myopic vs Fixed (paired CI, MC error, ties, complementarity)",
    )
    dmf.add_argument("--exp-dir", required=True)
    dmf.add_argument("--evaluation-rollouts", type=int, default=2000)
    dmf.add_argument("--seed", type=int, default=3579)

    t4 = sub.add_parser(
        "run-ieee5-t4",
        help="IEEE5 T=4 controlled four-method experiment (frozen margin 0.55; no IEEE9)",
    )
    t4.add_argument("--exp-dir", default=None, help="Default: experiments/ieee5_T4")
    t4.add_argument("--frozen-rule-path", default=None)
    t4.add_argument("--evaluation-rollouts", type=int, default=None)
    t4.add_argument("--skip-observability", action="store_true")

    t4x = sub.add_parser(
        "run-ieee5-t4-fixed-exact",
        help="IEEE5 T=4 exhaustive Fixed fairness correction (no DAD retrain; no IEEE9)",
    )
    t4x.add_argument("--exp-dir", default=None, help="Default: experiments/ieee5_T4")

    avd = sub.add_parser(
        "run-ieee5-adaptive-value-diagnosis",
        help="IEEE5 bank-based adaptive-value diagnosis (no test; no IEEE9)",
    )
    avd.add_argument("--exp-dir", default=None, help="Default: experiments/ieee5_T4")
    avd.add_argument("--out-dir", default=None)
    avd.add_argument("--k-outer", type=int, default=192)
    avd.add_argument("--n-hyp-inner", type=int, default=96)

    t3 = sub.add_parser(
        "run-ieee5-t3",
        help="IEEE5 T=3 controlled four-method experiment (frozen margin 0.55; no T=4)",
    )
    t3.add_argument(
        "--exp-dir",
        default=None,
        help="Default: experiments/ieee5_T3",
    )
    t3.add_argument(
        "--frozen-rule-path",
        default=None,
        help="Path to selected_policy_robust_rule.json (margin 0.55)",
    )
    t3.add_argument("--evaluation-rollouts", type=int, default=None)
    t3.add_argument(
        "--skip-observability",
        action="store_true",
        help="Skip observability gate (not recommended)",
    )

    prc = sub.add_parser(
        "run-policy-robust-calibration",
        help="IEEE5 T=2 policy-robust common margin calibration (no T=3/T=4)",
    )
    prc.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: experiments/ieee5_policy_robust_calibration_T2)",
    )
    prc.add_argument(
        "--skip-collect",
        action="store_true",
        help="Reuse existing policy_rollout_details.csv if present",
    )
    prc.add_argument(
        "--skip-retrain-pilot",
        action="store_true",
        help="Skip DAD retrain and four-method T=2 rerun",
    )
    prc.add_argument(
        "--skip-seal",
        action="store_true",
        help="Skip sealing a new final test bank",
    )

    hsweep = sub.add_parser(
        "run-ieee5-horizon-sweep",
        help="IEEE5 T=1..4 sweep with frozen Myopic n_h and terminal rule (no IEEE9/14)",
    )
    hsweep.add_argument(
        "--source-rule-exp",
        default="experiments/07132026_220727_ieee5_T2",
        help="Experiment providing the certified calibrated terminal rule",
    )
    hsweep.add_argument("--from-T", type=int, default=1, dest="from_t")
    hsweep.add_argument("--to-T", type=int, default=4, dest="to_t")
    hsweep.add_argument("--evaluation-rollouts", type=int, default=None)
    hsweep.add_argument(
        "--skip-observability",
        action="store_true",
        help="Skip observability gate (not recommended)",
    )

    diag = sub.add_parser(
        "diagnose-control-objective",
        help="Diagnose U-bank degeneracy / binding constraints (no method training)",
    )
    diag.add_argument("--config", required=True)
    diag.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="both",
    )

    train = sub.add_parser("train", help="Train DAD policy (metadata from linked data)")
    train.add_argument("--exp-dir", required=True)
    train.add_argument("--method", default="dad", choices=["dad"])
    train.add_argument(
        "--reuse-policy",
        action="store_true",
        help="Skip training if policy exists (resume same experiment dir only)",
    )

    ev = sub.add_parser("evaluate", help="Evaluate methods (metadata from linked data)")
    ev.add_argument("--exp-dir", required=True)
    ev.add_argument("--method", default=None, choices=ALL_METHODS)

    summ = sub.add_parser(
        "summarize",
        help="Print comparison table and refresh eval/summary.csv",
    )
    summ.add_argument("--exp-dir", required=True)

    plot = sub.add_parser(
        "plot-summaries",
        help="Plot cumulative ΔH curves, MSE bars, and train/test time bars from summary.csv",
    )
    plot.add_argument(
        "--run-prefix",
        "--config-prefix",
        dest="run_prefix",
        default="ieee14",
        help="Experiment folder prefix (default: ieee14). Legacy *_config_* dirs still match.",
    )
    plot.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: experiments/_plots)",
    )

    plot_det = sub.add_parser(
        "plot-detailed",
        help="Detailed plots: per-T metrics, training curves, all-system overview",
    )
    plot_det.add_argument(
        "--run-prefix",
        "--config-prefix",
        dest="run_prefix",
        default=None,
        help="Single system prefix (default: all ieee5/ieee9/ieee14)",
    )
    plot_det.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: experiments/_plots)",
    )

    plot_sweep = sub.add_parser(
        "plot-sweep",
        help="Three sweep figures: ΔH (top) and sPCE (bottom) per system with terminal tables",
    )
    plot_sweep.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: experiments/_plots/sweep_six)",
    )

    run = sub.add_parser("run", help="Full pipeline: generate-data → train → evaluate")
    run.add_argument("--config", required=True)
    run.add_argument("--exp-dir", default=None)
    run.add_argument("--method", default=None, choices=ALL_METHODS)
    _add_T(run)

    eig = sub.add_parser(
        "stepwise-eig",
        help="Evaluation-only stepwise ΔH/EIG report (fresh noise on banked y_sim)",
    )
    eig.add_argument(
        "--system",
        choices=("ieee5", "ieee9", "ieee14", "all"),
        default="all",
        help="IEEE benchmark system (default: all)",
    )
    eig.add_argument("--exp-dir", default=None, help="Primary experiment folder (default: latest T)")
    eig.add_argument("--T-max", type=int, default=None, dest="t_max", help="Primary horizon if auto-picking exp")
    eig.add_argument("--out-dir", default=None, help="Output directory override")
    eig.add_argument("--noise-seed", type=int, default=None)
    eig.add_argument("--support-seed", type=int, default=None)
    eig.add_argument("--rollout-seed", type=int, default=None)

    args = parser.parse_args(argv)
    root = repo_root()

    if args.command == "generate-data":
        cfg = load_config_for_run(args.config, root, step_number=args.step_number)
        splits = ("train", "test") if args.split == "both" else (args.split,)
        theta_ranges = {
            s: (int(args.theta_start), args.theta_end) for s in splits
        }
        exp_dir, data_path, train_systems, test_systems = generate_tables(
            cfg,
            root,
            resolve_exp_dir(root, args.exp_dir),
            splits=splits,
            theta_ranges=theta_ranges,
        )
        print_experiment_banner(cfg, exp_dir, data_path, train_systems, test_systems, cfg.methods)
        print(f"\nData → {data_path}")
        print(f"DATA_DIR={data_path}")
        print(f"EXP_DIR={exp_dir}")
        return

    if args.command == "generate-control-bank":
        from src.control.generate import generate_control_bank
        from src.control.diagnose import control_bank_nondegenerate
        from src.data import resolve_data_path

        splits = ("train", "test") if args.split == "both" else (args.split,)
        reports = generate_control_bank(args.config, splits=splits)
        ok = True
        for split, rep in reports.get("splits", {}).items():
            ub = float(rep.get("u_bank_particle_safety_rate", 0.0))
            um = float(rep.get("maximum_control_safety_rate", 0.0))
            oc = float(rep.get("oracle_control_safety_rate", 0.0))
            split_ok = ub >= 1.0 - 1e-12 and um >= 1.0 - 1e-12 and oc >= 1.0 - 1e-12
            ok = ok and split_ok
            print(
                f"[{split}] U-bank={ub:.3f}  u_max={um:.3f}  oracle={oc:.3f}  "
                f"{'PASS' if split_ok else 'FAIL'}"
            )
        if not ok:
            raise SystemExit(
                "Control-bank invariants FAILED. Do not compare methods until "
                "oracle/u_max/U-bank safety rates are all 1.0."
            )
        print("Control-bank invariants PASS.")
        cfg = load_config_for_run(args.config, root)
        data_path = resolve_data_path(root, cfg)
        nd_ok, nd_detail = control_bank_nondegenerate(data_path)
        if not nd_ok:
            raise SystemExit(
                "Control-bank is DEGENERATE (std(U)=0 or |unique|<=1). "
                "Run diagnose-control-objective and retune the control scenario "
                "before method training.\n"
                f"Detail: {nd_detail}"
            )
        print("Control-bank nondegeneracy PASS.")
        return

    if args.command == "diagnose-control-objective":
        from src.control.diagnose import diagnose_control_objective

        splits = ("train", "test") if args.split == "both" else (args.split,)
        report = diagnose_control_objective(args.config, splits=splits)
        bad = [
            s
            for s, rep in report.get("splits", {}).items()
            if not rep.get("verdict", {}).get("nondegenerate", False)
        ]
        if bad:
            raise SystemExit(
                f"Degenerate U-bank on splits {bad}. Retune control scenario before training."
            )
        return

    if args.command == "check-objective-observability":
        from src.control.observability import check_objective_observability

        summary = check_objective_observability(
            args.exp_dir,
            project_root=root,
            num_rollouts=args.num_rollouts,
            seed=args.seed,
        )
        if not summary.get("gate", {}).get("passed", False):
            raise SystemExit(
                "Objective-observability gate FAILED. "
                "Do not train DAD or evaluate methods until probe histories "
                "change the posterior terminal control."
            )
        print("Objective-observability gate PASS.")
        return

    if args.command == "calibrate-control-safety":
        from src.control.safety_calibration import calibrate_control_safety

        result = calibrate_control_safety(
            args.exp_dir,
            project_root=root,
            num_rollouts=args.num_rollouts,
            seed=args.seed,
            skip_diagnosis=bool(args.skip_diagnosis),
        )
        if not result.get("passed", False):
            raise SystemExit(
                "Control-safety calibration FAILED. "
                "No terminal rule achieved the required calibration/validation safety. "
                "Do not train methods until a safe rule is found."
            )
        print("Control-safety calibration PASS.")
        return

    if args.command == "run-pilot":
        from src.control.pilot import run_pilot

        report = run_pilot(
            args.exp_dir,
            project_root=root,
            debug_one_seed=bool(args.debug_one_seed),
            n_eval_rollouts=args.num_rollouts,
        )
        if not report.get("pilot_passed", False):
            raise SystemExit("Pilot FAILED (see eval/summary.json).")
        print("Pilot PASS.")
        return

    if args.command == "select-myopic-n-hypothetical":
        from src.control.myopic_convergence import (
            freeze_into_source_config,
            run_myopic_convergence,
        )
        from pathlib import Path

        report = run_myopic_convergence(
            args.exp_dir,
            project_root=root,
            out_dir=Path(args.out_dir) if args.out_dir else None,
        )
        if args.freeze_config:
            freeze_into_source_config(
                Path(args.freeze_config),
                int(report["selected_n_hypothetical"]),
                report,
            )
            print(f"Froze into source config → {args.freeze_config}")
        print(
            f"SELECTED_N_HYPOTHETICAL={report['selected_n_hypothetical']}"
        )
        return

    if args.command == "diagnose-myopic-fixed":
        from src.control.diagnose_myopic_fixed import run_diagnose_myopic_fixed

        report = run_diagnose_myopic_fixed(
            args.exp_dir,
            project_root=root,
            evaluation_rollouts=int(args.evaluation_rollouts),
            seed=int(args.seed),
        )
        print("Myopic-vs-Fixed diagnosis complete.")
        if report["verdicts"]["5_implementation_inconsistency"]["answer"]:
            raise SystemExit("Implementation inconsistency detected.")
        return

    if args.command == "run-ieee5-t4":
        from pathlib import Path

        from src.control.ieee5_t4 import run_ieee5_t4

        status = run_ieee5_t4(
            project_root=root,
            exp_dir=Path(args.exp_dir) if args.exp_dir else None,
            frozen_rule_path=Path(args.frozen_rule_path) if args.frozen_rule_path else None,
            evaluation_rollouts=args.evaluation_rollouts,
            skip_observability=bool(args.skip_observability),
        )
        print(f"IEEE5 T=4 → {status['exp_dir']}")
        print(f"TERMINAL_RULE_HASH={status['terminal_rule_hash']}")
        print(f"CAN_FREEZE_IEEE5={status['can_freeze_ieee5_before_ieee9']}")
        if status.get("stop_reasons"):
            raise SystemExit(f"T=4 stopped: {status['stop_reasons']}")
        if not status.get("pilot_passed"):
            raise SystemExit("T=4 experiment FAILED (see eval/T4_report.md).")
        print("IEEE5 T=4 PASS.")
        return

    if args.command == "run-ieee5-t4-fixed-exact":
        from pathlib import Path

        from src.control.ieee5_t4_fixed_exact import run_ieee5_t4_fixed_exact

        report = run_ieee5_t4_fixed_exact(
            project_root=root,
            exp_dir=Path(args.exp_dir) if args.exp_dir else None,
        )
        print(f"EXACT_FIXED_SUBSET={report['exact_fixed_subset']}")
        print(f"DAD_SUBSET_RANK={report['DAD_subset_rank']}")
        print(f"EXACT_FIXED_TEST_MEAN={report['exact_fixed_test_mean_u_ctrl']}")
        print(f"CAN_FREEZE_IEEE5={report['can_freeze_ieee5']}")
        return

    if args.command == "run-ieee5-adaptive-value-diagnosis":
        from pathlib import Path

        from src.control.adaptive_value_diagnosis import run_adaptive_value_diagnosis

        summary = run_adaptive_value_diagnosis(
            project_root=root,
            exp_dir=Path(args.exp_dir) if args.exp_dir else None,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            K_outer=int(args.k_outer),
            n_hyp_inner=int(args.n_hyp_inner),
        )
        print(f"OVERALL_CASE={summary['overall_case']}")
        print(f"PROCEED_DAD_IMPROVEMENT={summary['proceed_to_dad_improvement']}")
        print(f"MOVE_TO_IEEE9_RECOMMENDED={summary['move_to_ieee9_recommended']}")
        return

    if args.command == "run-ieee5-t3":
        from pathlib import Path

        from src.control.ieee5_t3 import run_ieee5_t3

        status = run_ieee5_t3(
            project_root=root,
            exp_dir=Path(args.exp_dir) if args.exp_dir else None,
            frozen_rule_path=Path(args.frozen_rule_path) if args.frozen_rule_path else None,
            evaluation_rollouts=args.evaluation_rollouts,
            skip_observability=bool(args.skip_observability),
        )
        print(f"IEEE5 T=3 → {status['exp_dir']}")
        print(f"TERMINAL_RULE_HASH={status['terminal_rule_hash']}")
        print(f"CAN_PROCEED_TO_T4={status['can_proceed_to_T4']}")
        if status.get("stop_reasons"):
            raise SystemExit(f"T=3 stopped: {status['stop_reasons']}")
        if not status.get("pilot_passed"):
            raise SystemExit("T=3 experiment FAILED (see eval/T3_report.md).")
        print("IEEE5 T=3 PASS.")
        return

    if args.command == "run-policy-robust-calibration":
        from pathlib import Path

        from src.control.policy_robust_calibration import run_policy_robust_calibration

        summary = run_policy_robust_calibration(
            project_root=root,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            skip_collect=bool(args.skip_collect),
            skip_retrain_pilot=bool(args.skip_retrain_pilot),
            skip_seal=bool(args.skip_seal),
        )
        print(f"Policy-robust calibration complete → {summary['out_dir']}")
        print(f"SELECTED_MARGIN={summary['selected_margin']}")
        return

    if args.command == "run-ieee5-horizon-sweep":
        from src.control.ieee5_horizon_sweep import run_ieee5_horizon_sweep

        T_values = list(range(int(args.from_t), int(args.to_t) + 1))
        result = run_ieee5_horizon_sweep(
            project_root=root,
            source_rule_exp=args.source_rule_exp,
            T_values=T_values,
            evaluation_rollouts=args.evaluation_rollouts,
            skip_observability=bool(args.skip_observability),
        )
        if result.get("stop_reason"):
            raise SystemExit(f"Sweep stopped: {result['stop_reason']}")
        print("IEEE5 horizon sweep PASS.")
        return

    if args.command == "train":
        exp_dir = resolve_exp_dir(root, args.exp_dir)
        assert exp_dir is not None
        run = load_experiment_run(exp_dir, root)
        print(f"  data={run.data_path}  T={run.meta.step_number} (from tables)")
        methods = [args.method] if args.method else _dad_methods_from_cfg(list(run.cfg.methods))
        if not methods:
            raise ValueError(
                "No DAD method found in config methods. "
                "Add dad, or pass --method dad."
            )
        for method in methods:
            policy_path = train_dad_policy(
                run,
                method_name=method,
                reuse_policy=args.reuse_policy or None,
            )
            print(f"Policy ({method}) → {policy_path}")
        return

    if args.command == "summarize":
        exp_dir = resolve_exp_dir(root, args.exp_dir)
        assert exp_dir is not None
        eval_experiment(exp_dir)
        return

    if args.command == "plot-summaries":
        out_dir = Path(args.out_dir) if args.out_dir else None
        print("Writing summary plots:")
        plot_all_summaries(
            root,
            out_dir=out_dir,
            run_prefix=args.run_prefix,
        )
        return

    if args.command == "plot-detailed":
        out_dir = Path(args.out_dir) if args.out_dir else None
        prefixes = (args.run_prefix,) if args.run_prefix else None
        print("Writing detailed plots:")
        plot_all_detailed(
            root,
            out_dir=out_dir,
            run_prefixes=prefixes,
        )
        return

    if args.command == "plot-sweep":
        out_dir = Path(args.out_dir) if args.out_dir else None
        print("Writing sweep figures (one per system: ΔH top, sPCE bottom):")
        plot_sweep_six_metrics(root, out_dir=out_dir)
        return

    if args.command == "evaluate":
        exp_dir = resolve_exp_dir(root, args.exp_dir)
        assert exp_dir is not None
        run = load_experiment_run(exp_dir, root)
        methods = [args.method] if args.method else list(run.cfg.methods)
        print_experiment_banner(
            run.cfg, run.exp_dir, run.data_path,
            run.train_systems, run.test_systems, methods,
        )
        run_evaluation(run, methods=methods)
        return

    if args.command == "stepwise-eig":
        kwargs = {
            "noise_seed": args.noise_seed,
            "support_seed": args.support_seed,
            "rollout_seed": args.rollout_seed,
        }
        if args.system == "all":
            outputs = run_all_systems(root, t_max=args.t_max, **kwargs)
            for prefix, path in outputs.items():
                print(f"{prefix} → {path}")
            return
        out = run_system_stepwise_eig(
            args.system,
            root,
            exp_dir=resolve_exp_dir(root, args.exp_dir) if args.exp_dir else None,
            t_max=args.t_max,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            **kwargs,
        )
        print(f"Stepwise EIG report → {out}")
        return

    if args.command == "run":
        methods = [args.method] if args.method else None
        exp_dir = resolve_exp_dir(root, args.exp_dir)
        run_experiment(
            resolve_config_path(args.config, root),
            root,
            methods=methods,
            exp_dir=exp_dir,
            step_number=args.step_number,
        )
        return


if __name__ == "__main__":
    main()
