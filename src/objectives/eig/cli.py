"""CLI for the EIG-based (table / ΔH) pipeline. Prefer ``python -m src.experiment``."""

from __future__ import annotations

import argparse

from src.config import (
    ALL_METHODS,
    DEFAULT_STEP_NUMBER,
    load_config_for_run,
    repo_root,
    resolve_config_path,
    resolve_exp_dir,
)
from src.objectives.eig.pipeline import (
    eval_experiment,
    generate_tables,
    print_experiment_banner,
    run_evaluation,
    run_experiment,
    train_dad_policy,
)
from src.layout import load_experiment_run


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

    run = sub.add_parser("run", help="Full pipeline: generate-data → train → evaluate")
    run.add_argument("--config", required=True)
    run.add_argument("--exp-dir", default=None)
    run.add_argument("--method", default=None, choices=ALL_METHODS)
    _add_T(run)

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
        from src.banks.generate_control import generate_control_bank
        from src.banks.diagnose_control import control_bank_nondegenerate
        from src.banks.tables import resolve_data_path

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
        from src.banks.diagnose_control import diagnose_control_objective

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
