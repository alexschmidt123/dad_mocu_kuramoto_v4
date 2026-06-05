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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Swing-equation DAD experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-data", help="Build or reuse data/<config>_T<T>/")
    gen.add_argument("--config", required=True)
    gen.add_argument("--exp-dir", default=None)
    _add_T(gen)

    train = sub.add_parser("train", help="Train DAD policy (metadata from linked data)")
    train.add_argument("--exp-dir", required=True)

    ev = sub.add_parser("evaluate", help="Evaluate methods (metadata from linked data)")
    ev.add_argument("--exp-dir", required=True)
    ev.add_argument("--method", default=None, choices=ALL_METHODS)

    summ = sub.add_parser("summarize", help="Print results.json")
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
        exp_dir, data_path, train_systems, test_systems = generate_tables(
            cfg, root, resolve_exp_dir(root, args.exp_dir),
        )
        print_experiment_banner(cfg, exp_dir, data_path, train_systems, test_systems, cfg.methods)
        print(f"\nData → {data_path}")
        print(f"DATA_DIR={data_path}")
        print(f"EXP_DIR={exp_dir}")
        return

    if args.command == "train":
        exp_dir = resolve_exp_dir(root, args.exp_dir)
        assert exp_dir is not None
        run = load_experiment_run(exp_dir, root)
        print(f"  data={run.data_path}  T={run.meta.step_number} (from tables)")
        policy_path = train_dad_policy(run)
        print(f"Policy → {policy_path}")
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
