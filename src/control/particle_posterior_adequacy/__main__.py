"""CLI for particle-posterior-adequacy study."""

from __future__ import annotations

import argparse
import json

from src.control.particle_posterior_adequacy.master_bank import (
    generate_master_for_system,
    generate_masters,
)
from src.control.particle_posterior_adequacy.plots import plot_comparison, plot_system
from src.control.particle_posterior_adequacy.report import (
    write_final_report,
    write_system_report,
)
from src.control.particle_posterior_adequacy.run_study import (
    analyze_system,
    write_comparison,
)


def cmd_generate(args: argparse.Namespace) -> None:
    if args.system == "both":
        summaries = generate_masters(("ieee5", "ieee9"))
    else:
        summaries = [generate_master_for_system(args.system)]
    print(json.dumps({"summaries": summaries}, indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    systems = ("ieee5", "ieee9") if args.system == "both" else (args.system,)
    summaries = []
    for system in systems:
        print(f"[particle_posterior_adequacy] analyze {system}")
        summary = analyze_system(
            system,
            smoke=args.smoke,
            max_histories=args.max_histories,
            n_hyp=args.n_hyp,
        )
        plot_system(system)
        write_system_report(summary)
        summaries.append(summary)
    write_comparison(summaries)
    plot_comparison(tuple(s["system"] for s in summaries))
    report = write_final_report(summaries)
    print(json.dumps({"n_systems": len(summaries), "report": str(report)}, indent=2))


def cmd_report(args: argparse.Namespace) -> None:
    del args
    from pathlib import Path

    from src.control.particle_posterior_adequacy import OUT

    summaries = []
    for system in ("ieee5", "ieee9"):
        path = OUT / f"{system}_T3" / "summary" / "system_summary.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
            plot_system(system)
            write_system_report(summaries[-1])
    if summaries:
        write_comparison(summaries)
        plot_comparison(tuple(s["system"] for s in summaries))
        print(write_final_report(summaries))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="particle_posterior_adequacy")
    sub = p.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-master")
    gen.add_argument("--system", choices=("ieee5", "ieee9", "both"), default="both")
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="Convergence + adaptive-value diagnostic (no training)")
    run.add_argument("--system", choices=("ieee5", "ieee9", "both"), default="both")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--max-histories", type=int, default=None)
    run.add_argument("--n-hyp", type=int, default=None)
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report")
    rep.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
