"""CLI for continuous u_ctrl + amplitude adaptive-value study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.control.continuous_uctrl_amplitude import OUT
from src.control.continuous_uctrl_amplitude.audit import (
    freeze_continuous_terminal_rule,
    write_u_bank_audit,
)
from src.control.continuous_uctrl_amplitude.diagnostic import analyze_system
from src.control.continuous_uctrl_amplitude.report import write_final_reports


def cmd_audit(args: argparse.Namespace) -> None:
    path = write_u_bank_audit()
    rules = {}
    systems = ("ieee5", "ieee9") if args.system == "both" else (args.system,)
    for system in systems:
        rule = freeze_continuous_terminal_rule(system)
        rules[system] = rule.metadata()
        print(f"[audit] {system} continuous hash={rule.terminal_rule_hash}")
    print(json.dumps({"u_bank_audit": str(path), "rules": rules}, indent=2))


def cmd_diagnose(args: argparse.Namespace) -> None:
    systems = ("ieee5", "ieee9") if args.system == "both" else (args.system,)
    summaries = []
    for system in systems:
        print(f"[diagnose] {system}")
        summaries.append(
            analyze_system(
                system,
                max_histories=args.max_histories,
                n_hyp=args.n_hyp,
                smoke=args.smoke,
            )
        )
    _write_comparison(summaries)
    report = write_final_reports(summaries)
    print(json.dumps({"summaries": summaries, "report": str(report)}, indent=2, default=str))


def cmd_report(args: argparse.Namespace) -> None:
    summaries = []
    for system in ("ieee5", "ieee9"):
        path = OUT / f"{system}_T3" / "summary" / "system_summary.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    report = write_final_reports(summaries)
    print(f"wrote {report}")


def cmd_run(args: argparse.Namespace) -> None:
    cmd_audit(argparse.Namespace(system=args.system))
    cmd_diagnose(args)


def _write_comparison(summaries: list[dict[str, Any]]) -> None:
    import csv

    comp = OUT / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in summaries:
        rows.append(
            {
                "system": s["system"],
                "case": s["case"],
                "dominant_amplitude": s["dominant_amplitude"],
                "dominant_amplitude_fraction": s["dominant_amplitude_fraction"],
                "unique_optimal_amplitudes_continuous": s[
                    "number_of_unique_optimal_amplitudes"
                ],
                "unique_optimal_amplitudes_snapped": s[
                    "unique_optimal_amplitudes_snapped"
                ],
                "mean_best_second_gap_continuous": s["mean_best_second_gap"],
                "mean_best_second_gap_snapped": s["mean_best_second_gap_snapped"],
                "continuous_J_std": s["continuous_J_std"],
                "snapped_J_std": s["snapped_J_std"],
                "mean_wrong_amplitude_regret": s["wrong_amplitude_regret"]["mean"],
                "mean_cross_history_amplitude_regret": s["cross_history_amplitude_regret"][
                    "mean"
                ],
                "continuous_terminal_rule_hash": s["continuous_terminal_rule_hash"],
            }
        )
    path = comp / "continuous_vs_snapped.csv"
    if rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    # specialization + regret flatten
    spec_rows = []
    regret_rows = []
    for s in summaries:
        spec_rows.append(
            {
                "system": s["system"],
                "dominant_amplitude": s["dominant_amplitude"],
                "dominant_fraction": s["dominant_amplitude_fraction"],
                "unique_amplitudes": s["number_of_unique_optimal_amplitudes"],
                "entropy": s["entropy_optimal_amplitude"],
                "frac_pairs_different": s["fraction_history_pairs_different_amplitude"],
            }
        )
        regret_rows.append(
            {
                "system": s["system"],
                **{f"wrong_{k}": v for k, v in s["wrong_amplitude_regret"].items()},
                **{f"cross_{k}": v for k, v in s["cross_history_amplitude_regret"].items()},
                **{f"dominant_{k}": v for k, v in s["dominant_amplitude_regret"].items()},
                **{f"fixed_{k}": v for k, v in s["fixed_amplitude_regret"].items()},
            }
        )
    for name, data in (
        ("amplitude_specialization.csv", spec_rows),
        ("amplitude_regret.csv", regret_rows),
    ):
        if not data:
            continue
        with (comp / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="continuous_uctrl_amplitude")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("--system", choices=("ieee5", "ieee9", "both"), default="both")
    audit.set_defaults(func=cmd_audit)

    diag = sub.add_parser("diagnose")
    diag.add_argument("--system", choices=("ieee5", "ieee9", "both"), default="both")
    diag.add_argument("--max-histories", type=int, default=200)
    diag.add_argument("--n-hyp", type=int, default=64)
    diag.add_argument("--smoke", action="store_true")
    diag.set_defaults(func=cmd_diagnose)

    rep = sub.add_parser("report")
    rep.set_defaults(func=cmd_report)

    run = sub.add_parser("run")
    run.add_argument("--system", choices=("ieee5", "ieee9", "both"), default="both")
    run.add_argument("--max-histories", type=int, default=200)
    run.add_argument("--n-hyp", type=int, default=64)
    run.add_argument("--smoke", action="store_true")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
