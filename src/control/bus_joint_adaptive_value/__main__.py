"""CLI for bus + joint adaptive-value diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any

from src.control.bus_joint_adaptive_value import OUT
from src.control.bus_joint_adaptive_value.diagnostic import analyze_system
from src.control.bus_joint_adaptive_value.report import write_final_reports


def cmd_run(args: argparse.Namespace) -> None:
    systems = ("ieee5", "ieee9") if args.system == "both" else (args.system,)
    summaries = []
    for system in systems:
        print(f"[bus_joint] {system}")
        summaries.append(
            analyze_system(
                system,
                max_histories=args.max_histories,
                n_hyp=args.n_hyp,
                smoke=args.smoke,
                n_decomp_rollouts=args.n_decomp_rollouts,
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
    print(f"wrote {write_final_reports(summaries)}")


def _write_comparison(summaries: list[dict[str, Any]]) -> None:
    comp = OUT / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    bus_spec, bus_reg, joint, cvs = [], [], [], []
    for s in summaries:
        bus_spec.append(
            {
                "system": s["system"],
                "dominant_bus": s["dominant_bus"],
                "dominant_bus_fraction": s["dominant_bus_fraction"],
                "unique_buses": s["number_of_unique_optimal_buses"],
                "entropy": s["entropy_optimal_bus"],
                "frac_pairs_different": s["fraction_history_pairs_different_bus"],
                "case": s["case"],
            }
        )
        bus_reg.append(
            {
                "system": s["system"],
                **{f"cont_{k}": v for k, v in s["wrong_bus_regret_cont"].items()},
                **{f"snap_{k}": v for k, v in s["wrong_bus_regret_snap"].items()},
                "prior_wrong_amp_mean": s["prior_wrong_amplitude_regret_mean"],
                "mean_dominant_bus_regret": s["mean_dominant_bus_regret"],
                "mean_fixed_bus_regret": s["mean_fixed_bus_regret"],
            }
        )
        for row in (s.get("decomposition") or {}).get("rows") or []:
            joint.append(row)
        cvs.append(
            {
                "system": s["system"],
                "mean_gap_cont": s["mean_best_second_bus_gap"],
                "mean_gap_snap": s["mean_best_second_bus_gap_snapped"],
                "wrong_bus_mean_cont": s["wrong_bus_regret_cont"]["mean"],
                "wrong_bus_mean_snap": s["wrong_bus_regret_snap"]["mean"],
                "case": s["case"],
            }
        )
    for name, rows in (
        ("bus_specialization.csv", bus_spec),
        ("bus_regret.csv", bus_reg),
        ("joint_decomposition.csv", joint),
        ("continuous_vs_snapped.csv", cvs),
    ):
        if not rows:
            continue
        with (comp / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bus_joint_adaptive_value")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--system", choices=("ieee5", "ieee9", "both"), default="both")
    run.add_argument("--max-histories", type=int, default=200)
    run.add_argument("--n-hyp", type=int, default=64)
    run.add_argument("--n-decomp-rollouts", type=int, default=128)
    run.add_argument("--smoke", action="store_true")
    run.set_defaults(func=cmd_run)
    rep = sub.add_parser("report")
    rep.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
