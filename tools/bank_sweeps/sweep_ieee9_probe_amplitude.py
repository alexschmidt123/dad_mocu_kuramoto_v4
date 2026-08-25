#!/usr/bin/env python3
"""Generate and structurally audit fixed-amplitude IEEE9 probe/MOCU banks.

No BOED policy is trained or evaluated.  Each cell changes only the fixed
probe amplitude; theta seeds, buses, durations, observation model, noise, and
terminal-control physics remain fixed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AMPLITUDES = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)


def slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def run(command: list[str], log_path: Path, *, allow_failure: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode and not allow_failure:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return int(completed.returncode)


def load_audit(path: Path) -> dict:
    if not path.is_file():
        return {"audit_available": False}
    report = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "verdict",
        "myopic_beatable",
        "adaptive_room",
        "monotone_adaptive_room",
        "near_duplicate_fraction",
        "best_fixed_value",
        "best_myopic_value",
        "best_adaptive_value",
        "adaptive_minus_myopic_gap",
        "adaptive_minus_fixed_gap",
    )
    summary = {key: report.get(key) for key in keys if key in report}
    summary["audit_available"] = True
    summary["audit_json"] = str(path.relative_to(ROOT))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/ieee9.yaml")
    parser.add_argument(
        "--amplitudes",
        default=",".join(str(x) for x in DEFAULT_AMPLITUDES),
        help="Comma-separated fixed probe amplitudes in p.u.",
    )
    parser.add_argument("--n-obs", type=int, default=5)
    parser.add_argument("--noise-sigma", type=float, default=0.01)
    parser.add_argument("--support-size", type=int, default=96)
    parser.add_argument("--n-outer", type=int, default=24)
    parser.add_argument("--n-inner", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    amplitudes = tuple(float(x) for x in args.amplitudes.split(",") if x.strip())
    if not amplitudes or any(x <= 0 for x in amplitudes):
        raise SystemExit("All amplitudes must be positive")

    base_path = (ROOT / args.base_config).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    durations = list(base["swing_equation"]["probe_durations"])
    buses = list(base["swing_equation"]["probe_buses"])
    if len(durations) != 6:
        raise SystemExit(f"Expected exactly six durations, found {durations}")

    config_dir = ROOT / "configs" / "sweeps" / "ieee9_probe_amplitude"
    report_root = ROOT / "reports" / "ieee9_probe_amplitude_sweep"
    config_dir.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    for amplitude in amplitudes:
        amp_slug = slug(amplitude)
        cell = report_root / f"amp_{amp_slug}"
        probe_rel = Path("data") / "ieee9_probe_amplitude_sweep" / f"amp_{amp_slug}" / "probe"
        mocu_rel = Path("data") / "ieee9_probe_amplitude_sweep" / f"amp_{amp_slug}" / "mocu"
        cfg = deepcopy(base)
        cfg["data"]["dataset_dir"] = str(probe_rel)
        cfg["data"]["mocu_dataset_dir"] = str(mocu_rel)
        cfg["swing_equation"]["probe_amplitudes"] = [float(amplitude)]
        cfg.setdefault("amplitude_sweep", {})
        cfg["amplitude_sweep"].update(
            {
                "fixed_probe_amplitude_pu": float(amplitude),
                "buses": buses,
                "durations_s": durations,
                "n_actions": len(buses) * len(durations),
                "methods_trained": [],
            }
        )
        cfg_path = config_dir / f"ieee9_amp_{amp_slug}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        cell.mkdir(parents=True, exist_ok=True)
        log = cell / "run.log"

        run(
            [
                sys.executable,
                "-m",
                "src.experiment",
                "generate-data",
                "--config",
                str(cfg_path.relative_to(ROOT)),
                "--experiment-type",
                "eig_based",
                "--N_obs",
                str(args.n_obs),
                "--noise_sigma",
                str(args.noise_sigma),
                "--seed",
                "101",
                "--exp-dir",
                str(cell / "generation_record"),
            ],
            log,
        )
        run(
            [
                sys.executable,
                "tools/regenerate_mocu_bank.py",
                "--config",
                str(cfg_path.relative_to(ROOT)),
                "--output",
                str(mocu_rel),
                "--batch-size",
                str(args.batch_size),
            ],
            log,
        )
        audit_dir = cell / "audit"
        audit_rc = run(
            [
                sys.executable,
                "-m",
                "src.experiment",
                "bank-structure-audit",
                "--config",
                str(cfg_path.relative_to(ROOT)),
                "--experiment-type",
                "objective_based",
                "-T",
                "2",
                "--N_obs",
                str(args.n_obs),
                "--noise_sigma",
                str(args.noise_sigma),
                "--support-size",
                str(args.support_size),
                "--n-outer",
                str(args.n_outer),
                "--n-inner",
                str(args.n_inner),
                "--top-k",
                str(args.top_k),
                "--seed",
                str(args.seed),
                "--exp-dir",
                str(audit_dir),
            ],
            log,
            allow_failure=True,
        )
        audit_json = audit_dir / "diagnostics" / "bank_structure_audit.json"
        summary = {
            "amplitude_pu": amplitude,
            "durations_s": durations,
            "buses": buses,
            "n_actions": len(buses) * len(durations),
            "N_obs": args.n_obs,
            "noise_sigma": args.noise_sigma,
            "probe_bank": str(probe_rel),
            "mocu_bank": str(mocu_rel),
            "config": str(cfg_path.relative_to(ROOT)),
            "audit_exit_code": audit_rc,
            **load_audit(audit_json),
        }
        summaries.append(summary)
        (cell / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    aggregate = {
        "purpose": "method-independent IEEE9 fixed-probe-amplitude structural sweep",
        "methods_trained": [],
        "controlled_factors": {
            "durations_s": durations,
            "buses": buses,
            "n_actions": len(buses) * len(durations),
            "N_obs": args.n_obs,
            "noise_sigma": args.noise_sigma,
            "theta_train_seed": int(base["data_generation"]["train_seed"]),
            "theta_test_seed": int(base["data_generation"]["test_seed"]),
        },
        "cells": summaries,
    }
    out = report_root / "amplitude_sweep_summary.json"
    out.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
