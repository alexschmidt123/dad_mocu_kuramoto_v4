#!/usr/bin/env python3
"""Fixed-bus duration-scale vs amp-scale structure audit (test harness).

Generates two physical banks under ``tools/fixed_bus_scale_audit/data/`` and
runs Myopic-trap + adaptive-room checks for each (pass/fail only; no filtering).

Usage (repo root)::

    python3 tools/fixed_bus_scale_audit/run_fixed_bus_scale_audit.py
    python3 tools/fixed_bus_scale_audit/run_fixed_bus_scale_audit.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config_for_run, repo_root
from src.banks.audit import (
    run_bank_structure_audit,
    write_audit_report,
)
from src.banks.power_grid import generate_physical_bank
from src.domains.swing.design import build_catalog

TEST_DIR = Path(__file__).resolve().parent
CONFIGS = {
    "duration_scale": TEST_DIR / "configs" / "ieee5_fixedbus_duration_scale.yaml",
    "amp_scale": TEST_DIR / "configs" / "ieee5_fixedbus_amp_scale.yaml",
}
DEFAULT_SIGMAS = (0.01, 0.001)
DEFAULT_N_OBS = 200


def _compact_audit(report: dict[str, Any]) -> dict[str, Any]:
    t2 = report.get("t2_adaptive_screen") or {}
    trap = report.get("myopic_trap") or {}
    red = report.get("action_redundancy") or {}
    return {
        "N_obs": report.get("N_obs"),
        "noise_sigma": report.get("noise_sigma"),
        "verdict": report.get("verdict"),
        "myopic_trap": bool(trap.get("trap_present")),
        "strong_trap": bool(trap.get("strong_trap")),
        "fixed_beatable": bool(report.get("fixed_beatable")),
        "branching_ok": bool(report.get("branching_ok")),
        "adaptive_room": bool(report.get("adaptive_room")),
        "planning_minus_myopic": t2.get("planning_minus_myopic"),
        "planning_minus_fixed": t2.get("planning_minus_fixed"),
        "n_distinct_second_actions": t2.get("n_distinct_second_actions"),
        "second_action_entropy": t2.get("second_action_entropy"),
        "J_myopic": t2.get("J_myopic_T2"),
        "J_planning": t2.get("J_planning_T2"),
        "J_fixed": t2.get("J_fixed_T2_approx"),
        "amp_scale_redundant": red.get("amp_scale_redundant"),
        "near_duplicate_frac": red.get("near_duplicate_frac"),
        "same_bus_near_dup_frac": red.get("same_bus_near_dup_frac"),
        "recommendations": list(report.get("recommendations") or []),
    }


def _run_variant(
    name: str,
    config_path: Path,
    *,
    force: bool,
    smoke: bool,
    n_obs: int,
    sigmas: list[float],
    results_dir: Path,
) -> dict[str, Any]:
    root = repo_root()
    cfg = load_config_for_run(str(config_path), root, step_number=2)
    catalog = build_catalog(cfg)
    print(f"\n=== {name} ===")
    print(f"config={config_path}")
    print(f"n_actions={len(catalog)} designs={[d.as_tuple() for d in catalog]}")

    gen = generate_physical_bank(cfg, project_root=root, smoke=smoke, force=force)
    data_dir = Path(gen["data_dir"])
    print(f"bank → {data_dir} (reused={gen.get('reused')})")

    variant_out = results_dir / name
    variant_out.mkdir(parents=True, exist_ok=True)
    audits: list[dict[str, Any]] = []
    for sigma in sigmas:
        report = run_bank_structure_audit(
            cfg,
            n_obs=int(n_obs),
            noise_sigma=float(sigma),
            support_size=96 if not smoke else 24,
            n_outer=24 if not smoke else 8,
            n_inner=16 if not smoke else 8,
            top_k=min(12, len(catalog)),
            seed=20260811,
            project_root=root,
        )
        write_audit_report(report, variant_out)
        # write_audit_report uses fixed filenames; rename per sigma
        src_json = variant_out / "bank_structure_audit.json"
        src_md = variant_out / "bank_structure_audit.md"
        sigma_tag = str(sigma).replace(".", "p")
        dst_json = variant_out / f"bank_structure_audit_sigma{sigma_tag}.json"
        dst_md = variant_out / f"bank_structure_audit_sigma{sigma_tag}.md"
        if src_json.is_file():
            src_json.replace(dst_json)
        if src_md.is_file():
            src_md.replace(dst_md)
        compact = _compact_audit(report)
        audits.append(compact)
        print(
            f"  σ={sigma}: trap={compact['myopic_trap']} "
            f"adaptive_room={compact['adaptive_room']} "
            f"(fixed_beatable={compact['fixed_beatable']}, "
            f"branch={compact['n_distinct_second_actions']}) "
            f"plan−fixed={compact['planning_minus_fixed']:.6f} "
            f"amp_scale_redundant={compact['amp_scale_redundant']}"
        )

    return {
        "variant": name,
        "config": str(config_path.relative_to(root)),
        "data_dir": str(data_dir),
        "n_actions": len(catalog),
        "designs": [list(d.as_tuple()) for d in catalog],
        "generation": {
            "reused": bool(gen.get("reused")),
            "smoke": bool(smoke),
        },
        "audits": audits,
        "pass_all_sigmas": {
            "myopic_trap": all(a["myopic_trap"] for a in audits),
            "adaptive_room": all(a["adaptive_room"] for a in audits),
        },
    }


def _write_summary(summary: dict[str, Any], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "summary.json"
    md_path = results_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Fixed-bus scale audit summary\n",
        "Compare **duration scale** vs **amp scale** on bus 0 "
        "(Myopic trap + adaptive room). Pass/fail only — no data filtering.\n",
        f"- N_obs={summary['N_obs']}",
        f"- noise_sigmas={summary['noise_sigmas']}",
        f"- smoke={summary['smoke']}\n",
        "| Variant | σ | Myopic trap | Adaptive room | plan−Fixed | "
        "ξ₂ distinct | amp_scale_redundant |",
        "|---------|---|-------------|---------------|------------|"
        "------------|---------------------|",
    ]
    for v in summary["variants"]:
        for a in v["audits"]:
            lines.append(
                f"| {v['variant']} | {a['noise_sigma']} | "
                f"{a['myopic_trap']} | {a['adaptive_room']} | "
                f"{a['planning_minus_fixed']:.6f} | "
                f"{a['n_distinct_second_actions']} | "
                f"{a['amp_scale_redundant']} |"
            )
    lines.append("\n## Takeaway\n")
    lines.append(summary.get("takeaway", ""))
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


def _takeaway(variants: list[dict[str, Any]]) -> str:
    by = {v["variant"]: v for v in variants}
    dur = by.get("duration_scale")
    amp = by.get("amp_scale")
    bits: list[str] = []
    if dur and amp:
        dur_trap = dur["pass_all_sigmas"]["myopic_trap"]
        amp_trap = amp["pass_all_sigmas"]["myopic_trap"]
        dur_adapt = dur["pass_all_sigmas"]["adaptive_room"]
        amp_adapt = amp["pass_all_sigmas"]["adaptive_room"]
        bits.append(
            f"duration_scale: myopic_trap_all_σ={dur_trap}, "
            f"adaptive_room_all_σ={dur_adapt}."
        )
        bits.append(
            f"amp_scale: myopic_trap_all_σ={amp_trap}, "
            f"adaptive_room_all_σ={amp_adapt}."
        )
        amp_red = any(
            bool(a.get("amp_scale_redundant")) for a in amp.get("audits") or []
        )
        if amp_red:
            bits.append(
                "amp_scale flagged amp_scale_redundant (same-bus multi-amp ≈ "
                "ROCOF scaling) — expected; this is why multi-amp alone rarely "
                "creates a Fixed-beatable adaptive problem."
            )
        if dur_adapt and not amp_adapt:
            bits.append(
                "Duration scaling on a fixed bus creates more adaptive room "
                "than amp scaling — prefer duration diversity in Plan-2 YAML."
            )
        elif not dur_adapt and not amp_adapt:
            bits.append(
                "Neither fixed-bus catalog alone provides adaptive_room on all "
                "σ — need bus diversity and/or stronger U heterogeneity in the "
                "generator YAML (not data filtering)."
            )
        elif amp_adapt and not dur_adapt:
            bits.append(
                "Unexpected: amp_scale shows adaptive_room while duration_scale "
                "does not — inspect per-σ audits before changing Plan-2."
            )
    return " ".join(bits)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate both banks (overwrite under tools/.../data/)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny banks for pipeline smoke (structure numbers less reliable)",
    )
    p.add_argument("--N_obs", type=int, default=DEFAULT_N_OBS)
    p.add_argument(
        "--noise_sigma",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SIGMAS),
        help="Comma-separated sigmas for the structure screen",
    )
    args = p.parse_args()
    sigmas = [float(x) for x in str(args.noise_sigma).split(",") if x.strip()]
    results_dir = TEST_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    variants: list[dict[str, Any]] = []
    for name, cfg_path in CONFIGS.items():
        if not cfg_path.is_file():
            raise FileNotFoundError(cfg_path)
        variants.append(
            _run_variant(
                name,
                cfg_path,
                force=bool(args.force),
                smoke=bool(args.smoke),
                n_obs=int(args.N_obs),
                sigmas=sigmas,
                results_dir=results_dir,
            )
        )

    summary = {
        "purpose": (
            "Test whether duration-scale vs amp-scale on a fixed bus creates "
            "Myopic trap and/or adaptive room (planning beats Fixed + branching)."
        ),
        "N_obs": int(args.N_obs),
        "noise_sigmas": sigmas,
        "smoke": bool(args.smoke),
        "variants": variants,
    }
    summary["takeaway"] = _takeaway(variants)
    _write_summary(summary, results_dir)
    print("\n" + summary["takeaway"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
