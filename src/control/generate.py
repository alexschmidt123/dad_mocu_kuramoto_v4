"""Generate / validate the control-requirement U-bank (PyCUDA)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import SBOEDConfig, load_config_for_run, repo_root
from src.banks.control_u import (
    attach_control_bank_to_probe_json,
    generate_control_bank_for_split,
    save_control_bank_sidecar,
    validate_control_invariants,
)
from src.control.cuda_control import CudaControlEngine
from src.control.u_req import ControlSpec
from src.banks.tables import load_tables, resolve_data_path, get_systems
from src.domains.swing.design import build_simulator
from src.domains.swing.simulator import system_mk


def generate_control_bank(
    config_name: str,
    *,
    project_root: Path | None = None,
    splits: tuple[str, ...] = ("train", "test"),
) -> dict[str, Any]:
    """
    Build U-bank for existing probe banks via PyCUDA.

    Does not regenerate probe Y-banks. Writes ``{split}_control_bank.json`` and
    merges ``u_req`` into ``{split}.json``.
    """
    root = project_root or repo_root()
    cfg = load_config_for_run(config_name, root)
    if str(cfg.data.get("backend", "cuda")).lower() != "cuda":
        raise RuntimeError("generate-control-bank requires data_generation.backend: cuda")

    data_path = resolve_data_path(root, cfg)
    if not (data_path / "train.json").is_file():
        raise FileNotFoundError(
            f"Probe bank missing at {data_path}. Run generate-data first."
        )

    spec = ControlSpec.from_cfg(cfg)
    sim = build_simulator(cfg)
    # Align simulator integrator with control spec (shared horizon / dt).
    sim.T_obs_sec = float(spec.T_obs_sec)
    sim.ode_dt = float(spec.ode_dt)
    sim.fs_hz = float(spec.fs_hz)
    engine = CudaControlEngine(sim, spec)
    batch = int(cfg.data.get("cuda_batch_size", 512))

    reports: dict[str, Any] = {"data_path": str(data_path), "splits": {}}
    for split in splits:
        payload = load_tables(data_path / f"{split}.json")
        systems = list(get_systems(payload))
        print(f"\n=== generate-control-bank [{split}] n={len(systems)} ===")
        gen_report = generate_control_bank_for_split(
            systems, engine, spec, batch_size=batch, progress=True,
        )
        inv = validate_control_invariants(systems, engine, spec, split_name=split)
        # Oracle: safe(θ, u_req(θ)) — same as bank particle check when U is from grid.
        oracle_ok = []
        for sys in systems:
            M, K = system_mk(sys, engine.N)
            m = engine.evaluate_one(M, K, float(sys["u_req"]))
            oracle_ok.append(m["safe"] >= 0.5)
        oracle_rate = float(sum(oracle_ok) / max(len(oracle_ok), 1))

        side = save_control_bank_sidecar(data_path, split, systems, {**gen_report, **inv}, spec)
        # Persist u_req into probe JSON for loaders.
        probe_path = data_path / f"{split}.json"
        payload["systems"] = systems
        payload.setdefault("meta", {})["control"] = {
            "rocof_limit_hz_s": spec.rocof_limit_hz_s,
            "delta_f_nadir_hz": spec.delta_f_nadir_hz,
            "profile": {
                "bus": spec.profile.bus,
                "t_start": spec.profile.t_start,
                "duration": spec.profile.duration,
                "shape": spec.profile.shape,
                "units": spec.profile.units,
            },
            "contingency": {
                "bus": spec.contingency.bus,
                "magnitude": spec.contingency.magnitude,
                "units": spec.contingency.units,
            },
            "u_candidates": list(spec.u_candidates),
            "T_obs_sec": spec.T_obs_sec,
            "ode_dt": spec.ode_dt,
            "control_model": "supplementary_active_power_injection",
        }
        payload["meta"]["control_bank_report"] = {
            **gen_report,
            "u_bank_particle_safety_rate": inv["u_bank_particle_safety_rate"],
            "maximum_control_safety_rate": inv["maximum_control_safety_rate"],
            "oracle_control_safety_rate": oracle_rate,
            "invariants_passed": inv["passed"] and oracle_rate >= 1.0 - 1e-12,
        }
        probe_path.write_text(json.dumps(payload), encoding="utf-8")

        split_report = {
            **gen_report,
            **inv,
            "oracle_control_safety_rate": oracle_rate,
            "sidecar": str(side),
        }
        reports["splits"][split] = split_report
        print(
            f"  U-bank safety={inv['u_bank_particle_safety_rate']:.3f}  "
            f"u_max safety={inv['maximum_control_safety_rate']:.3f}  "
            f"oracle safety={oracle_rate:.3f}  "
            f"infeasible={gen_report['n_infeasible_u_req']}  "
            f"std(U)={gen_report.get('std_u_req', float('nan')):.4f}  "
            f"n_unique={gen_report.get('n_unique_u_req', '?')}  "
            f"→ {side.name}"
        )
        if not (inv["passed"] and oracle_rate >= 1.0 - 1e-12):
            print(
                "  FAIL: control-bank invariants not satisfied. "
                "Enlarge u_candidates / soften limits / retune contingency before method comparison."
            )

    out_path = data_path / "control_bank_validation.json"
    out_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nControl-bank validation → {out_path}")
    return reports


def control_banks_certified(data_path: Path) -> tuple[bool, dict[str, Any]]:
    """Return whether train/test control banks meet the mandatory safety rates of 1.0."""
    path = data_path / "control_bank_validation.json"
    if not path.is_file():
        return False, {"error": f"missing {path}; run generate-control-bank"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    ok = True
    detail: dict[str, Any] = {}
    for split, rep in (payload.get("splits") or {}).items():
        ub = float(rep.get("u_bank_particle_safety_rate", 0.0))
        um = float(rep.get("maximum_control_safety_rate", 0.0))
        oc = float(rep.get("oracle_control_safety_rate", 0.0))
        split_ok = ub >= 1.0 - 1e-12 and um >= 1.0 - 1e-12 and oc >= 1.0 - 1e-12
        detail[split] = {
            "u_bank_particle_safety_rate": ub,
            "maximum_control_safety_rate": um,
            "oracle_control_safety_rate": oc,
            "passed": split_ok,
        }
        ok = ok and split_ok
    return ok, detail
