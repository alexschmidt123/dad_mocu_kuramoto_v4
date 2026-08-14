"""Diagnose U-bank degeneracy / binding constraints (no method training).

Core audit used by the EIG control-bank generation path
(``src.objectives.eig.cli``). Offline shim: ``tools.diagnostics.diagnose_control``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.config import SBOEDConfig, load_config_for_run, repo_root
from src.control.cuda_control import CudaControlEngine
from src.control.u_req import ControlSpec
from src.banks.tables import get_systems, load_tables, resolve_data_path
from src.domains.swing.design import build_simulator
from src.domains.swing.simulator import system_mk


def _u_stats(U: np.ndarray) -> dict[str, Any]:
    U = np.asarray(U, dtype=np.float64).reshape(-1)
    uniq = sorted({float(x) for x in U.tolist()})
    return {
        "n": int(U.size),
        "mean": float(np.mean(U)) if U.size else float("nan"),
        "std": float(np.std(U)) if U.size else float("nan"),
        "min": float(np.min(U)) if U.size else float("nan"),
        "max": float(np.max(U)) if U.size else float("nan"),
        "n_unique": int(len(uniq)),
        "unique": uniq,
        "counts": {str(k): int(v) for k, v in sorted(Counter(U.tolist()).items())},
        "frac_at_0": float(np.mean(np.isclose(U, 0.0))) if U.size else float("nan"),
        "nondegenerate": bool(U.size > 0 and float(np.std(U)) > 0.0 and len(uniq) > 1),
    }


def diagnose_split(
    systems: list[dict[str, Any]],
    engine: CudaControlEngine,
    spec: ControlSpec,
    *,
    split: str,
    batch_size: int = 512,
) -> dict[str, Any]:
    """GPU sweep of (θ, u) for one split; classify degeneracy causes."""
    n = len(systems)
    cands = spec.u_grid()
    n_c = len(cands)
    N = engine.N
    M = np.zeros((n, N), dtype=np.float64)
    K = np.zeros((n, N), dtype=np.float64)
    for i, sys in enumerate(systems):
        Mi, Ki = system_mk(sys, N)
        M[i] = Mi
        K[i] = Ki

    rocof, nadir = engine.simulate_metrics_batch(
        np.repeat(M, n_c, axis=0),
        np.repeat(K, n_c, axis=0),
        np.tile(cands, n),
        batch_size=batch_size,
    )
    rocof = rocof.reshape(n, n_c)
    nadir = nadir.reshape(n, n_c)
    rocof_ok = rocof <= spec.rocof_limit_hz_s
    nadir_ok = nadir >= spec.delta_f_nadir_hz
    safe = rocof_ok & nadir_ok

    U = np.empty(n, dtype=np.float64)
    stored = np.array([float(s.get("u_req", np.nan)) for s in systems], dtype=np.float64)
    binding_prev: list[str] = []
    fail_u0: list[str] = []
    first_safe_idx = np.full(n, -1, dtype=np.int32)
    n_infeasible = 0
    for i in range(n):
        ok = np.where(safe[i])[0]
        if ok.size == 0:
            n_infeasible += 1
            j = n_c - 1
            U[i] = float(cands[j])
            first_safe_idx[i] = -1
            binding_prev.append("infeasible_clipped_umax")
        else:
            j = int(ok[0])
            U[i] = float(cands[j])
            first_safe_idx[i] = j
            if j == 0:
                binding_prev.append("already_safe_at_0")
            else:
                jp = j - 1
                r_f = not bool(rocof_ok[i, jp])
                n_f = not bool(nadir_ok[i, jp])
                if r_f and n_f:
                    binding_prev.append("both")
                elif n_f:
                    binding_prev.append("nadir")
                elif r_f:
                    binding_prev.append("rocof")
                else:
                    binding_prev.append("unknown")

        if safe[i, 0]:
            fail_u0.append("safe")
        elif (not rocof_ok[i, 0]) and (not nadir_ok[i, 0]):
            fail_u0.append("both")
        elif not nadir_ok[i, 0]:
            fail_u0.append("nadir")
        else:
            fail_u0.append("rocof")

    # Non-monotonic safety (larger u becomes unsafe) → first-safe may be fragile.
    n_nonmono = 0
    for i in range(n):
        for j in range(n_c - 1):
            if safe[i, j] and not safe[i, j + 1]:
                n_nonmono += 1
                break

    # Grid resolution: fraction whose continuous-ish gap is a single candidate step.
    dU = np.diff(cands) if n_c > 1 else np.array([])
    median_step = float(np.median(dU)) if dU.size else float("nan")

    frac_umax = float(np.mean(np.isclose(U, spec.u_max)))
    frac_0 = float(np.mean(np.isclose(U, 0.0)))
    stats = _u_stats(U)
    stored_stats = _u_stats(stored[~np.isnan(stored)]) if np.any(~np.isnan(stored)) else None

    # Cause flags (not mutually exclusive).
    causes: dict[str, Any] = {
        "grid_resolution": {
            "suspected": bool(stats["n_unique"] <= 2 and n_c <= 3),
            "n_candidates": n_c,
            "median_candidate_step_pu": median_step,
            "note": (
                "Coarse u_candidates can quantize a continuous u_req onto few levels; "
                "not itself a reason for a single constant bank."
            ),
        },
        "upper_bound_clipping": {
            "suspected": bool(frac_umax >= 0.95 or n_infeasible > 0),
            "frac_at_u_max": frac_umax,
            "n_infeasible": int(n_infeasible),
            "u_max": float(spec.u_max),
            "note": (
                "Infeasible particles are assigned u_max; a bank of all u_max usually means "
                "the contingency/limits make every θ unsafe until (or beyond) u_max."
            ),
        },
        "incorrect_first_safe_search": {
            "suspected": bool(n_nonmono > 0),
            "n_particles_nonmonotonic_safe": int(n_nonmono),
            "note": (
                "First ascending safe candidate is correct iff safety is eventually monotone "
                "in u for this injection model. Non-monotonicity warns of ROCOF overshoot."
            ),
        },
        "physically_similar_requirements": {
            "suspected": bool(stats["std"] < 1e-12 or stats["n_unique"] <= 1),
            "std": stats["std"],
            "n_unique": stats["n_unique"],
            "note": (
                "All particles may truly need the same first-safe grid point under a mild "
                "or uniform contingency relative to θ variation."
            ),
        },
        "one_safety_constraint_dominates": {
            "suspected": True,
            "fail_at_u0_counts": dict(Counter(fail_u0)),
            "binding_before_first_safe_counts": dict(Counter(binding_prev)),
            "note": (
                "If almost all unsafe-at-0 particles fail only nadir (or only ROCOF), "
                "that constraint sets u_req for the bank."
            ),
        },
        "t0_contingency_rocof_definition": {
            "suspected": bool(
                Counter(fail_u0).get("rocof", 0) + Counter(fail_u0).get("both", 0) > 0
                and spec.profile.shape != "step"
            ),
            "profile_shape": spec.profile.shape,
            "contingency_magnitude_pu": float(spec.contingency.magnitude),
            "rocof_limit_hz_s": float(spec.rocof_limit_hz_s),
            "p95_rocof_at_u0": float(np.percentile(rocof[:, 0], 95)),
            "p95_rocof_at_umax": float(np.percentile(rocof[:, -1], 95)),
            "note": (
                "A t=0 contingency step sets an instantaneous ROCOF. Shapes that start at 0 "
                "(hann/ramp) cannot cancel that spike; step injection from t_start=0 can."
            ),
        },
    }

    # Correlations with θ (diagnostic only; prior ranges are not widened here).
    corr = {
        "corr_u_req_mean_M": float(np.corrcoef(U, M.mean(axis=1))[0, 1]) if n > 1 else float("nan"),
        "corr_u_req_min_M": float(np.corrcoef(U, M.min(axis=1))[0, 1]) if n > 1 else float("nan"),
        "corr_u_req_mean_K": float(np.corrcoef(U, K.mean(axis=1))[0, 1]) if n > 1 else float("nan"),
    }

    return {
        "split": split,
        "control_model": "supplementary_active_power_injection",
        "spec_summary": {
            "rocof_limit_hz_s": spec.rocof_limit_hz_s,
            "delta_f_nadir_hz": spec.delta_f_nadir_hz,
            "contingency": {
                "bus": spec.contingency.bus,
                "magnitude": spec.contingency.magnitude,
                "units": spec.contingency.units,
            },
            "profile": {
                "bus": spec.profile.bus,
                "t_start": spec.profile.t_start,
                "duration": spec.profile.duration,
                "shape": spec.profile.shape,
                "units": spec.profile.units,
            },
            "u_candidates": list(spec.u_candidates),
            "T_obs_sec": spec.T_obs_sec,
            "ode_dt": spec.ode_dt,
        },
        "u_req_recomputed": stats,
        "u_req_stored_in_json": stored_stats,
        "frac_at_u_max": frac_umax,
        "frac_at_0": frac_0,
        "n_infeasible": int(n_infeasible),
        "u_max_safety_rate": float(np.mean(safe[:, -1])),
        "oracle_proxy_first_safe_safety_rate": float(np.mean(safe[np.arange(n), np.clip(first_safe_idx, 0, n_c - 1)]))
        if n_infeasible == 0
        else float("nan"),
        "metric_summaries": {
            "rocof_at_u0": {
                "median": float(np.median(rocof[:, 0])),
                "p95": float(np.percentile(rocof[:, 0], 95)),
            },
            "nadir_at_u0": {
                "median": float(np.median(nadir[:, 0])),
                "p05": float(np.percentile(nadir[:, 0], 5)),
            },
            "rocof_at_umax": {
                "median": float(np.median(rocof[:, -1])),
                "p95": float(np.percentile(rocof[:, -1], 95)),
            },
            "nadir_at_umax": {
                "median": float(np.median(nadir[:, -1])),
                "p05": float(np.percentile(nadir[:, -1], 5)),
            },
        },
        "theta_correlations": corr,
        "cause_assessment": causes,
        "verdict": _verdict(stats, causes, frac_umax, n_infeasible),
    }


def _verdict(
    stats: dict[str, Any],
    causes: dict[str, Any],
    frac_umax: float,
    n_infeasible: int,
) -> dict[str, Any]:
    if not stats["nondegenerate"]:
        primary = []
        if causes["upper_bound_clipping"]["suspected"]:
            primary.append("upper_bound_clipping")
        if causes["physically_similar_requirements"]["suspected"]:
            primary.append("physically_similar_requirements")
        if causes["t0_contingency_rocof_definition"]["suspected"]:
            primary.append("t0_contingency_rocof_definition")
        if causes["incorrect_first_safe_search"]["suspected"]:
            primary.append("incorrect_first_safe_search")
        if causes["grid_resolution"]["suspected"]:
            primary.append("grid_resolution")
        return {
            "nondegenerate": False,
            "primary_causes": primary or ["physically_similar_requirements"],
            "message": (
                "U-bank is degenerate (std=0 or a single unique value). "
                "Retune contingency / limits / profile before method training."
            ),
        }
    notes = []
    if stats["frac_at_0"] > 0.4:
        notes.append("Many particles are safe at u=0; objective signal is weak though nondegenerate.")
    if frac_umax > 0.2:
        notes.append("Nontrivial mass sits at u_max; consider enlarging the safe candidate range carefully.")
    if n_infeasible:
        notes.append("Infeasible particles exist — safety invariants will fail.")
    return {
        "nondegenerate": True,
        "primary_causes": [],
        "message": "U-bank varies across particles." + ((" " + " ".join(notes)) if notes else ""),
    }


def diagnose_control_objective(
    config_name: str,
    *,
    project_root: Path | None = None,
    splits: tuple[str, ...] = ("train", "test"),
) -> dict[str, Any]:
    """Run diagnosis for configured splits; write JSON under the data directory."""
    root = project_root or repo_root()
    cfg = load_config_for_run(config_name, root)
    if str(cfg.data.get("backend", "cuda")).lower() != "cuda":
        raise RuntimeError("diagnose-control-objective requires data_generation.backend: cuda")

    data_path = resolve_data_path(root, cfg)
    if not (data_path / "train.json").is_file():
        raise FileNotFoundError(f"Probe bank missing at {data_path}. Run generate-data first.")

    spec = ControlSpec.from_cfg(cfg)
    sim = build_simulator(cfg)
    sim.T_obs_sec = float(spec.T_obs_sec)
    sim.ode_dt = float(spec.ode_dt)
    sim.fs_hz = float(spec.fs_hz)
    engine = CudaControlEngine(sim, spec)
    batch = int(cfg.data.get("cuda_batch_size", 512))

    report: dict[str, Any] = {
        "data_path": str(data_path),
        "config": str(cfg.config_path),
        "splits": {},
    }
    for split in splits:
        systems = list(get_systems(load_tables(data_path / f"{split}.json")))
        print(f"\n=== diagnose-control-objective [{split}] n={len(systems)} ===")
        split_rep = diagnose_split(systems, engine, spec, split=split, batch_size=batch)
        report["splits"][split] = split_rep
        _print_split(split_rep)

    out = data_path / "control_objective_diagnosis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDiagnosis → {out}")
    return report


def _print_split(rep: dict[str, Any]) -> None:
    st = rep["u_req_recomputed"]
    print(
        f"  u_req: mean={st['mean']:.4f}  std={st['std']:.4f}  "
        f"min={st['min']:.4f}  max={st['max']:.4f}  "
        f"n_unique={st['n_unique']}  frac0={st['frac_at_0']:.3f}  "
        f"frac_umax={rep['frac_at_u_max']:.3f}"
    )
    print(f"  unique levels: {st['unique']}")
    print(f"  counts: {st['counts']}")
    print(f"  fail@u=0: {rep['cause_assessment']['one_safety_constraint_dominates']['fail_at_u0_counts']}")
    print(
        "  binding before first-safe: "
        f"{rep['cause_assessment']['one_safety_constraint_dominates']['binding_before_first_safe_counts']}"
    )
    print(f"  θ correlations: {rep['theta_correlations']}")
    print(f"  VERDICT: {rep['verdict']['message']}")
    if rep["verdict"]["primary_causes"]:
        print(f"  primary causes: {rep['verdict']['primary_causes']}")


def control_bank_nondegenerate(data_path: Path) -> tuple[bool, dict[str, Any]]:
    """Check stored U-banks for std>0 and >1 unique values on every split present."""
    detail: dict[str, Any] = {}
    ok = True
    found = False
    for split in ("train", "test"):
        side = data_path / f"{split}_control_bank.json"
        if not side.is_file():
            continue
        found = True
        payload = json.loads(side.read_text(encoding="utf-8"))
        U = np.asarray(payload.get("u_req", []), dtype=np.float64)
        st = _u_stats(U)
        detail[split] = st
        ok = ok and bool(st["nondegenerate"])
    if not found:
        return False, {"error": f"no *_control_bank.json under {data_path}"}
    return ok, detail
