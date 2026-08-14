"""U-bank generation via PyCUDA and mandatory safety invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from src.control.u_req import ControlSpec, is_control_safe
from src.domains.swing.simulator import system_mk

if TYPE_CHECKING:
    from src.control.cuda_control import CudaControlEngine


def _cuda_engine_cls():
    from src.control.cuda_control import CudaControlEngine

    return CudaControlEngine


def evaluate_control_metrics(
    engine: Any,
    M: np.ndarray | float,
    K: np.ndarray | float,
    u_ctrl: float,
) -> dict[str, float]:
    """True-system / oracle evaluation — identical physics to U-bank GPU kernel."""
    return engine.evaluate_one(np.asarray(M, dtype=np.float64), np.asarray(K, dtype=np.float64), float(u_ctrl))


def u_req_for_theta(
    engine: Any,
    M: np.ndarray | float,
    K: np.ndarray | float,
    spec: ControlSpec,
) -> tuple[float, dict[str, float]]:
    """Smallest candidate u that is safe; returns (u_req, metrics_at_u_req)."""
    M_v = np.asarray(M, dtype=np.float64).reshape(-1)
    K_v = np.asarray(K, dtype=np.float64).reshape(-1)
    for u in spec.u_grid():
        m = engine.evaluate_one(M_v, K_v, float(u))
        if m["safe"] >= 0.5:
            return float(u), m
    # No feasible candidate — return u_max metrics (caller must fail invariant).
    m = engine.evaluate_one(M_v, K_v, float(spec.u_max))
    return float(spec.u_max), m


def extract_U_bank(systems: list[dict[str, Any]]) -> np.ndarray:
    out = np.empty(len(systems), dtype=np.float64)
    for i, sys in enumerate(systems):
        if "u_req" not in sys:
            raise KeyError(f"system[{i}] missing u_req; run generate-control-bank")
        out[i] = float(sys["u_req"])
    return out


def generate_control_bank_for_split(
    systems: list[dict[str, Any]],
    engine: CudaControlEngine,
    spec: ControlSpec,
    *,
    batch_size: int = 512,
    progress: bool = True,
) -> dict[str, Any]:
    """
    Compute U_bank[n] = u_req(θ_n) on GPU by sweeping candidates per particle.

    Vectorizes as (n_systems × n_candidates) trajectories per batch sweep.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **kwargs):  # type: ignore
            return x

    n = len(systems)
    cands = spec.u_grid()
    n_c = len(cands)
    N = engine.N
    M_rows = np.zeros((n, N), dtype=np.float64)
    K_rows = np.zeros((n, N), dtype=np.float64)
    for i, sys in enumerate(systems):
        M, K = system_mk(sys, N)
        M_rows[i] = M
        K_rows[i] = K

    # Evaluate all (θ, u) pairs.
    M_big = np.repeat(M_rows, n_c, axis=0)
    K_big = np.repeat(K_rows, n_c, axis=0)
    u_big = np.tile(cands, n)
    if progress:
        print(f"  control-bank GPU: {n} θ × {n_c} candidates = {n * n_c} trajectories")
    rocof, nadir = engine.simulate_metrics_batch(M_big, K_big, u_big, batch_size=batch_size)
    rocof = rocof.reshape(n, n_c)
    nadir = nadir.reshape(n, n_c)
    safe = (rocof <= spec.rocof_limit_hz_s) & (nadir >= spec.delta_f_nadir_hz)

    # The posterior-quantile U-bank is valid only when every particle has a
    # feasible restorative control and safety remains true for larger grid
    # values.  Never encode infeasibility as u_max: that makes unsafe particles
    # indistinguishable from genuinely conservative requirements.
    has_safe = safe.any(axis=1)
    u_max_safe = safe[:, -1]
    nonmonotonic = np.any(safe[:, :-1] & ~safe[:, 1:], axis=1)
    if not (has_safe.all() and u_max_safe.all() and not nonmonotonic.any()):
        raise RuntimeError(
            "Invalid terminal-control bank: posterior quantile semantics require "
            "a feasible, monotone safe-control curve for every particle. "
            f"infeasible={int((~has_safe).sum())}/{n}, "
            f"u_max_unsafe={int((~u_max_safe).sum())}/{n}, "
            f"nonmonotonic={int(nonmonotonic.sum())}/{n}. "
            "Balance the initial operating point and retune the contingency, "
            "limits, profile, or candidate range before regenerating."
        )

    U = np.empty(n, dtype=np.float64)
    metrics_at_req: list[dict[str, float]] = []
    for i in range(n):
        ok = np.where(safe[i])[0]
        j = int(ok[0])
        U[i] = float(cands[j])
        systems[i]["u_req"] = float(U[i])
        metrics_at_req.append(
            {
                "rocof_max": float(rocof[i, j]),
                "delta_f_nadir": float(nadir[i, j]),
                "safe": float(safe[i, j]),
            }
        )

    report = {
        "n_systems": n,
        "n_candidates": n_c,
        "u_candidates": [float(x) for x in cands],
        "n_infeasible_u_req": 0,
        "n_nonmonotonic_safe_curves": 0,
        "u_max_safety_rate": float(np.mean(u_max_safe)),
        "u_bank_particle_safety_rate": float(
            np.mean([m["safe"] for m in metrics_at_req])
        ),
        "mean_u_req": float(np.mean(U)),
        "std_u_req": float(np.std(U)),
        "min_u_req": float(np.min(U)),
        "max_u_req": float(np.max(U)),
        "n_unique_u_req": int(len({float(x) for x in U.tolist()})),
        "frac_u_req_zero": float(np.mean(np.isclose(U, 0.0))),
        "frac_u_req_umax": float(np.mean(np.isclose(U, spec.u_max))),
        "nondegenerate": bool(float(np.std(U)) > 0.0 and len({float(x) for x in U.tolist()}) > 1),
    }
    return report


def validate_control_invariants(
    systems: list[dict[str, Any]],
    engine: CudaControlEngine,
    spec: ControlSpec,
    *,
    split_name: str = "split",
) -> dict[str, Any]:
    """
    Mandatory invariants:
      safe(θ_n, U_bank[n]) == True
      safe(θ, u_max) == True for all θ
      (oracle checked separately on true θ with u_req(θ))
    """
    n = len(systems)
    bank_ok = np.zeros(n, dtype=bool)
    umax_ok = np.zeros(n, dtype=bool)
    details = []
    for i, sys in enumerate(systems):
        M, K = system_mk(sys, engine.N)
        u_req = float(sys["u_req"])
        m_req = engine.evaluate_one(M, K, u_req)
        m_max = engine.evaluate_one(M, K, float(spec.u_max))
        bank_ok[i] = m_req["safe"] >= 0.5
        umax_ok[i] = m_max["safe"] >= 0.5
        details.append(
            {
                "index": i,
                "u_req": u_req,
                "bank_safe": bool(bank_ok[i]),
                "u_max_safe": bool(umax_ok[i]),
                "rocof_at_req": m_req["rocof_max"],
                "nadir_at_req": m_req["delta_f_nadir"],
                "rocof_at_umax": m_max["rocof_max"],
                "nadir_at_umax": m_max["delta_f_nadir"],
            }
        )

    report = {
        "split": split_name,
        "n": n,
        "u_bank_particle_safety_rate": float(np.mean(bank_ok)),
        "maximum_control_safety_rate": float(np.mean(umax_ok)),
        "passed": bool(bank_ok.all() and umax_ok.all()),
        "n_bank_failures": int(np.sum(~bank_ok)),
        "n_umax_failures": int(np.sum(~umax_ok)),
        "failures": [d for d in details if not (d["bank_safe"] and d["u_max_safe"])],
    }
    return report


def save_control_bank_sidecar(
    data_path: Path,
    split: str,
    systems: list[dict[str, Any]],
    report: dict[str, Any],
    spec: ControlSpec,
) -> Path:
    """Write U values + metadata next to the probe bank (does not rewrite probe JSON schema lightly)."""
    out = data_path / f"{split}_control_bank.json"
    payload = {
        "split": split,
        "control_model": "supplementary_active_power_injection",
        "spec": {
            "alpha": spec.alpha,
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
            "fs_hz": spec.fs_hz,
        },
        "u_req": [float(s["u_req"]) for s in systems],
        "report": report,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def attach_control_bank_to_probe_json(data_path: Path, split: str) -> None:
    """Merge u_req from sidecar into train/test.json for downstream loaders."""
    sidecar = data_path / f"{split}_control_bank.json"
    probe = data_path / f"{split}.json"
    if not sidecar.is_file() or not probe.is_file():
        raise FileNotFoundError(f"missing {sidecar} or {probe}")
    side = json.loads(sidecar.read_text(encoding="utf-8"))
    payload = json.loads(probe.read_text(encoding="utf-8"))
    u_list = side["u_req"]
    systems = payload["systems"]
    if len(u_list) != len(systems):
        raise ValueError(f"{split}: U-bank length {len(u_list)} != n_systems {len(systems)}")
    for s, u in zip(systems, u_list):
        s["u_req"] = float(u)
    payload.setdefault("meta", {})["control"] = side.get("spec")
    payload["meta"]["control_bank_report"] = side.get("report")
    probe.write_text(json.dumps(payload), encoding="utf-8")
