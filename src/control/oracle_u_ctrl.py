"""True-θ numerically refined minimum safe control (evaluation oracle only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.control.cuda_control import CudaControlEngine
from src.control.u_req import ControlSpec

# Bump when oracle semantics change so stale caches are ignored.
ORACLE_CACHE_VERSION = 2


def _safe(engine: CudaControlEngine, M: np.ndarray, K: np.ndarray, u: float) -> bool:
    m = engine.evaluate_one(M, K, float(u))
    return bool(m["safe_total"] >= 0.5)


def compute_u_ctrl_opt(
    engine: CudaControlEngine,
    M: np.ndarray,
    K: np.ndarray,
    spec: ControlSpec,
    *,
    tolerance: float = 1e-4,
    coarse_points: int = 33,
) -> dict[str, Any]:
    """
    Minimum safe control under true θ, searched on ``[0, u_max]`` only.

    Does **not** assume safety is monotone in ``u`` and never extends past
    ``max(u_candidates)`` (the old ``×1.5`` loop produced fake optima like
    ``1.2 * 1.5**8 ≈ 30.75`` when large ``u`` made the system unsafe).

    Algorithm:
      1) Dense scan on ``[0, u_max]`` (candidates ∪ linspace).
      2) If any safe ``u``: take the smallest; optionally bisect in the local
         bracket ``(last_unsafe, first_safe)`` when that bracket is valid.
      3) If none safe: ``feasible=False``, ``u_ctrl_opt = u_max`` (same as bank).
    """
    u_max = float(max(spec.u_candidates)) if len(spec.u_candidates) else 1.0
    u_lo = 0.0
    cands = [float(u) for u in spec.u_candidates]
    n_coarse = max(int(coarse_points), len(cands) * 2, 33)
    grid = np.unique(
        np.concatenate(
            [
                np.asarray(cands, dtype=np.float64),
                np.linspace(u_lo, u_max, n_coarse),
            ]
        )
    )
    safes = [_safe(engine, M, K, float(u)) for u in grid]
    first_safe = next((i for i, s in enumerate(safes) if s), None)
    if first_safe is None:
        return {
            "u_ctrl_opt": float(u_max),
            "feasible": False,
            "monotonic": False,
            "message": "no_safe_control_in_u_max",
        }

    monotonic = all(safes[first_safe:])
    # Local refine: bisect only inside the first unsafe→safe step (works for
    # both monotone [u*,∞) and a first safe island).
    if first_safe == 0:
        return {
            "u_ctrl_opt": 0.0,
            "feasible": True,
            "monotonic": monotonic,
            "message": "safe_at_zero",
        }

    unsafe = float(grid[first_safe - 1])
    safe = float(grid[first_safe])
    # Guard: bracket must be unsafe then safe.
    if _safe(engine, M, K, unsafe) or not _safe(engine, M, K, safe):
        return {
            "u_ctrl_opt": safe,
            "feasible": True,
            "monotonic": monotonic,
            "message": "grid_first_safe",
        }

    tol = float(tolerance)
    while safe - unsafe > tol:
        mid = 0.5 * (safe + unsafe)
        if _safe(engine, M, K, mid):
            safe = mid
        else:
            unsafe = mid

    return {
        "u_ctrl_opt": float(safe),
        "feasible": True,
        "monotonic": bool(monotonic),
        "message": "bisection" if monotonic else "bisection_first_island",
        "bracket": [float(unsafe), float(safe)],
        "tolerance": tol,
    }


def load_or_compute_oracle_cache(
    cache_path: Path,
    test_systems: list[dict[str, Any]],
    engine: CudaControlEngine,
    spec: ControlSpec,
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Cache u_ctrl_opt per test θ index; reuse across methods."""
    if cache_path.is_file():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            isinstance(raw, dict)
            and int(raw.get("version", 0)) == ORACLE_CACHE_VERSION
            and isinstance(raw.get("rows"), list)
            and len(raw["rows"]) == len(test_systems)
        ):
            return list(raw["rows"])
        # Legacy list caches (v1) are discarded — they may contain ×1.5 ceilings.

    rows: list[dict[str, Any]] = []
    for i, sys in enumerate(test_systems):
        M = np.asarray(sys["M"], dtype=np.float64)
        K = np.asarray(sys["K"], dtype=np.float64)
        out = compute_u_ctrl_opt(engine, M, K, spec, tolerance=tolerance)
        rows.append({"theta_id": i, **out})
        if (i + 1) % max(1, len(test_systems) // 5) == 0:
            print(f"  oracle θ {i + 1}/{len(test_systems)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"version": ORACLE_CACHE_VERSION, "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def check_oracle_consistency(
    u_ctrl_method: float,
    u_ctrl_opt: float,
    method_safe: bool,
    *,
    tolerance: float,
    oracle_feasible: bool = True,
) -> str | None:
    """Return error message if a safe method control is below a feasible oracle."""
    if not oracle_feasible:
        return None
    if method_safe and float(u_ctrl_method) < float(u_ctrl_opt) - float(tolerance):
        return (
            f"oracle_consistency_error: safe u_ctrl={u_ctrl_method} "
            f"< u_ctrl_opt={u_ctrl_opt} - tol={tolerance}"
        )
    return None
