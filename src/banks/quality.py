"""Default physical-bank quality checks for objective sBOED.

Runs after data generation and again when a bank is loaded for train/eval.
Failed checks raise ``BankQualityError`` so later pipeline stages stop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import SBOEDConfig


class BankQualityError(RuntimeError):
    """Physical bank failed sBOED quality gates."""


# Production defaults: ieee5-like headroom; ieee9 (soft contingency) fails today.
DEFAULT_BANK_QUALITY: dict[str, Any] = {
    "enabled": True,
    # Skip strict U/sBOED gates under --smoke (tiny n).
    "skip_sboed_gates_on_smoke": True,
    "min_u_positive_frac": 0.50,
    "min_u_q95": 0.05,
    "min_u_headroom": 0.05,  # Q95(U) - mean(U)
    "min_max_abs_corr_rocof_u": 0.30,
    "max_flat_traj_frac": 0.01,
    "delta_f_sample_theta": 32,
    "delta_f_var_eps": 1.0e-6,
}


def resolve_bank_quality_cfg(
    cfg: SBOEDConfig | None,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    out = dict(DEFAULT_BANK_QUALITY)
    if cfg is not None:
        raw = dict(cfg.raw.get("bank_quality") or {})
        # Also allow under data_generation.bank_quality
        dg = dict(cfg.raw.get("data_generation") or {})
        raw = {**dict(dg.get("bank_quality") or {}), **raw}
        for k, v in raw.items():
            if k in out or k in ("enabled",):
                out[k] = v
    out["smoke"] = bool(smoke)
    return out


def _corr_abs_max(rocof: np.ndarray, U: np.ndarray) -> float:
    U = np.asarray(U, dtype=np.float64).reshape(-1)
    R = np.asarray(rocof, dtype=np.float64)
    if U.size < 3 or R.ndim != 2 or R.shape[0] != U.size:
        return 0.0
    if float(np.std(U)) < 1e-12:
        return 0.0
    best = 0.0
    for a in range(R.shape[1]):
        x = R[:, a]
        if float(np.std(x)) < 1e-12:
            continue
        c = float(np.corrcoef(x, U)[0, 1])
        if np.isfinite(c):
            best = max(best, abs(c))
    return float(best)


def _flat_traj_frac(
    delta_f: np.ndarray,
    *,
    sample_theta: int,
    var_eps: float,
) -> float:
    """Fraction of sampled (θ, action) trajectories with near-zero time variance."""
    arr = np.asarray(delta_f)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return 1.0
    n = int(arr.shape[0])
    k = int(min(max(sample_theta, 1), n))
    idx = np.linspace(0, n - 1, k, dtype=int)
    sample = np.asarray(arr[idx], dtype=np.float64)
    var_t = sample.var(axis=-1)
    return float(np.mean(var_t < float(var_eps)))


def evaluate_split_quality(
    *,
    split: str,
    U: np.ndarray,
    max_rocof: np.ndarray | None,
    delta_f: np.ndarray | None,
    thresholds: dict[str, Any],
    run_sboed_gates: bool,
) -> dict[str, Any]:
    U = np.asarray(U, dtype=np.float64).reshape(-1)
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def _add(name: str, ok: bool, detail: dict[str, Any]) -> None:
        checks.append({"name": name, "ok": bool(ok), **detail})
        if not ok:
            failures.append(name)

    finite_u = bool(np.isfinite(U).all())
    _add("finite_U", finite_u, {"n": int(U.size)})

    if max_rocof is not None:
        R = np.asarray(max_rocof, dtype=np.float64)
        _add(
            "finite_max_rocof",
            bool(np.isfinite(R).all()),
            {"shape": list(R.shape)},
        )
        _add(
            "max_rocof_shape",
            R.ndim == 2 and R.shape[0] == U.size,
            {"shape": list(R.shape), "n_U": int(U.size)},
        )
    else:
        _add("max_rocof_present", False, {})

    if delta_f is not None:
        df = delta_f
        # Cheap finite check on a subsample
        n = int(df.shape[0])
        k = int(min(8, n))
        idx = np.linspace(0, n - 1, k, dtype=int)
        sample = np.asarray(df[idx], dtype=np.float64)
        _add("finite_delta_f_sample", bool(np.isfinite(sample).all()), {})
        flat_frac = _flat_traj_frac(
            df,
            sample_theta=int(thresholds["delta_f_sample_theta"]),
            var_eps=float(thresholds["delta_f_var_eps"]),
        )
        _add(
            "delta_f_not_flat",
            flat_frac <= float(thresholds["max_flat_traj_frac"]),
            {
                "flat_frac": flat_frac,
                "max_flat_traj_frac": float(thresholds["max_flat_traj_frac"]),
            },
        )
    else:
        _add("delta_f_present", False, {})

    u_pos_frac = float(np.mean(U > 1e-12)) if U.size else 0.0
    u_mean = float(U.mean()) if U.size else 0.0
    u_q95 = float(np.quantile(U, 0.95)) if U.size else 0.0
    headroom = float(u_q95 - u_mean)
    corr_max = (
        _corr_abs_max(np.asarray(max_rocof), U) if max_rocof is not None else 0.0
    )

    stats = {
        "n": int(U.size),
        "u_positive_frac": u_pos_frac,
        "u_mean": u_mean,
        "u_q95": u_q95,
        "u_headroom_q95_minus_mean": headroom,
        "u_max": float(U.max()) if U.size else 0.0,
        "max_abs_corr_rocof_u": corr_max,
    }

    if run_sboed_gates:
        _add(
            "u_positive_frac",
            u_pos_frac >= float(thresholds["min_u_positive_frac"]),
            {
                "value": u_pos_frac,
                "min": float(thresholds["min_u_positive_frac"]),
            },
        )
        _add(
            "u_q95",
            u_q95 >= float(thresholds["min_u_q95"]),
            {"value": u_q95, "min": float(thresholds["min_u_q95"])},
        )
        _add(
            "u_headroom",
            headroom >= float(thresholds["min_u_headroom"]),
            {
                "value": headroom,
                "min": float(thresholds["min_u_headroom"]),
            },
        )
        # Only require ROCOF–U correlation when U has spread.
        if u_pos_frac >= 0.05 and float(np.std(U)) > 1e-9:
            _add(
                "rocof_u_correlation",
                corr_max >= float(thresholds["min_max_abs_corr_rocof_u"]),
                {
                    "value": corr_max,
                    "min": float(thresholds["min_max_abs_corr_rocof_u"]),
                },
            )

    return {
        "split": split,
        "passed": len(failures) == 0,
        "failures": failures,
        "checks": checks,
        "stats": stats,
    }


def validate_physical_bank_quality(
    path: Path,
    cfg: SBOEDConfig | None = None,
    *,
    smoke: bool = False,
    write_report: bool = False,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """
    Validate train+test banks for objective sBOED.

    Raises ``BankQualityError`` if any required check fails.
    Never writes reports into ``data/<system>/`` (bank dirs stay data-only).
    """
    path = Path(path)
    # Ensure neat layout (lazy import avoids circular dependency with banks.py).
    from src.banks.power_grid import migrate_flat_bank_to_neat

    migrate_flat_bank_to_neat(path)

    thresholds = resolve_bank_quality_cfg(cfg, smoke=smoke)
    if not bool(thresholds.get("enabled", True)):
        report = {
            "path": str(path.resolve()),
            "passed": True,
            "skipped": True,
            "reason": "bank_quality.enabled=false",
        }
        return report

    run_sboed = True
    if smoke and bool(thresholds.get("skip_sboed_gates_on_smoke", True)):
        run_sboed = False

    required_core = [
        "train/delta_f.npy",
        "test/delta_f.npy",
        "train/max_rocof.npy",
        "test/max_rocof.npy",
    ]
    missing = [n for n in required_core if not (path / n).is_file()]
    for split in ("train", "test"):
        has_psi = (path / split / "psi_star.npy").is_file() or (
            path / split / "U.npy"
        ).is_file()
        if not has_psi:
            missing.append(f"{split}/psi_star.npy")
    if missing:
        raise BankQualityError(
            f"Physical bank incomplete at {path}: missing {missing}"
        )

    splits = {}
    for split in ("train", "test"):
        psi_path = path / split / "psi_star.npy"
        if not psi_path.is_file():
            psi_path = path / split / "U.npy"
        U = np.load(psi_path)
        rocof = np.load(path / split / "max_rocof.npy", mmap_mode="r")
        df = np.load(path / split / "delta_f.npy", mmap_mode="r")
        splits[split] = evaluate_split_quality(
            split=split,
            U=U,
            max_rocof=rocof,
            delta_f=df,
            thresholds=thresholds,
            run_sboed_gates=run_sboed,
        )

    passed = all(splits[s]["passed"] for s in splits)
    report: dict[str, Any] = {
        "path": str(path.resolve()),
        "passed": passed,
        "skipped": False,
        "smoke": bool(smoke),
        "sboed_gates": bool(run_sboed),
        "thresholds": {
            k: thresholds[k]
            for k in (
                "min_u_positive_frac",
                "min_u_q95",
                "min_u_headroom",
                "min_max_abs_corr_rocof_u",
                "max_flat_traj_frac",
            )
        },
        "train": splits["train"],
        "test": splits["test"],
    }

    if write_report and report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not passed:
        fail_bits = []
        for split in ("train", "test"):
            if not splits[split]["passed"]:
                fail_bits.append(
                    f"{split}: {splits[split]['failures']} "
                    f"stats={splits[split]['stats']}"
                )
        hint = (
            "Bank failed sBOED quality gates. "
            "Tighten control.contingency magnitude and/or delta_f_nadir_hz / "
            "rocof_limit_hz_s, then regenerate with --force. "
            "To temporarily bypass (not recommended): set bank_quality.enabled: false."
        )
        raise BankQualityError(hint + "\n" + "\n".join(fail_bits))

    print(
        f"[bank-quality] PASSED → {path} "
        f"(train U>0={splits['train']['stats']['u_positive_frac']:.3f} "
        f"Q95={splits['train']['stats']['u_q95']:.4f}; "
        f"test U>0={splits['test']['stats']['u_positive_frac']:.3f} "
        f"Q95={splits['test']['stats']['u_q95']:.4f})"
    )
    return report
