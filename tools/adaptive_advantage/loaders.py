"""Read-only loaders for existing physical banks (no ODE / CUDA)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.config import load_config_for_run, repo_root
from src.control.u_req import ControlSpec

from .config import REPO_ROOT


@dataclass(frozen=True)
class DesignInfo:
    design_id: int
    amp: float
    bus: int
    duration: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "design_id": int(self.design_id),
            "Amp": float(self.amp),
            "bus_location": int(self.bus),
            "duration": float(self.duration),
        }


@dataclass
class SystemBank:
    system: str
    split_support: str
    split_eval: str
    M_support: np.ndarray
    K_support: np.ndarray
    Y_support: np.ndarray  # (n, n_designs) clean max-|ROCOF|
    U_support: np.ndarray
    M_eval: np.ndarray
    K_eval: np.ndarray
    Y_eval: np.ndarray
    U_eval: np.ndarray
    designs: tuple[DesignInfo, ...]
    sigma_y: float
    alpha: float
    safety_margin: float
    u_grid: np.ndarray
    meta: dict[str, Any]
    catalog: dict[str, Any]
    config_path: str
    data_dir: str

    @property
    def n_designs(self) -> int:
        return int(self.Y_support.shape[1])

    @property
    def n_support(self) -> int:
        return int(self.Y_support.shape[0])

    @property
    def n_eval(self) -> int:
        return int(self.Y_eval.shape[0])

    def design_table(self) -> list[dict[str, Any]]:
        return [d.as_dict() for d in self.designs]


def _parse_designs(catalog: dict[str, Any]) -> tuple[DesignInfo, ...]:
    raw = catalog.get("designs") or []
    duration = float(catalog.get("duration_s", 0.2))
    out: list[DesignInfo] = []
    for i, d in enumerate(raw):
        if isinstance(d, (list, tuple)) and len(d) >= 2:
            amp = float(d[0])
            bus = int(d[1])
            dur = float(d[2]) if len(d) >= 3 else duration
        elif isinstance(d, dict):
            amp = float(d.get("amp", d.get("amplitude", 0.0)))
            bus = int(d.get("bus", d.get("bus_location", 0)))
            dur = float(d.get("duration", duration))
        else:
            raise ValueError(f"Unrecognized design entry: {d!r}")
        out.append(DesignInfo(design_id=i, amp=amp, bus=bus, duration=dur))
    return tuple(out)


def load_system_bank(
    system: str,
    *,
    support_size: int | None = None,
    eval_size: int | None = None,
    support_seed: int = 101,
    eval_seed: int = 202,
) -> SystemBank:
    """Load existing bank arrays + control settings. Never runs a simulator."""
    root = repo_root() if (REPO_ROOT / "data").is_dir() else REPO_ROOT
    data_dir = root / "data" / system
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing bank directory: {data_dir}")

    cfg_path = root / "configs" / f"{system}.yaml"
    cfg = load_config_for_run(str(cfg_path), root)
    spec = ControlSpec.from_cfg(cfg)

    catalog = json.loads((data_dir / "meta" / "catalog.json").read_text(encoding="utf-8"))
    meta = yaml.safe_load((data_dir / "meta" / "bank.yaml").read_text(encoding="utf-8")) or {}
    designs = _parse_designs(catalog)

    def _load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sdir = data_dir / split
        M = np.load(sdir / "theta_M.npy")
        K = np.load(sdir / "theta_K.npy")
        Y = np.load(sdir / "max_rocof.npy")
        U = np.load(sdir / "U.npy").reshape(-1).astype(np.float64)
        return M, K, Y, U

    M_tr, K_tr, Y_tr, U_tr = _load_split("train")
    M_te, K_te, Y_te, U_te = _load_split("test")

    rng_s = np.random.default_rng(support_seed)
    rng_e = np.random.default_rng(eval_seed)
    n_tr = Y_tr.shape[0]
    n_te = Y_te.shape[0]
    idx_s = np.arange(n_tr)
    idx_e = np.arange(n_te)
    if support_size is not None and support_size < n_tr:
        idx_s = np.sort(rng_s.choice(n_tr, size=int(support_size), replace=False))
    if eval_size is not None and eval_size < n_te:
        idx_e = np.sort(rng_e.choice(n_te, size=int(eval_size), replace=False))

    sigma_y = float(cfg.sigma_y)
    if "sigma_y" in meta and meta["sigma_y"] is not None:
        # Prefer bank-embedded value when present (matches generation-time setting).
        sigma_y = float(meta["sigma_y"])

    return SystemBank(
        system=system,
        split_support="train",
        split_eval="test",
        M_support=np.asarray(M_tr[idx_s], dtype=np.float64),
        K_support=np.asarray(K_tr[idx_s], dtype=np.float64),
        Y_support=np.asarray(Y_tr[idx_s], dtype=np.float64),
        U_support=np.asarray(U_tr[idx_s], dtype=np.float64),
        M_eval=np.asarray(M_te[idx_e], dtype=np.float64),
        K_eval=np.asarray(K_te[idx_e], dtype=np.float64),
        Y_eval=np.asarray(Y_te[idx_e], dtype=np.float64),
        U_eval=np.asarray(U_te[idx_e], dtype=np.float64),
        designs=designs,
        sigma_y=sigma_y,
        alpha=float(spec.alpha),
        safety_margin=float(spec.safety_margin),
        u_grid=np.asarray(spec.u_grid(), dtype=np.float64),
        meta=dict(meta),
        catalog=catalog,
        config_path=str(cfg_path),
        data_dir=str(data_dir),
    )


def inventory_dict(bank: SystemBank) -> dict[str, Any]:
    amps = sorted({d.amp for d in bank.designs})
    buses = sorted({d.bus for d in bank.designs})
    durs = sorted({d.duration for d in bank.designs})
    return {
        "system": bank.system,
        "available_splits": ["train", "test"],
        "support_split": bank.split_support,
        "eval_split": bank.split_eval,
        "theta_support_shape_M": list(bank.M_support.shape),
        "theta_eval_shape_M": list(bank.M_eval.shape),
        "Y_support_shape": list(bank.Y_support.shape),
        "Y_eval_shape": list(bank.Y_eval.shape),
        "U_support_shape": list(bank.U_support.shape),
        "U_eval_shape": list(bank.U_eval.shape),
        "n_designs": bank.n_designs,
        "Amp_values": amps,
        "bus_locations": buses,
        "fixed_duration": durs[0] if len(durs) == 1 else durs,
        "sigma_y": bank.sigma_y,
        "alpha": bank.alpha,
        "safety_margin": bank.safety_margin,
        "control_grid": [float(x) for x in bank.u_grid.tolist()],
        "source_data_dir": bank.data_dir,
        "source_config": bank.config_path,
        "catalog_path": str(Path(bank.data_dir) / "meta" / "catalog.json"),
        "bank_yaml_path": str(Path(bank.data_dir) / "meta" / "bank.yaml"),
        "physical_simulator_called": False,
    }
