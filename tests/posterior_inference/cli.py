"""
Single-step posterior / MOCU check (T=1, YAML physics only).

  python tests/posterior_inference/cli.py

Uses ``config/early_test.yaml``, grid 8×8, seed 1. Writes
``tests/posterior_inference/output/single_step_early_test.json``.

**sBOED:** this run is **T = 1** (one sequential experimental step—the single-step special case).

**Device:** uses **PyTorch CUDA** when a GPU is available (same ODE path as ``likelihood`` /
``swing_equation_mocu``). Not the separate **PyCUDA** MOCU kernels. Force CPU with
``POSTERIOR_DEVICE=cpu``.

**This can sit with no output for many minutes:** 64× γ* + ODEs (``T_obs_sec`` is ODE horizon, not sBOED T).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent.parent))
sys.path.insert(0, str(_DIR))

from episode_helpers import (  # noqa: E402
    DEFAULT_SWING_YAML,
    initial_setup_document,
    resolve_inference_device,
    run_single_step_physics_episode,
    swing_physics_kwargs_from_yaml,
)

CONFIG = DEFAULT_SWING_YAML
GRID_SIDE = 8
SEED = 1
DEVICE = resolve_inference_device()
_OUTPUT_DIR = _DIR / "output"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _json_safe(x: Any) -> Any:
    import numpy as np

    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    return str(x)


def main() -> None:
    if not CONFIG.is_file():
        raise SystemExit(f"Config not found: {CONFIG}")

    print(
        "Running single-step physics (no progress until done): "
        f"device={DEVICE}, grid {GRID_SIDE}×{GRID_SIDE}, config {CONFIG.name} — "
        "often several minutes. Reduce GRID_SIDE in cli.py for a quicker check.",
        flush=True,
    )

    kw = swing_physics_kwargs_from_yaml(CONFIG)
    initial_setup = initial_setup_document(
        CONFIG,
        grid_side=GRID_SIDE,
        seed=SEED,
        device=DEVICE,
        T=1,
    )

    out = run_single_step_physics_episode(
        seed=SEED,
        grid_side=GRID_SIDE,
        device=DEVICE,
        config_path=CONFIG,
    )
    rep = out["single_step_report"]

    out_path = _OUTPUT_DIR / f"single_step_{CONFIG.stem}.json"
    document: dict[str, Any] = {
        "initial_setup": initial_setup,
        "device": DEVICE,
        "physics_kw_from_config": {k: list(v) if isinstance(v, tuple) else v for k, v in kw.items()},
        "physics_meta": _json_safe(out["physics_meta"]),
        "xi": list(out["xi"]),
        "grid_side": int(out["grid_side"]),
        "n_support": int(out["n"]),
        "theta_true": [float(x) for x in out["theta_true"]],
        "sigma_feat": float(rep["sigma_feat"]),
        "y_clean": float(out["y_clean"]),
        "y_step": float(out["y_steps"][0]),
        "noise": float(rep["y"]) - float(out["y_clean"]),
        "gamma_star_true": float(out["gamma_star_true"]),
        "u_ctrl_max_abs": float(out["u_ctrl_max_abs"]),
        "gamma_hat_prior": float(rep["gamma_hat_prior"]),
        "gamma_hat_post": float(rep["gamma_hat_post"]),
        "mocu_prior": float(rep["mocu_prior"]),
        "mocu_post": float(rep["mocu_post"]),
        "single_step_report": {
            "y": float(rep["y"]),
            "sigma_feat": float(rep["sigma_feat"]),
            "mu": _json_safe(rep["mu"]),
            "p0": _json_safe(rep["p0"]),
            "p1": _json_safe(rep["p1"]),
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)

    print(f"Wrote {out_path.resolve()}")
    print(f"mocu_prior={document['mocu_prior']:.6g}  mocu_post={document['mocu_post']:.6g}")


if __name__ == "__main__":
    main()
