"""
Sweep Gaussian likelihood scale ``sigma_feat`` for a single-step physics run to find
settings where:

  * prior MOCU is not trivially small (spread in γ* over the grid), and
  * posterior MOCU < prior MOCU (learning), but
  * posterior MOCU is not numerically ~0 (likelihood not too sharp).

**Mechanism:** MOCU is L1 dispersion of γ*(θ) under p(θ). A **too small** ``sigma_feat``
makes the posterior almost a point mass → MOCU_post → 0. A **larger** ``sigma_feat`` softens
the Gaussian likelihood and leaves residual posterior spread.

**Prior MOCU** depends only on {γ*(θ_n)} on the support (uniform prior), not on ``sigma_feat``.
If prior MOCU is too small, increase dispersion of γ* across the grid: wider (M,K) bounds
in ``run_physics_episode`` / grid construction, or a coarser grid over the same box (fewer
points can sometimes miss extremes—usually **widen the physical box** or check γ* batch).

Usage (slow: one full physics episode per σ):

  python tests/posterior_inference/tune_single_step_mocu.py
  python tests/posterior_inference/tune_single_step_mocu.py --grid-side 4 --sigmas 0.08,0.12,0.18,0.25

Set ``POSTERIOR_DEVICE=cpu`` to force CPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent.parent))
sys.path.insert(0, str(_DIR))

from episode_helpers import (  # noqa: E402
    DEFAULT_SWING_YAML,
    resolve_inference_device,
    run_single_step_physics_episode,
)


def _entropy(p) -> float:
    import numpy as np

    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep sigma_feat for single-step MOCU behavior")
    ap.add_argument("--config", type=Path, default=DEFAULT_SWING_YAML)
    ap.add_argument("--grid-side", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--sigmas",
        type=str,
        default="0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30",
        help="Comma-separated sigma_feat values",
    )
    ap.add_argument(
        "--min-prior-mocu",
        type=float,
        default=1.0,
        help="Flag rows where prior MOCU exceeds this (γ* spread diagnostic)",
    )
    ap.add_argument(
        "--min-post-mocu",
        type=float,
        default=1e-6,
        help="Flag rows where posterior MOCU stays above this (avoid collapse)",
    )
    args = ap.parse_args()

    sigmas = [float(x.strip()) for x in args.sigmas.split(",") if x.strip()]
    dev = resolve_inference_device()

    print(
        f"config={args.config.name}  grid={args.grid_side}  seed={args.seed}  device={dev}\n",
        "sigma_feat  mocu_prior  mocu_post  post/prior  H(p1)  notes",
        sep="",
    )
    g_min = g_max = None
    for s in sigmas:
        out = run_single_step_physics_episode(
            seed=args.seed,
            grid_side=args.grid_side,
            device=dev,
            config_path=args.config,
            sigma_feat=s,
        )
        rep = out["single_step_report"]
        mp, mo = float(rep["mocu_prior"]), float(rep["mocu_post"])
        ratio = mo / mp if mp > 0 else float("nan")
        H = _entropy(rep["p1"])
        gn = out["gamma_star_support"]
        if g_min is None:
            import numpy as np

            g_min = float(np.nanmin(gn))
            g_max = float(np.nanmax(gn))
        notes = []
        if mp < args.min_prior_mocu:
            notes.append("low_prior_mocu")
        if mo < args.min_post_mocu:
            notes.append("post_near_zero")
        if mo >= mp:
            notes.append("no_reduction")
        note = ",".join(notes) if notes else "ok"
        print(f"{s:10.4f}  {mp:11.6g}  {mo:9.6g}  {ratio:10.6g}  {H:6.3f}  {note}")

    print(f"\nγ* range on support (for prior MOCU scale): [{g_min:.6g}, {g_max:.6g}]")


if __name__ == "__main__":
    main()
