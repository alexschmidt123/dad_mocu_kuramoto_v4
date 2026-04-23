"""
Plot prior / posterior on the (M, K) grid from ``single_step_*.json`` (CLI output).

  python tests/posterior_inference/plot_single_step_posterior.py
  python tests/posterior_inference/plot_single_step_posterior.py path/to/single_step_early_test.json

Writes PNG next to the JSON (same stem + ``_posterior_pi.png``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    here = Path(__file__).resolve().parent
    default_json = here / "output" / "single_step_early_test.json"
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_json
    if not json_path.is_file():
        raise SystemExit(f"JSON not found: {json_path}")

    data = _load(json_path)
    grid_side = int(data["grid_side"])
    sw = data["initial_setup"]["swing_equation"]
    M_lo, M_hi = float(sw["M_lower"]), float(sw["M_upper"])
    K_lo, K_hi = float(sw["K_lower"]), float(sw["K_upper"])
    rep = data["single_step_report"]
    p0 = np.asarray(rep["p0"], dtype=np.float64).reshape(grid_side, grid_side)
    p1 = np.asarray(rep["p1"], dtype=np.float64).reshape(grid_side, grid_side)
    y = float(rep["y"])
    sigma = float(rep["sigma_feat"])

    theta_true = data.get("theta_true")
    M_true, K_true = (float(theta_true[0]), float(theta_true[1])) if theta_true else (None, None)

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for this script. Install with: pip install matplotlib"
        ) from e

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    extent = [K_lo, K_hi, M_lo, M_hi]

    n_tick = min(5, max(3, grid_side))
    k_ticks = np.linspace(K_lo, K_hi, n_tick)
    m_ticks = np.linspace(M_lo, M_hi, n_tick)

    def _style_param_axes(ax) -> None:
        ax.set_xticks(k_ticks)
        ax.set_xticklabels([f"{t:.2f}" for t in k_ticks])
        ax.set_yticks(m_ticks)
        ax.set_yticklabels([f"{t:.3f}" for t in m_ticks])
        ax.set_xlabel(rf"$K$ — support $[{K_lo:.2f},\,{K_hi:.2f}]$")
        ax.set_ylabel(rf"$M$ — support $[{M_lo:.3f},\,{M_hi:.3f}]$")

    ax0, ax1 = axes
    im0 = ax0.imshow(
        p0,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=p0.max() * 1.05 or 1.0,
    )
    ax0.set_title(r"Prior $p_0(\theta)$ — uniform on grid")
    _style_param_axes(ax0)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="mass")

    im1 = ax1.imshow(
        p1,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="Oranges",
        vmin=0,
        vmax=max(p1.max(), 1e-12),
    )
    ax1.set_title(r"Posterior $p_1(\theta \mid y)$ after one Gaussian update")
    _style_param_axes(ax1)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="mass")

    for ax in axes:
        if M_true is not None:
            ax.scatter(
                [K_true],
                [M_true],
                c="lime",
                s=120,
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
                label=r"$\theta_{\mathrm{true}}$",
            )
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        rf"Single-step discrete Bayes ($N={grid_side}\times{grid_side}$) | "
        rf"$y={y:.4f}$, $\sigma_{{\mathrm{{feat}}}}={sigma}$",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        rf"Tensor-product grid on the support box above. Minimum $K={K_lo:.2f}$ (left edge of the plots).",
        ha="center",
        fontsize=8,
        color="0.25",
    )

    out_path = json_path.with_name(json_path.stem + "_posterior_pi.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
