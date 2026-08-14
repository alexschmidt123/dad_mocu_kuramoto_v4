"""Diagnose continuous-duration R(θ,d)=max|RoCoF| bank before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.banks.power_grid import load_bank_from_path, resolve_dataset_dir
from src.config import load_config, repo_root


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def diagnose(
    *,
    config: str | Path,
    out_dir: Path | None = None,
    n_curve_theta: int = 12,
    seed: int = 101,
) -> dict:
    cfg = load_config(config)
    root = repo_root()
    data_dir = resolve_dataset_dir(cfg, root)
    bank = load_bank_from_path(data_dir, cfg=cfg, skip_quality_check=True)
    R = np.asarray(bank["max_rocof_train"], dtype=np.float64)
    M = np.asarray(bank["M_train"], dtype=np.float64)
    K = np.asarray(bank["K_train"], dtype=np.float64)
    M_s = M.mean(axis=1) if M.ndim > 1 else M
    K_s = K.mean(axis=1) if K.ndim > 1 else K

    cat = json.loads((data_dir / "meta" / "catalog.json").read_text(encoding="utf-8"))
    designs = list(cat.get("designs") or [])
    durs = np.asarray(
        [
            float(d[2]) if isinstance(d, (list, tuple)) else float(d.get("duration"))
            for d in designs
        ],
        dtype=np.float64,
    )
    if durs.size != R.shape[1]:
        raise RuntimeError(f"catalog durations ({durs.size}) != n_actions ({R.shape[1]})")

    out = out_dir or (data_dir / "diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    n_theta = int(R.shape[0])
    pick = np.sort(rng.choice(n_theta, size=min(int(n_curve_theta), n_theta), replace=False))

    d_idx = {
        "short": int(np.argmin(np.abs(durs - 0.05))),
        "mid": int(np.argmin(np.abs(durs - 0.15))),
        "long": int(np.argmin(np.abs(durs - 0.30))),
    }
    pairs: list[dict] = []
    sample_i = rng.choice(n_theta, size=min(80, n_theta), replace=False)
    for a, b in ((d_idx["short"], d_idx["long"]), (d_idx["mid"], d_idx["long"])):
        ra = R[sample_i, a]
        rb = R[sample_i, b]
        for i in range(len(sample_i)):
            for j in range(i + 1, len(sample_i)):
                close = abs(float(ra[i] - ra[j]))
                far = abs(float(rb[i] - rb[j]))
                if close <= 0.02 * max(1.0, float(np.median(np.abs(ra)))) and far >= 0.08 * max(
                    1.0, float(np.median(np.abs(rb)))
                ):
                    pairs.append(
                        {
                            "d1": float(durs[a]),
                            "d2": float(durs[b]),
                            "theta_i": int(sample_i[i]),
                            "theta_j": int(sample_i[j]),
                            "abs_diff_d1": close,
                            "abs_diff_d2": far,
                        }
                    )
    pairs.sort(key=lambda p: p["abs_diff_d2"] - p["abs_diff_d1"], reverse=True)
    pairs = pairs[:20]

    corr_M = [_pearson(R[:, j], M_s) for j in range(R.shape[1])]
    corr_K = [_pearson(R[:, j], K_s) for j in range(R.shape[1])]
    corr_M_arr = np.asarray(corr_M, dtype=np.float64)
    corr_K_arr = np.asarray(corr_K, dtype=np.float64)

    report = {
        "data_dir": str(data_dir.resolve()),
        "n_theta": n_theta,
        "n_actions": int(R.shape[1]),
        "duration_min": float(durs.min()),
        "duration_max": float(durs.max()),
        "complementary_pair_count": int(len(pairs)),
        "complementary_pairs_top": pairs[:8],
        "corr_R_vs_M": {
            "max_abs": float(np.nanmax(np.abs(corr_M_arr))),
            "at_mid": float(corr_M_arr[d_idx["mid"]]),
        },
        "corr_R_vs_K": {
            "max_abs": float(np.nanmax(np.abs(corr_K_arr))),
            "at_mid": float(corr_K_arr[d_idx["mid"]]),
        },
        "k_informative": bool(np.nanmax(np.abs(corr_K_arr)) >= 0.15),
        "m_informative": bool(np.nanmax(np.abs(corr_M_arr)) >= 0.15),
        "curve_theta_indices": pick.tolist(),
    }
    (out / "duration_response_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i in pick:
            ax.plot(durs, R[i], lw=1.0, alpha=0.85)
        ax.set_xlabel("duration d (s)")
        ax.set_ylabel(r"$R(\theta,d)=\max|\mathrm{RoCoF}|$")
        ax.set_title("Duration-response curves (subset of θ)")
        fig.tight_layout()
        fig.savefig(out / "duration_response_curves.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.0))
        ax.plot(durs, corr_M_arr, label=r"corr$(R, M)$")
        ax.plot(durs, corr_K_arr, label=r"corr$(R, K)$")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xlabel("duration d (s)")
        ax.legend()
        ax.set_title("Scalar max-RoCoF sensitivity to M vs K")
        fig.tight_layout()
        fig.savefig(out / "duration_MK_sensitivity.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        report["plot_error"] = str(exc)

    print(json.dumps(report, indent=2))
    print(f"wrote {out / 'duration_response_report.json'}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-curve-theta", type=int, default=12)
    p.add_argument("--seed", type=int, default=101)
    args = p.parse_args()
    diagnose(
        config=args.config,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        n_curve_theta=args.n_curve_theta,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
