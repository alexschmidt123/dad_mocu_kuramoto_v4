#!/usr/bin/env python3
"""IEEE9 EIG Stage-1 Experiment 1: physics and identifiability audit.

This is a read-only bank diagnostic.  It never trains a BOED method and never
runs the swing-equation simulator.  Exact trajectory/safety/SNR statistics are
kept separate from empirical linear sensitivity and Fisher diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.observations.compress import build_centres_bank  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def standardized_linear_diagnostic(
    theta_train: np.ndarray,
    y_train: np.ndarray,
    theta_test: np.ndarray,
    y_test: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return standardized beta, held-out R2, and residual variance by output.

    Beta rows correspond to one-standard-deviation changes in theta.  This is
    an empirical global linear surrogate, not an exact simulator Jacobian.
    """
    x_mean = theta_train.mean(axis=0)
    x_std = theta_train.std(axis=0, ddof=1)
    x_std = np.where(x_std > 0, x_std, 1.0)
    xt = (theta_train - x_mean) / x_std
    xv = (theta_test - x_mean) / x_std
    y_mean = y_train.mean(axis=0)
    yc = y_train - y_mean
    beta = np.linalg.solve(xt.T @ xt + ridge * np.eye(xt.shape[1]), xt.T @ yc)
    pred = y_mean + xv @ beta
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    residual_var = np.var(y_train - (y_mean + xt @ beta), axis=0, ddof=1)
    return beta, r2, residual_var


def safe_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ieee9.yaml")
    parser.add_argument("--output", default="reports/ieee9_eig_stage1/experiment1_physics_identifiability")
    parser.add_argument("--n-obs", type=int, default=5)
    parser.add_argument("--noise-sigmas", default="0.0025,0.005,0.01,0.015")
    parser.add_argument("--near-duplicate-corr", type=float, default=0.98)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_dir = (ROOT / cfg["data"]["dataset_dir"]).resolve()
    output = (ROOT / args.output).resolve()
    tables = output / "tables"
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    catalog_path = data_dir / "meta" / "catalog.json"
    bank_meta_path = data_dir / "meta" / "bank.yaml"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    meta = yaml.safe_load(bank_meta_path.read_text(encoding="utf-8"))
    designs = catalog["designs"]
    latent_names = list(meta.get("latent_names") or ["M_1", "M_2", "M_3", "K_1", "K_2", "K_3"])

    def load(split: str):
        split_dir = data_dir / split
        M = np.load(split_dir / "theta_M.npy")
        K = np.load(split_dir / "theta_K.npy")
        theta = np.concatenate([M, K], axis=1).astype(np.float64)
        full = np.load(split_dir / "delta_f.npy", mmap_mode="r")
        rocof = np.load(split_dir / "max_rocof.npy")
        centres, indices, mode = build_centres_bank(full, rocof, int(args.n_obs))
        # build_centres_bank returns action, theta, observation.
        return theta, full, rocof.astype(np.float64), centres, indices, mode

    theta_tr, full_tr, rocof_tr, centres_tr, indices, mode = load("train")
    theta_te, full_te, rocof_te, centres_te, indices_te, mode_te = load("test")
    if not np.array_equal(indices, indices_te) or mode != mode_te:
        raise RuntimeError("train/test observation compression mismatch")
    if len(designs) != centres_tr.shape[0]:
        raise RuntimeError("catalog/action count mismatch")

    n_actions, n_train, obs_dim = centres_tr.shape
    n_test = centres_te.shape[1]
    n_sim = int(full_tr.shape[-1])
    ode_dt = float(meta["ode_dt"])
    # CUDA stores y after each RK4 step, so array index s represents (s+1)dt.
    sample_times = (indices.astype(np.float64) + 1.0) * ode_dt
    sigmas = [float(x) for x in args.noise_sigmas.split(",") if x.strip()]
    reference_sigma = float(cfg["observation"]["eig_based"]["noise_sigma"])
    control_cfg = cfg.get("control") or {}
    provisional_df_limit = abs(float(control_cfg.get("delta_f_nadir_hz", -0.2)))
    provisional_rocof_limit = float(control_cfg.get("rocof_limit_hz_s", 22.0))

    all_centres = np.concatenate([centres_tr, centres_te], axis=1)
    obs_variance_by_dim = np.var(centres_tr, axis=1, ddof=1)
    # A dimension is practically uninformative when even its largest
    # across-theta variance is <1e-8 of the configured noise variance.
    max_signal_to_noise_variance = np.max(obs_variance_by_dim, axis=0) / (reference_sigma**2)
    uninformative_dims = max_signal_to_noise_variance < 1.0e-8

    action_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    betas: list[np.ndarray] = []
    fims: list[np.ndarray] = []
    heldout_r2: list[float] = []

    for action_id, design in enumerate(designs):
        amp, bus, duration = map(float, design)
        ytr = centres_tr[action_id]
        yte = centres_te[action_id]
        beta, r2, residual_var = standardized_linear_diagnostic(
            theta_tr, ytr, theta_te, yte, float(args.ridge)
        )
        betas.append(beta)
        heldout_r2.append(r2)
        fim = beta @ beta.T / (reference_sigma**2)
        fims.append(fim)
        sensitivity = np.linalg.norm(beta, axis=1) / np.sqrt(max(obs_dim, 1))
        for name, value in zip(latent_names, sensitivity):
            sensitivity_rows.append(
                {
                    "action_id": action_id,
                    "amplitude_pu": amp,
                    "physical_bus": int(bus) + 1,
                    "duration_s": duration,
                    "parameter": name,
                    "standardized_linear_sensitivity_hz": float(value),
                }
            )

        param_signal_var = float(np.mean(np.var(ytr, axis=0, ddof=1)))
        row = {
            "action_id": action_id,
            "amplitude_pu": amp,
            "physical_bus": int(bus) + 1,
            "duration_s": duration,
            "train_parameter_signal_sd_hz": float(np.sqrt(param_signal_var)),
            "test_parameter_signal_sd_hz": float(np.sqrt(np.mean(np.var(yte, axis=0, ddof=1)))),
            "heldout_linear_surrogate_r2": safe_float(r2),
            "train_max_abs_delta_f_hz": float(np.max(np.abs(full_tr[:, action_id, :]))),
            "test_max_abs_delta_f_hz": float(np.max(np.abs(full_te[:, action_id, :]))),
            "train_max_rocof_hz_s": float(np.max(rocof_tr[:, action_id])),
            "test_max_rocof_hz_s": float(np.max(rocof_te[:, action_id])),
            "provisional_test_df_safe_rate": float(np.mean(np.max(np.abs(full_te[:, action_id, :]), axis=1) <= provisional_df_limit)),
            "provisional_test_rocof_safe_rate": float(np.mean(rocof_te[:, action_id] <= provisional_rocof_limit)),
            "residual_sd_hz": float(np.sqrt(np.mean(residual_var))),
        }
        for sigma in sigmas:
            row[f"effective_snr_sigma_{sigma:g}"] = param_signal_var / (sigma**2)
        action_rows.append(row)

    # Proper Pearson correlations of centered, flattened response signatures.
    signatures = all_centres.reshape(n_actions, -1)
    signatures = signatures - signatures.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(signatures, axis=1)
    denominator = norms[:, None] * norms[None, :]
    similarity = np.divide(signatures @ signatures.T, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    similarity = np.clip(similarity, -1.0, 1.0)
    pair_rows: list[dict] = []
    near_count = 0
    for i in range(n_actions):
        for j in range(i + 1, n_actions):
            corr = float(similarity[i, j])
            near = abs(corr) >= float(args.near_duplicate_corr)
            near_count += int(near)
            di, dj = designs[i], designs[j]
            pair_rows.append(
                {
                    "action_i": i,
                    "action_j": j,
                    "bus_i": int(di[1]) + 1,
                    "bus_j": int(dj[1]) + 1,
                    "duration_i_s": float(di[2]),
                    "duration_j_s": float(dj[2]),
                    "pearson_response_correlation": corr,
                    "near_duplicate": near,
                }
            )

    combined_fim = np.sum(np.stack(fims), axis=0)
    eigvals, eigvecs = np.linalg.eigh(combined_fim)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    positive = eigvals[eigvals > max(float(eigvals[0]), 1.0) * 1.0e-12]
    condition = float(positive[0] / positive[-1]) if positive.size else float("inf")
    weakest_vector = eigvecs[:, -1]

    write_csv(tables / "action_metrics.csv", action_rows)
    write_csv(tables / "parameter_sensitivity.csv", sensitivity_rows)
    write_csv(tables / "action_pair_similarity.csv", pair_rows)
    np.save(tables / "action_similarity.npy", similarity)
    np.save(tables / "combined_standardized_fisher.npy", combined_fim)

    summary = {
        "experiment": "IEEE9 EIG Stage-1 Experiment 1: physics and identifiability",
        "scope": "read-only offline-bank diagnostic; no BOED methods and no simulator calls",
        "exact_vs_surrogate": {
            "exact": "trajectory extrema, recorded-PMU provisional safety, observation variance, SNR, and action response correlation",
            "surrogate": "standardized global linear theta-to-observation sensitivities and Fisher matrix",
        },
        "bank": {
            "path": str(data_dir.relative_to(ROOT)),
            "train_theta": n_train,
            "test_theta": n_test,
            "latent_names": latent_names,
            "n_actions": n_actions,
            "n_sim": n_sim,
            "amplitudes": sorted({float(x[0]) for x in designs}),
            "physical_buses": sorted({int(x[1]) + 1 for x in designs}),
            "durations_s": sorted({float(x[2]) for x in designs}),
            "observation_bus_physical": int(meta["observation_bus_physical"]),
            "catalog_sha256": digest(catalog_path),
            "bank_metadata_sha256": digest(bank_meta_path),
        },
        "observation": {
            "mode": mode,
            "N_obs": int(args.n_obs),
            "indices": indices.tolist(),
            "sample_times_s": sample_times.tolist(),
            "reference_sigma_hz": reference_sigma,
            "audited_sigmas_hz": sigmas,
            "uninformative_dimensions_across_all_actions": np.flatnonzero(uninformative_dims).tolist(),
            "max_signal_to_noise_variance_ratio_by_dimension": max_signal_to_noise_variance.tolist(),
            "practically_uninformative_threshold": "max_theta_variance / sigma^2 < 1e-8",
            "effective_informative_dimension_upper_bound": int(obs_dim - uninformative_dims.sum()),
        },
        "physics": {
            "provisional_limits_note": "The project control limits are reused only as a screening reference; this is not a certified probing-safety standard.",
            "provisional_abs_delta_f_limit_hz": provisional_df_limit,
            "provisional_rocof_limit_hz_s": provisional_rocof_limit,
            "global_train_max_abs_delta_f_hz": float(np.max(np.abs(full_tr))),
            "global_test_max_abs_delta_f_hz": float(np.max(np.abs(full_te))),
            "global_train_max_rocof_hz_s": float(np.max(rocof_tr)),
            "global_test_max_rocof_hz_s": float(np.max(rocof_te)),
            "all_test_actions_pass_provisional_recorded_pmu_limits": bool(
                all(r["provisional_test_df_safe_rate"] == 1.0 and r["provisional_test_rocof_safe_rate"] == 1.0 for r in action_rows)
            ),
            "spatial_limit": "delta_f and max_rocof are recorded only at the fixed PMU; system-wide probing safety cannot be certified from this bank.",
        },
        "redundancy": {
            "correlation_threshold": float(args.near_duplicate_corr),
            "n_pairs": len(pair_rows),
            "n_near_duplicate_pairs": near_count,
            "near_duplicate_fraction": near_count / len(pair_rows),
            "mean_absolute_pair_correlation": float(np.mean(np.abs(similarity[np.triu_indices(n_actions, 1)]))),
            "max_absolute_pair_correlation": float(np.max(np.abs(similarity[np.triu_indices(n_actions, 1)]))),
        },
        "linear_identifiability": {
            "reference_sigma_hz": reference_sigma,
            "combined_fisher_eigenvalues_desc": eigvals.tolist(),
            "positive_eigenvalue_condition_number": safe_float(condition),
            "weakest_standardized_direction": {name: float(value) for name, value in zip(latent_names, weakest_vector)},
            "heldout_surrogate_r2_mean": safe_float(np.nanmean(heldout_r2)),
            "heldout_surrogate_r2_min": safe_float(np.nanmin(heldout_r2)),
            "warning": "These are empirical global-linear diagnostics, not exact simulator Jacobians or exact EIG.",
        },
    }

    # Plots are optional presentation artifacts; numeric tables remain canonical.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sens_matrix = np.asarray([[r["standardized_linear_sensitivity_hz"] for r in sensitivity_rows if r["action_id"] == a] for a in range(n_actions)])
    fig, ax = plt.subplots(figsize=(8, 11))
    im = ax.imshow(sens_matrix, aspect="auto", cmap="viridis")
    for boundary in range(9, n_actions, 9):
        ax.axhline(boundary - 0.5, color="white", linewidth=0.7, alpha=0.8)
    ax.set_xticks(range(len(latent_names)), latent_names)
    ax.set_ylabel("Action ID (six durations × nine buses)")
    ax.set_title("Empirical standardized parameter sensitivity")
    fig.colorbar(im, ax=ax, label="Hz per 1-SD parameter change")
    fig.tight_layout(); fig.savefig(figures / "parameter_sensitivity_heatmap.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(similarity, vmin=-1, vmax=1, cmap="coolwarm")
    for boundary in range(9, n_actions, 9):
        ax.axhline(boundary - 0.5, color="black", linewidth=0.45, alpha=0.5)
        ax.axvline(boundary - 0.5, color="black", linewidth=0.45, alpha=0.5)
    ax.set_xlabel("Action ID"); ax.set_ylabel("Action ID")
    ax.set_title("Clean response-signature correlation")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout(); fig.savefig(figures / "action_similarity_heatmap.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(range(1, len(eigvals) + 1), np.maximum(eigvals, np.finfo(float).tiny), marker="o")
    ax.set_xlabel("Eigenvalue rank"); ax.set_ylabel("Standardized Fisher eigenvalue")
    ax.set_title("Combined empirical Fisher spectrum")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(figures / "combined_fisher_spectrum.png", dpi=180); plt.close(fig)

    action_signal = np.asarray([r["train_parameter_signal_sd_hz"] for r in action_rows])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(np.arange(n_actions), action_signal)
    colors = plt.cm.plasma(np.linspace(0.12, 0.88, len(sigmas)))
    for sigma, color in zip(sigmas, colors):
        ax.axhline(sigma, color=color, linestyle="--", linewidth=1.4, label=f"sigma={sigma:g} Hz")
    for boundary in range(9, n_actions, 9):
        ax.axvline(boundary - 0.5, color="0.5", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Action ID"); ax.set_ylabel("Across-theta signal SD (Hz)")
    ax.set_title("Decision-independent parameter signal versus noise")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "action_signal_vs_noise.png", dpi=180); plt.close(fig)

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    weakest = ", ".join(f"{k}={v:+.3f}" for k, v in summary["linear_identifiability"]["weakest_standardized_direction"].items())
    best_snr = sorted(action_rows, key=lambda r: r[f"effective_snr_sigma_{reference_sigma:g}"], reverse=True)[:5]
    report = f"""# IEEE9 EIG Stage-1 Experiment 1

## Scope

This is a read-only audit of the existing IEEE9 EIG bank. No DAD, RL-sBOED, Myopic, Fixed, Random, or swing-equation simulation was run. Exact bank statistics are separated from empirical global-linear sensitivity/Fisher diagnostics.

## Bank and observation

- Theta: `{n_train}` train and `{n_test}` held-out samples of `{', '.join(latent_names)}`.
- Actions: `{n_actions}` = 9 physical buses × 6 durations at fixed amplitude `{designs[0][0]}` p.u.
- PMU: physical bus `{meta['observation_bus_physical']}` only.
- Method-visible observation: `N_obs={args.n_obs}`, indices `{indices.tolist()}`, times `{[round(x, 6) for x in sample_times.tolist()]}` s.
- Reference EIG noise: `sigma={reference_sigma}` Hz.

## Main findings

1. **The first sampled value is practically uninformative.** Dimensions `{summary['observation']['uninformative_dimensions_across_all_actions']}` satisfy `max across-theta variance / sigma^2 < 1e-8` for every action. Thus `N_obs={args.n_obs}` has at most `{summary['observation']['effective_informative_dimension_upper_bound']}` informative scalar values. CUDA stores after each RK4 step, so index 0 is time `dt`, not exact equilibrium, but its signal is negligible relative to noise.
2. **Action redundancy is high.** `{near_count}/{len(pair_rows)}` action pairs (`{100*near_count/len(pair_rows):.1f}%`) have `|correlation| >= {args.near_duplicate_corr}`; mean absolute pair correlation is `{summary['redundancy']['mean_absolute_pair_correlation']:.3f}`.
3. **The current bank is spatially limited.** Every trajectory is observed at one fixed PMU, so this bank cannot establish spatial observability or certify system-wide probe safety.
4. **Recorded-PMU probe screen:** held-out maximum `|delta_f|={summary['physics']['global_test_max_abs_delta_f_hz']:.6f}` Hz and maximum RoCoF `{summary['physics']['global_test_max_rocof_hz_s']:.6f}` Hz/s. The comparison limits are provisional control limits, not a probing standard.
5. **Linear identifiability is uneven.** Combined standardized Fisher eigenvalues are `{[float(f'{x:.6g}') for x in eigvals]}` with condition number `{summary['linear_identifiability']['positive_eigenvalue_condition_number']}`. The weakest direction is `{weakest}`.
6. **Surrogate caution:** held-out global-linear R2 averages `{summary['linear_identifiability']['heldout_surrogate_r2_mean']:.3f}` (minimum `{summary['linear_identifiability']['heldout_surrogate_r2_min']:.3f}`). Sensitivity and Fisher results are diagnostics, not exact derivatives or EIG estimates.

## Highest parameter-signal actions at sigma={reference_sigma} Hz

| Action | Bus | Duration (s) | Signal SD (Hz) | Effective SNR |
|---:|---:|---:|---:|---:|
"""
    for row in best_snr:
        report += f"| {row['action_id']} | {row['physical_bus']} | {row['duration_s']} | {row['train_parameter_signal_sd_hz']:.6g} | {row[f'effective_snr_sigma_{reference_sigma:g}']:.4f} |\n"
    report += """

## Interpretation

The present bank is sufficient for a reproducible single-PMU reference experiment, but it is not yet sufficient to claim that all six regional parameters are spatially identifiable or that the 54 actions provide distinct information. The next Stage-1 experiment should correct the sampling schedule so it does not spend one of five observations on the negligible near-equilibrium response at the first RK4 step, then compare the current durations with separated durations and one PMU with three generator-bus PMUs.

## Files

- `summary.json`: machine-readable conclusions and provenance.
- `tables/action_metrics.csv`: exact action safety/SNR plus held-out surrogate fit.
- `tables/parameter_sensitivity.csv`: empirical standardized sensitivities.
- `tables/action_pair_similarity.csv`: all 1,431 action-pair correlations.
- `tables/combined_standardized_fisher.npy`: combined empirical Fisher matrix.
- `figures/`: sensitivity, similarity, Fisher-spectrum, and SNR plots.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"REPORT={output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
