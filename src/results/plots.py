"""Application-agnostic T-sweep plot bundles (EIG and MOCU).

Written only by ``sweep_run.sh``. Works for any config stem in the result
folder name (ieee9, ieee14, sir_ode, …) and both ``eig_based`` (terminal EIG)
and ``objective_based`` (MOCU). Do not put a copy under ``objectives/eig`` or
``objectives/mocu`` — those packages are optimization goals, not plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.config import repo_root
from src.layout import (
    horizon_token,
    load_run_config_doc,
    make_plots_dir_name,
    parse_result_dir_name,
    run_methods_from_doc,
)

POSTER_METHOD_ORDER = (
    "DAD",
    "RL-SBOED",
    "Myopic",
    "Fixed",
    "Random",
    "MoE-sBOED",
    "MatchedDense",
)

_METHOD_LABELS = {
    "dad": "DAD",
    "dad_eig": "DAD",
    "DAD": "DAD",
    "rl_sboed": "RL-SBOED",
    "rl_sboed_eig": "RL-SBOED",
    "RL-sBOED": "RL-SBOED",
    "RL-SBOED": "RL-SBOED",
    "myopic": "Myopic",
    "myopic_delta_h": "Myopic",
    "Myopic": "Myopic",
    "fixed": "Fixed",
    "fixed_open_loop": "Fixed",
    "Fixed": "Fixed",
    "random": "Random",
    "Random": "Random",
    "moe_sboed": "MoE-sBOED",
    "MoE-sBOED": "MoE-sBOED",
    "matched_dense": "MatchedDense",
    "MatchedDense": "MatchedDense",
    "Oracle": "Oracle",
}

_BUNDLE_FILES = ("metric.md", "time.md", "metric_vs_T.png", "time_vs_T.md", "meta.json")


def _poster_label(method: Any) -> str | None:
    raw = str(method).strip()
    if not raw or raw.lower() in {"oracle"}:
        return None
    if raw in _METHOD_LABELS:
        return _METHOD_LABELS[raw]
    key = raw.replace("-", "_").lower()
    return _METHOD_LABELS.get(key, raw)


def _fmt_pm(mean: float, std: float | None, n: int) -> str:
    if n < 2 or std is None or not math.isfinite(std):
        return f"{mean:.4f} ± nan"
    return f"{mean:.4f} ± {std:.4f}"


def _sample_std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)


def _offline_seconds(row: dict[str, Any]) -> float:
    for key in (
        "training_time_seconds",
        "offline_training_or_calibration_seconds",
        "train_s",
    ):
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def _online_seconds(row: dict[str, Any]) -> float:
    for key in ("online_seconds_per_rollout", "seconds_per_rollout"):
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def _metric_value(row: dict[str, Any], *, eig: bool) -> float | None:
    keys = (
        ("terminal_eig_mean", "mean_eig", "ΔH")
        if eig
        else ("mean_mocu", "mean_gap")
    )
    for key in keys:
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _load_summary_rows(exp_dir: Path, *, eig: bool) -> list[dict[str, Any]]:
    eval_dir = exp_dir / "eval"
    path = (
        eval_dir / "terminal_eig_summary.csv"
        if eig
        else eval_dir / "summary.csv"
    )
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rel_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def group_completed_runs(
    exp_dirs: list[Path],
    *,
    want_type: str | None = None,
    want_seeds: set[int] | None = None,
) -> dict[tuple[str, str, int, str], dict[int, dict[int, Path]]]:
    """Group result dirs by (config, experiment_type, N_obs, sigma_token)."""
    grouped: dict[tuple[str, str, int, str], dict[int, dict[int, tuple[str, Path]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    type_filter = (
        None
        if want_type is None
        else str(want_type).strip().lower().replace("-", "_")
    )
    skipped_no_summary = 0
    skipped_no_seed = 0
    for raw in exp_dirs:
        path = Path(raw)
        if not path.is_dir():
            continue
        parsed = parse_result_dir_name(path.name)
        if parsed is None:
            continue
        etype = str(parsed["experiment_type"])
        if type_filter is not None and etype != type_filter:
            continue
        eig = etype == "eig_based"
        rows = _load_summary_rows(path, eig=eig)
        if not rows:
            skipped_no_summary += 1
            continue
        doc = load_run_config_doc(path)
        seed = doc.get("seed")
        if seed is None or str(seed).strip() == "":
            skipped_no_seed += 1
            continue
        seed_i = int(seed)
        if want_seeds is not None and seed_i not in want_seeds:
            continue
        key = (
            str(parsed["config"]),
            etype,
            int(parsed["N_obs"]),
            str(parsed["sigma_token"]),
        )
        t = int(parsed["T"])
        stamp = str(parsed["stamp"])
        prev = grouped[key][t].get(seed_i)
        if prev is None or stamp > prev[0]:
            grouped[key][t][seed_i] = (stamp, path)
    if skipped_no_summary or skipped_no_seed:
        print(
            f"[plots] skipped incomplete dirs: "
            f"{skipped_no_summary} no eval summary, {skipped_no_seed} no run_config seed"
        )
    out: dict[tuple[str, str, int, str], dict[int, dict[int, Path]]] = {}
    for key, by_t in grouped.items():
        out[key] = {
            t: {seed: pair[1] for seed, pair in seeds.items()}
            for t, seeds in by_t.items()
        }
    return out


def discover_seed_runs(
    project_root: Path,
    *,
    config_stems: list[str],
    experiment_type: str,
    horizons: list[int],
    n_obs_values: list[int],
    noise_sigmas: list[float],
    seeds: list[int] | None = None,
) -> list[Path]:
    """Newest completed result dir per (config, T, N_obs, sigma, seed)."""
    experiments = project_root / "experiments"
    if not experiments.is_dir():
        return []
    want_type = str(experiment_type).strip().lower().replace("-", "_")
    want_configs = {str(c).strip() for c in config_stems if str(c).strip()}
    want_t = {int(t) for t in horizons}
    want_nobs = {int(n) for n in n_obs_values}
    want_sigma = [float(s) for s in noise_sigmas]
    want_seeds = {int(s) for s in seeds} if seeds else None
    found: list[Path] = []
    for path in sorted(experiments.iterdir()):
        if not path.is_dir():
            continue
        parsed = parse_result_dir_name(path.name)
        if parsed is None:
            continue
        if parsed["config"] not in want_configs:
            continue
        if parsed["experiment_type"] != want_type:
            continue
        if int(parsed["T"]) not in want_t:
            continue
        if int(parsed["N_obs"]) not in want_nobs:
            continue
        if not any(
            math.isclose(float(parsed["noise_sigma"]), sigma, rel_tol=0.0, abs_tol=1e-12)
            for sigma in want_sigma
        ):
            continue
        found.append(path)
    grouped = group_completed_runs(
        found,
        want_type=want_type,
        want_seeds=want_seeds,
    )
    out: list[Path] = []
    for by_t in grouped.values():
        for seeds_map in by_t.values():
            out.extend(seeds_map.values())
    return out


def find_seed_runs(
    project_root: Path,
    *,
    run_prefix: str,
    experiment_type: str,
    n_obs: int | None = None,
    noise_sigma: float | None = None,
) -> dict[tuple[str, int, str], dict[int, dict[int, Path]]]:
    experiments = project_root / "experiments"
    if not experiments.is_dir():
        return {}
    paths = [p for p in experiments.iterdir() if p.is_dir()]
    grouped = group_completed_runs(paths, want_type=experiment_type)
    out: dict[tuple[str, int, str], dict[int, dict[int, Path]]] = {}
    for (config, etype, nobs, sigma_token), by_t in grouped.items():
        if config != run_prefix or etype != experiment_type:
            continue
        if n_obs is not None and int(nobs) != int(n_obs):
            continue
        if noise_sigma is not None and not math.isclose(
            float(sigma_token.replace("p", ".")),
            float(noise_sigma),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            continue
        out[(config, nobs, sigma_token)] = by_t
    return out


def _ordered_methods(names: set[str]) -> list[str]:
    ordered = [m for m in POSTER_METHOD_ORDER if m in names]
    extra = sorted(names - set(POSTER_METHOD_ORDER))
    return ordered + extra


def _collect_cells(
    by_t: dict[int, dict[int, Path]],
    *,
    eig: bool,
) -> tuple[
    dict[str, dict[int, list[float]]],
    dict[str, dict[int, list[float]]],
    dict[str, dict[int, list[float]]],
]:
    metric: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    offline: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    online: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t, seeds in by_t.items():
        for _seed, exp_dir in seeds.items():
            for row in _load_summary_rows(exp_dir, eig=eig):
                label = _poster_label(row.get("method") or row.get("Method"))
                if label is None:
                    continue
                value = _metric_value(row, eig=eig)
                if value is None:
                    continue
                metric[label][t].append(value)
                offline[label][t].append(_offline_seconds(row))
                online[label][t].append(_online_seconds(row))
    return metric, offline, online


_EIG_PAIRED_COMPARISONS = (
    ("dad_eig", "fixed_open_loop"),
    ("dad_eig", "myopic_delta_h"),
    ("dad_eig", "random"),
    ("rl_sboed_eig", "fixed_open_loop"),
    ("rl_sboed_eig", "myopic_delta_h"),
    ("rl_sboed_eig", "random"),
)


def _paired_eig_differences(exp_dir: Path) -> dict[str, list[float]]:
    """Recover per-theta paired differences from the compact rollout artifact."""
    path = exp_dir / "eval" / "vector_eig_results.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in raw.get("rollouts", []):
        method = str(row.get("method", ""))
        if not method or row.get("theta_id") is None:
            continue
        grouped[method][int(row["theta_id"])].append(float(row["terminal_eig"]))
    output: dict[str, list[float]] = {}
    for left, right in _EIG_PAIRED_COMPARISONS:
        if left not in grouped or right not in grouped:
            continue
        theta_ids = sorted(set(grouped[left]) & set(grouped[right]))
        if not theta_ids:
            continue
        output[f"{left} - {right}"] = [
            sum(grouped[left][theta]) / len(grouped[left][theta])
            - sum(grouped[right][theta]) / len(grouped[right][theta])
            for theta in theta_ids
        ]
    return output


def _hierarchical_ci(
    seed_samples: list[list[float]], *, bootstrap_seed: int, n_bootstrap: int
) -> tuple[float, float, float, int]:
    """Seed-then-system bootstrap; avoids treating repeated θ banks as IID seeds."""
    clean = [values for values in seed_samples if values]
    if not clean:
        return float("nan"), float("nan"), float("nan"), 0
    seed_means = [sum(values) / len(values) for values in clean]
    estimate = sum(seed_means) / len(seed_means)
    rng = random.Random(int(bootstrap_seed))
    draws: list[float] = []
    for _ in range(max(1, int(n_bootstrap))):
        sampled_seeds = [clean[rng.randrange(len(clean))] for _ in clean]
        sampled_means = []
        for values in sampled_seeds:
            sampled = [values[rng.randrange(len(values))] for _ in values]
            sampled_means.append(sum(sampled) / len(sampled))
        draws.append(sum(sampled_means) / len(sampled_means))
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return estimate, lo, hi, len(clean)


def _collect_hierarchical_eig_pairs(
    by_t: dict[int, dict[int, Path]], *, n_bootstrap: int = 5000
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t, seed_dirs in sorted(by_t.items()):
        samples: dict[str, list[list[float]]] = defaultdict(list)
        for _seed, exp_dir in sorted(seed_dirs.items()):
            for comparison, values in _paired_eig_differences(exp_dir).items():
                samples[comparison].append(values)
        for index, (comparison, seed_samples) in enumerate(sorted(samples.items())):
            mean, low, high, n_seeds = _hierarchical_ci(
                seed_samples,
                bootstrap_seed=7919 + 101 * int(t) + index,
                n_bootstrap=n_bootstrap,
            )
            rows.append(
                {
                    "T": int(t),
                    "comparison": comparison,
                    "mean_diff": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_seeds": n_seeds,
                    "n_systems_per_seed": [len(values) for values in seed_samples],
                    "bootstrap_replicates": int(n_bootstrap),
                    "method": "hierarchical seed-then-system bootstrap",
                }
            )
    return rows


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    path.write_text(text, encoding="utf-8")


def _mean_cell(values: list[float], *, digits: int) -> str:
    if not values:
        return "—"
    return f"{sum(values) / len(values):.{digits}f}"


def _plot_metric_errorbars(
    metric: dict[str, dict[int, list[float]]],
    *,
    methods: list[str],
    horizons: list[int],
    out_path: Path,
    title: str,
    ylabel: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for method in methods:
        xs: list[int] = []
        ys: list[float] = []
        yerr: list[float] = []
        for t in horizons:
            values = metric.get(method, {}).get(t, [])
            if not values:
                continue
            xs.append(t)
            ys.append(sum(values) / len(values))
            std = _sample_std(values)
            yerr.append(0.0 if not math.isfinite(std) else std)
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=method)
    ax.set_xlabel("Horizon T")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _unique_seeds(by_t: dict[int, dict[int, Path]]) -> list[int]:
    seeds = {seed for seeds in by_t.values() for seed in seeds}
    return sorted(seeds)


def _canonical_methods(by_t: dict[int, dict[int, Path]], extra: list[str] | None) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for token in extra or []:
        key = str(token).strip()
        if key and key not in seen:
            seen.add(key)
            names.append(key)
    for seeds in by_t.values():
        for exp_dir in seeds.values():
            for method in run_methods_from_doc(load_run_config_doc(exp_dir)):
                if method not in seen:
                    seen.add(method)
                    names.append(method)
    return names


def write_sweep_plot_bundle(
    by_t: dict[int, dict[int, Path]],
    *,
    out_dir: Path,
    stamp: str,
    config: str,
    experiment_type: str,
    n_obs: int,
    sigma_token: str,
    methods_cli: list[str] | None = None,
    root: Path | None = None,
    sweep_horizons: list[int] | None = None,
) -> Path:
    """Write exactly five files into a stamped sweep plots folder."""
    root = root or repo_root()
    eig = experiment_type == "eig_based"
    metric, offline, online = _collect_cells(by_t, eig=eig)
    paired_eig = _collect_hierarchical_eig_pairs(by_t) if eig else []
    methods = _ordered_methods(set(metric))
    horizons = sorted(by_t)
    n_seeds = len(_unique_seeds(by_t))
    metric_name = "terminal EIG" if eig else "MOCU"
    better = "higher better" if eig else "lower better"
    title_metric = f"Mean {metric_name} ± std (n = {n_seeds} seeds) ({better})"
    title_time = f"Mean time consumption (n = {n_seeds} seeds)"

    headers_metric = ["Method"] + [f"T={t}" for t in horizons]
    rows_metric: list[list[str]] = []
    for method in methods:
        row = [method]
        for t in horizons:
            values = metric.get(method, {}).get(t, [])
            n = len(values)
            if not n:
                row.append("—")
                continue
            mean = sum(values) / n
            row.append(_fmt_pm(mean, _sample_std(values), n))
        rows_metric.append(row)

    headers_time = ["Method"]
    for t in horizons:
        headers_time.extend([f"T={t} Offline (s)", f"T={t} Online (s/rollout)"])
    rows_time: list[list[str]] = []
    for method in methods:
        row = [method]
        for t in horizons:
            row.append(_mean_cell(offline.get(method, {}).get(t, []), digits=2))
            row.append(_mean_cell(online.get(method, {}).get(t, []), digits=6))
        rows_time.append(row)

    headers_vs = ["T"] + methods
    rows_off: list[list[str]] = []
    rows_on: list[list[str]] = []
    for t in horizons:
        off_row = [str(t)]
        on_row = [str(t)]
        for method in methods:
            off_row.append(_mean_cell(offline.get(method, {}).get(t, []), digits=2))
            on_row.append(_mean_cell(online.get(method, {}).get(t, []), digits=6))
        rows_off.append(off_row)
        rows_on.append(on_row)

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    metric_path = dest / "metric.md"
    time_path = dest / "time.md"
    fig_path = dest / "metric_vs_T.png"
    time_vs_path = dest / "time_vs_T.md"
    meta_path = dest / "meta.json"

    metric_lines = [f"# {title_metric}", ""] + _md_table(headers_metric, rows_metric)
    if eig:
        metric_lines.extend(
            [
                "",
                "Random is averaged over 32 randomized design sequences per held-out "
                "system; deterministic methods use one sequence per system.",
                "",
                "## Paired EIG improvements",
                "",
                "Intervals use a hierarchical seed-then-system bootstrap. Positive "
                "differences favor the method on the left.",
                "",
            ]
        )
        paired_table_rows = [
            [
                str(row["T"]),
                str(row["comparison"]),
                f'{float(row["mean_diff"]):.4f}',
                f'[{float(row["ci95_low"]):.4f}, {float(row["ci95_high"]):.4f}]',
                str(row["n_seeds"]),
            ]
            for row in paired_eig
        ]
        metric_lines.extend(
            _md_table(
                ["T", "Comparison", "Mean difference", "Hierarchical 95% CI", "Seeds"],
                paired_table_rows,
            )
        )
    metric_lines.append("")
    _write_text(metric_path, metric_lines)
    _write_text(
        time_path,
        [f"# {title_time}", ""] + _md_table(headers_time, rows_time) + [""],
    )
    _write_text(
        time_vs_path,
        [
            f"# Mean time vs T (n = {n_seeds} seeds)",
            "",
            "## Offline (s)",
            "",
            *_md_table(headers_vs, rows_off),
            "",
            "## Online (s / rollout)",
            "",
            *_md_table(headers_vs, rows_on),
            "",
        ],
    )
    _plot_metric_errorbars(
        metric,
        methods=methods,
        horizons=horizons,
        out_path=fig_path,
        title=title_metric,
        ylabel=f"Mean {metric_name}",
    )

    runs: list[dict[str, Any]] = []
    for t in horizons:
        for seed, exp_dir in sorted(by_t[t].items()):
            parsed = parse_result_dir_name(exp_dir.name) or {}
            runs.append(
                {
                    "dir": _rel_to_root(exp_dir, root),
                    "T": int(t),
                    "seed": int(seed),
                    "stamp": parsed.get("stamp"),
                }
            )
    meta = {
        "stamp": stamp,
        "folder": dest.name,
        "experiment_type": experiment_type,
        "config": config,
        "T": list(sweep_horizons) if sweep_horizons else horizons,
        "T_completed": horizons,
        "T_token": horizon_token(sweep_horizons if sweep_horizons else horizons),
        "N_obs": int(n_obs),
        "noise_sigma": float(str(sigma_token).replace("p", ".")),
        "sigma_token": sigma_token,
        "seeds": _unique_seeds(by_t),
        "methods": _canonical_methods(by_t, methods_cli),
        "n_seeds": n_seeds,
        "n_runs": len(runs),
        "runs": runs,
        "run_selection": (
            "newest completed result per config/T/N_obs/noise_sigma/seed; exact "
            "selected folders are listed in runs"
        ),
        "paired_eig": paired_eig if eig else None,
        "random_evaluation": (
            "32 randomized designs per held-out system, averaged before method "
            "comparison"
            if eig
            else None
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    allowed = set(_BUNDLE_FILES)
    for child in list(dest.iterdir()):
        if child.name in allowed:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    missing = [name for name in _BUNDLE_FILES if not (dest / name).is_file()]
    if missing:
        raise RuntimeError(f"plots folder missing files {missing}: {dest}")
    extra = [p.name for p in dest.iterdir() if p.name not in allowed]
    if extra:
        raise RuntimeError(f"plots folder has extra files {extra}: {dest}")
    return dest


def write_sweep_plot_folders(
    exp_dirs: list[Path],
    *,
    stamp: str,
    experiment_type: str | None = None,
    methods_cli: list[str] | None = None,
    project_root: Path | None = None,
    sweep_horizons: list[int] | None = None,
) -> list[Path]:
    root = project_root or repo_root()
    grouped = group_completed_runs(exp_dirs, want_type=experiment_type)
    if not grouped:
        print("[plots] no completed result dirs to aggregate")
        return []
    written: list[Path] = []
    for (config, etype, n_obs, sigma_token), by_t in sorted(grouped.items()):
        name_horizons = list(sweep_horizons) if sweep_horizons else sorted(by_t)
        if len(set(int(t) for t in name_horizons)) < 2 or len(by_t) < 2:
            print(
                f"[plots] skip {config} {etype} Nobs{n_obs} sigma{sigma_token}: "
                "plots folders are only for a T sweep (at least two horizons)"
            )
            continue
        name = make_plots_dir_name(
            config,
            etype,
            name_horizons,
            stamp=stamp,
            n_obs=n_obs,
            sigma_token=sigma_token,
        )
        dest = root / "experiments" / name
        print(
            f"[plots] {name} T={sorted(by_t)} "
            f"seeds={_unique_seeds(by_t)} n_runs={sum(len(s) for s in by_t.values())}"
        )
        written.append(
            write_sweep_plot_bundle(
                by_t,
                out_dir=dest,
                stamp=stamp,
                config=config,
                experiment_type=etype,
                n_obs=n_obs,
                sigma_token=sigma_token,
                methods_cli=methods_cli,
                root=root,
                sweep_horizons=name_horizons,
            )
        )
        print(f"  → {dest}")
    return written


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write stamped T-sweep plot folders (called from sweep_run.sh). "
            "Works for EIG and MOCU on any application config."
        ),
    )
    parser.add_argument(
        "--stamp",
        required=True,
        help="Sweep start time MMDDYYYY_HHMMSS (from sweep_run.sh).",
    )
    parser.add_argument(
        "--experiment-type",
        default=None,
        choices=("eig_based", "objective_based"),
    )
    parser.add_argument("--exp-dir", action="append", default=[], dest="exp_dirs")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Find existing result dirs matching --config-stem / --T / --N_obs / --noise_sigma / --seeds.",
    )
    parser.add_argument("--config-stem", action="append", default=[], dest="config_stems")
    parser.add_argument("--T", dest="horizons", default=None)
    parser.add_argument("--N_obs", dest="n_obs", default=None)
    parser.add_argument("--noise_sigma", dest="noise_sigma", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--methods", default=None)
    args = parser.parse_args(argv)

    root = repo_root()
    methods = _split_csv(args.methods)
    sweep_horizons = [int(t) for t in _split_csv(args.horizons)]
    exp_dirs = [Path(p) for p in args.exp_dirs]
    if args.discover:
        stems = [s for s in args.config_stems if s]
        if not stems:
            raise SystemExit("--discover requires --config-stem")
        n_obs_values = [int(n) for n in _split_csv(args.n_obs)]
        sigmas = [float(s) for s in _split_csv(args.noise_sigma)]
        seeds = [int(s) for s in _split_csv(args.seeds)] if args.seeds else None
        if not args.experiment_type:
            raise SystemExit("--discover requires --experiment-type")
        if not sweep_horizons or not n_obs_values or not sigmas:
            raise SystemExit("--discover requires --T, --N_obs, and --noise_sigma")
        exp_dirs.extend(
            discover_seed_runs(
                root,
                config_stems=stems,
                experiment_type=args.experiment_type,
                horizons=sweep_horizons,
                n_obs_values=n_obs_values,
                noise_sigmas=sigmas,
                seeds=seeds,
            )
        )
    if not exp_dirs:
        raise SystemExit("no experiment dirs (pass --exp-dir or --discover)")
    write_sweep_plot_folders(
        exp_dirs,
        stamp=args.stamp,
        experiment_type=args.experiment_type,
        methods_cli=methods or None,
        project_root=root,
        sweep_horizons=sweep_horizons or None,
    )


if __name__ == "__main__":
    main()
