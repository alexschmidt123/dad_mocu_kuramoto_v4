"""Write a single experiment-root ``summary.md`` comparison table.

Observation mode follows ``N_obs`` (max_rocof if 0, sampled Δf otherwise).
Primary metric: ``mean_mocu`` for objective_based, ``mean_eig`` for eig_based.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from src.observations.compress import observation_mode


def _fmt(value: Any, *, digits: int = 6) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |",
    ]
    for row in rows:
        cells = list(row) + [""] * max(0, len(headers) - len(row))
        lines.append("| " + " | ".join(cells[: len(headers)]) + " |")
    return lines


def _observation_blurb(n_obs: int, mode: str) -> str:
    if int(n_obs) == 0 or mode == "max_rocof":
        return (
            "Observation: scalar max-|ROCOF| (`N_obs=0`). "
            "Methods do not see full Δf trajectories."
        )
    return (
        f"Observation: {int(n_obs)} evenly spaced probe-bus Δf samples "
        f"(`observation_mode={mode}`)."
    )


def write_summary_md(
    exp_dir: Path,
    *,
    system: str,
    experiment_type: str,
    meta: dict[str, Any] | None = None,
    table_headers: Sequence[str],
    table_rows: Sequence[Sequence[str]],
    extra_lines: Sequence[str] | None = None,
) -> Path:
    """Write ``{exp_dir}/summary.md`` (no ``summary/`` folder)."""
    exp_dir = Path(exp_dir)
    meta = dict(meta or {})
    n_obs = int(meta.get("N_obs", meta.get("n_obs", 0)) or 0)
    mode = str(meta.get("observation_mode") or observation_mode(n_obs))
    n_sim = meta.get("N_sim", meta.get("n_sim"))
    step_number = meta.get("T", meta.get("step_number"))

    lines = [
        f"# Summary — {system} ({experiment_type})",
        "",
        _observation_blurb(n_obs, mode),
        "",
        f"- system: `{system}`",
        f"- experiment_type: `{experiment_type}`",
        f"- observation_mode: `{mode}`",
        f"- N_obs: {n_obs}",
    ]
    if n_sim is not None:
        lines.append(f"- N_sim: {n_sim}")
    if step_number is not None:
        lines.append(f"- T: {step_number}")
    if meta.get("config_path"):
        lines.append(f"- config: `{meta['config_path']}`")
    lines += ["", "## Comparison", ""]
    lines.extend(_md_table(table_headers, table_rows))
    if extra_lines:
        lines += ["", *extra_lines]
    lines.append("")

    out = exp_dir / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_objective_summary_md(
    exp_dir: Path,
    *,
    system: str | None = None,
    eval_meta: dict[str, Any] | None = None,
) -> Path:
    """Build objective_based table from ``eval/summary.csv`` (+ Oracle)."""
    exp_dir = Path(exp_dir)
    eval_dir = exp_dir / "eval"
    summary_csv = eval_dir / "summary.csv"
    meta: dict[str, Any] = {}
    meta_path = eval_dir / "eval_meta.json"
    if meta_path.is_file():
        meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
    if eval_meta:
        meta.update(eval_meta)

    system = system or str(meta.get("system") or exp_dir.name)
    if "T" not in meta and "step_number" not in meta:
        from src.layout import parse_result_dir_name

        parsed = parse_result_dir_name(exp_dir.name)
        if parsed and parsed.get("step_number") is not None:
            meta["T"] = parsed["step_number"]

    headers = [
        "Method",
        "mean_MOCU",
        "mean_u_ctrl",
        "safety_rate",
        "valid",
        "n_unique_sequences",
    ]
    parsed_rows: list[dict[str, Any]] = []
    if summary_csv.is_file():
        with summary_csv.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                method = str(row.get("method", ""))
                if method.endswith("_stochastic"):
                    continue
                parsed_rows.append(row)
    rows: list[list[str]] = []
    for row in parsed_rows:
        method = str(row.get("method", ""))
        mean_gap = row.get("mean_gap")
        mean_excess = row.get("mean_excess")
        under = row.get("under_control_rate")
        if mean_excess in (None, "") and mean_gap not in (None, ""):
            # Legacy CSV: approximate excess from raw gap (clamped).
            try:
                mean_excess = max(float(mean_gap), 0.0)
            except (TypeError, ValueError):
                mean_excess = ""
        rows.append(
            [
                method,
                _fmt(row.get("mean_mocu", row.get("mean_gap"))),
                _fmt(row.get("mean_u_ctrl")),
                _fmt(row.get("safety_rate"), digits=3),
                (
                    "VALID"
                    if str(row.get("method", "")) == "Oracle"
                    or str(row.get("valid", "0")) in ("1", "True", "true")
                    else "INVALID"
                ),
                str(row.get("n_unique_sequences") or "—"),
            ]
        )
    if not rows:
        rows.append(["(summary.csv missing)", "—", "—", "—", "—", "—"])

    def _rank_key(row: dict[str, Any]) -> tuple[float]:
        def _f(key: str, default: float) -> float:
            try:
                v = row.get(key)
                if v in (None, ""):
                    return default
                return float(v)
            except (TypeError, ValueError):
                return default

        return (_f("mean_mocu", _f("mean_gap", float("inf"))),)

    ranking = [
        m
        for m, _ in sorted(
            (
                (str(r["method"]), _rank_key(r))
                for r in parsed_rows
                if str(r.get("method", "")) not in ("", "Oracle")
                and str(r.get("valid", "0")) in ("1", "True", "true")
            ),
            key=lambda x: x[1],
        )
    ]
    extra: list[str] = [
        "Notes:",
        "",
        "- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.",
        "- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.",
        "- Methods with safety rate below 0.95 are INVALID and receive no rank.",
        "- Valid methods are ranked by lower mean_MOCU.",
        "- `mean_u_ctrl` remains a secondary physical-control metric.",
        "- Policies use argmax actions only (no `*_stochastic` rows).",
    ]
    if ranking:
        extra.extend(
            [
                "",
                "## Ranking (safety ≥ 0.95; mean_MOCU ↓)",
                "",
                ", ".join(ranking),
            ]
        )

    return write_summary_md(
        exp_dir,
        system=system,
        experiment_type="objective_based",
        meta=meta,
        table_headers=headers,
        table_rows=rows,
        extra_lines=extra or None,
    )


def write_eig_summary_md(
    exp_dir: Path,
    *,
    system: str,
    horizon: int,
    method_results: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> Path:
    """Build eig_based table; primary metric is mean (terminal) EIG."""
    meta = dict(meta or {})
    meta.setdefault("T", horizon)
    meta.setdefault("N_obs", meta.get("n_obs", 0))
    if "observation_mode" not in meta:
        meta["observation_mode"] = observation_mode(int(meta["N_obs"]))

    headers = ["Method", "mean_eig", "mean_eig_step1", "sum_stepwise_eig", "n_rollouts"]
    rows: list[list[str]] = []
    for _key, payload in method_results.items():
        steps = list(payload.get("mean_eig_by_step") or [])
        mean_eig = payload.get("terminal_eig_mean")
        if mean_eig is None and steps:
            mean_eig = float(sum(steps))
        step1 = steps[0] if steps else None
        sum_steps = float(sum(steps)) if steps else None
        rows.append(
            [
                str(payload.get("method_label", _key)),
                _fmt(mean_eig, digits=4),
                _fmt(step1, digits=4),
                _fmt(sum_steps, digits=4),
                str(payload.get("n_rollouts", "—")),
            ]
        )

    return write_summary_md(
        exp_dir,
        system=system,
        experiment_type="eig_based",
        meta=meta,
        table_headers=headers,
        table_rows=rows,
    )
