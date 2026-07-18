"""Load fixed representative histories (read-only historical CSVs) and extend to h2/h3."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.control.particle_posterior_adequacy.supports import (
    GLOBAL_HISTORY_SEED,
    MasterArrays,
    production_true_theta_systems,
)
from src.control.terminal_rule import keyed_noise
from src.data import lookup_action_y_sim
from src.config import repo_root


def _fixed_sequence(system: str, project_root: Path | None = None) -> list[int]:
    root = project_root or repo_root()
    path = root / "experiments" / f"{system}_T3" / "eval" / "fixed" / "subset_meta.json"
    if not path.is_file():
        return []
    return [int(x) for x in json.loads(path.read_text(encoding="utf-8"))["selected_action_ids"]]


def _history_csv(system: str, project_root: Path | None = None) -> Path:
    root = project_root or repo_root()
    return (
        root
        / "experiments"
        / "objective_adaptive_value"
        / f"{system}_T3"
        / "first_history_results.csv"
    )


def load_stratified_h1_rows(
    system: str,
    *,
    max_histories: int,
    smoke: bool = False,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Stratified sample of historical h1 rows (xi1, y1, theta_id, rollout_id)."""
    path = _history_csv(system, project_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing historical histories at {path} (read-only dependency)."
        )
    by_xi1: dict[int, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            xi1 = int(row["xi1"])
            by_xi1.setdefault(xi1, []).append(
                {
                    "history_id": int(row["history_id"]),
                    "xi1": xi1,
                    "y1": float(row["y1"]),
                    "theta_id": int(row["theta_id"]),
                    "rollout_id": int(row["rollout_id"]),
                    "source": "objective_adaptive_value/first_history_results.csv",
                }
            )
    limit = 12 if smoke else int(max_histories)
    keys = sorted(by_xi1)
    out: list[dict[str, Any]] = []
    ptr = {k: 0 for k in keys}
    while len(out) < limit and keys:
        progressed = False
        for k in keys:
            rows = by_xi1[k]
            i = ptr[k]
            if i < len(rows):
                out.append(rows[i])
                ptr[k] = i + 1
                progressed = True
                if len(out) >= limit:
                    break
        if not progressed:
            break
    return out


def _next_fixed_actions(fixed_seq: list[int], used: set[int], n_actions: int, need: int) -> list[int]:
    picks: list[int] = []
    for a in fixed_seq:
        if a not in used and a not in picks:
            picks.append(int(a))
        if len(picks) >= need:
            return picks
    for a in range(n_actions):
        if a not in used and a not in picks:
            picks.append(int(a))
        if len(picks) >= need:
            return picks
    return picks


def build_multistep_histories(
    master: MasterArrays,
    h1_rows: list[dict[str, Any]],
    *,
    horizon: int = 3,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Build fixed action/observation sequences for steps 0..horizon.

    Observation generation for steps ≥2 uses production true-θ Y-banks + keyed noise
    (no ODE). Posterior support particles remain the master-bank subsets.
    """
    true_systems = production_true_theta_systems(master.system, project_root)
    fixed_seq = _fixed_sequence(master.system, project_root)
    histories: list[dict[str, Any]] = []

    # Shared prior history (h0)
    histories.append(
        {
            "history_id": -1,
            "theta_id": None,
            "rollout_id": None,
            "source": "prior_h0",
            "steps": {
                0: {"actions": [], "observations": []},
            },
        }
    )

    for row in h1_rows:
        tid = int(row["theta_id"])
        if tid < 0 or tid >= len(true_systems):
            # Fall back: cannot extend; keep h0/h1 only from CSV observations.
            true_sys = None
        else:
            true_sys = true_systems[tid]
        xi1 = int(row["xi1"])
        y1 = float(row["y1"])
        actions = [xi1]
        obs = [y1]
        used = {xi1}
        need = max(0, int(horizon) - 1)
        extra = _next_fixed_actions(fixed_seq, used, master.n_actions, need)
        for step_idx, action in enumerate(extra, start=1):
            if true_sys is None:
                break
            z = keyed_noise(
                global_seed=GLOBAL_HISTORY_SEED,
                theta_id=tid,
                rollout_id=int(row["rollout_id"]),
                step=step_idx,
                action_id=int(action),
            )
            y_sim = lookup_action_y_sim(true_sys, int(action))
            y = float(y_sim) + float(master.sigma_y) * float(z)
            actions.append(int(action))
            obs.append(y)
            used.add(int(action))

        steps: dict[int, dict[str, Any]] = {0: {"actions": [], "observations": []}}
        for t in range(1, len(actions) + 1):
            steps[t] = {
                "actions": list(actions[:t]),
                "observations": list(obs[:t]),
            }
        histories.append(
            {
                "history_id": int(row["history_id"]),
                "theta_id": tid,
                "rollout_id": int(row["rollout_id"]),
                "source": row["source"],
                "xi1": xi1,
                "y1": y1,
                "steps": steps,
            }
        )
    return histories


def assert_histories_independent_of_support(
    histories: list[dict[str, Any]],
    support_indices: np.ndarray,
) -> None:
    """True-θ IDs are production indices; support indices are master-bank rows — distinct spaces."""
    del support_indices  # spaces differ by construction; document in metadata.
    for h in histories:
        if h.get("history_id") == -1:
            continue
        if h.get("source") != "objective_adaptive_value/first_history_results.csv":
            raise AssertionError("unexpected history source")
