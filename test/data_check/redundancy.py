"""Pairwise redundancy and complementarity diagnostics (information redundancy)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.table_scoring import TableThetaSupport, y_sim_last_step_from_tables
from test.data_check.oracle_eig import mc_conditional_eig_given_action, mc_particle_eig


@dataclass(frozen=True)
class RedundancyDiagnostics:
    eig_matrix: np.ndarray
    redundancy_matrix: np.ndarray
    response_signatures: np.ndarray
    response_cosine: np.ndarray
    response_correlation: np.ndarray
    top_eig_actions: list[tuple[int, float]]
    top_redundant_pairs: list[tuple[int, int, float]]
    top_complementary_pairs: list[tuple[int, int, float]]


def build_action_centres(
    table_support: TableThetaSupport,
    n_actions: int,
) -> dict[int, np.ndarray]:
    """Clean response centres m_n(a) from banked y_sim for each action."""
    centres: dict[int, np.ndarray] = {}
    for a in range(n_actions):
        centres[a] = y_sim_last_step_from_tables(table_support, [a])
    return centres


def response_signature_matrix(centres: dict[int, np.ndarray], n_actions: int) -> np.ndarray:
    """Shape (A, N): r_a = [y_clean(theta^(1), xi^a), ...]."""
    return np.stack([centres[a] for a in range(n_actions)], axis=0)


def cosine_similarity_matrix(signatures: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(signatures, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    unit = signatures / norms
    return unit @ unit.T


def correlation_matrix(signatures: np.ndarray) -> np.ndarray:
    if signatures.shape[1] < 2:
        return np.eye(signatures.shape[0])
    return np.corrcoef(signatures)


def compute_redundancy_diagnostics(
    table_support: TableThetaSupport,
    n_actions: int,
    sigma_y: float,
    K: int,
    rng: np.random.Generator,
    *,
    action_subset: list[int] | None = None,
) -> RedundancyDiagnostics:
    """Full directional redundancy matrix and rankings."""
    centres = build_action_centres(table_support, n_actions)
    actions = action_subset if action_subset is not None else list(range(n_actions))
    n = len(actions)
    log_p0 = table_support.log_p0

    eig_one_step: dict[int, float] = {}
    for a in actions:
        eig_one_step[a] = mc_particle_eig(log_p0, centres[a], sigma_y, K, rng).eig

    red_mat = np.full((n, n), np.nan, dtype=np.float64)
    for i, a in enumerate(actions):
        red_mat[i, i] = 0.0
        for j, b in enumerate(actions):
            if a == b:
                continue
            cond_eig, _ = mc_conditional_eig_given_action(
                log_p0, centres[a], centres[b], sigma_y, K, rng,
            )
            denom = eig_one_step[b]
            red_mat[i, j] = 0.0 if denom <= 1e-12 else 1.0 - cond_eig / denom

    eig_mat = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(actions):
        eig_mat[i, i] = eig_one_step[a]

    sig = response_signature_matrix(centres, n_actions)
    if action_subset is not None:
        sig = sig[np.asarray(actions)]

    eig_diag = {actions[i]: float(eig_mat[i, i]) for i in range(n)}
    top_eig = sorted(eig_diag.items(), key=lambda x: -x[1])

    pairs_red: list[tuple[int, int, float]] = []
    pairs_comp: list[tuple[int, int, float]] = []
    for i, a in enumerate(actions):
        for j, b in enumerate(actions):
            if a == b:
                continue
            r = float(red_mat[i, j])
            pairs_red.append((a, b, r))
            pairs_comp.append((a, b, 1.0 - r))

    pairs_red.sort(key=lambda x: -x[2])
    pairs_comp.sort(key=lambda x: -x[2])

    return RedundancyDiagnostics(
        eig_matrix=eig_mat,
        redundancy_matrix=red_mat,
        response_signatures=sig,
        response_cosine=cosine_similarity_matrix(sig),
        response_correlation=correlation_matrix(sig),
        top_eig_actions=top_eig[: min(20, len(top_eig))],
        top_redundant_pairs=pairs_red[: min(30, len(pairs_red))],
        top_complementary_pairs=pairs_comp[: min(30, len(pairs_comp))],
    )


def save_redundancy_artifacts(out_dir: Path, diag: RedundancyDiagnostics, meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "redundancy_diagnostics.npz",
        eig_matrix=diag.eig_matrix,
        redundancy_matrix=diag.redundancy_matrix,
        response_signatures=diag.response_signatures,
        response_cosine=diag.response_cosine,
        response_correlation=diag.response_correlation,
    )
    import csv

    from src.data import save_json

    save_json(meta | {"top_eig_actions": diag.top_eig_actions}, out_dir / "redundancy_summary.json")
    with (out_dir / "top_redundant_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action_a", "action_b", "redundancy_a_to_b"])
        for a, b, r in diag.top_redundant_pairs:
            w.writerow([a, b, r])
    with (out_dir / "top_complementary_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action_a", "action_b", "complementarity"])
        for a, b, c in diag.top_complementary_pairs:
            w.writerow([a, b, c])
