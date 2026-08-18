"""T=2 planning utilities over existing Y/U banks (no simulator)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .posterior_utils import (
    terminal_u,
    uniform_log_prior,
    update_log_weights,
    weights_from_log,
)


@dataclass
class SearchMeta:
    mode: str  # EXACT | APPROXIMATE
    designs_considered: int
    designs_total: int
    label: str  # adaptive_planning_oracle | approx_adaptive_planner | fixed_search

    def as_dict(self) -> dict[str, Any]:
        return {
            "Adaptive_search": self.mode,
            "Candidate_designs_used": f"{self.designs_considered} / total {self.designs_total}",
            "planner_label": self.label,
        }


@dataclass
class CRNBundle:
    """Common-random-number tables for paired policy comparison.

    Observation noise is shared across policies at (theta_i, replicate_r, step_t):
        y = Y_eval[i, a] + eps_obs[i, r, t]
    regardless of which design a the policy selects at that step.

    Hypothetical scoring noise for one-step expectations is shared across
    candidate designs compared at the same decision index (i, r, t).
    """

    eps_obs: np.ndarray  # (n_eval, n_rep, T)
    hyp_idx: np.ndarray  # (n_eval, n_rep, T, n_hyp) particle indices
    hyp_noise: np.ndarray  # (n_eval, n_rep, T, n_hyp)

    @property
    def n_eval(self) -> int:
        return int(self.eps_obs.shape[0])

    @property
    def n_rep(self) -> int:
        return int(self.eps_obs.shape[1])

    @property
    def horizon(self) -> int:
        return int(self.eps_obs.shape[2])


def make_crn_bundle(
    *,
    n_eval: int,
    n_rep: int,
    horizon: int,
    n_support: int,
    n_hyp: int,
    sigma_y: float,
    rng: np.random.Generator,
) -> CRNBundle:
    return CRNBundle(
        eps_obs=rng.normal(0.0, sigma_y, size=(n_eval, n_rep, horizon)),
        # Uniform prior indices; reweighted decisions still share the same draws.
        hyp_idx=rng.integers(0, n_support, size=(n_eval, n_rep, horizon, n_hyp)),
        hyp_noise=rng.normal(0.0, sigma_y, size=(n_eval, n_rep, horizon, n_hyp)),
    )


def _centres_a(Y: np.ndarray, a: int) -> np.ndarray:
    return np.asarray(Y[:, a], dtype=np.float64)[:, None]


def observe_crn(Y_eval: np.ndarray, i: int, a: int, eps: float) -> float:
    return float(Y_eval[int(i), int(a)]) + float(eps)


def expected_u_one_step(
    Y: np.ndarray,
    U: np.ndarray,
    design_id: int,
    *,
    log_w: np.ndarray,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    idx: np.ndarray,
    noise: np.ndarray,
) -> float:
    """E_y[u_ctrl] after one design, CRN via (idx, noise)."""
    a = int(design_id)
    centres = _centres_a(Y, a)
    scores: list[float] = []
    for r, n_idx in enumerate(idx):
        # Re-index under current posterior: map uniform-support index through
        # multinomial using current weights when possible.
        y = float(centres[int(n_idx) % centres.shape[0], 0] + noise[r])
        log_w2 = update_log_weights(log_w, y, centres, sigma_y)
        w2 = weights_from_log(log_w2)
        scores.append(terminal_u(U, w2, alpha=alpha, margin=margin, u_grid=u_grid))
    return float(np.mean(scores))


def expected_u_one_step_posterior_crn(
    Y: np.ndarray,
    U: np.ndarray,
    design_id: int,
    *,
    log_w: np.ndarray,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    hyp_idx_raw: np.ndarray,
    hyp_noise: np.ndarray,
) -> float:
    """One-step expectation with CRN remapped through current posterior weights."""
    a = int(design_id)
    centres = _centres_a(Y, a)
    w = weights_from_log(log_w)
    cdf = np.cumsum(w)
    scores: list[float] = []
    n = len(U)
    for r in range(len(hyp_noise)):
        # Convert raw integer to u~(0,1) then inverse-CDF sample under posterior.
        u = (float(int(hyp_idx_raw[r]) % max(n, 1)) + 0.5) / float(max(n, 1))
        n_idx = int(np.searchsorted(cdf, min(u, 1.0 - 1e-12), side="left"))
        n_idx = min(max(n_idx, 0), n - 1)
        y = float(centres[n_idx, 0] + hyp_noise[r])
        log_w2 = update_log_weights(log_w, y, centres, sigma_y)
        w2 = weights_from_log(log_w2)
        scores.append(terminal_u(U, w2, alpha=alpha, margin=margin, u_grid=u_grid))
    return float(np.mean(scores))


def score_all_one_step(
    Y: np.ndarray,
    U: np.ndarray,
    *,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_hyp: int,
    rng: np.random.Generator,
    used: set[int] | None = None,
    log_w: np.ndarray | None = None,
    candidates: Sequence[int] | None = None,
    hyp_idx: np.ndarray | None = None,
    hyp_noise: np.ndarray | None = None,
) -> dict[int, float]:
    used = used or set()
    n = len(U)
    log_w = uniform_log_prior(n) if log_w is None else np.asarray(log_w, dtype=np.float64)
    w = weights_from_log(log_w)
    if hyp_idx is None or hyp_noise is None:
        idx = rng.choice(n, size=int(n_hyp), p=w)
        noise = rng.normal(0.0, sigma_y, size=int(n_hyp))
        use_posterior_crn = False
    else:
        idx = np.asarray(hyp_idx, dtype=np.int64).reshape(-1)[: int(n_hyp)]
        noise = np.asarray(hyp_noise, dtype=np.float64).reshape(-1)[: int(n_hyp)]
        use_posterior_crn = True
    acts = list(candidates) if candidates is not None else list(range(Y.shape[1]))
    out: dict[int, float] = {}
    for a in acts:
        if a in used:
            continue
        if use_posterior_crn:
            out[int(a)] = expected_u_one_step_posterior_crn(
                Y,
                U,
                a,
                log_w=log_w,
                sigma_y=sigma_y,
                alpha=alpha,
                margin=margin,
                u_grid=u_grid,
                hyp_idx_raw=idx,
                hyp_noise=noise,
            )
        else:
            out[int(a)] = expected_u_one_step(
                Y,
                U,
                a,
                log_w=log_w,
                sigma_y=sigma_y,
                alpha=alpha,
                margin=margin,
                u_grid=u_grid,
                idx=idx,
                noise=noise,
            )
    return out


def V1_continuation(
    Y: np.ndarray,
    U: np.ndarray,
    *,
    log_w: np.ndarray,
    used: set[int],
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_hyp: int,
    rng: np.random.Generator,
    second_candidates: Sequence[int] | None = None,
    hyp_idx: np.ndarray | None = None,
    hyp_noise: np.ndarray | None = None,
) -> tuple[float, int, dict[int, float]]:
    """min_xi2 E_y2[u_ctrl] under current posterior; returns (V, best_a, scores)."""
    scores = score_all_one_step(
        Y,
        U,
        sigma_y=sigma_y,
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        n_hyp=n_hyp,
        rng=rng,
        used=used,
        log_w=log_w,
        candidates=second_candidates,
        hyp_idx=hyp_idx,
        hyp_noise=hyp_noise,
    )
    if not scores:
        raise RuntimeError("no feasible second designs")
    best_a = min(scores, key=scores.get)
    return float(scores[best_a]), int(best_a), scores


def J_adaptive_T2(
    Y: np.ndarray,
    U: np.ndarray,
    *,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_hyp: int,
    rng: np.random.Generator,
    first_candidates: Sequence[int] | None = None,
    second_candidates: Sequence[int] | None = None,
    exact: bool = True,
) -> tuple[float, int, SearchMeta, dict[int, float]]:
    """
    J = min_xi1 E_y1[ V1(xi1,y1) ] with V1 = min_xi2 E_y2[u_ctrl].
    """
    n_actions = Y.shape[1]
    n = len(U)
    log_w0 = uniform_log_prior(n)
    w0 = weights_from_log(log_w0)
    firsts = list(first_candidates) if first_candidates is not None else list(range(n_actions))
    idx = rng.choice(n, size=int(n_hyp), p=w0)
    noise = rng.normal(0.0, sigma_y, size=int(n_hyp))

    j1_scores: dict[int, float] = {}
    for a1 in firsts:
        vals: list[float] = []
        centres = _centres_a(Y, int(a1))
        for r, n_idx in enumerate(idx):
            y1 = float(centres[int(n_idx), 0] + noise[r])
            log_w1 = update_log_weights(log_w0, y1, centres, sigma_y)
            # Deterministic CRN stream for second-step scoring (shared across a1 at r).
            rng2 = np.random.default_rng(10_000_003 + 1_000_003 * int(a1) + int(r))
            v1, _best, _ = V1_continuation(
                Y,
                U,
                log_w=log_w1,
                used={int(a1)},
                sigma_y=sigma_y,
                alpha=alpha,
                margin=margin,
                u_grid=u_grid,
                n_hyp=n_hyp,
                rng=rng2,
                second_candidates=second_candidates,
            )
            vals.append(v1)
        j1_scores[int(a1)] = float(np.mean(vals))

    best_first = min(j1_scores, key=j1_scores.get)
    n_first = len(firsts)
    n_second = n_actions if second_candidates is None else len(second_candidates)
    # Report the binding restriction (first-design screen when second is full).
    n_cons = n_first if n_second >= n_actions else max(n_first, n_second)
    meta = SearchMeta(
        mode="EXACT" if exact else "APPROXIMATE",
        designs_considered=int(n_cons),
        designs_total=int(n_actions),
        label="adaptive_planning_oracle" if exact else "approx_adaptive_planner",
    )
    return float(j1_scores[best_first]), int(best_first), meta, j1_scores


def J_myopic_T2_on_eval(
    Y_support: np.ndarray,
    U_support: np.ndarray,
    Y_eval: np.ndarray,
    *,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_hyp: int,
    crn: CRNBundle,
    candidates: Sequence[int] | None = None,
    fixed_first_design: int | None = None,
) -> np.ndarray:
    """Per-eval-θ terminal u under Myopic Control for T=2 with CRN observations.

    If ``fixed_first_design`` is provided, step-0 uses that design for all
    (theta, replicate) pairs (deterministic prior Myopic choice). Step-1 remains
    myopic under the posterior. This matches the open-loop-first structure of the
    T=2 planner and makes the dominance sanity check well-posed.
    """
    n_eval = Y_eval.shape[0]
    n_rep = crn.n_rep
    out = np.empty((n_eval, n_rep), dtype=np.float64)
    for i in range(n_eval):
        for r in range(n_rep):
            log_w = uniform_log_prior(len(U_support))
            used: set[int] = set()
            for t in range(2):
                if t == 0 and fixed_first_design is not None:
                    a = int(fixed_first_design)
                else:
                    scores = score_all_one_step(
                        Y_support,
                        U_support,
                        sigma_y=sigma_y,
                        alpha=alpha,
                        margin=margin,
                        u_grid=u_grid,
                        n_hyp=n_hyp,
                        rng=np.random.default_rng(0),  # unused when hyp_* provided
                        used=used,
                        log_w=log_w,
                        candidates=candidates,
                        hyp_idx=crn.hyp_idx[i, r, t],
                        hyp_noise=crn.hyp_noise[i, r, t],
                    )
                    a = min(scores, key=scores.get)
                y = observe_crn(Y_eval, i, a, crn.eps_obs[i, r, t])
                log_w = update_log_weights(log_w, y, _centres_a(Y_support, a), sigma_y)
                used.add(int(a))
            w = weights_from_log(log_w)
            out[i, r] = terminal_u(
                U_support, w, alpha=alpha, margin=margin, u_grid=u_grid
            )
    return out


def J_fixed_T2_on_eval(
    Y_support: np.ndarray,
    U_support: np.ndarray,
    Y_eval: np.ndarray,
    sequence: Sequence[int],
    *,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    crn: CRNBundle,
) -> np.ndarray:
    seq = [int(a) for a in sequence]
    n_eval = Y_eval.shape[0]
    n_rep = crn.n_rep
    out = np.empty((n_eval, n_rep), dtype=np.float64)
    for i in range(n_eval):
        for r in range(n_rep):
            log_w = uniform_log_prior(len(U_support))
            for t, a in enumerate(seq):
                y = observe_crn(Y_eval, i, a, crn.eps_obs[i, r, t])
                log_w = update_log_weights(log_w, y, _centres_a(Y_support, a), sigma_y)
            w = weights_from_log(log_w)
            out[i, r] = terminal_u(
                U_support, w, alpha=alpha, margin=margin, u_grid=u_grid
            )
    return out


def J_adaptive_planner_on_eval(
    Y_support: np.ndarray,
    U_support: np.ndarray,
    Y_eval: np.ndarray,
    *,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_hyp: int,
    crn: CRNBundle,
    best_first: int,
    first_candidates: Sequence[int] | None = None,
    second_candidates: Sequence[int] | None = None,
) -> np.ndarray:
    """Closed-loop T=2 planning policy with CRN observations.

    First design is the open-loop planning choice ``best_first`` (already computed
    on support). Second design is chosen by exact/approx V1 after y1.
    """
    a1 = int(best_first)
    _ = first_candidates  # documented for callers; first action fixed by planning
    n_eval = Y_eval.shape[0]
    n_rep = crn.n_rep
    out = np.empty((n_eval, n_rep), dtype=np.float64)
    for i in range(n_eval):
        for r in range(n_rep):
            log_w = uniform_log_prior(len(U_support))
            y1 = observe_crn(Y_eval, i, a1, crn.eps_obs[i, r, 0])
            log_w = update_log_weights(log_w, y1, _centres_a(Y_support, a1), sigma_y)
            _v, a2, _ = V1_continuation(
                Y_support,
                U_support,
                log_w=log_w,
                used={a1},
                sigma_y=sigma_y,
                alpha=alpha,
                margin=margin,
                u_grid=u_grid,
                n_hyp=n_hyp,
                rng=np.random.default_rng(0),
                second_candidates=second_candidates,
                hyp_idx=crn.hyp_idx[i, r, 1],
                hyp_noise=crn.hyp_noise[i, r, 1],
            )
            y2 = observe_crn(Y_eval, i, a2, crn.eps_obs[i, r, 1])
            log_w = update_log_weights(log_w, y2, _centres_a(Y_support, a2), sigma_y)
            w = weights_from_log(log_w)
            out[i, r] = terminal_u(
                U_support, w, alpha=alpha, margin=margin, u_grid=u_grid
            )
    return out


def select_fixed_T2(
    Y: np.ndarray,
    U: np.ndarray,
    *,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    rng: np.random.Generator,
    exact_threshold: int,
    candidates: Sequence[int] | None = None,
) -> tuple[tuple[int, int], float, SearchMeta]:
    """Best length-2 Fixed sequence by MC mean u_ctrl on support."""
    from src.objectives.mocu.context import _score_fixed_subset
    from src.control.posterior_ctrl import log_prior_uniform_discrete

    acts = list(candidates) if candidates is not None else list(range(Y.shape[1]))
    centres_by_theta = np.asarray(Y[:, :, None], dtype=np.float64)
    log_p0 = log_prior_uniform_discrete(len(U))
    n_actions_full = Y.shape[1]
    n_pairs = len(acts) * (len(acts) - 1) // 2
    # Exact only when searching the full catalog under the pair budget.
    exact = candidates is None and n_pairs <= exact_threshold
    best_seq: tuple[int, int] | None = None
    best_score = float("inf")

    if exact or candidates is not None or n_pairs <= exact_threshold:
        for i, a in enumerate(acts):
            for b in acts[i + 1 :]:
                s_ab = _score_fixed_subset(
                    (a, b),
                    centres_by_theta=centres_by_theta,
                    U_support=U,
                    log_p0=log_p0,
                    sigma_y=sigma_y,
                    alpha=alpha,
                    margin=margin,
                    u_grid=u_grid,
                    rng=rng,
                )
                s_ba = _score_fixed_subset(
                    (b, a),
                    centres_by_theta=centres_by_theta,
                    U_support=U,
                    log_p0=log_p0,
                    sigma_y=sigma_y,
                    alpha=alpha,
                    margin=margin,
                    u_grid=u_grid,
                    rng=rng,
                )
                if s_ab <= s_ba and s_ab < best_score:
                    best_score = float(s_ab)
                    best_seq = (int(a), int(b))
                elif s_ba < best_score:
                    best_score = float(s_ba)
                    best_seq = (int(b), int(a))
        mode = "EXACT" if (candidates is None and n_pairs <= exact_threshold) else "APPROXIMATE"
    else:
        from src.objectives.mocu.context import _greedy_fixed_sequence

        best_seq_list, best_score = _greedy_fixed_sequence(
            centres_by_theta=centres_by_theta,
            U_support=U,
            log_p0=log_p0,
            sigma_y=sigma_y,
            alpha=alpha,
            margin=margin,
            u_grid=u_grid,
            horizon=2,
            seed=int(rng.integers(0, 10_000)),
        )
        best_seq = (int(best_seq_list[0]), int(best_seq_list[1]))
        mode = "APPROXIMATE"

    assert best_seq is not None
    meta = SearchMeta(
        mode=mode,
        designs_considered=len(acts),
        designs_total=n_actions_full,
        label="fixed_search",
    )
    return best_seq, float(best_score), meta


def screen_top_designs(
    Y: np.ndarray,
    U: np.ndarray,
    *,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_hyp: int,
    rng: np.random.Generator,
    top_k: int,
) -> list[int]:
    scores = score_all_one_step(
        Y,
        U,
        sigma_y=sigma_y,
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        n_hyp=n_hyp,
        rng=rng,
    )
    ranked = sorted(scores, key=scores.get)
    return [int(a) for a in ranked[: max(1, int(top_k))]]


def myopic_first_design(
    Y: np.ndarray,
    U: np.ndarray,
    *,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    n_hyp: int,
    rng: np.random.Generator,
) -> tuple[int, dict[int, float]]:
    """Prior one-step Myopic Control design (full catalog)."""
    scores = score_all_one_step(
        Y,
        U,
        sigma_y=sigma_y,
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        n_hyp=n_hyp,
        rng=rng,
    )
    best = min(scores, key=scores.get)
    return int(best), scores


def build_adaptive_candidate_set(
    *,
    n_actions: int,
    screened: Sequence[int],
    myopic_designs: Sequence[int],
    fixed_designs: Sequence[int],
    exact: bool,
) -> tuple[list[int], list[int]]:
    """Build first/second candidate sets for adaptive planning.

    Requirement: approximate adaptive search always includes every design
    selected by Myopic and by the best Fixed search.

    Second-step candidates are the **full catalog** whenever we are already
    approximate on the first step, so last-step continuation matches Myopic
    optimality and restores J_planning <= J_myopic whenever the myopic first
    design is included in the first-step candidate set.
    """
    if exact:
        full = list(range(n_actions))
        return full, full
    first = sorted(
        {
            int(a)
            for a in list(screened) + list(myopic_designs) + list(fixed_designs)
            if 0 <= int(a) < n_actions
        }
    )
    second = list(range(n_actions))
    return first, second


def planner_sanity_check(
    *,
    j_planning: float,
    j_myopic: float,
    exact: bool,
    tol: float,
) -> dict[str, Any]:
    """Exact adaptive planner cannot be worse than Myopic (admissible policy)."""
    gap = float(j_planning - j_myopic)
    ok = gap <= float(tol)
    if ok:
        status = "OK"
        note = "J_planning <= J_myopic within tolerance."
    elif exact:
        status = "EXACT_VIOLATION"
        note = (
            "Exact planner exceeded Myopic beyond tolerance; indicates a bug in "
            "planning/evaluation, not a scientific finding that planning is harmful."
        )
    else:
        status = "APPROXIMATE_VIOLATION"
        note = (
            "Approximate planner exceeded Myopic. This is a search-quality failure "
            "(candidate screening), not evidence that non-myopic planning is "
            "intrinsically worse than Myopic Control."
        )
    return {
        "status": status,
        "ok": bool(ok),
        "J_planning": float(j_planning),
        "J_myopic": float(j_myopic),
        "gap_planning_minus_myopic": gap,
        "tolerance": float(tol),
        "exact_search": bool(exact),
        "note": note,
    }
