"""Experiment context: method-visible observations only (no full-trajectory leakage)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_EXPECTED_U_TORCH_CACHE: dict[tuple, tuple[torch.Tensor, ...]] = {}

from src.config import SBOEDConfig, load_config, repo_root, resolve_config_path
from src.banks.power_grid import (
    load_bank_from_path,
    resolve_dataset_dir,
    system_name_from_cfg,
)
from src.control.posterior_ctrl import (
    batch_u_ctrl,
    log_prior_uniform_discrete,
    normalize_log_weights,
    posterior_control_decision,
    posterior_ess as ess_from_weights,
    posterior_safe_u_ctrl,
)
from src.control.terminal_rule import load_frozen_terminal_rule
from src.control.u_req import ControlSpec
from src.observations.compress import (
    build_centres_bank,
    build_observation_clean,
    observation_mode,
    observation_report_fields,
    validate_n_obs,
)
from src.observations.likelihood import vector_gaussian_loglik
from src.observations.noise import keyed_noise_vector
from src.domains.swing.design import build_simulator

BELIEF_DIM = 33
GLOBAL_SEED = 77311

METHOD_ALIASES = {
    "dad": "DAD",
    "rl_sboed": "RL-sBOED",
    "rl-sboed": "RL-sBOED",
    "moe_sboed": "MoE-sBOED",
    "moe-sboed": "MoE-sBOED",
    "matched_dense": "MatchedDense",
    "matched-dense": "MatchedDense",
    "step_dad": "Step-DAD",
    "step-dad": "Step-DAD",
    "myopic": "Myopic",
    "fixed": "Fixed",
    "random": "Random",
}
# Default main-table methods.  Step-DAD and MatchedDense are ablation /
# modern-baseline methods selected explicitly (they cost extra online time or
# are only meaningful as MoE controls).
ALL_METHOD_KEYS = (
    "dad",
    "rl_sboed",
    "moe_sboed",
    "myopic",
    "fixed",
    "random",
)
EXTENDED_METHOD_KEYS = ALL_METHOD_KEYS + ("matched_dense", "step_dad")

# Methods that require an offline train step (shell scripts skip training.sh when
# the selected evaluate set contains only keys outside this tuple).
TRAINABLE_METHOD_KEYS = (
    "dad",
    "rl_sboed",
    "moe_sboed",
    "matched_dense",
)


@dataclass
class ExperimentContext:
    """Shared context for learned, hybrid, and baseline design methods."""

    system: str
    cfg: SBOEDConfig
    horizon: int
    n_actions: int
    n_obs: int
    n_sim: int
    obs_dim: int
    obs_indices: np.ndarray
    observation_mode: str
    sigma_y: float
    alpha: float
    margin: float
    u_grid: np.ndarray
    robust_rule: str
    # Method-visible centres only: (n_actions, n_support, obs_dim)
    centres_support: np.ndarray
    U_support: np.ndarray
    log_p0: np.ndarray
    M_support: np.ndarray
    K_support: np.ndarray
    particle_features: np.ndarray
    obs_mean: float
    obs_std: float
    test_systems: list[dict[str, Any]]
    train_systems: list[dict[str, Any]]
    validation_systems: list[dict[str, Any]]
    U_test: np.ndarray
    M_test: np.ndarray
    K_test: np.ndarray
    data_dir: Path
    out_dir: Path
    oracle_tolerance: float
    fixed_sequence: list[int]
    terminal_rule_hash: str
    config_hash: str
    continuous_duration_mode: bool = False
    reset_after_probe: bool = True


def resolve_n_obs(cfg: SBOEDConfig) -> int:
    """N_obs from YAML only (observation.N_obs or legacy top-level N_obs)."""
    obs = dict(cfg.raw.get("observation") or {})
    if "N_obs" in obs:
        return int(obs["N_obs"])
    if "N_obs" in cfg.raw:
        return int(cfg.raw["N_obs"])
    return 5


def resolve_oracle_tolerance(cfg: SBOEDConfig) -> float:
    oracle = dict(cfg.raw.get("oracle") or {})
    if "tolerance" in oracle:
        return float(oracle["tolerance"])
    if "oracle_tolerance" in cfg.raw:
        return float(cfg.raw["oracle_tolerance"])
    return 1e-4


def resolve_sigma_y(cfg: SBOEDConfig) -> float:
    obs = dict(cfg.raw.get("observation") or {})
    if "noise_sigma" in obs:
        return float(obs["noise_sigma"])
    return float(cfg.sigma_y)


def config_sha256(cfg: SBOEDConfig) -> str:
    data = cfg.config_path.read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def normalize_method_key(method: str) -> str:
    raw = str(method).strip()
    key = raw.lower().replace("-", "_")
    compact = "".join(ch for ch in key if ch.isalnum())
    # Accept canonical keys, hyphen aliases, and display names (MatchedDense).
    for canon in EXTENDED_METHOD_KEYS:
        display = METHOD_ALIASES[canon]
        display_key = display.lower().replace("-", "_")
        display_compact = "".join(ch for ch in display_key if ch.isalnum())
        if key in (canon, display_key) or compact == display_compact:
            return canon
    if key in METHOD_ALIASES:
        # Non-canonical alias such as "moe-sboed" -> resolve via display.
        display = METHOD_ALIASES[key]
        return normalize_method_key(display)
    raise ValueError(
        f"Unknown method {method!r}. Allowed: {', '.join(EXTENDED_METHOD_KEYS)}"
    )


def method_display_name(method_key: str) -> str:
    return METHOD_ALIASES[normalize_method_key(method_key)]


def _dedupe_method_keys(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        canon = normalize_method_key(key)
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def methods_from_args(
    cfg: SBOEDConfig, method: str | None
) -> list[str]:
    """Return canonical method keys to run.

    ``method`` may be a single key/display name or a comma-separated list such
    as ``dad,random``.  When omitted, use ``experiment.methods`` from the config
    (with the legacy MoE auto-insert when the config leaves methods implicit).
    """
    if method is not None and str(method).strip() != "":
        raw = str(method).strip()
        if "," in raw:
            return _dedupe_method_keys(
                part.strip() for part in raw.split(",") if part.strip()
            )
        return [normalize_method_key(raw)]

    explicit = list(cfg.methods) if cfg.methods else []
    configured = explicit if explicit else list(ALL_METHOD_KEYS)
    keys = []
    for m in configured:
        try:
            keys.append(normalize_method_key(m))
        except ValueError:
            continue
    if not keys:
        keys = list(ALL_METHOD_KEYS)
    out = _dedupe_method_keys(keys)
    # Only auto-add MoE when the config did not declare experiment.methods
    # (Plan-2 configs omit MoE until DAD/RL beat Fixed/Myopic).
    if not explicit and "moe_sboed" not in out:
        out.insert(2 if "rl_sboed" in out else len(out), "moe_sboed")
    if not explicit and set(out) == {
        "dad",
        "moe_sboed",
        "myopic",
        "fixed",
        "random",
    }:
        out = list(ALL_METHOD_KEYS)
    return out


def training_method_keys(method_keys: list[str]) -> list[str]:
    """Offline trainers required for an evaluate set (preserves train order)."""
    normalized = _dedupe_method_keys(method_keys)
    selected = set(normalized)
    out = [key for key in TRAINABLE_METHOD_KEYS if key in selected]
    if "step_dad" in selected and "dad" not in out:
        out.insert(0, "dad")
    return out


def experiment_out_dir(
    cfg: SBOEDConfig,
    project_root: Path | None = None,
    *,
    experiment_type: str = "objective_based",
    exp_dir: Path | None = None,
    create_new: bool = False,
) -> Path:
    """Result folder: ``date_time_configname_experimentType_Tnum`` under experiments/."""
    from src.layout import resolve_result_dir

    return resolve_result_dir(
        cfg,
        experiment_type,
        exp_dir=exp_dir,
        project_root=project_root,
        create_new=create_new,
    )


def _resolve_terminal_rule(
    cfg: SBOEDConfig,
    *,
    root: Path,
    fallback_exp: Path,
    out_dir: Path | None = None,
) -> tuple[float, float, np.ndarray, str]:
    """
    Resolve (alpha, margin, u_grid, rule_hash).

    ``control_safety_calibration.mode: frozen`` requires a real rule file and
    must not silently fall back to YAML margin (that produced u_ctrl≡1.0).
    """
    import json

    cal = dict(cfg.raw.get("control_safety_calibration") or {})
    mode = str(cal.get("mode", "config")).strip().lower()
    expected_margin = cal.get("expected_margin", None)
    expected_margin_f = None if expected_margin is None else float(expected_margin)
    spec = ControlSpec.from_cfg(cfg)
    u_grid = np.asarray(spec.u_candidates, dtype=np.float64)

    def _from_frozen(path_or_dir: Path) -> tuple[float, float, np.ndarray, str]:
        path_or_dir = Path(path_or_dir)
        if path_or_dir.is_file():
            raw = json.loads(path_or_dir.read_text(encoding="utf-8"))
            rule = raw.get("rule")
            if not isinstance(rule, dict):
                rule = raw
            from src.control.terminal_rule import FrozenTerminalRule

            frozen = FrozenTerminalRule(
                alpha=float(rule["alpha"]),
                margin=float(rule["margin"]),
                u_candidates=tuple(float(x) for x in rule["u_candidates"]),
                snap_up=bool(rule.get("snap_up", True)),
                source=str(path_or_dir.resolve()),
            )
            if expected_margin_f is not None and abs(
                frozen.margin - expected_margin_f
            ) > 1e-12:
                raise RuntimeError(
                    f"Frozen rule margin={frozen.margin} != expected_margin={expected_margin_f} "
                    f"({path_or_dir})"
                )
            return (
                float(frozen.alpha),
                float(frozen.margin),
                np.asarray(frozen.u_candidates, dtype=np.float64),
                str(frozen.metadata().get("terminal_rule_hash", "")),
            )
        frozen = load_frozen_terminal_rule(
            path_or_dir, expected_margin=expected_margin_f
        )
        return (
            float(frozen.alpha),
            float(frozen.margin),
            np.asarray(frozen.u_candidates, dtype=np.float64),
            str(frozen.metadata().get("terminal_rule_hash", "")),
        )

    if mode == "calibrate":
        from src.objectives.mocu.calibrate import ensure_calibrated_rule

        if out_dir is None:
            raise RuntimeError(
                "control_safety_calibration.mode=calibrate requires experiment out_dir "
                "(frozen rule is stored under <exp>/model/, not data/)"
            )
        rule_path = ensure_calibrated_rule(
            cfg, project_root=root, out_dir=out_dir
        )
        return _from_frozen(rule_path)

    if mode in ("frozen", "policy_robust", "policy-robust"):
        from src.layout import model_dir

        candidates: list[Path] = []
        if out_dir is not None:
            candidates.append(model_dir(Path(out_dir)) / "selected_policy_robust_rule.json")
            candidates.append(Path(out_dir) / "selected_policy_robust_rule.json")
        raw_path = str(cal.get("frozen_rule_path", "") or "").strip()
        if raw_path:
            p = Path(raw_path)
            if not p.is_absolute():
                p = root / p
            # Ignore legacy data/<system>/… paths; rules are not bank data.
            if "data/" not in str(p).replace("\\", "/"):
                candidates.append(p)
        candidates.append(fallback_exp / "model" / "selected_policy_robust_rule.json")
        candidates.append(fallback_exp)
        errors: list[str] = []
        for cand in candidates:
            try:
                return _from_frozen(cand)
            except Exception as exc:  # noqa: BLE001 — collect and re-raise clearly
                errors.append(f"{cand}: {exc}")
        raise FileNotFoundError(
            "control_safety_calibration.mode=frozen but no usable terminal rule was found.\n"
            "  Tried:\n    - "
            + "\n    - ".join(errors)
            + "\n  Run with mode: calibrate first, or:\n"
            "  python -m src.objectives.mocu.calibrate --config <yaml> --exp-dir <exp>"
        )

    # mode: config / yaml / off — use ControlSpec from the active config
    alpha = float(spec.alpha)
    margin = float(getattr(spec, "safety_margin", 0.0) or 0.0)
    return alpha, margin, u_grid, ""


def _load_fixed_sequence_file(
    path: Path, *, n_actions: int, horizon: int
) -> tuple[list[int], dict[str, Any]] | None:
    import json

    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    seq_raw = raw.get("selected_action_ids") or raw.get("subset") or []
    if not seq_raw:
        return None
    seq = [int(x) for x in seq_raw]
    if len(seq) != horizon:
        raise RuntimeError(
            f"Fixed sequence length {len(seq)} != horizon T={horizon} in {path}"
        )
    if any(a < 0 or a >= n_actions for a in seq):
        raise RuntimeError(f"Fixed sequence has out-of-range action ids in {path}")
    if len(set(seq)) != len(seq):
        raise RuntimeError(f"Fixed sequence has repeats (no-repeat required) in {path}")
    return seq, raw


def _score_fixed_subset(
    subset: list[int] | tuple[int, ...],
    *,
    centres_by_theta: np.ndarray,
    U_support: np.ndarray,
    log_p0: np.ndarray,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    rng: np.random.Generator,
    max_theta: int | None = None,
) -> float:
    """Mean posterior-safe u_ctrl for an unordered probe subset (order sorted)."""
    subset = tuple(sorted(int(a) for a in subset))
    n_theta, _n_actions, obs_dim = centres_by_theta.shape
    n_use = n_theta if max_theta is None else min(n_theta, int(max_theta))
    scores: list[float] = []
    for tid in range(n_use):
        log_w = np.asarray(log_p0, dtype=np.float64).copy()
        for act in subset:
            clean = np.asarray(centres_by_theta[tid, act], dtype=np.float64)
            y = clean + rng.normal(0.0, sigma_y, size=obs_dim)
            log_w = log_w + vector_gaussian_loglik(
                y, centres_by_theta[:, act, :], sigma_y
            )
        w = normalize_log_weights(log_w)
        scores.append(
            posterior_safe_u_ctrl(U_support, w, alpha, margin=margin, u_grid=u_grid)
        )
    return float(np.mean(scores))


def _greedy_fixed_sequence(
    *,
    centres_by_theta: np.ndarray,
    U_support: np.ndarray,
    log_p0: np.ndarray,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    horizon: int,
    seed: int = 101,
    start_action: int | None = None,
    chronological: bool = False,
) -> tuple[list[int], float]:
    """Greedy no-repeat Fixed design; optional forced first action for restarts."""
    _n_theta, n_actions, _obs_dim = centres_by_theta.shape
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    if start_action is not None:
        chosen.append(int(start_action))
    for step in range(len(chosen), horizon):
        best_a = None
        best_score = float("inf")
        last = max(chosen) if chosen else -1
        remaining = horizon - step
        for a in range(n_actions):
            if a in chosen:
                continue
            if chronological:
                if a <= last or a > n_actions - remaining:
                    continue
            trial = chosen + [a]
            mean_u = _score_fixed_subset(
                trial,
                centres_by_theta=centres_by_theta,
                U_support=U_support,
                log_p0=log_p0,
                sigma_y=sigma_y,
                alpha=alpha,
                margin=margin,
                u_grid=u_grid,
                rng=rng,
            )
            if mean_u < best_score - 1e-12:
                best_score = mean_u
                best_a = a
        if best_a is None:
            raise RuntimeError("Greedy Fixed search failed to select an action")
        chosen.append(int(best_a))
        print(
            f"[fixed] greedy step {step + 1}/{horizon}: action={best_a} "
            f"mean_u_ctrl={best_score:.4f}"
        )
    final = _score_fixed_subset(
        chosen,
        centres_by_theta=centres_by_theta,
        U_support=U_support,
        log_p0=log_p0,
        sigma_y=sigma_y,
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        rng=rng,
    )
    return chosen, final


def _exhaustive_fixed_sequence(
    *,
    centres_by_theta: np.ndarray,
    U_support: np.ndarray,
    log_p0: np.ndarray,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    horizon: int,
    seed: int = 101,
) -> tuple[list[int], float, int]:
    """Enumerate all unordered size-T subsets; return best ordered by increasing id."""
    import itertools

    n_actions = int(centres_by_theta.shape[1])
    rng = np.random.default_rng(seed)
    best_subset: tuple[int, ...] | None = None
    best_score = float("inf")
    n_eval = 0
    total = math.comb(n_actions, horizon)
    print(f"[fixed] exhaustive search C({n_actions},{horizon})={total}")
    report_every = max(1, total // 20)
    for subset in itertools.combinations(range(n_actions), horizon):
        score = _score_fixed_subset(
            subset,
            centres_by_theta=centres_by_theta,
            U_support=U_support,
            log_p0=log_p0,
            sigma_y=sigma_y,
            alpha=alpha,
            margin=margin,
            u_grid=u_grid,
            rng=rng,
        )
        n_eval += 1
        if score < best_score - 1e-12:
            best_score = score
            best_subset = subset
        if n_eval % report_every == 0:
            print(f"  evaluated {n_eval}/{total}  best_u={best_score:.4f}")
    if best_subset is None:
        raise RuntimeError("Exhaustive Fixed search found no subset")
    # Deterministic rollout order: sorted action ids (nonadaptive set)
    return list(best_subset), best_score, n_eval


def _resolve_fixed_sequence(
    cfg: SBOEDConfig,
    *,
    fallback_exp: Path,
    out_dir: Path,
    n_actions: int,
    horizon: int,
    centres_by_theta: np.ndarray | None = None,
    U_support: np.ndarray | None = None,
    log_p0: np.ndarray | None = None,
    sigma_y: float = 0.08,
    alpha: float = 0.05,
    margin: float = 0.0,
    u_grid: np.ndarray | None = None,
    smoke: bool = False,
) -> list[int]:
    """Load a length-T Fixed design; search if missing (never silent range(T)).

    Artifact lives under the result folder ``model/fixed_subset_T{T}.json``
    (not under shared ``data/<system>/``).
    """
    import json
    import time

    from src.layout import model_dir

    save_path = model_dir(Path(out_dir)) / f"fixed_subset_T{horizon}.json"
    sigma_tag = f"{float(sigma_y):g}".replace(".", "p")
    shared_path = (
        resolve_dataset_dir(cfg)
        / "fixed_cache"
        / (
            f"objective_fixed_T{horizon}_Nobs{resolve_n_obs(cfg)}"
            f"_sigma{sigma_tag}.json"
        )
    )
    ctrl = dict(cfg.raw.get("control") or {})
    threshold = int(ctrl.get("fixed_exhaustive_threshold", 5000))
    n_comb = math.comb(n_actions, horizon)
    want_exhaustive = n_comb <= threshold
    # Power-grid duration probes are independent experiments. Fixed chooses an
    # unordered nonrepeated subset; chronological ordering is reserved for SIR.
    chronological = False
    candidates = [
        save_path,
        shared_path,
        Path(out_dir) / "eval" / "fixed" / "subset_meta.json",
        fallback_exp / "eval" / "fixed" / "subset_meta.json",
    ]
    for path in candidates:
        loaded = _load_fixed_sequence_file(path, n_actions=n_actions, horizon=horizon)
        if loaded is None:
            continue
        seq, meta = loaded
        mode = str(meta.get("search_mode", "") or "")
        # Stale greedy / unknown caches must not block publication exhaustive Fixed.
        if want_exhaustive and "exhaustive" not in mode.lower():
            print(
                f"[fixed] ignoring cached {path.name} (search_mode={mode!r}); "
                f"C({n_actions},{horizon})={n_comb} ≤ threshold={threshold} → exhaustive"
            )
            continue
        print(f"[fixed] loaded sequence from {path}: {seq}")
        return seq

    allow_fallback = bool(
        (cfg.raw.get("experiment") or {}).get("allow_trivial_fixed", False)
    ) or bool(smoke)
    if allow_fallback:
        print(
            "[fixed] WARNING: using trivial sequence [0..T-1] "
            "(smoke / allow_trivial_fixed); not a valid Fixed baseline"
        )
        return list(range(horizon))

    if (
        centres_by_theta is None
        or U_support is None
        or log_p0 is None
        or u_grid is None
    ):
        raise FileNotFoundError(
            f"No Fixed subset of length T={horizon} found under {candidates}."
        )

    seed = int(cfg.data.get("train_seed", 101))
    t0 = time.time()
    n_eval = 0

    if want_exhaustive:
        seq, score, n_eval = _exhaustive_fixed_sequence(
            centres_by_theta=centres_by_theta,
            U_support=U_support,
            log_p0=log_p0,
            sigma_y=sigma_y,
            alpha=alpha,
            margin=margin,
            u_grid=u_grid,
            horizon=horizon,
            seed=seed,
        )
        search_mode = "exhaustive_offline"
    else:
        restarts = int(ctrl.get("fixed_greedy_restarts", 4))
        print(
            f"[fixed] C({n_actions},{horizon})={n_comb} > threshold={threshold}; "
            f"multi-restart greedy (restarts={restarts})"
        )
        best_seq: list[int] | None = None
        best_score = float("inf")
        starts = [None] + list(
            np.random.default_rng(seed).choice(n_actions, size=min(restarts, n_actions), replace=False)
        )
        n_eval = 0
        for r_i, start in enumerate(starts):
            seq_r, score_r = _greedy_fixed_sequence(
                centres_by_theta=centres_by_theta,
                U_support=U_support,
                log_p0=log_p0,
                sigma_y=sigma_y,
                alpha=alpha,
                margin=margin,
                u_grid=u_grid,
                horizon=horizon,
                seed=seed + 17 * r_i,
                start_action=None if start is None else int(start),
                chronological=chronological,
            )
            n_eval += 1
            if score_r < best_score - 1e-12:
                best_score = score_r
                best_seq = seq_r
            print(f"  restart {r_i}: score={score_r:.4f} seq={seq_r}")
        assert best_seq is not None
        seq, score = best_seq, best_score
        search_mode = "greedy_multirestart_offline"

    elapsed = time.time() - t0
    print(
        f"[fixed] done mode={search_mode} score={score:.4f} "
        f"seq={seq} ({elapsed:.1f}s) → {save_path}"
    )
    payload = {
        "selected_action_ids": seq,
        "subset": seq,
        "horizon": horizon,
        "search_mode": search_mode,
        "n_actions": n_actions,
        "objective_mean_u_ctrl": score,
        "n_candidates_evaluated": int(n_eval),
        "elapsed_seconds": elapsed,
        "system": system_name_from_cfg(cfg),
        "N_obs": resolve_n_obs(cfg),
        "noise_sigma": float(sigma_y),
    }
    for path in (save_path, shared_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return seq


def build_context_from_config(
    cfg: SBOEDConfig,
    *,
    project_root: Path | None = None,
    ensure_bank: bool = True,
    smoke: bool = False,
    out_dir: Path | None = None,
    experiment_type: str = "objective_based",
) -> ExperimentContext:
    from src.domains.sir.context import build_sir_context, is_sir_config

    if is_sir_config(cfg):
        return build_sir_context(
            cfg,
            project_root=project_root,
            ensure_bank=ensure_bank,
            smoke=smoke,
            out_dir=out_dir,
            experiment_type=experiment_type,
        )

    root = project_root or repo_root()
    system = system_name_from_cfg(cfg)
    data_dir = resolve_dataset_dir(cfg, root)
    from src.banks.power_grid import bank_is_complete

    if ensure_bank and not bank_is_complete(data_dir):
        raise FileNotFoundError(
            f"Physical databank missing or incomplete at {data_dir}. "
            "Training/evaluation is databank-only and will not run the power-grid "
            "simulator on the fly. Restore the complete bank before running the "
            "experiment."
        )

    bank = load_bank_from_path(data_dir, project_root=root, cfg=cfg, smoke=smoke)
    n_sim = int(bank["meta"]["N_sim"])
    n_obs = validate_n_obs(resolve_n_obs(cfg), n_sim)
    centres, indices, mode = build_centres_bank(
        None if n_obs == 0 else bank["full_train"],
        bank["max_rocof_train"],
        n_obs,
    )
    if bool(getattr(cfg, "continuous_duration_mode", False)):
        mode = (
            "continuous_duration_rocof"
            if int(n_obs) == 0
            else f"continuous_duration_{mode}"
        )
    if n_obs == 0 and bank["max_rocof_train"] is None:
        raise RuntimeError(
            f"N_obs=0 requires train/test max_rocof.npy under {data_dir}; bank backfill failed"
        )

    U_train_full = np.asarray(bank["U_train"], dtype=np.float64)
    n_train_bank = int(U_train_full.shape[0])
    # Split before constructing posterior support. Calibration/validation theta
    # must be genuinely off-support; otherwise coverage is optimistically leaked.
    n_val_bank = (
        max(1, n_train_bank // 4)
        if n_train_bank >= 4
        else max(1, n_train_bank // 2)
    )
    n_fit_bank = max(1, n_train_bank - n_val_bank)
    fit_indices = np.arange(n_fit_bank)
    requested_mc = int((cfg.raw.get("prior") or {}).get("mc_samples", n_fit_bank))
    mc_target = min(requested_mc, n_fit_bank)
    if requested_mc > n_fit_bank:
        print(
            f"[prior] capping mc_samples={requested_mc} to {n_fit_bank}; "
            f"{n_val_bank} particles are reserved as strict off-support validation"
        )
    mc_seed = int((cfg.raw.get("prior") or {}).get("mc_support_seed", 1))
    if mc_target < n_fit_bank:
        pick = np.random.default_rng(mc_seed).choice(
            fit_indices, size=mc_target, replace=False
        )
        pick = np.sort(pick)
        print(
            f"[prior] using mc_samples={mc_target}/{n_fit_bank} fit "
            f"particles (seed={mc_seed})"
        )
    else:
        pick = fit_indices
    U_support = U_train_full[pick]
    log_p0 = log_prior_uniform_discrete(U_support.shape[0])

    if out_dir is None:
        out_dir = experiment_out_dir(
            cfg, root, experiment_type=experiment_type, create_new=False
        )
    else:
        from src.layout import assert_experiments_result_dir

        out_dir = assert_experiments_result_dir(
            Path(out_dir).resolve(), project_root=root
        )
        out_dir.mkdir(parents=True, exist_ok=True)
    from src.layout import ensure_result_layout

    ensure_result_layout(out_dir)

    prod_exp = root / "experiments" / f"{system}_T3"
    alpha, margin, u_grid, terminal_rule_hash = _resolve_terminal_rule(
        cfg, root=root, fallback_exp=prod_exp, out_dir=out_dir
    )
    robust_rule = str(ControlSpec.from_cfg(cfg).robust_rule)

    # Per-θ method-visible clean curves: (n_actions, obs_dim)
    def _systems(M, K, U, centres_by_theta) -> list[dict[str, Any]]:
        # centres_by_theta: (n_theta, n_actions, obs_dim)
        out = []
        for i in range(M.shape[0]):
            out.append(
                {
                    "M": M[i].tolist(),
                    "K": K[i].tolist(),
                    "u_req": float(U[i]),
                    "obs_clean": centres_by_theta[i],
                }
            )
        return out

    # centres is (n_actions, n_theta, obs_dim) → transpose to (n_theta, n_actions, obs_dim)
    train_centres_nt_full = np.transpose(centres, (1, 0, 2))
    # Posterior particle support (may be a subset of the train bank).
    centres = np.asarray(centres[:, pick, :], dtype=np.float64)
    train_centres_support_nt = np.transpose(centres, (1, 0, 2))
    test_centres, _, _ = build_centres_bank(
        None if n_obs == 0 else bank["full_test"],
        bank["max_rocof_test"],
        n_obs,
        obs_indices=indices if n_obs > 0 else None,
    )
    test_centres_nt = np.transpose(test_centres, (1, 0, 2))

    train_systems = _systems(
        bank["M_train"], bank["K_train"], bank["U_train"], train_centres_nt_full
    )
    test_systems = _systems(
        bank["M_test"], bank["K_test"], bank["U_test"], test_centres_nt
    )
    validation_systems = train_systems[n_fit_bank:]
    train_fit = train_systems[:n_fit_bank]

    M_mean_full = np.asarray(bank["M_train"], dtype=np.float64).mean(axis=1)
    K_mean_full = np.asarray(bank["K_train"], dtype=np.float64).mean(axis=1)
    M_mean = M_mean_full[pick]
    K_mean = K_mean_full[pick]
    if str(experiment_type).lower() == "eig_based":
        # Pure EIG must retain the spatial latent state.  Collapsing an IEEE
        # system to mean(M), mean(K) makes distinct machine-wise hypotheses
        # indistinguishable to every learned policy, even though the Bayesian
        # likelihood still distinguishes them.  U is a MOCU-only latent and is
        # deliberately excluded from the EIG particle representation.
        M_nodes = np.asarray(bank["M_train"], dtype=np.float64)[pick]
        K_nodes = np.asarray(bank["K_train"], dtype=np.float64)[pick]
        raw_particles = np.concatenate([M_nodes, K_nodes], axis=1)
    else:
        raw_particles = np.column_stack([M_mean, K_mean, U_support]).astype(
            np.float64
        )
    p_mean = raw_particles.mean(axis=0)
    p_std = np.maximum(raw_particles.std(axis=0), 1e-8)
    particles = ((raw_particles - p_mean) / p_std).astype(np.float32)

    obs_flat = train_centres_support_nt.reshape(-1)
    obs_mean = float(obs_flat.mean())
    obs_std = float(max(obs_flat.std(), 1e-8))
    obs_dim = int(centres.shape[-1])

    if str(experiment_type).lower() == "eig_based":
        # Vector-EIG computes its own entropy-optimized Fixed sequence in
        # ``eig_based.vector``.  Running the objective/MOCU exhaustive search
        # here is both scientifically wrong and needlessly CPU-bound.
        fixed_seq = list(range(int(cfg.step_number)))
    else:
        fixed_seq = _resolve_fixed_sequence(
            cfg,
            fallback_exp=prod_exp,
            out_dir=out_dir,
            n_actions=int(centres.shape[0]),
            horizon=int(cfg.step_number),
            centres_by_theta=train_centres_support_nt,
            U_support=U_support,
            log_p0=log_p0,
            sigma_y=resolve_sigma_y(cfg),
            alpha=alpha,
            margin=margin,
            u_grid=u_grid,
            smoke=smoke,
        )

    return ExperimentContext(
        system=system,
        cfg=cfg,
        horizon=int(cfg.step_number),
        n_actions=int(centres.shape[0]),
        n_obs=n_obs,
        n_sim=n_sim,
        obs_dim=obs_dim,
        obs_indices=indices,
        observation_mode=mode,
        sigma_y=resolve_sigma_y(cfg),
        alpha=alpha,
        margin=margin,
        u_grid=u_grid,
        robust_rule=robust_rule,
        centres_support=centres,
        U_support=U_support,
        log_p0=log_p0,
        M_support=M_mean,
        K_support=K_mean,
        particle_features=particles,
        obs_mean=obs_mean,
        obs_std=obs_std,
        test_systems=test_systems,
        train_systems=train_fit,
        validation_systems=validation_systems,
        U_test=np.asarray(bank["U_test"], dtype=np.float64),
        M_test=np.asarray(bank["M_test"], dtype=np.float64),
        K_test=np.asarray(bank["K_test"], dtype=np.float64),
        data_dir=Path(bank["path"]),
        out_dir=out_dir,
        oracle_tolerance=resolve_oracle_tolerance(cfg),
        fixed_sequence=fixed_seq,
        terminal_rule_hash=terminal_rule_hash,
        config_hash=config_sha256(cfg),
        continuous_duration_mode=bool(getattr(cfg, "continuous_duration_mode", False)),
        reset_after_probe=bool(getattr(cfg, "reset_after_probe", True)),
    )


def build_context(
    system: str,
    *,
    project_root=None,
    n_obs: int | None = None,
) -> ExperimentContext:
    """Legacy helper: load study config for a system name."""
    from src.config import SYSTEM_CONFIGS

    root = project_root or repo_root()
    cfg = load_config(resolve_config_path(SYSTEM_CONFIGS[system], root))
    from src.config import with_step_number

    cfg = with_step_number(cfg, int(cfg.step_number))
    if n_obs is not None:
        obs = dict(cfg.raw.get("observation") or {})
        obs["N_obs"] = int(n_obs)
        cfg.raw["observation"] = obs
    return build_context_from_config(cfg, project_root=root)


def update_posterior_vector(
    ctx: ExperimentContext,
    log_w: np.ndarray,
    action: int,
    y_obs: np.ndarray,
) -> np.ndarray:
    centres = ctx.centres_support[int(action)]
    y = np.asarray(y_obs, dtype=np.float64).reshape(-1)
    return log_w + vector_gaussian_loglik(y, centres, ctx.sigma_y)


def observe_compressed(
    system_row: dict[str, Any],
    action: int,
    *,
    sigma_y: float,
    n_obs: int,
    global_seed: int,
    theta_id: int,
    rollout_id: int,
    step: int,
) -> np.ndarray:
    """Noisy method-visible observation via the shared observation interface."""
    forbidden = ("full_delta_f", "delta_f_full", "full_delta_f_bank", "max_rocof_bank")
    for key in forbidden:
        if key in system_row:
            raise RuntimeError(
                f"{key} must not enter method observation path (use obs_clean only)"
            )
    if "obs_clean" not in system_row and "delta_f_obs_clean" not in system_row:
        raise KeyError("system row missing obs_clean")
    clean_bank = system_row.get("obs_clean", system_row.get("delta_f_obs_clean"))
    clean = np.asarray(clean_bank[int(action)], dtype=np.float64).reshape(-1)
    dim = max(int(n_obs), 1)
    if clean.size != dim:
        raise ValueError(f"expected clean shape ({dim},), got {clean.shape}")
    z = keyed_noise_vector(
        global_seed=global_seed,
        theta_id=theta_id,
        rollout_id=rollout_id,
        step=step,
        action_id=int(action),
        n_obs=n_obs,
    )
    return clean + float(sigma_y) * z


def terminal_u_ctrl(ctx: ExperimentContext, log_w: np.ndarray) -> float:
    w = normalize_log_weights(log_w)
    return float(
        posterior_safe_u_ctrl(
            ctx.U_support,
            w,
            ctx.alpha,
            margin=ctx.margin,
            u_grid=ctx.u_grid,
            snap_up=True,
            robust_rule=getattr(ctx, "robust_rule", "quantile"),
        )
    )


def control_from_log_weights(ctx: ExperimentContext, log_w: np.ndarray):
    w = normalize_log_weights(log_w)
    return posterior_control_decision(
        ctx.U_support,
        w,
        ctx.alpha,
        margin=ctx.margin,
        u_grid=ctx.u_grid,
        robust_rule=getattr(ctx, "robust_rule", "quantile"),
    )


def belief_summary(
    ctx: ExperimentContext,
    log_w: np.ndarray,
    observations: list[np.ndarray],
) -> np.ndarray:
    w = normalize_log_weights(log_w)
    feats = np.zeros(BELIEF_DIM, dtype=np.float32)
    feats[0] = float(len(observations)) / float(ctx.horizon)
    ess = float(1.0 / np.sum(w * w))
    feats[1] = ess / float(len(w))
    feats[2] = float(np.max(w))
    feats[3] = float(np.sum(w * ctx.M_support))
    feats[4] = float(
        np.sqrt(max(np.sum(w * (ctx.M_support - feats[3]) ** 2), 0.0))
    )
    feats[5] = float(np.sum(w * ctx.K_support))
    feats[6] = float(
        np.sqrt(max(np.sum(w * (ctx.K_support - feats[5]) ** 2), 0.0))
    )
    order = np.argsort(ctx.U_support, kind="mergesort")
    u_sorted = ctx.U_support[order]
    cdf = np.cumsum(w[order])
    for i, q in enumerate((0.05, 0.25, 0.50, 0.75, 0.95)):
        idx = int(np.searchsorted(cdf, q, side="left"))
        idx = min(max(idx, 0), u_sorted.size - 1)
        feats[7 + i] = float(u_sorted[idx])
    decision = posterior_control_decision(
        ctx.U_support,
        w,
        ctx.alpha,
        margin=ctx.margin,
        u_grid=ctx.u_grid,
        robust_rule=getattr(ctx, "robust_rule", "quantile"),
    )
    feats[12] = float(decision.u_ctrl)
    for i, level in enumerate(ctx.u_grid[:16]):
        feats[13 + i] = float(np.sum(w[np.isclose(ctx.U_support, level)]))
    if observations:
        y = np.concatenate(
            [np.asarray(o, dtype=np.float64).reshape(-1) for o in observations]
        )
        feats[29] = float(y.mean())
        feats[30] = float(y.std() if y.size > 1 else 0.0)
        feats[31] = float(y.min())
        feats[32] = float(y.max())
    return feats


def posterior_mocu(
    ctx: ExperimentContext,
    log_w: np.ndarray,
    *,
    undercontrol_penalty: float = 10.0,
    violation_penalty: float = 0.10,
) -> float:
    """Yoon belief MOCU for the current robust decision.

    With ``robust_rule=ibr_max`` (primary):
        MOCU = Σ_n w_n (u_ctrl - U_n),  u_ctrl = max{U_n : w_n > 0}.

    With ``robust_rule=quantile`` (legacy / chance-constrained), under-control is
    penalized so unsafe shortfalls cannot produce negative MOCU.
    """
    from src.control.posterior_ctrl import belief_mocu

    w = normalize_log_weights(log_w)
    decision = posterior_control_decision(
        ctx.U_support,
        w,
        ctx.alpha,
        margin=ctx.margin,
        u_grid=ctx.u_grid,
        robust_rule=getattr(ctx, "robust_rule", "quantile"),
    )
    u = float(decision.u_ctrl)
    required = np.asarray(ctx.U_support, dtype=np.float64)
    rule = str(getattr(ctx, "robust_rule", "quantile")).lower()
    if rule in {"ibr", "ibr_max", "max", "yoon_ibr"}:
        return belief_mocu(required, w, u)

    shortfall = np.maximum(required - u, 0.0)
    realized_cost = (
        u
        + float(undercontrol_penalty) * shortfall
        + float(violation_penalty) * (shortfall > 0.0)
    )
    regret = realized_cost - required
    return float(np.sum(w * regret))


def expected_u_after_action_vector(
    action: int,
    log_w: np.ndarray,
    *,
    centres: np.ndarray,
    U: np.ndarray,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    idx: np.ndarray,
    noise: np.ndarray,
) -> float:
    c = centres[int(action)]
    y = c[idx] + noise
    s2 = float(sigma_y) ** 2
    d = y.shape[-1]
    resid = y[:, None, :] - c[None, :, :]
    quad = np.sum(resid * resid, axis=-1) / s2
    log_L = -0.5 * d * math.log(2.0 * math.pi * s2) - 0.5 * quad
    log_w_h = log_w[None, :] + log_L
    return float(
        np.mean(
            batch_u_ctrl(
                U,
                log_w_h,
                alpha=alpha,
                margin=margin,
                u_grid=u_grid,
                snap_up=True,
            )
        )
    )


def expected_u_all_actions_torch(
    log_w: np.ndarray,
    *,
    centres: np.ndarray,
    U: np.ndarray,
    sigma_y: float,
    alpha: float,
    margin: float,
    u_grid: np.ndarray,
    idx: np.ndarray,
    noise: np.ndarray,
    feasible: np.ndarray,
    device: str | None = None,
    action_chunk: int = 32,
) -> np.ndarray:
    """CUDA-batched equivalent of ``expected_u_after_action_vector``."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    cache_key = (
        str(dev),
        int(centres.__array_interface__["data"][0]),
        tuple(centres.shape),
        int(U.__array_interface__["data"][0]),
        tuple(np.asarray(u_grid, dtype=np.float64).tolist()),
    )
    cached = _EXPECTED_U_TORCH_CACHE.get(cache_key)
    if cached is None:
        centres_t = torch.as_tensor(
            centres, dtype=torch.float32, device=dev
        )  # A,P,D
        U_t = torch.as_tensor(U, dtype=torch.float32, device=dev)
        order = torch.argsort(U_t)
        U_sorted = U_t[order]
        grid = torch.as_tensor(u_grid, dtype=torch.float32, device=dev)
        cached = (centres_t, order, U_sorted, grid)
        _EXPECTED_U_TORCH_CACHE[cache_key] = cached
    centres_t, order, U_sorted, grid = cached
    log_w_t = torch.as_tensor(log_w, dtype=torch.float32, device=dev)
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=dev)
    noise_t = torch.as_tensor(noise, dtype=torch.float32, device=dev)
    q = 1.0 - float(alpha)
    s2 = float(sigma_y) ** 2
    out = np.full(centres.shape[0], np.inf, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=int)
    with torch.no_grad():
        for start in range(0, len(feasible), int(action_chunk)):
            acts_np = feasible[start : start + int(action_chunk)]
            acts = torch.as_tensor(acts_np, dtype=torch.long, device=dev)
            c = centres_t[acts]  # C,P,D
            y = c[:, idx_t, :] + noise_t[None, :, :]  # C,S,D
            distances = torch.cdist(
                y,
                c,
                p=2.0,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            quad = distances * distances
            log_w_h = log_w_t[None, None, :] - 0.5 * quad / s2
            weights = torch.softmax(log_w_h, dim=-1)
            cdf = torch.cumsum(weights[..., order], dim=-1)
            quantile_index = torch.sum(cdf < q, dim=-1).clamp(
                max=len(U_sorted) - 1
            )
            target = U_sorted[quantile_index] + float(margin)
            grid_index = torch.sum(
                grid[None, None, :] + 1e-12 < target[..., None], dim=-1
            ).clamp(max=len(grid) - 1)
            u_ctrl = grid[grid_index]
            out[acts_np] = u_ctrl.mean(dim=1).cpu().numpy()
    return out


def posterior_ess(log_w: np.ndarray) -> float:
    return ess_from_weights(normalize_log_weights(log_w))


def control_engine_for(ctx: ExperimentContext):
    from src.control.cuda_control import CudaControlEngine

    spec = ControlSpec.from_cfg(ctx.cfg)
    sim = build_simulator(ctx.cfg)
    sim.T_obs_sec = float(spec.T_obs_sec)
    sim.ode_dt = float(spec.ode_dt)
    sim.fs_hz = float(spec.fs_hz)
    return CudaControlEngine(sim, spec), spec


def context_report_meta(ctx: ExperimentContext) -> dict[str, Any]:
    return {
        "system": ctx.system,
        "config_path": str(ctx.cfg.config_path),
        "config_hash": ctx.config_hash,
        "n_theta_train": len(ctx.train_systems) + len(ctx.validation_systems),
        "n_theta_val": len(ctx.validation_systems),
        "n_theta_test": len(ctx.test_systems),
        "n_designs": ctx.n_actions,
        **observation_report_fields(
            n_obs=ctx.n_obs, n_sim=ctx.n_sim, obs_indices=ctx.obs_indices
        ),
        "data_dir": str(ctx.data_dir),
        "out_dir": str(ctx.out_dir),
    }
