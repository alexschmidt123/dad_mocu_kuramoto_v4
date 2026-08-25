"""Frozen calibrated terminal-control rule shared by all methods."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.control.posterior_ctrl import TerminalControlRule, posterior_safe_u_ctrl
from src.control.u_req import ControlSpec


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class FrozenTerminalRule:
    """Immutable calibrated rule used identically by dad/myopic/fixed/random."""

    alpha: float
    margin: float
    u_candidates: tuple[float, ...]
    snap_up: bool = True
    source: str = ""

    @property
    def quantile_level(self) -> float:
        return 1.0 - float(self.alpha)

    @property
    def additive_margin(self) -> float:
        return float(self.margin)

    @property
    def terminal_rule_hash(self) -> str:
        return _stable_hash(
            {
                "alpha": self.alpha,
                "margin": self.margin,
                "quantile_level": self.quantile_level,
                "snap_up": self.snap_up,
                "u_candidates": list(self.u_candidates),
            }
        )

    @property
    def control_grid_hash(self) -> str:
        return _stable_hash(list(self.u_candidates))

    def metadata(self) -> dict[str, Any]:
        formula = (
            "snap_up(Q_{1-alpha}(U|w) + margin)"
            if self.snap_up
            else "Q_{1-alpha}(U|w) + margin"
        )
        return {
            "terminal_rule_hash": self.terminal_rule_hash,
            "quantile_level": self.quantile_level,
            "additive_margin": self.additive_margin,
            "alpha": self.alpha,
            "snap_up": self.snap_up,
            "control_grid_hash": self.control_grid_hash,
            "u_candidates": list(self.u_candidates),
            "source": self.source,
            "rule": formula,
        }

    def as_continuous(self, *, source: str | None = None) -> FrozenTerminalRule:
        """Same α/margin/grid with snap_up disabled (new continuous-control studies)."""
        return FrozenTerminalRule(
            alpha=self.alpha,
            margin=self.margin,
            u_candidates=self.u_candidates,
            snap_up=False,
            source=source or f"{self.source}|continuous_no_snap",
        )

    def to_control_spec(self, base: ControlSpec) -> ControlSpec:
        """Return a ControlSpec copy with frozen alpha/margin/grid."""
        return ControlSpec(
            alpha=float(self.alpha),
            safety_margin=float(self.margin),
            snap_up=bool(self.snap_up),
            robust_rule=base.robust_rule,
            rocof_limit_hz_s=base.rocof_limit_hz_s,
            delta_f_nadir_hz=base.delta_f_nadir_hz,
            profile=base.profile,
            contingency=base.contingency,
            u_candidates=tuple(self.u_candidates),
            myopic_hypothetical=base.myopic_hypothetical,
            fixed_exhaustive_threshold=base.fixed_exhaustive_threshold,
            fixed_noise_replicas=base.fixed_noise_replicas,
            fixed_greedy_restarts=base.fixed_greedy_restarts,
            T_obs_sec=base.T_obs_sec,
            ode_dt=base.ode_dt,
            fs_hz=base.fs_hz,
        )


def load_frozen_terminal_rule(
    exp_dir: Path,
    *,
    expected_margin: float | None = None,
    allow_policy_robust: bool = True,
) -> FrozenTerminalRule:
    """Load calibrated rule from experiment diagnostics; never silently invent one."""
    exp_dir = Path(exp_dir)
    robust_model = exp_dir / "model" / "selected_policy_robust_rule.json"
    robust = exp_dir / "selected_policy_robust_rule.json"
    legacy = (
        exp_dir
        / "diagnostics"
        / "control_safety_calibration"
        / "calibrated_terminal_rule.json"
    )
    if allow_policy_robust and robust_model.is_file():
        path = robust_model
    elif allow_policy_robust and robust.is_file():
        path = robust
    elif legacy.is_file():
        path = legacy
    else:
        raise FileNotFoundError(
            f"Frozen terminal rule missing under {exp_dir}/model/. "
            "Run control_safety_calibration or policy-robust calibration first."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    rule = raw.get("rule")
    # Nested "rule" must be a dict; a formula string must not shadow the body.
    if not isinstance(rule, dict):
        rule = raw
    cands = tuple(float(x) for x in rule["u_candidates"])
    frozen = FrozenTerminalRule(
        alpha=float(rule["alpha"]),
        margin=float(rule["margin"]),
        u_candidates=cands,
        snap_up=bool(rule.get("snap_up", True)),
        source=str(path.resolve()),
    )
    if not (0.0 < frozen.alpha < 1.0):
        raise RuntimeError(f"Invalid calibrated alpha={frozen.alpha}")
    if expected_margin is not None and abs(frozen.margin - float(expected_margin)) > 1e-12:
        raise RuntimeError(
            f"Unexpected margin={frozen.margin}; expected {expected_margin}."
        )
    # Legacy pilot freeze: if loading the original calibrated file without an
    # explicit expected_margin and without a policy-robust override, keep 0.40.
    if (
        expected_margin is None
        and path.resolve() == legacy.resolve()
        and not robust.is_file()
        and abs(frozen.margin - 0.40) > 1e-12
    ):
        raise RuntimeError(
            f"Unexpected calibrated rule α={frozen.alpha}, margin={frozen.margin}; "
            "legacy pilot expects α=0.05, margin=0.40 unless a policy-robust rule is present."
        )
    if not (0.0 < frozen.quantile_level < 1.0):
        raise RuntimeError(
            f"Invalid quantile_level={frozen.quantile_level}"
        )
    return frozen


def posterior_to_u_ctrl(
    weights: np.ndarray,
    U_bank: np.ndarray,
    calibrated_rule: FrozenTerminalRule,
) -> float:
    """
    Shared terminal map used by every method.

    Historical (snap_up=True):  snap_up(Q_{1-α}(U|w) + margin)
    Continuous (snap_up=False): Q_{1-α}(U|w) + margin
    """
    return float(
        posterior_safe_u_ctrl(
            U_bank,
            weights,
            calibrated_rule.alpha,
            margin=calibrated_rule.margin,
            u_grid=calibrated_rule.u_candidates,
            snap_up=calibrated_rule.snap_up,
        )
    )


def assert_shared_rule_metadata(method_metas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Stop evaluation if methods disagree on the frozen rule."""
    if not method_metas:
        raise ValueError("no method metadata")
    keys = (
        "terminal_rule_hash",
        "quantile_level",
        "additive_margin",
        "control_grid_hash",
    )
    ref_name = next(iter(method_metas))
    ref = method_metas[ref_name]
    for name, meta in method_metas.items():
        for k in keys:
            if meta.get(k) != ref.get(k):
                raise RuntimeError(
                    f"Method metadata mismatch: {ref_name}.{k}={ref.get(k)} "
                    f"vs {name}.{k}={meta.get(k)}"
                )
    return {k: ref[k] for k in keys}


def keyed_noise(
    *,
    global_seed: int,
    theta_id: int,
    rollout_id: int,
    step: int,
    action_id: int,
) -> float:
    """Deterministic N(0,1) draw keyed by (seed, θ, rollout, step, action)."""
    seed = (
        int(global_seed) * 1_000_003
        + int(theta_id) * 97_451
        + int(rollout_id) * 1_039
        + int(step) * 31
        + int(action_id)
    ) % (2**31 - 1)
    return float(np.random.default_rng(seed).normal())


def observe_with_keyed_noise(
    system: dict[str, Any],
    action: int,
    *,
    sigma_y: float,
    global_seed: int,
    theta_id: int,
    rollout_id: int,
    step: int,
) -> float:
    """Banked y_sim + keyed Gaussian noise (reproducible, action-specific)."""
    from src.banks.tables import lookup_action_y_sim

    y0 = float(lookup_action_y_sim(system, int(action)))
    z = keyed_noise(
        global_seed=global_seed,
        theta_id=theta_id,
        rollout_id=rollout_id,
        step=step,
        action_id=int(action),
    )
    return y0 + float(sigma_y) * z


def run_keyed_history(
    *,
    system: dict[str, Any],
    theta_id: int,
    rollout_id: int,
    selector: Any,
    table_support: Any,
    U_support: np.ndarray,
    frozen: FrozenTerminalRule,
    horizon: int,
    sigma_y: float,
    global_seed: int,
    rng: np.random.Generator,
    margin_override: float | None = None,
) -> dict[str, Any]:
    """
    One complete T-step history with keyed observation noise.

    Shared by observability, pilot Random, and frozen calibration diagnostics so
    identical seeds yield identical histories.
    """
    from src.control.posterior_ctrl import (
        normalize_log_weights,
        posterior_ess,
        posterior_safe_u_ctrl,
        weighted_quantile,
    )
    from src.objectives.eig.rollout import update_log_weights
    from src.banks.tables import y_sim_last_step_from_tables

    log_w = np.asarray(table_support.log_p0, dtype=np.float64).copy()
    used: set[int] = set()
    seq: list[int] = []
    y_list: list[float] = []
    for step in range(horizon):
        weights = normalize_log_weights(log_w)
        a = int(
            selector.select(
                step=step,
                history_actions=list(seq),
                history_obs=list(y_list),
                used=set(used),
                log_weights=log_w,
                weights=weights,
                rng=rng,
            )
        )
        if a in used:
            raise RuntimeError(f"repeat action {a}")
        y = observe_with_keyed_noise(
            system,
            a,
            sigma_y=sigma_y,
            global_seed=global_seed,
            theta_id=theta_id,
            rollout_id=rollout_id,
            step=step,
        )
        centres = y_sim_last_step_from_tables(table_support, [a])
        log_w = update_log_weights(log_w, y, centres, sigma_y)
        seq.append(a)
        y_list.append(y)
        used.add(a)

    weights = normalize_log_weights(log_w)
    q95 = float(weighted_quantile(U_support, weights, 1.0 - float(frozen.alpha)))
    margin = float(frozen.margin if margin_override is None else margin_override)
    u_ctrl = float(
        posterior_safe_u_ctrl(
            U_support,
            weights,
            frozen.alpha,
            margin=margin,
            u_grid=frozen.u_candidates,
        )
    )
    u_req = float(system["u_req"])
    residual_raw = max(0.0, u_req - q95)
    under = u_ctrl - u_req
    mean_U = float(np.sum(weights * U_support))
    return {
        "sequence": seq,
        "y_obs": y_list,
        "weights": weights,
        "posterior_ess": float(posterior_ess(weights)),
        "max_posterior_weight": float(np.max(weights)),
        "posterior_mean_U": mean_U,
        "posterior_std_U": float(
            np.sqrt(max(np.sum(weights * (U_support - mean_U) ** 2), 0.0))
        ),
        "posterior_quantile": q95,
        "true_u_req": u_req,
        "selected_u_ctrl": u_ctrl,
        "under_control_residual": float(under),
        "raw_residual_r": float(residual_raw),
        "proxy_safe": bool(u_ctrl + 1e-12 >= u_req),
    }
