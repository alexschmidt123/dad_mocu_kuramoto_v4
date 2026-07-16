"""Frozen calibrated terminal-control rule shared by all methods."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.control.posterior_ctrl import TerminalControlRule, posterior_safe_u_ctrl, snap_up_to_grid
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
        return {
            "terminal_rule_hash": self.terminal_rule_hash,
            "quantile_level": self.quantile_level,
            "additive_margin": self.additive_margin,
            "alpha": self.alpha,
            "snap_up": self.snap_up,
            "control_grid_hash": self.control_grid_hash,
            "u_candidates": list(self.u_candidates),
            "source": self.source,
            "rule": "snap_up(Q_{1-alpha}(U|w) + margin)",
        }

    def to_control_spec(self, base: ControlSpec) -> ControlSpec:
        """Return a ControlSpec copy with frozen alpha/margin/grid."""
        return ControlSpec(
            alpha=float(self.alpha),
            safety_margin=float(self.margin),
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
    robust = exp_dir / "selected_policy_robust_rule.json"
    legacy = (
        exp_dir
        / "diagnostics"
        / "control_safety_calibration"
        / "calibrated_terminal_rule.json"
    )
    if allow_policy_robust and robust.is_file():
        path = robust
    elif legacy.is_file():
        path = legacy
    else:
        raise FileNotFoundError(
            f"Frozen terminal rule missing under {exp_dir}. "
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
        snap_up=True,
        source=str(path.resolve()),
    )
    if abs(frozen.alpha - 0.05) > 1e-12:
        raise RuntimeError(
            f"Unexpected calibrated α={frozen.alpha}; expected α=0.05."
        )
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
    if abs(frozen.quantile_level - 0.95) > 1e-12:
        raise RuntimeError(f"quantile_level={frozen.quantile_level} != 0.95")
    return frozen


def posterior_to_u_ctrl(
    weights: np.ndarray,
    U_bank: np.ndarray,
    calibrated_rule: FrozenTerminalRule,
) -> float:
    """
    Shared terminal map used by every method:

        u = snap_up( Q_{1-α}(U|w) + margin )
    """
    u = posterior_safe_u_ctrl(
        U_bank,
        weights,
        calibrated_rule.alpha,
        margin=calibrated_rule.margin,
        u_grid=calibrated_rule.u_candidates if calibrated_rule.snap_up else None,
    )
    if calibrated_rule.snap_up:
        return snap_up_to_grid(u, calibrated_rule.u_candidates)
    return float(u)


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
    from src.data import lookup_action_y_sim

    y0 = float(lookup_action_y_sim(system, int(action)))
    z = keyed_noise(
        global_seed=global_seed,
        theta_id=theta_id,
        rollout_id=rollout_id,
        step=step,
        action_id=int(action),
    )
    return y0 + float(sigma_y) * z
