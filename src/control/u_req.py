"""Terminal control: supplementary active-power injection (not droop gain)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ShapeName = Literal["step", "hann", "ramp"]


@dataclass(frozen=True)
class ControlProfile:
    """Configured terminal-control injection profile (probe_amplitude = 0)."""

    bus: int
    t_start: float
    duration: float
    shape: ShapeName
    units: str = "pu"  # per-unit active power (same scale as P_m)

    def amplitude_at(self, t: float, u_magnitude: float) -> float:
        """Scalar injection at time ``t`` for magnitude ``u_magnitude`` (in ``units``)."""
        u = float(u_magnitude)
        if u == 0.0:
            return 0.0
        t0 = float(self.t_start)
        T = float(self.duration)
        if t < t0 or t > t0 + T or T <= 0.0:
            return 0.0
        tau = (t - t0) / T
        if self.shape == "step":
            return u
        if self.shape == "hann":
            return u * 0.5 * (1.0 - np.cos(2.0 * np.pi * tau))
        if self.shape == "ramp":
            # Linear ramp 0→u over duration, then hold is not used (ends at T).
            return u * float(tau)
        raise ValueError(f"Unknown control shape {self.shape!r}")


@dataclass(frozen=True)
class ContingencySpec:
    bus: int
    magnitude: float  # pu step on P_m from t=0
    units: str = "pu"


@dataclass(frozen=True)
class ControlSpec:
    """Full terminal-control configuration from YAML ``control:``."""

    alpha: float
    rocof_limit_hz_s: float
    delta_f_nadir_hz: float  # frequency *deviation* threshold (Hz), not absolute f
    profile: ControlProfile
    contingency: ContingencySpec
    u_candidates: tuple[float, ...]
    safety_margin: float = 0.0  # additive pu after quantile; snapped up to grid
    # "ibr_max" = Yoon IBR max{U_n:w_n>0}; "quantile" = Q_{1-α}+margin
    robust_rule: str = "quantile"
    myopic_hypothetical: int = 16
    fixed_exhaustive_threshold: int = 5000
    fixed_noise_replicas: int = 2
    fixed_greedy_restarts: int = 4
    # Shared integrator settings (must match bank + true-system eval)
    T_obs_sec: float = 10.0
    ode_dt: float = 1.0 / 160.0
    fs_hz: float = 12.0

    @property
    def u_min(self) -> float:
        return float(min(self.u_candidates)) if self.u_candidates else 0.0

    @property
    def u_max(self) -> float:
        return float(max(self.u_candidates)) if self.u_candidates else 0.0

    def u_grid(self) -> np.ndarray:
        return np.asarray(self.u_candidates, dtype=np.float64)

    def terminal_rule(self) -> "TerminalControlRule":
        from src.control.posterior_ctrl import TerminalControlRule

        return TerminalControlRule(
            alpha=float(self.alpha),
            margin=float(self.safety_margin),
            u_candidates=tuple(self.u_candidates),
            robust_rule=(
                "ibr_max"
                if str(self.robust_rule).lower() in {"ibr", "ibr_max", "max", "yoon_ibr"}
                else "quantile"
            ),
        )

    @classmethod
    def from_cfg(cls, cfg: Any) -> ControlSpec:
        raw = dict(getattr(cfg, "raw", {}).get("control") or {})
        sw = dict(getattr(cfg, "raw", {}).get("swing_equation") or {})

        # Explicit profile block (preferred) with flat-key fallbacks.
        prof = dict(raw.get("profile") or {})
        shape = str(prof.get("shape", raw.get("shape", "hann"))).lower()
        if shape not in {"step", "hann", "ramp"}:
            raise ValueError(f"control.profile.shape must be step|hann|ramp, got {shape}")

        units = str(prof.get("units", raw.get("units", "pu"))).lower()
        if units not in {"pu", "mw"}:
            raise ValueError(f"control units must be 'pu' or 'mw', got {units}")
        if units == "mw":
            raise ValueError(
                "control.units='mw' requires an explicit base-power conversion; "
                "use units: pu (same scale as P_m) for this codebase."
            )

        # Candidate magnitudes: explicit list preferred over linspace.
        if "u_candidates" in raw:
            cands = tuple(float(x) for x in raw["u_candidates"])
        else:
            u_min = float(raw.get("u_min", 0.0))
            u_max = float(raw.get("u_max", 5.0))
            u_levels = int(raw.get("u_levels", 26))
            cands = tuple(float(x) for x in np.linspace(u_min, u_max, u_levels))
        if not cands:
            raise ValueError("control.u_candidates must be non-empty")
        if sorted(cands) != list(cands):
            raise ValueError("control.u_candidates must be sorted ascending")

        cont = dict(raw.get("contingency") or {})
        # Nadir: prefer delta_f_nadir_hz; reject ambiguous legacy key without rename.
        if "delta_f_nadir_hz" in raw:
            nadir = float(raw["delta_f_nadir_hz"])
        elif "f_nadir_hz" in raw:
            raise ValueError(
                "Use control.delta_f_nadir_hz for frequency *deviation* thresholds. "
                "Absolute-frequency key f_nadir_hz is not supported in this model "
                "(ω trajectories are deviations from nominal)."
            )
        elif "f_nadir_limit" in raw:
            # Temporary bridge: treat legacy key as deviation, but require rename.
            nadir = float(raw["f_nadir_limit"])
        else:
            nadir = -0.30

        rocof = float(raw.get("rocof_limit_hz_s", raw.get("rocof_limit", 15.0)))

        T_obs = float(raw.get("T_obs_sec", sw.get("T_obs_sec", getattr(cfg, "T_obs_sec", 10.0))))
        fs = float(raw.get("fs_hz", sw.get("fs_hz", getattr(cfg, "fs_hz", 12.0))))
        ode_dt = float(raw.get("ode_dt", 1.0 / 160.0))

        rule_raw = str(raw.get("robust_rule", "quantile")).strip().lower()
        robust_rule = (
            "ibr_max"
            if rule_raw in {"ibr", "ibr_max", "max", "yoon_ibr"}
            else "quantile"
        )
        return cls(
            alpha=float(raw.get("alpha", 0.05)),
            safety_margin=float(raw.get("safety_margin", raw.get("margin", 0.0))),
            robust_rule=robust_rule,
            rocof_limit_hz_s=rocof,
            delta_f_nadir_hz=nadir,
            profile=ControlProfile(
                bus=int(prof.get("bus", raw.get("control_bus", 0))),
                t_start=float(prof.get("t_start", raw.get("t_start", 0.0))),
                duration=float(prof.get("duration", raw.get("duration", 2.0))),
                shape=shape,  # type: ignore[arg-type]
                units=units,
            ),
            contingency=ContingencySpec(
                bus=int(cont.get("bus", raw.get("contingency_bus", 0))),
                magnitude=float(cont.get("magnitude", raw.get("contingency_magnitude", -0.35))),
                units=str(cont.get("units", "pu")),
            ),
            u_candidates=cands,
            myopic_hypothetical=int(raw.get("myopic_hypothetical", 16)),
            fixed_exhaustive_threshold=int(raw.get("fixed_exhaustive_threshold", 5000)),
            fixed_noise_replicas=int(raw.get("fixed_noise_replicas", 2)),
            fixed_greedy_restarts=int(raw.get("fixed_greedy_restarts", 4)),
            T_obs_sec=T_obs,
            ode_dt=ode_dt,
            fs_hz=fs,
        )


def is_control_safe(
    rocof_max: float,
    delta_f_nadir: float,
    spec: ControlSpec,
) -> bool:
    return (rocof_max <= spec.rocof_limit_hz_s) and (delta_f_nadir >= spec.delta_f_nadir_hz)


def metrics_from_omega_traj(
    omega_traj: np.ndarray,
    *,
    ode_dt: float,
) -> dict[str, float]:
    """
    Shared ROCOF / nadir definitions for bank generation and true-system eval.

    ``omega_traj`` shape (n_steps, N): angular frequency *deviation* [rad/s].
    ``delta_f = omega / (2π)`` [Hz deviation].
    ROCOF = max |d(delta_f)/dt| over buses and time via forward differences.
    Nadir = min delta_f over buses and time (most negative deviation).
    """
    omega = np.asarray(omega_traj, dtype=np.float64)
    if omega.ndim != 2 or omega.shape[0] < 1:
        raise ValueError("omega_traj must be (n_steps, N)")
    delta_f = omega / (2.0 * np.pi)
    if omega.shape[0] < 2:
        rocof_max = 0.0
    else:
        df = np.diff(delta_f, axis=0) / float(ode_dt)
        rocof_max = float(np.max(np.abs(df)))
    delta_f_nadir = float(np.min(delta_f))
    return {
        "rocof_max": rocof_max,
        "delta_f_nadir": delta_f_nadir,
        "frequency_nadir": delta_f_nadir,  # alias: deviation, not absolute f
    }
