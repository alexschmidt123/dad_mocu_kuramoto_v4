"""
Helpers for posterior/MOCU episode tests (pseudocode.tex Alg. dad_mocu_loop, no π_φ).

**Physics “main body” (same code for any sBOED horizon T)**  
:func:`run_physics_episode` — precompute ``γ*(θ_n)``, ``Map(θ_n,ξ)``, then sequential
Gaussian likelihood updates. **T = 1** (single design step) is the same loop with one
probe–observe round—a special case of sequential BOED, not a different model.

**Convenience**  
:func:`run_single_step_physics_episode` wraps ``run_physics_episode(..., T=1)``
and adds :func:`~src.core.discrete_bayes.single_step_discrete_bayes_report` plus
``γ*(θ_true)`` / ``u_ctrl`` diagnostics. Real swing parameters come **only** from
``DEFAULT_SWING_YAML`` (``config/early_test.yaml``) via :func:`swing_physics_kwargs_from_yaml`.

**Staged T=1 (ξ → y → posterior → γ/u_ctrl → MOCU)**  
:func:`main_body_single_step_simulation` — same core as the multi-step physics body, nested
by pipeline stage (no policy, DAD, baselines, or eval scripts).

**Fast tests only**  
:func:`run_synthetic_episode` replaces Map and ``γ*`` with closed-form surrogates (no ODE).
Use it for cheap unit tests, not as the one-step version of the physics body.

- Posterior / MOCU: ``src.core.discrete_bayes``
- Map / likelihood: ``src.core.likelihood``
- γ*(θ): ``src.core.swing_equation_mocu.binary_search_gamma_star_batch``
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# --- sBOED: sequential experiment length T (probe–observe rounds); not ODE horizon ---
DEFAULT_T = 4

# --- ODE integration horizon (seconds), not sBOED T ---
ODE_HORIZON_DEFAULT = 5.0
ODE_HORIZON_QUICK = 3.0
QUICK_PHYSICS_GRID = 2
QUICK_PHYSICS_TIMEOUT = 15.0

# Real swing physics always uses this YAML (see ``cli.py``).
DEFAULT_SWING_YAML = Path(__file__).resolve().parents[2] / "config" / "early_test.yaml"


def resolve_inference_device(explicit: str | None = None) -> str:
    """
    Device string for PyTorch swing ODE / γ* (``device='cuda'`` or ``'cpu'``).

    This path uses **PyTorch + CUDA**, not the separate **PyCUDA** MOCU kernels in
    ``mocu_pycuda`` (those are for bulk MOCU Monte Carlo elsewhere).

    Resolution: ``explicit`` if given; else env ``POSTERIOR_DEVICE`` (``cpu`` / ``cuda``);
    else ``cuda`` when ``torch.cuda.is_available()``, else ``cpu``.
    """
    if explicit is not None:
        return explicit
    v = os.environ.get("POSTERIOR_DEVICE", "").strip().lower()
    if v in ("cpu", "cuda"):
        return v
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw if raw is not None else {}


def initial_setup_document(
    config_path: str | Path,
    *,
    grid_side: int,
    seed: int,
    device: str,
    T: int = 1,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    cfg = load_yaml_config(path)
    sw = copy.deepcopy(cfg.get("swing_equation") or {})
    exp = copy.deepcopy(cfg.get("experiment") or {})
    top = {k: cfg[k] for k in ("N", "N_global", "model_type") if k in cfg}
    return {
        "config_path": str(path),
        "top_level": top,
        "swing_equation": sw,
        "experiment": exp,
        "discrete_bayes_single_step": {
            "T": int(T),
            "grid_side": int(grid_side),
            "n_support": int(grid_side) ** 2,
            "prior_on_theta": "uniform on (M,K) tensor-product grid in [M_lower,M_upper]×[K_lower,K_upper]",
            "seed": int(seed),
            "device": str(device),
            "note": (
                "T is the sBOED sequential experiment length (probe–observe rounds). "
                "experiment.update_count is the baseline evaluation horizon, not T."
            ),
        },
    }


def swing_physics_kwargs_from_yaml(config_path: str | Path) -> dict[str, Any]:
    cfg = load_yaml_config(config_path)
    sw = cfg.get("swing_equation") or {}
    probe_duration = float(sw.get("probe_duration", 2.0))
    amps = list(sw.get("probe_amplitudes") or [0.2])
    mid = len(amps) // 2
    amp = float(amps[mid])
    xi = (4, amp, probe_duration)
    ode_horizon = float(sw.get("T_obs_sec", 3.0))
    ode_timeout = max(15.0, 3.0 * ode_horizon)
    return {
        "ode_horizon_sec": ode_horizon,
        "ode_timeout": ode_timeout,
        "sigma_feat": float(sw.get("sigma", 0.05)),
        "xi": xi,
        "r_max": float(sw.get("r_max", 0.1)),
        "f_min": float(sw.get("f_min", 49.8)),
        "reference_probe_bus": int(sw.get("reference_probe_bus", 3)),
        "reference_probe_amplitude": float(sw.get("reference_probe_amplitude", 0.2)),
        "reference_probe_duration": float(sw.get("reference_probe_duration", 2.0)),
    }


def _synthetic_grid(grid_side: int = 3) -> tuple[np.ndarray, int]:
    """Fixed (M,K) support on a grid (discrete θ_n)."""
    Ms = np.linspace(0.02, 0.05, grid_side)
    Ks = np.linspace(0.1, 0.4, grid_side)
    grid = np.array([(M, K) for M in Ms for K in Ks])
    return grid, len(grid)


def _map_surrogate(theta_m: float, theta_k: float) -> float:
    """Surrogate Map(θ, ξ) for synthetic runs (ROCOF_max scale); ξ fixed in demo."""
    return 0.3 + 8.0 * theta_m + 0.5 * theta_k


def _gamma_star_surrogate(theta_m: float, theta_k: float) -> float:
    """Surrogate γ*(θ) for synthetic runs (evaluation model stand-in)."""
    return 5.0 + 120.0 * theta_m + 15.0 * theta_k


def run_synthetic_episode(
    T: int = DEFAULT_T,
    seed: int = 0,
    grid_side: int = 3,
    sigma_feat: float = 0.05,
    log_p0: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Alg. 1-style loop with **surrogate** Map / γ* (no ODE).

    This is **not** the one-step reduction of :func:`run_physics_episode`; it is a cheap
    stand-in for tests. For the real single-step body, use ``run_physics_episode(...,
    T=1)`` or :func:`run_single_step_physics_episode`.

    ``log_p0``: optional length-``N`` log prior on the support (default uniform).
    Pass the **same** ``log_p0`` to ``single_step_discrete_bayes_report`` when testing T=1.
    """
    from src.core.discrete_bayes import (
        mocu_gamma_star,
        posterior_after_sequential_gaussian_observations,
    )

    rng = np.random.default_rng(seed)
    grid, n = _synthetic_grid(grid_side)
    M_true, K_true = 0.035, 0.25
    xi = (4, 0.2, 2.0)

    mu_row = np.array(
        [_map_surrogate(float(M), float(K)) for M, K in grid],
        dtype=np.float64,
    )
    gamma_star_n = np.array(
        [_gamma_star_surrogate(float(M), float(K)) for M, K in grid],
        dtype=np.float64,
    )

    mu_steps = np.tile(mu_row, (T, 1))
    y_steps = np.array(
        [
            _map_surrogate(M_true, K_true) + float(rng.normal(0.0, sigma_feat))
            for _ in range(T)
        ],
        dtype=np.float64,
    )

    _, p_trace = posterior_after_sequential_gaussian_observations(
        mu_steps, y_steps, sigma_feat, log_p0=log_p0
    )

    mocu_trace: list[float] = []
    ghat_trace: list[float] = []
    for p in p_trace:
        m, gh = mocu_gamma_star(p, gamma_star_n)
        mocu_trace.append(m)
        ghat_trace.append(gh)

    return {
        "mode": "synthetic",
        "T": T,
        "n": n,
        "grid": grid,
        "mu_predictions": mu_steps,
        "xi": xi,
        "theta_true": (M_true, K_true),
        "sigma_feat": sigma_feat,
        "y_steps": y_steps,
        "p_trace": p_trace,
        "mocu_trace": mocu_trace,
        "ghat_trace": ghat_trace,
        "gamma_star_support": gamma_star_n,
    }


def run_physics_episode(
    T: int = DEFAULT_T,
    seed: int = 42,
    grid_side: int = QUICK_PHYSICS_GRID,
    device: str = "cpu",
    ode_horizon_sec: float = ODE_HORIZON_QUICK,
    ode_timeout: float = QUICK_PHYSICS_TIMEOUT,
    *,
    sigma_feat: float | None = None,
    xi: tuple | None = None,
    r_max: float | None = None,
    f_min: float | None = None,
    reference_probe_bus: int | None = None,
    reference_probe_amplitude: float | None = None,
    reference_probe_duration: float | None = None,
) -> dict[str, Any]:
    """
    IEEE-14 physics body: precompute ``γ*(θ_n)``, ``μ_n = Map(θ_n,ξ)``, then sequential
    Gaussian likelihood updates and MOCU on the support.

    **T** is the sBOED sequential experiment length (probe–observe rounds). **T = 1** is the
    single-step test—same pipeline as multi-step, one observation and one posterior update
    (see :func:`run_single_step_physics_episode` for T = 1 plus a detailed report).
    """
    from src.core.discrete_bayes import (
        log_prior_uniform_discrete,
        mocu_gamma_star,
        sequential_posterior_from_log_likelihoods,
    )
    from src.core.discrete_bayes import log_gaussian_observation_density
    from src.core.likelihood import DEFAULT_SIGMA, mu_theta_xi, mu_theta_xi_batch
    from src.core.swing_equation_mocu import binary_search_gamma_star_batch
    from src.core.swing_equation_params import get_default_swing_equation_params

    rng = np.random.default_rng(seed)
    params = get_default_swing_equation_params(
        N=14,
        topology="ieee14",
        coupling_strength=1.0,
        damping=0.1,
        base_power=1.0,
        M_lower=0.01,
        M_upper=0.06,
        K_lower=0.05,
        K_upper=0.50,
    )
    B, P_m, D, g = params["B"], params["P_m"], params["D"], params["g"]
    M_lo, M_hi = float(params["M_lower"]), float(params["M_upper"])
    K_lo, K_hi = float(params["K_lower"]), float(params["K_upper"])

    Ms = np.linspace(M_lo, M_hi, grid_side)
    Ks = np.linspace(K_lo, K_hi, grid_side)
    grid = np.array([(M, K) for M in Ms for K in Ks])
    n = len(grid)
    M_true = float(rng.uniform(M_lo, M_hi))
    K_true = float(rng.uniform(K_lo, K_hi))

    h = 1.0 / 160.0
    M_steps = int(ode_horizon_sec / h)
    sigma_feat = DEFAULT_SIGMA if sigma_feat is None else float(sigma_feat)
    xi = (4, 0.2, 2.0) if xi is None else xi
    ref_bus = 3 if reference_probe_bus is None else int(reference_probe_bus)
    ref_amp = 0.2 if reference_probe_amplitude is None else float(reference_probe_amplitude)
    ref_dur = 2.0 if reference_probe_duration is None else float(reference_probe_duration)
    r_max_v = 0.1 if r_max is None else float(r_max)
    f_min_v = 49.8 if f_min is None else float(f_min)

    gamma_star_n = binary_search_gamma_star_batch(
        grid[:, 0],
        grid[:, 1],
        B,
        P_m,
        D,
        g,
        r_max=r_max_v,
        f_min=f_min_v,
        h=h,
        T=ode_horizon_sec,
        M_steps=M_steps,
        reference_probe_bus=ref_bus,
        reference_probe_amplitude=ref_amp,
        reference_probe_duration=ref_dur,
        device=device,
    )

    y_clean = mu_theta_xi(
        (M_true, K_true),
        xi,
        B,
        P_m,
        D,
        g,
        h=h,
        T=ode_horizon_sec,
        M_steps=M_steps,
        device=device,
        timeout=ode_timeout,
        T_obs_sec=ode_horizon_sec,
    )

    mu_map = mu_theta_xi_batch(
        grid,
        xi,
        B,
        P_m,
        D,
        g,
        h=h,
        T=ode_horizon_sec,
        M_steps=M_steps,
        device=device,
        timeout=ode_timeout,
        T_obs_sec=ode_horizon_sec,
    )

    log_L_steps = np.zeros((T, n), dtype=np.float64)
    y_steps = np.zeros(T, dtype=np.float64)
    for t in range(T):
        y_steps[t] = float(y_clean + rng.normal(0.0, sigma_feat))
        log_L_steps[t] = log_gaussian_observation_density(
            float(y_steps[t]), mu_map, sigma_feat
        )

    log_p0 = log_prior_uniform_discrete(n)
    _, p_trace = sequential_posterior_from_log_likelihoods(log_L_steps, log_p0)

    mocu_trace = []
    ghat_trace = []
    for p in p_trace:
        m, gh = mocu_gamma_star(p, gamma_star_n)
        mocu_trace.append(m)
        ghat_trace.append(gh)

    return {
        "mode": "physics",
        "T": T,
        "ode_horizon_sec": ode_horizon_sec,
        "n": n,
        "grid": grid,
        "mu_predictions": np.tile(mu_map, (T, 1)),
        "grid_side": grid_side,
        "xi": xi,
        "theta_true": (M_true, K_true),
        "sigma_feat": sigma_feat,
        "y_clean": float(y_clean),
        "y_steps": y_steps,
        "p_trace": p_trace,
        "mocu_trace": mocu_trace,
        "ghat_trace": ghat_trace,
        "gamma_star_support": gamma_star_n,
        "physics_meta": {
            "sigma_feat": sigma_feat,
            "xi": xi,
            "reference_probe_bus": ref_bus,
            "reference_probe_amplitude": ref_amp,
            "reference_probe_duration": ref_dur,
            "r_max": r_max_v,
            "f_min": f_min_v,
            "ode_horizon_sec": ode_horizon_sec,
        },
    }


def run_single_step_physics_episode(
    seed: int = 42,
    grid_side: int = 8,
    device: str | None = None,
    config_path: str | Path | None = None,
    *,
    sigma_feat: float | None = None,
) -> dict[str, Any]:
    """
    **T = 1** (single sBOED step): :func:`run_physics_episode` with ``T=1``, plus Bayes report
    and ``γ*(θ_true)`` / ``u_ctrl``. Swing / likelihood / reference probe come **only** from
    ``config_path`` (default ``config/early_test.yaml`` via :func:`swing_physics_kwargs_from_yaml`).

    ``device``: pass ``\"cpu\"`` / ``\"cuda\"`` or ``None`` for :func:`resolve_inference_device`.

    ``sigma_feat``: if set, overrides YAML ``swing_equation.sigma`` for the Gaussian likelihood
    (and synthetic observation noise). Larger values soften the update and typically keep
    posterior MOCU away from numerical zero; smaller values sharpen the posterior.
    """
    dev = resolve_inference_device(device)
    path = Path(config_path) if config_path else DEFAULT_SWING_YAML
    merged = swing_physics_kwargs_from_yaml(path)
    ode_horizon_sec = float(merged.pop("ode_horizon_sec"))
    ode_timeout = float(merged.pop("ode_timeout"))
    if sigma_feat is not None:
        merged["sigma_feat"] = float(sigma_feat)

    from src.core.discrete_bayes import (
        log_prior_uniform_discrete,
        single_step_discrete_bayes_report,
    )
    from src.core.swing_equation_mocu import binary_search_gamma_star
    from src.core.swing_equation_ode import solve_swing_equation_ode
    from src.core.swing_equation_params import get_default_swing_equation_params

    base = run_physics_episode(
        T=1,
        seed=seed,
        grid_side=grid_side,
        device=dev,
        ode_horizon_sec=ode_horizon_sec,
        ode_timeout=ode_timeout,
        **merged,
    )

    n = int(base["n"])
    log_p0 = log_prior_uniform_discrete(n)
    mu_row = base["mu_predictions"][0]
    y = float(base["y_steps"][0])
    sigma_feat = float(base["sigma_feat"])
    gamma_star_n = base["gamma_star_support"]

    single_step_report = single_step_discrete_bayes_report(
        y, mu_row, sigma_feat, gamma_star_n, log_p0=log_p0
    )

    params = get_default_swing_equation_params(
        N=14,
        topology="ieee14",
        coupling_strength=1.0,
        damping=0.1,
        base_power=1.0,
        M_lower=0.01,
        M_upper=0.06,
        K_lower=0.05,
        K_upper=0.50,
    )
    B = params["B"]
    P_m = params["P_m"]
    D = float(params["D"])
    g = params["g"]
    M_true, K_true = base["theta_true"]

    meta = base["physics_meta"]
    h = 1.0 / 160.0
    M_steps = int(float(meta["ode_horizon_sec"]) / h)
    r_max = float(meta["r_max"])
    f_min = float(meta["f_min"])
    ref_bus = int(meta["reference_probe_bus"])
    ref_amp = float(meta["reference_probe_amplitude"])
    ref_dur = float(meta["reference_probe_duration"])
    probe_bus_internal = (ref_bus - 1) if ref_bus >= 1 else ref_bus

    gamma_star_true = binary_search_gamma_star(
        B,
        P_m,
        D,
        M_true,
        K_true,
        g,
        r_max=r_max,
        f_min=f_min,
        h=h,
        T=float(meta["ode_horizon_sec"]),
        M_steps=M_steps,
        reference_probe_bus=ref_bus,
        reference_probe_amplitude=ref_amp,
        reference_probe_duration=ref_dur,
        device=dev,
    )

    state = solve_swing_equation_ode(
        B,
        P_m,
        D,
        M_true,
        K_true,
        g,
        gamma=float(gamma_star_true),
        probe_bus=probe_bus_internal,
        probe_amplitude=ref_amp,
        probe_duration=ref_dur,
        h=h,
        M_steps=M_steps,
        T=float(meta["ode_horizon_sec"]),
        device=dev,
        timeout=ode_timeout,
    )
    n_bus = len(P_m)
    omega = state[:, n_bus:]
    g_row = np.asarray(g, dtype=np.float64).reshape(1, -1)
    u_ctrl = -float(gamma_star_true) * g_row * omega

    result = {
        **base,
        "log_p0": log_p0,
        "single_step_report": single_step_report,
        "gamma_star_true": float(gamma_star_true),
        "u_ctrl_trajectory": u_ctrl,
        "u_ctrl_max_abs": float(np.max(np.abs(u_ctrl))),
        "u_ctrl_reference_probe": (ref_bus, ref_amp, ref_dur),
        "device": dev,
    }
    return result


def main_body_single_step_simulation(
    seed: int = 42,
    grid_side: int = 8,
    device: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    **T = 1** (single sBOED step) slice of the swing experiment-design core: same machinery as multi-step
    :func:`run_physics_episode`, staged for readability. **Omits** policy/DAD training,
    baseline comparisons, and evaluation harnesses — only the physics + discrete Bayes + MOCU chain.

    Pipeline (order):

    1. **ξ** — probe design ``(b, A, T_p)`` (1-based bus); ``Map`` uses ``gamma=None`` (no
       planning control in the measurement simulator; see ``likelihood.mu_theta_xi``).
    2. **γ*(θ_n)** on the discrete support — binary search under the **reference** contingency
       (same as ``binary_search_gamma_star_batch`` in :func:`run_physics_episode`).
    3. **Observation** — ``y = μ(θ_true, ξ) + ε`` with ``μ_n = Map(θ_n, ξ)`` and Gaussian
       ``ε`` (``sigma_feat``); ``y_clean = μ(θ_true, ξ)``.
    4. **Posterior** — discrete Bayes update ``p(θ|y)`` (``single_step_discrete_bayes_report``).
    5. **Evaluation (safety)** — ``γ*(θ_true)`` and ``u_ctrl = -γ g ⊙ ω`` from a forward ODE
       at that γ (reference probe); this is the **evaluation** control, not the probe Map.
    6. **MOCU** — ``MOCU(p)`` and ``γ̂`` before/after the update on ``{γ*(θ_n)}``.

    Returns a nested dict; the full helper output is under ``"raw"`` for tests that need
    arrays (``p_trace``, ``mu_predictions``, …).
    """
    raw = run_single_step_physics_episode(
        seed=seed,
        grid_side=grid_side,
        device=device,
        config_path=config_path,
    )
    rep = raw["single_step_report"]
    y = float(rep["y"])
    y_clean = float(raw["y_clean"])
    return {
        "xi": raw["xi"],
        "grid": raw["grid"],
        "theta_true": raw["theta_true"],
        "gamma_star_support": raw["gamma_star_support"],
        "map_likelihood": {
            "mu_n": rep["mu"].copy(),
            "y_clean": y_clean,
            "y": y,
            "sigma_feat": float(rep["sigma_feat"]),
            "noise": y - y_clean,
        },
        "posterior": {
            "p0": rep["p0"].copy(),
            "p1": rep["p1"].copy(),
            "log_Z": float(rep["log_Z"]),
            "single_step_report": rep,
        },
        "evaluation_control": {
            "gamma_star_true": raw["gamma_star_true"],
            "u_ctrl_trajectory": raw["u_ctrl_trajectory"],
            "u_ctrl_max_abs": raw["u_ctrl_max_abs"],
            "u_ctrl_reference_probe": raw["u_ctrl_reference_probe"],
        },
        "mocu": {
            "mocu_prior": float(rep["mocu_prior"]),
            "gamma_hat_prior": float(rep["gamma_hat_prior"]),
            "mocu_post": float(rep["mocu_post"]),
            "gamma_hat_post": float(rep["gamma_hat_post"]),
        },
        "raw": raw,
    }
