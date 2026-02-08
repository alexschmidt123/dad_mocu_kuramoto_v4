"""
PI-ready test suite: Design ξ → forward simulation (2nd-order Kuramoto / IEEE-14) → observation y (ROCOF) → prior→posterior sharpening.

NO MOCU predictor, NO DAD policy, NO sequential decisions.
All outputs (tables, plots) under tests/output/.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.core.swing_equation_ode import solve_swing_equation_ode, extract_frequency_features
from src.core.rocof import extract_max_rocof
from src.core.likelihood import log_likelihood_batch
from src.core.swing_equation_params import generate_ieee14_coupling_matrix
from src.core.probe_signal import hann_window

# Optional fallback for obs only (not for plots)
try:
    from scripts.data_generation.generate_dad_data import perform_probe_experiment
except ImportError:
    perform_probe_experiment = None

# Outputs must stay under tests/
_TESTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _TESTS_DIR / "output"
_resolved = OUTPUT_DIR.resolve()
_tests_resolved = _TESTS_DIR.resolve()
assert _resolved == _tests_resolved or str(_resolved).startswith(str(_tests_resolved) + os.sep)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_design(B, P_m, D, g, M_true, K_true, probe_bus, probe_amplitude,
                     probe_duration, h, T, device="cpu", timeout=10.0, use_fallback=False):
    """
    Run one probe experiment. probe_bus is B in {1..14} (1-based). Returns obs and omega_traj.
    Converts B to 0-based for ODE internally.
    """
    M_steps = int(round(T / h))
    N = len(P_m)
    probe_bus_internal = (probe_bus - 1) if probe_bus >= 1 else probe_bus
    try:
        state_traj = solve_swing_equation_ode(
            B, P_m, D, M_true, K_true, g,
            probe_bus=probe_bus_internal,
            probe_amplitude=probe_amplitude,
            probe_duration=probe_duration,
            h=h, M_steps=M_steps, T=T,
            device=device, timeout=timeout,
        )
        omega_traj = state_traj[:, N:]
        rocof_max = extract_max_rocof(
            omega_traj, fs=12.0, window_sec=min(10.0, T), h=h
        )
        obs = extract_frequency_features(omega_traj, h, fs=12.0)
        obs["ROCOF_max"] = rocof_max
        return obs, omega_traj
    except Exception as e:
        if use_fallback and perform_probe_experiment is not None:
            obs = perform_probe_experiment(
                B, P_m, D, M_true, K_true, g,
                probe_bus=probe_bus_internal,
                probe_amplitude=probe_amplitude,
                probe_duration=probe_duration,
                h=h, T=T, M_steps=M_steps,
                device=device, timeout=timeout,
            )
            return obs, None
        raise RuntimeError(f"ODE solve failed (use_fallback={use_fallback}): {e}") from e


def compute_rocof_timeseries(omega_traj, h, fs=12.0):
    """
    ROCOF time-series consistent with src/core/rocof.py and design_part1.tex:
    Δf = ω/(2π), downsample to fs using h, then ROCOF = diff(Δf)/dt with dt=1/fs.
    Return time array and max |ROCOF| across buses at each time (Hz/s).
    """
    M, N = omega_traj.shape
    dt = 1.0 / fs
    if h > 0 and (1.0 / h) > fs:
        downsample = max(1, int(round((1.0 / h) / fs)))
        indices = np.arange(0, M, downsample)
        omega_ds = omega_traj[indices, :]
    else:
        omega_ds = omega_traj
    delta_f = omega_ds / (2.0 * np.pi)
    rocof_series = np.diff(delta_f, axis=0) / dt
    rocof_max_over_buses = np.max(np.abs(rocof_series), axis=1)
    t_rocof = np.arange(rocof_max_over_buses.shape[0], dtype=np.float64) * dt
    return t_rocof, rocof_max_over_buses


def posterior_on_grid(y_obs, xi, M_lower, M_upper, K_lower, K_upper, n_grid, sigma,
                      B, P_m, D, g, h, T, device="cpu", timeout=10.0):
    """
    Build uniform grid over (M,K), compute log likelihood with log_likelihood_batch,
    normalize to posterior p(M,K|y,ξ). Returns p_grid [n_grid, n_grid], M_vals, K_vals.
    """
    M_steps = int(round(T / h))
    M_vals = np.linspace(M_lower, M_upper, n_grid)
    K_vals = np.linspace(K_lower, K_upper, n_grid)
    MM, KK = np.meshgrid(M_vals, K_vals, indexing="ij")
    thetas = np.column_stack([MM.ravel(), KK.ravel()])
    try:
        logp = log_likelihood_batch(
            float(y_obs), thetas, xi, sigma,
            B, P_m, D, g,
            h=h, T=T, M_steps=M_steps,
            fs=12.0, device=device, timeout=timeout,
            rocof_method="full_window", T_obs_sec=min(10.0, T),
        )
    except Exception as e:
        raise RuntimeError(f"posterior_on_grid failed: {e}") from e
    logp = np.clip(logp, -1e10, None)
    p = np.exp(logp - logp.max())
    p = p / p.sum()
    p_grid = p.reshape(n_grid, n_grid)
    return p_grid, M_vals, K_vals


def marginal_M(p_grid, M_vals):
    """
    pM = sum over K axis; normalize. Prior on grid is uniform: pM_prior = 1/n_grid each.
    Returns pM_prior, pM_post, var_M_post.
    """
    n_grid = p_grid.shape[0]
    pM_post = np.sum(p_grid, axis=1)
    pM_post = pM_post / pM_post.sum()
    pM_prior = np.ones(n_grid) / n_grid
    E_M = np.sum(pM_post * M_vals)
    var_M_post = np.sum(pM_post * (M_vals - E_M) ** 2)
    return pM_prior, pM_post, var_M_post


def marginal_K(p_grid, K_vals):
    """
    pK = sum over M axis; normalize. Prior on grid is uniform.
    Returns pK_prior, pK_post, var_K_post.
    """
    n_grid = p_grid.shape[1]
    pK_post = np.sum(p_grid, axis=0)
    pK_post = pK_post / pK_post.sum()
    pK_prior = np.ones(n_grid) / n_grid
    E_K = np.sum(pK_post * K_vals)
    var_K_post = np.sum(pK_post * (K_vals - E_K) ** 2)
    return pK_prior, pK_post, var_K_post


def test_single_design_produces_observation(ieee14_params, prior_bounds, simulation_settings):
    """Run 1 design (bus=0, amp=0.2) using ODE. Assert obs has ROCOF_max, f_min; omega_traj shape[1]==14."""
    B = ieee14_params["B"]
    P_m = ieee14_params["P_m"]
    D = ieee14_params["D"]
    g = ieee14_params["g"]
    M_true = prior_bounds["M_true"]
    K_true = prior_bounds["K_true"]
    h = simulation_settings["h"]
    T = simulation_settings["T"]
    device = simulation_settings["device"]
    timeout = simulation_settings["timeout"]
    probe_duration = simulation_settings["probe_duration"]

    obs, omega_traj = run_single_design(
        B, P_m, D, g, M_true, K_true,
        probe_bus=1,
        probe_amplitude=0.2,
        probe_duration=probe_duration,
        h=h, T=T, device=device, timeout=timeout,
        use_fallback=False,
    )
    assert isinstance(obs, dict)
    assert "ROCOF_max" in obs
    assert "f_min" in obs
    assert obs["ROCOF_max"] >= 0
    assert omega_traj is not None
    assert omega_traj.ndim == 2
    assert omega_traj.shape[1] == 14


def test_design_comparison_table_saved(ieee14_params, prior_bounds, simulation_settings, design_candidates):
    """Loop design_candidates; run_single_design, posterior_on_grid (n_grid=7), info_gain; save CSV. Sanity: >=2 distinct ROCOF_max."""
    B = ieee14_params["B"]
    P_m = ieee14_params["P_m"]
    D = ieee14_params["D"]
    g = ieee14_params["g"]
    M_true = prior_bounds["M_true"]
    K_true = prior_bounds["K_true"]
    M_lower = prior_bounds["M_lower"]
    M_upper = prior_bounds["M_upper"]
    K_lower = prior_bounds["K_lower"]
    K_upper = prior_bounds["K_upper"]
    h = simulation_settings["h"]
    T = simulation_settings["T"]
    device = simulation_settings["device"]
    timeout = simulation_settings["timeout"]
    probe_duration = simulation_settings["probe_duration"]
    n_grid = 7
    sigma = 0.05

    rows = []
    for (probe_bus, probe_amplitude) in design_candidates:
        obs, _ = run_single_design(
            B, P_m, D, g, M_true, K_true,
            probe_bus=probe_bus,
            probe_amplitude=probe_amplitude,
            probe_duration=probe_duration,
            h=h, T=T, device=device, timeout=timeout,
            use_fallback=True,
        )
        y_obs = obs.get("ROCOF_max", 0.0)
        xi = (probe_bus, probe_amplitude, probe_duration)
        try:
            p_grid, M_vals, K_vals = posterior_on_grid(
                y_obs, xi, M_lower, M_upper, K_lower, K_upper, n_grid, sigma,
                B, P_m, D, g, h, T, device=device, timeout=timeout,
            )
            _, _, var_M_post = marginal_M(p_grid, M_vals)
            _, _, var_K_post = marginal_K(p_grid, K_vals)
            # Discrete entropy: H = -sum p log p; prior uniform H_prior = log(n_points)
            n_points = p_grid.size
            p_flat = p_grid.ravel()
            p_flat = p_flat / p_flat.sum()
            p_pos = p_flat[p_flat > 0]
            H_post = -np.sum(p_pos * np.log(p_pos))
            H_prior = np.log(n_points)
            info_gain = H_prior - H_post
        except Exception:
            var_M_post = np.nan
            var_K_post = np.nan
            info_gain = np.nan
        rows.append({
            "bus": probe_bus,
            "amplitude": probe_amplitude,
            "ROCOF_max": obs.get("ROCOF_max", np.nan),
            "f_min": obs.get("f_min", np.nan),
            "var_M_post": var_M_post,
            "var_K_post": var_K_post,
            "info_gain": info_gain,
        })

    # Save CSV
    table_path = OUTPUT_DIR / "design_comparison_table.csv"
    with open(table_path, "w") as f:
        f.write("bus,amplitude,ROCOF_max,f_min,var_M_post,var_K_post,info_gain\n")
        for r in rows:
            v_m = r["var_M_post"]
            v_k = r["var_K_post"]
            i = r["info_gain"]
            v_m_str = f"{v_m:.10g}" if not np.isnan(v_m) else "nan"
            v_k_str = f"{v_k:.10g}" if not np.isnan(v_k) else "nan"
            i_str = f"{i:.10g}" if not np.isnan(i) else "nan"
            f.write(f"{r['bus']},{r['amplitude']},{r['ROCOF_max']:.6f},{r['f_min']:.6f},{v_m_str},{v_k_str},{i_str}\n")

    assert len(rows) == 140, "Table must have 140 rows (14 B × 10 A)"
    rocofs = [r["ROCOF_max"] for r in rows]
    assert len(set(np.round(np.asarray(rocofs), 6))) >= 2, "Expected at least 2 distinct ROCOF_max across designs"


# Design sets for ROCOF plots: B in {1,4,7,10,13,14}, A in 6 values (aligned with Parameter_references_table.md: tests up to 0.5)
ROCOF_BUSES = [1, 4, 7, 10, 13, 14]
ROCOF_AMPLITUDES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def _highlight_probe_interval(ax, probe_duration, T):
    """Blue shaded region and text for ROCOF plots: first 2 s is probing time. No legend entry."""
    ax.axvspan(0, probe_duration, alpha=0.25, color="steelblue", zorder=0)
    ax.axvline(probe_duration, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    # Label centered over blue region, fixed at top of axes
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(probe_duration * 0.5, 0.96, "First 2 s: probing time", transform=trans,
            ha="center", va="top", fontsize=8, color="navy", alpha=0.95,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))


def test_rocof_timeseries_by_bus_plot(ieee14_params, prior_bounds, simulation_settings):
    """Same A for different B: 6 subplots (one per A), 6 curves (B=1,4,7,10,13,14) each. Highlight probe 0..2 s."""
    if not HAS_MATPLOTLIB:
        pytest.skip("matplotlib required for plots")
    B = ieee14_params["B"]
    P_m = ieee14_params["P_m"]
    D = ieee14_params["D"]
    g = ieee14_params["g"]
    M_true = prior_bounds["M_true"]
    K_true = prior_bounds["K_true"]
    h = simulation_settings["h"]
    T = simulation_settings["T"]
    device = simulation_settings["device"]
    timeout = simulation_settings["timeout"]
    probe_duration = simulation_settings["probe_duration"]
    amplitudes = ROCOF_AMPLITUDES
    buses = ROCOF_BUSES

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=False)
    axes = axes.flatten()
    for idx, A in enumerate(amplitudes):
        ax = axes[idx]
        _highlight_probe_interval(ax, probe_duration, T)
        for bus in buses:
            obs, omega_traj = run_single_design(
                B, P_m, D, g, M_true, K_true,
                probe_bus=bus,
                probe_amplitude=A,
                probe_duration=probe_duration,
                h=h, T=T, device=device, timeout=timeout,
                use_fallback=False,
            )
            assert omega_traj is not None
            t, rocof = compute_rocof_timeseries(omega_traj, h)
            ax.plot(t, rocof, label=f"B={bus}", zorder=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ROCOF (Hz/s)")
        ax.set_title(f"A={A}")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.suptitle(f"ROCOF(t): same A, different B (probe on 0..{probe_duration} s, T={T} s)")
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])  # tight; legend on the right in small margin
    fig.savefig(OUTPUT_DIR / "rocof_timeseries_by_bus.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def test_rocof_timeseries_by_amplitude_plot(ieee14_params, prior_bounds, simulation_settings):
    """Same B for different A: 6 subplots (one per B in {1,4,7,10,13,14}), 6 curves (A values) each. Highlight probe 0..2 s."""
    if not HAS_MATPLOTLIB:
        pytest.skip("matplotlib required for plots")
    B = ieee14_params["B"]
    P_m = ieee14_params["P_m"]
    D = ieee14_params["D"]
    g = ieee14_params["g"]
    M_true = prior_bounds["M_true"]
    K_true = prior_bounds["K_true"]
    h = simulation_settings["h"]
    T = simulation_settings["T"]
    device = simulation_settings["device"]
    timeout = simulation_settings["timeout"]
    probe_duration = simulation_settings["probe_duration"]
    buses = ROCOF_BUSES
    amplitudes = ROCOF_AMPLITUDES

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=False)
    axes = axes.flatten()
    for idx, bus in enumerate(buses):
        ax = axes[idx]
        _highlight_probe_interval(ax, probe_duration, T)
        for A in amplitudes:
            obs, omega_traj = run_single_design(
                B, P_m, D, g, M_true, K_true,
                probe_bus=bus,
                probe_amplitude=A,
                probe_duration=probe_duration,
                h=h, T=T, device=device, timeout=timeout,
                use_fallback=False,
            )
            assert omega_traj is not None
            t, rocof = compute_rocof_timeseries(omega_traj, h)
            ax.plot(t, rocof, label=f"A={A}", zorder=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ROCOF (Hz/s)")
        ax.set_title(f"B={bus}")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.suptitle(f"ROCOF(t): same B, different A (probe on 0..{probe_duration} s, T={T} s)")
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])  # tight; legend on the right in small margin
    fig.savefig(OUTPUT_DIR / "rocof_timeseries_by_amplitude.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def test_posterior_sharpens_plot(ieee14_params, prior_bounds, simulation_settings):
    """2–3 designs: plot prior vs posterior for both M and K. Y-axis = probability density (1/width for prior)."""
    if not HAS_MATPLOTLIB:
        pytest.skip("matplotlib required for plots")
    B = ieee14_params["B"]
    P_m = ieee14_params["P_m"]
    D = ieee14_params["D"]
    g = ieee14_params["g"]
    M_true = prior_bounds["M_true"]
    K_true = prior_bounds["K_true"]
    M_lower = prior_bounds["M_lower"]
    M_upper = prior_bounds["M_upper"]
    K_lower = prior_bounds["K_lower"]
    K_upper = prior_bounds["K_upper"]
    h = simulation_settings["h"]
    T = simulation_settings["T"]
    device = simulation_settings["device"]
    timeout = simulation_settings["timeout"]
    probe_duration = simulation_settings["probe_duration"]
    n_grid = 55  # 5–10x resolution for smooth marginal curves (was 9)
    sigma = 0.05

    designs = [
        (1, 0.3, "design1: A=0.3, B=1"),
        (7, 0.3, "design2: A=0.3, B=7"),
        (14, 0.3, "design3: A=0.3, B=14"),
        (7, 0.1, "design4: A=0.1, B=7"),
        (7, 0.5, "design5: A=0.5, B=7"),
    ]

    # Prior variance: continuous uniform Var = (b-a)^2/12
    var_M_prior = ((M_upper - M_lower) ** 2) / 12.0
    var_K_prior = ((K_upper - K_lower) ** 2) / 12.0
    # Bin widths = grid spacing (for density: mass / width)
    dM = (M_upper - M_lower) / (n_grid - 1)
    dK = (K_upper - K_lower) / (n_grid - 1)
    prior_density_M = 1.0 / (M_upper - M_lower)
    prior_density_K = 1.0 / (K_upper - K_lower)

    fig, (ax_m, ax_k) = plt.subplots(1, 2, figsize=(10, 4))

    at_least_one_sharper = False
    for (probe_bus, probe_amplitude, label) in designs:
        obs, _ = run_single_design(
            B, P_m, D, g, M_true, K_true,
            probe_bus=probe_bus,
            probe_amplitude=probe_amplitude,
            probe_duration=probe_duration,
            h=h, T=T, device=device, timeout=timeout,
            use_fallback=True,
        )
        y_obs = obs.get("ROCOF_max", 0.0)
        xi = (probe_bus, probe_amplitude, probe_duration)
        p_grid, M_vals, K_vals = posterior_on_grid(
            y_obs, xi, M_lower, M_upper, K_lower, K_upper, n_grid, sigma,
            B, P_m, D, g, h, T, device=device, timeout=timeout,
        )
        pM_prior, pM_post, var_M_post = marginal_M(p_grid, M_vals)
        pK_prior, pK_post, var_K_post = marginal_K(p_grid, K_vals)
        if var_M_post < var_M_prior or var_K_post < var_K_prior:
            at_least_one_sharper = True
        # Plot probability density: posterior_density = p_post / bin_width
        ax_m.plot(M_vals, pM_post / dM, label=label)
        ax_k.plot(K_vals, pK_post / dK, label=label)

    # Prior: uniform density over [lower, upper]
    ax_m.hlines(prior_density_M, M_lower, M_upper, colors="gray", linestyles="--", label="Prior")
    ax_m.set_xlabel("M (inertia)")
    ax_m.set_ylabel("Probability density")
    ax_m.set_title("Prior vs posterior p(M|y,ξ)")
    ax_m.grid(True, alpha=0.3)
    ax_m.set_ylim(bottom=0)

    ax_k.hlines(prior_density_K, K_lower, K_upper, colors="gray", linestyles="--", label="Prior")
    ax_k.set_xlabel("K (gain)")
    ax_k.set_ylabel("Probability density")
    ax_k.set_title("Prior vs posterior p(K|y,ξ)")
    ax_k.grid(True, alpha=0.3)
    ax_k.set_ylim(bottom=0)

    handles, labels = ax_m.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)
    fig.suptitle("Prior vs posterior marginals for different designs")
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])  # tight; legend on the right in small margin
    fig.savefig(OUTPUT_DIR / "posterior_marginals_by_design.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    assert at_least_one_sharper, "Posterior variance < prior variance for at least one design (M or K)"


def test_ieee14_diagram_plot():
    """Draw IEEE 14-bus network diagram (same topology as in project and papers). Save to tests/output/ieee14_diagram.png."""
    if not HAS_MATPLOTLIB:
        pytest.skip("matplotlib required for plots")
    # Same coupling matrix as used in swing equation and design_part1.tex / published work
    B = generate_ieee14_coupling_matrix(1.0)
    N = 14
    # Edges from B (1-based bus labels for display)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if B[i, j] != 0:
                edges.append((i + 1, j + 1))
    # Layout: IEEE 14 single-line style (standard test case topology)
    pos = {
        1: (0, 2),
        2: (1, 2.5),
        3: (1, 1.5),
        4: (1.5, 2),
        5: (0.5, 1),
        6: (1.5, 1),
        7: (2.5, 2.5),
        8: (3, 2.5),
        9: (2.5, 1.5),
        10: (3, 1.5),
        11: (2.5, 0.5),
        12: (2, 0),
        13: (2.5, 0),
        14: (3, 0),
    }
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    for (i, j) in edges:
        xi, yi = pos[i]
        xj, yj = pos[j]
        ax.plot([xi, xj], [yi, yj], "k-", linewidth=2, zorder=0)
    # Larger circles and numbers for readability
    node_markersize = 22
    for bus in range(1, N + 1):
        x, y = pos[bus]
        ax.plot(x, y, "o", markersize=node_markersize, color="steelblue", markeredgecolor="navy", markeredgewidth=2, zorder=1)
        ax.text(x, y, str(bus), ha="center", va="center", fontsize=14, fontweight="bold", color="white", zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("IEEE 14-bus network (project / published topology)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ieee14_diagram.png", dpi=150)
    plt.close(fig)


def test_probe_signal_wave_plot(simulation_settings):
    """Plot in-use probe signal (Hann window, A=0.3) and save to tests/output/probe_signal_wave.png."""
    if not HAS_MATPLOTLIB:
        pytest.skip("matplotlib required for plots")
    probe_duration = simulation_settings["probe_duration"]  # 2.0 s
    A = 0.3
    t = np.linspace(0, 2.5, 301)  # 0 to 2.5 s to show probe on then off
    u = np.array([A * hann_window(ti, probe_duration) for ti in t])
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(t, u, "b-", linewidth=2, label=f"Probe signal (A={A}, Hann 0–{probe_duration} s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Probe signal (Hann window)")
    ax.legend(loc="upper right", fontsize=7, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "probe_signal_wave.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s", "--tb=short"])
