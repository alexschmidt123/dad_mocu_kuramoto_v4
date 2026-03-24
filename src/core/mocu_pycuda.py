"""
**PyCUDA + embedded CUDA C++** MOCU for the swing equation (second-order Kuramoto).

This module is **not** plain PyTorch: kernels are **CUDA C++ source strings**, compiled at
runtime via **PyCUDA** (`pycuda.compiler.SourceModule`). Each GPU thread runs RK4 + binary
search for one ``(M, K)`` sample.

**When to use:** optional **fast** path from :func:`swing_equation_mocu.get_mocu_swing_computer`
when ``USE_PYCUDA=1`` and nvcc/driver work. Same MOCU definition as
:class:`~swing_equation_mocu.MOCU_swing_equation` (median \\(\\hat\\gamma\\), mean \\(|\\gamma^*-\\hat\\gamma|\\)).

**See also (same math, different backends):**

- :mod:`mocu_particles` — NumPy, **given** particle weights (no i.i.d. sampling).
- :mod:`swing_equation_mocu` — **PyTorch** + torchdiffeq (default pipeline).
- :mod:`mocu_torchdiffeq` — thin torchdiffeq helpers built on ``swing_equation_mocu``.
"""

import numpy as np

# Lazy init; cache by N
_mod_cache = {}


def _get_module(N):
    global _mod_cache
    if N in _mod_cache:
        return _mod_cache[N]

    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule

    # CUDA kernel: swing equation RK4 + binary search for γ*
    kernel_src = f"""
#define N_SWING {N}
#define PI 3.14159265358979323846

__device__ double hann_window(double t, double T) {{
    if (t > T) return 0.0;
    return 0.5 * (1.0 - cos(2.0 * PI * t / T));
}}

// RK4 step; returns 1 if constraints satisfied, 0 otherwise
// Constraints: ROCOF_max <= r_max, f_min_actual >= f_min
// ROCOF = |d(omega/2π)/dt| = |domega_dt|/(2π); f = 50 + omega/(2π) Hz
__device__ int swing_check_constraints(
    double M_val, double K_val, double gamma,
    double *B, double *P_m, double *g, double D,
    double ref_A, int ref_bus, double ref_T,
    double r_max, double omega_min_req,  // omega_min_req = (f_min - 50)*2*PI
    double h, int M_steps,
    double *theta, double *omega, double *domega_dt_store
) {{
    int i, j, k;
    double t = 0.0;
    double coupling;
    double u_probe = 0.0, u_ctrl = 0.0;
    double k1_t[N_SWING], k1_o[N_SWING], k2_t[N_SWING], k2_o[N_SWING];
    double k3_t[N_SWING], k3_o[N_SWING], k4_o[N_SWING];
    double theta_tmp[N_SWING], omega_tmp[N_SWING];
    double rocof_max = 0.0;
    double omega_min = 1e9;
    double omega_old[N_SWING];

    for (i = 0; i < N_SWING; i++) {{
        theta[i] = 0.0;
        omega[i] = 0.0;
    }}

    for (k = 0; k < M_steps; k++) {{
        for (i = 0; i < N_SWING; i++) {{
            theta_tmp[i] = theta[i];
            omega_tmp[i] = omega[i];
            omega_old[i] = omega[i];
        }}

        // k1
        for (i = 0; i < N_SWING; i++) {{
            coupling = 0.0;
            for (j = 0; j < N_SWING; j++)
                coupling += B[i*N_SWING+j] * sin(theta_tmp[j] - theta_tmp[i]);
            u_ctrl = -gamma * g[i] * omega_tmp[i];
            u_probe = (i == ref_bus && ref_A > 0) ? ref_A * hann_window(t, ref_T) : 0.0;
            domega_dt_store[i] = (P_m[i] - coupling - D*omega_tmp[i] - K_val*omega_tmp[i] + u_probe + u_ctrl) / M_val;
        }}
        for (i = 0; i < N_SWING; i++) {{
            k1_t[i] = h * omega_tmp[i];
            k1_o[i] = h * domega_dt_store[i];
        }}

        for (i = 0; i < N_SWING; i++) {{
            theta_tmp[i] = theta[i] + 0.5*k1_t[i];
            omega_tmp[i] = omega[i] + 0.5*k1_o[i];
        }}
        t += 0.5*h;

        // k2
        for (i = 0; i < N_SWING; i++) {{
            coupling = 0.0;
            for (j = 0; j < N_SWING; j++)
                coupling += B[i*N_SWING+j] * sin(theta_tmp[j] - theta_tmp[i]);
            u_ctrl = -gamma * g[i] * omega_tmp[i];
            u_probe = (i == ref_bus && ref_A > 0) ? ref_A * hann_window(t, ref_T) : 0.0;
            k2_o[i] = h * (P_m[i] - coupling - D*omega_tmp[i] - K_val*omega_tmp[i] + u_probe + u_ctrl) / M_val;
        }}
        for (i = 0; i < N_SWING; i++) {{
            k2_t[i] = h * omega_tmp[i];
            theta_tmp[i] = theta[i] + 0.5*k2_t[i];
            omega_tmp[i] = omega[i] + 0.5*k2_o[i];
        }}

        // k3
        for (i = 0; i < N_SWING; i++) {{
            coupling = 0.0;
            for (j = 0; j < N_SWING; j++)
                coupling += B[i*N_SWING+j] * sin(theta_tmp[j] - theta_tmp[i]);
            u_ctrl = -gamma * g[i] * omega_tmp[i];
            u_probe = (i == ref_bus && ref_A > 0) ? ref_A * hann_window(t, ref_T) : 0.0;
            k3_o[i] = h * (P_m[i] - coupling - D*omega_tmp[i] - K_val*omega_tmp[i] + u_probe + u_ctrl) / M_val;
        }}
        for (i = 0; i < N_SWING; i++) {{
            k3_t[i] = h * omega_tmp[i];
            theta_tmp[i] = theta[i] + k3_t[i];
            omega_tmp[i] = omega[i] + k3_o[i];
        }}
        t = (k + 1) * h;

        // k4: use theta_tmp = theta + k3_t, omega_tmp = omega + k3_o
        for (i = 0; i < N_SWING; i++) {{
            coupling = 0.0;
            for (j = 0; j < N_SWING; j++)
                coupling += B[i*N_SWING+j] * sin(theta_tmp[j] - theta_tmp[i]);
            u_ctrl = -gamma * g[i] * omega_tmp[i];
            u_probe = (i == ref_bus && ref_A > 0) ? ref_A * hann_window(t, ref_T) : 0.0;
            k4_o[i] = h * (P_m[i] - coupling - D*omega_tmp[i] - K_val*omega_tmp[i] + u_probe + u_ctrl) / M_val;
        }}

        for (i = 0; i < N_SWING; i++) {{
            double k4_t_i = h * omega_tmp[i];
            theta[i] += (k1_t[i] + 2*k2_t[i] + 2*k3_t[i] + k4_t_i) / 6.0;
            omega[i] += (k1_o[i] + 2*k2_o[i] + 2*k3_o[i] + k4_o[i]) / 6.0;
        }}

        for (i = 0; i < N_SWING; i++) {{
            double do_dt = (omega[i] - omega_old[i]) / h;
            double rocof = fabs(do_dt) / (2.0 * PI);
            if (rocof > rocof_max) rocof_max = rocof;
            if (omega[i] < omega_min) omega_min = omega[i];
        }}
        t += h;
    }}

    if (rocof_max <= r_max && omega_min >= omega_min_req) return 1;
    return 0;
}}

__global__ void swing_mocu_task(
    double *M_batch, double *K_batch,
    double *B, double *P_m, double *g,
    double D, double ref_A, int ref_bus, double ref_T,
    double r_max, double f_min,
    double h, int M_steps,
    double *gamma_star_out
) {{
    int i_c = blockDim.x * blockIdx.x + threadIdx.x;
    double M_val = M_batch[i_c];
    double K_val = K_batch[i_c];
    double omega_min_req = (f_min - 50.0) * 2.0 * PI;

    double theta[N_SWING], omega[N_SWING], domega_dt[N_SWING];

    double gamma_lower = 0.0;
    double gamma_upper = 200.0;
    int iter, found = 0;

    for (iter = 1; iter <= 100; iter++) {{
        double gamma_test = 2.0 * iter;
        int ok = swing_check_constraints(
            M_val, K_val, gamma_test,
            B, P_m, g, D,
            ref_A, ref_bus, ref_T,
            r_max, omega_min_req,
            h, M_steps,
            theta, omega, domega_dt
        );
        if (ok) {{
            gamma_upper = gamma_test;
            found = 1;
            break;
        }}
    }}

    if (found) {{
        for (iter = 0; iter < 15; iter++) {{
            double mid = (gamma_lower + gamma_upper) / 2.0;
            int ok = swing_check_constraints(
                M_val, K_val, mid,
                B, P_m, g, D,
                ref_A, ref_bus, ref_T,
                r_max, omega_min_req,
                h, M_steps,
                theta, omega, domega_dt
            );
            if (ok) gamma_upper = mid;
            else gamma_lower = mid;
            if (gamma_upper - gamma_lower < 0.01) break;
        }}
    }}

    gamma_star_out[i_c] = found ? gamma_upper : 1000000.0;
}}
"""

    try:
        # Suppress nvcc warnings so compile succeeds on strict systems
        mod = SourceModule(kernel_src, options=['-w'])
    except Exception as e:
        raise RuntimeError(
            f"PyCUDA kernel compile failed for N={N}: {e}. "
            "Ensure nvcc is in PATH and CUDA driver matches. Use USE_PYCUDA=0 to fall back to torchdiffeq."
        ) from e
    task = mod.get_function("swing_mocu_task")
    _mod_cache[N] = (mod, task)
    return mod, task


def MOCU_swing_pycuda(
    K_max: int,
    B: np.ndarray,
    P_m: np.ndarray,
    D: float,
    M_lower: float, M_upper: float,
    K_lower: float, K_upper: float,
    g: np.ndarray,
    r_max: float = 0.5,
    f_min: float = 49.8,
    h: float = 1.0 / 160.0,
    T: float = 10.0,
    reference_probe_bus: int = 0,
    reference_probe_amplitude: float = 0.5,
    reference_probe_duration: float = 2.0,
    seed: int = 0,
) -> float:
    """
    MOCU via PyCUDA: sample ``(M,K)``, compute ``γ*`` per sample on GPU, then median / mean |·|.

    Returns:
        MOCU value (float)
    """
    N = B.shape[0]

    M_steps = int(round(T / h))

    # Sample (M, K)
    if seed != 0:
        rng = np.random.default_rng(seed)
        M_batch = rng.uniform(M_lower, M_upper, size=K_max).astype(np.float64)
        K_batch = rng.uniform(K_lower, K_upper, size=K_max).astype(np.float64)
    else:
        M_batch = np.random.uniform(M_lower, M_upper, size=K_max).astype(np.float64)
        K_batch = np.random.uniform(K_lower, K_upper, size=K_max).astype(np.float64)

    B_flat = np.ascontiguousarray(B.astype(np.float64))
    P_m_flat = np.ascontiguousarray(P_m.astype(np.float64))
    g_flat = np.ascontiguousarray(g.astype(np.float64))

    gamma_star_out = np.zeros(K_max, dtype=np.float64)

    _, task = _get_module(N)
    import pycuda.driver as drv

    task(
        drv.In(M_batch),
        drv.In(K_batch),
        drv.In(B_flat),
        drv.In(P_m_flat),
        drv.In(g_flat),
        np.float64(D),
        np.float64(reference_probe_amplitude),
        np.int32(reference_probe_bus),
        np.float64(reference_probe_duration),
        np.float64(r_max),
        np.float64(f_min),
        np.float64(h),
        np.int32(M_steps),
        drv.Out(gamma_star_out),
        block=(min(256, K_max), 1, 1),
        grid=((K_max + 255) // 256, 1),
    )

    valid = (gamma_star_out < 1e6) & (gamma_star_out >= 0) & np.isfinite(gamma_star_out)
    if np.sum(valid) == 0:
        n_fail = np.sum(gamma_star_out >= 1e6)
        raise RuntimeError(f"All PyCUDA MOCU samples failed ({n_fail}/{K_max} threads). "
                          "Check constraints (r_max, f_min) and parameter bounds.")
    gamma_star = gamma_star_out[valid]
    gamma_hat = np.median(gamma_star)
    mocu_val = np.mean(np.abs(gamma_star - gamma_hat))
    return float(mocu_val)
