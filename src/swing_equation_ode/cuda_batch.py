"""
PyCUDA batch simulator for independent (theta, xi) one-step probes.

Mapping:
- one CUDA block = one (theta, xi) pair
- one CUDA thread = one bus update
- output per pair = scalar max |ROCOF| at the probed bus
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyCUDA is required for cuda_batch. Install with: pip install pycuda"
    ) from exc


MAX_N_BUS = 64
CUDA_BATCH_KERNEL = r"""
#define MAX_N_BUS 64

extern "C" __global__ void simulate_max_rocof_pairs_rk2(
    const int n_pairs,
    const int N,
    const int n_steps,
    const float dt,
    const float fs_hz,
    const float probe_sign,
    const float unstable_state_limit,
    const float unstable_rocof_limit,
    const float *pair_M,
    const float *pair_K,
    const int *pair_action_id,
    const float *action_amp,
    const int *action_bus,
    const float *action_dt,
    const float *B_flat,
    const float *P_m,
    const float *D_nodes,
    const float *theta0,
    const float *omega0,
    float *out_y,
    int *out_flags
) {
    const int pair = blockIdx.x;
    const int tid = threadIdx.x;
    if (pair >= n_pairs) return;
    if (N > MAX_N_BUS) return;

    __shared__ float theta[MAX_N_BUS];
    __shared__ float omega[MAX_N_BUS];
    __shared__ float k1_theta[MAX_N_BUS];
    __shared__ float k1_omega[MAX_N_BUS];
    __shared__ float theta_mid[MAX_N_BUS];
    __shared__ float omega_mid[MAX_N_BUS];
    __shared__ float coupling[MAX_N_BUS];
    __shared__ float rocof_max;
    __shared__ float omega_prev;
    __shared__ int flags;
    __shared__ int pb;
    __shared__ float A;
    __shared__ float Tp;
    __shared__ int down;
    __shared__ int invalid_mass;

    if (tid == 0) {
        const int a = pair_action_id[pair];
        pb = action_bus[a];
        A = action_amp[a];
        Tp = action_dt[a];
        down = (int)(1.0f / (fs_hz * dt));
        if (down < 1) down = 1;
        rocof_max = 0.0f;
        omega_prev = 0.0f;
        flags = 0;
        invalid_mass = 0;
    }
    __syncthreads();

    if (tid < N) {
        theta[tid] = theta0[tid];
        omega[tid] = omega0[tid];
        if (pair_M[pair * N + tid] <= 0.0f) invalid_mass = 1;
    }
    __syncthreads();

    if (tid == 0) {
        if (pb >= 0 && pb < N) omega_prev = omega[pb];
        if (invalid_mass) flags |= 1;  // invalid M
    }
    __syncthreads();

    if (invalid_mass) {
        if (tid == 0) {
            out_y[pair] = 0.0f;
            out_flags[pair] = flags;
        }
        return;
    }

    for (int s = 0; s < n_steps; ++s) {
        const float t = s * dt;

        if (tid < N) {
            float c = 0.0f;
            const float th_i = theta[tid];
            for (int j = 0; j < N; ++j) {
                const float Bij = B_flat[tid * N + j];
                if (Bij != 0.0f) {
                    c += Bij * sinf(th_i - theta[j]);
                }
            }
            coupling[tid] = c;
        }
        __syncthreads();

        if (tid < N) {
            float u = 0.0f;
            if (tid == pb && t <= Tp && Tp > 0.0f) {
                const float pi = 3.14159265358979323846f;
                u = probe_sign * A * 0.5f * (1.0f - cosf(2.0f * pi * t / Tp));
            }
            const float M_i = pair_M[pair * N + tid];
            const float decay = pair_K[pair * N + tid] / (2.0f * 3.14159265358979323846f) + D_nodes[tid];
            k1_theta[tid] = omega[tid];
            k1_omega[tid] = (P_m[tid] - coupling[tid] - decay * omega[tid] + u) / M_i;
            theta_mid[tid] = theta[tid] + 0.5f * dt * k1_theta[tid];
            omega_mid[tid] = omega[tid] + 0.5f * dt * k1_omega[tid];
        }
        __syncthreads();

        if (tid < N) {
            float c_mid = 0.0f;
            const float th_i = theta_mid[tid];
            for (int j = 0; j < N; ++j) {
                const float Bij = B_flat[tid * N + j];
                if (Bij != 0.0f) {
                    c_mid += Bij * sinf(th_i - theta_mid[j]);
                }
            }
            coupling[tid] = c_mid;
        }
        __syncthreads();

        if (tid < N) {
            float u_mid = 0.0f;
            const float t_mid = t + 0.5f * dt;
            if (tid == pb && t_mid <= Tp && Tp > 0.0f) {
                const float pi = 3.14159265358979323846f;
                u_mid = probe_sign * A * 0.5f * (1.0f - cosf(2.0f * pi * t_mid / Tp));
            }
            const float M_i = pair_M[pair * N + tid];
            const float decay = pair_K[pair * N + tid] / (2.0f * 3.14159265358979323846f) + D_nodes[tid];
            const float k2_theta = omega_mid[tid];
            const float k2_omega = (P_m[tid] - coupling[tid] - decay * omega_mid[tid] + u_mid) / M_i;
            theta[tid] += dt * k2_theta;
            omega[tid] += dt * k2_omega;
        }
        __syncthreads();

        if (tid == 0) {
            if (pb >= 0 && pb < N && s >= down && (s % down) == 0) {
                const float rocof = (omega[pb] - omega_prev) / (2.0f * 3.14159265358979323846f) / (down * dt);
                const float av = fabsf(rocof);
                if (av > rocof_max) rocof_max = av;
                if (!isfinite(av) || av > unstable_rocof_limit) flags |= 8;  // unstable ROCOF
                omega_prev = omega[pb];
            }
        }
        __syncthreads();

        if (tid < N) {
            const float th = fabsf(theta[tid]);
            const float om = fabsf(omega[tid]);
            if (!isfinite(theta[tid]) || !isfinite(omega[tid])) flags |= 2;  // NaN/Inf state
            if (th > unstable_state_limit || om > unstable_state_limit) flags |= 4;  // blown state
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_y[pair] = isfinite(rocof_max) ? rocof_max : 0.0f;
        if (!isfinite(rocof_max)) flags |= 16;  // NaN/Inf output
        out_flags[pair] = flags;
    }
}
"""

_mod = SourceModule(CUDA_BATCH_KERNEL, options=["--use_fast_math"])
_rk2_kernel = _mod.get_function("simulate_max_rocof_pairs_rk2")


@dataclass
class BatchTiming:
    h2d_ms: float
    kernel_ms: float
    d2h_ms: float
    total_ms: float


def _as_f32(name: str, x: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} != {shape}")
    return arr


def _as_i32(name: str, x: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.int32))
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} != {shape}")
    return arr


def _cpu_reference_rk2(
    pair_M: np.ndarray,
    pair_K: np.ndarray,
    pair_action_id: np.ndarray,
    action_amp: np.ndarray,
    action_bus: np.ndarray,
    action_dt: np.ndarray,
    B: np.ndarray,
    P_m: np.ndarray,
    D_nodes: np.ndarray,
    theta0: np.ndarray,
    omega0: np.ndarray,
    *,
    T_sim: float,
    solver_dt: float,
    fs_hz: float,
    probe_sign: float,
) -> np.ndarray:
    n_pairs, n_bus = pair_M.shape
    out = np.zeros(n_pairs, dtype=np.float32)
    n_steps = int(np.ceil(T_sim / solver_dt))
    down = max(1, int(1.0 / (fs_hz * solver_dt)))
    two_pi = np.float32(2.0 * np.pi)
    for p in range(n_pairs):
        theta = theta0.copy()
        omega = omega0.copy()
        a = int(pair_action_id[p])
        pb = int(action_bus[a])
        A = float(action_amp[a])
        Tp = float(action_dt[a])
        rocof_max = 0.0
        omega_prev = float(omega[pb])
        for s in range(n_steps):
            t = s * solver_dt
            coupling = np.zeros(n_bus, dtype=np.float32)
            for i in range(n_bus):
                for j in range(n_bus):
                    Bij = float(B[i, j])
                    if Bij != 0.0:
                        coupling[i] += np.float32(Bij * np.sin(float(theta[i] - theta[j])))
            u = np.zeros(n_bus, dtype=np.float32)
            if t <= Tp and Tp > 0.0:
                u[pb] = np.float32(
                    probe_sign * A * 0.5 * (1.0 - np.cos(2.0 * np.pi * t / Tp))
                )
            decay = pair_K[p] / two_pi + D_nodes
            k1_theta = omega
            k1_omega = (P_m - coupling - decay * omega + u) / pair_M[p]
            theta_mid = theta + 0.5 * solver_dt * k1_theta
            omega_mid = omega + 0.5 * solver_dt * k1_omega

            coupling_mid = np.zeros(n_bus, dtype=np.float32)
            for i in range(n_bus):
                for j in range(n_bus):
                    Bij = float(B[i, j])
                    if Bij != 0.0:
                        coupling_mid[i] += np.float32(Bij * np.sin(float(theta_mid[i] - theta_mid[j])))
            u_mid = np.zeros(n_bus, dtype=np.float32)
            t_mid = t + 0.5 * solver_dt
            if t_mid <= Tp and Tp > 0.0:
                u_mid[pb] = np.float32(
                    probe_sign * A * 0.5 * (1.0 - np.cos(2.0 * np.pi * t_mid / Tp))
                )
            k2_theta = omega_mid
            k2_omega = (P_m - coupling_mid - decay * omega_mid + u_mid) / pair_M[p]
            theta = theta + solver_dt * k2_theta
            omega = omega + solver_dt * k2_omega

            if s >= down and (s % down) == 0:
                rocof = (float(omega[pb]) - omega_prev) / (2.0 * np.pi) / (down * solver_dt)
                rocof_max = max(rocof_max, abs(rocof))
                omega_prev = float(omega[pb])
        out[p] = np.float32(rocof_max)
    return out


def simulate_max_rocof_batch_pycuda(
    M_samples: np.ndarray,
    K_samples: np.ndarray,
    actions: np.ndarray,
    P: np.ndarray,
    D: np.ndarray,
    x0: np.ndarray,
    B: np.ndarray,
    T_sim: float,
    solver_dt: float,
    *,
    fs_hz: float = 12.0,
    probe_sign: float = 1.0,
    batch_size: int = 8192,
    unstable_state_limit: float = 1e3,
    unstable_rocof_limit: float = 1e5,
    return_diagnostics: bool = False,
    return_timing: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, BatchTiming]:
    """
    GPU batched one-step simulator.

    Args:
        M_samples: (n_theta, n_bus) inertia samples.
        K_samples: (n_theta, n_bus) stiffness samples.
        actions: (n_actions, 3) as [amplitude, bus, duration].
        P: (n_bus,) base mechanical power.
        D: (n_bus,) damping.
        x0: (2, n_bus) or (2*n_bus,) initial [theta0, omega0].
        B: (n_bus, n_bus) coupling matrix.
        T_sim: simulation time horizon.
        solver_dt: fixed integration step.

    Returns:
        Y reshaped to (n_theta, n_actions), and optionally flags and timing.
    """
    if solver_dt <= 0.0:
        raise ValueError("solver_dt must be > 0")
    n_theta, n_bus = np.asarray(M_samples).shape
    if n_bus > MAX_N_BUS:
        raise ValueError(f"n_bus={n_bus} exceeds MAX_N_BUS={MAX_N_BUS}")
    if np.asarray(K_samples).shape != (n_theta, n_bus):
        raise ValueError("K_samples shape mismatch")
    if np.any(np.asarray(M_samples) <= 0):
        raise ValueError("All M_i must be > 0")

    actions_arr = np.asarray(actions)
    if actions_arr.ndim != 2 or actions_arr.shape[1] != 3:
        raise ValueError("actions must have shape (n_actions, 3): [A, bus, duration]")
    n_actions = int(actions_arr.shape[0])
    if n_actions <= 0:
        raise ValueError("actions cannot be empty")

    M_samples = _as_f32("M_samples", M_samples, (n_theta, n_bus))
    K_samples = _as_f32("K_samples", K_samples, (n_theta, n_bus))
    B = _as_f32("B", B, (n_bus, n_bus))
    P = _as_f32("P", P, (n_bus,))
    D = _as_f32("D", D, (n_bus,))

    x0_arr = np.asarray(x0, dtype=np.float32)
    if x0_arr.shape == (2, n_bus):
        theta0 = _as_f32("theta0", x0_arr[0], (n_bus,))
        omega0 = _as_f32("omega0", x0_arr[1], (n_bus,))
    elif x0_arr.shape == (2 * n_bus,):
        theta0 = _as_f32("theta0", x0_arr[:n_bus], (n_bus,))
        omega0 = _as_f32("omega0", x0_arr[n_bus:], (n_bus,))
    else:
        raise ValueError("x0 must have shape (2, n_bus) or (2*n_bus,)")

    action_amp = _as_f32("action_amp", actions_arr[:, 0], (n_actions,))
    action_bus = _as_i32("action_bus", actions_arr[:, 1], (n_actions,))
    action_dt = _as_f32("action_dt", actions_arr[:, 2], (n_actions,))
    if np.any(action_bus < 0) or np.any(action_bus >= n_bus):
        raise ValueError("action bus index out of range")

    pair_theta_id = np.repeat(np.arange(n_theta, dtype=np.int32), n_actions)
    pair_action_id = np.tile(np.arange(n_actions, dtype=np.int32), n_theta)
    n_pairs = int(pair_theta_id.size)
    pair_M = np.ascontiguousarray(M_samples[pair_theta_id], dtype=np.float32)
    pair_K = np.ascontiguousarray(K_samples[pair_theta_id], dtype=np.float32)

    y_flat = np.zeros(n_pairs, dtype=np.float32)
    flag_flat = np.zeros(n_pairs, dtype=np.int32)
    n_steps = int(np.ceil(float(T_sim) / float(solver_dt)))

    h2d_ms = 0.0
    kernel_ms = 0.0
    d2h_ms = 0.0
    total_start, total_end = cuda.Event(), cuda.Event()
    total_start.record()

    block = (32, 1, 1)  # IEEE-14-friendly default; threads >= n_bus are idle
    for start in range(0, n_pairs, int(batch_size)):
        end = min(n_pairs, start + int(batch_size))
        n_batch = int(end - start)

        h2d_start, h2d_end = cuda.Event(), cuda.Event()
        ker_start, ker_end = cuda.Event(), cuda.Event()
        d2h_start, d2h_end = cuda.Event(), cuda.Event()

        h2d_start.record()
        d_pair_M = cuda.mem_alloc(pair_M[start:end].nbytes)
        d_pair_K = cuda.mem_alloc(pair_K[start:end].nbytes)
        d_pair_action = cuda.mem_alloc(pair_action_id[start:end].nbytes)
        d_amp = cuda.mem_alloc(action_amp.nbytes)
        d_bus = cuda.mem_alloc(action_bus.nbytes)
        d_dur = cuda.mem_alloc(action_dt.nbytes)
        d_B = cuda.mem_alloc(B.nbytes)
        d_P = cuda.mem_alloc(P.nbytes)
        d_D = cuda.mem_alloc(D.nbytes)
        d_theta0 = cuda.mem_alloc(theta0.nbytes)
        d_omega0 = cuda.mem_alloc(omega0.nbytes)
        d_out_y = cuda.mem_alloc(y_flat[start:end].nbytes)
        d_out_flags = cuda.mem_alloc(flag_flat[start:end].nbytes)

        cuda.memcpy_htod(d_pair_M, pair_M[start:end])
        cuda.memcpy_htod(d_pair_K, pair_K[start:end])
        cuda.memcpy_htod(d_pair_action, pair_action_id[start:end])
        cuda.memcpy_htod(d_amp, action_amp)
        cuda.memcpy_htod(d_bus, action_bus)
        cuda.memcpy_htod(d_dur, action_dt)
        cuda.memcpy_htod(d_B, B)
        cuda.memcpy_htod(d_P, P)
        cuda.memcpy_htod(d_D, D)
        cuda.memcpy_htod(d_theta0, theta0)
        cuda.memcpy_htod(d_omega0, omega0)
        h2d_end.record()

        ker_start.record()
        _rk2_kernel(
            np.int32(n_batch),
            np.int32(n_bus),
            np.int32(n_steps),
            np.float32(solver_dt),
            np.float32(fs_hz),
            np.float32(probe_sign),
            np.float32(unstable_state_limit),
            np.float32(unstable_rocof_limit),
            d_pair_M,
            d_pair_K,
            d_pair_action,
            d_amp,
            d_bus,
            d_dur,
            d_B,
            d_P,
            d_D,
            d_theta0,
            d_omega0,
            d_out_y,
            d_out_flags,
            block=block,
            grid=(n_batch, 1, 1),
        )
        ker_end.record()

        d2h_start.record()
        cuda.memcpy_dtoh(y_flat[start:end], d_out_y)
        cuda.memcpy_dtoh(flag_flat[start:end], d_out_flags)
        d2h_end.record()

        d2h_end.synchronize()
        h2d_ms += h2d_start.time_till(h2d_end)
        kernel_ms += ker_start.time_till(ker_end)
        d2h_ms += d2h_start.time_till(d2h_end)

        for ptr in (
            d_pair_M, d_pair_K, d_pair_action, d_amp, d_bus, d_dur, d_B, d_P,
            d_D, d_theta0, d_omega0, d_out_y, d_out_flags
        ):
            ptr.free()

    total_end.record()
    total_end.synchronize()
    total_ms = total_start.time_till(total_end)
    timing = BatchTiming(h2d_ms=h2d_ms, kernel_ms=kernel_ms, d2h_ms=d2h_ms, total_ms=total_ms)

    y_out = y_flat.reshape(n_theta, n_actions)
    f_out = flag_flat.reshape(n_theta, n_actions)

    if np.any(~np.isfinite(y_out)):
        raise RuntimeError("NaN/Inf detected in GPU output")

    if return_diagnostics and return_timing:
        return y_out, f_out, timing
    if return_diagnostics:
        return y_out, f_out
    if return_timing:
        return y_out, timing
    return y_out


def test_pycuda_matches_cpu(
    *,
    n_theta: int = 4,
    n_actions: int = 6,
    n_bus: int = 9,
    seed: int = 0,
) -> dict[str, float]:
    """
    Small CPU-vs-GPU consistency check for RK2 batch simulator.
    """
    rng = np.random.default_rng(seed)
    M_samples = rng.uniform(0.01, 0.06, size=(n_theta, n_bus)).astype(np.float32)
    K_samples = rng.uniform(0.05, 0.50, size=(n_theta, n_bus)).astype(np.float32)

    amps = rng.choice(np.array([0.05, 0.1, 0.2], dtype=np.float32), size=n_actions, replace=True)
    buses = rng.integers(0, n_bus, size=n_actions, dtype=np.int32)
    dts = np.full(n_actions, 0.2, dtype=np.float32)
    actions = np.stack([amps, buses.astype(np.float32), dts], axis=1).astype(np.float32)

    B = rng.normal(0.0, 0.1, size=(n_bus, n_bus)).astype(np.float32)
    B = 0.5 * (B + B.T)
    np.fill_diagonal(B, 0.0)
    P = rng.normal(0.0, 0.2, size=n_bus).astype(np.float32)
    D = rng.uniform(0.05, 0.2, size=n_bus).astype(np.float32)
    theta0 = np.zeros(n_bus, dtype=np.float32)
    omega0 = np.zeros(n_bus, dtype=np.float32)
    x0 = np.stack([theta0, omega0], axis=0).astype(np.float32)

    T_sim = 1.0
    solver_dt = 1.0 / 160.0
    fs_hz = 12.0

    y_gpu, flags = simulate_max_rocof_batch_pycuda(
        M_samples, K_samples, actions, P, D, x0, B, T_sim, solver_dt,
        fs_hz=fs_hz, return_diagnostics=True, batch_size=1024
    )

    pair_theta_id = np.repeat(np.arange(n_theta, dtype=np.int32), n_actions)
    pair_action_id = np.tile(np.arange(n_actions, dtype=np.int32), n_theta)
    y_cpu_flat = _cpu_reference_rk2(
        M_samples[pair_theta_id],
        K_samples[pair_theta_id],
        pair_action_id,
        actions[:, 0].astype(np.float32),
        actions[:, 1].astype(np.int32),
        actions[:, 2].astype(np.float32),
        B, P, D, theta0, omega0,
        T_sim=T_sim, solver_dt=solver_dt, fs_hz=fs_hz, probe_sign=1.0,
    )
    y_cpu = y_cpu_flat.reshape(n_theta, n_actions)

    abs_err = np.abs(y_gpu - y_cpu)
    rel_err = abs_err / np.maximum(np.abs(y_cpu), 1e-8)
    stats = {
        "max_abs_error": float(np.max(abs_err)),
        "mean_abs_error": float(np.mean(abs_err)),
        "max_rel_error": float(np.max(rel_err)),
        "mean_rel_error": float(np.mean(rel_err)),
        "flag_rate": float(np.mean(flags != 0)),
    }
    print(
        "CPU/GPU RK2 validation:",
        f"max_abs={stats['max_abs_error']:.6e}",
        f"max_rel={stats['max_rel_error']:.6e}",
        f"flag_rate={stats['flag_rate']:.3f}",
    )
    return stats


def profile_batch_sizes(
    *,
    n_theta: int = 64,
    n_actions: int = 27,
    n_bus: int = 9,
    batch_sizes: tuple[int, ...] = (1024, 4096, 8192, 16384),
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Timing helper for requested batch sizes.
    """
    rng = np.random.default_rng(seed)
    M_samples = rng.uniform(0.01, 0.06, size=(n_theta, n_bus)).astype(np.float32)
    K_samples = rng.uniform(0.05, 0.50, size=(n_theta, n_bus)).astype(np.float32)
    actions = np.zeros((n_actions, 3), dtype=np.float32)
    actions[:, 0] = rng.choice(np.array([0.05, 0.1, 0.2], dtype=np.float32), size=n_actions)
    actions[:, 1] = rng.integers(0, n_bus, size=n_actions)
    actions[:, 2] = 0.2
    B = rng.normal(0.0, 0.1, size=(n_bus, n_bus)).astype(np.float32)
    B = 0.5 * (B + B.T)
    np.fill_diagonal(B, 0.0)
    P = rng.normal(0.0, 0.2, size=n_bus).astype(np.float32)
    D = rng.uniform(0.05, 0.2, size=n_bus).astype(np.float32)
    x0 = np.zeros((2, n_bus), dtype=np.float32)

    rows: list[dict[str, Any]] = []
    for bs in batch_sizes:
        _, flags, t = simulate_max_rocof_batch_pycuda(
            M_samples, K_samples, actions, P, D, x0, B,
            T_sim=1.0, solver_dt=1.0 / 160.0,
            batch_size=bs,
            return_diagnostics=True,
            return_timing=True,
        )
        row = {
            "batch_size": int(bs),
            "h2d_ms": float(t.h2d_ms),
            "kernel_ms": float(t.kernel_ms),
            "d2h_ms": float(t.d2h_ms),
            "total_ms": float(t.total_ms),
            "flag_rate": float(np.mean(flags != 0)),
        }
        rows.append(row)
        print(
            f"batch={bs:5d} h2d={row['h2d_ms']:.2f}ms kernel={row['kernel_ms']:.2f}ms "
            f"d2h={row['d2h_ms']:.2f}ms total={row['total_ms']:.2f}ms flags={row['flag_rate']:.3f}"
        )
    return rows
