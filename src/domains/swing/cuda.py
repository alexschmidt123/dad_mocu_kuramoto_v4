"""
PyCUDA batch swing-equation integration for offline one-step data generation.

One CUDA thread simulates one reset-based (theta, action) probe. Sequential BOED
histories are assembled by table lookup; the physical response is not
history-dependent.
Required for data generation (``data_generation.backend: cuda``, default).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.domains.swing.design import Design
    from src.domains.swing.simulator import SwingSimulator

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyCUDA is required for data generation. Install with: pip install pycuda"
    ) from exc


CUDA_KERNEL = r"""
extern "C" __global__ void simulate_trajectories(
    const int n_traj,
    const int T,
    const int n_steps,
    const int N,
    const int *sequences,
    const double *M_bus,
    const double *K_bus,
    const double *B_flat,
    const double *P_m,
    const double *D_nodes,
    const double *theta0,
    const double *omega0,
    const double *amp,
    const double *input_map,
    const int observation_bus,
    const double *duration,
    const double dt,
    const double fs_hz,
    double *out_y
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_traj) return;

    const double pi = 3.14159265358979323846;
    double y[28];
    double k1[28], k2[28], k3[28], k4[28], yt[28];

    for (int i = 0; i < N; ++i) {
        y[i] = theta0[i];
        y[N + i] = omega0[i];
    }

    for (int step = 0; step < T; ++step) {
        const int a = sequences[idx * T + step];
        const double A = amp[a];
        const int pb = observation_bus;
        const double Tp = duration[a];
        double rocof_max = 0.0;
        double omega_prev = y[N + pb];
        const int down = (int)fmax(1.0, floor(1.0 / (fs_hz * dt)));

        for (int s = 0; s < n_steps; ++s) {
            const double t = s * dt;

            for (int i = 0; i < N; ++i) {
                const double th = y[i];
                const double dw = y[N + i];
                double coupling = 0.0;
                for (int j = 0; j < N; ++j) {
                    const double Bij = B_flat[i * N + j];
                    if (Bij != 0.0) coupling += Bij * sin(th - y[j]);
                }
                double u = 0.0;
                if (t <= Tp) u = input_map[a * N + i] * A * 0.5 * (1.0 - cos(2.0 * pi * t / Tp));
                const double decay = K_bus[idx * N + i] / (2.0 * pi) + D_nodes[i];
                k1[i] = dw;
                k1[N + i] = (P_m[i] - coupling - decay * dw + u) / M_bus[idx * N + i];
            }

            for (int i = 0; i < 2 * N; ++i) yt[i] = y[i] + 0.5 * dt * k1[i];
            for (int i = 0; i < N; ++i) {
                const double th = yt[i];
                const double dw = yt[N + i];
                double coupling = 0.0;
                for (int j = 0; j < N; ++j) {
                    const double Bij = B_flat[i * N + j];
                    if (Bij != 0.0) coupling += Bij * sin(th - yt[j]);
                }
                double u = 0.0;
                const double tt2 = t + 0.5 * dt;
                if (tt2 <= Tp) u = input_map[a * N + i] * A * 0.5 * (1.0 - cos(2.0 * pi * tt2 / Tp));
                const double decay = K_bus[idx * N + i] / (2.0 * pi) + D_nodes[i];
                k2[i] = dw;
                k2[N + i] = (P_m[i] - coupling - decay * dw + u) / M_bus[idx * N + i];
            }

            for (int i = 0; i < 2 * N; ++i) yt[i] = y[i] + 0.5 * dt * k2[i];
            for (int i = 0; i < N; ++i) {
                const double th = yt[i];
                const double dw = yt[N + i];
                double coupling = 0.0;
                for (int j = 0; j < N; ++j) {
                    const double Bij = B_flat[i * N + j];
                    if (Bij != 0.0) coupling += Bij * sin(th - yt[j]);
                }
                double u = 0.0;
                const double tt3 = t + 0.5 * dt;
                if (tt3 <= Tp) u = input_map[a * N + i] * A * 0.5 * (1.0 - cos(2.0 * pi * tt3 / Tp));
                const double decay = K_bus[idx * N + i] / (2.0 * pi) + D_nodes[i];
                k3[i] = dw;
                k3[N + i] = (P_m[i] - coupling - decay * dw + u) / M_bus[idx * N + i];
            }

            for (int i = 0; i < 2 * N; ++i) yt[i] = y[i] + dt * k3[i];
            for (int i = 0; i < N; ++i) {
                const double th = yt[i];
                const double dw = yt[N + i];
                double coupling = 0.0;
                for (int j = 0; j < N; ++j) {
                    const double Bij = B_flat[i * N + j];
                    if (Bij != 0.0) coupling += Bij * sin(th - yt[j]);
                }
                double u = 0.0;
                const double tt4 = t + dt;
                if (tt4 <= Tp) u = input_map[a * N + i] * A * 0.5 * (1.0 - cos(2.0 * pi * tt4 / Tp));
                const double decay = K_bus[idx * N + i] / (2.0 * pi) + D_nodes[i];
                k4[i] = dw;
                k4[N + i] = (P_m[i] - coupling - decay * dw + u) / M_bus[idx * N + i];
            }

            for (int i = 0; i < 2 * N; ++i) {
                y[i] += dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
            }

            if (s >= down && (s % down) == 0) {
                const double df = (y[N + pb] - omega_prev) / (2.0 * pi) / (down * dt);
                const double av = fabs(df);
                if (av > rocof_max) rocof_max = av;
                omega_prev = y[N + pb];
            }
        }

        out_y[idx * T + step] = rocof_max;
    }
}
"""

_mod = SourceModule(CUDA_KERNEL, options=["--use_fast_math"])
_sim_kernel = _mod.get_function("simulate_trajectories")

# Full probe-bus Δf(t) at every ODE step (uncompressed physical observation bank).
CUDA_DELTA_F_KERNEL = r"""
extern "C" __global__ void simulate_delta_f_trajectories(
    const int n_traj,
    const int n_steps,
    const int N,
    const int *actions,
    const double *M_bus,
    const double *K_bus,
    const double *B_flat,
    const double *P_m,
    const double *D_nodes,
    const double *theta0,
    const double *omega0,
    const double *amp,
    const double *input_map,
    const int observation_bus,
    const double *duration,
    const double dt,
    double *out_df
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_traj) return;

    const double pi = 3.14159265358979323846;
    double y[28];
    double k1[28], k2[28], k3[28], k4[28], yt[28];

    for (int i = 0; i < N; ++i) {
        y[i] = theta0[i];
        y[N + i] = omega0[i];
    }

    const int a = actions[idx];
    const double A = amp[a];
    const int pb = observation_bus;
    const double Tp = duration[a];

    for (int s = 0; s < n_steps; ++s) {
        const double t = s * dt;

        for (int i = 0; i < N; ++i) {
            const double th = y[i];
            const double dw = y[N + i];
            double coupling = 0.0;
            for (int j = 0; j < N; ++j) {
                const double Bij = B_flat[i * N + j];
                if (Bij != 0.0) coupling += Bij * sin(th - y[j]);
            }
            double u = 0.0;
            if (t <= Tp) u = input_map[a * N + i] * A * 0.5 * (1.0 - cos(2.0 * pi * t / Tp));
            const double decay = K_bus[idx * N + i] / (2.0 * pi) + D_nodes[i];
            k1[i] = dw;
            k1[N + i] = (P_m[i] - coupling - decay * dw + u) / M_bus[idx * N + i];
        }

        for (int i = 0; i < 2 * N; ++i) yt[i] = y[i] + 0.5 * dt * k1[i];
        for (int i = 0; i < N; ++i) {
            const double th = yt[i];
            const double dw = yt[N + i];
            double coupling = 0.0;
            for (int j = 0; j < N; ++j) {
                const double Bij = B_flat[i * N + j];
                if (Bij != 0.0) coupling += Bij * sin(th - yt[j]);
            }
            double u = 0.0;
            const double tt2 = t + 0.5 * dt;
            if (tt2 <= Tp) u = input_map[a * N + i] * A * 0.5 * (1.0 - cos(2.0 * pi * tt2 / Tp));
            const double decay = K_bus[idx * N + i] / (2.0 * pi) + D_nodes[i];
            k2[i] = dw;
            k2[N + i] = (P_m[i] - coupling - decay * dw + u) / M_bus[idx * N + i];
        }

        for (int i = 0; i < 2 * N; ++i) yt[i] = y[i] + 0.5 * dt * k2[i];
        for (int i = 0; i < N; ++i) {
            const double th = yt[i];
            const double dw = yt[N + i];
            double coupling = 0.0;
            for (int j = 0; j < N; ++j) {
                const double Bij = B_flat[i * N + j];
                if (Bij != 0.0) coupling += Bij * sin(th - yt[j]);
            }
            double u = 0.0;
            const double tt3 = t + 0.5 * dt;
            if (tt3 <= Tp) u = input_map[a * N + i] * A * 0.5 * (1.0 - cos(2.0 * pi * tt3 / Tp));
            const double decay = K_bus[idx * N + i] / (2.0 * pi) + D_nodes[i];
            k3[i] = dw;
            k3[N + i] = (P_m[i] - coupling - decay * dw + u) / M_bus[idx * N + i];
        }

        for (int i = 0; i < 2 * N; ++i) yt[i] = y[i] + dt * k3[i];
        for (int i = 0; i < N; ++i) {
            const double th = yt[i];
            const double dw = yt[N + i];
            double coupling = 0.0;
            for (int j = 0; j < N; ++j) {
                const double Bij = B_flat[i * N + j];
                if (Bij != 0.0) coupling += Bij * sin(th - yt[j]);
            }
            double u = 0.0;
            const double tt4 = t + dt;
            if (tt4 <= Tp) u = input_map[a * N + i] * A * 0.5 * (1.0 - cos(2.0 * pi * tt4 / Tp));
            const double decay = K_bus[idx * N + i] / (2.0 * pi) + D_nodes[i];
            k4[i] = dw;
            k4[N + i] = (P_m[i] - coupling - decay * dw + u) / M_bus[idx * N + i];
        }

        for (int i = 0; i < 2 * N; ++i) {
            y[i] += dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }

        // Probe-bus frequency deviation [Hz] at this ODE step.
        out_df[idx * n_steps + s] = y[N + pb] / (2.0 * pi);
    }
}
"""

_mod_df = SourceModule(CUDA_DELTA_F_KERNEL, options=["--use_fast_math"])
_sim_delta_f_kernel = _mod_df.get_function("simulate_delta_f_trajectories")


class CudaTrajectoryEngine:
    """GPU batch engine: one thread = one ordered T-step trajectory."""

    def __init__(self, sim: SwingSimulator, catalog: list[Design]):
        self.N = sim.N
        self.fs_hz = sim.fs_hz
        self.T_obs = sim.T_obs_sec
        self.ode_dt = sim.ode_dt
        self._M_bus = np.zeros(self.N, dtype=np.float64)
        self._K_bus = np.zeros(self.N, dtype=np.float64)
        self._B = np.ascontiguousarray(sim.B, dtype=np.float64)
        self._P_m = np.ascontiguousarray(sim.P_m, dtype=np.float64)
        self._D = np.ascontiguousarray(sim.D_nodes, dtype=np.float64)
        self._theta0 = np.ascontiguousarray(sim.theta0, dtype=np.float64)
        self._omega0 = np.ascontiguousarray(sim.omega0, dtype=np.float64)
        self._amp = np.ascontiguousarray(
            np.array([d.amplitude for d in catalog], dtype=np.float64)
        )
        self._input_map = np.ascontiguousarray(
            np.stack([sim.physical_input_map[d.bus] for d in catalog]),
            dtype=np.float64,
        )
        self._observation_bus = int(sim.observation_bus)
        self._dur = np.ascontiguousarray(
            np.array([d.duration for d in catalog], dtype=np.float64)
        )

    def simulate_all_sequences(
        self,
        M: np.ndarray,
        K: np.ndarray,
        sequences: list[tuple[int, ...]],
        sigma_y: float,
        rng: np.random.Generator,
        *,
        batch_size: int = 512,
        progress_label: str = "",
    ) -> list[dict]:
        T = len(sequences[0]) if sequences else 0
        if T != 1:
            raise ValueError(
                "reset-based data generation only supports one-step action rows; "
                "pass sequences like [(action,)]"
            )
        n_traj = len(sequences)
        n_steps = int(math.ceil(max(self.T_obs, float(np.max(self._dur)) + 0.5) / self.ode_dt))
        seq_arr = np.ascontiguousarray(np.array(sequences, dtype=np.int32).reshape(-1))

        M_vec = np.ascontiguousarray(np.asarray(M, dtype=np.float64).reshape(-1))
        K_vec = np.ascontiguousarray(np.asarray(K, dtype=np.float64).reshape(-1))
        if M_vec.shape[0] != self.N:
            raise ValueError(f"M length {M_vec.shape[0]} != N={self.N}")

        out_all = np.zeros(n_traj * T, dtype=np.float64)
        y_sim_all = np.zeros(n_traj * T, dtype=np.float64)
        block = 128

        for start in range(0, n_traj, batch_size):
            end = min(n_traj, start + batch_size)
            n_batch = end - start
            if progress_label:
                print(f"    {progress_label} GPU trajectories {end}/{n_traj}")

            out_batch = np.zeros(n_batch * T, dtype=np.float64)
            seq_batch = seq_arr[start * T : end * T]
            M_tiled = np.ascontiguousarray(np.tile(M_vec, (n_batch, 1)))
            K_tiled = np.ascontiguousarray(np.tile(K_vec, (n_batch, 1)))

            _sim_kernel(
                np.int32(n_batch),
                np.int32(T),
                np.int32(n_steps),
                np.int32(self.N),
                cuda.In(seq_batch),
                cuda.In(M_tiled),
                cuda.In(K_tiled),
                cuda.In(self._B),
                cuda.In(self._P_m),
                cuda.In(self._D),
                cuda.In(self._theta0),
                cuda.In(self._omega0),
                cuda.In(self._amp),
                cuda.In(self._input_map),
                np.int32(self._observation_bus),
                cuda.In(self._dur),
                np.float64(self.ode_dt),
                np.float64(self.fs_hz),
                cuda.Out(out_batch),
                block=(block, 1, 1),
                grid=((n_batch + block - 1) // block, 1),
            )

            y_sim_batch = out_batch.copy()
            if sigma_y > 0:
                out_batch += rng.normal(0.0, sigma_y, size=out_batch.shape)
            out_all[start * T : end * T] = out_batch
            y_sim_all[start * T : end * T] = y_sim_batch

        return [
            {
                "sequence": list(seq),
                "y_sim": y_sim_all[i * T : (i + 1) * T].tolist(),
                "y": out_all[i * T : (i + 1) * T].tolist(),
            }
            for i, seq in enumerate(sequences)
        ]

    def n_sim_steps(self) -> int:
        """Number of ODE steps stored for full Δf banks."""
        return int(math.ceil(max(self.T_obs, float(np.max(self._dur)) + 0.5) / self.ode_dt))

    def simulate_delta_f_batch(
        self,
        M_rows: np.ndarray,
        K_rows: np.ndarray,
        action_indices: np.ndarray,
        *,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Probe-bus Δf(t) at every ODE step for each (θ, action) row.

        Returns:
            (n_traj, N_sim) float64 array, units Hz deviation.
        """
        M_rows = np.ascontiguousarray(M_rows, dtype=np.float64)
        K_rows = np.ascontiguousarray(K_rows, dtype=np.float64)
        actions = np.ascontiguousarray(action_indices, dtype=np.int32).reshape(-1)
        n_traj = int(actions.shape[0])
        if M_rows.shape != (n_traj, self.N) or K_rows.shape != (n_traj, self.N):
            raise ValueError("M_rows/K_rows must be (n_traj, N)")
        n_steps = self.n_sim_steps()
        out_all = np.zeros((n_traj, n_steps), dtype=np.float64)
        block = 128
        for start in range(0, n_traj, batch_size):
            end = min(n_traj, start + batch_size)
            n_batch = end - start
            out_batch = np.zeros(n_batch * n_steps, dtype=np.float64)
            _sim_delta_f_kernel(
                np.int32(n_batch),
                np.int32(n_steps),
                np.int32(self.N),
                cuda.In(actions[start:end]),
                cuda.In(M_rows[start:end]),
                cuda.In(K_rows[start:end]),
                cuda.In(self._B),
                cuda.In(self._P_m),
                cuda.In(self._D),
                cuda.In(self._theta0),
                cuda.In(self._omega0),
                cuda.In(self._amp),
                cuda.In(self._input_map),
                np.int32(self._observation_bus),
                cuda.In(self._dur),
                np.float64(self.ode_dt),
                cuda.Out(out_batch),
                block=(block, 1, 1),
                grid=((n_batch + block - 1) // block, 1),
            )
            out_all[start:end] = out_batch.reshape(n_batch, n_steps)
        return out_all

    def simulate_one_step_f_batch(
        self,
        M_rows: np.ndarray,
        K_rows: np.ndarray,
        action_indices: np.ndarray,
        *,
        batch_size: int = 512,
    ) -> np.ndarray:
        """
        One equilibrium probe per row: noiseless max-ROCOF F(θ, ξ).

        Args:
            M_rows: (n_traj, N) per-bus inertia per trajectory
            K_rows: (n_traj, N) per-bus stiffness per trajectory
            action_indices: (n_traj,) catalog indices

        Returns:
            (n_traj,) F values
        """
        M_rows = np.ascontiguousarray(M_rows, dtype=np.float64)
        K_rows = np.ascontiguousarray(K_rows, dtype=np.float64)
        actions = np.ascontiguousarray(action_indices, dtype=np.int32).reshape(-1)
        n_traj = int(actions.shape[0])
        if M_rows.shape != (n_traj, self.N) or K_rows.shape != (n_traj, self.N):
            raise ValueError("M_rows/K_rows must be (n_traj, N)")

        T = 1
        n_steps = int(math.ceil(max(self.T_obs, float(np.max(self._dur)) + 0.5) / self.ode_dt))
        f_all = np.zeros(n_traj, dtype=np.float64)
        block = 128

        for start in range(0, n_traj, batch_size):
            end = min(n_traj, start + batch_size)
            n_batch = end - start
            seq_batch = actions[start:end].reshape(-1)
            out_batch = np.zeros(n_batch, dtype=np.float64)

            _sim_kernel(
                np.int32(n_batch),
                np.int32(T),
                np.int32(n_steps),
                np.int32(self.N),
                cuda.In(seq_batch),
                cuda.In(M_rows[start:end]),
                cuda.In(K_rows[start:end]),
                cuda.In(self._B),
                cuda.In(self._P_m),
                cuda.In(self._D),
                cuda.In(self._theta0),
                cuda.In(self._omega0),
                cuda.In(self._amp),
                cuda.In(self._input_map),
                np.int32(self._observation_bus),
                cuda.In(self._dur),
                np.float64(self.ode_dt),
                np.float64(self.fs_hz),
                cuda.Out(out_batch),
                block=(block, 1, 1),
                grid=((n_batch + block - 1) // block, 1),
            )
            f_all[start:end] = out_batch

        return f_all

    def build_equilibrium_f_grid(
        self,
        M_support: np.ndarray,
        K_support: np.ndarray,
        n_actions: int,
        *,
        batch_size: int = 512,
    ) -> np.ndarray:
        """
        F(a, θ_n) for all one-step designs and MC support rows.

        Returns array shape ``(n_actions, n_support)`` with ``n_support = len(M_support)``.
        """
        n_mc = int(M_support.shape[0])
        M_big = np.ascontiguousarray(np.tile(M_support, (n_actions, 1)))
        K_big = np.ascontiguousarray(np.tile(K_support, (n_actions, 1)))
        actions = np.repeat(np.arange(n_actions, dtype=np.int32), n_mc)
        f_vals = self.simulate_one_step_f_batch(M_big, K_big, actions, batch_size=batch_size)
        return f_vals.reshape(n_actions, n_mc)
