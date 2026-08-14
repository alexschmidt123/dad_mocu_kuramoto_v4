"""
PyCUDA batch control-only simulations (probe_amplitude = 0).

Each thread evaluates one (θ, u_candidate) pair under the configured contingency
and supplementary active-power injection profile. Outputs max |ROCOF| and
frequency-deviation nadir using the same definitions as true-system evaluation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from src.control.u_req import ControlSpec, is_control_safe, metrics_from_omega_traj

if TYPE_CHECKING:
    from src.domains.swing.simulator import SwingSimulator

try:
    # Do not use pycuda.autoinit: share Torch's primary context via retain_primary_context.
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyCUDA is required for control-bank generation. Install with: pip install pycuda"
    ) from exc


# N<=14 → state length 28 (matches probe kernel convention).
CUDA_CONTROL_KERNEL = r"""
extern "C" __global__ void simulate_control_metrics(
    const int n_traj,
    const int n_steps,
    const int N,
    const double *M_bus,          // (n_traj, N)
    const double *K_bus,          // (n_traj, N)
    const double *B_flat,         // (N, N)
    const double *P_m,            // (N,)
    const double *D_nodes,        // (N,)
    const double *theta0,         // (N,)
    const double *omega0,         // (N,)
    const double *u_mag,          // (n_traj,)
    const int cont_bus,
    const double cont_mag,
    const int ctrl_bus,
    const double ctrl_t0,
    const double ctrl_dur,
    const int ctrl_shape,         // 0=step, 1=hann, 2=ramp
    const double dt,
    double *out_rocof,            // (n_traj,)
    double *out_nadir             // (n_traj,) delta_f nadir [Hz]
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

    const double U = u_mag[idx];
    double rocof_max = 0.0;
    double nadir = 1.0e300;
    double omega_prev[14];
    for (int i = 0; i < N; ++i) omega_prev[i] = y[N + i];

    for (int s = 0; s < n_steps; ++s) {
        const double t = s * dt;

        // Control injection shape (probe amplitude identically 0).
        double u_inj = 0.0;
        if (U != 0.0 && ctrl_dur > 0.0 && t >= ctrl_t0 && t <= ctrl_t0 + ctrl_dur) {
            const double tau = (t - ctrl_t0) / ctrl_dur;
            if (ctrl_shape == 0) {
                u_inj = U;
            } else if (ctrl_shape == 1) {
                u_inj = U * 0.5 * (1.0 - cos(2.0 * pi * tau));
            } else {
                u_inj = U * tau;
            }
        }

        // RK4
        for (int i = 0; i < N; ++i) {
            const double th = y[i];
            const double dw = y[N + i];
            double coupling = 0.0;
            for (int j = 0; j < N; ++j) {
                const double Bij = B_flat[i * N + j];
                if (Bij != 0.0) coupling += Bij * sin(th - y[j]);
            }
            double P = P_m[i];
            if (i == cont_bus) P += cont_mag;
            double u = 0.0;
            if (i == ctrl_bus) u = u_inj;
            const double decay = K_bus[idx * N + i] / (2.0 * pi) + D_nodes[i];
            k1[i] = dw;
            k1[N + i] = (P - coupling - decay * dw + u) / M_bus[idx * N + i];
        }
        for (int i = 0; i < 2 * N; ++i) yt[i] = y[i] + 0.5 * dt * k1[i];

        // mid control at t+0.5dt
        double u_inj_m = 0.0;
        const double tm = t + 0.5 * dt;
        if (U != 0.0 && ctrl_dur > 0.0 && tm >= ctrl_t0 && tm <= ctrl_t0 + ctrl_dur) {
            const double tau = (tm - ctrl_t0) / ctrl_dur;
            if (ctrl_shape == 0) u_inj_m = U;
            else if (ctrl_shape == 1) u_inj_m = U * 0.5 * (1.0 - cos(2.0 * pi * tau));
            else u_inj_m = U * tau;
        }

        for (int i = 0; i < N; ++i) {
            const double th = yt[i];
            const double dw = yt[N + i];
            double coupling = 0.0;
            for (int j = 0; j < N; ++j) {
                const double Bij = B_flat[i * N + j];
                if (Bij != 0.0) coupling += Bij * sin(th - yt[j]);
            }
            double P = P_m[i];
            if (i == cont_bus) P += cont_mag;
            double u = 0.0;
            if (i == ctrl_bus) u = u_inj_m;
            const double decay = K_bus[idx * N + i] / (2.0 * pi) + D_nodes[i];
            k2[i] = dw;
            k2[N + i] = (P - coupling - decay * dw + u) / M_bus[idx * N + i];
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
            double P = P_m[i];
            if (i == cont_bus) P += cont_mag;
            double u = 0.0;
            if (i == ctrl_bus) u = u_inj_m;
            const double decay = K_bus[idx * N + i] / (2.0 * pi) + D_nodes[i];
            k3[i] = dw;
            k3[N + i] = (P - coupling - decay * dw + u) / M_bus[idx * N + i];
        }
        for (int i = 0; i < 2 * N; ++i) yt[i] = y[i] + dt * k3[i];

        double u_inj_e = 0.0;
        const double te = t + dt;
        if (U != 0.0 && ctrl_dur > 0.0 && te >= ctrl_t0 && te <= ctrl_t0 + ctrl_dur) {
            const double tau = (te - ctrl_t0) / ctrl_dur;
            if (ctrl_shape == 0) u_inj_e = U;
            else if (ctrl_shape == 1) u_inj_e = U * 0.5 * (1.0 - cos(2.0 * pi * tau));
            else u_inj_e = U * tau;
        }

        for (int i = 0; i < N; ++i) {
            const double th = yt[i];
            const double dw = yt[N + i];
            double coupling = 0.0;
            for (int j = 0; j < N; ++j) {
                const double Bij = B_flat[i * N + j];
                if (Bij != 0.0) coupling += Bij * sin(th - yt[j]);
            }
            double P = P_m[i];
            if (i == cont_bus) P += cont_mag;
            double u = 0.0;
            if (i == ctrl_bus) u = u_inj_e;
            const double decay = K_bus[idx * N + i] / (2.0 * pi) + D_nodes[i];
            k4[i] = dw;
            k4[N + i] = (P - coupling - decay * dw + u) / M_bus[idx * N + i];
        }
        for (int i = 0; i < 2 * N; ++i) {
            y[i] = y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }

        // Metrics from ω deviation
        for (int i = 0; i < N; ++i) {
            const double dw = y[N + i];
            const double df = dw / (2.0 * pi);
            if (df < nadir) nadir = df;
            const double rocof = fabs((dw - omega_prev[i]) / (2.0 * pi) / dt);
            if (rocof > rocof_max) rocof_max = rocof;
            omega_prev[i] = dw;
        }
    }

    out_rocof[idx] = rocof_max;
    out_nadir[idx] = nadir;
}
"""

# Lazy compile: PyTorch training steals/shares the primary CUDA context and
# invalidates kernels compiled at import time (cuFuncSetBlockShape / invalid handle).
_mod = None
_ctrl_kernel = None
_primary_ctx = None

_SHAPE_CODE = {"step": 0, "hann": 1, "ramp": 2}


def _ensure_ctrl_kernel(*, force_recompile: bool = False, manage_ctx: bool = True):
    """Compile (or recompile) the control kernel on the device primary context."""
    global _mod, _ctrl_kernel, _primary_ctx
    cuda.init()
    if _primary_ctx is None:
        # Share Torch's primary context when both use the same device.
        try:
            import torch

            dev_id = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        except Exception:
            dev_id = 0
        _primary_ctx = cuda.Device(dev_id).retain_primary_context()
    if _ctrl_kernel is None or force_recompile:

        def _compile():
            global _mod, _ctrl_kernel
            _mod = SourceModule(CUDA_CONTROL_KERNEL)
            _ctrl_kernel = _mod.get_function("simulate_control_metrics")

        if manage_ctx:
            _primary_ctx.push()
            try:
                _compile()
            finally:
                _primary_ctx.pop()
        else:
            _compile()
    return _ctrl_kernel


class _PrimaryCtx:
    def __enter__(self):
        if _primary_ctx is None:
            _ensure_ctrl_kernel()
        _primary_ctx.push()
        return _primary_ctx

    def __exit__(self, *exc):
        _primary_ctx.pop()
        return False


class CudaControlEngine:
    """GPU control-only batch evaluator sharing system matrices with the probe bank."""

    def __init__(self, sim: SwingSimulator, spec: ControlSpec):
        self.sim = sim
        self.spec = spec
        self.N = int(sim.N)
        if self.N > 14:
            raise ValueError("CudaControlEngine supports N<=14 (fixed state buffer)")
        self._B = np.ascontiguousarray(sim.B.astype(np.float64).reshape(-1))
        self._P_m = np.ascontiguousarray(sim.P_m.astype(np.float64))
        self._D = np.ascontiguousarray(sim.D_nodes.astype(np.float64))
        self._theta0 = np.ascontiguousarray(sim.theta0.astype(np.float64))
        self._omega0 = np.ascontiguousarray(sim.omega0.astype(np.float64))
        self.ode_dt = float(spec.ode_dt)
        self.T_obs = float(spec.T_obs_sec)
        self.n_steps = int(math.ceil(self.T_obs / self.ode_dt))
        _ensure_ctrl_kernel()

    def simulate_metrics_batch(
        self,
        M_rows: np.ndarray,
        K_rows: np.ndarray,
        u_mags: np.ndarray,
        *,
        batch_size: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            M_rows, K_rows: (n_traj, N)
            u_mags: (n_traj,) supplementary injection magnitudes [pu]
        Returns:
            rocof_max (n_traj,), delta_f_nadir (n_traj,)
        """
        M_rows = np.ascontiguousarray(M_rows, dtype=np.float64)
        K_rows = np.ascontiguousarray(K_rows, dtype=np.float64)
        u_mags = np.ascontiguousarray(np.asarray(u_mags, dtype=np.float64).reshape(-1))
        n_traj = int(u_mags.shape[0])
        if M_rows.shape != (n_traj, self.N) or K_rows.shape != (n_traj, self.N):
            raise ValueError("M_rows/K_rows must be (n_traj, N)")

        rocof = np.zeros(n_traj, dtype=np.float64)
        nadir = np.zeros(n_traj, dtype=np.float64)
        block = 128
        shape_code = int(_SHAPE_CODE[self.spec.profile.shape])
        prof = self.spec.profile
        cont = self.spec.contingency
        kernel = _ensure_ctrl_kernel()

        with _PrimaryCtx():
            for start in range(0, n_traj, batch_size):
                end = min(n_traj, start + batch_size)
                n_batch = end - start
                out_r = np.zeros(n_batch, dtype=np.float64)
                out_n = np.zeros(n_batch, dtype=np.float64)
                args = (
                    np.int32(n_batch),
                    np.int32(self.n_steps),
                    np.int32(self.N),
                    cuda.In(M_rows[start:end]),
                    cuda.In(K_rows[start:end]),
                    cuda.In(self._B),
                    cuda.In(self._P_m),
                    cuda.In(self._D),
                    cuda.In(self._theta0),
                    cuda.In(self._omega0),
                    cuda.In(u_mags[start:end]),
                    np.int32(cont.bus),
                    np.float64(cont.magnitude),
                    np.int32(prof.bus),
                    np.float64(prof.t_start),
                    np.float64(prof.duration),
                    np.int32(shape_code),
                    np.float64(self.ode_dt),
                    cuda.Out(out_r),
                    cuda.Out(out_n),
                )
                try:
                    kernel(
                        *args,
                        block=(block, 1, 1),
                        grid=((n_batch + block - 1) // block, 1),
                    )
                except cuda.LogicError:
                    kernel = _ensure_ctrl_kernel(force_recompile=True, manage_ctx=False)
                    kernel(
                        *args,
                        block=(block, 1, 1),
                        grid=((n_batch + block - 1) // block, 1),
                    )
                rocof[start:end] = out_r
                nadir[start:end] = out_n
        return rocof, nadir

    def evaluate_one(
        self,
        M: np.ndarray,
        K: np.ndarray,
        u_mag: float,
    ) -> dict[str, float]:
        M_row = np.ascontiguousarray(np.asarray(M, dtype=np.float64).reshape(1, -1))
        K_row = np.ascontiguousarray(np.asarray(K, dtype=np.float64).reshape(1, -1))
        r, n = self.simulate_metrics_batch(M_row, K_row, np.array([u_mag]))
        rocof_max = float(r[0])
        delta_f_nadir = float(n[0])
        safe = is_control_safe(rocof_max, delta_f_nadir, self.spec)
        return {
            "rocof_max": rocof_max,
            "delta_f_nadir": delta_f_nadir,
            "frequency_nadir": delta_f_nadir,
            "rocof_safe": float(rocof_max <= self.spec.rocof_limit_hz_s),
            "nadir_safe": float(delta_f_nadir >= self.spec.delta_f_nadir_hz),
            "safe": float(safe),
            "safe_total": float(safe),
            "u_ctrl": float(u_mag),
        }
