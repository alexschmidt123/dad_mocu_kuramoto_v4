# ----------------------------
# Paper-grounded fixed probe settings (HICSS 2024 / NREL 2024)
# ----------------------------
PROBE = {
    "waveform": "hann",       # Hann window: 0.5 * (1 - cos(2πt/T))
    "T_p": 2.0,               # Duration: 2s (excites inertial/FFR dynamics)
    "A_set": [0.05, 0.1, 0.2],# Discrete amplitudes A
    "fs_hz": 12.0,            # Sampling frequency (PMU standard)
    "T_obs": 10.0,            # Observation window [0, 10]s
    "sigma": 0.01             # Measurement noise standard deviation
}

# ----------------------------
# Core sequential design loop (Decision-Aware DAD-MOCU)
# ----------------------------
def dad_sequential_design(T_horizon, candidate_buses, prior_particles):
    """
    Non-myopic sequential design optimizing terminal MOCU.
    Based on DAD framework.
    """
    history = []  # h_t = {(xi_1, y_1), ..., (xi_t, y_t)}
    particles = prior_particles # p(theta) where theta = (M, K)

    for t in range(1, T_horizon + 1):

        # 1) Select next probing action using DAD Policy
        # Policy pi_phi(h_{t-1}) uses a fast neural surrogate (MPNN/MLP) 
        # to estimate the Expected MOCU Matrix (R-matrix)
        action_scores = fast_mocu_estimator(particles, history, candidate_buses, PROBE)
        best_bus, best_amp = select_action_from_policy(action_scores)
        xi_t = (best_bus, best_amp, PROBE["T_p"])

        # 2) Run experiment (Simulator or Field)
        # Solve Batched Swing Equation ODE
        raw_omega = run_swing_equation_ode(xi_t, theta_true)

        # 3) Extract ROCOF-only observation y_t
        y_t = extract_max_rocof(raw_omega, fs=PROBE["fs_hz"], window=PROBE["T_obs"])

        # 4) Bayesian Update: Compute posterior particles via Likelihood
        # Likelihood p(y_t | theta, xi_t) = N(mu(theta, xi_t), sigma^2)
        history.append((xi_t, y_t))
        particles = update_particles_with_likelihood(particles, history, PROBE)

        # 5) Compute terminal MOCU for tracking (using Batched ODE for validation)
        # MOCU(h_T) = E[gamma*(A_T) - gamma*(theta)]
        current_mocu = compute_true_mocu(particles)
        print(f"step={t}, bus={best_bus}, amp={best_amp}, MOCU={current_mocu:.4f}")

    return particles

# ----------------------------
# Helper: ROCOF extraction (Design Part 1, Section 4)
# ----------------------------
def extract_max_rocof(omega_series, fs, window):
    """
    Extract peak ROCOF from frequency deviation.
    """
    delta_f = omega_series / (2.0 * 3.14159) # Convert to Hz
    dt = 1.0 / fs
    
    # Numerical derivative over discrete samples
    rocof_series = np.diff(delta_f, axis=0) / dt
    
    # Return max absolute ROCOF within the window
    return np.max(np.abs(rocof_series))

# ----------------------------
# Helper: Fast MOCU Estimator (Neural Proxy)
# ----------------------------
def fast_mocu_estimator(particles, history, candidate_buses, PROBE):
    """
    Accelerated look-ahead using MPNN/MLP surrogate.
    Predicts remaining MOCU for each candidate action (xi).
    """
    # 1) Calculate current uncertainty bounds (M_low, M_up, K_low, K_up)
    bounds = get_credible_set_bounds(particles)
    
    # 2) Feed bounds + history into Neural Surrogate to get expected MOCU reduction
    # This bypasses the expensive gamma_star binary search
    expected_mocu_matrix = surrogate_model.predict(bounds, history, candidate_buses)
    
    return expected_mocu_matrix