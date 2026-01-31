# ----------------------------
# Paper-grounded fixed probe settings (HICSS 2024)
# ----------------------------
PROBE = {
  "waveform": "hann",     # smooth windowed probe
  "T_sec": 2.0,           # keep energy < 0.5 Hz
  "A_max": 0.05,          # example magnitude used in paper
  "fs_hz": 12.0,          # sampling frequency
  "rocof_window_sec": 0.5,# 500 ms sliding window
  "rocof_eval_sec": 1.0   # evaluate max ROCOF over first 1 s
}

# ----------------------------
# Core sequential design loop (2–3 steps)
# ----------------------------
def sequential_location_design(K_steps, candidate_buses, prior_belief):
    belief = prior_belief  # p(theta)

    for k in range(1, K_steps+1):

        # 1) Select next probing location (bus) by expected utility
        # Utility should reflect how much ROCOF would reduce uncertainty in theta
        best_bus = None
        best_score = -float("inf")

        for bus in candidate_buses:
            # Predict distribution of ROCOF observation y if we probe at this bus:
            # y ~ p(y | bus, belief, PROBE)
            # Use either:
            #   - a fast surrogate model, or
            #   - a small set of forward simulations sampled from belief
            score = expected_information_from_rocof(bus, belief, PROBE)

            if score > best_score:
                best_score = score
                best_bus = bus

        # 2) Run experiment at selected bus (inject probe at IBR Pref)
        raw_freq = run_probe_and_collect_frequency(best_bus, PROBE)

        # 3) Convert raw frequency to ROCOF observation (your choice)
        y_rocof = max_rocof(raw_freq,
                            fs=PROBE["fs_hz"],
                            win=PROBE["rocof_window_sec"],
                            horizon=PROBE["rocof_eval_sec"])

        # 4) Bayesian update: belief <- p(theta | y_rocof, bus, PROBE)
        belief = bayes_update_inertia(belief, y_rocof, best_bus, PROBE)

        # (optional) log for debugging
        print(f"step={k}, bus={best_bus}, rocof={y_rocof}, score={best_score}")

    return belief


# ----------------------------
# Helper: ROCOF extraction aligned with HICSS method
# ----------------------------
def max_rocof(freq_series, fs, win, horizon):
    # freq_series: array of f(t) starting at probe onset
    # compute df/dt via linear fit in a sliding window of length win
    # evaluate max |df/dt| over first 'horizon' seconds
    N = int(horizon * fs)
    W = int(win * fs)

    rocof_vals = []
    for i in range(0, max(1, N - W + 1)):
        segment = freq_series[i:i+W]
        # simple least squares slope for df/dt
        slope = linear_slope(segment, fs)
        rocof_vals.append(abs(slope))

    return max(rocof_vals)


# ----------------------------
# Helper: Expected utility for a location (simple prototype)
# ----------------------------
def expected_information_from_rocof(bus, belief, PROBE):
    # simplest version: Monte Carlo
    # 1) sample theta ~ belief
    # 2) simulate predicted ROCOF y for each theta at this bus
    # 3) compute reduction in variance / entropy proxy
    # Replace with your MOCU utility once you have your control objective wired.
    samples = belief.sample(64)
    y_samples = [simulate_rocof(theta, bus, PROBE) for theta in samples]
    return variance_reduction_proxy(belief, y_samples)
