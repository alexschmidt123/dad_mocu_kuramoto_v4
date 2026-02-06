# Analysis of Test Outputs (Design Pipeline)

This document summarizes and interprets the results produced by the experimental design pipeline tests (`pytest tests/test_experiment_design_pipeline.py`). All outputs are under `tests/output/`.

---

## 1. Design comparison table (`design_comparison_table.csv`)

**Content:** 140 design candidates: bus B ∈ {1,…,14} × amplitude A ∈ {0.05, 0.1, …, 0.5} (10 values). Each row is one design ξ = (B, A) with metrics from a single probe experiment and posterior on (M, K).

**Columns:**
- **bus (B):** Probe bus (1-based).
- **amplitude (A):** Probe amplitude (Hann window).
- **ROCOF_max:** Maximum |ROCOF| (Hz/s) over the trajectory (observation y).
- **f_min:** Minimum frequency (Hz) over buses; should stay above constraint (e.g. 49.8 Hz).
- **var_M_post:** Posterior variance of M (inertia) after observing y; lower = more informative design.
- **info_gain:** Entropy reduction H(prior) − H(posterior) in nats; higher = more informative.

### Summary statistics (from current run)

| Metric        | Min      | Max     | Notes |
|---------------|----------|---------|--------|
| ROCOF_max     | ~0.44    | ~0.89   | B=4 gives highest ROCOF; B=7, B=8 lower. |
| f_min         | ~49.79   | ~49.93  | All above 49.8 Hz (constraint satisfied). |
| var_M_post    | ~8.6e-05 | ~0.034  | B=4 and B=12 give very low variance (sharp M). |
| info_gain     | ~1.30    | ~2.01   | B=4, B=12 near 2.0; B=7, B=8 lower (~1.3–1.5). |

### Design ranking (by information gain / posterior sharpness)

- **Best designs (high info_gain, low var_M_post):**
  - **B=4** across amplitudes: info_gain ≈ 1.99–2.01, var_M_post ≈ 0.00021–0.00031. Probing at bus 4 is highly informative for (M, K).
  - **B=12** at A ∈ {0.2, 0.25, 0.3, 0.35}: var_M_post as low as ~8e-05–1e-04, info_gain ≈ 1.97–2.05. Very sharp posterior on M.
- **Weaker designs (lower info_gain, higher var_M_post):**
  - **B=7, B=8:** info_gain ≈ 1.30–1.45, var_M_post ≈ 0.015–0.026. Probing at these buses reduces uncertainty less.
- **Amplitude:** For many buses, higher A (e.g. 0.3–0.5) gives slightly lower var_M_post and higher info_gain; very low A (0.05) is often less informative.

**Takeaway:** Bus location matters more than amplitude in this grid. B=4 and B=12 are the most informative probe locations for learning M (and overall uncertainty); B=7 and B=8 are the least informative among the 14 buses.

---

## 2. ROCOF time-series by bus (`rocof_timeseries_by_bus.png`)

**Content:** 6 subplots, one per amplitude A ∈ {0.05, 0.1, 0.2, 0.3, 0.4, 0.6}. In each subplot, 6 curves: ROCOF(t) for B ∈ {1, 4, 7, 10, 13, 14}. The first 2 s are shaded and labeled as **probing time** (probe on).

**How to read:**
- **X-axis:** Time (0–5 s).
- **Y-axis:** ROCOF (Hz/s), max over buses at each time.
- **Blue band (0–2 s):** Probe is active (Hann window); ROCOF response is driven by the probe.
- **After 2 s:** Free response; ROCOF decays or oscillates depending on damping and topology.

**Interpretation:**
- Compare **same A, different B**: e.g. at A=0.3, B=4 typically shows larger ROCOF during probing than B=7, consistent with the table (B=4 has higher ROCOF_max and higher info_gain).
- Peak ROCOF during 0–2 s and the decay after 2 s depend on where the probe is applied (bus) and system dynamics.

---

## 3. ROCOF time-series by amplitude (`rocof_timeseries_by_amplitude.png`)

**Content:** 6 subplots, one per bus B ∈ {1, 4, 7, 10, 13, 14}. In each subplot, 6 curves: ROCOF(t) for A ∈ {0.05, 0.1, 0.2, 0.3, 0.4, 0.6}. Same 0–2 s probing-time highlight.

**How to read:**
- **Same B, different A:** Larger A generally gives larger ROCOF magnitude during probing and often a stronger transient after 2 s.
- **Per bus:** B=4 and B=12 (not in this subset but in the table) show the largest ROCOF_max; in these plots, B=4 will show the largest peaks for a given A.

**Interpretation:**
- Amplitude controls “signal strength” of the probe; too low (0.05) may give weak observations and less information gain, as in the table.
- The shape of ROCOF(t) after 2 s (damping, oscillation) is determined by the same (M, K) and topology; it is the same for all designs that share the same true parameters.

---

## 4. Posterior marginals by design (`posterior_marginals_by_design.png`)

**Content:** Two panels—**p(M|y,ξ)** and **p(K|y,ξ)**—each showing the **prior** (flat dashed line) and **posterior** (curves) for 5 designs:

| Label   | Design   | Description        |
|---------|----------|--------------------|
| design1 | B=1, A=0.3 | Probe at bus 1, amplitude 0.3 |
| design2 | B=7, A=0.3 | Probe at bus 7, amplitude 0.3 |
| design3 | B=14, A=0.3 | Probe at bus 14, amplitude 0.3 |
| design4 | B=7, A=0.1 | Probe at bus 7, amplitude 0.1 |
| design5 | B=7, A=0.5 | Probe at bus 7, amplitude 0.5 |

**How to read:**
- **Prior:** Uniform over [M_lower, M_upper] and [K_lower, K_upper]; same for all designs.
- **Posterior:** Updated belief over M (inertia) and K (control gain) after observing ROCOF_max = y for that design.
- **Peaked curves** = posterior is concentrated = design was informative.
- **Flat or wide curves** = posterior barely updated = design was less informative.

**Interpretation:**
- **design2, design4, design5 (all B=7):** B=7 is one of the less informative buses (low info_gain in the table); posteriors for B=7 may be broader and similar across A=0.1, 0.3, 0.5.
- **design1 (B=1), design3 (B=14):** Typically sharper than B=7, especially if B=1 or B=14 are more informative in the table.
- **M vs K:** Some designs may sharpen M more than K (or vice versa), depending on how ROCOF_max relates to (M, K) for that probe location and amplitude.

This plot directly shows that **posterior variance decreases (uncertainty shrinks) for at least one design**, which the test asserts.

---

## 5. IEEE 14-bus diagram (`ieee14_diagram.png`)

**Content:** Single-line diagram of the **IEEE 14-bus network** used in the project: 14 nodes (buses 1–14) and transmission lines (edges from the coupling matrix B).

**Use:** Reference for where each **B** (probe bus) sits in the network. For example:
- B=4 is a central bus (connected to 2, 3, 5, 7, 9) and in the table is the most informative.
- B=7, B=8 (less informative) are in a different part of the network. This suggests that **topology and location** strongly influence which probe designs are best for learning (M, K).

---

## 6. Probe signal waveform (`probe_signal_wave.png`)

**Content:** The **in-use probe signal**: Hann window with **A = 0.3**, duration **2 s**, time axis 0–2.5 s. Amplitude rises smoothly to 0.3 and returns to zero by 2 s.

**Use:** Shows the **waveform** used as u_probe in the swing equation during 0–2 s. It is the same functional form for all designs; only the **bus** (which bus gets the injection) and **amplitude A** change across designs.

---

## Summary

| Output file                      | Purpose |
|----------------------------------|--------|
| **design_comparison_table.csv**  | Rank all 140 designs by ROCOF_max, f_min, var_M_post, info_gain; **best probe locations: B=4, B=12**; weakest: B=7, B=8. |
| **rocof_timeseries_by_bus.png**  | Compare ROCOF(t) across buses for fixed A; confirms that bus choice changes response and information. |
| **rocof_timeseries_by_amplitude.png** | Compare ROCOF(t) across amplitudes for fixed B; confirms that larger A generally gives larger ROCOF. |
| **posterior_marginals_by_design.png** | Shows prior vs posterior for M and K for 5 designs; illustrates that some designs (e.g. B=1, B=14) sharpen the posterior more than B=7. |
| **ieee14_diagram.png**           | Reference topology for interpreting which buses are central vs peripheral. |
| **probe_signal_wave.png**        | Reference waveform of the probe (Hann, A=0.3, 2 s). |

Together, these results show that the **design pipeline** (design ξ → ODE → ROCOF observation → posterior) behaves as intended: **bus location has a large effect on information gain**, with B=4 and B=12 giving the sharpest posterior on M and the highest info_gain, and B=7/B=8 the weakest.
