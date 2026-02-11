# Brief Summary: Original .mdl, Dynamic .mdl, Python ODE — Comparison and Validation

---

## 1. Detailed analysis of all current original .mdl and dynamic .mdl results

### 1.1 Original .mdl (Fourteen_bus) — current results

**Location:** `matlab/results/fourteen_bus/`

| File | Description and content |
|------|--------------------------|
| **tout.csv** | Time vector from Simulink (one column). Typical: **1000 points**, range **[0.04, 0.12] s** (StopTime 0.12 s). No header; each row is one time value. |
| **summary.txt** | Short text: model name, save date, tout range and point count. Optional: lines for xout/yout if those were saved. |
| **xout.csv**, **yout.csv** | Present only if the model exports state (xout) and output (yout) to workspace; run script writes them when variables exist. |
| **xout_vs_time.png**, **yout_vs_time.png** | Optional PNGs: state or output vs time, if xout/yout exist. |

**Interpretation:** The original .mdl is a **short (0.12 s) run** for steady-state or initial transient. It does **not** log per-bus voltage to CSV; you get tout and, if configured, xout/yout. So for the original .mdl we have **no ROCOF_max or f_min** — only time and whatever signals the run script saves. Use as **reference topology and time base** only.

---

### 1.2 Dynamic .mdl (Fourteen_bus_dynamic) — current results

**Location:** `matlab/results/fourteen_bus_dynamic/`

| File | Description and content |
|------|--------------------------|
| **summary.txt** | Model name (Fourteen_bus_dynamic), save date, then **one line per bus**: e.g. `ScopeBusN: time [4.56144, 5] s, 5000 points`. So scope data covers the **tail** of the 5 s run (about 4.56–5 s), 5000 points per bus. |
| **ScopeBus1.csv … ScopeBus14.csv** | Per-bus **voltage time series**. **No header.** Column 0 = time (s); columns 1–3 = three signals per bus (e.g. 3-phase V_a, V_b, V_c). Typical: **5000 rows**, time from ~4.56 to 5 s. Values in roughly [−1, 1] (normalized or per-unit). |
| **ScopeBus1.png … ScopeBus14.png** | One PNG per bus: time vs the 3 signal columns (voltage components). |
| **all_buses.png** | Single figure with 14 subplots: first voltage component vs time for each bus (overview of all buses). |
| **observation_from_voltage.csv** | **Derived** (not from Simulink directly): run `python matlab/derive_observation_from_voltage.py` to create it. Columns: **bus**, **ROCOF_max** (Hz/s), **f_min** (Hz). One row per bus (1–14). Current typical values: ROCOF_max in **[0.012, 0.028] Hz/s**, f_min in **[49.99977, 49.99992] Hz** (very close to 50 Hz). |

**Interpretation:** The dynamic .mdl gives **5 s simulation** but scope logging may only store the **last portion** (e.g. 4.56–5 s) depending on Simulink/scope settings. From that voltage we **derive** ROCOF_max and f_min using the same math as Python (see §5 in this document). The derived observation shows **small ROCOF** and **f_min ≈ 50 Hz** because the .mdl has **no probe** — only electrical dynamics and small frequency deviation.

**Why you see “ODE things” in the dynamic .mdl — is it reasonable?**  
Yes. The dynamic .mdl **is** a dynamic simulation: Simulink uses an **ODE solver** (e.g. **ode23t**) to integrate **differential equations** from 0 to 5 s. The physics (Simscape/Power System Blocks) are **electrical ODEs**: inductor currents, capacitor voltages, and source dynamics evolve according to d(state)/dt = f(state, t). So you correctly see “ODE things” in the sense of: (1) continuous-time integration, (2) state evolving over time, (3) transients in voltage/current. What the .mdl does **not** implement is the **reduced swing ODE** (phase θ, frequency ω, M dω/dt = …) or the **Hann probe** that the Python code uses. So: **ODE in the .mdl = electrical network ODEs** (voltages/currents); **ODE in Python = swing equation** (θ, ω). Both are ODEs; the **equations** are different. It is reasonable that the dynamic .mdl shows ODE behavior — it is just a different (more detailed) dynamic model.

**Can you do a probe operation on the dynamic .mdl?**  
**Yes, but only by editing the .mdl.** The current dynamic .mdl has **no** probe (no disturbance injection). To add a probe you would:

1. **Open the model** in Simulink (`Fourteen_bus_dynamic.mdl`).
2. **Choose a bus** to probe (e.g. bus 1 to match Python design (1, 0.3)).
3. **Inject a disturbance** at that bus with the same **time shape** as the Python probe (Hann window: 0.5·A·(1 − cos(2πt/T_p)) for t ≤ T_p = 2 s, 0 otherwise). In Simulink you can:
   - Use a **Controlled Voltage Source** or **Controlled Current Source** (Simscape Electrical) at the bus, driven by a signal **u(t)** from a **Clock** + **MATLAB Function** block that outputs the Hann window, or
   - Use a **Three-Phase Programmable Voltage Source** (or similar) whose amplitude is modulated by the Hann signal, or
   - Add a **current injection** in parallel at the bus, with the injection current proportional to the Hann signal.
4. **Set A and T_p** (e.g. A = 0.3, T_p = 2 s) in the block parameters or in the MATLAB Function.
5. **Run** the simulation and save scope data as now; then run `derive_observation_from_voltage.py` to get ROCOF_max and f_min from the **probed** voltage.

Then you would get **larger ROCOF** and **lower f_min** from the dynamic .mdl (closer in spirit to Python), though the **physics** still differ (electrical vs swing ODE), so numerical match is not guaranteed. The repo does **not** include a pre-modified “probed” .mdl; you would create a copy (e.g. `Fourteen_bus_dynamic_probe.mdl`) and add the injection blocks yourself.

---

## 2. Detailed analysis of all current Python ODE test results and plots

**Location:** `tests/output/` (filled by `pytest tests/test_experiment_design_pipeline.py`).

### 2.1 Table: design_comparison_table.csv

| Column | Meaning |
|--------|--------|
| bus | Probe bus (1–14). |
| amplitude | Probe amplitude A (0.05, 0.1, …, 0.5). |
| ROCOF_max | Max \|d(Δf)/dt\| (Hz/s) over the observation window (12 Hz, 5 s). |
| f_min | Minimum absolute frequency (Hz) = 50 + min(Δf). |
| var_M_post | Posterior variance of M (inertia) after observing ROCOF_max for this design. |
| var_K_post | Posterior variance of K (gain) after observing ROCOF_max. |
| info_gain | Information gain in nats: H(prior) − H(posterior). |

**Content:** **140 rows** (14 buses × 10 amplitudes). Each row is one **design** (bus, amplitude): one ODE run with that probe, then ROCOF_max and f_min extracted, then posterior on a 7×7 (M,K) grid, then variances and info gain. **Typical ranges:** ROCOF_max ~4–8 Hz/s, f_min ~49.1–49.7 Hz (probe causes large ROCOF and frequency drop). Use this table for design comparison and for numerical comparison with Simulink-derived observation (see §5).

---

### 2.2 Plots (all generated by the test pipeline)

| File | What it shows | How it is produced |
|------|----------------|-------------------|
| **rocof_timeseries_by_bus.png** | **ROCOF(t)** (Hz/s) vs time. **6 subplots** (one per amplitude A ∈ {0.05, 0.1, 0.2, 0.3, 0.4, 0.5}). In each subplot, **6 curves** (one per bus B ∈ {1, 4, 7, 10, 13, 14}). Blue shaded region: **probe interval 0–2 s**. Red dotted line: **r_max** reference (e.g. 0.1 Hz/s). | For each (A, B) the test runs the ODE, computes ROCOF time series at 12 Hz (max over buses at each time), plots ROCOF(t). So: **same A, different B** to see effect of probe bus. |
| **rocof_timeseries_by_amplitude.png** | **ROCOF(t)** vs time. **6 subplots** (one per bus B ∈ {1, 4, 7, 10, 13, 14}). In each subplot, **6 curves** (one per amplitude A). Same probe interval and r_max reference as above. | Same ROCOF(t) computation; layout is **same B, different A** to see effect of probe amplitude. |
| **posterior_marginals_by_design.png** | **Prior vs posterior marginals** for M and for K. **Two panels:** (1) p(M\|y,ξ) for 5 designs (e.g. (1,0.3), (7,0.3), (14,0.3), (7,0.1), (7,0.5)); (2) p(K\|y,ξ) for the same. Gray dashed = **prior** (uniform); black vertical line = **M_true** or **K_true**. | For each design, run ODE → ROCOF_max → posterior on 55×55 grid → marginalize to M and K; plot posterior density. Test asserts at least one design **sharpens** the posterior (variance decreases). |
| **posterior_2d_design1.png** | **2D heatmap** of posterior **p(M,K\|y,ξ)** for a **single design** (bus 1, amplitude 0.3). **Red star** = (M_true, K_true). Axes: M (inertia), K (gain). | One ODE run for (1, 0.3) → ROCOF_max → posterior on 41×41 grid → pcolormesh; mark true parameter. Use to check that (M_true, K_true) lies in a high-probability region. |
| **ieee14_diagram.png** | **IEEE 14-bus network diagram**: nodes and edges from the same coupling matrix as the swing ODE (slack=1, gen=2,3,6,8, rest=load). Colors: slack (gold), generator (green), load (blue). | No ODE run; builds graph from `generate_ieee14_coupling_matrix(1.0)`, fixed layout, one PNG. Reference topology for reports. |
| **probe_signal_wave.png** | **Probe signal** (Hann window) vs time: u(t) = A·0.5·(1−cos(2πt/T_p)) for t ≤ T_p (2 s), 0 otherwise; A=0.3. Time axis 0–2.5 s. | No ODE run; plots the analytical probe used in the pipeline. Documents probe shape. |

**Summary:** The **table** gives the numerical design comparison (ROCOF_max, f_min, posteriors); the **plots** give ROCOF time evolution by bus/amplitude, prior vs posterior marginals, one 2D posterior, the network diagram, and the probe waveform. Together they show that the Python ODE pipeline is consistent and that observations (ROCOF_max, f_min) update the posterior over M and K as expected.

---

## 3. How each works (settings, probing, results)

### 3.1 Original .mdl (Fourteen_bus.mdl)

| Item | Value |
|------|--------|
| **What it is** | Simulink/Simscape IEEE 14-bus **electrical** model (3-phase sources, lines, loads). |
| **Initial/parameters** | Solver ode23t; 50 Hz sources; R, L, Pref from .mdl; no user-set θ₀, ω₀ (Simulink initializes from power flow). |
| **Probing** | **None.** Only network dynamics from initial conditions. |
| **StopTime** | **0.12 s** (short run). |
| **Scope logging** | Off (or default); run script may save tout only. |
| **Result after run** | Short transient; `results/fourteen_bus/` (e.g. tout.csv, summary.txt). **No** per-bus voltage CSV unless script adds it. Use as **steady-state / reference** only. |

---

### 3.2 Dynamic .mdl (Fourteen_bus_dynamic.mdl)

| Item | Value |
|------|--------|
| **What it is** | **Same** physics as original .mdl; copy with longer run and scope logging. |
| **Initial/parameters** | Same as original (ode23t, 50 Hz, same R, L, Pref). |
| **Probing** | **None.** Same as original; no Hann probe or external injection. |
| **StopTime** | **5 s** (aligned with Python T=5). |
| **Scope logging** | **On** for all 14 buses → ScopeBus1 … ScopeBus14 (time + 3 signals per bus, e.g. 3-phase voltage). |
| **Result after run** | `matlab/results/fourteen_bus_dynamic/`: **ScopeBus1.csv … ScopeBus14.csv** (time + 3 columns per bus), summary.txt, PNGs. Each CSV = voltage time series (e.g. V_a, V_b, V_c) over the simulated window. |

---

### 3.3 Python ODE (swing equation)

| Item | Value |
|------|--------|
| **What it is** | **Reduced** model: swing ODE (θ, ω), one ODE per bus; B, P_m, D, M, K, g from `swing_equation_params` (IEEE 14). |
| **Initial/parameters** | θ₀ = 0, ω₀ = 0 (or from conftest); T=5 s, h=1/160 s, f_nominal=50 Hz; M,K from prior (e.g. M∈[0.01,0.06], K∈[0.05,0.5]); B, P_m, D, g fixed. |
| **Probing** | **Yes.** Hann window u_probe(t) = A·0.5·(1−cos(2πt/T_p)) at **one** bus (design = bus + amplitude A); T_p=2 s. |
| **Result after run** | **ROCOF_max** (Hz/s), **f_min** (Hz) per design; optionally θ(t), ω(t). Pipeline: `pytest tests/test_experiment_design_pipeline.py` → `tests/output/design_comparison_table.csv` (140 rows: 14 buses × 10 amplitudes) with columns bus, amplitude, ROCOF_max, f_min, var_M_post, var_K_post, info_gain. |

---

## 4. Numerical comparison of results

| Source | ROCOF_max (Hz/s) | f_min (Hz) | Note |
|--------|-------------------|------------|------|
| **Original .mdl** | — | — | No ROCOF/f_min produced; voltage only, 0.12 s; not used for this comparison. |
| **Dynamic .mdl** (derived from voltage, see §3) | **~0.01–0.03** | **~49.9998–50** | 14 buses; no probe ⇒ small deviation. |
| **Python ODE** | **~4–8** | **~49.1–49.7** | 140 designs (bus × amplitude); probe ⇒ large ROCOF and f_min drop. |

**Conclusion:** Same observation **definition** (ROCOF_max, f_min at 12 Hz, 50 Hz nominal), but **different dynamics** (electrical vs swing + probe) ⇒ **numerical values differ by design**. Simulink gives small ROCOF and f_min ≈ 50; Python gives large ROCOF and lower f_min when probing.

Run side-by-side:  
`python matlab/compare_matlab_python_results.py`  
(requires `observation_from_voltage.csv` from §3 and `design_comparison_table.csv` from pytest).

---

## 5. Getting ROCOF_max and f_min from the dynamic .mdl (math and comparison)

### 5.1 What the dynamic .mdl gives you

After running `run_fourteen_bus_dynamic_save.m`, each **ScopeBusN.csv** has:

- **Column 0:** time t (s)  
- **Columns 1–3:** 3-phase voltage (e.g. V_a, V_b, V_c) per bus.

So the raw result is **voltage time series**, not frequency or ROCOF. We **derive** ROCOF_max and f_min to match the Python observation definition.

### 5.2 Math: from voltage to ROCOF_max and f_min

Same definitions as in Python (`extract_frequency_features` + ROCOF), so the comparison is apples-to-apples.

**Step 1 — Phase from 3-phase voltage (Clarke + angle)**  
For each bus, from V_a, V_b, V_c:

- **Clarke (αβ):**  
  V_α = (2/3)(V_a − ½V_b − ½V_c),  
  V_β = (1/√3)(V_b − V_c).

- **Phase (rad):**  
  φ(t) = atan2(V_β, V_α), then **unwrap** φ so it is continuous in t.

**Step 2 — Instantaneous frequency and deviation**

- **Instantaneous frequency (Hz):**  
  f(t) = (1/(2π)) · dφ/dt  
  (numerically: `gradient(φ, t) / (2π)`).

- **Frequency deviation:**  
  Δf(t) = f(t) − f_nominal,  
  with **f_nominal = 50 Hz**.

**Step 3 — Downsample to observation rate (match Python)**

- Resample (t, Δf) to a **uniform grid at fs = 12 Hz** over the same time window (e.g. interpolate onto t_obs with spacing h_obs = 1/12 s).

**Step 4 — ROCOF_max and f_min (same as Python)**

- **ROCOF:**  
  ROCOF(t) = d(Δf)/dt at the 12 Hz grid.  
  **ROCOF_max** = max_t |ROCOF(t)| (Hz/s).

- **f_min:**  
  **f_min** = f_nominal + min_t Δf(t) = 50 + min(Δf) (Hz).

This is implemented in **`matlab/derive_observation_from_voltage.py`**. Run after the dynamic .mdl:

```bash
python matlab/derive_observation_from_voltage.py
```

→ Reads ScopeBus1.csv … ScopeBus14.csv, applies the steps above per bus, prints ROCOF_max and f_min per bus, and writes **`matlab/results/fourteen_bus_dynamic/observation_from_voltage.csv`**.

### 5.3 Final comparison

- **Script:**  
  `python matlab/compare_matlab_python_results.py`  
  loads `observation_from_voltage.csv` (Simulink-derived) and `design_comparison_table.csv` (Python), then prints:
  - Ranges of ROCOF_max and f_min for both,
  - Short explanation of why they differ (no probe vs probe).

- **Interpretation:**  
  - **Same formulas** for ROCOF_max and f_min (12 Hz, 50 Hz, same downsampling and max/min).  
  - **Different physics:** dynamic .mdl = electrical, no probe; Python = swing ODE with probe.  
  - So **numerical agreement is not expected**; the comparison checks that both sides use the same observation definition and that the **ranges** are consistent with “no probe” vs “probe”.

---

## 6. Is the Python ODE reliable? (Validation conclusion)

- **Structural checks (no Simulink needed)**  
  - Python pipeline is **internally consistent**: ODE → ω(t) → ROCOF_max, f_min (same formulas as above) → likelihood → posterior.  
  - design_comparison_table.csv has plausible ranges (ROCOF in Hz/s, f_min in 49–50 Hz); posteriors sharpen for some designs; (M_true, K_true) lie in high-probability regions in 2D posterior plots.  
  → This supports that the **Python ODE and observation pipeline are implemented correctly** for the swing + probe model.

- **Comparison with dynamic .mdl**  
  - We **cannot** validate “same numbers” because the models differ (electrical vs swing, no probe vs probe).  
  - We **can** validate:  
    - **Same observation type:** ROCOF_max and f_min from dynamic .mdl (via voltage → φ → f → ROCOF, f_min) use the **same math** as Python.  
    - **Qualitative consistency:** Simulink gives small ROCOF and f_min ≈ 50; Python gives large ROCOF and lower f_min under probe, which is **expected** from the physics.  
  - So the dynamic .mdl supports that the **observation definition** is consistent and that the **order of magnitude** of responses is reasonable (probe causes much larger ROCOF and frequency drop than unprobed Simulink).

**Summary:**  
- **Reliability of the Python ODE (and ROCOF/f_min):** Supported by (1) internal consistency and plausible outputs, and (2) same observation definition and qualitatively consistent comparison with the dynamic .mdl.  
- **Not claimed:** Numerical match between Simulink and Python (different models and excitation). The Python ODE is **reliable for the reduced swing + probe model**; the dynamic .mdl is used as a **reference for observation definition and qualitative behavior**, not for point-by-point validation of the same dynamics.
