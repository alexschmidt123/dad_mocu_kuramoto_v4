# Test suite: Experimental design pipeline (PI-ready)

Validates **Design ξ → forward simulation (2nd-order Kuramoto / IEEE-14) → observation y (ROCOF) → prior→posterior sharpening**. No MOCU predictor, no DAD policy, no sequential decisions.

## How to run

```bash
pytest -v -s tests/test_experiment_design_pipeline.py
```

All outputs are written under `tests/output/`.

## Output files (what PI should look at)

| File | Description |
|------|-------------|
| **design_comparison_table.csv** | **140 rows:** B ∈ {1..14}, A ∈ {0.05, 0.1, …, 0.5} (10 values). Columns: `bus`, `amplitude`, `ROCOF_max`, `f_min`, `var_M_post`, `info_gain`. |
| **rocof_timeseries_by_bus.png** | **Same A, different B:** 6 subplots; 6 curves per subplot (B=1, 4, 7, 10, 13, 14). Blue window = **probing time (0–2 s)** with in-plot label and legend. |
| **rocof_timeseries_by_amplitude.png** | **Same B, different A:** 6 subplots; 6 curves per subplot. Blue window = **probing time (0–2 s)** with in-plot label and legend. |
| **posterior_marginals_by_design.png** | **Two panels:** prior vs posterior p(M) and p(K) for **5 designs**. Legend: design1: A=0.3, B=1; design2: A=0.3, B=7; design3: A=0.3, B=14; design4: A=0.1, B=7; design5: A=0.5, B=7. Y-axis = probability density. |
| **ieee14_diagram.png** | **IEEE 14-bus network diagram:** nodes 1..14 and transmission lines (from coupling matrix). |
| **probe_signal_wave.png** | **In-use probe signal:** Hann window with A=0.3, duration 2 s; time axis 0–2.5 s. Neat wave curve only (no blue shadow). |

**Time axis and probe highlight in ROCOF plots:** The x-axis is **simulation time** in seconds (0..T = 5 s). The **blue shaded region (0–2 s)** is labeled **"probing time"** and appears in the legend as "Probing time (0–2 s)". After 2 s the system is in free response.

## Tests implemented

1. **test_single_design_produces_observation** – One design (B=1, A=0.2); assert obs has ROCOF_max, f_min and omega_traj shape (., 14).
2. **test_design_comparison_table_saved** – Loop over 140 design candidates (B 1..14, A 10 values); save full table; assert 140 rows and ≥2 distinct ROCOF_max.
3. **test_rocof_timeseries_by_bus_plot** – ROCOF(t) by bus (fixed amplitude); save PNG.
4. **test_rocof_timeseries_by_amplitude_plot** – ROCOF(t) by amplitude (fixed bus); save PNG.
5. **test_posterior_sharpens_plot** – Prior + posterior p(M) and p(K) for 5 designs; legend entries like "design1: A=0.3, B=1"; assert posterior variance < prior for at least one design.
6. **test_ieee14_diagram_plot** – IEEE 14-bus network diagram; save `ieee14_diagram.png`.
7. **test_probe_signal_wave_plot** – Probe signal (Hann, A=0.3); save `probe_signal_wave.png`.
