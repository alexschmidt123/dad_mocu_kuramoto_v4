# MATLAB/Simulink reference: IEEE 14-bus

This folder holds a **third-party Simulink model** of the IEEE 14-bus system for reference only. **All simulation and analysis in this project are done in Python** (see `src/`, `scripts/`, `documents/design.md`). We do not repeat the swing ODE or probing logic in MATLAB.

## What to run (Python test + MATLAB, aligned)

To run **both** and keep the **dynamic MATLAB run aligned with the Python test** (same time window 5 s, 50 Hz nominal):

1. **MATLAB (from repo root or from `matlab/`):**
   - **Steady-state (optional):** In MATLAB, `cd` to the `matlab/` folder, then run:
     ```matlab
     run('run_fourteen_bus_save.m')
     ```
     → Saves to `matlab/results/fourteen_bus/` (0.12 s run).
   - **Dynamic (aligned with Python T=5 s):** In MATLAB, same folder, run:
     ```matlab
     run('run_fourteen_bus_dynamic_save.m')
     ```
     → Saves to `matlab/results/fourteen_bus_dynamic/` (5 s run; StopTime in the .mdl is 5 s to match Python test).

2. **Python tests (from repo root):**
   ```bash
   pytest tests/posterior_inference/ -k experiment_design -v
   ```
   → Fills `tests/posterior_inference/output/design_comparison_table.csv` (140 designs, T=5 s, 50 Hz nominal; fixtures in `tests/posterior_inference/conftest.py`).

3. **Compare (from repo root):** MATLAB vs Python ODE (`tests/simulink_reference/`):
   ```bash
   python -m tests.simulink_reference.ode_validation
   ```
   or: `pytest tests/simulink_reference/ -v`
   → Runs Python ODE (probe at bus 1), derives MATLAB observation if needed, prints side-by-side comparison.

**Alignment:** The experiment-design test uses `T=5.0` s and 50 Hz nominal (`simulation_settings` in `tests/posterior_inference/conftest.py`). The dynamic .mdl uses **StopTime 5 s** and the .mdl is 50 Hz; both produce results over a 5 s window for comparison.

## Difference between fourteen_bus and fourteen_bus_dynamic

| | **fourteen_bus** (original) | **fourteen_bus_dynamic** |
|---|----------------------------|---------------------------|
| **File** | `fourteen_bus.mdl` | `fourteen_bus_dynamic.mdl` (copy with different settings) |
| **StopTime** | **0.12 s** | **5 s** (aligned with Python test T=5) |
| **Purpose** | Short transient / **steady-state reference** (Step 1): quick run to see equilibrium or initial transient. | **Dynamic run** (Step 2): longer window for comparison with Python; scope data over 0–5 s. |
| **Scope logging** | Off (or default); saves tout only in the run script. | **On** for all 14 buses → ScopeBus1 … ScopeBus14 in workspace and CSV in `results/fourteen_bus_dynamic/`. |
| **Run script** | `run_fourteen_bus_save.m` → `results/fourteen_bus/` | `run_fourteen_bus_dynamic_save.m` → `results/fourteen_bus_dynamic/` |

Same underlying model (same physics, 50 Hz, IEEE 14-bus); only **simulation length** and **scope logging** differ.

## What is here

**fourteen_bus.mdl** – Original IEEE 14-bus Simulink model from MATLAB Central File Exchange (Bharath Yk, [46067](https://www.mathworks.com/matlabcentral/fileexchange/46067-ieee-14-bus-system-simulink-model)). Short transient (StopTime 0.12 s). Use for **steady-state reference** (Step 1).

- **Open:** `open_system('fourteen_bus')`

**fourteen_bus_dynamic.mdl** – Same model, configured for **dynamic simulation** (Step 2): **StopTime 5 s** (aligned with Python test `T=5.0`), scope logging on for all 14 buses. When you run the simulation, bus scope data is saved to the MATLAB workspace as **ScopeBus1** … **ScopeBus14** (each with `.time` and `.signals`; use for mapping to θ/ω and comparison with Python).

- **Open:** `open_system('fourteen_bus_dynamic')`
- **Run and save results (CSV/TXT/PNG):** `run('run_fourteen_bus_save.m')` for steady-state; `run('run_fourteen_bus_dynamic_save.m')` for dynamic. Outputs go to `results/fourteen_bus/` and `results/fourteen_bus_dynamic/` respectively.

**fourteen_bus_dynamic_probe.mdl** – Copy of the dynamic model for **probing / OED validation** (same 5 s, same scopes). Run **`run_fourteen_bus_dynamic_with_probe_save.m`**: it adds the Hann-window probe (A=0.2, T_p=2 s), **ProbeInjection** (Controlled Current Source), **ProbeGround**, and **ProbeSnubber** (1 MΩ in parallel), wires the probe to Bus 1 and ground, then runs the sim and saves to `results/fourteen_bus_dynamic_probe/`. You should see probe influence (e.g. larger ROCOF, lower f_min during 0–2 s). See **Probe injection (active probing at bus 1)** below if wiring fails.

### ROCOF / ROCOF_max from MATLAB results

The **ScopeBus plots show voltage vs time**, not ROCOF. ROCOF (rate of change of frequency) and **ROCOF_max** are **derived** from voltage: phase from 3-phase voltage (Clarke) → instantaneous frequency → ROCOF. The run scripts **compute** these automatically (same definition as Python: 50 Hz nominal, 12 Hz observation rate) and write:

- **observation_from_voltage.csv** — per-bus `ROCOF_max` and `f_min`
- **summary.txt** — same values appended
- **ROCOF_bus1.png** (probe run only) — ROCOF(t) for bus 1 with ROCOF_max annotated

So you **do use calculation** to get ROCOF/ROCOF_max from MATLAB results; the scripts do it for you. For Python-side comparison, run `python -m tests.simulink_reference.ode_validation` (it can also derive from ScopeBus CSVs if the CSV was not written by MATLAB).

### Where are the results after Run?

1. **MATLAB Workspace** (after simulation finishes):
   - **ScopeBus1** … **ScopeBus14** — one struct per bus. Each has:
     - `ScopeBusN.time` — time vector
     - `ScopeBusN.signals` — struct array with signal values (e.g. voltage, angle)
   - **ScopeData** — from the main scope (if present)
   - In the MATLAB desktop, open the **Workspace** panel to see and double‑click these variables to inspect or plot.

2. **Scope windows** — Simulink Scope blocks open as figures during/after the run; you can view plots there.

3. **To plot in MATLAB:** e.g. `plot(ScopeBus1.time, ScopeBus1.signals.values)` (adjust `.signals` layout to your block output).

Both use **Power System Blocks** (Simscape Electrical). They are a detailed electrical network, not the same as the repo’s **reduced swing equation** (θ, ω, B, Hann probe) in Python. Use them as reference topology and for mapping/validation.

**Running with probe:** Use **fourteen_bus_dynamic_probe.mdl** and run **`run_fourteen_bus_dynamic_with_probe_save.m`**. That script adds the Hann-window probe blocks if needed, runs the sim, and saves ScopeBus + summary + PNGs to `results/fourteen_bus_dynamic_probe/`. To get **active probing at bus 1**, add injection as below.

### Probe injection (active probing at bus 1)

The run script **automatically** adds **ProbeInjection** (Controlled Current Source), **ProbeGround**, and **ProbeSnubber** (1 MΩ resistor in parallel), and wires ProbeInjection’s **+** to **Bus 1** (one phase) and **−** to Ground. The snubber is required by the SPS solver when a current source is in series with inductive branches. If automatic wiring fails (e.g. block names differ), connect manually: **ProbeInjection** **+** and **−** to the **Bus 1** node and **ground**, add a high-value resistor (e.g. 1 MΩ) in parallel, save, and re-run. Then run `python -m tests.simulink_reference.ode_validation` to compare with Python; you should see larger ROCOF and lower f_min during 0–2 s.

**Alternative:** Use a **Three-Phase Dynamic Load** with external P control at bus 1 and drive its P input from the HannProbe signal (scaled).

**Python (same idea, probe at bus 1):** `tests/simulink_reference/ode_validation.py` and `tests/simulink_reference/test_simulink_reference.py` run the IEEE 14 swing ODE with probe at bus 1 (A=0.2, Tp=2 s, T=5 s) and compare with MATLAB. Run: `python -m tests.simulink_reference.ode_validation` or `pytest tests/simulink_reference/ -v`.

## Relation to the project

| Item        | Python (this repo)                    | This folder (MATLAB)     |
|------------|----------------------------------------|---------------------------|
| Simulation | `src/core/`, swing ODE + probe, ROCOF  | Not duplicated here       |
| IEEE 14    | `swing_equation_params.py`              | fourteen_bus.mdl (reference)      |

Parameter conventions (M, K, D, f_nominal, ROCOF limits, etc.) are in `documents/Parameter_references_table.md` and `documents/design.md`. For **inspiration from the .mdl** (base values, P_m from Pref, B from line reactances, reference θ), see **`matlab/mdl_to_python_params.md`**.

## Letting scripts / CI / AI use MATLAB (e.g. on macOS)

So that tools (terminal, Cursor, scripts) can run MATLAB without the GUI:

1. **Use the full path to the executable** (if `matlab` is not on your PATH):
   ```bash
   /Applications/MATLAB_R2025b.app/bin/matlab -batch "cd('path/to/matlab'); run('run_fourteen_bus_dynamic_save.m')"
   ```
2. **Or add MATLAB to your PATH** (in `~/.zshrc`):
   ```bash
   export PATH="/Applications/MATLAB_R2025b.app/bin:$PATH"
   ```
   Then: `matlab -batch "cd('path/to/matlab'); run('run_fourteen_bus_dynamic_save.m')"`

Use `-batch` for non-interactive runs (exits when done). From the project root, the `matlab` folder is at `matlab/` relative to the repo.

## “Output Port 2 is not connected” warnings

The 14-bus model has Bus blocks with a second output port that is not wired. Simulink reports this as 14 “Output Port 2 … is not connected” messages.

- **Suppressed in this repo:** `fourteen_bus_dynamic.mdl` has **Unconnected block output ports** set to `none`, and the run scripts set `UnconnectedOutputMsg` to `'none'` before `sim()`, so these warnings no longer appear when you run via the script or open this .mdl.
- **To change it in the GUI:** Model Configuration Parameters → **Diagnostics** → **Connectivity** → **Unconnected block output ports** → set to `none` (or `warning` if you want to see them again).
- **If you still see the 15 warnings when you click Run:** Close the model (File → Close), then open it again from disk (`open_system('fourteen_bus_dynamic')`). The saved .mdl already has the diagnostic set to `none`; reopening loads that. Or run once in the Command Window: `set_param('fourteen_bus_dynamic','UnconnectedOutputMsg','none')` (with the model loaded), then click Run.

## Run scripts (3)

- **run_fourteen_bus_save.m** – Runs **fourteen_bus** (0.12 s steady-state). Saves results to `results/fourteen_bus/`: CSV (tout, xout, yout), summary.txt, PNG plots.
- **run_fourteen_bus_dynamic_save.m** – Runs **fourteen_bus_dynamic** (5 s, aligned with Python test T). Saves results to `results/fourteen_bus_dynamic/`: CSV (ScopeBus1…14), summary.txt, PNG plots (per bus + all_buses.png).
- **run_fourteen_bus_dynamic_with_probe_save.m** – Runs **fourteen_bus_dynamic_probe** (5 s with Hann-window probe). Adds probe blocks to the model if not present, then runs and saves to `results/fourteen_bus_dynamic_probe/`: ScopeBus1…14, summary.txt, PNGs (and ProbeOut if logged).

**Analysis vs Python:** After running the MATLAB scripts and `pytest tests/posterior_inference/ -k experiment_design -v`, run `python -m tests.simulink_reference.ode_validation` to run the Python ODE (probe at bus 1) and print a side-by-side comparison with MATLAB. See **`tests/simulink_reference/`**.

## Files (sorted)

All files in `matlab/`, grouped and sorted.

### Documentation
| File | Description |
|------|--------------|
| **instruction.md** | This file: how to run models, scripts, and comparison; includes **Probe injection (active probing at bus 1)**. |
| **mdl_to_python_params.md** | Mapping from .mdl (B, P_m, topology) to Python swing-equation parameters. |

### Simulink models (.mdl)
| File | Description |
|------|--------------|
| **fourteen_bus.mdl** | Original IEEE 14-bus (File Exchange 46067); StopTime 0.12 s, steady-state reference. |
| **fourteen_bus_dynamic.mdl** | Same model, 5 s run, scope logging (ScopeBus1…14) for validation vs Python. |
| **fourteen_bus_dynamic_probe.mdl** | Copy for OED/probing; probe blocks are added automatically by `run_fourteen_bus_dynamic_with_probe_save.m` when you run it. |

### MATLAB scripts (.m)
| File | Description |
|------|--------------|
| **run_fourteen_bus_save.m** | Runs fourteen_bus (0.12 s); saves to `results/fourteen_bus/`. |
| **run_fourteen_bus_dynamic_save.m** | Runs fourteen_bus_dynamic (5 s); saves to `results/fourteen_bus_dynamic/`. |
| **run_fourteen_bus_dynamic_with_probe_save.m** | Runs fourteen_bus_dynamic_probe (5 s with probe). Adds probe blocks if not present; saves to `results/fourteen_bus_dynamic_probe/`. |

### Validation tests (run from repo root)
| Test / script | Description |
|---------------|-------------|
| **tests/simulink_reference/** | Helpers (`ode_validation.py`) + pytest (`test_simulink_reference.py`): Python ODE (probe at bus 1), derives ROCOF_max/f_min from MATLAB ScopeBus CSVs when present. Run: `python -m tests.simulink_reference.ode_validation` · `pytest tests/simulink_reference/ -v`. |

### Results (generated by run scripts)
| Path | Description |
|------|--------------|
| **results/fourteen_bus/** | tout.csv, summary.txt (from run_fourteen_bus_save.m). |
| **results/fourteen_bus_dynamic/** | ScopeBus1.csv … ScopeBus14.csv, summary.txt, observation_from_voltage.csv (optional), ode_params_single_design.json (optional). |
| **results/COMPARISON_TABLE.md** | Summary table: MATLAB dynamic + probe (ROCOF_max, f_min per bus) and Python ODE (probe at bus 1 per-bus, probe at 1..14 system-wide). Generated by `python -m tests.simulink_reference.ode_validation --table-only`. |