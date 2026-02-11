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
   pytest tests/test_experiment_design_pipeline.py -v
   ```
   → Fills `tests/output/design_comparison_table.csv` (140 designs, T=5 s, 50 Hz nominal, same as `tests/conftest.py`).

3. **Compare (from repo root):**
   ```bash
   python matlab/compare_matlab_python_results.py
   ```
   → Prints summary of MATLAB and Python results and comparison notes.

**Alignment:** Python test uses `T=5.0` s and 50 Hz nominal (`tests/conftest.py`). The dynamic .mdl uses **StopTime 5 s** and the .mdl is 50 Hz; both produce results over a 5 s window for comparison.

## Difference between Fourteen_bus and Fourteen_bus_dynamic

| | **Fourteen_bus** (original) | **Fourteen_bus_dynamic** |
|---|----------------------------|---------------------------|
| **File** | `Fourteen_bus.mdl` | `Fourteen_bus_dynamic.mdl` (copy with different settings) |
| **StopTime** | **0.12 s** | **5 s** (aligned with Python test T=5) |
| **Purpose** | Short transient / **steady-state reference** (Step 1): quick run to see equilibrium or initial transient. | **Dynamic run** (Step 2): longer window for comparison with Python; scope data over 0–5 s. |
| **Scope logging** | Off (or default); saves tout only in the run script. | **On** for all 14 buses → ScopeBus1 … ScopeBus14 in workspace and CSV in `results/fourteen_bus_dynamic/`. |
| **Run script** | `run_fourteen_bus_save.m` → `results/fourteen_bus/` | `run_fourteen_bus_dynamic_save.m` → `results/fourteen_bus_dynamic/` |

Same underlying model (same physics, 50 Hz, IEEE 14-bus); only **simulation length** and **scope logging** differ.

## What is here

**Fourteen_bus.mdl** – Original IEEE 14-bus Simulink model from MATLAB Central File Exchange (Bharath Yk, [46067](https://www.mathworks.com/matlabcentral/fileexchange/46067-ieee-14-bus-system-simulink-model)). Short transient (StopTime 0.12 s). Use for **steady-state reference** (Step 1).

- **Open:** `open_system('Fourteen_bus.mdl')` or `open_system('Fourteen_bus')`

**Fourteen_bus_dynamic.mdl** – Same model, configured for **dynamic simulation** (Step 2): **StopTime 5 s** (aligned with Python test `T=5.0`), scope logging on for all 14 buses. When you run the simulation, bus scope data is saved to the MATLAB workspace as **ScopeBus1** … **ScopeBus14** (each with `.time` and `.signals`; use for mapping to θ/ω and comparison with Python).

- **Open:** `open_system('Fourteen_bus_dynamic')`
- **Run and save results (CSV/TXT/PNG):** `run('run_fourteen_bus_save.m')` for steady-state; `run('run_fourteen_bus_dynamic_save.m')` for dynamic. Outputs go to `results/fourteen_bus/` and `results/fourteen_bus_dynamic/` respectively.

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

**Adding a probe to the dynamic .mdl:** The current .mdl has no probe. To do a probe operation in Simulink you must **edit the model**: add a controlled source or current injection at a chosen bus, driven by a Hann-window signal (e.g. MATLAB Function block: `0.5*A*(1-cos(2*pi*t/2))` for t≤2, 0 else). See **documents/brief_summary_validation.md** (section 1.2, "Can you do a probe operation on the dynamic .mdl?") for steps. The repo does not ship a probed .mdl; you can save a copy (e.g. `Fourteen_bus_dynamic_probe.mdl`) and add the blocks.

## Relation to the project

| Item        | Python (this repo)                    | This folder (MATLAB)     |
|------------|----------------------------------------|---------------------------|
| Simulation | `src/core/`, swing ODE + probe, ROCOF  | Not duplicated here       |
| IEEE 14    | `swing_equation_params.py`              | Fourteen_bus.mdl (reference)      |

Parameter conventions (M, K, D, f_nominal, ROCOF limits, etc.) are in `documents/Parameter_references_table.md` and `documents/design.md`. For **inspiration from the .mdl** (base values, P_m from Pref, B from line reactances, reference θ), see **`matlab/ieee14_mdl_to_python_params.md`**.

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

- **Suppressed in this repo:** `Fourteen_bus_dynamic.mdl` has **Unconnected block output ports** set to `none`, and the run scripts set `UnconnectedOutputMsg` to `'none'` before `sim()`, so these warnings no longer appear when you run via the script or open this .mdl.
- **To change it in the GUI:** Model Configuration Parameters → **Diagnostics** → **Connectivity** → **Unconnected block output ports** → set to `none` (or `warning` if you want to see them again).
- **If you still see the 15 warnings when you click Run:** Close the model (File → Close), then open it again from disk (`open_system('Fourteen_bus_dynamic')`). The saved .mdl already has the diagnostic set to `none`; reopening loads that. Or run once in the Command Window: `set_param('Fourteen_bus_dynamic','UnconnectedOutputMsg','none')` (with the model loaded), then click Run.

## Run scripts (2)

- **run_fourteen_bus_save.m** – Runs **Fourteen_bus** (0.12 s steady-state). Saves results to `results/fourteen_bus/`: CSV (tout, xout, yout), summary.txt, PNG plots.
- **run_fourteen_bus_dynamic_save.m** – Runs **Fourteen_bus_dynamic** (5 s, aligned with Python test T). Saves results to `results/fourteen_bus_dynamic/`: CSV (ScopeBus1…14), summary.txt, PNG plots (per bus + all_buses.png).

**Analysis vs Python:** After running both MATLAB scripts and the Python design tests (`pytest tests/test_experiment_design_pipeline.py`), run `python matlab/compare_matlab_python_results.py` for a summary. See **`matlab/results/ANALYSIS_MATLAB_vs_PYTHON.md`** for a full analysis and comparison of MATLAB vs Python results.

## Files

- **instruction.md** (this file)
- **Fourteen_bus.mdl** – Original IEEE 14-bus model (File Exchange 46067); steady-state reference.
- **Fourteen_bus_dynamic.mdl** – Same model, 5 s dynamic run (aligned with Python test T=5) with scope logging (ScopeBus1 … ScopeBus14) for validation vs Python.
