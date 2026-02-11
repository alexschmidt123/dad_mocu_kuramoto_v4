# MATLAB/Simulink reference: IEEE 14-bus

This folder holds a **third-party Simulink model** of the IEEE 14-bus system for reference only. **All simulation and analysis in this project are done in Python** (see `src/`, `scripts/`, `documents/design.md`). We do not repeat the swing ODE or probing logic in MATLAB.

## What is here

**Fourteen_bus.mdl** – Original IEEE 14-bus Simulink model from MATLAB Central File Exchange (Bharath Yk, [46067](https://www.mathworks.com/matlabcentral/fileexchange/46067-ieee-14-bus-system-simulink-model)). Short transient (StopTime 0.12 s). Use for **steady-state reference** (Step 1).

- **Open:** `open_system('Fourteen_bus.mdl')` or `open_system('Fourteen_bus')`

**Fourteen_bus_dynamic.mdl** – Same model, configured for **dynamic simulation** (Step 2): StopTime 10 s, scope logging on for all 14 buses. When you run the simulation, bus scope data is saved to the MATLAB workspace as **ScopeBus1** … **ScopeBus14** (each with `.time` and `.signals`; use for mapping to θ/ω and comparison with Python).

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
- **run_fourteen_bus_dynamic_save.m** – Runs **Fourteen_bus_dynamic** (10 s). Saves results to `results/fourteen_bus_dynamic/`: CSV (ScopeBus1…14), summary.txt, PNG plots (per bus + all_buses.png).

## Files

- **instruction.md** (this file)
- **Fourteen_bus.mdl** – Original IEEE 14-bus model (File Exchange 46067); steady-state reference.
- **Fourteen_bus_dynamic.mdl** – Same model, 10 s dynamic run with scope logging (ScopeBus1 … ScopeBus14) for validation vs Python.
