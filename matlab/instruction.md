# MATLAB/Simulink reference: IEEE 14-bus

This folder holds a **third-party Simulink model** of the IEEE 14-bus system for reference only. **All simulation and analysis in this project are done in Python** (see `src/`, `scripts/`, `documents/design.md`). We do not repeat the swing ODE or probing logic in MATLAB.

## What is here

**Fourteen_bus.mdl/** – Original IEEE 14-bus Simulink model from MATLAB Central File Exchange (Bharath Yk, [46067](https://www.mathworks.com/matlabcentral/fileexchange/46067-ieee-14-bus-system-simulink-model)). Short transient (StopTime 0.12 s). Use for **steady-state reference** (Step 1).

- **Open:** `open_system('Fourteen_bus.mdl/Fourteen_bus.mdl')`

**Fourteen_bus_dynamic.mdl** – Same model, configured for **dynamic simulation** (Step 2): StopTime 10 s, scope logging on for all 14 buses. When you run the simulation, bus scope data is saved to the MATLAB workspace as **ScopeBus1** … **ScopeBus14** (each with `.time` and `.signals`; use for mapping to θ/ω and comparison with Python).

- **Open:** `open_system('Fourteen_bus_dynamic')`
- **Run:** Click Run in Simulink, or `sim('Fourteen_bus_dynamic')`. Then use ScopeBus1 … ScopeBus14 in the workspace for export or comparison.

Both use **Power System Blocks** (Simscape Electrical). They are a detailed electrical network, not the same as the repo’s **reduced swing equation** (θ, ω, B, Hann probe) in Python. Use them as reference topology and for mapping/validation.

## Relation to the project

| Item        | Python (this repo)                    | This folder (MATLAB)     |
|------------|----------------------------------------|---------------------------|
| Simulation | `src/core/`, swing ODE + probe, ROCOF  | Not duplicated here       |
| IEEE 14    | `swing_equation_params.py`              | Fourteen_bus.mdl (reference only) |

Parameter conventions (M, K, D, f_nominal, ROCOF limits, etc.) are in `documents/Parameter_references_table.md` and `documents/design.md`. For **inspiration from the .mdl** (base values, P_m from Pref, B from line reactances, reference θ), see **`matlab/ieee14_mdl_to_python_params.md`**.

## Files

- **instruction.md** (this file)
- **Fourteen_bus.mdl/** – Original IEEE 14-bus model (File Exchange 46067); steady-state reference.
- **Fourteen_bus_dynamic.mdl** – Same model, 10 s dynamic run with scope logging (ScopeBus1 … ScopeBus14) for validation vs Python.
