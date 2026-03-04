# Parameters You Can Grab from the IEEE 14-Bus .mdl

This note is about **fourteen_bus.mdl** (and **fourteen_bus_dynamic.mdl**). It states exactly which quantities you can take from the .mdl for your IEEE 14 swing-ODE, and which you cannot.

---

## Parameters you **can** get from the .mdl

| What you need for Python ODE | Where in the .mdl | How you get it |
|-----------------------------|-------------------|----------------|
| **Base power (S_base)** | Measurement blocks: `Pbase` | **Grab directly:** 100 MVA (100e6). Use for p.u. power and susceptances. |
| **Base voltage per bus (V_base)** | Measurement blocks: `Vbase` | **Grab directly:** per-bus values ≈ 0.92–1.0 p.u. (e.g. 0.92, 0.94, 0.95, 1.0). |
| **Generator active power setpoints** | Three-Phase Source blocks: `Pref` | **Grab directly:** e.g. 0.4, 2.324 (p.u.). Use at generator buses to build **P_m** (and use loads for negative P_m at load buses). |
| **Generator reactive setpoints** | Three-Phase Source blocks: `Qref` | **Grab directly:** e.g. 0, −0.169. Optional for power-flow / P_m balance. |
| **Line resistance R** | Series RLC Branch blocks: `Resistance` | **Grab directly:** e.g. 0.01335, 0.01938, 0.06615, … (p.u.). Used for lossy models; for lossless swing equation often ignored. |
| **Line reactance X** | Series RLC Branch blocks: `Inductance` | **Derive:** Inductance is `L_value/(2*pi*f)` → **X = L_value** in p.u. (e.g. 0.04211, 0.05917, 0.13027, …). **Grab** the numeric part from each branch. |
| **Coupling B_ij (susceptance)** | From line reactance X | **Derive:** **B_ij = 1 / X_ij** for each line (lossless). You need a **bus–branch mapping** (which branch connects which bus pair); that mapping is **not** in the .mdl block names—use the schematic or MATPOWER case14. |
| **Reference phase angle (θ)** | Three-Phase Source: `PhaseAngle`; or scope outputs after run | **Grab directly:** e.g. **−30°** at sources. After a **steady-state run**, **grab** bus voltage angles from ScopeBus1…ScopeBus14 → use as reference θ (and ω ≈ 0) for your ODE equilibrium or initial conditions. |
| **Reference voltage magnitude** | Three-Phase Source: `Voltage` | **Grab directly:** e.g. 1.045, 1.06, 1.09 (p.u.). Optional for alignment with .mdl outputs. |

So from the .mdl you can **grab**: S_base, V_base per bus, Pref (and Qref), R and X per branch, PhaseAngle (−30°), Voltage, and after a run the steady-state bus angles (and any logged signals). You **derive**: B_ij = 1/X_ij once you have the bus–branch mapping.

---

## Parameters you **cannot** get from the .mdl

| Parameter | Why not in .mdl |
|-----------|------------------|
| **M (inertia)** | The .mdl uses detailed electrical (Simscape) sources, not swing-equation states. M = 2H/ω_s is not a block parameter. |
| **D (damping)** | No swing-equation damping parameter in the .mdl. |
| **K (droop)** | No primary frequency (droop) gain in the .mdl. |
| **Which bus pair each branch connects** | Block names are "Branch1", "Branch2", … with no (from_bus, to_bus). Get the mapping from the .mdl schematic or from MATPOWER case14. |

Use **M, D, K** from literature as in `documents/Parameter_references_table.md` (e.g. M ∈ [0.01, 0.06], D = 0.1, K ∈ [0.05, 0.50]).

---

## Short reference: what to grab where

- **Three-Phase Source blocks:** `Pref`, `Qref`, `Voltage`, `PhaseAngle` (−30°).
- **Series RLC Branch blocks:** `Resistance`, `Inductance` (→ X = numeric part in p.u.).
- **Measurement (V-I) blocks:** `Pbase` (100e6), `Vbase` (per bus).
- **After a simulation:** Scope data (e.g. ScopeBus1…ScopeBus14 in fourteen_bus_dynamic.mdl) → steady-state bus angles and time series for θ (and frequency if you compute it from angle).

Use MATPOWER case14 (or the .mdl schematic) to map Branch1…Branch20 to (from_bus, to_bus), then build **B** from **B_ij = 1/X_ij** and **P_m** from Pref/loads.
