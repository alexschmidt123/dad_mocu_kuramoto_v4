# ieee9 bus + joint adaptive-value report

- Case: **BUS-B** — one-step bus gaps/branching exist, but four-way terminal decomposition does not significantly beat Fully Fixed (practical terminal bus adaptive value still low; also BUS-E for policy training)
- Histories (h1): 200
- Design: 6 amps × 9 buses = 54 (duration=0.2 s)
- Dominant bus: 0 (fraction 0.685)
- Unique optimal buses: 5
- Mean wrong-bus regret (cont): 0.00677996
- Mean wrong-bus regret (snap): 0.00426897
- Prior wrong-amplitude regret: 0.00026875
- Mean Fixed-bus regret: 0.00025

## Four-way decomposition (continuous terminal)

- fully_fixed: mean u_cont=0.919375, mean u_snap=0.97125
- fixed_bus_adaptive_amp: mean u_cont=0.919375, mean u_snap=0.97125
- adaptive_bus_fixed_amp: mean u_cont=0.918125, mean u_snap=0.97
- adaptive_bus_adaptive_amp: mean u_cont=0.918125, mean u_snap=0.97

- fixed_bus_adaptive_amp - fully_fixed: +0 CI95=[+0, +0]
- adaptive_bus_fixed_amp - fully_fixed: -0.00125 CI95=[-0.00375, +0]
- adaptive_bus_adaptive_amp - fully_fixed: -0.00125 CI95=[-0.00375, +0]
- adaptive_bus_adaptive_amp - adaptive_bus_fixed_amp: +0 CI95=[+0, +0]
