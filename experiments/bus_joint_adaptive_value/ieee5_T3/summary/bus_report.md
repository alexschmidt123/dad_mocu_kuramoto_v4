# ieee5 bus + joint adaptive-value report

- Case: **BUS-B** — one-step bus gaps/branching exist, but four-way terminal decomposition does not significantly beat Fully Fixed (practical terminal bus adaptive value still low; also BUS-E for policy training)
- Histories (h1): 200
- Design: 6 amps × 5 buses = 30 (duration=0.2 s)
- Dominant bus: 1 (fraction 0.745)
- Unique optimal buses: 4
- Mean wrong-bus regret (cont): 0.0191064
- Mean wrong-bus regret (snap): 0.0160611
- Prior wrong-amplitude regret: 0.000578125
- Mean Fixed-bus regret: 0.0563398

## Four-way decomposition (continuous terminal)

- fully_fixed: mean u_cont=0.848828, mean u_snap=0.85
- fixed_bus_adaptive_amp: mean u_cont=0.845313, mean u_snap=0.846875
- adaptive_bus_fixed_amp: mean u_cont=0.851172, mean u_snap=0.853906
- adaptive_bus_adaptive_amp: mean u_cont=0.852734, mean u_snap=0.853906

- fixed_bus_adaptive_amp - fully_fixed: -0.00351562 CI95=[-0.00859375, +0]
- adaptive_bus_fixed_amp - fully_fixed: +0.00234375 CI95=[-0.0113281, +0.0175781]
- adaptive_bus_adaptive_amp - fully_fixed: +0.00390625 CI95=[-0.0113281, +0.0214844]
- adaptive_bus_adaptive_amp - adaptive_bus_fixed_amp: +0.0015625 CI95=[-0.00273438, +0.0078125]
