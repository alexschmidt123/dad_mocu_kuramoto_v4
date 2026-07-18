# Particle posterior adequacy — ieee9

## Setup (unchanged)
- Latent dim: 18 (`theta=(M_1..M_N,K_1..K_N)`)
- Designs: 54 = 6 amplitudes × 9 buses
- Duration: 0.2 s
- Amplitudes: [0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
- Dataset: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/data/ieee9_particle_adequacy_master_2048`
- Official metric: `u_ctrl = snap_up(Q_{1-alpha}(U|w)+margin)`
- Diagnostic: `u_cont = Q_{1-alpha}(U|w)+margin`
- Particle counts: [128, 256, 512, 1024, 2048]
- Support seeds: [101, 202, 303, 404, 505]
- Diagnostic histories: 120 (plus h0)

## True-θ sample counts (diagnostic only)
- Production train θ: 128
- Production test θ: 32

## ESS by history step (median normalized ESS at N={ref_n})
- h0: 1
- h1: 0.1682
- h2: 0.0085
- h3: 0.0009759

## Δ_adaptive by particle count (mean over seeds)
- N=128: Δ_adaptive=0  case=D  bus=BUS-B
- N=256: Δ_adaptive=0  case=D  bus=BUS-B
- N=512: Δ_adaptive=0  case=D  bus=BUS-B
- N=1024: Δ_adaptive=0  case=D  bus=BUS-B
- N=2048: Δ_adaptive=0  case=D  bus=BUS-B

## Smallest practically adequate N
- **512** (objective_stable_uctrl_regret_delta_bus_case)

## u_ctrl / design stability vs reference
- N=128: median|Δu_ctrl|=0.0  frac_changed=0.367983367983368
- N=256: median|Δu_ctrl|=0.0  frac_changed=0.3808731808731809
- N=512: median|Δu_ctrl|=0.0  frac_changed=0.4964656964656965
- N=1024: median|Δu_ctrl|=0.0  frac_changed=0.30353430353430355
- N=2048: median|Δu_ctrl|=0.0  frac_changed=0.0
- N=128: design_agree=0.2033195020746888  bus_agree=0.6896265560165975  median_regret=0.0025000000000000577
- N=256: design_agree=0.1908713692946058  bus_agree=0.7095435684647303  median_regret=0.0025000000000001688
- N=512: design_agree=0.3203319502074689  bus_agree=0.7618257261410788  median_regret=0.0024999999999999467
- N=1024: design_agree=0.25809128630705397  bus_agree=0.8497925311203319  median_regret=0.0024999999999999467
- N=2048: design_agree=nan  bus_agree=nan  median_regret=nan
