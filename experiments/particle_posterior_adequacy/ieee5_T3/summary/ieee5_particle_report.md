# Particle posterior adequacy — ieee5

## Setup (unchanged)
- Latent dim: 10 (`theta=(M_1..M_N,K_1..K_N)`)
- Designs: 30 = 6 amplitudes × 5 buses
- Duration: 0.2 s
- Amplitudes: [0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
- Dataset: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/data/ieee5_particle_adequacy_master_2048`
- Official metric: `u_ctrl = snap_up(Q_{1-alpha}(U|w)+margin)`
- Diagnostic: `u_cont = Q_{1-alpha}(U|w)+margin`
- Particle counts: [128, 256, 512, 1024, 2048]
- Support seeds: [101, 202, 303, 404, 505]
- Diagnostic histories: 120 (plus h0)

## True-θ sample counts (diagnostic only)
- Production train θ: 64
- Production test θ: 16

## ESS by history step (median normalized ESS at N={ref_n})
- h0: 1
- h1: 0.1173
- h2: 0.01466
- h3: 0.003055

## Δ_adaptive by particle count (mean over seeds)
- N=128: Δ_adaptive=0  case=D  bus=BUS-B
- N=256: Δ_adaptive=0  case=D  bus=BUS-B
- N=512: Δ_adaptive=0  case=D  bus=BUS-B
- N=1024: Δ_adaptive=0  case=B  bus=BUS-B
- N=2048: Δ_adaptive=0  case=B  bus=BUS-B

## Smallest practically adequate N
- **256** (objective_stable_uctrl_regret_delta_bus_case)

## u_ctrl / design stability vs reference
- N=128: median|Δu_ctrl|=0.0  frac_changed=0.26112266112266114
- N=256: median|Δu_ctrl|=0.0  frac_changed=0.1945945945945946
- N=512: median|Δu_ctrl|=0.0  frac_changed=0.16632016632016633
- N=1024: median|Δu_ctrl|=0.0  frac_changed=0.13762993762993764
- N=2048: median|Δu_ctrl|=0.0  frac_changed=0.0
- N=128: design_agree=0.4970954356846473  bus_agree=0.870539419087137  median_regret=0.0
- N=256: design_agree=0.5526970954356847  bus_agree=0.9112033195020747  median_regret=0.0
- N=512: design_agree=0.4473029045643154  bus_agree=0.9543568464730291  median_regret=0.0
- N=1024: design_agree=0.6514522821576764  bus_agree=0.9717842323651452  median_regret=0.0
- N=2048: design_agree=nan  bus_agree=nan  median_regret=nan
