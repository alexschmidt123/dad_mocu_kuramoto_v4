# Summary — ieee5 (objective_based)

Observation: 200 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee5`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 200
- N_sim: 1600
- T: 3
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee5.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.083474 | 1.000 | VALID | — |
| DAD | 0.141526 | 0.225000 | 1.000 | VALID | 4 |
| RL-sBOED | 0.179026 | 0.262500 | 1.000 | VALID | 4 |
| MoE-sBOED | 0.134026 | 0.217500 | 1.000 | VALID | 4 |
| Myopic | 0.111526 | 0.195000 | 1.000 | VALID | 4 |
| Fixed | 0.164026 | 0.247500 | 1.000 | VALID | 1 |
| Random | 0.150901 | 0.234375 | 1.000 | VALID | 32 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

Myopic, MoE-sBOED, DAD, Random, Fixed, RL-sBOED
