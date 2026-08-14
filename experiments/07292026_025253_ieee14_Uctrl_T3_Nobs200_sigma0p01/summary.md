# Summary — ieee14 (objective_based)

Observation: 200 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee14`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 200
- N_sim: 1600
- T: 3
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee14.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.124920 | 1.000 | VALID | — |
| DAD | 0.153205 | 0.278125 | 1.000 | VALID | 7 |
| RL-sBOED | 0.155705 | 0.280625 | 1.000 | VALID | 6 |
| Myopic | 0.145705 | 0.270625 | 1.000 | VALID | 61 |
| Fixed | 0.148205 | 0.273125 | 1.000 | VALID | 1 |
| Random | 0.258830 | 0.383750 | 0.984 | VALID | 64 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

Myopic, Fixed, DAD, RL-sBOED, Random
