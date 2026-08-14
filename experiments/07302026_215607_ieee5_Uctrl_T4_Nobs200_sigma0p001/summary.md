# Summary — ieee5 (objective_based)

Observation: 200 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee5`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 200
- N_sim: 1600
- T: 4
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee5.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.091393 | 1.000 | VALID | — |
| DAD | 0.070325 | 0.161719 | 1.000 | VALID | 5 |
| RL-sBOED | 0.072669 | 0.164062 | 1.000 | VALID | 8 |
| MoE-sBOED | 0.070325 | 0.161719 | 1.000 | VALID | 5 |
| Myopic | 0.078294 | 0.169687 | 0.984 | VALID | 30 |
| Fixed | 0.077825 | 0.169219 | 0.844 | INVALID | 1 |
| Random | 0.072230 | 0.163623 | 0.916 | INVALID | 2042 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

DAD, MoE-sBOED, RL-sBOED, Myopic
