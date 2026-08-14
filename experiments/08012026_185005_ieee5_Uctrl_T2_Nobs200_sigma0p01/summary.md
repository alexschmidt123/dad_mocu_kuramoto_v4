# Summary — ieee5 (objective_based)

Observation: 200 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee5`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 200
- N_sim: 1600
- T: 2
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee5.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.091393 | 1.000 | VALID | — |
| DAD | 0.084857 | 0.176250 | 0.984 | VALID | 2 |
| RL-sBOED | 0.093763 | 0.185156 | 1.000 | VALID | 1 |
| MoE-sBOED | 0.093763 | 0.185156 | 1.000 | VALID | 1 |
| Myopic | 0.103138 | 0.194531 | 1.000 | VALID | 36 |
| Fixed | 0.096107 | 0.187500 | 1.000 | VALID | 1 |
| Random | 0.150379 | 0.241772 | 0.990 | VALID | 796 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

DAD, RL-sBOED, MoE-sBOED, Fixed, Myopic, Random
