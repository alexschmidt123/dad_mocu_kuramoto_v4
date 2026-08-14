# Summary — ieee5 (objective_based)

Observation: 20 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee5`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 20
- N_sim: 1600
- T: 4
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee5.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.091393 | 1.000 | VALID | — |
| DAD | 0.078294 | 0.169687 | 0.969 | VALID | 27 |
| RL-sBOED | 0.074075 | 0.165469 | 0.953 | VALID | 12 |
| Myopic | 0.075013 | 0.166406 | 0.922 | INVALID | 44 |
| Fixed | 0.106419 | 0.197813 | 0.844 | INVALID | 1 |
| Random | 0.089075 | 0.180469 | 0.969 | VALID | 64 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

RL-sBOED, DAD, Random
