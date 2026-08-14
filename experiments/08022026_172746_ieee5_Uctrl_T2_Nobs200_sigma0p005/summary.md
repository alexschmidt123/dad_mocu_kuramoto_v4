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
| DAD | 0.081575 | 0.172969 | 0.984 | VALID | 4 |
| RL-sBOED | 0.076419 | 0.167813 | 0.984 | VALID | 2 |
| MoE-sBOED | 0.074075 | 0.165469 | 0.984 | VALID | 2 |
| Myopic | 0.091888 | 0.183281 | 0.953 | VALID | 45 |
| Fixed | 0.078763 | 0.170156 | 0.859 | INVALID | 1 |
| Random | 0.121536 | 0.212930 | 0.966 | VALID | 796 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

MoE-sBOED, RL-sBOED, DAD, Myopic, Random
