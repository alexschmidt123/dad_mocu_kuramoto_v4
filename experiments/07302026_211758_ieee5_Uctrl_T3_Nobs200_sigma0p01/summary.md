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
| Oracle | 0.000000 | 0.091393 | 1.000 | VALID | — |
| DAD | 0.086263 | 0.177656 | 1.000 | VALID | 5 |
| RL-sBOED | 0.089075 | 0.180469 | 1.000 | VALID | 4 |
| MoE-sBOED | 0.078763 | 0.170156 | 0.984 | VALID | 16 |
| Myopic | 0.090950 | 0.182344 | 0.984 | VALID | 57 |
| Fixed | 0.087200 | 0.178594 | 1.000 | VALID | 1 |
| Random | 0.133768 | 0.225161 | 0.984 | VALID | 1964 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

MoE-sBOED, DAD, Fixed, RL-sBOED, Myopic, Random
