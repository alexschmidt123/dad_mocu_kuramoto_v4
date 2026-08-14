# Summary — ieee5 (objective_based)

Observation: 20 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee5`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 20
- N_sim: 1600
- T: 5
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee5.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.091393 | 1.000 | VALID | — |
| DAD | 0.073607 | 0.165000 | 0.938 | INVALID | 10 |
| RL-sBOED | 0.074544 | 0.165937 | 0.984 | VALID | 10 |
| Myopic | 0.074544 | 0.165937 | 0.922 | INVALID | 45 |
| Fixed | 0.071263 | 0.162656 | 0.922 | INVALID | 1 |
| Random | 0.079700 | 0.171094 | 0.922 | INVALID | 64 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

RL-sBOED
