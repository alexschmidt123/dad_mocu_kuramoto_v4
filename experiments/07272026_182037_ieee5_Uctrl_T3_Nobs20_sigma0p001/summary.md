# Summary — ieee5 (objective_based)

Observation: 20 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee5`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 20
- N_sim: 1600
- T: 3
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee5.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.091393 | 1.000 | VALID | — |
| DAD | 0.079700 | 0.171094 | 0.969 | VALID | 5 |
| RL-sBOED | 0.075013 | 0.166406 | 0.969 | VALID | 3 |
| Myopic | 0.074544 | 0.165938 | 0.938 | INVALID | 38 |
| Fixed | 0.082044 | 0.173437 | 0.828 | INVALID | 1 |
| Random | 0.106888 | 0.198281 | 0.953 | VALID | 64 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

RL-sBOED, DAD, Random
