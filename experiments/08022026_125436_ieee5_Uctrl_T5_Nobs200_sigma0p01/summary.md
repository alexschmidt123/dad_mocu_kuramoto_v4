# Summary — ieee5 (objective_based)

Observation: 200 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee5`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 200
- N_sim: 1600
- T: 5
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee5.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.091393 | 1.000 | VALID | — |
| DAD | 0.081107 | 0.172500 | 1.000 | VALID | 26 |
| RL-sBOED | 0.081107 | 0.172500 | 1.000 | VALID | 45 |
| MoE-sBOED | 0.068450 | 0.159844 | 0.969 | VALID | 14 |
| Myopic | 0.082044 | 0.173437 | 0.984 | VALID | 63 |
| Fixed | 0.075013 | 0.166406 | 0.969 | VALID | 1 |
| Random | 0.106112 | 0.197505 | 0.963 | VALID | 2048 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

MoE-sBOED, Fixed, DAD, RL-sBOED, Myopic, Random
