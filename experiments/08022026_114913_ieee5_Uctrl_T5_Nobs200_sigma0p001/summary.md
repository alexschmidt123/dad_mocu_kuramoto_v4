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
| DAD | 0.063294 | 0.154687 | 1.000 | VALID | 4 |
| RL-sBOED | 0.072200 | 0.163594 | 1.000 | VALID | 17 |
| MoE-sBOED | 0.072669 | 0.164062 | 1.000 | VALID | 21 |
| Myopic | 0.067044 | 0.158437 | 0.984 | VALID | 30 |
| Fixed | 0.064700 | 0.156094 | 0.922 | INVALID | 1 |
| Random | 0.068919 | 0.160312 | 0.910 | INVALID | 2048 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

DAD, Myopic, RL-sBOED, MoE-sBOED
