# Summary — ieee5 (objective_based)

Observation: 120 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee5`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 120
- N_sim: 1600
- T: 3
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee5.yaml`

## Comparison

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.091393 | 0.000000 | 0.000 | 1.000 | VALID | — |
| DAD | 0.225000 | 0.133607 | 0.000 | 1.000 | VALID | 7 |
| RL-sBOED | 0.223594 | 0.132612 | 0.016 | 0.984 | VALID | 19 |
| Myopic | 0.227344 | 0.135950 | 0.000 | 1.000 | VALID | 36 |
| Fixed | 0.240469 | 0.151404 | 0.094 | 0.906 | VALID | 1 |
| Random | 0.256406 | 0.165013 | 0.000 | 1.000 | VALID | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.90 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_u_ctrl.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.90; mean_u_ctrl ↓)

RL-sBOED, DAD, Myopic, Fixed, Random
