# Summary — ieee5 (objective_based)

Observation: 120 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee5`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 120
- N_sim: 1600
- T: 4
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee5.yaml`

## Comparison

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.091393 | 0.000000 | 0.000 | 1.000 | VALID | — |
| DAD | 0.225469 | 0.134075 | 0.000 | 1.000 | VALID | 2 |
| RL-sBOED | 0.231094 | 0.139700 | 0.000 | 1.000 | VALID | 20 |
| Myopic | 0.221719 | 0.130325 | 0.000 | 1.000 | VALID | 52 |
| Fixed | 0.225469 | 0.134075 | 0.000 | 1.000 | VALID | 1 |
| Random | 0.247031 | 0.155638 | 0.000 | 1.000 | VALID | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.90 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_u_ctrl.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.90; mean_u_ctrl ↓)

Myopic, DAD, Fixed, RL-sBOED, Random
