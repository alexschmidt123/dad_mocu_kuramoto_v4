# Summary — ieee14 (objective_based)

Observation: 120 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee14`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 120
- N_sim: 1600
- T: 3
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee14.yaml`

## Comparison

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.124920 | 0.000000 | 0.000 | 1.000 | VALID | — |
| DAD | 0.268125 | 0.143205 | 0.000 | 1.000 | VALID | 3 |
| RL-sBOED | 0.258750 | 0.133830 | 0.000 | 1.000 | VALID | 4 |
| Myopic | 0.267500 | 0.142580 | 0.000 | 1.000 | VALID | 51 |
| Fixed | 0.265000 | 0.140080 | 0.000 | 1.000 | VALID | 1 |
| Random | 0.381875 | 0.259281 | 0.016 | 0.984 | VALID | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.90 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_u_ctrl.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.90; mean_u_ctrl ↓)

RL-sBOED, Fixed, Myopic, DAD, Random
