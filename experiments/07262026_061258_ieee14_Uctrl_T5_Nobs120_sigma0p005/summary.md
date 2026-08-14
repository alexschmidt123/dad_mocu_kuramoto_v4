# Summary — ieee14 (objective_based)

Observation: 120 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee14`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 120
- N_sim: 1600
- T: 5
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee14.yaml`

## Comparison

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.124920 | 0.000000 | 0.000 | 1.000 | VALID | — |
| DAD | 0.266875 | 0.141955 | 0.000 | 1.000 | VALID | 13 |
| RL-sBOED | 0.270000 | 0.145080 | 0.000 | 1.000 | VALID | 19 |
| Myopic | 0.258125 | 0.133205 | 0.000 | 1.000 | VALID | 56 |
| Fixed | 0.265625 | 0.140705 | 0.000 | 1.000 | VALID | 1 |
| Random | 0.303750 | 0.182956 | 0.047 | 0.953 | VALID | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.90 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_u_ctrl.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.90; mean_u_ctrl ↓)

Myopic, Fixed, DAD, RL-sBOED, Random
