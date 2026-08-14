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

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.124920 | 0.000000 | 0.000 | 1.000 | — |
| DAD | 0.156875 | 0.043589 | 0.234 | 0.766 | 15 |
| RL-sBOED | 0.138750 | 0.031029 | 0.312 | 0.688 | 22 |
| Myopic | 0.136250 | 0.017373 | 0.312 | 0.688 | 49 |
| Fixed | 0.145000 | 0.027043 | 0.297 | 0.703 | 1 |
| Random | 0.230625 | 0.119505 | 0.188 | 0.812 | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (under_control ↑, mean_excess ↑, mean_u_ctrl ↑)

Random, DAD, Fixed, Myopic, RL-sBOED
