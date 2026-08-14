# Summary — ieee9 (objective_based)

Observation: 120 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 120
- N_sim: 1600
- T: 4
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee9.yaml`

## Comparison

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.082331 | 0.000000 | 0.000 | 1.000 | — |
| DAD | 0.100898 | 0.028945 | 0.312 | 0.688 | 10 |
| RL-sBOED | 0.102305 | 0.028423 | 0.250 | 0.750 | 28 |
| Myopic | 0.089648 | 0.013385 | 0.328 | 0.672 | 56 |
| Fixed | 0.094570 | 0.019318 | 0.344 | 0.656 | 1 |
| Random | 0.125156 | 0.049284 | 0.188 | 0.812 | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (under_control ↑, mean_excess ↑, mean_u_ctrl ↑)

Random, RL-sBOED, DAD, Myopic, Fixed
