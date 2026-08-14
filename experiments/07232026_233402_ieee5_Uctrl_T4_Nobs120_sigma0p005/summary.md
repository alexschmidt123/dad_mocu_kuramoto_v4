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

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.091393 | 0.000000 | 0.000 | 1.000 | — |
| DAD | 0.106875 | 0.024070 | 0.328 | 0.672 | 37 |
| RL-sBOED | 0.097031 | 0.023655 | 0.422 | 0.578 | 24 |
| Myopic | 0.099844 | 0.016266 | 0.391 | 0.609 | 42 |
| Fixed | 0.103125 | 0.040505 | 0.453 | 0.547 | 1 |
| Random | 0.126562 | 0.043946 | 0.281 | 0.719 | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (under_control ↑, mean_excess ↑, mean_u_ctrl ↑)

Random, DAD, Myopic, RL-sBOED, Fixed
