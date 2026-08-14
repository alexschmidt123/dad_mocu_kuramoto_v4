# Summary — ieee9 (objective_based)

Observation: 120 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 120
- N_sim: 1600
- T: 2
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee9.yaml`

## Comparison

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.082331 | 0.000000 | 0.000 | 1.000 | — |
| DAD | 0.104414 | 0.024949 | 0.172 | 0.828 | 2 |
| RL-sBOED | 0.106172 | 0.026477 | 0.156 | 0.844 | 2 |
| Myopic | 0.102305 | 0.021820 | 0.141 | 0.859 | 14 |
| Fixed | 0.101250 | 0.020708 | 0.156 | 0.844 | 1 |
| Random | 0.153281 | 0.071631 | 0.047 | 0.953 | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (under_control ↑, mean_excess ↑, mean_u_ctrl ↑)

Random, Myopic, Fixed, RL-sBOED, DAD
