# Summary — ieee5 (objective_based)

Observation: 120 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee5`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 120
- N_sim: 1600
- T: 2
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee5.yaml`

## Comparison

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.091393 | 0.000000 | 0.000 | 1.000 | — |
| DAD | 0.111562 | 0.031259 | 0.250 | 0.750 | 3 |
| RL-sBOED | 0.113906 | 0.028156 | 0.203 | 0.797 | 3 |
| Myopic | 0.107344 | 0.021595 | 0.297 | 0.703 | 9 |
| Fixed | 0.107812 | 0.019856 | 0.203 | 0.797 | 1 |
| Random | 0.157031 | 0.071722 | 0.125 | 0.875 | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (under_control ↑, mean_excess ↑, mean_u_ctrl ↑)

Random, Fixed, RL-sBOED, DAD, Myopic
