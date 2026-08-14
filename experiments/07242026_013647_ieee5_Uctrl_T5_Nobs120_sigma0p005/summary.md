# Summary — ieee5 (objective_based)

Observation: 120 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee5`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 120
- N_sim: 1600
- T: 5
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee5.yaml`

## Comparison

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.091393 | 0.000000 | 0.000 | 1.000 | — |
| DAD | 0.103125 | 0.020670 | 0.328 | 0.672 | 28 |
| RL-sBOED | 0.104062 | 0.019601 | 0.328 | 0.672 | 37 |
| Myopic | 0.099844 | 0.016869 | 0.391 | 0.609 | 45 |
| Fixed | 0.104531 | 0.022673 | 0.312 | 0.688 | 1 |
| Random | 0.114375 | 0.035992 | 0.344 | 0.656 | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (under_control ↑, mean_excess ↑, mean_u_ctrl ↑)

Fixed, RL-sBOED, DAD, Random, Myopic
