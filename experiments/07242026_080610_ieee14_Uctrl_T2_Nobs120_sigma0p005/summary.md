# Summary — ieee14 (objective_based)

Observation: 120 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee14`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 120
- N_sim: 1600
- T: 2
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee14.yaml`

## Comparison

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.124920 | 0.000000 | 0.000 | 1.000 | — |
| DAD | 0.150000 | 0.040547 | 0.219 | 0.781 | 4 |
| RL-sBOED | 0.157500 | 0.034456 | 0.125 | 0.875 | 2 |
| Myopic | 0.150000 | 0.028565 | 0.188 | 0.812 | 18 |
| Fixed | 0.150000 | 0.027583 | 0.172 | 0.828 | 1 |
| Random | 0.265000 | 0.150260 | 0.125 | 0.875 | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (under_control ↑, mean_excess ↑, mean_u_ctrl ↑)

RL-sBOED, Random, Fixed, Myopic, DAD
