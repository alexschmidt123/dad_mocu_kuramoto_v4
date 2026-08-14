# Summary — ieee9 (objective_based)

Observation: 120 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 120
- N_sim: 1600
- T: 3
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee9.yaml`

## Comparison

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.082331 | 0.000000 | 0.000 | 1.000 | — |
| DAD | 0.103008 | 0.024921 | 0.234 | 0.766 | 12 |
| RL-sBOED | 0.097383 | 0.020259 | 0.297 | 0.703 | 7 |
| Myopic | 0.094219 | 0.016009 | 0.250 | 0.750 | 53 |
| Fixed | 0.093164 | 0.015258 | 0.297 | 0.703 | 1 |
| Random | 0.139219 | 0.059568 | 0.156 | 0.844 | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (under_control ↑, mean_excess ↑, mean_u_ctrl ↑)

Random, DAD, Myopic, Fixed, RL-sBOED
