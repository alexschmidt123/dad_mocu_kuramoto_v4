# Summary — ieee5 (objective_based)

Observation: 120 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee5`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 120
- N_sim: 1600
- T: 3
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee5.yaml`

## Comparison

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.091393 | 0.000000 | 0.000 | 1.000 | — |
| DAD | 0.107812 | 0.022542 | 0.297 | 0.703 | 5 |
| RL-sBOED | 0.104531 | 0.035734 | 0.359 | 0.641 | 5 |
| Myopic | 0.105469 | 0.019819 | 0.266 | 0.734 | 35 |
| Fixed | 0.104062 | 0.022003 | 0.312 | 0.688 | 1 |
| Random | 0.139219 | 0.056576 | 0.250 | 0.750 | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (under_control ↑, mean_excess ↑, mean_u_ctrl ↑)

Random, Myopic, DAD, Fixed, RL-sBOED
