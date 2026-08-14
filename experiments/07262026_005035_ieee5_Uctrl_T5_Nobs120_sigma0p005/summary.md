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

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.091393 | 0.000000 | 0.000 | 1.000 | VALID | — |
| DAD | 0.167812 | 0.076419 | 0.000 | 1.000 | VALID | 16 |
| RL-sBOED | 0.166406 | 0.075817 | 0.047 | 0.953 | VALID | 17 |
| Myopic | 0.157500 | 0.066231 | 0.031 | 0.969 | VALID | 52 |
| Fixed | 0.162188 | 0.070862 | 0.016 | 0.984 | VALID | 1 |
| Random | 0.169219 | 0.080798 | 0.078 | 0.922 | VALID | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.90 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_u_ctrl.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.90; mean_u_ctrl ↓)

Myopic, Fixed, RL-sBOED, DAD, Random
