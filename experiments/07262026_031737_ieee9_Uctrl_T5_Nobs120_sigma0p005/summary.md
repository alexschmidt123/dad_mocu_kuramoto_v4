# Summary — ieee9 (objective_based)

Observation: 120 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 120
- N_sim: 1600
- T: 5
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee9.yaml`

## Comparison

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.082331 | 0.000000 | 0.000 | 1.000 | VALID | — |
| DAD | 0.143437 | 0.061546 | 0.047 | 0.953 | VALID | 4 |
| RL-sBOED | 0.133945 | 0.053551 | 0.094 | 0.906 | VALID | 4 |
| Myopic | 0.134297 | 0.052060 | 0.016 | 0.984 | VALID | 61 |
| Fixed | 0.139570 | 0.057247 | 0.016 | 0.984 | VALID | 1 |
| Random | 0.157148 | 0.078530 | 0.062 | 0.938 | VALID | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.90 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_u_ctrl.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.90; mean_u_ctrl ↓)

RL-sBOED, Myopic, Fixed, DAD, Random
