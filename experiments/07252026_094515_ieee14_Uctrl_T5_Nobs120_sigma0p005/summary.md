# Summary — ieee14 (objective_based)

Observation: 120 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee14`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 120
- N_sim: 1600
- T: 5
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee14.yaml`

## Comparison

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.124920 | 0.000000 | 0.000 | 1.000 | VALID | — |
| DAD | 0.300625 | 0.175705 | 0.000 | 1.000 | VALID | 27 |
| RL-sBOED | 0.306250 | 0.181586 | 0.016 | 0.984 | VALID | 29 |
| Myopic | 0.305625 | 0.180705 | 0.000 | 1.000 | VALID | 56 |
| Fixed | 0.310625 | 0.185705 | 0.000 | 1.000 | VALID | 1 |
| Random | 0.354375 | 0.232075 | 0.031 | 0.969 | VALID | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.90 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_u_ctrl.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.90; mean_u_ctrl ↓)

DAD, Myopic, RL-sBOED, Fixed, Random
