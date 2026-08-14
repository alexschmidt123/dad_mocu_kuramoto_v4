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

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.082331 | 0.000000 | 0.000 | 1.000 | VALID | — |
| DAD | 0.163125 | 0.080794 | 0.000 | 1.000 | VALID | 11 |
| RL-sBOED | 0.163125 | 0.080794 | 0.000 | 1.000 | VALID | 10 |
| Myopic | 0.161016 | 0.078685 | 0.000 | 1.000 | VALID | 49 |
| Fixed | 0.157852 | 0.075521 | 0.000 | 1.000 | VALID | 1 |
| Random | 0.210586 | 0.128255 | 0.000 | 1.000 | VALID | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.90 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_u_ctrl.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.90; mean_u_ctrl ↓)

Fixed, Myopic, DAD, RL-sBOED, Random
