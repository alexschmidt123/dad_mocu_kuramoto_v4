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

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.082331 | 0.000000 | 0.000 | 1.000 | VALID | — |
| DAD | 0.146602 | 0.064677 | 0.031 | 0.969 | VALID | 3 |
| RL-sBOED | 0.142383 | 0.060052 | 0.000 | 1.000 | VALID | 2 |
| Myopic | 0.143437 | 0.061107 | 0.000 | 1.000 | VALID | 21 |
| Fixed | 0.145898 | 0.063568 | 0.000 | 1.000 | VALID | 1 |
| Random | 0.186680 | 0.104681 | 0.016 | 0.984 | VALID | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.90 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_u_ctrl.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.90; mean_u_ctrl ↓)

RL-sBOED, Myopic, Fixed, DAD, Random
