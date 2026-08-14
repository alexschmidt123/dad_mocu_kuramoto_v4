# Summary — ieee9 (objective_based)

Observation: 200 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 200
- N_sim: 1600
- T: 5
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee9.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.082331 | 1.000 | VALID | — |
| DAD | 0.061107 | 0.143437 | 0.953 | VALID | 44 |
| RL-sBOED | 0.062513 | 0.144844 | 0.953 | VALID | 44 |
| MoE-sBOED | 0.057591 | 0.139922 | 0.984 | VALID | 30 |
| Myopic | 0.067435 | 0.149766 | 0.969 | VALID | 63 |
| Fixed | 0.060755 | 0.143086 | 0.938 | INVALID | 1 |
| Random | 0.093187 | 0.175518 | 0.965 | VALID | 2048 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

MoE-sBOED, DAD, RL-sBOED, Myopic, Random
