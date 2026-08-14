# Summary — ieee14 (objective_based)

Observation: 200 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee14`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 200
- N_sim: 1600
- T: 3
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee14.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.124920 | 1.000 | VALID | — |
| DAD | 0.131330 | 0.256250 | 0.953 | VALID | 3 |
| RL-sBOED | 0.146330 | 0.271250 | 1.000 | VALID | 11 |
| MoE-sBOED | 0.131955 | 0.256875 | 1.000 | VALID | 3 |
| Myopic | 0.175080 | 0.300000 | 0.922 | INVALID | 47 |
| Fixed | 0.132580 | 0.257500 | 1.000 | VALID | 1 |
| Random | 0.156291 | 0.281211 | 0.895 | INVALID | 2044 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

DAD, MoE-sBOED, Fixed, RL-sBOED
