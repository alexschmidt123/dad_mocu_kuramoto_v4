# Summary — ieee9 (objective_based)

Observation: 20 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 20
- N_sim: 1600
- T: 3
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee9.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.082331 | 1.000 | VALID | — |
| DAD | 0.062864 | 0.145195 | 0.984 | VALID | 9 |
| RL-sBOED | 0.063568 | 0.145898 | 0.984 | VALID | 5 |
| Myopic | 0.068138 | 0.150469 | 0.984 | VALID | 40 |
| Fixed | 0.071302 | 0.153633 | 0.891 | INVALID | 1 |
| Random | 0.101888 | 0.184219 | 0.938 | INVALID | 64 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

DAD, RL-sBOED, Myopic
