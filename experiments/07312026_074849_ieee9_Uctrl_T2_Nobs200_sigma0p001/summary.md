# Summary — ieee9 (objective_based)

Observation: 200 evenly spaced probe-bus Δf samples (`observation_mode=sampled_delta_f`).

- system: `ieee9`
- experiment_type: `objective_based`
- observation_mode: `sampled_delta_f`
- N_obs: 200
- N_sim: 1600
- T: 2
- config: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/configs/ieee9.yaml`

## Comparison

| Method | mean_MOCU | mean_u_ctrl | safety_rate | valid | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.000000 | 0.082331 | 1.000 | VALID | — |
| DAD | 0.052318 | 0.134648 | 0.969 | VALID | 4 |
| RL-sBOED | 0.054427 | 0.136758 | 1.000 | VALID | 1 |
| MoE-sBOED | 0.052318 | 0.134648 | 0.969 | VALID | 4 |
| Myopic | 0.065677 | 0.148008 | 0.984 | VALID | 29 |
| Fixed | 0.067786 | 0.150117 | 0.859 | INVALID | 1 |
| Random | 0.066677 | 0.149008 | 0.887 | INVALID | 1477 |

Notes:

- `mean_MOCU` = mean(u_ctrl − u_ctrl_optimal) on common held-out systems.
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Methods with safety rate below 0.95 are INVALID and receive no rank.
- Valid methods are ranked by lower mean_MOCU.
- `mean_u_ctrl` remains a secondary physical-control metric.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (safety ≥ 0.95; mean_MOCU ↓)

DAD, MoE-sBOED, RL-sBOED, Myopic
