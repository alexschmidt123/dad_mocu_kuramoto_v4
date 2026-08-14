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

| Method | mean_u_ctrl | mean_excess | under_control_rate | safety_rate | n_unique_sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.082331 | 0.000000 | 0.000 | 1.000 | — |
| DAD | 0.093867 | 0.020344 | 0.344 | 0.656 | 37 |
| RL-sBOED | 0.107227 | 0.031802 | 0.281 | 0.719 | 53 |
| Myopic | 0.090352 | 0.014267 | 0.312 | 0.688 | 60 |
| Fixed | 0.093867 | 0.018615 | 0.344 | 0.656 | 1 |
| Random | 0.116367 | 0.042153 | 0.203 | 0.797 | 64 |

Notes:

- `mean_excess` = mean max(u_ctrl − u_ctrl_opt, 0) (overshoot only).
- `u_ctrl < u_ctrl_opt` is under-control (often unsafe); it does not improve ranking.
- Policies use argmax actions only (no `*_stochastic` rows).

## Ranking (under_control ↑, mean_excess ↑, mean_u_ctrl ↑)

Random, RL-sBOED, Myopic, Fixed, DAD
