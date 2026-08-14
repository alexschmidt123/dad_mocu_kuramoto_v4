# Summary — IEEE-5 (eig_based)

Observation: scalar max-|ROCOF| (`N_obs=0`). Methods do not see full Δf trajectories.

- system: `IEEE-5`
- experiment_type: `eig_based`
- observation_mode: `max_rocof`
- N_obs: 0
- T: 2

## Comparison

| Method | mean_eig | mean_eig_step1 | sum_stepwise_eig | n_rollouts |
| --- | ---: | ---: | ---: | ---: |
| DAD-EIG | 1.5557 | 1.0735 | 1.5557 | 128 |
| RL-sBOED-EIG | 0.8940 | 0.1016 | 0.8940 | 128 |
| MoE-sBOED | 2.0372 | 1.0735 | 2.0372 | 128 |
| Myopic ΔH | 2.0372 | 1.0735 | 2.0372 | 128 |
| Random | 0.8108 | 0.4314 | 0.8108 | 128 |
| Fixed design | 1.0984 | 0.6131 | 1.0984 | 128 |
