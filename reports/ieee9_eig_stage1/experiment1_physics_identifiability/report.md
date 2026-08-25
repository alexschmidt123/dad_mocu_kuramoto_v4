# IEEE9 EIG Stage-1 Experiment 1

## Scope

This is a read-only audit of the existing IEEE9 EIG bank. No DAD, RL-sBOED, Myopic, Fixed, Random, or swing-equation simulation was run. Exact bank statistics are separated from empirical global-linear sensitivity/Fisher diagnostics.

## Bank and observation

- Theta: `512` train and `128` held-out samples of `M_1, M_2, M_3, K_1, K_2, K_3`.
- Actions: `54` = 9 physical buses × 6 durations at fixed amplitude `0.05` p.u.
- PMU: physical bus `1` only.
- Method-visible observation: `N_obs=5`, indices `[0, 640, 1280, 1919, 2559]`, times `[0.001563, 1.001563, 2.001562, 3.0, 4.0]` s.
- Reference EIG noise: `sigma=0.01` Hz.

## Main findings

1. **The first sampled value is practically uninformative.** Dimensions `[0]` satisfy `max across-theta variance / sigma^2 < 1e-8` for every action. Thus `N_obs=5` has at most `4` informative scalar values. CUDA stores after each RK4 step, so index 0 is time `dt`, not exact equilibrium, but its signal is negligible relative to noise.
2. **Action redundancy is high.** `270/1431` action pairs (`18.9%`) have `|correlation| >= 0.98`; mean absolute pair correlation is `0.745`.
3. **The current bank is spatially limited.** Every trajectory is observed at one fixed PMU, so this bank cannot establish spatial observability or certify system-wide probe safety.
4. **Recorded-PMU probe screen:** held-out maximum `|delta_f|=0.061498` Hz and maximum RoCoF `0.103020` Hz/s. The comparison limits are provisional control limits, not a probing standard.
5. **Linear identifiability is uneven.** Combined standardized Fisher eigenvalues are `[19.1582, 4.47193, 0.254005, 0.00472209, 0.00127741, 0.000197723]` with condition number `96894.43617295446`. The weakest direction is `M_1=-0.008, M_2=-0.011, M_3=-0.011, K_1=+0.003, K_2=-0.528, K_3=+0.849`.
6. **Surrogate caution:** held-out global-linear R2 averages `0.922` (minimum `0.523`). Sensitivity and Fisher results are diagnostics, not exact derivatives or EIG estimates.

## Highest parameter-signal actions at sigma=0.01 Hz

| Action | Bus | Duration (s) | Signal SD (Hz) | Effective SNR |
|---:|---:|---:|---:|---:|
| 46 | 2 | 3.0 | 0.00421729 | 0.1779 |
| 47 | 3 | 3.0 | 0.00421702 | 0.1778 |
| 52 | 8 | 3.0 | 0.00421658 | 0.1778 |
| 51 | 7 | 3.0 | 0.00421654 | 0.1778 |
| 50 | 6 | 3.0 | 0.00421651 | 0.1778 |


## Interpretation

The present bank is sufficient for a reproducible single-PMU reference experiment, but it is not yet sufficient to claim that all six regional parameters are spatially identifiable or that the 54 actions provide distinct information. The next Stage-1 experiment should correct the sampling schedule so it does not spend one of five observations on the negligible near-equilibrium response at the first RK4 step, then compare the current durations with separated durations and one PMU with three generator-bus PMUs.

## Files

- `summary.json`: machine-readable conclusions and provenance.
- `tables/action_metrics.csv`: exact action safety/SNR plus held-out surrogate fit.
- `tables/parameter_sensitivity.csv`: empirical standardized sensitivities.
- `tables/action_pair_similarity.csv`: all 1,431 action-pair correlations.
- `tables/combined_standardized_fisher.npy`: combined empirical Fisher matrix.
- `figures/`: sensitivity, similarity, Fisher-spectrum, and SNR plots.
