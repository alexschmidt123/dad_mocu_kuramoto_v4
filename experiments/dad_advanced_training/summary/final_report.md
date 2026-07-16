# Stronger random-initialized DAD training: final report

## Scientific invariants

All policies started from random neural-network weights, used one shared complete-history policy across T=3, used training-bank-only observation normalization and posterior features, and optimized expected terminal u_ctrl. No Fixed/Myopic labels, behavior cloning, EIG, parameter MSE, confirmation tuning, IEEE14 run, physical ODE call, or alternative scientific objective was used.

## 1–10. Training diagnosis and algorithm

1. Previous REINFORCE assigned a quantized terminal signal to a sampled sequence and had very weak action-conditional evidence for observation branches.
2. The old history encoder changed numerically with y, but direct policy L1 changes were small; see `audit/current_training_audit.md`.
3. Belief-state effects are isolated by R1/R2 and R3/R4 validation ablations.
4. PPO performance is reported by R1/R2 below and was not assumed superior.
5. Counterfactual performance is reported by R3/R4.
6. Replay performance is reported by R5 versus R4.
7. Critic validation rows: 173; MAE/RMSE and calibration are in `critic_accuracy.csv`.
8. Critic action ranking is reported by validation Spearman, top-1 agreement, and top-5 recall.
9. Adaptive-positive diagnostic passed: True.
10. Fraction of known adaptive advantage recovered: 0.881.

## Validation-only variant selection

Selected configuration: **R2 (ppo_belief)**, using the lowest summed validation rank across IEEE5 and IEEE9. Confirmation results were unavailable to selection.

### IEEE5 screening

- R0: mean validation u_ctrl 0.827778 (seed SD 0.007857).
- R1: mean validation u_ctrl 0.827778 (seed SD 0.027499).
- R2: mean validation u_ctrl 0.816667 (seed SD 0.018002).
- R3: mean validation u_ctrl 0.858333 (seed SD 0.000000).
- R4: mean validation u_ctrl 0.855556 (seed SD 0.020787).
- R5: mean validation u_ctrl 0.855556 (seed SD 0.010393).

### IEEE9 screening

- R0: mean validation u_ctrl 0.967778 (seed SD 0.003143).
- R1: mean validation u_ctrl 0.970000 (seed SD 0.005443).
- R2: mean validation u_ctrl 0.968889 (seed SD 0.001571).
- R3: mean validation u_ctrl 0.973333 (seed SD 0.002722).
- R4: mean validation u_ctrl 0.976667 (seed SD 0.002722).
- R5: mean validation u_ctrl 0.974444 (seed SD 0.004157).

## 11–20. Sealed confirmation and adaptivity

11. **IEEE5 five-seed aggregate:** mean 0.842440, SD 0.092821, median 0.840000, q10 0.720000, q90 0.980000, safety 1.000.
12. **IEEE9 five-seed aggregate:** mean 0.981792, SD 0.029034, median 0.960000, q10 0.960000, q90 1.008000, safety 1.000.
13. Seed variability (SD of seed means): IEEE5 0.013501; IEEE9 0.015088.
14. Advanced DAD versus previous_improved_dad: IEEE5 -0.006860, 95% CI [-0.012260, -0.001480] (advanced DAD is better); IEEE9 0.007280, 95% CI [0.005520, 0.009040] (advanced DAD is worse).
15. Advanced DAD versus fixed: IEEE5 -0.000860, 95% CI [-0.010060, 0.008680] (no supported difference); IEEE9 0.002672, 95% CI [-0.000880, 0.006128] (no supported difference).
16. Advanced DAD versus myopic: IEEE5 -0.015360, 95% CI [-0.022400, -0.008260] (advanced DAD is better); IEEE9 0.017472, 95% CI [0.015072, 0.019808] (advanced DAD is worse).
17. Advanced DAD versus random: IEEE5 -0.033160, 95% CI [-0.040520, -0.025740] (advanced DAD is better); IEEE9 -0.000368, 95% CI [-0.004240, 0.003344] (no supported difference).
18. Observation dependence for every seed is listed in `adaptivity_summary.csv`; classifications: {'effectively nonadaptive': 6, 'observation-dependent adaptive': 4}.
19. DAD versus dominant replay: 2/10 seed policies have a negative paired CI excluding zero.
20. Genuine adaptive performance gain supported: **False**. This requires both observation dependence and superiority to dominant replay.

## 21–23. Interpretation, tests, and files

21. Because the trainer passed the positive diagnostic, continued nonadaptivity on IEEE5/IEEE9 would favor low adaptive value under the frozen objective over a general inability of the optimizer to branch. A performance gain is called adaptive only when the confirmation and dominant-replay criteria above are met.
22. All 25 required advanced-training invariant tests passed.
23. New source files: `src/neural/advanced_dad.py`, `src/control/advanced_dad_core.py`, `src/control/dad_advanced_audit.py`, `src/control/adaptive_positive_diagnostic.py`, `src/control/dad_advanced_training.py`, `src/control/dad_advanced_report.py`, and `tests/test_dad_advanced_training.py`. Outputs are isolated under `experiments/dad_advanced_training/`.
