# DAD PPO Stage-2 final report

## Scientific contract

Stage 2 uses random initialization, one complete-history and Bayesian-belief policy, bank-only observations, gamma=1 PPO reward equal to negative terminal u_ctrl, validation-only selection, and frozen sealed confirmation. No EIG, imitation, counterfactual critic, adaptivity reward, diversity reward, physical ODE call, or confirmation tuning is used.

## 1–8. Selected training design

1. Current R2 belief representation is documented in `audit/belief_input_audit.md` (33 legal posterior/history summaries).
2. Best belief encoder: **B2**.
3. Best history encoder: **H1**. Final actor/value sizes are 372,558/368,817 parameters on IEEE5 and 377,190/370,353 on IEEE9.
4. Best entropy strategy: **E1**.
5. Best PPO batch size: **P2**.
6. Best normalization: **A0**, GAE lambda **0.98**. A0 and A1 are equivalent because R2 already normalizes advantages.
7. Curriculum selected: **none**; IEEE5 direct 0.852778 vs curriculum 0.875000; IEEE9 direct 0.967778 vs curriculum 0.973333.
8. Branch survival through C2/C3: IEEE5 mean final survival 0.556 (mean KL 1.201); IEEE9 survival 1.000 (mean KL 0.198, but all final probes selected one action). These values were diagnostic and never optimized.

## 9–11. Adaptive-positive diagnostic and stability

- IEEE5: learned mean 0.835156, seed SD 0.001105, mean adaptive advantage recovered 1.002, observation-dependent rate 1.000.
- IEEE9: learned mean 1.072695, seed SD 0.006837, mean adaptive advantage recovered 0.464, observation-dependent rate 0.500.
9. Diagnostic details are in `adaptive_positive_diagnostic/*/diagnostic_results.csv`.
10. Mean recovered advantage is reported across all eight seeds, not a lucky seed.
11. Seed variance: IEEE5 Stage-2 0.024373 vs R2 0.013501; IEEE9 Stage-2 0.005796 vs R2 0.015088.
- IEEE5 checkpoints: median best update 20, median final update 80, mean best/final validation 0.845833/0.881250, mean degradation 0.035417.
- IEEE9 checkpoints: median best update 15, median final update 75, mean best/final validation 0.969583/0.977083, mean degradation 0.007500.

## 12–21. Sealed confirmation and adaptivity

12. IEEE5 Stage-2: mean 0.855300, SD 0.084412, median 0.862500, q10 0.725000, q90 0.962500, safety 1.000, mean excess 0.536250, runtime 0.004585s.
13. IEEE9 Stage-2: mean 0.971220, SD 0.029033, median 0.960000, q10 0.960000, q90 0.990000, safety 1.000, mean excess 0.958500, runtime 0.006778s.
14. Stage-2 versus previous_R2_dad: IEEE5 0.012860, 95% CI [0.009940, 0.015815] (Stage-2 worse); IEEE9 -0.010572, 95% CI [-0.012570, -0.008522] (Stage-2 better).
15. Stage-2 versus previous_reinforce_dad: IEEE5 0.006000, 95% CI [0.001650, 0.010403] (Stage-2 worse); IEEE9 -0.003292, 95% CI [-0.004768, -0.001828] (Stage-2 better).
16. Stage-2 versus fixed: IEEE5 0.012000, 95% CI [0.002362, 0.021938] (Stage-2 worse); IEEE9 -0.007900, 95% CI [-0.011620, -0.004310] (Stage-2 better).
17. Stage-2 versus myopic: IEEE5 -0.002500, 95% CI [-0.008488, 0.003588] (no supported difference); IEEE9 0.006900, 95% CI [0.004690, 0.009170] (Stage-2 worse).
18. Stage-2 versus random: IEEE5 -0.020300, 95% CI [-0.027263, -0.013250] (Stage-2 better); IEEE9 -0.010940, 95% CI [-0.014800, -0.007120] (Stage-2 better).
19. Observation-dependent policy fraction: 0.688.
20. Fraction significantly beating dominant replay: 0.125; A/B/C counts: {'A_nonadaptive_tied_with_replay': 5, 'B_observation_dependent_no_replay_benefit': 9, 'C_observation_dependent_with_benefit': 2}. Seed fractions beating Fixed/Myopic by mean are IEEE5 0.125/0.625 and IEEE9 0.875/0.250.
21. Stable genuine adaptive value across the population: **False**.

## 22–24. Interpretation and transfer

22. Interpretation is system-specific. IEEE9's 8/8 observation-dependent policies but only 2 replay-beating policies favor limited adaptive value. IEEE5's 3/8 observation-dependent policies together with worse cost than R2/Fixed show that optimization/transfer failure remains plausible; it cannot be attributed only to low adaptive value.
23. All required Stage-2 tests passed.
24. New code and outputs are isolated under `src/neural/ppo_stage2.py`, `src/control/ppo_stage2_*.py`, `src/control/dad_ppo_stage2*.py`, `tests/test_dad_ppo_stage2.py`, and `experiments/dad_ppo_stage2/`.

## IEEE14 transfer decision

Ready to transfer unchanged to IEEE14: **False**. Transfer is recommended only if Stage-2 does not degrade R2 on either development system under paired confirmation.
