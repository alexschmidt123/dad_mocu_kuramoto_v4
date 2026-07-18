# Continuous u_ctrl + amplitude adaptive-value — final report

## Validity of continuous u_ctrl

- U_n nature: `B_discrete_grid_selected`
- Continuous u_ctrl status: **approximation_based_on_discrete_U_bank**
- Physically validated continuous intermediates: **False**

U_n itself is a discrete-grid safe injection level. Continuous u_ctrl = Q_{1-α}(U|w) + margin interpolates between those banked discrete levels in posterior-quantile space. Intermediate continuous values are NOT individually re-simulated against ROCOF/nadir for this study. Safety thresholds (ROCOF, nadir) and calibrated margin are unchanged; only snap_up is removed from the terminal selector.

Primary definition for this study:

    u_ctrl = Q_{1-α}(U|w) + margin   (no snap_up)

Historical snapped control retained only as `u_ctrl_snapped` diagnostic.
Safety constraints (ROCOF, nadir) and calibrated margin are unchanged.

## Design space (from repository configs)

Amplitudes remain the existing six options; duration is the configured
probe duration (0.2 s), not expanded. Buses = all system buses.

### ieee5

- N_design = 30 (6 × 5)
- Case **B**: nominal_amplitude_branching_near_zero_regret; continuous still low practical value => also Case E (snap not the main amplitude bottleneck); continuous u_ctrl more variable than snapped, but amplitude regret still near-zero (not Case D)
- Dominant A* fraction: 0.690
- Unique A* (continuous / snapped): 6 / 6
- Mean wrong-amplitude regret: 0.000578125
- Continuous vs snapped J std: 0.0883626 / 0.0639643

### ieee9

- N_design = 54 (6 × 9)
- Case **B**: nominal_amplitude_branching_near_zero_regret; continuous still low practical value => also Case E (snap not the main amplitude bottleneck); continuous u_ctrl more variable than snapped, but amplitude regret still near-zero (not Case D)
- Dominant A* fraction: 0.885
- Unique A* (continuous / snapped): 6 / 6
- Mean wrong-amplitude regret: 0.00026875
- Continuous vs snapped J std: 0.0156381 / 0.0109093

## Answers to Part XVII

1. **After removing snap_up, how much more variable is terminal u_ctrl?**  
   IEEE5: continuous std=0.0883626, snapped std=0.0639643, unique J continuous/snapped=137/88. IEEE9: continuous std=0.0156381, snapped std=0.0109093, unique J continuous/snapped=34/28.

2. **Does continuous u_ctrl produce larger objective gaps among designs?**  
   IEEE5 mean best−second continuous=0.000578125 vs snapped=0.000523438. IEEE9 continuous=0.00026875 vs snapped=0.00025625.

3. **Do different histories select different optimal amplitudes?**  
   IEEE5: unique A*=6, non-dominant fraction=0.310.
   IEEE9: unique A*=6, non-dominant fraction=0.115.

4. **Systematic or mostly random/tied?**  
   Interpret via case labels and near-zero regrets: if Case A/B, changes are nominal/tied rather than practically meaningful.

5. **Does preferred amplitude change with bus held fixed?**  
   See `results/bus_conditional_amplitude.csv` (per-bus A*(h,b) diversity).

6. **Regret of wrong amplitude?**  
   IEEE5: mean=0.000578125, p95=0.00234375, max=0.0046875.
   IEEE9: mean=0.00026875, p95=0.00125, max=0.01125.

7. **Does the globally dominant amplitude perform nearly as well?**  
   IEEE5: mean dominant-amp regret=0.000984375.
   IEEE9: mean dominant-amp regret=0.00059375.

8. **Does Fixed plan amplitude perform nearly as well?**  
   IEEE5: Fixed amp=0.15, mean regret=0.00183984.
   IEEE9: Fixed amp=0.05, mean regret=0.00059375.

9. **Does continuous u_ctrl reveal adaptive amplitude value hidden by snap_up?**  
   Compare unique A* and gaps continuous vs snapped in the system tables above; Case D only if continuous shows meaningful regret while snapped does not.

10. **IEEE5 amplitude adaptivity:**  
   **nominal only** (Case B).

11. **IEEE9 amplitude adaptivity:**  
   **nominal only** (Case B).

12. **Is a finer 0.01 amplitude grid justified next?**  
   **Not yet.** Preference barely changes and/or wrong-amplitude regret is approximately zero under the existing six amplitudes. Increasing amplitude resolution alone may not materially raise intrinsic adaptive value. Do **not** generate the 0.01 grid based on this study.

## Decision rule outcome

This study is diagnostic only. No DAD/RL-sBOED retraining and no amplitude grid expansion were performed.

