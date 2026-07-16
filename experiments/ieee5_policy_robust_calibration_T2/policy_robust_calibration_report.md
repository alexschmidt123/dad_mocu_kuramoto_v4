# Policy-robust safety calibration (IEEE5 T=2)

## Verdict

Selected common rule: `α=0.05`, **margin=0.55**, snap-up on the frozen u-grid.
All four methods achieve development-test safety **1.0** after DAD retrain with safety-first checkpoints.
IEEE5 T=3/T=4 may resume under this frozen rule.

## 1. Random path consistency

- Shared implementation identical under matched seeds: `True`
- Cause of Random 0.986: Keyed-noise + support-particle Random produced ESS-collapse under-control on a few development-test systems (u_ctrl < u_req); observability used a different observation/particle path and therefore disagreed.
- Observability previously used banked `lookup_action_y` + full-train particles; pilot used keyed noise + support particles. Unified on `run_keyed_history`.

## 2. Metadata consistency

- Consistent: `True`
- Unsafe⇒under-control implication holds: `True`
- terminal_rule_hash (legacy 0.40): `dc0dc35332b394b7` → new hash under margin 0.55 (see selected rule).

## 3. Causes of under-control at margin 0.40

- **DAD 0.936**: ESS≈1 posterior collapse on DAD histories → `Q_0.95` too low → `u_ctrl=snap(Q+0.40)` below `u_req` (e.g. 0.45 < 0.5).
- **Random 0.986**: same under-control mechanism on a few systems; observability disagreed because of a different observation/particle path.

## 4. Residual distributions (margin 0.40 proxy)

- **DAD**: n=12000, unsafe@0.40=0.0025, resid_q95=0.250, resid_max=0.500, ESS=1.365, max_w=0.862
- **DAD_exploratory**: n=4000, unsafe@0.40=0.0147, resid_q95=0.250, resid_max=0.500, ESS=1.952, max_w=0.763
- **DAD_seed_101**: n=4000, unsafe@0.40=0.0075, resid_q95=0.350, resid_max=0.500, ESS=1.428, max_w=0.838
- **DAD_seed_202**: n=4000, unsafe@0.40=0.0000, resid_q95=0.050, resid_max=0.250, ESS=1.431, max_w=0.836
- **DAD_seed_303**: n=4000, unsafe@0.40=0.0000, resid_q95=0.150, resid_max=0.250, ESS=1.238, max_w=0.910
- **Fixed**: n=4000, unsafe@0.40=0.0000, resid_q95=0.050, resid_max=0.250, ESS=1.531, max_w=0.803
- **Myopic**: n=4000, unsafe@0.40=0.0000, resid_q95=0.050, resid_max=0.200, ESS=1.336, max_w=0.875
- **Random**: n=4000, unsafe@0.40=0.0143, resid_q95=0.250, resid_max=0.500, ESS=1.936, max_w=0.766

DAD (esp. seed 101) shows lower ESS, higher max weight, and larger residuals than Myopic/Fixed — consistent with exploiting an overconfident terminal rule.

## 5. Common-margin candidates

Admissible only if **every** policy has cal and val proxy safety = 1.0.

- margin=0.4: admissible=False DAD_val=0.99375 Random_val=0.991 mean_u_val=0.7557642857142857
- margin=0.45: admissible=True DAD_val=1.0 Random_val=1.0 mean_u_val=0.7896642857142858
- margin=0.5: admissible=True DAD_val=1.0 Random_val=1.0 mean_u_val=0.8501928571428572
- margin=0.55: admissible=True DAD_val=1.0 Random_val=1.0 mean_u_val=0.8803714285714286
- margin=0.6: admissible=True DAD_val=1.0 Random_val=1.0 mean_u_val=0.9214857142857142
- margin=0.65: admissible=True DAD_val=1.0 Random_val=1.0 mean_u_val=0.945557142857143
- margin=0.7: admissible=True DAD_val=1.0 Random_val=1.0 mean_u_val=0.9710857142857143
- margin=0.75: admissible=True DAD_val=1.0 Random_val=1.0 mean_u_val=0.9751285714285713
- margin=0.8: admissible=True DAD_val=1.0 Random_val=1.0 mean_u_val=0.9893571428571428
- margin=0.9: admissible=True DAD_val=1.0 Random_val=1.0 mean_u_val=1.0
- margin=1.0: admissible=True DAD_val=1.0 Random_val=1.0 mean_u_val=1.0

## 6. Selected margin

- Cal/val lexicographic pick: **0.45** (smallest mean `u_ctrl` among fully safe candidates).
- Development-test debug: Random safety 0.998 at 0.45 (same `u=0.5 < u_req=0.6` pattern).
- One documented bump: **0.45 → 0.55**. Observability gate then PASS (safety 1.0, nondegenerate).
- Empirical certification only (Wilson lower bound recorded in rule JSON).

## 7. Observability after recalibration

- true_safety_rate: `1.0`
- unique_final_u_ctrl_count: `5`
- final_u_ctrl_std: `0.12736263188235394`
- fraction_changed_from_prior: `0.583`
- real Spearman: `0.39284937787096685` vs shuffled `-0.04250647778309007`

## 8. DAD checkpoint selection

- Changed to safety-first: reject `validation_safety < 1.0`, else minimize validation mean `u_ctrl`.
- See `dad_checkpoint_safety.csv`.

## 9. Rerun T=2 (development_test, n=1000)

- **dad**: safety=1.0, mean_u=0.8246000000000001, excess=0.50555
- **myopic**: safety=1.0, mean_u=0.8688, excess=0.54975
- **fixed**: safety=1.0, mean_u=0.8501000000000002, excess=0.5310499999999999
- **random**: safety=1.0, mean_u=0.8916000000000002, excess=0.5725500000000001

## 10. Sealed final test

- Seed: `917531`
- Hash: `c422c4667bf29e00`
- Path: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/experiments/ieee5_policy_robust_calibration_T2/sealed_final_test/final_test.json`
- Not evaluated; do not recalibrate from this set.

## 11. T=3/T=4 resume

**Yes — can resume**, because DAD/Myopic/Fixed/Random all have safety 1.0 under the frozen common rule (margin 0.55).

