# IEEE5 T=4 exact Fixed fairness report

Approximate Fixed (`greedy_multistart`, 340 subsets) is archived.
This report uses exhaustive search over all 27,405 size-4 subsets.

## Frozen rule (unchanged)

- terminal_rule_hash = `c2e2af33cb68a5ea`
- alpha = 0.05, additive_margin = 0.55

## Exact Fixed selection

- exact_fixed_subset: `[8, 19, 20, 28]`
- exact_fixed_validation_mean: `0.7791666666666668`
- DAD_subset: `[10, 19, 23, 29]`
- DAD_subset_rank: `2223`
- DAD_subset_validation_mean: `0.8333333333333334`
- total_subsets_evaluated: `27405`
- search_runtime: `39.42s`

## Validation subset comparison

- DAD_subset: subset=`10 19 23 29` mean=`0.8333333333333334` safety=`1.0` rank=`2223`
- exact_Fixed: subset=`8 19 20 28` mean=`0.7791666666666668` safety=`1.0` rank=`1`
- archived_approximate_Fixed: subset=`0 1 4 8` mean=`0.8833333333333334` safety=`1.0` rank=`14508`

## Test evaluation (paired rollouts)

- exact Fixed mean u_ctrl: `0.846100`
- exact Fixed safety: `1.000000`

## DAD across seeds (test)

- seed means: `{'101': 0.8207000000000001, '202': 0.8632000000000001, '303': 0.8531000000000002}`
- mean across seeds: `0.845667`
- std across seeds: `0.022204`
- selected-model (seed 101): `0.820700`

## Paired differences (DAD − exact Fixed)

- `dad_seed_101_minus_exact_fixed`: mean=-0.025400 CI95=[-0.033100, -0.018100] tied=0.710
- `dad_seed_202_minus_exact_fixed`: mean=0.017100 CI95=[0.004600, 0.029500] tied=0.326
- `dad_seed_303_minus_exact_fixed`: mean=0.007000 CI95=[-0.003700, 0.017800] tied=0.409
- `dad_primary_minus_exact_fixed`: mean=-0.025400 CI95=[-0.033100, -0.018100] tied=0.710
- `dad_seed_mean_minus_exact_fixed`: mean=-0.000433 CI95=[-0.007933, 0.007000] tied=0.226
- `dad_dominant_sequence_minus_exact_fixed`: mean=-0.025400 CI95=[-0.033100, -0.018100] tied=0.710

## Interpretation

DAD did not demonstrate an adaptive advantage at IEEE5 T=4. Its previous advantage resulted from a stronger learned fixed sequence compared with an underoptimized approximate Fixed baseline. Across training seeds, DAD is statistically tied with exact Fixed; DAD remains effectively nonadaptive (one sequence on every rollout).

- adaptive_benefit_demonstrated: `False`
- dad_equals_exact_fixed: `False`
- can_freeze_ieee5: `True`
