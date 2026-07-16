# IEEE9 adaptive-value diagnosis

Frozen IEEE9 T=2 / T=3 experiment outputs were **not** modified.
Final test systems were **not** used for this diagnosis.
Offline probe/U banks only.

**Overall case: A**

IEEE9 currently provides little measurable value for observation-dependent adaptation under the frozen objective.

- improve_dad_before_T4: `False`
- run_ieee9_T4: `False`
- move_to_ieee14_recommended: `True`

Shared terminal_rule_hash = `5b4c2191621b1bbc` (α=0.05, margin=0.9).

## Fixed reference

- T=2: **exhaustive** J_fixed=0.963333 subset=`[13, 18]`
- T=3: **exhaustive** J_fixed=0.960000 subset=`[0, 4, 6]` (evaluated 24804/24804 combinations)

## Adaptive reference

- T=2: exact_nested_MC_validation_outer J_adapt=0.966536 Δ=-0.003203 CI=[-0.007825520833333877, -0.0003906250000005329]
- T=3 bin-beam: J_adapt≈0.960000 (method=`posterior_predictive_binning_plus_beam`, bins=5, beam=10)
- T=3 CRN myopic-adaptive: J_adapt≈0.978333 Δ_crn=-0.018333 CI=[-0.036666666666666674, -0.005]

## Observation → later action dependence

- T=2: `True` (MI=0.690988460409074)
- T=3: `True` (MI=6.425169210251804, frac histories with branch-dependent next=0.3465909090909091)

## DAD comparison (validation)

| T | J_fixed | J_adapt | J_DAD | J_dom_replay | gap captured |
|---|---|---|---|---|---|
| 2 | 0.963333 | 0.966536 | 0.972800 | 0.977200 | nan |
| 3 | 0.960000 | 0.960000 | 0.972400 | 0.972400 | nan |

## Interpretation

- Case T=2: `A`
- Case T=3: `A`
- Overall: `A`

