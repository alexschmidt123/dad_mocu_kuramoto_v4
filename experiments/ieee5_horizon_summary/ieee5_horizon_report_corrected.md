# IEEE5 horizon report (corrected: exact Fixed at T=4)

Frozen terminal rule: α=0.05, margin=0.55, hash=`c2e2af33cb68a5ea`.

## Adaptivity

- T=3 DAD: effectively nonadaptive (one sequence, fraction 1.0).
- T=4 DAD: effectively nonadaptive (sequence `[23,29,10,19]`, fraction 1.0).
- No observation-dependent adaptation benefit is claimed.

## T=4 Fixed correction

- Exact Fixed subset: `[8, 19, 20, 28]`
- Exact Fixed validation mean: `0.7791666666666668`
- DAD subset rank: `2223` / 27405
- Exact Fixed test mean u_ctrl: `0.846100`
- Exact Fixed safety: `1.000000`
- Previous approximate Fixed (`[0,1,4,8]`, 340 subsets) remains archived.

## T=4 DAD results

- Selected-model (seed 101): `0.820700`
- Across-seed mean ± std: `0.845667 ± 0.022204`
- Seed means: `{'101': 0.8207000000000001, '202': 0.8632000000000001, '303': 0.8531000000000002}`

## Paired DAD − exact Fixed

- `dad_seed_101_minus_exact_fixed`: mean=-0.025400 CI95=[-0.033100, -0.018100]
- `dad_seed_202_minus_exact_fixed`: mean=0.017100 CI95=[0.004600, 0.029500]
- `dad_seed_303_minus_exact_fixed`: mean=0.007000 CI95=[-0.003700, 0.017800]
- `dad_primary_minus_exact_fixed`: mean=-0.025400 CI95=[-0.033100, -0.018100]
- `dad_seed_mean_minus_exact_fixed`: mean=-0.000433 CI95=[-0.007933, 0.007000]
- `dad_dominant_sequence_minus_exact_fixed`: mean=-0.025400 CI95=[-0.033100, -0.018100]

## Interpretation

DAD did not demonstrate an adaptive advantage at IEEE5 T=4. Its previous advantage resulted from a stronger learned fixed sequence compared with an underoptimized approximate Fixed baseline. Across training seeds, DAD is statistically tied with exact Fixed; DAD remains effectively nonadaptive (one sequence on every rollout).

- Adaptive benefit demonstrated: `False`
- IEEE5 scientifically freezable: `True`

## Mean u_ctrl by T (reference; T=4 Fixed = exact)

| T | dad (primary) | fixed | myopic | random |
|---|---:|---:|---:|---:|
| 2 | 0.8246 | 0.8501 | 0.8688 | 0.8916 |
| 3 | 0.8433 | 0.8420 | 0.8604 | 0.8704 |
| 4 | 0.8207 | 0.8461 | 0.8528 | 0.8688 |
