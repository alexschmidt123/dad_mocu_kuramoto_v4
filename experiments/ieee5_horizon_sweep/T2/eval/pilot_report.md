# IEEE5 T=2 four-method pilot report

**Pilot passed: False**

Terminal rule hash: `dc0dc35332b394b7`
α=0.05, margin=0.4, quantile=0.95

## Per-method metrics

| method | mean u_ctrl | safety | mean excess | runtime |
|---|---|---|---|---|
| dad | 0.7753 | 0.936 | 0.4562 | 0.0712s |
| myopic | 0.7588 | 1.000 | 0.4398 | 0.7206s |
| fixed | 0.7570 | 1.000 | 0.4380 | 0.0653s |
| random | 0.7880 | 0.986 | 0.4689 | 0.0658s |

## Paired differences (u_A - u_B)

- `dad_minus_myopic`: mean=0.0165  CI95=[0.0022,0.0308]  frac_A_lower=0.296
- `dad_minus_fixed`: mean=0.0183  CI95=[0.0036,0.0333]  frac_A_lower=0.333
- `dad_minus_random`: mean=-0.0126  CI95=[-0.0252,-0.0002]  frac_A_lower=0.369
- `myopic_minus_fixed`: mean=0.0019  CI95=[-0.0031,0.0069]  frac_A_lower=0.157
- `myopic_minus_random`: mean=-0.0291  CI95=[-0.0402,-0.0180]  frac_A_lower=0.394
- `fixed_minus_random`: mean=-0.0310  CI95=[-0.0423,-0.0199]  frac_A_lower=0.402

Fixed subset: `[10, 11]`

