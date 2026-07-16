# IEEE5 T=2 four-method pilot report

**Pilot passed: True**

Terminal rule hash: `c2e2af33cb68a5ea`
α=0.05, margin=0.55, quantile=0.95

## Per-method metrics

| method | mean u_ctrl | safety | mean excess | runtime |
|---|---|---|---|---|
| dad | 0.8246 | 1.000 | 0.5056 | 0.0711s |
| myopic | 0.8688 | 1.000 | 0.5497 | 0.7353s |
| fixed | 0.8501 | 1.000 | 0.5310 | 0.0651s |
| random | 0.8916 | 1.000 | 0.5726 | 0.0660s |

## Paired differences (u_A - u_B)

- `dad_minus_myopic`: mean=-0.0442  CI95=[-0.0566,-0.0315]  frac_A_lower=0.464
- `dad_minus_fixed`: mean=-0.0255  CI95=[-0.0381,-0.0127]  frac_A_lower=0.360
- `dad_minus_random`: mean=-0.0670  CI95=[-0.0787,-0.0551]  frac_A_lower=0.465
- `myopic_minus_fixed`: mean=0.0187  CI95=[0.0144,0.0229]  frac_A_lower=0.069
- `myopic_minus_random`: mean=-0.0228  CI95=[-0.0319,-0.0137]  frac_A_lower=0.326
- `fixed_minus_random`: mean=-0.0415  CI95=[-0.0508,-0.0320]  frac_A_lower=0.371

Fixed subset: `[1, 8]`

