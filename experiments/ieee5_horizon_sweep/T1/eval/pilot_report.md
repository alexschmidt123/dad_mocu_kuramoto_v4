# IEEE5 T=2 four-method pilot report

**Pilot passed: True**

Terminal rule hash: `dc0dc35332b394b7`
α=0.05, margin=0.4, quantile=0.95

## Per-method metrics

| method | mean u_ctrl | safety | mean excess | runtime |
|---|---|---|---|---|
| dad | 0.7839 | 1.000 | 0.4649 | 0.0697s |
| myopic | 0.7830 | 1.000 | 0.4640 | 0.3960s |
| fixed | 0.7875 | 1.000 | 0.4685 | 0.0657s |
| random | 0.8690 | 1.000 | 0.5500 | 0.0651s |

## Paired differences (u_A - u_B)

- `dad_minus_myopic`: mean=0.0009  CI95=[-0.0014,0.0033]  frac_A_lower=0.066
- `dad_minus_fixed`: mean=-0.0036  CI95=[-0.0062,-0.0009]  frac_A_lower=0.110
- `dad_minus_random`: mean=-0.0851  CI95=[-0.0950,-0.0753]  frac_A_lower=0.531
- `myopic_minus_fixed`: mean=-0.0045  CI95=[-0.0073,-0.0017]  frac_A_lower=0.123
- `myopic_minus_random`: mean=-0.0860  CI95=[-0.0960,-0.0762]  frac_A_lower=0.529
- `fixed_minus_random`: mean=-0.0815  CI95=[-0.0913,-0.0717]  frac_A_lower=0.522

Fixed subset: `[16]`

