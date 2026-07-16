# IEEE5 T=2 four-method pilot report

**Pilot passed: True**

Terminal rule hash: `dc0dc35332b394b7`
α=0.05, margin=0.4, quantile=0.95

Primary DAD seed: **202** (best validation mean u_ctrl)

## Per-method metrics

| method | mean u_ctrl | safety | mean excess | runtime |
|---|---|---|---|---|
| dad | 0.7053 | 1.000 | 0.3862 | 0.0601s |
| myopic | 0.7605 | 1.000 | 0.4415 | 0.0718s |
| fixed | 0.7570 | 1.000 | 0.4380 | 0.0536s |
| random | 0.7880 | 1.000 | 0.4689 | 0.0536s |

## DAD per-seed

- seed 101: mean_u=0.7711, safety=1.000
- seed 202: mean_u=0.7053, safety=1.000
- seed 303: mean_u=0.7315, safety=1.000
- aggregate mean±std: 0.7359 ± 0.0271

## Paired differences (u_A - u_B)

- `dad_minus_myopic`: mean=-0.0552  CI95=[-0.0687,-0.0415]  frac_A_lower=0.512
- `dad_minus_fixed`: mean=-0.0517  CI95=[-0.0659,-0.0381]  frac_A_lower=0.481
- `dad_minus_random`: mean=-0.0826  CI95=[-0.0963,-0.0689]  frac_A_lower=0.521
- `myopic_minus_fixed`: mean=0.0035  CI95=[-0.0016,0.0086]  frac_A_lower=0.140

Fixed subset: `[10, 11]`

