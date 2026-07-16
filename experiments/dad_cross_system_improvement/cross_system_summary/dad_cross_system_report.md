# Cross-system Fixed-init DAD improvement report

**Selected training variant:** `V1_fixed_init_terminal`
**Screen decision:** `V1_fixed_init_terminal` on `ieee5_T2`

## Fixed solutions (loaded from diagnosis/exact metadata)

- `ieee5_T2`: seq=`[23, 24]` (exact) val_u=0.8083333333333332 hash=c2e2af33cb68a5ea
- `ieee5_T3`: seq=`[18, 19, 25]` (exact) val_u=0.7833333333333333 hash=c2e2af33cb68a5ea
- `ieee5_T4`: seq=`[8, 19, 20, 28]` (exact) val_u=0.7791666666666668 hash=c2e2af33cb68a5ea
- `ieee9_T2`: seq=`[13, 18]` (exact) val_u=0.9633333333333333 hash=5b4c2191621b1bbc
- `ieee9_T3`: seq=`[0, 4, 6]` (exact) val_u=0.96 hash=5b4c2191621b1bbc

## Confirmation mean u_ctrl (safety must be 1.0)

| system_T | Fixed | DAD | Myopic | Random |
|---|---|---|---|---|
| ieee5_T2 | 0.8305 | 0.8305 | 0.8692 | 0.8888 |
| ieee5_T3 | 0.8480 | 0.8480 | 0.8588 | 0.8745 |
| ieee5_T4 | 0.8455 | 0.8180 | 0.8553 | 0.8735 |
| ieee9_T2 | 0.9874 | 0.9874 | 0.9678 | 0.9974 |
| ieee9_T3 | 0.9788 | 0.9788 | 0.9648 | 0.9854 |

## Adaptivity

- `ieee5_T2`: effectively nonadaptive (obs_dep=False, ck=0)
- `ieee5_T3`: effectively nonadaptive (obs_dep=False, ck=24)
- `ieee5_T4`: effectively nonadaptive (obs_dep=False, ck=22)
- `ieee9_T2`: effectively nonadaptive (obs_dep=False, ck=0)
- `ieee9_T3`: effectively nonadaptive (obs_dep=False, ck=0)

## Notes
- Frozen IEEE5/IEEE9 experiment trees were not modified.
- Confirmation uses sealed frozen test banks; not used for model selection.
- No IEEE14; no ADP-sOED; no EIG in DAD objective.

