# IEEE9 DAD improvement report

Frozen IEEE9 T=2/T=3 outputs were **not** modified.
Finalized test systems were **not** used for tuning.

**DAD optimization resolved:** `True`
**Proceed to IEEE14:** `True`

Fixed sequence target: `[0, 3, 47]`
terminal_rule_hash: `5b4c2191621b1bbc`

## Fixed initialization

- reproduction_rate: `1.0`
- vs Fixed replay passed: `True`
- Fixed-init val u_ctrl: `0.98`
- Fixed open-loop val u_ctrl: `0.98`
- Original DAD val u_ctrl: `0.9700000000000001`

## Variant summary (validation)

- **V1_fixed_init_terminal**: mean=0.975333±0.004000 frac_ck0=0.40 improved/tied/degraded=3/2/0
- **V2_fixed_init_potential**: mean=0.976667±0.002981 frac_ck0=0.40 improved/tied/degraded=3/2/0
- **V3_fixed_init_potential_normadv**: mean=0.980000±0.000000 frac_ck0=1.00 improved/tied/degraded=0/5/0

Best variant: `V1_fixed_init_terminal` seed `404`
Fraction selecting checkpoint 0: `0.4`
Fine-tuning improved any seed: `True`
Observation-dependent: `False`

## Confirmation (calibration systems)

- improved_dad: mean_u=0.976667 safety=1.0
- fixed_init_dad: mean_u=0.973333 safety=1.0
- fixed: mean_u=0.973333 safety=1.0
- original_dad: mean_u=0.976667 safety=1.0

## Paired contrasts

- improved_dad_minus_fixed: mean=0.0033333333333333457 CI=[-0.02666666666666666, 0.030000000000000027]
- improved_dad_minus_original_dad: mean=9.25185853854297e-18 CI=[-0.033333333333333326, 0.030000000000000027]
- fixed_init_dad_minus_fixed: mean=0.0 CI=[0.0, 0.0]
