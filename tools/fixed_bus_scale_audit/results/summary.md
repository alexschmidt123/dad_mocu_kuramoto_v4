# Fixed-bus scale audit summary

Compare **duration scale** vs **amp scale** on bus 0 (Myopic trap + adaptive room). Pass/fail only — no data filtering.

- N_obs=200
- noise_sigmas=[0.01, 0.001]
- smoke=False

| Variant | σ | Myopic trap | Adaptive room | plan−Fixed | ξ₂ distinct | amp_scale_redundant |
|---------|---|-------------|---------------|------------|------------|---------------------|
| duration_scale | 0.01 | False | True | -0.003828 | 4 | False |
| duration_scale | 0.001 | True | False | 0.000234 | 2 | False |
| amp_scale | 0.01 | True | True | -0.005234 | 3 | True |
| amp_scale | 0.001 | True | False | 0.000547 | 2 | True |

## Takeaway

duration_scale: myopic_trap_all_σ=False, adaptive_room_all_σ=False. amp_scale: myopic_trap_all_σ=True, adaptive_room_all_σ=False. amp_scale flagged amp_scale_redundant (same-bus multi-amp ≈ ROCOF scaling) — expected; this is why multi-amp alone rarely creates a Fixed-beatable adaptive problem. Neither fixed-bus catalog alone provides adaptive_room on all σ — need bus diversity and/or stronger U heterogeneity in the generator YAML (not data filtering).
