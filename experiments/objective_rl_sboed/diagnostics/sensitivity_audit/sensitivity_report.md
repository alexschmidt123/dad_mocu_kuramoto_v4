# Terminal u_ctrl sensitivity audit

Train/validation systems only. Confirmation/test unused.
Primary metric: snapped terminal `u_ctrl`. Diagnostic: continuous `u_raw`.

## ieee5 T=3

- Sequences evaluated: 400
- Unique terminal u_ctrl: 49
- Unique terminal u_raw: 98
- std(u_ctrl): 0.018257
- std(u_raw): 0.025013
- Modal identical fraction: 0.140
- Best−second gap: 0.000000
- Mean |u_raw−u_ctrl|: 0.004477
- Mean order gap: 0.0012500000000000844
- Mean single-probe replace gap: nan
- Likely drivers: C_snap_up_quantization, E_probes_produce_similar_posterior_changes

## ieee9 T=3

- Sequences evaluated: 400
- Unique terminal u_ctrl: 44
- Unique terminal u_raw: 93
- std(u_ctrl): 0.011714
- std(u_raw): 0.027748
- Modal identical fraction: 0.195
- Best−second gap: 0.001000
- Mean |u_raw−u_ctrl|: 0.042052
- Mean order gap: nan
- Mean single-probe replace gap: 0.008499999999999897
- Likely drivers: C_snap_up_quantization

## Decision (Part III)

**No experiment-design or control-grid modification in this study version.**

Reasons:

1. Objective differences across sequences exist but are small (std ~0.01–0.02), consistent with the completed Case-B adaptive-value study.
2. `u_raw` shows somewhat finer variation than snapped `u_ctrl`, indicating snap_up quantization contributes — changing the grid/safety rule would create a new experiment version and must not be mixed into the primary DAD vs RL-sBOED contrast.
3. The controlled comparison proceeds under the **frozen** terminal rules already used by IEEE5/IEEE9 T=3 authoritative experiments.

Optional later ablation (not primary): RL-sBOED-raw-reward diagnostic using `u_raw` stepwise differences, still evaluated by snapped terminal `u_ctrl`.

