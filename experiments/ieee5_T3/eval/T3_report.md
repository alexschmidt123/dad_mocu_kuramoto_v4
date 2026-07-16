# IEEE5 T=3 controlled experiment report

**Passed: True**

## Frozen terminal rule

- α = `0.05`
- additive_margin = `0.55`
- terminal_rule_hash = `c2e2af33cb68a5ea`
- control_grid_hash = `3697ea3799e74809`

### Per-method rule hashes

- `myopic`: `c2e2af33cb68a5ea` (α=0.05, margin=0.55)
- `fixed`: `c2e2af33cb68a5ea` (α=0.05, margin=0.55)
- `random`: `c2e2af33cb68a5ea` (α=0.05, margin=0.55)
- `dad_seed_101`: `c2e2af33cb68a5ea` (α=0.05, margin=0.55)
- `dad_seed_202`: `c2e2af33cb68a5ea` (α=0.05, margin=0.55)
- `dad_seed_303`: `c2e2af33cb68a5ea` (α=0.05, margin=0.55)
- `dad`: `c2e2af33cb68a5ea` (α=0.05, margin=0.55)

## Objective observability

- true_safety_rate: `1.0`
- unique_final_u_ctrl_count: `5`
- final_u_ctrl_std: `0.1258576974205392`
- fraction_changed_from_prior: `0.643`
- real Spearman: `0.4545019677782575`
- shuffled Spearman: `0.018766422616686658`
- gate_passed: `None`

## Per-method results

| method | mean u_ctrl | safety | mean excess | runtime |
|---|---:|---:|---:|---:|
| dad | 0.8433 | 1.000 | 0.5242 | 0.0726s |
| myopic | 0.8604 | 1.000 | 0.5413 | 1.0196s |
| fixed | 0.8420 | 1.000 | 0.5230 | 0.0654s |
| random | 0.8704 | 1.000 | 0.5514 | 0.0658s |

## DAD seeds

- seed 101: mean_u=0.8911 safety=1.000
- seed 202: mean_u=0.8433 safety=1.000
- seed 303: mean_u=0.8621 safety=1.000
- selected primary: seed `202` (safety-first then min val mean u_ctrl)

## Fixed subset

- search_mode: `exhaustive`
- selected_action_ids: `[0, 19, 28]`
- subsets_evaluated: `4060`
- estimated_mean_u_ctrl: `0.846875`
- search_runtime: `8.07095106691122`
- search_seed: `7`

## Random uniformity

`{
  "repeat_action_count": 0,
  "action_count_cv": 0.09983319421247959,
  "n_unique_actions_used": 30,
  "n_actions": 30
}`

## Paired differences

- `dad_minus_myopic`: mean=-0.0171 CI95=[-0.0224,-0.0118] frac_A_lower=0.261 frac_B_lower=0.099 tied=0.640
- `dad_minus_fixed`: mean=0.0013 CI95=[-0.0124,0.0151] frac_A_lower=0.317 frac_B_lower=0.324 tied=0.359
- `dad_minus_random`: mean=-0.0271 CI95=[-0.0370,-0.0173] frac_A_lower=0.350 frac_B_lower=0.200 tied=0.450
- `myopic_minus_fixed`: mean=0.0184 CI95=[0.0051,0.0316] frac_A_lower=0.321 frac_B_lower=0.434 tied=0.245
- `myopic_minus_random`: mean=-0.0100 CI95=[-0.0195,-0.0005] frac_A_lower=0.307 frac_B_lower=0.250 tied=0.443
- `fixed_minus_random`: mean=-0.0284 CI95=[-0.0398,-0.0169] frac_A_lower=0.378 frac_B_lower=0.267 tied=0.355

## DAD adaptivity

- dominant_sequence: `[1, 28, 23]`
- dominant_sequence_fraction: `1.0`
- n_unique_sequences: `1`
- effectively_nonadaptive: `True`

## T=4 resume

**Yes** — T=4 may proceed only if all four methods have safety 1.0 under the frozen rule (current: [1.0, 1.0, 1.0, 1.0]).
