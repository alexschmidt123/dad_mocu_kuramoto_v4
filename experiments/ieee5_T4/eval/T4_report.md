# IEEE5 T=4 controlled experiment report

**Passed: True**

## Frozen terminal rule

- α = `0.05`
- additive_margin = `0.55`
- terminal_rule_hash = `c2e2af33cb68a5ea` (expected `c2e2af33cb68a5ea`)

### Per-method hashes

- `myopic`: `c2e2af33cb68a5ea`
- `fixed`: `c2e2af33cb68a5ea`
- `random`: `c2e2af33cb68a5ea`
- `dad_seed_101`: `c2e2af33cb68a5ea`
- `dad_seed_202`: `c2e2af33cb68a5ea`
- `dad_seed_303`: `c2e2af33cb68a5ea`
- `dad`: `c2e2af33cb68a5ea`

## Objective observability

- true_safety_rate: `1.0`
- unique_final_u_ctrl_count: `5`
- final_u_ctrl_std: `0.12668034575260678`
- fraction_changed_from_prior: `0.671`
- real Spearman: `0.528085274350811` vs shuffled `0.02077109471449168`

### Stepwise

- t=0: unique=1 mean=1.0 std=0.0 changed_from_prev=0.0 ESS=39.99999999999999
- t=1: unique=5 mean=0.9486 std=0.0868218866415606 changed_from_prev=0.339 ESS=5.244713836658685
- t=2: unique=5 mean=0.8874000000000001 std=0.12736263188235394 changed_from_prev=0.443 ESS=1.8156261847584236
- t=3: unique=5 mean=0.8772000000000001 std=0.1258576974205392 changed_from_prev=0.323 ESS=1.2648662219559041
- t=4: unique=5 mean=0.8703 std=0.12668034575260678 changed_from_prev=0.255 ESS=1.1031867492936358

## Per-method results

| method | mean u_ctrl | safety | excess | runtime |
|---|---:|---:|---:|---:|
| dad | 0.8207 | 1.000 | 0.5016 | 0.0739s |
| myopic | 0.8528 | 1.000 | 0.5337 | 1.3147s |
| fixed | 0.8542 | 1.000 | 0.5352 | 0.0659s |
| random | 0.8688 | 1.000 | 0.5497 | 0.0657s |

## DAD seeds

- seed 101: mean_u=0.8207 safety=1.000
- seed 202: mean_u=0.8632 safety=1.000
- seed 303: mean_u=0.8531 safety=1.000
- primary: seed `101`

## DAD adaptivity

- unique sequences: `1`
- dominant sequence: `[23, 29, 10, 19]`
- dominant fraction: `1.0`
- sequence entropy: `-0.0`
- unique actions step 2/3/4: `1` / `1` / `1`
- interpretation: **effectively_nonadaptive**

## DAD vs dominant-sequence diagnostic

- label: `dad_dominant_sequence_diagnostic` (not a primary method)
- mean paired diff (DAD − dominant): `0.0`
- CI95: `[0.0, 0.0]`
- fraction tied: `1.0`

## Myopic ties

`{
  "exact_tie_rate": 0.999,
  "near_tie_rate": 1.0,
  "mean_exact_ties_per_rollout": 50.272,
  "mean_near_ties_per_rollout": 57.895,
  "median_co_tied_action_count": 14.0,
  "mean_score_gap": 0.0004112548828125001,
  "action_index_tie_break_count": 2315,
  "mean_runtime_per_decision": 0.31116078552085674
}`

## Fixed

- exhaustive_or_approximate: `approximately optimized Fixed baseline`
- search_mode: `greedy_multistart`
- selected: `[0, 1, 4, 8]`
- subsets_evaluated: `340`
- validation_mean_u_ctrl: `0.85`
- runtime: `0.6580761109944433`

## Random uniformity

`{
  "repeat_action_count": 0,
  "action_count_cv": 0.07292976072907412,
  "n_unique_actions_used": 30,
  "n_actions": 30
}`

## Paired differences

- `dad_minus_myopic`: mean=-0.0321 CI95=[-0.0459,-0.0185] tied=0.309
- `dad_minus_fixed`: mean=-0.0335 CI95=[-0.0448,-0.0226] tied=0.490
- `dad_minus_random`: mean=-0.0481 CI95=[-0.0603,-0.0365] tied=0.358
- `myopic_minus_fixed`: mean=-0.0014 CI95=[-0.0077,0.0049] tied=0.563
- `myopic_minus_random`: mean=-0.0160 CI95=[-0.0243,-0.0078] tied=0.497
- `fixed_minus_random`: mean=-0.0146 CI95=[-0.0222,-0.0071] tied=0.548
