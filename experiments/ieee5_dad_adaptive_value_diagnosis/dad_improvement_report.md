# IEEE5 DAD adaptive-value diagnosis

Frozen IEEE5 final reports were **not** modified.
Final test systems were **not** used.

**Overall case:** `A_no_measurable_adaptive_value`

IEEE5 provides little measurable value of observation-dependent adaptation under the current objective and design space.

## Adaptive reference vs Exact Fixed

| T | mode | J_adaptive | J_fixed | Δ_adapt | CI |
|---|---|---:|---:|---:|---|
| 2 | exact_nested_MC_validation_outer | 0.841992 | 0.808333 | -0.033659 | [-0.0591813151041668, -0.006705729166666785] |
| 3 | approximate_crn_myopic_adaptive_scenario | 0.891667 | 0.783333 | -0.108333 | [-0.19166666666666665, -0.029166666666666657] |
| 4 | approximate_crn_myopic_adaptive_scenario | 0.879167 | 0.779167 | -0.100000 | [-0.16666666666666666, -0.03333333333333333] |

## Decision rule

- proceed_to_dad_improvement: `False`
- move_to_ieee9_recommended: `True`

## Existing DAD observation sensitivity

- seed 101 step 2: unique_actions_across_y_bins=1 mean_TV=0.0004380912120853152 MI=3.3496782913295966e-07
- seed 101 step 3: unique_actions_across_y_bins=1 mean_TV=0.00022406097767608508 MI=9.776963198614638e-08
- seed 101 step 4: unique_actions_across_y_bins=1 mean_TV=0.0001377197913825512 MI=3.6269004129215915e-08
- seed 202 step 2: unique_actions_across_y_bins=1 mean_TV=0.001106845713885767 MI=2.005072842548974e-06
- seed 202 step 3: unique_actions_across_y_bins=1 mean_TV=0.00032593323183911186 MI=1.73748285226953e-07
- seed 202 step 4: unique_actions_across_y_bins=1 mean_TV=0.0004928506511662688 MI=4.7444410666027465e-07
- seed 303 step 2: unique_actions_across_y_bins=1 mean_TV=0.0008665248419025115 MI=1.393583646059535e-06
- seed 303 step 3: unique_actions_across_y_bins=1 mean_TV=0.0004477354232221842 MI=3.5997519541045677e-07
- seed 303 step 4: unique_actions_across_y_bins=1 mean_TV=0.00025232761566128047 MI=9.630994122255684e-08

## Reward quantization

```json
{
  "unique_terminal_cost_count": 5,
  "fraction_at_modal_terminal_cost": 0.3125,
  "terminal_cost_std": 0.125,
  "fraction_of_equal_cost_pairs": 0.21614583333333334,
  "pre_snap_quantile_mean": 0.8197916666666667,
  "pre_snap_quantile_std": 0.1460378030872676,
  "control_grid_level_distribution": {
    "0.6": 6,
    "0.7": 6,
    "0.8": 15,
    "0.9": 12,
    "1.0": 9
  },
  "n_trajectories": 48
}
```

## Training variants

Skipped: Case A says do not force DAD to become adaptive.

## Interpretation

1. At T=2, nested adaptive planning does **not** improve on Exact Fixed
   (Δ CI entirely below 0 under matched validation outer draws).
2. At T=3/T=4, a bank-based myopic-adaptive scenario policy is likewise
   worse than Exact Fixed on paired CRN validation draws.
3. Existing DAD checkpoints select the **same** next action across observation
   bins (unique_actions=1; MI≈0) — they ignore y, consistent with frozen
   nonadaptive rollouts.
4. Therefore IEEE5 should remain a **no-adaptive-value** case under this
   objective; do not retune DAD to chase Fixed. Move to IEEE9 for systems
   where adaptation may matter.

## Complete-history note

DAD `HistoryEncoder` receives `(action_indices, observations, mask)` for all past steps (`src/neural/policy.py`). Call path: rollout buffers → `DADPolicy.forward` → `HistoryEncoder.forward` → attention pool → logits head.

Tests confirm changing an earlier observation (latest pair fixed) changes the
history embedding and logits. The encoder is correct; the learned policy still
collapses to observation-insensitive actions.

