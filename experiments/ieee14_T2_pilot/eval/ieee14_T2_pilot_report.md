# IEEE14 T=2 objective-based SBOED pilot report

**Passed: True**
**can_proceed_to_T3: True**

T=2 is a pipeline/system pilot — do not overinterpret adaptive value.

## Action space
- N = 14
- amplitudes = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
- N_ξ = 84
- C(N_ξ, 2) = 3486

## Bank hashes
```json
{
  "probe_bank_hash": {
    "train": "b989acde0c3dff81",
    "test": "0b1d9b02d90b6f02"
  },
  "U_bank_hash": {
    "train": "79af356895d4415a",
    "test": "2367f0542c1aa706"
  },
  "theta_particle_hash": {
    "train": "4497d32c822ad4ae",
    "test": "23b22df6bf551a4e",
    "train_n": 128,
    "test_n": 32
  },
  "action_space_hash": "97e3d2c6573d3008",
  "config_hash": "64203af24cd94a9c",
  "n_actions": 84,
  "backend_probe": "cuda",
  "particle_ordering_match": {
    "train": {
      "ok": true,
      "n": 128,
      "max_abs_diff": 0.0
    },
    "test": {
      "ok": true,
      "n": 32,
      "max_abs_diff": 0.0
    }
  }
}
```

## Control certification
```json
{
  "passed": true,
  "rates": {
    "train": {
      "u_bank_particle_safety_rate": 1.0,
      "maximum_control_safety_rate": 1.0,
      "oracle_control_safety_rate": 1.0,
      "passed": true
    },
    "test": {
      "u_bank_particle_safety_rate": 1.0,
      "maximum_control_safety_rate": 1.0,
      "oracle_control_safety_rate": 1.0,
      "passed": true
    }
  },
  "infeasible_particle_count": {
    "train": 0,
    "test": 0
  },
  "requirement": {
    "P_safe_oracle": 1.0,
    "P_safe_u_max": 1.0,
    "P_safe_U_bank": 1.0,
    "infeasible_particle_count": 0
  }
}
```

## U-bank variation
```json
{
  "support": {
    "n": 80,
    "mean_U": 0.091,
    "std_U": 0.24091284731205184,
    "unique_U_count": 8,
    "min_U": 0.0,
    "max_U": 1.28,
    "fraction_at_u_min": 0.7,
    "fraction_at_u_max": 0.0,
    "q10": 0.0,
    "q50": 0.0,
    "q90": 0.24,
    "nondegenerate": true
  },
  "calibration": {
    "n": 24,
    "mean_U": 0.16333333333333336,
    "std_U": 0.33978751530266027,
    "unique_U_count": 7,
    "min_U": 0.0,
    "max_U": 1.36,
    "fraction_at_u_min": 0.625,
    "fraction_at_u_max": 0.0,
    "q10": 0.0,
    "q50": 0.0,
    "q90": 0.4319999999999999,
    "nondegenerate": true
  },
  "validation": {
    "n": 24,
    "mean_U": 0.10333333333333333,
    "std_U": 0.2097352828898583,
    "unique_U_count": 7,
    "min_U": 0.0,
    "max_U": 0.96,
    "fraction_at_u_min": 0.625,
    "fraction_at_u_max": 0.0,
    "q10": 0.0,
    "q50": 0.0,
    "q90": 0.29599999999999993,
    "nondegenerate": true
  },
  "test": {
    "n": 32,
    "mean_U": 0.0775,
    "std_U": 0.22758240265890506,
    "unique_U_count": 5,
    "min_U": 0.0,
    "max_U": 1.04,
    "fraction_at_u_min": 0.8125,
    "fraction_at_u_max": 0.0,
    "q10": 0.0,
    "q50": 0.0,
    "q90": 0.08,
    "nondegenerate": true
  }
}
```

## Terminal rule (IEEE14-specific)
- α = 0.05
- margin = 1.4
- terminal_rule_hash = `85a3a2babf9b7d51`
- reused IEEE5/IEEE9 margin: False

## Observability
- passed = True

## Exact Fixed T=2
```json
{
  "selected_action_ids": [
    55,
    56
  ],
  "selected_amplitudes": [
    0.15,
    0.2
  ],
  "selected_buses": [
    13,
    0
  ],
  "estimated_mean_u_ctrl": 1.49,
  "validation_mean_u_ctrl": 1.49,
  "number_of_subsets_evaluated": 3486,
  "search_space_size": 3486,
  "search_runtime": 21.20814689504914,
  "search_seed": 7,
  "search_mode": "exhaustive",
  "exact_or_approximate": "exact",
  "used_test_systems": false,
  "horizon": 2,
  "n_actions": 84,
  "terminal_rule": {
    "terminal_rule_hash": "85a3a2babf9b7d51",
    "alpha": 0.05,
    "additive_margin": 1.4
  },
  "validation_safety": 1.0
}
```

## Fixed-BC DAD
- sequence = `[55, 56]`
- reproduction = 1.0
- vs Fixed equality passed = True

## Fine-tuned DAD selection
- selected seed = 101
- selected checkpoint = 0
- is checkpoint 0 = True
- V1 ck0 count = 5/5
- second action depends on y1 = False

## Test evaluation

| method | mean u_ctrl | safety | runtime/rollout |
|---|---|---|---|
| fixed | 1.4951999999999999 | 1.0 | 0.4905987810442457 |
| dad_fixed_init | 1.4951999999999999 | 1.0 | 0.5064373763470212 |
| dad | 1.4951999999999999 | 1.0 | 0.48965066150587516 |
| myopic | 1.4924000000000002 | 1.0 | 1.24970743892889 |
| random | 1.5582 | 1.0 | 0.4646945044043241 |

## Paired contrasts
```json
{
  "dad_fixed_init_minus_fixed": {
    "mean_paired_diff": 0.0,
    "median_paired_diff": 0.0,
    "std_error": 0.0,
    "ci95_low": 0.0,
    "ci95_high": 0.0,
    "fraction_a_lower": 0.0,
    "fraction_b_lower": 0.0,
    "fraction_tied": 1.0,
    "n": 400,
    "n_bootstrap": 10000
  },
  "dad_minus_dad_fixed_init": {
    "mean_paired_diff": 0.0,
    "median_paired_diff": 0.0,
    "std_error": 0.0,
    "ci95_low": 0.0,
    "ci95_high": 0.0,
    "fraction_a_lower": 0.0,
    "fraction_b_lower": 0.0,
    "fraction_tied": 1.0,
    "n": 400,
    "n_bootstrap": 10000
  },
  "dad_minus_fixed": {
    "mean_paired_diff": 0.0,
    "median_paired_diff": 0.0,
    "std_error": 0.0,
    "ci95_low": 0.0,
    "ci95_high": 0.0,
    "fraction_a_lower": 0.0,
    "fraction_b_lower": 0.0,
    "fraction_tied": 1.0,
    "n": 400,
    "n_bootstrap": 10000
  },
  "dad_minus_dominant_replay": {
    "mean_paired_diff": 0.0,
    "median_paired_diff": 0.0,
    "std_error": 0.0,
    "ci95_low": 0.0,
    "ci95_high": 0.0,
    "fraction_a_lower": 0.0,
    "fraction_b_lower": 0.0,
    "fraction_tied": 1.0,
    "n": 400,
    "n_bootstrap": 10000
  },
  "dad_minus_myopic": {
    "mean_paired_diff": 0.0028000000000000026,
    "median_paired_diff": 0.0,
    "std_error": 0.002398913705704846,
    "ci95_low": -0.0018000000000000017,
    "ci95_high": 0.007600000000000007,
    "fraction_a_lower": 0.09,
    "fraction_b_lower": 0.1125,
    "fraction_tied": 0.7975,
    "n": 400,
    "n_bootstrap": 10000
  },
  "dad_minus_random": {
    "mean_paired_diff": -0.06300000000000006,
    "median_paired_diff": -0.08000000000000007,
    "std_error": 0.004666040058182252,
    "ci95_low": -0.07220000000000006,
    "ci95_high": -0.05400000000000005,
    "fraction_a_lower": 0.56,
    "fraction_b_lower": 0.1125,
    "fraction_tied": 0.3275,
    "n": 400,
    "n_bootstrap": 10000
  }
}
```

- stop_reasons = []
- can_proceed_to_T3 = True

Do not run IEEE14 T=3/T=4 from this pilot.

