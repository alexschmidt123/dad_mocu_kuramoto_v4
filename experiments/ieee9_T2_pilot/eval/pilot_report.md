# IEEE9 T=2 objective-based SBOED pilot report

**Passed: True**
**May proceed to IEEE9 T=3: True**

## 1. Action space
- N_b = 9
- amplitudes = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3] (6 options)
- N_ξ = 6 N_b = 54
- C(N_ξ, 2) = 1431

## 2. Bank hashes
```json
{
  "probe_bank_hash": {
    "train": "9606950d48493034",
    "test": "2ee0773e1b7d6864"
  },
  "U_bank_hash": {
    "train": "e92d41caf87eac58",
    "test": "0ab75bab90f86d19"
  },
  "theta_particle_hash": {
    "train": "4497d32c822ad4ae",
    "test": "23b22df6bf551a4e",
    "train_n": 128,
    "test_n": 32
  },
  "action_space_hash": "1158467507c384e4",
  "config_hash": "937da74c41562dbc",
  "n_actions": 54,
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

## 3. U-bank variation
```json
{
  "support": {
    "n": 80,
    "mean_U": 0.019000000000000003,
    "std_U": 0.10059324032955694,
    "unique_U_count": 5,
    "min_U": 0.0,
    "max_U": 0.8,
    "fraction_at_u_min": 0.9375,
    "fraction_at_u_max": 0.0,
    "q10": 0.0,
    "q50": 0.0,
    "q90": 0.0,
    "nondegenerate": true
  },
  "calibration": {
    "n": 24,
    "mean_U": 0.04666666666666667,
    "std_U": 0.16152055252780956,
    "unique_U_count": 3,
    "min_U": 0.0,
    "max_U": 0.72,
    "fraction_at_u_min": 0.9166666666666666,
    "fraction_at_u_max": 0.0,
    "q10": 0.0,
    "q50": 0.0,
    "q90": 0.0,
    "nondegenerate": true
  },
  "validation": {
    "n": 24,
    "mean_U": 0.049999999999999996,
    "std_U": 0.19570385790780923,
    "unique_U_count": 3,
    "min_U": 0.0,
    "max_U": 0.96,
    "fraction_at_u_min": 0.9166666666666666,
    "fraction_at_u_max": 0.0,
    "q10": 0.0,
    "q50": 0.0,
    "q90": 0.0,
    "nondegenerate": true
  },
  "test": {
    "n": 32,
    "mean_U": 0.0125,
    "std_U": 0.05695392874947258,
    "unique_U_count": 3,
    "min_U": 0.0,
    "max_U": 0.32,
    "fraction_at_u_min": 0.9375,
    "fraction_at_u_max": 0.0,
    "q10": 0.0,
    "q50": 0.0,
    "q90": 0.0,
    "nondegenerate": true
  }
}
```

## 4. Terminal-control rule
- α = 0.05
- margin = 0.9
- terminal_rule_hash = `5b4c2191621b1bbc`

## 5. Calibration / validation safety by policy
```json
{
  "random": {
    "safety_rate": 1.0,
    "mean_u_ctrl": 1.0062,
    "n_rollouts": 400
  },
  "fixed": {
    "safety_rate": 1.0,
    "mean_u_ctrl": 0.9960000000000002,
    "n_rollouts": 400
  },
  "myopic": {
    "safety_rate": 1.0,
    "mean_u_ctrl": 0.9864000000000002,
    "n_rollouts": 400
  }
}
```
```json
{
  "random": {
    "safety_rate": 1.0,
    "mean_u_ctrl": 0.9984000000000001,
    "n_rollouts": 400
  },
  "fixed": {
    "safety_rate": 1.0,
    "mean_u_ctrl": 1.0054000000000003,
    "n_rollouts": 400
  },
  "myopic": {
    "safety_rate": 1.0,
    "mean_u_ctrl": 0.9846000000000003,
    "n_rollouts": 400
  },
  "exploratory_dad_proxy": {
    "safety_rate": 1.0,
    "mean_u_ctrl": 0.9940000000000002,
    "n_rollouts": 400
  }
}
```

## 6. Objective observability
- passed = True

## 7–8. DAD seeds and across-seed
```json
{
  "n_seeds": 3,
  "mean_u_ctrl_across_seeds": 0.9681866666666671,
  "std_u_ctrl_across_seeds": 0.0017676600980455028,
  "min_true_safety_rate": 1.0,
  "all_seeds_safe": true,
  "seeds": [
    {
      "seed": "101",
      "mean_u_ctrl": 0.9657600000000004,
      "std_u_ctrl": 0.02980641541681927,
      "true_safety_rate": 1.0,
      "mean_excess_control": 0.9530400000000003,
      "mean_runtime_per_rollout": 0.10411585608008318,
      "terminal_rule_hash": "5b4c2191621b1bbc"
    },
    {
      "seed": "202",
      "mean_u_ctrl": 0.9699200000000004,
      "std_u_ctrl": 0.04386335144514154,
      "true_safety_rate": 1.0,
      "mean_excess_control": 0.9572000000000003,
      "mean_runtime_per_rollout": 0.1041446163309738,
      "terminal_rule_hash": "5b4c2191621b1bbc"
    },
    {
      "seed": "303",
      "mean_u_ctrl": 0.9688800000000003,
      "std_u_ctrl": 0.044158188368636685,
      "true_safety_rate": 1.0,
      "mean_excess_control": 0.9561600000000003,
      "mean_runtime_per_rollout": 0.10373325719637796,
      "terminal_rule_hash": "5b4c2191621b1bbc"
    }
  ]
}
```

## 9. DAD adaptivity
```json
{
  "number_of_unique_sequences": 1,
  "dominant_sequence": [
    44,
    3
  ],
  "dominant_sequence_fraction": 1.0,
  "sequence_entropy": -0.0,
  "observation_to_second_action_dependence": false,
  "second_action_distribution_by_observation_bin": [
    {
      "first_action": 44,
      "first_observation_bin": 0,
      "most_common_second_action": 3,
      "second_action_distribution": {
        "3": 1.0
      },
      "n": 200
    },
    {
      "first_action": 44,
      "first_observation_bin": 1,
      "most_common_second_action": 3,
      "second_action_distribution": {
        "3": 1.0
      },
      "n": 200
    },
    {
      "first_action": 44,
      "first_observation_bin": 2,
      "most_common_second_action": 3,
      "second_action_distribution": {
        "3": 1.0
      },
      "n": 200
    },
    {
      "first_action": 44,
      "first_observation_bin": 3,
      "most_common_second_action": 3,
      "second_action_distribution": {
        "3": 1.0
      },
      "n": 200
    },
    {
      "first_action": 44,
      "first_observation_bin": 4,
      "most_common_second_action": 3,
      "second_action_distribution": {
        "3": 1.0
      },
      "n": 200
    }
  ],
  "interpretation": "effectively_nonadaptive"
}
```

## 10–12. Myopic / Fixed / Random
- myopic n_hypothetical = 256
- fixed = ```json
{
  "selected_action_ids": [
    0,
    3
  ],
  "selected_amplitudes": [
    0.05,
    0.05
  ],
  "selected_buses": [
    0,
    3
  ],
  "estimated_mean_u_ctrl": 0.9675,
  "number_of_subsets_evaluated": 1431,
  "search_runtime": 2.6858336930163205,
  "search_seed": 7,
  "search_mode": "exhaustive",
  "used_test_systems": false,
  "terminal_rule": {
    "terminal_rule_hash": "5b4c2191621b1bbc",
    "quantile_level": 0.95,
    "additive_margin": 0.9,
    "alpha": 0.05,
    "snap_up": true,
    "control_grid_hash": "2a35b5c585aa23b1",
    "u_candidates": [
      0.0,
      0.08,
      0.16,
      0.24,
      0.32,
      0.4,
      0.48,
      0.56,
      0.64,
      0.72,
      0.8,
      0.88,
      0.96,
      1.04,
      1.12,
      1.2
    ],
    "source": "/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/experiments/ieee9_T2_pilot/selected_policy_robust_rule.json",
    "rule": "snap_up(Q_{1-alpha}(U|w) + margin)"
  }
}
```
- random uniformity = ```json
{
  "repeat_action_count": 0,
  "action_count_cv": 0.16183015788165073,
  "n_unique_actions_used": 54,
  "n_actions": 54
}
```

## 13. Per-method mean u_ctrl and safety

| method | mean u_ctrl | safety | runtime/rollout |
|---|---|---|---|
| dad | 0.9688800000000003 | 1.0 | 0.10373325719637796 |
| myopic | 0.9667200000000004 | 1.0 | 0.4295053581991233 |
| fixed | 0.9657600000000003 | 1.0 | 0.10150869260681793 |
| random | 0.9952000000000003 | 1.0 | 0.10115518288267776 |

## 14. Paired confidence intervals
```json
{
  "dad_minus_myopic": {
    "mean_paired_diff": 0.002159999999999995,
    "median_paired_diff": 0.0,
    "std_error": 0.001754084151690338,
    "ci95_low": -0.0012800000000000062,
    "ci95_high": 0.005679999999999994,
    "fraction_a_lower": 0.044,
    "fraction_b_lower": 0.04,
    "fraction_tied": 0.916,
    "n": 1000,
    "n_bootstrap": 10000
  },
  "dad_minus_fixed": {
    "mean_paired_diff": 0.0031199999999999956,
    "median_paired_diff": 0.0,
    "std_error": 0.00168555827745786,
    "ci95_low": -0.00016000000000000546,
    "ci95_high": 0.006479999999999995,
    "fraction_a_lower": 0.034,
    "fraction_b_lower": 0.038,
    "fraction_tied": 0.928,
    "n": 1000,
    "n_bootstrap": 10000
  },
  "dad_minus_random": {
    "mean_paired_diff": -0.02632000000000001,
    "median_paired_diff": 0.0,
    "std_error": 0.0026361727953620583,
    "ci95_low": -0.03152000000000001,
    "ci95_high": -0.021200000000000014,
    "fraction_a_lower": 0.219,
    "fraction_b_lower": 0.031,
    "fraction_tied": 0.75,
    "n": 1000,
    "n_bootstrap": 10000
  },
  "myopic_minus_fixed": {
    "mean_paired_diff": 0.0009600000000000009,
    "median_paired_diff": 0.0,
    "std_error": 0.0009190882020367383,
    "ci95_low": -0.0008000000000000007,
    "ci95_high": 0.0028800000000000023,
    "fraction_a_lower": 0.014,
    "fraction_b_lower": 0.022,
    "fraction_tied": 0.964,
    "n": 1000,
    "n_bootstrap": 10000
  },
  "myopic_minus_random": {
    "mean_paired_diff": -0.028480000000000005,
    "median_paired_diff": 0.0,
    "std_error": 0.002527623088689942,
    "ci95_low": -0.03360000000000001,
    "ci95_high": -0.023600000000000006,
    "fraction_a_lower": 0.221,
    "fraction_b_lower": 0.039,
    "fraction_tied": 0.74,
    "n": 1000,
    "n_bootstrap": 10000
  },
  "fixed_minus_random": {
    "mean_paired_diff": -0.029440000000000004,
    "median_paired_diff": 0.0,
    "std_error": 0.0024625496036731025,
    "ci95_low": -0.034400000000000014,
    "ci95_high": -0.024720000000000006,
    "fraction_a_lower": 0.223,
    "fraction_b_lower": 0.029,
    "fraction_tied": 0.748,
    "n": 1000,
    "n_bootstrap": 10000
  }
}
```

## 16. Stop conditions
- stop_reasons = []
- can_proceed_to_T3 = True

Do not run IEEE9 T=3/T=4 or IEEE14 from this pilot.
