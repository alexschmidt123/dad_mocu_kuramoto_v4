# IEEE9 T=3 controlled experiment report

**Passed: True**

## Frozen rule (from IEEE9 T=2 pilot)
- source: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/experiments/ieee9_T2_pilot`
- terminal_rule_hash = `5b4c2191621b1bbc`
- α=0.05, margin=0.9
- probe/U hashes verified against T=2 finalized banks

## Objective observability
- passed = True

## Frozen-rule safety at T=3
- passed = True

## Per-method

| method | mean u_ctrl | safety | runtime |
|---|---|---|---|
| dad | 0.9697600000000003 | 1.0 | 0.10542648044321686 |
| myopic | 0.9661600000000004 | 1.0 | 0.5963829814761411 |
| fixed | 0.9617600000000004 | 1.0 | 0.10227266684663482 |
| random | 0.9830400000000001 | 1.0 | 0.10230921367160045 |

## DAD adaptivity
- unique sequences = 1
- dominant = `[47, 3, 19]` (frac=1.0)
- interpretation = effectively_nonadaptive
- vs dominant replay: {'mean_paired_diff': 0.0, 'median_paired_diff': 0.0, 'std_error': 0.0, 'ci95_low': 0.0, 'ci95_high': 0.0, 'fraction_a_lower': 0.0, 'fraction_b_lower': 0.0, 'fraction_tied': 1.0, 'n': 1000, 'n_bootstrap': 10000}

Fixed (approximately optimized Fixed baseline; C(54,3)=24804 > 5000): `{'selected_action_ids': [0, 3, 47], 'selected_amplitudes': [0.05, 0.05, 0.3], 'selected_buses': [0, 3, 2], 'estimated_mean_u_ctrl': 0.9675, 'number_of_subsets_evaluated': 424, 'search_runtime': 0.8696843211073428, 'search_seed': 7, 'search_mode': 'greedy_multistart', 'used_test_systems': False, 'terminal_rule': {'terminal_rule_hash': '5b4c2191621b1bbc', 'quantile_level': 0.95, 'additive_margin': 0.9, 'alpha': 0.05, 'snap_up': True, 'control_grid_hash': '2a35b5c585aa23b1', 'u_candidates': [0.0, 0.08, 0.16, 0.24, 0.32, 0.4, 0.48, 0.56, 0.64, 0.72, 0.8, 0.88, 0.96, 1.04, 1.12, 1.2], 'source': '/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/experiments/ieee9_T3/selected_policy_robust_rule.json', 'rule': 'snap_up(Q_{1-alpha}(U|w) + margin)'}}`
Myopic n_h = 256
Random uniformity = `{'repeat_action_count': 0, 'action_count_cv': 0.14406942770761602, 'n_unique_actions_used': 54, 'n_actions': 54}`
