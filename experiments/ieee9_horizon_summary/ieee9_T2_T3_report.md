# IEEE9 T=2 vs T=3 horizon summary

Authoritative T=2 source: `experiments/ieee9_T2_pilot/`.
Shared terminal_rule_hash = `5b4c2191621b1bbc` (α=0.05, margin=0.9).

## Mean u_ctrl

| T | dad | myopic | fixed | random |
|---|---|---|---|---|
| 2 | 0.9688800000000003 | 0.9667200000000004 | 0.9657600000000003 | 0.9952000000000003 |
| 3 | 0.9697600000000003 | 0.9661600000000004 | 0.9617600000000004 | 0.9830400000000001 |

## Answers

- **T3_reduces_control_vs_T2**: `{'dad': False, 'myopic': True, 'fixed': True, 'random': True}`
- **dad_more_adaptive_at_T3**: `False`
- **dad_beats_dominant_replay**: `False`
- **dad_beats_fair_fixed**: `False`
- **myopic_improves_vs_fixed**: `False`
- **random_weakest**: `True`

## DAD adaptivity by T
```json
[
  {
    "T": 2,
    "n_unique": 1,
    "dominant": [
      44,
      3
    ],
    "dominant_frac": 1.0,
    "entropy": -0.0,
    "interpretation": "effectively_nonadaptive"
  },
  {
    "T": 3,
    "n_unique": 1,
    "dominant": [
      47,
      3,
      19
    ],
    "dominant_frac": 1.0,
    "entropy": -0.0,
    "interpretation": "effectively_nonadaptive"
  }
]
```
