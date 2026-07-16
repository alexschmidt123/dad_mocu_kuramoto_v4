# IEEE5 horizon summary (T=2, T=3, T=4)

Frozen terminal rule: α=0.05, margin=0.55.

## Mean u_ctrl by T

| T | dad | myopic | fixed | random |
|---|---:|---:|---:|---:|
| 2 | 0.8246 | 0.8688 | 0.8501 | 0.8916 |
| 3 | 0.8433 | 0.8604 | 0.8420 | 0.8704 |
| 4 | 0.8207 | 0.8528 | 0.8542 | 0.8688 |

## Answers

1. T=4 lower than T=3? `{'dad': True, 'myopic': True, 'fixed': False, 'random': True}`
2. DAD tied with Fixed at T=4? `False`
3. DAD adaptive at T=4? `False`
4. Myopic worse than Fixed at T=4? `False`
5. Random weakest at T=4? `True`
6. DAD−Random by T: `{'2': -0.067, '3': -0.027099999999999996, '4': -0.04810000000000001}`
7. T=4 cost justification: `{'dad_mean_u_reduction_T3_to_T4': 0.022599999999999953, 'dad_runtime_T4': 0.07389946051058359, 'note': 'Justified if control reduction is material relative to extra probe/runtime cost.'}`
