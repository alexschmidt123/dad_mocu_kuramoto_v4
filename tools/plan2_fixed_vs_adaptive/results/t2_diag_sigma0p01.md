# Plan-2 T=2 Fixed vs adaptive diagnostic

- config: `configs/ieee5_plan2_trap.yaml`
- bank: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/data/ieee5_plan2_trap_v3`
- sigma: `0.01`, N_obs=`200`, seed=`101`
- catalog note: Frozen Plan-2 v2 catalog from config (amp×duration×bus). ChatGPT brief listed D={0.05..0.20}; frozen YAML uses D={0.03,0.06,0.10,0.15,0.22,0.30}.

## Table A — screen objectives (expected terminal u_ctrl, lower better)

| Method | J (approx) | notes |
|--------|------------|-------|
| Myopic (greedy ξ1 + adaptive ξ2) | 0.675990 | |
| T=2 adaptive planner | 0.668203 | best ξ1 under branch-optimal ξ2 |
| Optimized Fixed (approx on candidates) | 0.673125 | pair=[26, 25] |
| planning − Fixed | -0.004922 | <0 ⇒ adaptive room |
| planning − Myopic | -0.007786 | Myopic trap if <0 |

## Fixed near-best subsets

- #1: pair=(26,25) J=0.673125
- #2: pair=(25,26) J=0.675000
- #3: pair=(26,20) J=0.675625
- #4: pair=(21,25) J=0.676250
- #5: pair=(20,26) J=0.677500
- #6: pair=(21,20) J=0.679375
- #7: pair=(20,21) J=0.680625
- #8: pair=(25,21) J=0.681250

## Table B — per-ξ1 branching structure

| a1 | bus | dur | J_adapt | n_a2 | P(mode a2) | H(a2) | eff_n | Δu_branch mean | frac Δu>0 | frac u changes |
|----|-----|-----|---------|------|------------|-------|------|----------------|-----------|----------------|
| 25 | 0 | 0.45 | 0.66820 | 14 | 0.46 | 2.005 | 4.17 | 0.00581 | 0.52 | 0.25 |
| 20 | 0 | 0.32 | 0.67534 | 9 | 0.54 | 1.581 | 3.06 | 0.00367 | 0.40 | 0.17 |
| 26 | 1 | 0.45 | 0.67599 | 10 | 0.21 | 2.008 | 6.44 | 0.00401 | 0.67 | 0.23 |
| 29 | 4 | 0.45 | 0.67750 | 11 | 0.50 | 1.757 | 3.52 | 0.00755 | 0.46 | 0.25 |
| 28 | 3 | 0.45 | 0.68057 | 11 | 0.50 | 1.694 | 3.43 | 0.01763 | 0.81 | 0.42 |
| 24 | 4 | 0.32 | 0.68357 | 11 | 0.54 | 1.639 | 3.08 | 0.00456 | 0.40 | 0.21 |
| 21 | 1 | 0.32 | 0.68422 | 10 | 0.38 | 1.859 | 4.63 | 0.00448 | 0.56 | 0.19 |
| 27 | 2 | 0.45 | 0.68630 | 9 | 0.71 | 1.171 | 1.94 | 0.00349 | 0.23 | 0.06 |
| 15 | 0 | 0.2 | 0.68812 | 9 | 0.71 | 1.178 | 1.94 | 0.00898 | 0.96 | 0.27 |
| 16 | 1 | 0.2 | 0.69677 | 9 | 0.33 | 1.743 | 4.59 | 0.00427 | 0.60 | 0.23 |
| 11 | 1 | 0.12 | 0.70557 | 8 | 0.27 | 1.813 | 5.36 | 0.00953 | 0.77 | 0.48 |
| 6 | 1 | 0.07 | 0.70930 | 6 | 0.75 | 0.942 | 1.73 | 0.00154 | 0.23 | 0.12 |

## Branching at planning-optimal ξ1

- a1=25 meta={'amp': 0.15, 'duration': 0.45, 'bus': 0}
- a2 mass={'26': 0.4583333333333333, '28': 0.10416666666666667, '27': 0.0625, '29': 0.0625, '16': 0.041666666666666664, '17': 0.041666666666666664, '22': 0.041666666666666664, '23': 0.041666666666666664, '24': 0.041666666666666664, '0': 0.020833333333333332, '1': 0.020833333333333332, '13': 0.020833333333333332, '18': 0.020833333333333332, '21': 0.020833333333333332}
- most_common_prob=0.458 (≈open-loop if ≳0.9)
- mean branch value Δu=0.00581 (Fixed-forced − adapt)
- frac histories with Δu>0: 0.521
- frac terminal discrete u changes: 0.250

## Table H — learned policies on held-out rollouts

| method | mean_gap | uniq | H_seq | frac≡Fixed mode | uniq a2 | P(mode a2) |
|--------|----------|------|-------|-----------------|---------|------------|

Fixed mode sequence: `None`

## Preliminary category (T=2 screen)

Category 4 / mixed — planner edge and branching evidence are both modest; interpret with rollout Table H.
