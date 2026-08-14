# Plan-2 T=2 Fixed vs adaptive diagnostic

- config: `configs/ieee5_plan2_trap.yaml`
- bank: `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/data/ieee5_plan2_trap_v3`
- sigma: `0.005`, N_obs=`200`, seed=`101`
- catalog note: Frozen Plan-2 v2 catalog from config (amp×duration×bus). ChatGPT brief listed D={0.05..0.20}; frozen YAML uses D={0.03,0.06,0.10,0.15,0.22,0.30}.

## Table A — screen objectives (expected terminal u_ctrl, lower better)

| Method | J (approx) | notes |
|--------|------------|-------|
| Myopic (greedy ξ1 + adaptive ξ2) | 0.655521 | |
| T=2 adaptive planner | 0.641510 | best ξ1 under branch-optimal ξ2 |
| Optimized Fixed (approx on candidates) | 0.653750 | pair=[26, 20] |
| planning − Fixed | -0.012240 | <0 ⇒ adaptive room |
| planning − Myopic | -0.014010 | Myopic trap if <0 |

## Fixed near-best subsets

- #1: pair=(26,20) J=0.653750
- #2: pair=(26,25) J=0.653750
- #3: pair=(28,29) J=0.653750
- #4: pair=(25,26) J=0.654375
- #5: pair=(29,26) J=0.654375
- #6: pair=(20,26) J=0.654375
- #7: pair=(27,20) J=0.654375
- #8: pair=(28,24) J=0.655000

## Table B — per-ξ1 branching structure

| a1 | bus | dur | J_adapt | n_a2 | P(mode a2) | H(a2) | eff_n | Δu_branch mean | frac Δu>0 | frac u changes |
|----|-----|-----|---------|------|------------|-------|------|----------------|-----------|----------------|
| 25 | 0 | 0.45 | 0.64151 | 14 | 0.40 | 2.119 | 5.14 | 0.00260 | 0.25 | 0.06 |
| 20 | 0 | 0.32 | 0.64667 | 17 | 0.27 | 2.485 | 8.47 | 0.00258 | 0.40 | 0.12 |
| 29 | 4 | 0.45 | 0.64755 | 19 | 0.19 | 2.704 | 11.76 | 0.00432 | 0.52 | 0.10 |
| 24 | 4 | 0.32 | 0.65138 | 19 | 0.12 | 2.776 | 14.05 | 0.00430 | 0.48 | 0.19 |
| 28 | 3 | 0.45 | 0.65190 | 19 | 0.23 | 2.623 | 10.11 | 0.00417 | 0.35 | 0.04 |
| 26 | 1 | 0.45 | 0.65552 | 17 | 0.15 | 2.542 | 10.47 | 0.00435 | 0.56 | 0.15 |
| 27 | 2 | 0.45 | 0.65781 | 17 | 0.17 | 2.566 | 10.47 | 0.00521 | 0.58 | 0.12 |
| 21 | 1 | 0.32 | 0.65810 | 15 | 0.17 | 2.492 | 10.47 | 0.00518 | 0.67 | 0.23 |
| 15 | 0 | 0.2 | 0.65979 | 18 | 0.15 | 2.653 | 11.88 | 0.00375 | 0.58 | 0.21 |
| 16 | 1 | 0.2 | 0.66211 | 14 | 0.27 | 2.184 | 6.55 | 0.00240 | 0.52 | 0.15 |
| 11 | 1 | 0.12 | 0.66721 | 9 | 0.40 | 1.699 | 4.13 | 0.00385 | 0.54 | 0.12 |
| 6 | 1 | 0.07 | 0.67253 | 8 | 0.33 | 1.753 | 4.70 | 0.00578 | 0.69 | 0.38 |

## Branching at planning-optimal ξ1

- a1=25 meta={'amp': 0.15, 'duration': 0.45, 'bus': 0}
- a2 mass={'0': 0.3958333333333333, '8': 0.10416666666666667, '27': 0.08333333333333333, '6': 0.0625, '12': 0.0625, '13': 0.0625, '26': 0.0625, '18': 0.041666666666666664, '7': 0.020833333333333332, '11': 0.020833333333333332, '17': 0.020833333333333332, '22': 0.020833333333333332, '23': 0.020833333333333332, '24': 0.020833333333333332}
- most_common_prob=0.396 (≈open-loop if ≳0.9)
- mean branch value Δu=0.00260 (Fixed-forced − adapt)
- frac histories with Δu>0: 0.250
- frac terminal discrete u changes: 0.062

## Table H — learned policies on held-out rollouts

| method | mean_gap | uniq | H_seq | frac≡Fixed mode | uniq a2 | P(mode a2) |
|--------|----------|------|-------|-----------------|---------|------------|

Fixed mode sequence: `None`

## Preliminary category (T=2 screen)

Category 2 — Adaptive room exists (planner beats Fixed / meaningful branching); if learned methods still lose, training/collapse is suspect.
