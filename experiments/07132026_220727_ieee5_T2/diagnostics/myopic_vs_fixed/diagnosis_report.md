# Myopic vs Fixed diagnosis (IEEE5 T=2)

Seed `3579`, paired rollouts `2000`.
Frozen rule hash `dc0dc35332b394b7`.
Fixed subset `[10, 11]`.
Myopic `n_hypothetical=16`.

## Verdicts

1. Statistically tied? **True** — myopic−fixed mean=0.0029, 95% CI [-0.0006, 0.0064]
2. Excessive Myopic MC error? **True** — Top-1 unstable and/or best–second gap comparable to score SE
3. Many quantized ties? **True** — step0 exact-tie rate=0.825, frac gap<1e-3=0.825
4. Fixed complementarity? **True** — joint_gain_vs_better_single=0.0187, fixed_vs_naive_top2=0.0078
5. Implementation inconsistency? **False** — none

## Paired comparison (myopic − fixed)

- mean myopic u_ctrl: `0.7601`
- mean fixed u_ctrl: `0.7572`
- mean paired diff: `0.0029`
- 95% bootstrap CI: `[-0.0006, 0.0064]`
- fraction tied: `0.694`
- fraction identical sequences: `0.000`

## Recommendation

Treat Myopic and Fixed as statistically tied at T=2 on this pilot (95% CI for paired difference contains 0). Increase myopic_hypothetical before the sweep; current MC error may blur Myopic decisions. Quantized u_ctrl ties are common; Myopic often breaks ties by action index, which can make it behave like a near-fixed rule. Fixed benefits from joint probe complementarity that one-step Myopic cannot plan; expect Fixed ≥ Myopic at small T when pairs matter. Do not start the full IEEE5/9/14 sweep until this diagnosis is accepted.
