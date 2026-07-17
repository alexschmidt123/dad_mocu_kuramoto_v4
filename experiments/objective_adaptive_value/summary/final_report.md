# Objective-Based Adaptive-Value Test

Diagnostic only. Main manuscript methods remain:

- DAD
- Myopic
- Fixed
- Random

Question: after the first probe, does the optimal second probe change across histories in a way that meaningfully reduces expected final `u_ctrl`?

All calculations use offline Y/U banks and particle posteriors on the **validation diagnostic split**. No ODE simulation. No EIG/MSE/accuracy metrics.

## Setup

| System | Histories / ξ₁ | n_hyp outer/inner | Valid ξ₁ |
|--------|----------------|-------------------|----------|
| IEEE5 T=3 | 256 | 96/48 | 30 |
| IEEE9 T=3 | 100 | 48/24 | 54 |

`Δ_adaptive = J_common − J_adaptive` uses split-sample selection/evaluation and 10,000 paired bootstrap resamples. Case labels follow the protocol (A–D).

## System summaries

| Quantity | IEEE5 | IEEE9 |
|----------|-------|-------|
| Case | **B** | **B** |
| Best ξ₁ (lowest J_adaptive) | 20 | 0 |
| J_adaptive (that ξ₁) | 0.834530 | 0.965493 |
| J_common (that ξ₁) | 0.834539 | 0.965114 |
| Δ_adaptive | 9.74867e-06 | -0.000378516 |
| 95% CI | [-0.001427, 0.001451] | [-0.000929, 0.000147] |
| Unique ξ₂* (that ξ₁) | 25 | 30 |
| Dominant ξ₂* fraction | 0.082 | 0.640 |
| Near-tie rate | 0.062 | 0.725 |
| Mean Δ over all ξ₁ | -0.000621188 | -0.000785739 |
| Max Δ over all ξ₁ | 0.000253296 | 0.000436806 |
| Fraction ξ₁ with CI>0 | 0% | 0% |
| Mean # unique ξ₂* | 26.7 | 25.9 |
| Mean dominant fraction | 0.123 | 0.370 |
| Median gap (best−2nd) | 0.00164931 | 6.94444e-05 |
| DAD agreement vs ξ₂* | 0.037 | 0.031 |
| DAD mean regret | 0.000795 | 0.000106 |

Plots for each system are under `ieee{5,9}_T3/plots/`.

## Answers to the scientific questions

1. **Does the best second probe change across first-step histories in IEEE5?**  
   Nominally yes: ~27 unique ξ₂* on average per ξ₁, dominant fraction only ~12%. But best−second gaps are tiny (median ≈ 0.0016), so the argmin is largely Monte Carlo noise among near-ties.

2. **Does it change in IEEE9?**  
   Same pattern: ~26 unique ξ₂* on average, but with higher near-tie rates (~51%) and even smaller median gaps (≈ 0.00007).

3. **Are those changes systematic or mostly ties/noise?**  
   Mostly **ties/noise for the objective**. y₁-bin modal ξ₂ can differ, but objective gaps are negligible and **no ξ₁ has a bootstrap CI for Δ_adaptive excluding 0**. Changing ξ₂* labels without material J differences is not adaptive value.

4. **Does history-dependent second-probe selection significantly reduce expected final u_ctrl?**  
   **No** for either system under this protocol. Mean Δ_adaptive is slightly **negative** (IEEE5 -0.000621; IEEE9 -0.000786), consistent with residual selection noise after split-sample correction, not a real gain.

5. **What is Δ_adaptive for IEEE5?**  
   At best-J ξ₁=20: **9.74867e-06**, 95% CI **[-0.001427, 0.001451]** (includes 0). Across all ξ₁: mean -0.000621188, max 0.000253296, **0/30** significant.

6. **What is Δ_adaptive for IEEE9?**  
   At best-J ξ₁=0: **-0.000378516**, 95% CI **[-0.000929, 0.000147]** (includes 0). Across all ξ₁: mean -0.000785739, max 0.000436806, **0/54** significant.

7. **Is Fixed strong because one probe sequence is good for most histories?**  
   **Yes.** J_common ≈ J_adaptive for every ξ₁, so a single non-adaptive second probe matches history-adaptive selection in expected `u_ctrl`. Fixed is naturally competitive; DAD should mainly match Fixed rather than strongly beat it on these T=3 problems.

8. **Does DAD choose the objective-optimal second probe?**  
   Rarely vs the noisy ξ₂* label: agreement **3.7%** (IEEE5) and **3.1%** (IEEE9). That disagreement is expected when many actions are near-tied.

9. **What is DAD's average next-action regret?**  
   IEEE5 **0.000795**; IEEE9 **0.000106** (raw `J(h,ξ₂_DAD)−min J`; MC noise can make individual terms slightly negative). Absolute regret is tiny relative to baseline `u_ctrl` levels (~0.83 / ~0.96).

10. **Is the main limitation low intrinsic adaptive value or imperfect DAD training?**  
    **Low intrinsic objective-based adaptive value (Case B)** on both IEEE5 and IEEE9 T=3. ξ₂* labels fluctuate, but Δ_adaptive is indistinguishable from zero. The diagnostic does **not** show a large missed adaptive opportunity that DAD training failed to exploit. Improving DAD is secondary to this structural finding for these systems/horizon.

## Outputs

```
experiments/objective_adaptive_value/
  ieee5_T3/   first_history_results.csv, xi2_distribution.csv,
              adaptive_gain.csv, dad_action_regret.csv, plots/
  ieee9_T3/   (same)
  summary/    system_comparison.csv, final_report.md
```
