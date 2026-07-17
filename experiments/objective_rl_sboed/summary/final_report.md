# Objective RL-sBOED study — final report

Scientific methods: **DAD**, **RL-sBOED**, **Myopic**, **Fixed**, **Random**.

Controlled contrast: terminal-only DAD reward vs dense stepwise RL-sBOED reward
(`r_t = u_{t-1} - u_t`, γ = 1), same R2-style policy backbone and PPO trainer.
Primary metric for every method: terminal snapped `u_ctrl(h_T)` (lower is better).

Study directory: `experiments/objective_rl_sboed/`  
Historical adaptive-value and stamped IEEE runs were **not** overwritten.

Layout (standard): `run_config.yaml`, `train/`, `model/`, `eval/`, `diagnostics/`,
`logs/`, `summary/`, plus study-root alias `sensitivity_audit/` →
`diagnostics/sensitivity_audit/`.

---

## 1. Sensitivity audit (Part II)

Train/validation systems only. Confirmation/test unused.

| System | Unique `u_ctrl` | Unique `u_raw` | std(`u_ctrl`) | std(`u_raw`) | Best−second gap | Mean \|`u_raw`−`u_ctrl`\| | Drivers |
|--------|----------------:|---------------:|--------------:|-------------:|----------------:|--------------------------:|---------|
| IEEE5 T=3 | 49 | 98 | 0.0183 | 0.0250 | 0.0000 | 0.0045 | C snap_up; E similar posteriors |
| IEEE9 T=3 | 44 | 93 | 0.0117 | 0.0277 | 0.0010 | 0.0421 | C snap_up |

Details: `sensitivity_audit/` (alias) and `diagnostics/sensitivity_audit/`.

**Conclusion:** Terminal `u_ctrl` has *some* sequence dependence (dozens of unique
values), but gaps are tiny. Continuous `u_raw` is more dispersed than snapped
`u_ctrl`, especially on IEEE9 → snap_up quantization contributes, alongside
similar posterior changes across probes.

---

## 2. Experiment-design modifications (Part III)

**None in this study version.**

Frozen terminal rules and the existing `(amplitude, bus)` design space were
retained so the DAD vs RL-sBOED contrast is not confounded by a new grid or
recalibrated safety rule. Optional later ablation: RL-sBOED-raw-reward
diagnostic (still evaluated by snapped terminal `u_ctrl`).

---

## 3. IEEE5 T=3 results

### Initialization (validation)

| Method | Selected init | Mean validation `u_ctrl` |
|--------|---------------|--------------------------:|
| DAD | **fixed** | 0.8178 |
| RL-sBOED | **fixed** | 0.8203 |

Fixed initialization improved validation `u_ctrl` vs random for both methods
(DAD 0.8178 vs 0.8209; RL-sBOED 0.8203 vs 0.8434).

### Confirmation (selected init; 5 seeds)

| Method | Mean | Std (across seeds) |
|--------|-----:|-------------------:|
| DAD (fixed init) | 0.8388 | 0.0151 |
| RL-sBOED (fixed init) | 0.8356 | 0.0135 |
| Fixed baseline | 0.8367 | — |
| Myopic baseline | 0.8711 | — |
| Random baseline | 0.8633 | — |

### Paired bootstrap (confirmation; ≥10,000 resamples)

| Comparison | Mean diff | 95% CI | Significant? |
|------------|----------:|--------|--------------|
| RL-sBOED − DAD | −0.0031 | [−0.0244, +0.0150] | No |
| RL-sBOED − Myopic | −0.0355 | [−0.0473, −0.0236] | Yes (RL better) |
| RL-sBOED − Fixed | −0.0011 | [−0.0130, +0.0108] | No |
| RL-sBOED − Random | −0.0277 | [−0.0395, −0.0158] | Yes (RL better) |
| DAD − Myopic | −0.0323 | [−0.0448, −0.0186] | Yes (DAD better) |
| DAD − Fixed | +0.0020 | [−0.0105, +0.0158] | No |

### Adaptivity (selected init)

| Method | Mean unique sequences | Mean dominant fraction |
|--------|----------------------:|-----------------------:|
| DAD | 1.4 | 0.95 |
| RL-sBOED | 1.0 | 1.00 |

Policies remain nearly non-adaptive (Fixed-like sequences dominate).

### Action regret vs `xi2*` (adaptive-value histories)

Mean regret ≈ −0.0008 for both methods; `xi2*` agreement ≈ 0.02.
Dense reward did **not** improve second-action targeting relative to DAD.

### Reward sparsity (RL-sBOED training)

Mean fraction of trajectories with all-zero intermediate rewards ≈ 0.28–0.42
(random/fixed seeds). Stepwise signal is sparse but not entirely zero.

**IEEE5 interpretation: CASE 2** — RL-sBOED ≈ DAD; both near Fixed; both beat
Myopic/Random. Consistent with low intrinsic adaptive value (Case B).

---

## 4. IEEE9 T=3 results

### Initialization (validation)

| Method | Selected init | Mean validation `u_ctrl` |
|--------|---------------|--------------------------:|
| DAD | **random** | 0.9680 |
| RL-sBOED | **random** | 0.9710 |

Fixed init did **not** win on validation (though confirmation means for fixed
init were slightly better — selection correctly ignored confirmation).

### Confirmation (selected init; 5 seeds)

| Method | Mean | Std (across seeds) |
|--------|-----:|-------------------:|
| DAD (random init) | 0.9700 | 0.0051 |
| RL-sBOED (random init) | 0.9733 | 0.0053 |
| Fixed baseline | 0.9638 | — |
| Myopic baseline | 0.9613 | — |
| Random baseline | 0.9838 | — |

### Paired bootstrap

| Comparison | Mean diff | 95% CI | Significant? |
|------------|----------:|--------|--------------|
| RL-sBOED − DAD | +0.0033 | [−0.0023, +0.0093] | No |
| RL-sBOED − Myopic | +0.0120 | [+0.0075, +0.0165] | Yes (RL worse) |
| RL-sBOED − Fixed | +0.0095 | [+0.0050, +0.0140] | Yes (RL worse) |
| RL-sBOED − Random | −0.0105 | [−0.0150, −0.0060] | Yes (RL better) |
| DAD − Myopic | +0.0088 | [+0.0038, +0.0128] | Yes (DAD worse) |
| DAD − Fixed | +0.0063 | [+0.0013, +0.0103] | Yes (DAD worse) |

### Adaptivity (selected init)

| Method | Mean unique sequences | Mean dominant fraction |
|--------|----------------------:|-----------------------:|
| DAD | 1.8 | 0.90 |
| RL-sBOED | 2.0 | 0.80 |

Slightly more sequence diversity than IEEE5, still weakly adaptive.

### Action regret

Mean regret ≈ −0.00015 (DAD) vs −0.00004 (RL-sBOED); `xi2*` agreement ≈ 0.015–0.019.
No meaningful regret advantage for RL-sBOED.

### Reward sparsity

Zero-intermediate-reward fraction ≈ 0.01–0.05 (much less sparse than IEEE5).

**IEEE9 interpretation: CASE 2 (bordering CASE 4)** — RL-sBOED ≈ DAD (CI includes 0);
neither beats Fixed/Myopic; both beat Random. Again consistent with Case B.

---

## 5. Answers to Part XIX questions

1. **Is terminal `u_ctrl` sufficiently sensitive to probe combinations?**  
   Partially. Many unique values exist, but std ≈ 0.01–0.02 and best−second gaps
   are ~0–0.001 — weak discrimination for learning.

2. **Is coarseness mainly physical similarity or snap_up?**  
   Both. Drivers: **C snap_up quantization** (especially IEEE9, where `u_raw` is
   more dispersed) and **E similar posterior changes** (IEEE5).

3. **Was any experiment-design modification necessary?**  
   No — none applied in this version.

4. **Does Fixed initialization improve DAD?**  
   Yes on IEEE5 (selected by validation). No on IEEE9 (random selected).

5. **Does DAD fine-tuning move beyond the Fixed sequence?**  
   Weakly on IEEE5 fixed-init (mean unique sequences 1.4; dominant fraction 0.95).
   Mostly stays near a Fixed-like plan.

6. **Does DAD become observation-dependent?**  
   Only mildly (dominant fraction 0.90–0.95). Not strong branching.

7. **Does RL-sBOED reduce terminal `u_ctrl` vs terminal-reward DAD?**  
   No significant improvement. IEEE5: −0.003 (CI includes 0). IEEE9: +0.003
   (CI includes 0).

8. **Does RL-sBOED reduce next-action regret?**  
   No material difference vs DAD on either system.

9. **Does RL-sBOED learn more observation-dependent branching?**  
   IEEE5: no (dominant fraction 1.0). IEEE9: slightly higher unique-sequence
   count (2.0 vs 1.8), still weak.

10. **Does better adaptive behavior reduce terminal `u_ctrl`?**  
    Adaptive gains are too small to matter; Fixed remains competitive or better.

11. **Can RL-sBOED beat Myopic?**  
    IEEE5: yes (significant). IEEE9: no (significantly worse under selected init).

12. **Can RL-sBOED beat Fixed?**  
    IEEE5: no (CI includes 0). IEEE9: no (significantly worse).

13. **Are any gains statistically significant under paired bootstrap?**  
    Yes for beating Random (both systems) and Myopic on IEEE5. Not for
    RL-sBOED vs DAD. Not for beating Fixed.

14. **Consistent with prior low intrinsic adaptive value?**  
    **Yes.** Matches Case B from `experiments/objective_adaptive_value/`:
    dense stepwise rewards do not unlock large terminal-control gains when
    intrinsic adaptive value is tiny.

---

## 6. Scientific takeaway

| Method | Role | Finding |
|--------|------|---------|
| **DAD** | Terminal-reward full-horizon adaptive policy | Near Fixed on IEEE5; below Fixed/Myopic on IEEE9 |
| **RL-sBOED** | Stepwise-reward full-horizon adaptive policy | ≈ DAD; no significant terminal advantage |
| **Myopic** | One-step objective adaptive optimization | Strong on IEEE9; weaker on IEEE5 |
| **Fixed** | Optimized nonadaptive plan | Strong baseline on both systems |
| **Random** | Random reference | Worst or near-worst |

**Overall case:** CASE 2 — dense objective-based rewards do not materially improve
full-horizon RL under the current low-adaptive-value IEEE5/IEEE9 T=3 landscape.

Final performance criterion remains: **smallest safe terminal `u_ctrl`**.

---

## 7. Artifact index

```
experiments/objective_rl_sboed/
  sensitivity_audit/          # alias of diagnostics/sensitivity_audit/
  diagnostics/sensitivity_audit/
  ieee5_T3/
    train/{dad,rl_sboed}_{random,fixed}_init/seed_*/
    eval/{comparison,paired_bootstrap,adaptivity,action_regret}.csv
    eval/{fixed,myopic,random}/rollouts.csv
    summary/selected_initialization.json
  ieee9_T3/                   # same structure
  summary/
    final_method_comparison.csv
    reward_diagnostics.csv
    final_report.md
  logs/
```

Entry points:

```bash
./run.sh -study objective_rl_sboed -system ieee5 -stage run-system
./run.sh -study objective_rl_sboed -system ieee9 -stage report
pytest tests/test_objective_rl_sboed.py
```
