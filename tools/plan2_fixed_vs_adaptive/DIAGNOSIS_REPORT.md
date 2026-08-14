# Plan-2 Fixed vs adaptive diagnosis (IEEE-5 Solution-1 bank v2)

**Date:** 2026-08-11  
**Config:** `configs/ieee5_plan2_trap.yaml`  
**Bank (frozen):** `data/ieee5_plan2_trap_v2`  
**Catalog:** amp `{0.15}` × durations `{0.03,0.06,0.10,0.15,0.22,0.30}` × 5 buses → **N_ξ=30**  
**Note:** ChatGPT brief listed `D={0.05…0.20}`; diagnosis uses the **frozen YAML catalog**, not that older set.

**Scripts:**
```bash
python3 tools/plan2_fixed_vs_adaptive/run_t2_fixed_vs_adaptive.py \
  --noise_sigma 0.01 --seed 101 \
  --exp-dir experiments/08112026_145803_ieee5_plan2_trap_Uctrl_T2_Nobs200_sigma0p01

python3 tools/plan2_fixed_vs_adaptive/run_t2_fixed_vs_adaptive.py \
  --noise_sigma 0.005 --seed 101 \
  --exp-dir experiments/08112026_151142_ieee5_plan2_trap_Uctrl_T2_Nobs200_sigma0p005
```
Outputs: `tools/plan2_fixed_vs_adaptive/results/t2_diag_sigma{0p01,0p005}.{json,md}`

---

## §10 Implementation map

| # | Item | Location |
|---|------|----------|
| 1 | Plan-2 design catalog | `src/swing_equation_ode/design.py` → `build_catalog`; YAML `probe_amplitudes` / `probe_durations` |
| 2 | Probe-response bank Y_{n,a} | `src/observation/builder.py` `build_centres_bank`; bank load `src/control/full_delta_f/banks.py` (`delta_f.npy`) |
| 3 | Control-requirement bank U_n | `src/control/banks.py` + `data/.../train/U.npy` |
| 4 | Particle posterior update | `src/observation/delta_f.py` `vector_gaussian_loglik`; weight update in `full_delta_f/context.py` / `bank_structure_audit._update` |
| 5 | Posterior-safe `u_ctrl` | `src/control/posterior_ctrl.py` `posterior_safe_u_ctrl` |
| 6 | `u_ctrl_opt` | `src/control/oracle_u_ctrl.py` |
| 7 | MOCU evaluation | `src/control/full_delta_f/evaluate.py` (gap = u_ctrl − u_ctrl_opt) |
| 8 | Myopic candidate scoring | `src/control/full_delta_f/diagnostics.py` / adaptive one-step scores in context + `bank_structure_audit._one_step_scores` |
| 9 | Exhaustive Fixed subset search | `src/control/full_delta_f/context.py` `_exhaustive_fixed_sequence`, `_score_fixed_subset`, `_resolve_fixed_sequence` |
| 10 | DAD training cost | `src/control/rewards.py` + `full_delta_f/train.py` (terminal `u_ctrl`) |
| 11 | RL-sBOED dense reward | `src/control/rewards.py` (Δu_ctrl) |
| 12 | MoE PPO reward | same reward path; policy `src/neural/rl_policy.py` |
| 13 | MoE branching regularizer | `src/neural/rl_policy.py` (+ train `lambda_branch`) |
| 14 | MoE counterfactual fingerprints | `rl_policy.counterfactual_loss` |
| 15 | MoE checkpoint diversity | `full_delta_f/train.py` `checkpoint_score` (MOCU − w·unique_frac) |
| 16 | Action masking / no-repeat | `design.masked_action_indices` |
| 17 | Deterministic evaluation | eval path `eval_mode=deterministic` in summary |
| 18 | Sequence-diversity diagnostics | `train.sequence_diversity_stats`; `experiment.diagnose_conditional_action_diversity` |

---

## §11 Diagnostic A — Fixed optimization

| T | σ | search_mode | n_eval | = C(30,T)? | subset | offline mean u_ctrl |
|---|---|-------------|--------|------------|--------|---------------------|
| 2 | 0.01 | exhaustive_offline | **435** | yes | [26,28] | 0.50625 |
| 2 | 0.005 | exhaustive_offline | **435** | yes | [25,26] | 0.47797 |
| 3 | 0.01 | exhaustive_offline | **4060** | yes | [25,26,28] | 0.48922 |
| 3 | 0.005 | exhaustive_offline | **4060** | yes | [25,26,28] | 0.46805 |
| 4 | 0.01 | exhaustive_offline | **27405** | yes | [25,26,27,28] | 0.47453 |
| 4 | 0.005 | exhaustive_offline | **27405** | yes | [18,22,25,26] | 0.46641 |
| 5 | 0.01 | greedy_multirestart_offline | 9 | no (C(30,5)=142506) | [26,27,25,28,24] | 0.46969 |
| 5 | 0.005 | greedy_multirestart_offline | 9 | no | [3,26,25,27,28] | 0.46641 |

Fixed scoring uses posterior-predictive noisy Δf on particles (not true θ★), objective = expected terminal posterior-safe `u_ctrl`, order-invariant unordered subsets. Artifacts: `experiments/.../model/fixed_subset_T{T}.json`.

Near-best Fixed pairs (MC screen, σ=0.01): (25,26), (26,25), (26,28), … within ~0.004 — **no uniquely dominant pair**; many almost-equivalent complementary long-duration pairs on buses 0–3.

Actions: 25=(0.15,bus0,0.30), 26=(0.15,bus1,0.30), 28=(0.15,bus3,0.30).

---

## Table A — primary T=2 held-out performance (v2 sweep, seed 101)

### σ = 0.01 — `08112026_145803_..._T2_..._sigma0p01`

| method | mean u_ctrl | MOCU | safety | valid | uniq seq | H_seq |
|--------|-------------|------|--------|-------|----------|-------|
| Myopic | 0.5128 | 0.1225 | 1.00 | yes | 42 | 3.61 |
| Fixed | 0.5109 | 0.1206 | 1.00 | yes | 1 | 0 |
| T=2 planner (screen J) | ≈0.4779 | — | — | screen only | — | — |
| DAD | **0.5048** | **0.1145** | 1.00 | yes | **1** | 0 |
| RL-sBOED | 0.5081 | 0.1178 | 1.00 | yes | 2 | 0.69 |
| MoE-sBOED | 0.5105 | 0.1201 | 1.00 | yes | 5 | 1.02 |
| Random | 0.5714 | 0.1810 | 0.999 | yes | 796 | 6.54 |

**Fixed does NOT beat DAD/RL/MoE at T=2 σ=0.01.** DAD is #1 with a single open-loop sequence.

### σ = 0.005 — `08112026_151142_..._T2_..._sigma0p005`

| method | mean u_ctrl | MOCU | safety | valid | uniq seq | H_seq |
|--------|-------------|------|--------|-------|----------|-------|
| Myopic | 0.4992 | 0.1089 | 1.00 | yes | 40 | 3.53 |
| Fixed | **0.4814** | **0.0911** | 1.00 | yes | 1 | 0 |
| T=2 planner (screen J) | ≈0.4471 | — | — | screen only | — | — |
| DAD | 0.4875 | 0.0972 | 1.00 | yes | 3 | 0.35 |
| RL-sBOED | 0.4880 | 0.0976 | 1.00 | yes | 6 | 1.20 |
| MoE-sBOED | 0.4870 | 0.0967 | 1.00 | yes | 5 | 1.37 |
| Random | 0.5440 | 0.1537 | 0.988 | yes | 796 | 6.54 |

Here Fixed #1; learners within ~0.006 MOCU of Fixed; Myopic trap remains.

---

## Table B — T=2 adaptivity structure (MC planner screen)

| σ | plan ξ1 | P(mode ξ2) | H(ξ2) | eff_n | plan−Fixed | mean Δu_branch | frac Δu>0 | frac discrete u change |
|---|--------|------------|-------|-------|------------|----------------|-----------|------------------------|
| 0.01 | 28 (bus3,0.30) | 0.625 | 1.35 | 2.38 | **−0.0034** | 0.0027 | 0.31 | 0.21 |
| 0.005 | 25 (bus0,0.30) | 0.125 | 2.77 | 13.9 | **−0.0035** | 0.0041 | 0.52 | 0.12 |

**Interpretation:** planner edges Fixed by only ~0.003–0.004 in expected terminal `u_ctrl` (below the structure-audit gate of −0.01). High ξ2 entropy at σ=0.005 with tiny branch value is consistent with a **flat second-action landscape** (many near-ties), not high-value history branching.

---

## Table C / H — learned vs Fixed / collapse

| σ | method | seq | ≡Fixed seq | ≡Fixed set | verdict |
|---|--------|-----|------------|------------|---------|
| 0.01 | DAD | always `22 26` | 0% | 0% | **open-loop collapse to a non-Fixed near-optimal pair** (and wins) |
| 0.01 | RL | `26→{27,25}` | 0% | 0% | Fixed ξ1, wrong ξ2 |
| 0.01 | MoE | mostly `26 28` | 62% | 62% | **Fixed-in-disguise** majority |
| 0.005 | DAD | mostly `27 26` | 0% | 0% | open-loop near Fixed buses; loses to Fixed |
| 0.005 | RL | `26→{28,27,29}` | 0% | 0% | adaptive-looking but not Fixed-optimal |
| 0.005 | MoE | `26→{25,21,…}` | 0% | 42% set | partial Fixed set; still loses |

Myopic shows high uniq / entropy but worse MOCU → **Myopic trap = complementarity without good one-step credit**.

---

## Category (T=2 structural test)

### **Category 1 — Fixed wins (when it does) because adaptive room is intrinsically weak**

Evidence:
1. Myopic trap exists (Fixed ≻ Myopic) → open-loop complementarity.
2. T=2 belief-state planner ≃ Fixed (ΔJ ≈ −0.003 only).
3. Branch value ≪ 0.01; discrete `u_ctrl` changes on only ~12–21% of histories.
4. Many near-tied Fixed pairs (long-duration multi-bus complements).
5. At σ=0.01 DAD collapses to one non-Fixed sequence and still beats Fixed → landscape has several near-equivalent open-loop optima, not large history-contingent value.

**Not Category 2 as the primary story at T=2:** there is no clear planner≫Fixed gap for training to exploit.

**Caveat across T:** MoE already #1 at T3 (both σ) and T4 σ=0.01. “Fixed always beats learners” is **false** on the frozen v2 sweep; it is cell-dependent. T=2 σ=0.005 is the clearest Fixed-win cell matching the ChatGPT premise.

---

## Answers Q1–12

1. **Why Fixed beats DAD?** Only clearly at T=2 σ=0.005 (and some higher-T cells). DAD collapses to a near-open-loop sequence that is **near but not** the exhaustive Fixed optimum; gap ≈0.006 MOCU. Not because Fixed cheats — Fixed is exhaustive C(30,2)=435. At T=2 σ=0.01 DAD **beats** Fixed with a different collapsed pair (`22 26`).

2. **Why Fixed beats RL-sBOED?** Same structural reason + RL picks Fixed-like ξ1 but suboptimal ξ2 (credit/representation), without enough history-contingent value to justify branching.

3. **Why Fixed beats MoE-sBOED?** At T=2 σ=0.005 MoE is #2 (closest learner) but still ~0.006 behind. At T=2 σ=0.01 MoE is 62% Fixed-in-disguise and essentially ties Fixed. MoE routing/diversity is **not** the primary T=2 failure mode vs Fixed.

4. **Genuinely adaptive or only non-myopic?** **Primarily non-myopic / open-loop complementary.** History-dependent branching exists but with **small operational value**.

5. **Does first Δf change optimal ξ2?** Sometimes (P(mode) 0.62 at σ=0.01; flatter at 0.005), but mass concentrates on long-duration bus-1/0 probes.

6. **Does branching materially reduce terminal u_ctrl?** **No** — mean Δu_branch ≈ 0.003–0.004.

7. **Does it reduce held-out MOCU?** Planner edge too small to expect large held-out wins; MoE wins appear at **T≥3**, not from large T=2 branch value.

8. **Are learned policies using observation history?** Weakly: DAD often ignores history (uniq=1). RL/MoE show limited ξ2 variation; MoE sometimes copies Fixed.

9. **Collapsing to Fixed-in-disguise?** MoE at σ=0.01: **yes (62%)**. DAD: collapse to **other** Fixed-like open-loop pairs, not the optimized Fixed sequence.

10. **Is Fixed exploiting a universally good complementary subset?** **Yes** — long-duration probes across key buses; many near-equivalent pairs.

11. **Change benchmark or policy training?** **Benchmark structure first** if the publication goal requires large adaptive ≻ Fixed gaps at T=2. Training tweaks cannot invent branch value that is ≈0.003. Keep current training for T≥3 cells where MoE already wins.

12. **SINGLE highest-priority next change:** **Do not retune MoE capacity / λ yet.** Next scientific step: **strengthen history-contingent control-tail branching in the problem** (still under Plan-2 physics rules: e.g. larger control-relevant ambiguity near the (1−α) U-tail across early probes) **or** reframe the paper claim around **non-myopic complementarity + MoE wins at T=3–4**, not “T=2 adaptive ≫ Fixed”. Only after a planner−Fixed ≲ −0.01 **with** mean branch value ≳ 0.01 should training/collapse become the priority.

---

## Reproducibility artifacts

| Artifact | Path |
|----------|------|
| Diagnostic script | `tools/plan2_fixed_vs_adaptive/run_t2_fixed_vs_adaptive.py` |
| Results | `tools/plan2_fixed_vs_adaptive/results/` |
| T2 σ0.01 eval | `experiments/08112026_145803_ieee5_plan2_trap_Uctrl_T2_Nobs200_sigma0p01` |
| T2 σ0.005 eval | `experiments/08112026_151142_ieee5_plan2_trap_Uctrl_T2_Nobs200_sigma0p005` |
| Frozen bank | `data/ieee5_plan2_trap_v2` |
