# Bus-location and joint bus–amplitude adaptive-value — final report

## Context

Prior amplitude study: IEEE5/IEEE9 = **Case B** (nominal amplitude branching,
near-zero practical amplitude regret). No amplitude-grid expansion.
No DAD/RL-sBOED retraining in this diagnostic.

Continuous `u_cont` is a **diagnostic** objective (approximation_based_on_discrete_U_bank); snapped `u_ctrl` retained for comparison. Physically validated continuous intermediates: **False**.

## Per-system bus results

### ieee5

- **BUS-B**: one-step bus gaps/branching exist, but four-way terminal decomposition does not significantly beat Fully Fixed (practical terminal bus adaptive value still low; also BUS-E for policy training)
- Unique b*: 4, dominant fraction=0.745
- Mean wrong-bus regret (cont/snap): 0.0191064 / 0.0160611
- Mean best−second bus gap (cont/snap): 0.0219141 / 0.0161563
- Prior wrong-amp regret: 0.000578125

  - fully_fixed: u_cont=0.848828
  - fixed_bus_adaptive_amp: u_cont=0.845313
  - adaptive_bus_fixed_amp: u_cont=0.851172
  - adaptive_bus_adaptive_amp: u_cont=0.852734

### ieee9

- **BUS-B**: one-step bus gaps/branching exist, but four-way terminal decomposition does not significantly beat Fully Fixed (practical terminal bus adaptive value still low; also BUS-E for policy training)
- Unique b*: 5, dominant fraction=0.685
- Mean wrong-bus regret (cont/snap): 0.00677996 / 0.00426897
- Mean best−second bus gap (cont/snap): 0.00263125 / 0.00233125
- Prior wrong-amp regret: 0.00026875

  - fully_fixed: u_cont=0.919375
  - fixed_bus_adaptive_amp: u_cont=0.919375
  - adaptive_bus_fixed_amp: u_cont=0.918125
  - adaptive_bus_adaptive_amp: u_cont=0.918125

## Answers to Part XIX

1. **Different buses preferred on IEEE5?**  
   Yes nominally — unique b*=4, non-dominant fraction=0.255.

2. **Different buses preferred on IEEE9?**  
   Unique b*=5, non-dominant fraction=0.315.

3. **Systematic or near-tied?**  
   Interpret via case labels and median/mean wrong-bus regret (median≈0 with tiny mean ⇒ near-tied / Case BUS-B).

4. **Wrong-bus regret?**  
   IEEE5: mean=0.0191064, median=0, p95=0.100781, max=0.167187.
   IEEE9: mean=0.00677996, median=0, p95=0.0375, max=0.425.

5. **Wrong-bus vs prior wrong-amplitude regret?**  
   IEEE5: bus=0.0191064 vs amp=0.000578125.
   IEEE9: bus=0.00677996 vs amp=0.00026875.

6. **Does bus contain more adaptive value than amplitude?**  
   Compare the regrets above and four-way decomposition (Adaptive Bus + Fixed Amp vs Fixed Bus + Adaptive Amp).

7. **Did continuous resolution reveal bus value hidden by snap_up?**  
   Only if Case BUS-D / partial-D note; otherwise continuous still low ⇒ BUS-E.

8–10. **Four-way comparisons** — see per-system paired bootstrap in `results/joint_decomposition.csv` and system reports.

11. **Cause of low adaptive value?**  
   **Both dimensions + overall experiment structure** under the current 6×bus design: prior amplitude Case B; bus BUS-B/BUS-B with four-way terminal structures ≈ Fully Fixed.

12. **Meaningful adaptive value for DAD/RL-sBOED?**  
   **Not yet for policy training.** One-step bus gaps may exist, but Fully Fixed ≈ adaptive references on terminal u_ctrl. **Do not continue generic RL tuning.** Next: modify experimental design (probes / horizon / systems), not retrain DAD/RL-sBOED yet.

## Decision rule outcome

Bus Case A/B and amplitude Case B, with four-way ≈ Fixed → current probe design space has **low intrinsic terminal adaptive value**. **Do not retrain DAD/RL-sBOED yet.**

- IEEE5 bus adaptivity: **nominal only** (BUS-B)
- IEEE9 bus adaptivity: **nominal only** (BUS-B)
