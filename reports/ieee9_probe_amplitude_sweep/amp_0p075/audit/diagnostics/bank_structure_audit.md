# Plan-2 bank structure audit

- **system:** `ieee9`
- **data_dir:** `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/data/ieee9_probe_amplitude_sweep/amp_0p075/probe`
- **N_obs / sigma:** `5` / `0.01`
- **verdict:** `MYOPIC_TRAP_READY_FOR_DAD_RL`
- **Myopic beatable (trap):** `True`
- **Fixed beatable (planning−fixed):** `False`
- **Branching (distinct ξ₂≥2):** `False`
- **Adaptive room (Fixed-beatable ∧ branching):** `False` (need plan−Fixed ≤ −0.01, ξ₂≥2)
- **Monotone adaptive room (multi-T gap):** `False`
- **DAD/RL ready:** `True` (MoE deferred: `True`)

## Myopic trap

- trap_present=`True`, strong_trap=`True`
- myopic_first=`36` {'bus': 0, 'amp': 0.075, 'duration': 2.5}
- planning_first=`48` {'bus': 3, 'amp': 0.075, 'duration': 3.0}
- fixed_pair=`[48, 45]`, ξ1 in fixed=`False`
- ξ1 mean |corr| with others=`0.9725915351568547`
- MYOPIC_TRAP: ξ1 is one-step best but a different first design is better for T≥2 (information overlay / option-value).

## U heterogeneity

- mean=0.3748, std=0.0160, Q95=0.4000, headroom=0.0252, U>0 frac=1.000, unique≈8

## Action redundancy (max-|ROCOF| fingerprints)

- n_actions=54 (amps=1, buses=9)
- near-dup frac=0.429 (thr |corr|≥0.98)
- mean |corr|=0.963, max |corr|=1.002
- amp_scale_redundant=False, same-bus near-dup frac=0.378

## T=2 adaptive screen (lower J better)

- J_myopic=0.392109
- J_planning=0.391693
- J_fixed≈0.390833
- planning−myopic=-0.000417
- planning−fixed=0.000859
- distinct second actions=8, entropy=1.702
- mode ξ2 prob=0.3333333333333333, eff_n=4.430769230769231
- mean_branch_value=0.0001562500000000013

## Multi-T adaptive room (gap = J_plan − J_fixed)

- gaps=`{'2': 0.0004166666666667318, '3': -0.0012499999999999734, '4': 0.0004166666666666763}`
- monotone_ok=`False`
- monotone_failures=`['gap(2)=0.000417 (need ≤ -0.01)', 'gap(3)=-0.001250 not ≤ gap(2)-0.005=-0.004583', 'gap(4)=0.000417 not ≤ gap(3)-0.005=-0.006250']`
- min_gap_improve=`0.005`

## Recommendations

- High cross-action redundancy: drop near-duplicate buses/amps and regenerate into a new dataset_dir.
- Adaptive planning does not beat Fixed on this screen: increase U heterogeneity (stronger contingency / lower nadir) or reduce open-loop sufficiency by making probes more informative but belief-dependent.
- Weak U headroom: tighten control.contingency magnitude and regenerate.
- Multi-T adaptive room is not monotone: enlarge nested duration complementarity so gap(T)=J_plan−J_fixed deepens at T=3,4 (history-contingent residual U-tail), then regenerate with --force.

## Next steps

- Solution-1 FAIL: retune YAML (contingency / probe_durations), regenerate FULL bank with --force (no θ/design filtering).
- Re-run --bank-structure-audit at every sweep σ until myopic_trap, adaptive_room, and monotone_adaptive_room pass.
