# Plan-2 bank structure audit

- **system:** `ieee9`
- **data_dir:** `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/data/ieee9_probe_amplitude_sweep/amp_0p05/probe`
- **N_obs / sigma:** `5` / `0.01`
- **verdict:** `REDUNDANT_LITTLE_ADAPTIVE_ROOM`
- **Myopic beatable (trap):** `False`
- **Fixed beatable (planning−fixed):** `False`
- **Branching (distinct ξ₂≥2):** `False`
- **Adaptive room (Fixed-beatable ∧ branching):** `False` (need plan−Fixed ≤ −0.01, ξ₂≥2)
- **Monotone adaptive room (multi-T gap):** `False`
- **DAD/RL ready:** `False` (MoE deferred: `True`)

## Myopic trap

- trap_present=`False`, strong_trap=`False`
- myopic_first=`45` {'bus': 0, 'amp': 0.05, 'duration': 3.0}
- planning_first=`47` {'bus': 2, 'amp': 0.05, 'duration': 3.0}
- fixed_pair=`[48, 27]`, ξ1 in fixed=`False`
- ξ1 mean |corr| with others=`0.9659800877353522`
- NO_MYOPIC_TRAP: one-step greedy first matches (or is as good as) non-myopic first — Myopic is hard to beat; add complementary overlap structure (e.g. multi-duration waveforms), not just more amps.

## U heterogeneity

- mean=0.3748, std=0.0160, Q95=0.4000, headroom=0.0252, U>0 frac=1.000, unique≈8

## Action redundancy (max-|ROCOF| fingerprints)

- n_actions=54 (amps=1, buses=9)
- near-dup frac=0.429 (thr |corr|≥0.98)
- mean |corr|=0.963, max |corr|=1.002
- amp_scale_redundant=False, same-bus near-dup frac=0.378

## T=2 adaptive screen (lower J better)

- J_myopic=0.394557
- J_planning=0.394479
- J_fixed≈0.393750
- planning−myopic=-0.000078
- planning−fixed=0.000729
- distinct second actions=5, entropy=1.350
- mode ξ2 prob=0.3333333333333333, eff_n=3.512195121951219
- mean_branch_value=0.0009374999999999939

## Multi-T adaptive room (gap = J_plan − J_fixed)

- gaps=`{'2': 0.0, '3': 0.0008333333333333526, '4': 0.0012500000000000289}`
- monotone_ok=`False`
- monotone_failures=`['gap(2)=0.000000 (need ≤ -0.01)', 'gap(3)=0.000833 not ≤ gap(2)-0.005=-0.005000', 'gap(4)=0.001250 not ≤ gap(3)-0.005=-0.004167']`
- min_gap_improve=`0.005`

## Recommendations

- High cross-action redundancy: drop near-duplicate buses/amps and regenerate into a new dataset_dir.
- No Myopic trap yet: greedy ξ1 is also (near) optimal for T≥2. Need a one-step-best design that overlays other useful probes so the optimal T-set excludes ξ1. Try probe_durations=[short,mid,long] with one amp (short pulse = high-ROCOF bait; long pulse = complementary).
- Little non-myopic gap vs Myopic: enlarge overlay trap (durations) or use moderate noise with vector N_obs>=100 and T>=3.
- Adaptive planning does not beat Fixed on this screen: increase U heterogeneity (stronger contingency / lower nadir) or reduce open-loop sufficiency by making probes more informative but belief-dependent.
- Weak U headroom: tighten control.contingency magnitude and regenerate.
- Multi-T adaptive room is not monotone: enlarge nested duration complementarity so gap(T)=J_plan−J_fixed deepens at T=3,4 (history-contingent residual U-tail), then regenerate with --force.

## Next steps

- Solution-1 FAIL: retune YAML (contingency / probe_durations), regenerate FULL bank with --force (no θ/design filtering).
- Re-run --bank-structure-audit at every sweep σ until myopic_trap, adaptive_room, and monotone_adaptive_room pass.
