# Plan-2 bank structure audit

- **system:** `ieee5`
- **data_dir:** `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/tests/fixed_bus_scale_audit/data/duration_scale`
- **N_obs / sigma:** `200` / `0.01`
- **verdict:** `PARTIAL_ADAPTIVE_ROOM`
- **Myopic beatable (trap):** `False`
- **Fixed beatable (planning−fixed):** `True`
- **Branching (distinct ξ₂≥2):** `True`
- **Adaptive room (Fixed-beatable ∧ branching):** `True`
- **DAD/RL ready:** `True` (MoE deferred: `True`)

## Myopic trap

- trap_present=`False`, strong_trap=`False`
- myopic_first=`5` {'bus': 0, 'amp': 0.15, 'duration': 0.2}
- planning_first=`5` {'bus': 0, 'amp': 0.15, 'duration': 0.2}
- fixed_pair=`[5, 3]`, ξ1 in fixed=`True`
- ξ1 mean |corr| with others=`0.9714194926033345`
- NO_MYOPIC_TRAP: one-step greedy first matches (or is as good as) non-myopic first — Myopic is hard to beat; add complementary overlap structure (e.g. multi-duration waveforms), not just more amps.

## U heterogeneity

- mean=0.1065, std=0.0646, Q95=0.2100, headroom=0.1035, U>0 frac=0.934, unique≈10

## Action redundancy (max-|ROCOF| fingerprints)

- n_actions=6 (amps=1, buses=1)
- near-dup frac=0.600 (thr |corr|≥0.98)
- mean |corr|=0.985, max |corr|=1.002
- amp_scale_redundant=False, same-bus near-dup frac=0.600

## T=2 adaptive screen (lower J better)

- J_myopic=0.262422
- J_planning=0.262422
- J_fixed≈0.266250
- planning−myopic=0.000000
- planning−fixed=-0.003828
- distinct second actions=4, entropy=1.164

## Recommendations

- High cross-action redundancy: drop near-duplicate buses/amps and regenerate into a new dataset_dir.
- No Myopic trap yet: greedy ξ1 is also (near) optimal for T≥2. Need a one-step-best design that overlays other useful probes so the optimal T-set excludes ξ1. Try probe_durations=[short,mid,long] with one amp (short pulse = high-ROCOF bait; long pulse = complementary).
- Same-bus designs remain near-duplicates: keep one amplitude per bus.
- Little non-myopic gap vs Myopic: enlarge overlay trap (durations) or use moderate noise with vector N_obs>=100 and T>=3.

## Next steps

- No Myopic trap: regenerate with multi-duration designs (probe_durations) so one-step-best ξ1 overlays other probes and is excluded from the optimal T-set.
- Re-run bank-structure-audit until myopic_trap.trap_present=true.
- Keep MoE-sBOED disabled until DAD/RL win Fixed/Myopic.
