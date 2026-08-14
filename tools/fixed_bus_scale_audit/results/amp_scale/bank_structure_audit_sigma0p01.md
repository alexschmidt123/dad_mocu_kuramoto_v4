# Plan-2 bank structure audit

- **system:** `ieee5`
- **data_dir:** `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/tests/fixed_bus_scale_audit/data/amp_scale`
- **N_obs / sigma:** `200` / `0.01`
- **verdict:** `MYOPIC_TRAP_READY_FOR_DAD_RL`
- **Myopic beatable (trap):** `True`
- **Fixed beatable (planning−fixed):** `True`
- **Branching (distinct ξ₂≥2):** `True`
- **Adaptive room (Fixed-beatable ∧ branching):** `True`
- **DAD/RL ready:** `True` (MoE deferred: `True`)

## Myopic trap

- trap_present=`True`, strong_trap=`True`
- myopic_first=`5` {'bus': 0, 'amp': 0.2, 'duration': 0.15}
- planning_first=`4` {'bus': 0, 'amp': 0.15, 'duration': 0.15}
- fixed_pair=`[5, 4]`, ξ1 in fixed=`True`
- ξ1 mean |corr| with others=`1.0039214217325954`
- MYOPIC_TRAP: ξ1 is one-step best but a different first design is better for T≥2 (information overlay / option-value).

## U heterogeneity

- mean=0.1065, std=0.0646, Q95=0.2100, headroom=0.1035, U>0 frac=0.934, unique≈10

## Action redundancy (max-|ROCOF| fingerprints)

- n_actions=6 (amps=6, buses=1)
- near-dup frac=1.000 (thr |corr|≥0.98)
- mean |corr|=1.004, max |corr|=1.004
- amp_scale_redundant=True, same-bus near-dup frac=1.000

## T=2 adaptive screen (lower J better)

- J_myopic=0.261484
- J_planning=0.261016
- J_fixed≈0.266250
- planning−myopic=-0.000469
- planning−fixed=-0.005234
- distinct second actions=3, entropy=0.544

## Recommendations

- Probe amplitudes are scale-redundant (same-bus amp-normalized |corr|≈1): use a single probe_amplitude. Multi-amp does not create a Myopic trap — it only duplicates ξ.

## Next steps

- Train DAD and RL-sBOED only (exclude MoE) on this cell.
- Evaluate vs Fixed/Myopic/Random with paired system-level CIs.
- Only after DAD or RL beats Fixed and Myopic, enable MoE-sBOED.
