# Plan-2 bank structure audit

- **system:** `ieee5`
- **data_dir:** `/home/grads/g/g.lin/Documents/dad_mocu_kuramoto_v4/tests/fixed_bus_scale_audit/data/amp_scale`
- **N_obs / sigma:** `200` / `0.001`
- **verdict:** `AMP_SCALE_REDUNDANT`
- **Myopic beatable (trap):** `True`
- **Fixed beatable (planning−fixed):** `False`
- **Branching (distinct ξ₂≥2):** `True`
- **Adaptive room (Fixed-beatable ∧ branching):** `False`
- **DAD/RL ready:** `False` (MoE deferred: `True`)

## Myopic trap

- trap_present=`True`, strong_trap=`True`
- myopic_first=`4` {'bus': 0, 'amp': 0.15, 'duration': 0.15}
- planning_first=`5` {'bus': 0, 'amp': 0.2, 'duration': 0.15}
- fixed_pair=`[4, 5]`, ξ1 in fixed=`True`
- ξ1 mean |corr| with others=`1.0039215179302594`
- MYOPIC_TRAP: ξ1 is one-step best but a different first design is better for T≥2 (information overlay / option-value).

## U heterogeneity

- mean=0.1065, std=0.0646, Q95=0.2100, headroom=0.1035, U>0 frac=0.934, unique≈10

## Action redundancy (max-|ROCOF| fingerprints)

- n_actions=6 (amps=6, buses=1)
- near-dup frac=1.000 (thr |corr|≥0.98)
- mean |corr|=1.004, max |corr|=1.004
- amp_scale_redundant=True, same-bus near-dup frac=1.000

## T=2 adaptive screen (lower J better)

- J_myopic=0.156250
- J_planning=0.155547
- J_fixed≈0.155000
- planning−myopic=-0.000703
- planning−fixed=0.000547
- distinct second actions=2, entropy=0.173

## Recommendations

- Probe amplitudes are scale-redundant (same-bus amp-normalized |corr|≈1): use a single probe_amplitude. Multi-amp does not create a Myopic trap — it only duplicates ξ.
- Adaptive planning does not beat Fixed on this screen: increase U heterogeneity (stronger contingency / lower nadir) or reduce open-loop sufficiency by making probes more informative but belief-dependent.

## Next steps

- Train DAD and RL-sBOED only (exclude MoE) on this cell.
- Evaluate vs Fixed/Myopic/Random with paired system-level CIs.
- Only after DAD or RL beats Fixed and Myopic, enable MoE-sBOED.
