# Objective observability report

**Gate: PASS**

## Summary

- prior terminal control: `1.0`
- unique final controls: `5`
- final u_ctrl mean / std: `0.887400` / `0.127363`
- final u_ctrl min / median / max: `0.600000` / `0.900000` / `1.000000`
- fraction changed / reduced / increased from prior: `0.5830` / `0.5830` / `0.0000`
- true-system safety rate: `1.0000`
- mean excess control: `0.574100`
- real Spearman (posterior mean U vs u_req): `0.3928`
- shuffled Spearman: `-0.0425`
- no-update check passed: `True`

## Gate checks

- `unique_final_u_ctrl_count`: PASS (value=5, threshold=3)
- `final_u_ctrl_std`: PASS (value=0.12736263188235394, threshold=0.01)
- `fraction_changed_from_prior`: PASS (value=0.583, threshold=0.05)
- `true_safety_rate`: PASS (value=1.0, threshold=1.0)
- `real_spearman`: PASS (value=0.39284937787096685, threshold=0.1)
- `real_better_than_shuffled`: PASS (value=0.39284937787096685, threshold=-0.04250647778309007)
- `no_update_check`: PASS (value=True, threshold=True)
