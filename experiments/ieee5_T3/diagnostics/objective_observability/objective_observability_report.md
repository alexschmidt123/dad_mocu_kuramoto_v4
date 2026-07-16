# Objective observability report

**Gate: PASS**

## Summary

- prior terminal control: `1.0`
- unique final controls: `5`
- final u_ctrl mean / std: `0.877200` / `0.125858`
- final u_ctrl min / median / max: `0.600000` / `0.900000` / `1.000000`
- fraction changed / reduced / increased from prior: `0.6430` / `0.6430` / `0.0000`
- true-system safety rate: `1.0000`
- mean excess control: `0.563900`
- real Spearman (posterior mean U vs u_req): `0.4545`
- shuffled Spearman: `0.0188`
- no-update check passed: `True`

## Gate checks

- `unique_final_u_ctrl_count`: PASS (value=5, threshold=3)
- `final_u_ctrl_std`: PASS (value=0.1258576974205392, threshold=0.01)
- `fraction_changed_from_prior`: PASS (value=0.643, threshold=0.05)
- `true_safety_rate`: PASS (value=1.0, threshold=1.0)
- `real_spearman`: PASS (value=0.4545019677782575, threshold=0.1)
- `real_better_than_shuffled`: PASS (value=0.4545019677782575, threshold=0.018766422616686658)
- `no_update_check`: PASS (value=True, threshold=True)
