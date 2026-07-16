# Objective observability report

**Gate: PASS**

## Summary

- prior terminal control: `1.0`
- unique final controls: `7`
- final u_ctrl mean / std: `0.881300` / `0.127536`
- final u_ctrl min / median / max: `0.450000` / `0.900000` / `1.000000`
- fraction changed / reduced / increased from prior: `0.5790` / `0.5790` / `0.0000`
- true-system safety rate: `1.0000`
- mean excess control: `0.568000`
- real Spearman (posterior mean U vs u_req): `0.3455`
- shuffled Spearman: `-0.0055`
- no-update check passed: `True`

## Gate checks

- `unique_final_u_ctrl_count`: PASS (value=7, threshold=3)
- `final_u_ctrl_std`: PASS (value=0.1275355244627943, threshold=0.01)
- `fraction_changed_from_prior`: PASS (value=0.579, threshold=0.05)
- `true_safety_rate`: PASS (value=1.0, threshold=1.0)
- `real_spearman`: PASS (value=0.34552797269327057, threshold=0.1)
- `real_better_than_shuffled`: PASS (value=0.34552797269327057, threshold=-0.0054863277037167394)
- `no_update_check`: PASS (value=True, threshold=True)
