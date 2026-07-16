# Objective observability report

**Gate: PASS**

## Summary

- prior terminal control: `1.6`
- unique final controls: `3`
- final u_ctrl mean / std: `1.555520` / `0.064936`
- final u_ctrl min / median / max: `1.440000` / `1.600000` / `1.600000`
- fraction changed / reduced / increased from prior: `0.3500` / `0.3500` / `0.0000`
- true-system safety rate: `1.0000`
- mean excess control: `1.480080`
- real Spearman (posterior mean U vs u_req): `0.1361`
- shuffled Spearman: `-0.0137`
- no-update check passed: `True`

## Gate checks

- `unique_final_u_ctrl_count`: PASS (value=3, threshold=3)
- `final_u_ctrl_std`: PASS (value=0.0649363503748094, threshold=0.01)
- `fraction_changed_from_prior`: PASS (value=0.35, threshold=0.05)
- `true_safety_rate`: PASS (value=1.0, threshold=1.0)
- `real_spearman`: PASS (value=0.136093901296358, threshold=0.0)
- `real_better_than_shuffled`: PASS (value=0.136093901296358, threshold=-0.013653610369231583)
- `no_update_check`: PASS (value=True, threshold=True)
