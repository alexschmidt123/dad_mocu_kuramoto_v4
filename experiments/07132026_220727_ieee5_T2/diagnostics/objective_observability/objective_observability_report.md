# Objective observability report

**Gate: PASS**

## Summary

- prior terminal control: `1.0`
- unique final controls: `7`
- final u_ctrl mean / std: `0.790950` / `0.163495`
- final u_ctrl min / median / max: `0.450000` / `0.800000` / `1.000000`
- fraction changed / reduced / increased from prior: `0.8070` / `0.8070` / `0.0000`
- true-system safety rate: `1.0000`
- mean excess control: `0.477650`
- real Spearman (posterior mean U vs u_req): `0.4444`
- shuffled Spearman: `-0.0347`
- no-update check passed: `True`

## Gate checks

- `unique_final_u_ctrl_count`: PASS (value=7, threshold=3)
- `final_u_ctrl_std`: PASS (value=0.16349494640508008, threshold=0.01)
- `fraction_changed_from_prior`: PASS (value=0.807, threshold=0.05)
- `true_safety_rate`: PASS (value=1.0, threshold=1.0)
- `real_spearman`: PASS (value=0.4443705343785322, threshold=0.1)
- `real_better_than_shuffled`: PASS (value=0.4443705343785322, threshold=-0.034674855472798756)
- `no_update_check`: PASS (value=True, threshold=True)
