# Objective observability report

**Gate: PASS**

## Summary

- prior terminal control: `1.04`
- unique final controls: `4`
- final u_ctrl mean / std: `1.001120` / `0.079026`
- final u_ctrl min / median / max: `0.960000` / `0.960000` / `1.200000`
- fraction changed / reduced / increased from prior: `0.9150` / `0.7500` / `0.1650`
- true-system safety rate: `1.0000`
- mean excess control: `0.990960`
- real Spearman (posterior mean U vs u_req): `0.0448`
- shuffled Spearman: `-0.0857`
- no-update check passed: `True`

## Gate checks

- `unique_final_u_ctrl_count`: PASS (value=4, threshold=3)
- `final_u_ctrl_std`: PASS (value=0.07902623361896986, threshold=0.01)
- `fraction_changed_from_prior`: PASS (value=0.915, threshold=0.05)
- `true_safety_rate`: PASS (value=1.0, threshold=1.0)
- `real_spearman`: PASS (value=0.04481360082431009, threshold=0.0)
- `real_better_than_shuffled`: PASS (value=0.04481360082431009, threshold=-0.08565381937585212)
- `no_update_check`: PASS (value=True, threshold=True)
