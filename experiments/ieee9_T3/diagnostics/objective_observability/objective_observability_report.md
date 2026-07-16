# Objective observability report

**Gate: PASS**

## Summary

- prior terminal control: `1.04`
- unique final controls: `4`
- final u_ctrl mean / std: `0.985200` / `0.062981`
- final u_ctrl min / median / max: `0.960000` / `0.960000` / `1.200000`
- fraction changed / reduced / increased from prior: `0.9330` / `0.8350` / `0.0980`
- true-system safety rate: `1.0000`
- mean excess control: `0.975040`
- real Spearman (posterior mean U vs u_req): `0.0424`
- shuffled Spearman: `-0.0802`
- no-update check passed: `True`

## Gate checks

- `unique_final_u_ctrl_count`: PASS (value=4, threshold=3)
- `final_u_ctrl_std`: PASS (value=0.06298063194347928, threshold=0.01)
- `fraction_changed_from_prior`: PASS (value=0.933, threshold=0.05)
- `true_safety_rate`: PASS (value=1.0, threshold=1.0)
- `real_spearman`: PASS (value=0.042393045060597795, threshold=0.0)
- `real_better_than_shuffled`: PASS (value=0.042393045060597795, threshold=-0.08017751745333047)
- `no_update_check`: PASS (value=True, threshold=True)
