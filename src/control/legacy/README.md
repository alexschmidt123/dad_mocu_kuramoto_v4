# Legacy / frozen experiment runners

These modules are **not** part of the production control API.

They remain only so historical IEEE5 experiments can be re-verified:

| Module | Historical purpose |
|--------|--------------------|
| `ieee5_t3.py` | Controlled IEEE5 T=3 with frozen margin 0.55 |
| `ieee5_t4.py` | Controlled IEEE5 T=4 |
| `ieee5_t4_fixed_exact.py` | Exact Fixed search for T=4 fairness |
| `ieee5_horizon_sweep.py` | T=1..4 sweep |
| `policy_robust_calibration.py` | Produced frozen margin 0.55 |
| `myopic_convergence.py` | Froze Myopic `n_hypothetical` |
| `diagnose_myopic_fixed.py` | Myopic vs Fixed diagnosis |
| `adaptive_value_diagnosis.py` | Case A/B adaptive-value diagnosis |
| `objective_adaptive_value.py` | IEEE5/9 T=3 adaptive-value study runner |

Production code lives in the parent package (`posterior_ctrl`, `pilot`,
`safety_calibration`, `objective_rl_sboed`, …).

Invoke via CLI subcommands that import `src.control.legacy.*`, not as
manuscript methods.
