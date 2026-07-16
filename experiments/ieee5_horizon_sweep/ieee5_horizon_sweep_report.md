# IEEE5 horizon sweep report

## Status

**Stopped at T=2** under the declared stop condition: at least one method true safety rate `< 1.0`.

- Frozen Myopic `n_hypothetical = 1024` (validation convergence; no smaller count met 0.95 action agreement).
- Frozen terminal-rule hash `dc0dc35332b394b7` (α=0.05, margin=0.40).
- T=1 completed with all safeties = 1.0.
- T=2 method evaluation completed, then stopped before T=3/T=4.
- IEEE9 / IEEE14 not started.

### T=2 stop detail

| method | mean u_ctrl | true_safety_rate |
|--------|-------------|------------------|
| dad (primary seed 101 by best val) | 0.775 | **0.936** |
| dad seed 202 | 0.752 | 1.000 |
| dad seed 303 | 0.734 | 1.000 |
| myopic (n_h=1024) | 0.759 | 1.000 |
| fixed | 0.757 | 1.000 |
| random | 0.788 | **0.986** |

Unsafe random / DAD-101 rollouts show posterior **ESS ≈ 1** collapsing onto a wrong particle, yielding `u_ctrl=0.45 < u_req` (e.g. 0.5). This is particle-collapse under-control, not a Myopic sample-count issue.

T=2 observability informational metrics matched the certified pilot exactly; GPU safety was 0.996 (4/1000) and was waived under certified-rule reuse so evaluation could run.

## 1. Production Myopic sample count

**1024**

Selected by validation-only convergence against reference 1024. No count in {16,…,512} achieved `selected_action_agreement_with_1024 ≥ 0.95` (quantized ties make exact action agreement hard even when mean u_ctrl is close). Per the selection rule: use 1024.

Source: `experiments/ieee5_horizon_sweep/myopic_convergence/`  
Frozen into `config/ieee5_config.yaml`.

## 2. Convergence highlights

| n_h | agree@1024 | rank ρ | ‖Δmean u‖ | seed std | pass |
|----:|----------:|-------:|----------:|---------:|:----:|
| 16 | 0.052 | 0.744 | 0.0026 | 0.0055 | no |
| 32 | 0.073 | 0.806 | 0.0010 | 0.0050 | no |
| 64 | 0.094 | 0.785 | 0.0073 | 0.0106 | no |
| 128 | 0.104 | 0.828 | 0.0016 | 0.0063 | no |
| 256 | 0.229 | 0.861 | 0.0031 | 0.0050 | no |
| 512 | 0.208 | 0.866 | 0.0031 | 0.0039 | no |
| 1024 | 1.000 | 1.000 | 0.0000 | 0.0033 | yes |

## 3. Frozen T=2 results (n_h=1024)

Myopic − Fixed paired (1000 rollouts, 10k bootstrap):

- mean diff = **0.00185**
- 95% CI = **[−0.00315, 0.00690]** (contains 0)
- fraction tied = 0.685

**Statistically tied** under this experiment. Do not interpret Fixed as better than Myopic.

## 4. T=1 Myopic–Fixed equivalence

| | myopic | fixed |
|--|--------|-------|
| mean u_ctrl | 0.7830 | 0.7875 |
| safety | 1.0 | 1.0 |

Paired myopic−fixed: mean −0.0045, 95% CI [−0.0073, −0.0017] (excludes 0) but absolute gap 0.0045 is small. Fixed subset `[16]` (exhaustive). Same one-action objective; residual gap from offline Fixed MC vs online Myopic under quantized ties. **Not a material baseline failure.**

## 5–6. Per-T metrics and safety

### T=1 (all safety 1.0)

| method | mean | std | median | q10 | q90 | excess | runtime |
|--------|------|-----|--------|-----|-----|--------|---------|
| dad | 0.784 | — | — | — | — | — | — |
| myopic | 0.783 | — | — | — | — | 0.464 | — |
| fixed | 0.788 | — | — | — | — | 0.469 | — |
| random | 0.869 | — | — | — | — | 0.550 | — |

(See `T1/eval/summary.json` for full fields.)

### T=2

See stop table above. Myopic/Fixed safe; primary DAD and Random not.

## 7. DAD per-seed (T=2)

| seed | best val u_ctrl | test mean u | test safety |
|------|-----------------|-------------|-------------|
| 101 (primary) | 0.704 | 0.775 | 0.936 |
| 202 | 0.725 | 0.752 | 1.000 |
| 303 | 0.708 | 0.734 | 1.000 |

Aggregate of safe seeds 202/303: mean u ≈ 0.743. Primary rule (best val) selected the most aggressive seed, which under-controlled on test.

## 8. Paired CIs (selected)

**T=1 myopic−fixed:** mean −0.0045, CI [−0.0073, −0.0017]  
**T=2 myopic−fixed:** mean +0.0019, CI [−0.0032, +0.0069] → **tied**  
**T=2 dad(primary)−myopic:** not used for ranking (dad primary unsafe)

## 9. Myopic ties

Quantized ties remain common (see prior diagnosis and convergence). Exact action agreement to the 1024-reference is low for all smaller n_h despite similar mean u_ctrl.

## 10. Fixed subsets

| T | subset | search | note |
|---|--------|--------|------|
| 1 | [16] | exhaustive | C(30,1)=30 |
| 2 | [10,11] | exhaustive | C(30,2)=435 |

## 11. DAD adaptation

T=2 seed 101 primary showed under-control failures; seeds 202/303 remain candidates for adaptive analysis after a safe primary rule is enforced (validation under-control constraint).

## 12. Runtime

Myopic with n_h=1024 dominates wall time (~tens of minutes per T for 1000 rollouts). DAD train ~seconds/seed. Fixed exhaustive at T≤2 is cheap; T≥4 will need approximate search (C(30,4)=27405 > threshold 5000).

## 13–14. Horizon questions

- **DAD vs Myopic / Fixed:** not answerable for T≥2 until primary-seed safety is resolved; safe DAD seeds at T=2 have mean u similar to Myopic/Fixed.
- **Myopic vs Fixed:** **tied** at T=2 with n_h=1024 (CI contains 0).
- **vs Random:** at T=1 objective methods clearly beat Random (0.78–0.79 vs 0.87). At T=2 Random is unsafe under particle collapse.

## 15. Files

Created/updated:

- `src/control/myopic_convergence.py`
- `src/control/ieee5_horizon_sweep.py`
- `src/control/myopic.py` (tie diagnostics)
- `src/control/pilot.py` (n_h freeze, 10k bootstrap)
- `src/cli.py` (`select-myopic-n-hypothetical`, `run-ieee5-horizon-sweep`)
- `config/ieee5_config.yaml` (frozen n_h=1024)
- `tests/test_ieee5_horizon_sweep.py`
- `experiments/ieee5_horizon_sweep/` (myopic_convergence, T1, T2, summaries)

Original pilot `experiments/07132026_220727_ieee5_T2` not overwritten.

## 16. Tests / stop conditions

- Pre-sweep unit tests passed (validation-only selection, rule hash, keyed noise).
- Stop fired correctly on T=2 safety `< 1.0`.
- T=3/T=4 not run.

## 17. Ready for IEEE9?

**No.** Before freezing IEEE5 or moving to IEEE9:

1. Enforce a validation under-control / safety constraint in DAD checkpoint selection (best val u_ctrl alone selected an unsafe seed).
2. Address posterior collapse (ESS≈1) paths that let Random (and aggressive DAD) under-control despite margin 0.40.
3. Re-run T=2..4 with all safeties = 1.0.
4. Keep Myopic n_h=1024 (or revisit agreement metric to score-stable rather than exact-action agreement under ties).
