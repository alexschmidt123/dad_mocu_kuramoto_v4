# Repository cleanup report

**Date:** 2026-07-16 / 2026-07-17  
**Scope:** Project structure, experiment output format, `src/` / `scripts/` cleanup for the objective control-SBOED project.

---

## 1. Structure before → after

### Before (problem)

- New study outputs under `experiments/objective_rl_sboed/` used an **ad-hoc** layout (`dad_random_init/seed_*` at the system root, loose `*.log` files, `sensitivity_audit/` at study root).
- This **did not match** the authoritative control-objective template (`experiments/ieee5_T3`, `experiments/07082026_*`).
- Incomplete/smoke artifacts and top-level run logs lived beside authoritative results.
- `run.sh` hard-coded IEEE5 for `-T 3` / `-T 4` regardless of `-config`.

### After (standard)

Authoritative template followed (union of pre-2026-07-13 stamped runs + IEEE5/9 T=3 control-objective layout):

```
experiments/<experiment>/
  run_config.yaml
  run_metadata.json          # entry_point, git hash, seed, system, T, method, terminal-rule hash
  model/                    # checkpoints (symlinks/copies)
  train/<method_key>/seed_*/
  eval/                     # summary.csv, method dirs, paired_bootstrap.csv, plots/
  diagnostics/              # sensitivity, observability, calibration
  logs/                     # including logs/scratch for non-authoritative debris
  summary/                  # final_report.md, selection JSON
```

Study root `experiments/objective_rl_sboed/` now uses the same top-level folders.

---

## 2. Removed / relocated generated outputs

| Path | Action | Classification |
|------|--------|----------------|
| `experiments/ieee5_T3_run.log` | → `_archive_obsolete/` | incorrectly stored top-level log |
| `experiments/ieee5_T4_run.log` | → `_archive_obsolete/` | same |
| `experiments/ieee9_T2_pilot_run.log` | → `_archive_obsolete/` | same |
| `experiments/ieee5_dad_adaptive_value_diagnosis_run.log` | → `_archive_obsolete/` | same |
| `experiments/dad_ppo_stage2/smoke/` | → `_archive_obsolete/dad_ppo_stage2_smoke/` | incomplete smoke |
| `experiments/dad_rl_training_improvement/audit/smoke/` | → `_archive_obsolete/dad_rl_smoke/` | incomplete smoke |
| `scripts/__pycache__/` | → `_archive_obsolete/scripts_pycache/` | not a workflow script |
| `objective_rl_sboed/*.log` | → `objective_rl_sboed/logs/` | format migration |
| `objective_rl_sboed/sensitivity_audit/` | → `diagnostics/sensitivity_audit/` | format migration |
| `objective_rl_sboed/ieee5_T3/*_init/seed_*` | → `ieee5_T3/train/*_init/seed_*` | format migration |
| Checkpoints | published under `ieee5_T3/model/...` | format migration |
| `matlab/results/` | **retained**; pointer under `_archive_obsolete/matlab_results_pointer/` | outside `experiments/`; not deleted pending confirmation |

**Not deleted:** any authoritative historical experiment (`07082026_*`, `ieee5_T3`, `ieee9_T3`, `objective_adaptive_value`, `dad_advanced_training`, `dad_ppo_stage2` non-smoke, banks under `data/`).

Migration log: `experiments/objective_rl_sboed/logs/layout_migration.json`.

---

## 3. Files added / merged for formatting

| File | Role |
|------|------|
| `src/experiment_layout.py` | Unified layout I/O: stamped-run helpers + study writers (`ensure_standard_layout`, `RunMetadata`, `write_study_run_config`) — formerly split with `experiment_io.py` |
| `src/control/objective_rl_sboed/layout.py` | Study-specific migration + `train/<method>/seed_*` paths |
| `src/control/posterior_ctrl.py` | Shared `ControlDecision` / `u_raw` / `u_ctrl` |
| `src/neural/rl_policy.py` | Production `DADPolicy` / `RLSBOEDPolicy` / `PolicyTrainer` (internal encoder labels private) |

---

## 4. `src/` consolidation status

### Authoritative production pieces

| Responsibility | Module |
|----------------|--------|
| Terminal control (`u_ctrl`, `u_raw`) | `src/control/posterior_ctrl.py` |
| Frozen rule load | `src/control/terminal_rule.py` |
| U-bank | `src/control/banks.py` + `u_req.py` |
| Safety calibration | `src/control/safety_calibration.py` |
| Fixed search | `src/control/fixed_search.py` |
| Myopic | `src/control/myopic.py` |
| Legacy REINFORCE DAD (CLI train) | `src/neural/train.py` + `src/neural/policy.py` |
| Study DAD / RL-sBOED PPO | `src/control/objective_rl_sboed/` + `src/neural/rl_policy.py` |
| Experiment I/O | `src/experiment_layout.py` |

### Intentionally retained but non-production

Diagnostic / historical modules under `src/control/` (e.g. `objective_adaptive_value.py`, `adaptive_value_diagnosis.py`, `ieee5_t3.py`, `ieee5_t4.py`, `ppo_stage2.py` support code) remain for reproducibility of frozen experiments. They are **not** manuscript methods.

Deleted advanced-training sources (`dad_advanced_training.py`, etc.) stay deleted; their experiment trees are frozen.

### Neural production naming

Production API exposes only:

- `DADPolicy`
- `RLSBOEDPolicy`
- `PolicyTrainer` / `AdaptiveExperimentPolicy`

Internal backbone constants are private (`_PRODUCTION_BELIEF`, `_PRODUCTION_HISTORY`) and must not appear in manuscript method tables.

---

## 5. Scripts audit

| Script | Status | Workflow stage |
|--------|--------|----------------|
| `scripts/data_generation.sh` | kept | offline banks |
| `scripts/control_safety_calibration.sh` | kept | safety calibration |
| `scripts/objective_observability.sh` | kept | objective diagnostics |
| `scripts/dad_training.sh` | kept | legacy DAD train (REINFORCE path) |
| `scripts/evaluation.sh` | kept | legacy four-method eval |
| `scripts/objective_rl_sboed.sh` | kept / updated | DAD vs RL-sBOED study + `migrate-layout` |

All pass `bash -n`.

---

## 6. Entry points

### `./run.sh` (single complete run)

```bash
# Legacy full pipeline (banks → safety → observability → DAD → eval)
./run.sh -config ieee5_config -T 2
./run.sh -config ieee5_config -exp-dir experiments/<existing>

# Objective RL-sBOED study (official)
./run.sh -study objective_rl_sboed -system ieee5 -stage migrate-layout
./run.sh -study objective_rl_sboed -system ieee5 -stage sensitivity
./run.sh -study objective_rl_sboed -system ieee5 -stage run-system
./run.sh -study objective_rl_sboed -system ieee5 -stage run-system --smoke
```

`-T 3` / `-T 4` shortcuts are now gated on the **selected config** (no silent IEEE5 assumption for unrelated configs).

### `./sweep_run.sh` (multi-run)

```bash
./sweep_run.sh -config ieee5_config,ieee9_config -from 3 -to 3
```

Calls `./run.sh` under `.sweep.lock` (flock). Does not duplicate workflow logic.

### Complete-run rule

Authoritative results must record in `run_metadata.json`:

- `entry_point`: `run.sh` or `sweep_run.sh`
- timestamp, git commit, system, horizon, method, seed, terminal-rule hash, data_dir

Direct `python -m ...` calls are for smoke/debug only unless reproduced through `run.sh`.

---

## 7. Final workflow

```
Offline Data Generation
  -> Safety Calibration
  -> Objective Diagnostics (observability / sensitivity / adaptive-value)
  -> Fixed Optimization
  -> DAD / RL-sBOED Training
  -> Myopic / Fixed / Random Evaluation
  -> Statistical Comparison
  -> Final Report
```

---

## 8. Tests executed

```text
pytest tests/test_objective_rl_sboed.py
→ 6 passed

bash -n run.sh sweep_run.sh scripts/*.sh
→ OK

layout assertions on experiments/objective_rl_sboed/ieee5_T3
→ train/, model/, eval/, run_config.yaml, run_metadata.json present
```

Full historical suite against deleted advanced-training modules was not force-run (those sources remain absent by design).

---

## 9. Retained authoritative experiments

- `experiments/07082026_*` (pre-2026-07-13 format reference)
- `experiments/ieee5_T3`, `ieee9_T3`, `ieee5_T4`
- `experiments/objective_adaptive_value/` (Case B; frozen)
- `experiments/dad_advanced_training/`, `dad_ppo_stage2/` (non-smoke), `dad_rl_training_improvement/` (non-smoke)
- `data/*` offline banks
- Frozen terminal-rule JSON inside system experiment dirs

---

## 10. Remaining cleanup (documented, not silently deleted)

1. Aggressive deletion of every diagnostic `src/control/*` helper would risk breaking frozen experiment loaders — deferred.
2. `matlab/results/` still outside `experiments/`; relocate only with explicit approval.
3. IEEE9 `objective_rl_sboed` system dir will be created on first official `./run.sh -study ... -system ieee9` run using the same layout.
4. Incomplete mid-migration training seeds were reorganized under `train/`; any future training must go through `run.sh` so `run_metadata.json` stays authoritative.

---

## 11. src/control consolidation (2026-07-17)

Moved one-shot IEEE5 / historical diagnostic runners into `src/control/legacy/`:

- ieee5_t3.py, ieee5_t4.py, ieee5_t4_fixed_exact.py, ieee5_horizon_sweep.py
- policy_robust_calibration.py, myopic_convergence.py, diagnose_myopic_fixed.py
- adaptive_value_diagnosis.py, objective_adaptive_value.py

Extracted shared bank MC helpers into `src/control/posterior_batch.py`.

Merged `run_keyed_history` into `src/control/terminal_rule.py` (thin shim left at `shared_rollout.py`).

Production surface under `src/control/*.py` is now: banks, cuda_control, diagnose, eval_metrics, fixed_search, generate, myopic, observability, pilot, posterior_batch, posterior_ctrl, safety_calibration, terminal_rule, u_req, objective_rl_sboed/, plus deprecated shared_rollout shim.

---

## 12. src/neural consolidation (2026-07-17)

Production surface:
- `policy.py` — REINFORCE DADPolicy
- `train.py` — REINFORCE trainer
- `rl_policy.py` — self-contained PPO DAD / RL-sBOED backbone (merged former Stage-2 H0+B0 path)

Moved to `src/neural/legacy/`:
- `advanced_dad.py`, `ppo_stage2.py` (multi-encoder Stage-2 diagnostic stack)

Removed (unused; archived):
- `training_performance.py` → `experiments/_archive_obsolete/neural/training_performance.py`
  (legacy ΔH verdict writer; zero importers)

---

## 13. src/ root consolidation (2026-07-17)

Deleted / archived (zero or dead importers):
- `theta_support.py` → `_archive_obsolete/src_root/` (replaced by `table_scoring.TableThetaSupport`)
- `design_eig.py` → `_archive_obsolete/src_root/` (superseded by `stepwise_eig/`; unused import removed from `experiment.py`)
- `experiment_io.py` → merged into `experiment_layout.py`

Moved:
- `plot_summary.py` → `src/legacy/plot_summary.py` (Foster-era summarize/plot CLI only)

Production root modules remaining:
`cli.py`, `config.py`, `data.py`, `experiment.py`, `experiment_layout.py`, `methods.py`, `rollout.py`, `run_context.py`, `table_scoring.py`
