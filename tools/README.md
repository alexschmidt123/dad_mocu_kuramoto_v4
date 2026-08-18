# Optional tools (not part of the core `scripts/` pipeline)

Core pipeline shells live only in `scripts/`:

1. `data_generation.sh`
2. `training.sh`
3. `evaluation.sh`

Use them via `run.sh` / `sweep_run.sh`. T-sweep metric/time tables
(`src.results.plots`) work for EIG and MOCU on any application
config and are written by `sweep_run.sh` into stamped
`experiments/<stamp>_plots_…` folders (including `--plots-only`).
Diagnostics: `python -m src.experiment moe-mechanism`
or `python -m src.experiment diagnose-collapse`.

This `tools/` tree holds offline audits, diagnostics, and bank sweeps.
Product-gated audits used by `python -m src.experiment` live in core
(`src.banks.audit`, `src.banks.diagnose_control`, `src.banks.diagnose_duration`).

| Path | Purpose |
|------|---------|
| `adaptive_advantage/` | Planning / Fixed-vs-adaptive diagnostic library |
| `diagnostics/` | Offline CLI runners (shim for diagnose_control → `src.banks`) |
| `bank_sweeps/` | Contingency / equilibrium sweeps |
| `fixed_bus_scale_audit/` | Fixed-bus duration-scale vs amp-scale structure audit |
| `plan2_fixed_vs_adaptive/` | T=2 Fixed vs adaptive diagnosis harness |
| `calibrate_shared_margin.py` | Shared margin calibration |
| `check_order_invariance.py` | Order-invariance check |

Safe to omit for a minimal train/eval deliverable. `tests/` is separately
optional and is not required by this tree or by `src/`.
