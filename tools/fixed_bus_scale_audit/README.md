# Fixed-bus scale audit (test)

Diagnostic: on **one fixed bus**, compare two design catalogs and run the
**Myopic-trap** + **adaptive-room** structure checks (pass/fail only — no θ filtering).

| Variant | Catalog | Intent |
|---------|---------|--------|
| `duration_scale` | 6 durations × 1 amp × bus 0 | Non-scale waveform diversity |
| `amp_scale` | 1 duration × 6 amps × bus 0 | Often ROCOF scale-redundant |

## Run

From repo root (CUDA required for bank generation):

```bash
python3 tools/fixed_bus_scale_audit/run_fixed_bus_scale_audit.py
# optional:
python3 tools/fixed_bus_scale_audit/run_fixed_bus_scale_audit.py --force
python3 tools/fixed_bus_scale_audit/run_fixed_bus_scale_audit.py --smoke   # tiny bank, skips U gates
```

## Outputs (written under this folder)

- `data/duration_scale/` — physical bank
- `data/amp_scale/` — physical bank
- `results/summary.json` — compact comparison
- `results/summary.md` — human-readable
- `results/<variant>/bank_structure_audit_{sigma}.json|md` — per-σ audits
