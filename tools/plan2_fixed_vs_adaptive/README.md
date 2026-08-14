# Plan-2 Fixed vs adaptive (T=2) diagnosis

Frozen bank: `data/ieee5_plan2_trap_v3` (Solution-1 adaptive-room strengthen).

```bash
python3 tools/plan2_fixed_vs_adaptive/run_t2_fixed_vs_adaptive.py \\
  --noise_sigma 0.01 --seed 101 \\
  --exp-dir experiments/08112026_215946_ieee5_plan2_trap_Uctrl_T2_Nobs200_sigma0p01

python3 tools/plan2_fixed_vs_adaptive/run_t2_fixed_vs_adaptive.py \\
  --noise_sigma 0.005 --seed 101 \\
  --exp-dir experiments/08112026_221616_ieee5_plan2_trap_Uctrl_T2_Nobs200_sigma0p005
```

See `DIAGNOSIS_REPORT.md` (v2 diagnosis) and `documents/solution1_bank_v3_status.md` (v3 freeze).
