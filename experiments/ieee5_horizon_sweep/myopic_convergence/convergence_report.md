# Myopic n_hypothetical validation convergence

**Selected:** `1024`
**Reference:** `1024`
**Selection source:** validation only (test unused)
**Terminal rule hash:** `dc0dc35332b394b7`

## Thresholds

```
{
  "agreement_min": 0.95,
  "rank_corr_min": 0.95,
  "mean_u_diff_max": 0.005,
  "seed_std_max": 0.005
}
```

## Results

| n_h | agree@1024 | rank ρ | |Δmean u| | seed std | pass | runtime/roll |
|---:|---:|---:|---:|---:|:---:|---:|
| 16 | 0.052 | 0.744 | 0.0026 | 0.0055 | False | 0.0147 |
| 32 | 0.073 | 0.806 | 0.0010 | 0.0050 | False | 0.0247 |
| 64 | 0.094 | 0.785 | 0.0073 | 0.0106 | False | 0.0450 |
| 128 | 0.104 | 0.828 | 0.0016 | 0.0063 | False | 0.0854 |
| 256 | 0.229 | 0.861 | 0.0031 | 0.0050 | False | 0.1657 |
| 512 | 0.208 | 0.866 | 0.0031 | 0.0039 | False | 0.3281 |
| 1024 | 1.000 | 1.000 | 0.0000 | 0.0033 | True | 0.6513 |
