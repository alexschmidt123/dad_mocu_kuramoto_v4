# particle_posterior_adequacy

Posterior particle adequacy / convergence study.

## Physical master banks (do not overwrite historical)

| System | Dataset | Path |
|--------|---------|------|
| IEEE5 | `ieee5_particle_adequacy_master_2048` | `data/ieee5_particle_adequacy_master_2048/` |
| IEEE9 | `ieee9_particle_adequacy_master_2048` | `data/ieee9_particle_adequacy_master_2048/` |

Historical production banks remain at `data/ieee5/` and `data/ieee9/`.

IEEE14 is out of scope for this study.

## Nested supports

From each ordered master train bank:

`N ∈ {128, 256, 512, 1024, 2048}` as prefixes `theta_master[:N]` (same rows in Y/U).

## Regenerate

```bash
./run.sh -study particle_posterior_adequacy -system both -stage generate-master
```

Requires GPU + PyCUDA (no CPU ODE fallback).
