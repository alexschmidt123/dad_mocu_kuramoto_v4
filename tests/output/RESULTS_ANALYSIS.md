# Posterior and ROCOF Plot Analysis

Interpretation and test improvements.

## ROCOF plots

rocof_timeseries_by_bus: same A, different B. rocof_timeseries_by_amplitude: same B, different A. First 2s is probe (shaded).

Good: ROCOF rises in 0-2s, curves differ by design. Bad: all same, or explosion.

Improve: add r_max reference line; annotate M_true K_true; align time with ODE.

## Posterior plots

posterior_marginals_by_design: p(M|y) and p(K|y) for 5 designs; gray = prior.

Good: at least one posterior sharper; mode near true. Bad: all like prior; mode far from true; jagged.

Improve: vertical lines at M_true and K_true; optional 2D heatmap; configurable sigma and n_grid.

## Checklist

ROCOF: probe 0-2s visible; add r_max line; show true M,K in title.
Posterior: prior line; M_true and K_true vertical lines; smooth marginals.
