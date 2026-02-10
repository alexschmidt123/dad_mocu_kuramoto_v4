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

---

## Improvement suggestions: physics and ML reasonableness

### Power grid physics

1. **ROCOF vs inertia:** Lower M (less inertia) should generally give higher ROCOF for the same probe. Add a test that runs the same design for several M values (K fixed) and asserts ROCOF_max is decreasing in M (or at least monotonic trend).
2. **Frequency nadir:** f_min should stay above a safe bound (e.g. 59.5 Hz) for “moderate” probes, or document that open-loop probes exceed limits. Consider asserting f_min in a band (e.g. 59.0–60.0 Hz) for the current (M_true, K_true) and design set.
3. **Scaling:** Check that B, P_m, D, M, K use consistent per-unit or SI. The design doc uses p.u. for power and s²/rad for M; ensure ODE and config match (e.g. f_nominal = 60 Hz in swing_equation_ode).
4. **Probe amplitude:** Document that A ∈ [0.05, 0.5] yields ROCOF far above typical grid limits (0.1–1 Hz/s) so the setup is “high excitation” for estimation; for near–real-world ROCOF, use smaller A and/or shorter T_p.
5. **Single (M,K):** Design table and posterior plots use one random (M_true, K_true). For physics plausibility, add a test that samples 5–10 (M,K) from the prior and checks ROCOF_max and f_min ranges (no NaNs, no explosion, f_min in [59, 60] Hz).

### ML / simulation

1. **Likelihood scale (σ):** σ = 0.05 Hz/s is used in tests. Check that typical |y − μ(θ,ξ)| across the prior is on the order of σ (not 10σ or 0.1σ), so the likelihood is informative but not overconfident. Option: add a test that computes mean absolute residual for a few designs over a (M,K) grid.
2. **Posterior consistency:** For a single observation, posterior mode should be near (M_true, K_true) for at least some designs. Add an assertion that the design with highest info_gain in the table has posterior mode within a tolerance of the true (e.g. 20% of prior range).
3. **Grid resolution:** Design table uses n_grid=7; posterior plots use 55 or 41. For ML training data, use a consistent grid (e.g. 15–25) and document it so downstream MOCU/DAD scripts match.
4. **Data diversity:** When generating data for DAD/MPNN, ensure (M,K) are sampled from the full prior and designs (bus, amplitude) cover slack, gen, and load buses and low/medium/high A so the model sees diverse ROCOF and posteriors.
5. **Reproducibility:** conftest uses np.random.seed(42) for (M_true, K_true). Document this so that “reasonable” is evaluated on a fixed seed; optionally add a second test with seed=0 or 123 to spot-check another (M,K).

### Test additions (concrete)

- **test_rocof_decreases_with_M:** Fix K_true, vary M in [M_lower, M_upper]; same design; assert ROCOF_max is lower for larger M (or assert Spearman correlation M vs ROCOF < 0).
- **test_f_min_in_band:** For all design_candidates (or a subset), assert 59.0 ≤ f_min ≤ 60.0 (or your chosen band).
- **test_posterior_mode_near_true:** For the design (bus, amplitude) with largest info_gain in the table, compute posterior on a fine grid and assert argmax is within 0.5 × (prior range) of (M_true, K_true).
- **test_multiple_MK_samples:** Sample 5 (M,K) from prior; for each, run 3 designs; assert no NaN/inf in ROCOF_max and f_min, and variance of ROCOF across (M,K) is positive (designs are informative).
