# Posterior Particle Adequacy and Convergence — Final Report

Diagnostic only: no DAD / RL-sBOED training; scientific problem unchanged.

## Systems
- **ieee5**: latent_dim=10, adequate_N=256, cases={'128': 'D', '256': 'D', '512': 'D', '1024': 'B', '2048': 'B'}, bus={'128': 'BUS-B', '256': 'BUS-B', '512': 'BUS-B', '1024': 'BUS-B', '2048': 'BUS-B'}
- **ieee9**: latent_dim=18, adequate_N=512, cases={'128': 'D', '256': 'D', '512': 'D', '1024': 'D', '2048': 'D'}, bus={'128': 'BUS-B', '256': 'BUS-B', '512': 'BUS-B', '1024': 'BUS-B', '2048': 'BUS-B'}

## Answers

1. Is 128 enough for IEEE5?  **Not for exact argmin identity; see adequate_N** (adequate_N=256).

2. Is 256 enough for IEEE9?  **Not for exact argmin identity; see adequate_N** (adequate_N=512).

3. ESS collapses sharply after the first observation and is severe by h2–h3 (median normalized ESS ≪ 1 even at N=2048). Particle weights concentrate; this is a real finite-support stress, but it does not by itself create large Δ_adaptive.

4–5. Snapped **u_ctrl** median error vs N=2048 is ~0 (grid quantization); the *fraction* of histories with a different snapped value can be non-negligible. Continuous **u_cont** moves more with N and is the higher-resolution diagnostic.

6–8. Exact ξ* identity is only moderately stable (tied snapped landscape). Bus agreement is higher than full-design agreement. Amplitude identity moves more than bus.

9. When ξ* changes, **median reference regret is ≈ 0** — design changes are near-ties, not large objective mistakes.

10. **Δ_adaptive ≈ 0** across particle counts (snapped objective).

11. IEEE5 Case labels by N (A–D): **{'128': 'D', '256': 'D', '512': 'D', '1024': 'B', '2048': 'B'}**; BUS labels: **{'128': 'BUS-B', '256': 'BUS-B', '512': 'BUS-B', '1024': 'BUS-B', '2048': 'BUS-B'}** (BUS-B stable).

12. IEEE9 Case labels by N (A–D): **{'128': 'D', '256': 'D', '512': 'D', '1024': 'D', '2048': 'D'}**; BUS labels: **{'128': 'BUS-B', '256': 'BUS-B', '512': 'BUS-B', '1024': 'BUS-B', '2048': 'BUS-B'}** (BUS-B stable).

13. Smallest practically adequate N (IEEE5, objective-level): **256**

14. Smallest practically adequate N (IEEE9, objective-level): **512**

15. Low adaptive value robust to increased support? **Yes (Outcome 1 for objective-level adaptive value)**

16. Next step: **richer physically meaningful observations**

## Decision
OUTCOME 1 (objective-level): Increasing posterior particle support does **not** overturn low Δ_adaptive or BUS-B. Design *identity* can churn because many actions are snapped-tied (regret ≈ 0). ESS collapse is real and argues for large supports in production, but the previous low-adaptive-value conclusion is not primarily an artifact of using too few particles.

Recommended next step: richer physically meaningful observations, keeping θ and the design space unchanged. Prefer large particle supports (e.g. ≥1024) in future production runs for ESS, but do **not** retrain DAD/RL-sBOED before that observation study.

## True-θ sample-count note
- IEEE5 production train/test θ: 64/16
- IEEE9 production train/test θ: 128/32
IEEE5’s smaller true-θ pool can widen adaptive-value CIs; this is separate from particle-count effects.

## Artifacts
- `experiments/particle_posterior_adequacy/{ieee5,ieee9}_T3/results/`
- `experiments/particle_posterior_adequacy/comparison/`
- Per-system markdown reports under `summary/`
