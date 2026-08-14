# Conference poster report (living briefing)

Updated: 2026-08-13

**Status:** living poster/research briefing aligned to the publication north star.
Not a frozen ICML confirmatory evaluation. Older discrete amp×bus×duration
catalogs are **superseded** — do not put those numbers on the poster as the
main claim.

Related sources:
- `documents/publication_experiment_plan.txt` (protocol / win criteria)
- `documents/moe_sboed_workflow.txt` (architecture + training)
- `documents/sBOED_design.tex` (methods write-up)
- `configs/ieee5_continuous_duration.yaml`, `configs/sir_ode.yaml`

---

## 1. One-sentence pitch

**MoE-sBOED** learns a shared base probe policy plus belief-conditioned residual
experts so sequential BOED can adapt across posterior regimes — on power-grid
control (MOCU) and on a classic explicit-likelihood SIR ODE EIG benchmark.

---

## 2. Poster claim checklist (ultra goal)

| ID | Claim | Poster stance |
|---|---|---|
| W1 | Power-grid OBJECTIVE: DAD/RL/MoE beat Fixed/Myopic/Random; MoE best overall | Target; IEEE-5 continuous-duration first |
| W2 | Power-grid EIG: same ranking | Second panel after W1 |
| W3 | SIR ODE EIG transfer (explicit Gaussian) | Required “not only grids” panel |
| W4 | Adaptivity: uniq ≫ 1, router diagnostics | Must appear next to any MoE win |

**Do not claim** a universal MoE win from older multi-amp / multi-bus catalogs, or from
cells where MoE collapses to `uniq≈1` (Fixed-in-disguise).

---

## 3. Method (innovation panel)

Architecture (`BeliefConditionedMoEPolicy`):

```
history + posterior particles + belief summary
                 │
                 ▼
        shared encoder → h
         ┌──────┴──────┐
         ▼             ▼
    ℓ_base(h)    residual experts E=4
         │             │
         │      router top-2 → d_route
         └──────┬──────┘
                ▼
     ℓ = ℓ_base + softplus(λ)·d_route
```

- Experts are **learned residuals**, not Fixed/Myopic/DAD copies.
- Baselines remain external: Random, Fixed, Myopic, DAD, RL-sBOED (+ MatchedDense ablation).
- Training: PPO + telescoping MOCU (or EIG) rewards; **Fixed-BC off** (no Fixed
  cloning); branching regularizer on disagreeing CF fingerprints; belief-gated
  residual scale so corrections vary with posterior regime.
- Checkpointing: `joint_score = MOCU − λ·unique_frac` with **soft unique floor**.
- Legacy fusion router in `src/policies/moe.py` is retired for new runs.

---

## 4. Design space (must be on the poster)

**IEEE-5 continuous-duration** (primary power-grid Ξ):

| Factor | Values |
|---|---|
| Amplitude | fixed `{0.15}` |
| Bus | fixed `{0}` |
| Duration | continuous \(d\in[0.001,0.400]\), bank grid \(\Delta d=0.001\) |
| **N_ξ** | **400** |
| Observation | scalar max\|ROCOF\| (`N_obs=0`) |
| Probe state | **no reset** (carry physical state); chronological \(\xi_1<\xi_2<\cdots<\xi_T\) |
| Config / bank | `configs/ieee5_continuous_duration.yaml` / `data/ieee5_continuous_duration/` |

Why: multi-amp same-bus probes are mostly scale-redundant. Fixing amp/bus and
searching duration makes \(\xi\) a single engineering knob. Frame as a
**deliberate sequential-BOED regime** (look-ahead can beat myopic), not as
“the unique natural IEEE library.”

---

## 5. Experiment matrix (what to show)

### Panel A — Power-grid OBJECTIVE (W1, W4)

- System: IEEE-5 continuous-duration (then IEEE-9 after clean)
- `experiment_type=objective_based`, `N_obs=0`
- `T ∈ {2,3,4,5}`, `σ ∈ {0.01, 0.001}` (confirm 0.005)
- Methods: MoE, DAD, RL, Myopic, Fixed, Random
- Metrics: mean terminal MOCU ↓, safety ≥ 0.95, **n_unique**, paired CIs

Sweep:
```bash
bash sweep_run.sh --config configs/ieee5_continuous_duration.yaml \
  --experiment_type objective_based --T 2,3,4,5 \
  --N_obs 0 --noise_sigma 0.01,0.001 --seed 101
```

### Panel B — Power-grid EIG (W2)

- Same Ξ / systems; `experiment_type=eig_based`
- Metric: terminal vector EIG ↑ + uniq

```bash
bash sweep_run.sh --config configs/ieee5_continuous_duration.yaml \
  --experiment_type eig_based --T 2,3,4,5 \
  --N_obs 0 --noise_sigma 0.001,0.005 --seed 101
```

### Panel C — SIR ODE EIG (W3; required)

- `configs/sir_ode.yaml` — iDAD App. D.6 settings; ξ = time, y = I(τ) count + noise
- Full I(t) bank on `linspace(0,100,10000)`; chronological ξ₁ < ξ₂ < … < ξ_T
- ODE drift only (**no SIR SDE**); T=5
- Training: soft two-step BC (DAD/RL/MoE), PPO for RL+MoE, MoE CF/branching floor
- Same method list; purpose: transfer beyond power systems

```bash
bash run.sh --config configs/sir_ode.yaml --experiment_type eig_based -T 5 --seed 101
```

### Panel D — Mechanism

- `python -m src.experiment moe-mechanism …`
- Router top-2 mass vs step/ESS; expert disagreement; MatchedDense ablation
- Soft-floor before/after uniq–MOCU tradeoff

---

## 6. Current development snapshot (non-confirmatory)

As of 2026-08-13:

- Primary grid config is `ieee5_continuous_duration.yaml` (not older discrete catalogs).
- Soft unique-floor + branching + joint score is the active anti-collapse recipe.
- Pending for a clean poster claim: IEEE-5 continuous-duration objective win with
  uniq ≫ 1; power-grid EIG (adequate training budget); SIR ODE EIG; multi-seed
  confirmatory tables.

Fill quantitative poster cells **only** from continuous-duration (and SIR ODE)
`eval/summary.csv` / `terminal_eig_summary.csv`. Do not paste older multi-amp /
multi-bus means as the headline result.

---

## 7. Poster layout suggestion

1. **Title / pitch** — MoE-sBOED for sequential BOED (control + information)
2. **Problem** — probes → posterior → terminal control (MOCU); duration-only ξ
3. **Method** — shared base + residual MoE diagram (Panel / §3)
4. **Design space** — fixed amp/bus, continuous duration, max\|ROCOF\|
5. **Results** — IEEE-5 OBJECTIVE (MOCU + uniq); optional EIG; SIR ODE
6. **Mechanism** — router / MatchedDense / soft-floor
7. **Takeaway** — MoE wins *with* adaptivity; transfers to SIR ODE

---

## 8. What was removed and why

Previous discrete amp×bus×duration IEEE-5/9 pilots are no longer the publication
design space. Those tables remain recoverable from old `experiments/` folders if
needed for appendix honesty, but they are not poster-primary.
