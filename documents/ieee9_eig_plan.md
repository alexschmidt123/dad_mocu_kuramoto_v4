# IEEE9 EIG Research Plan

## 1. Purpose and paper role

The IEEE9 EIG study is the first stage of the full power-grid Bayesian optimal experimental design project. Its purpose is to establish, independently of any downstream control objective, whether temporary active-power probes can identify the uncertain IEEE9 swing-dynamics parameters and whether sequential feedback creates genuine adaptive and non-myopic value.

The scientific question is:

> Can a sequence of safe probing experiments identify the six uncertain regional parameters, and does the most valuable later probe depend meaningfully on earlier observations?

The uncertain parameter vector is fixed as

\[
\theta=(M_1,M_2,M_3,K_1,K_2,K_3).
\]

IEEE9 is the controlled methodology and diagnosis system. Once its physics, observation model, action space, numerical estimators, and adaptive value have been validated, the same framework will be extended to an objective-based IEEE9 problem and then to IEEE14, IEEE30, or larger systems.

MOCU is not required in this stage. IEEE9 EIG is parameter-focused and must remain independent of the objective-based experiments.

## 2. Current reference setting

The current reference bank is stored at:

```text
data/ieee9_duration_bus/
```

Its principal settings are:

| Component | Current value |
|---|---|
| Dynamic model | Kron-reduced IEEE9 swing system |
| Physical probe buses | 1-9 |
| Dynamic generator buses | 1, 2, 3 |
| Unknown parameters | \(M_1,M_2,M_3,K_1,K_2,K_3\) |
| Probe amplitude | 0.05 p.u. |
| Probe durations | 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 s |
| Candidate actions | 54 = 9 buses x 6 durations |
| Probe waveform | Hann-window active-power perturbation |
| Observation PMU | Physical bus 1 |
| Observation window | 4.0 s |
| Stored trajectory resolution | 2560 ODE-step samples |
| Default method-visible samples | \(N_{\mathrm{obs}}=5\) |
| Reference EIG noise | \(\sigma=0.01\) Hz |
| Training theta systems | 512, seed 101 |
| Held-out theta systems | 128, seed 202 |
| Reset rule | Every probe starts from the same equilibrium |

All alternative settings must preserve the theta distribution and train/test theta seeds unless the experiment explicitly studies prior sensitivity.

## 3. Experimental principles

1. Physical settings must be selected using safety, observability, identifiability, and method-independent oracle diagnostics, not according to where DAD or RL-sBOED wins.
2. One clean physical bank should be reused across observation-noise and observation-density sweeps whenever possible.
3. Adaptive and non-myopic value must be established before learned policies are trained at full scale.
4. All methods must receive the same observations, action constraints, theta systems, likelihood, and terminal EIG estimator.
5. Within a replicate seed, methods must share held-out systems and observation-noise realizations for paired comparison while retaining separate algorithm-specific random streams.
6. Every method must be reported with uncertainty across independent replicate seeds.
7. Action diversity is a mechanism diagnostic, not an auxiliary reward or success criterion.
8. Redundant actions may be clustered only using a method-independent physical or statistical criterion.
9. IEEE9 EIG and the later objective-based IEEE9 study must use separate objectives, configurations, results, and claims.

## 4. Experiment 1 - Current-bank physics and identifiability audit

### 4.1 Status

Completed on the existing IEEE9 EIG bank. No BOED method was trained and no simulator was called.

Code:

```text
tools/diagnostics/run_ieee9_eig_experiment1.py
```

Results:

```text
reports/ieee9_eig_stage1/experiment1_physics_identifiability/
```

### 4.2 Required outputs

- Exact trajectory extrema at the recorded PMU.
- Provisional probe frequency and RoCoF screens.
- Across-theta signal magnitude and effective SNR.
- Empirical standardized parameter sensitivity.
- Empirical Fisher-information spectrum.
- Held-out linear-surrogate diagnostics.
- Pairwise clean-response correlations.
- Near-duplicate action inventory.
- Machine-readable tables, plots, provenance, and limitations.

### 4.3 Main findings

- The first of five uniformly selected observations is taken after the first RK4 step and has negligible parameter-dependent signal relative to the configured noise. The present \(N_{\mathrm{obs}}=5\) representation therefore has at most four practically informative values.
- 270 of 1431 action pairs, or 18.9%, have absolute clean-response correlation at least 0.98.
- The mean absolute response correlation is 0.745.
- The observation bank contains only physical-bus-1 frequency trajectories, so it cannot establish spatial observability or certify system-wide probe safety.
- The empirical standardized Fisher condition number is approximately \(9.69\times10^4\), indicating strongly uneven identifiability.
- The weakest empirical direction is dominated by a \(K_2,K_3\) combination.
- At \(\sigma=0.01\) Hz, the median action-level parameter SNR is approximately 0.099 and the maximum is approximately 0.178. This is a low-SNR setting under the audit definition.
- At \(\sigma=0.0025\) Hz, the median SNR is approximately 1.58, providing a useful higher-SNR comparison.

### 4.4 Interpretation

The existing bank is a valid reproducible single-PMU reference, but it is not sufficient to claim that all six regional parameters are well identified or that all 54 probe actions are distinct. It should remain Setting A in subsequent ablation studies rather than being silently replaced.

## 5. Experiment 2 - Physical-model and unit validation

### 5.1 Objective

Document and verify the precise engineering meaning of the current IEEE9 simulator before generating redesigned banks.

### 5.2 Checks

- Verify the MATPOWER IEEE9 network source and Kron-reduction procedure.
- Record the mapping from physical buses 1-9 to the three retained dynamic buses.
- Verify the sign and units of probe injections.
- Verify \(M_i\), \(K_i\), damping, angular-frequency, frequency, and RoCoF units.
- Confirm that every theta system begins at an exact equilibrium.
- Confirm that every probe is removed and the state is reset before the next experiment.
- Compare CPU and CUDA trajectories for representative theta-action pairs.
- Verify ODE time stamps and the relationship among `ode_dt`, stored samples, PMU sampling, and method-visible observation indices.
- Verify that the chosen probe amplitudes remain in the intended safe small-signal regime at all monitored buses.
- Separate measurement noise from simulator error and model discrepancy in the documentation.

### 5.3 Outputs

```text
reports/ieee9_eig_stage1/experiment2_physics_validation/
```

The folder should contain a Markdown report, configuration snapshot, unit table, CPU/CUDA comparison table, representative trajectory plots, and machine-readable validation results.

## 6. Experiment 3 - Observation-time redesign

### 6.1 Problem

Uniform sampling currently spends one of five observations on a negligible near-equilibrium response. The remaining times may also miss early inertia-sensitive dynamics.

### 6.2 Candidate schedules

Compare at least:

1. Current uniform schedule.
2. Early-time enriched schedule.
3. Logarithmically spaced schedule.
4. A physics-informed schedule covering early RoCoF, transient propagation, and late decay.

Example five-point physics-informed schedule:

\[
t_{\mathrm{obs}}=[0.05,0.20,0.75,2.0,4.0]\ \mathrm{s}.
\]

The exact schedule must respect stored resolution and should be selected using sensitivity and identifiability diagnostics, not learned-method performance.

### 6.3 Evaluation

- Per-parameter sensitivity.
- Fisher eigenvalues and condition number.
- Held-out posterior calibration.
- Action distinguishability.
- Single-step EIG with numerical uncertainty.

## 7. Experiment 4 - Duration-catalogue redesign

### 7.1 Settings

Compare:

| Setting | Durations (s) |
|---|---|
| Current | 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 |
| Separated | 0.2, 0.4, 0.8, 1.5, 2.5, 4.0 |

Use fixed amplitude 0.05 p.u. initially and preserve all nine probe buses.

### 7.2 Physical motivation

- Short probes emphasize initial acceleration, RoCoF, and inertia.
- Intermediate probes mix inertia and network-coupling effects.
- Long probes emphasize slower damping/coupling and propagation behavior.

### 7.3 Evaluation

- Parameter sensitivity by bus and duration.
- Fisher-spectrum improvement.
- Reduction in near-duplicate action pairs.
- Single-step EIG distribution.
- Method-independent two-step adaptive gates.

## 8. Experiment 5 - Spatial PMU observability

### 8.1 Settings

Compare:

1. PMU at physical bus 1 only.
2. PMUs at the three dynamic generator buses 1, 2, and 3.
3. A smaller PMU subset selected by a prespecified observability criterion.

The multi-PMU observation may be written as

\[
y_\xi=[\Delta f_1(t_{1:N}),\Delta f_2(t_{1:N}),\Delta f_3(t_{1:N})].
\]

### 8.2 Questions

- Does spatial observation separate \(K_2\) from \(K_3\)?
- Does it improve sensitivity to \(M_3\)?
- Does it reduce the number of electrically equivalent probe buses?
- Does it create complementary actions rather than merely increasing observation dimension?
- Are posterior intervals calibrated on held-out theta systems?

## 9. Experiment 6 - Probe-amplitude validation

Amplitude is an engineering and SNR choice, not a method-tuning variable.

Compare only a small prespecified set, initially:

\[
A\in\{0.025,0.05,0.075\}\ \mathrm{p.u.}
\]

For each amplitude, report:

- Probe-induced maximum frequency deviation and RoCoF at every stored PMU.
- Parameter signal magnitude.
- Effective SNR.
- Normalized trajectory-shape similarity.
- Departure from the intended small-signal regime.
- Single-step EIG and posterior calibration.

Choose the smallest amplitude that produces adequate identifiability while satisfying the prespecified probe-safety conditions. Do not select amplitude from DAD/RL-sBOED rankings.

## 10. Experiment 7 - Method-independent adaptive and non-myopic gate

### 10.1 Purpose

This is the decisive problem-setting experiment. It must be run before full learned-policy sweeps.

### 10.2 Comparators

- High-budget adaptive two-step oracle approximation.
- Repeated Myopic EIG design.
- Optimized open-loop two-probe sequence.

### 10.3 Primary quantities

Non-myopic advantage:

\[
\Delta_{\mathrm{nonmyopic}}
=V_{\mathrm{adaptive}}^{(2)}-V_{\mathrm{myopic}}^{(2)}.
\]

Feedback advantage:

\[
\Delta_{\mathrm{feedback}}
=V_{\mathrm{adaptive}}^{(2)}-V_{\mathrm{open\text{-}loop}}^{(2)}.
\]

Both must be reported with Monte Carlo uncertainty.

### 10.4 Branching diagnostics

- Number of distinct valuable second actions.
- Modal second-action frequency.
- Fraction of histories changing the preferred second action.
- Continuation-value improvement from branching.
- Distribution of branch values.
- Difference between raw action disagreement and value-weighted meaningful branching.

### 10.5 Promotion rule

A setting is promoted to learned-policy training only when:

- Adaptive planning beats Myopic beyond numerical uncertainty.
- Adaptive planning beats optimized open-loop beyond numerical uncertainty.
- A meaningful fraction of histories changes the valuable next action.
- Simulator, posterior-particle, and nested-Monte-Carlo error are smaller than the expected method differences.

## 11. Experiment 8 - Noise and observation-density map

Once a clean physical bank is available, reuse it without new ODE simulations.

Initial grid:

\[
N_{\mathrm{obs}}\in\{2,5,10\},
\qquad
\sigma\in\{0.0025,0.005,0.01,0.015\}\ \mathrm{Hz}.
\]

Run the method-independent two-step gate first. For every cell report:

- Effective SNR.
- Single-step EIG.
- Adaptive-oracle EIG.
- Myopic EIG.
- Optimized open-loop EIG.
- Non-myopic and feedback gaps with uncertainty.
- Branching rate and branching value.
- Posterior calibration.

The final paper must report the complete prespecified SNR curve. A setting must not be selected only because a learned method wins there.

Expected qualitative regimes:

- High SNR: one probe may nearly resolve the system, making Myopic competitive.
- Intermediate SNR: partial early information may route later probes, producing the strongest adaptive value.
- Very low SNR: observations become uninformative and all methods should converge toward similar performance.

## 12. Experiment 9 - Action redundancy and catalogue ablation

Keep the complete physical bank, but create a representative BOED catalogue using a method-independent criterion such as:

- Normalized trajectory correlation.
- Likelihood distance.
- Fisher-information similarity.
- Electrical equivalence from the reduced network.

Compare:

1. Full 54-action catalogue.
2. Clustered distinguishable catalogue.

The clustered catalogue is acceptable only if it preserves oracle EIG and adaptive opportunity while reducing equivalent choices. Redundancy must not be retained or removed for the purpose of manufacturing a Myopic trap.

## 13. Experiment 10 - Learned-policy pilot

Run only after a physical/observation setting passes the method-independent gate.

### 13.1 Initial pilot

| Component | Values |
|---|---|
| Horizons | \(T=2,3,4\) |
| Seeds | 101, 202, 303 |
| Methods | DAD, RL-sBOED, Myopic, optimized open-loop, Random |
| Objective | Common terminal EIG |
| Evaluation | Paired held-out theta systems and observation noise |

### 13.2 Method identities

- DAD must retain its history-dependent trajectory-level adaptive-design objective.
- RL-sBOED must retain PPO with a stepwise/telescoping information objective whose total return matches terminal information gain.
- Myopic must optimize expected one-step EIG.
- Optimized open-loop must select all actions before observations.
- Random must sample from the same feasible catalogue.

No method may receive true theta, future observations, evaluation samples during calibration, or a method-specific likelihood/objective.

### 13.3 Reports

For every method and horizon report:

- Mean terminal EIG plus or minus sample standard deviation across seeds.
- Paired difference and confidence interval versus Myopic.
- Paired difference and confidence interval versus optimized open-loop.
- Fraction of adaptive-oracle advantage recovered.
- Greedy sequence diversity.
- Observation-conditioned next-action change rate.
- Training runtime and online selection runtime.
- Probe-safety statistics.

Three seeds are a pilot minimum. Final paper results should use at least five independent seeds, preferably more if training variance remains material.

## 14. Experiment 11 - Parameter-level interpretation

For every qualified method, report posterior behavior separately for:

\[
M_1,M_2,M_3,K_1,K_2,K_3.
\]

Metrics:

- Posterior mean RMSE.
- Posterior standard deviation.
- Marginal interval coverage.
- Joint calibration or simulation-based calibration.
- Parameter-wise information gain.
- Relationship between selected actions and physical parameter sensitivities.

This experiment connects EIG values to interpretable power-grid identification rather than treating EIG as an isolated machine-learning score.

## 15. Experiment 12 - Robustness and ablation study

After the primary setting is frozen, study:

- Prior-range sensitivity without changing the primary prior.
- Theta support size and held-out particle count.
- Observation-noise misspecification.
- Mild model discrepancy.
- Full versus clustered action catalogue.
- Current versus redesigned observation schedule.
- One PMU versus multiple PMUs.
- Current versus separated durations.
- Policy architecture and training-budget sensitivity.
- Exact/large-budget versus approximate EIG estimators.

These are robustness analyses. They must not redefine the primary setting after method results are known.

## 16. Statistical and computational protocol

### 16.1 Replicate semantics

One seed means one independent run of all methods. For replicate seed \(s\):

- All methods share held-out theta systems and evaluation-noise realizations.
- DAD and RL-sBOED are independently trained.
- Myopic fantasies, open-loop optimization, and Random action streams use deterministic method-specific substreams derived from \(s\).
- Evaluation uses the replicate seed rather than one fixed global seed.

### 16.2 Uncertainty reporting

For every method:

\[
\text{mean EIG}\pm\text{sample SD}.
\]

Also report paired method differences and confidence intervals. Numerical error from nested Monte Carlo, likelihood evaluation, posterior particles, and simulator discretization must be audited separately from training-seed variation.

### 16.3 Data separation

- Training theta particles: policy learning and posterior support only.
- Validation theta particles: checkpoint/model selection only.
- Held-out test theta particles: final evaluation only.
- No test leakage into Fixed/open-loop calibration or hyperparameter selection.

## 17. Recommended execution order

1. Preserve the current bank as Setting A.
2. Complete physical-model and unit validation.
3. Correct and compare observation-time schedules.
4. Generate the separated-duration single-PMU bank.
5. Generate the separated-duration three-PMU bank.
6. Run sensitivity, Fisher, redundancy, and posterior-calibration audits.
7. Run the method-independent \(T=2\) adaptive/Myopic/open-loop gate.
8. Map \(N_{\mathrm{obs}}\) and \(\sigma\) without training learned methods.
9. Evaluate full and clustered action catalogues.
10. Freeze one primary and one robustness setting.
11. Run the three-seed learned-policy pilot at \(T=2,3,4\).
12. Diagnose learned-policy fidelity and adaptivity.
13. Run at least five final seeds only after the pilot passes.
14. Produce parameter-level interpretation and robustness ablations.
15. Freeze IEEE9 EIG before beginning the objective-based IEEE9 study.

## 18. Stage-1 completion criteria

IEEE9 EIG is ready to freeze when:

1. The physical model, parameter definitions, units, equilibrium, probe waveform, and observation timing are documented and tested.
2. Probe safety is evaluated at every required monitored bus.
3. Every uncertain parameter has measurable held-out sensitivity, or any non-identifiable direction is explicitly documented and justified.
4. The action catalogue is not dominated by near-duplicate likelihoods.
5. Posterior inference is calibrated on held-out theta systems.
6. A high-budget adaptive policy significantly beats optimized open-loop design.
7. A high-budget non-myopic policy significantly beats repeated Myopic design.
8. Different observation histories generate valuable next-action branching.
9. Expected differences exceed simulator, likelihood, particle, and Monte Carlo uncertainty.
10. DAD and/or RL-sBOED recover a meaningful, reproducible fraction of the adaptive-oracle advantage without changing their reference identities.
11. Every method is reported with seed variation and paired uncertainty.
12. Results are interpretable through regional inertia/coupling sensitivity rather than method rankings alone.

## 19. Expected paper artifacts

- IEEE9 physical-model diagram and probe/PMU workflow.
- Bus-duration action catalogue and probe-safety table.
- Parameter-sensitivity heatmaps.
- Fisher-information spectra.
- Action-similarity and clustering figures.
- SNR/noise phase diagram.
- Adaptive-oracle versus Myopic versus open-loop gate table.
- Terminal EIG versus horizon curves.
- Parameter-level posterior calibration and RMSE plots.
- Learned-policy branching and selected-action maps.
- Runtime and scalability table.
- Clear separation between IEEE9 EIG conclusions and later objective-based conclusions.

## 20. Immediate next action

The next experiment should be the observation-time and PMU observability redesign. Specifically:

1. Replace the negligible first method-visible sample with an early informative time.
2. Compare the current duration catalogue against separated durations.
3. Compare one PMU at bus 1 against PMUs at generator buses 1-3.
4. Run the same read-only identifiability audit.
5. Run the high-budget \(T=2\) adaptive/Myopic/open-loop EIG gate before any new learned-method sweep.

This sequence directly addresses the problems found by Experiment 1: low effective SNR at \(\sigma=0.01\), uneven six-parameter identifiability, a weak \(K_2/K_3\) direction, single-PMU spatial limitation, and redundant probe actions.
