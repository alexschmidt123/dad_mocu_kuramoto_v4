# DAD-MOCU: Design Document

**Sequential Optimal Experimental Design for Power Systems**

*Gaoming Lin · Advisor: Dr. Byung-Jun Yoon*  
*January 2026*

---

## 1. Introduction and Goal

This document describes the full design of a **sequential Bayesian optimal experimental design (sBOED)** framework for power systems. The system is modeled by the **second-order Kuramoto (swing) equation** on an IEEE-14 bus network. The goal is to design **active probing experiments** that reduce uncertainty in a **decision-relevant quantity**—the minimal safe gain $\gamma^*(M,K)$—rather than estimating parameters $(M,K)$ for their own sake. The framework uses **Mean Objective Cost of Uncertainty (MOCU)** as the utility and trains a **Deep Adaptive Design (DAD)** policy to select probes non-myopically.

---

## 2. System Model, Latent Uncertainty, and Decision Objective

### 2.1 Dynamics (network-coupled swing equation)

For each bus $i=1,\dots,N$:

$$
\dot{\theta}_i(t) = \omega_i(t),
$$
$$
M\,\dot{\omega}_i(t) = P_{m,i} - \sum_{j=1}^{N} B_{ij} \sin\bigl(\theta_i(t)-\theta_j(t)\bigr) - (D+K)\omega_i(t) + u^{\text{probe}}_{\xi,i}(t) + u^{\text{ctrl}}_{\gamma,i}(t).
$$

- $i$ indexes the bus; $j$ indexes buses coupled to $i$; $B_{ij}=0$ if no line.
- Summation is the electrical power exchange with neighbors.

### 2.2 Latent parameters

$$
\vartheta = (M,K) \sim p(\vartheta), \quad \vartheta \in \mathbb{R}_+^2,
$$

where $M$ is equivalent system inertia and $K$ is aggregate fast frequency response (droop-like gain). Network $B_{ij}$ (IEEE-14), damping $D$, and nominal injections $P_{m,i}$ are known and fixed.

### 2.3 Planning-level control and security

Control: $u^{\text{ctrl}}_{\gamma,i}(t) = \gamma\, g_i\, \omega_i(t)$, with $\sum_i g_i = 1$, $g_i \ge 0$.

Security-constrained optimal gain:

$$
\gamma^*(\vartheta) = \min_{\gamma} \quad \text{s.t.} \quad \max_t |\dot{f}(t)| \le r_{\max},\;\; \min_t f(t) \ge f_{\min}.
$$

**Goal:** Design probing experiments that reduce uncertainty in $\gamma^*(M,K)$ [1], [2].

---

## 3. Finite-Horizon Sequential Probing Experiment

Episode: $\vartheta \sim p(\vartheta)$ is drawn once and fixed; steps $t=1,\dots,T$.

At step $t$, the experimenter chooses a probing design

$$
\xi_t = (b_t, A_t, T_p) \in \Xi = \mathcal{B} \times \mathcal{A} \times \{T_p\}.
$$

### 3.1 Active probing signal (IBR-style injection)

$$
u^{\text{probe}}_{\xi,i}(\tau) =
\begin{cases}
A_t\, s(\tau; T_p), & i = b_t, \\
0, & i \neq b_t,
\end{cases}
\qquad
s(\tau; T_p) = \frac{1}{2}\biggl(1 - \cos \frac{2\pi\tau}{T_p}\biggr).
$$

The probe is a smooth active-power pulse at bus $b_t$ to excite inertial and primary-frequency dynamics within operational limits.

### 3.2 Observation and history

Simulate the swing equation under $(\vartheta, \xi_t)$, measure frequency response, and compute observation $y_t$. The designer has history

$$
h_t = \{(\xi_1, y_1), \dots, (\xi_t, y_t)\},
$$

while $(M,K)$ remain unobserved [3], [4].

---

## 4. Probing Parameter Table (Fixed Design Choices)

| Parameter | Symbol | Value / Set | Justification |
|-----------|--------|-------------|----------------|
| Probe location | $b_t$ | $\mathcal{B} \subset \{1,\dots,14\}$ | Buses with IBR actuation and high observability [3], [4]. |
| Probe amplitude | $A_t$ | $\mathcal{A} = \{A_1, A_2, A_3\}$ | ROCOF above PMU noise, below security limits; PHIL-validated [3], [4]. |
| Probe duration | $T_p$ | 2 s | Excites inertial/FFR dynamics; avoids slower secondary control [3]. |
| Probe shape | $s(t)$ | Hann window | Smooth, band-limited; widely used in probing [3]. |
| Sampling rate | $f_s$ | 12 Hz | PMU/ROCOF standards [5], [12]. |
| Observation window | $T_{\mathrm{obs}}$ | [0,10] s | ROCOF peak and early transient [5], [6]. |

---

## 5. Observation Model and Feature Extraction (ROCOF-only)

PMU-like frequency: $\Delta f_i(t) = \omega_i(t)/(2\pi)$, $t = n\Delta t$, $\Delta t = 1/f_s$.

ROCOF-only observation used in this work:

$$
\widehat{\dot{f}}(n) = \frac{\Delta f((n+1)\Delta t) - \Delta f(n\Delta t)}{\Delta t},
\qquad
y_t = \mathrm{ROCOF}_{\max} = \max_{n \in \mathcal{W}_t} |\widehat{\dot{f}}(n)|.
$$

ROCOF is directly governed by inertia and fast frequency response and is the primary observable in inertia monitoring and probing studies [4], [5], [6].

---

## 6. Likelihood Modeling and Why DAD (not iDAD)

The swing equation is *deterministic*; uncertainty enters via measurement noise and unmodeled dynamics. We define an explicit likelihood at the *measurement level*.

Simulator-to-feature map: $\mu(\vartheta, \xi_t) = \mathrm{ROCOF}_{\max}(\Delta f(\cdot; \vartheta, \xi_t))$.

Explicit measurement likelihood:

$$
y_t = \mu(\vartheta, \xi_t) + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2),
\qquad
p(y_t \mid \vartheta, \xi_t) = \mathcal{N}(\mu(\vartheta, \xi_t), \sigma^2).
$$

The induced feature $y_t$ admits a tractable probabilistic noise model even though the ODE has no closed-form likelihood.

**Why DAD:** DAD requires numerical evaluation of $\log p(y \mid \vartheta, \xi)$, not an analytic physics likelihood. An explicit, testable likelihood exists at the feature level; the iDAD framework states that DAD should be used when such a likelihood is available [7], [8].

---

## 7. DAD Training, MOCU, and End-to-End Workflow

Design policy (finite-horizon, non-myopic): $\xi_t = \pi_\phi(h_{t-1})$, $t=1,\dots,T$.

History update: $h_t = h_{t-1} \cup \{(\xi_t, y_t)\}$.

Offline posterior (for evaluation): $p(\vartheta \mid h_T) \propto p(\vartheta) \prod_{t=1}^T p(y_t \mid \vartheta, \xi_t)$.

**Mean Objective Cost of Uncertainty (MOCU):**

$$
\mathrm{MOCU}(h_T) = \mathbb{E}_{\vartheta \sim p(\vartheta \mid h_T)} \bigl[ \gamma^*(\mathcal{A}_T) - \gamma^*(\vartheta) \bigr],
$$

where $\mathcal{A}_T$ is the credible set of $p(\vartheta \mid h_T)$.

Training objective (decision-aware DAD): $\phi^* = \arg\min_\phi \mathbb{E}[\mathrm{MOCU}(h_T)]$.

**Workflow:** Offline—simulate episodes, evaluate likelihood, train $\pi_\phi$ to minimize terminal MOCU. Online—apply $\pi_\phi$ sequentially using observed history only. Baselines are myopic; DAD optimizes the full probing sequence.

---

## 8. Implementation: Computational Acceleration

### 8.1 Batched ODE solver

Refactor the swing-equation simulator to handle state tensors of shape [Batch, 2N] and use `torchdiffeq.odeint` for all posterior particles on GPU, removing $O(N)$ Python-loop overhead and giving large speedups for ground-truth MOCU.

### 8.2 Fast MOCU estimator (neural surrogate)

An MPNN uses the IEEE-14 graph $B_{ij}$ to map latent bounds $(\vartheta_{\mathrm{low}}, \vartheta_{\mathrm{up}})$ and probe $\xi_t$ to MOCU. Optional axiom loss: $L_{\mathrm{axiom}} = \max(0, \mathrm{MOCU}_{t+1} - \mathrm{MOCU}_t)$ so the surrogate respects that experiments do not increase uncertainty.

### 8.3 DAD policy integration

The DAD policy uses the fast estimator to compute the expected MOCU matrix (R-matrix) for candidate actions. Final evaluation of all methods uses the batched ODE solver for physics-validated MOCU.

---

## 9. Parameter List and Sequential Design Pseudocode

### 9.1 Fixed probe and observation settings

*Paper-grounded fixed probe settings [3], [9].*

```python
PROBE = {
    "waveform": "hann",       # 0.5 * (1 - cos(2*pi*t/T))
    "T_p": 2.0,              # Duration: 2 s
    "A_set": [0.05, 0.1, 0.2],
    "fs_hz": 12.0,
    "T_obs": 10.0,
    "sigma": 0.01
}
```

### 9.2 Sequential design loop (DAD-MOCU)

**Algorithm: DAD sequential design (non-myopic, terminal MOCU)**

1. **Input:** $T_{\mathrm{horizon}}$, candidate buses, prior particles
2. Initialize history $h_0 \gets \emptyset$; particles $\gets$ prior
3. **For** $t = 1$ to $T_{\mathrm{horizon}}$:
   - Action scores $\gets$ `fast_mocu_estimator`(particles, history, buses, PROBE)
   - $(b_t, A_t) \gets$ `select_action_from_policy`(scores)
   - $\xi_t \gets (b_t, A_t, T_p)$
   - Run swing ODE with $\xi_t$, true $\vartheta$; get $\omega(\cdot)$
   - $y_t \gets$ `extract_max_rocof`($\omega$, $f_s$, $T_{\mathrm{obs}}$)
   - Append $(\xi_t, y_t)$ to history
   - Update particles via likelihood: $p(\vartheta \mid h_t)$
   - (Optional) Compute current MOCU for logging
4. **return** particles (or terminal MOCU)

ROCOF extraction: $\Delta f = \omega/(2\pi)$; discrete derivative over $\Delta t = 1/f_s$; $y_t = \max_n |\widehat{\dot{f}}(n)|$. The fast MOCU estimator takes current credible-set bounds and history, and returns expected MOCU for each candidate $(b, A)$ via the neural surrogate (MPNN), bypassing expensive $\gamma^*$ binary search.

---

## 10. Literature Grounding of Parameters

- **[3] Peng et al. (NREL 2024):** IBR probing; amplitude must exceed noise and stay within security; duration long enough for inertial transient, short vs. secondary control; defines design variable $u = \{A, T, \omega\}$ and ROCOF sensitivity to inertia.

- **[9] Jia et al. (IREC 2023 / NREL):** Real-time inertia estimation tool using probing signals; PHIL validation; motivates sequential probing and small active-power pulses within security limits.

- **[10] Du et al. (2022):** Approximations for optimal experimental design in power system parameter estimation; baseline for how excitation would be chosen for estimation alone; motivates MOCU-based utility instead of FIM for decision-relevant probing.

- **[11] Stanojev et al. (Energies 2021):** Perturbation-based methodology to estimate equivalent inertia of an area monitored by PMUs; supports observation model and ROCOF-based inference.

- **[13] Chakraborty et al. (2025):** Practical inertia estimation using ambient synchrophasor data; swing-equation-based; supports observation model and Bayesian updates.

---

## References

[1] P. Kundur, *Power System Stability and Control*. New York, NY, USA: McGraw-Hill, 1994.

[2] F. Dörfler and F. Bullo, "Synchronization in complex networks of phase oscillators: A survey," *Automatica*, vol. 50, no. 6, pp. 1539–1564, 2014.

[3] J. Peng et al., "Probing signal-based inertia and frequency response estimation for power systems with high penetration of inverter-based resources," in *Proc. IEEE PES General Meeting*, Seattle, WA, USA, 2024; NREL/CP-5D00-87925.

[4] Y. Zhang et al., "Synchrophasor data-based inertia estimation for regional grids in interconnected power systems," *Frontiers Energy Res.*, vol. 10, 2022, Art. 989430.

[5] ENTSO-E, "Inertia and rate of change of frequency (RoCoF)," ENTSO-E, Brussels, Belgium, Dec. 2020.

[6] J. Tan et al., "Power system inertia estimation: Review of methods and the impacts of converter-interfaced generations," *Int. J. Electr. Power Energy Syst.*, vol. 134, Jan. 2022, Art. 107362.

[7] A. Foster et al., "Deep adaptive design: Amortizing sequential Bayesian experimental design," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2021, pp. 3384–3395.

[8] D. R. Ivanova et al., "Implicit deep adaptive design: Policy-based experimental design without likelihoods," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 34, 2021, pp. 25785–25798.

[9] X. Jia et al., "Real-time inertia estimation tool implementation based on probing signals," in *Proc. 14th Int. Renewable Energy Congr. (IREC)*, 2023; NREL/CP-5000-89049.

[10] Y. Du, A. Engelmann, T. Faulwasser, and B. Houska, "Approximations for optimal experimental design in power system parameter estimation," *arXiv:2203.14011*, 2022.

[11] M. Stanojev et al., "A perturbation-based methodology to estimate the equivalent inertia of an area monitored by PMUs," *Energies*, vol. 14, no. 24, 2021, Art. 8477.

[12] NASPI, "Phasors or waveforms: Considerations for choosing measurements to match your application," PNNL-31215, Apr. 2021.

[13] T. Chakraborty et al., "A practical approach towards inertia estimation using ambient synchrophasor data," *arXiv:2505.02978*, 2025.
