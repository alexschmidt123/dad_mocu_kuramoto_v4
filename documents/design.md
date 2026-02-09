# DAD-MOCU: Design Document

**Sequential Optimal Experimental Design for Power Systems**

*Gaoming Lin · Advisor: Dr. Byung-Jun Yoon · January 2026*

*This document is the prototype for the project paper.*

---

## 1. Introduction and Goal

This document describes the design of a **sequential Bayesian optimal experimental design (sBOED)** framework for power systems. The system is modeled by the **second-order Kuramoto (swing) equation** on an IEEE-14 bus network. The goal is to design **active probing experiments** that reduce uncertainty in a **decision-relevant quantity**—the minimal safe gain $\gamma^*(M,K)$—rather than estimating parameters $(M,K)$ for their own sake. The framework uses **Mean Objective Cost of Uncertainty (MOCU)** as the utility and trains a **Deep Adaptive Design (DAD)** policy to select probes non-myopically.

---

## 2. System Model and Dynamics

### 2.1 Network Topology (IEEE 14-Bus Standard)

The system topology is fixed and defined by the standard IEEE 14-bus test case.
* **Bus Set ($\mathcal{V}$):** The set of $N=14$ buses, indexed $i \in \{1, \dots, 14\}$.
* **Branch Set ($\mathcal{E}$):** The specific set of 20 transmission lines and transformers as defined in the standard IEEE 14-bus data. A physical line exists between bus $i$ and $j$ if $(i, j) \in \mathcal{E}$.
* **Coupling Structure:** The network connectivity is encoded in the **Susceptance Matrix** $\mathbf{B} \in \mathbb{R}^{N \times N}$. The entry $B_{ij} > 0$ represents the magnitude of the line susceptance if $(i, j) \in \mathcal{E}$, and $B_{ij} = 0$ otherwise.

**Node types (standard power-system classification):**

| Type | Buses | Role |
|------|--------|------|
| **Slack** | 1 | Reference (angle/frequency); balances real/reactive power. |
| **Generator (PV)** | 2, 3, 6, 8 | Voltage-controlled; inject real power. |
| **Load (PQ)** | 4, 5, 7, 9, 10, 11, 12, 13, 14 | Consume P and Q. |

In the swing-equation model used here, every bus has phase and frequency dynamics; the types above are the conventional classification and guide where to probe or observe.

**Connectivity (degree and neighbors, 1-based bus labels):** Bus 4 is the only degree-5 node (hub); buses 2, 5, 6, 9 have degree 4. Symmetric pairs: (10, 14) and (11, 13) have identical connectivity and yield identical design outcomes for the same probe amplitude.

| Bus | Degree | Neighbors |
|-----|--------|-----------|
| 1 | 2 | 2, 5 |
| 2 | 4 | 1, 3, 4, 5 |
| 3 | 2 | 2, 4 |
| 4 | **5** | 2, 3, 5, 7, 9 |
| 5 | 4 | 1, 2, 4, 6 |
| 6 | 4 | 5, 11, 12, 13 |
| 7 | 3 | 4, 8, 9 |
| 8 | 1 | 7 |
| 9 | 4 | 4, 7, 10, 14 |
| 10 | 2 | 9, 11 |
| 11 | 2 | 6, 10 |
| 12 | 2 | 6, 13 |
| 13 | 3 | 6, 12, 14 |
| 14 | 2 | 9, 13 |

**Experiment design preference:** For maximum information (M/K estimation), prefer high-degree buses: **4** (hub), then **2, 5, 6, 9**. To cover node types use **1** (slack), **2** or **3** (gen), **4** (load hub), **7** (load), and **10** or **14** (load). For minimal redundancy, use one from each symmetric pair (e.g. 10 and 13, or 14 and 11).

### 2.2 Swing Equation (Second-Order Kuramoto Model)

The system state at time $t$ is $\mathbf{x}(t) = [\boldsymbol{\theta}(t), \boldsymbol{\omega}(t)]^\top \in \mathbb{R}^{2N}$. The dynamics follow the **Second-Order Kuramoto Model** (Swing Equation), adapted for structure-preserving power networks.

For each bus $i \in \mathcal{V}$:

$$
\dot{\theta}_i(t) = \omega_i(t)
$$

$$
M_i \dot{\omega}_i(t) = P_{m,i} - P_{e,i}(\boldsymbol{\theta}(t)) - (D_i + K_i)\omega_i(t) + u_i(t)
$$

**Where the electrical power flow $P_{e,i}$ is:**
$$
P_{e,i}(\boldsymbol{\theta}(t)) = \sum_{j \in \mathcal{N}_i} B_{ij} \sin\bigl(\theta_i(t) - \theta_j(t)\bigr)
$$
*(Note: $\mathcal{N}_i = \lbrace j \mid (i, j) \in \mathcal{E} \rbrace$ denotes the set of neighbors for bus $i$.)*

### 2.3 Notation and Units

| Symbol | Definition | Unit / Domain |
| :--- | :--- | :--- |
| $\theta_i$ | Voltage phase angle at bus $i$. | Rad |
| $\omega_i$ | Angular frequency deviation from synchronous speed $\omega_s$. | Rad/s |
| $P_{m,i}$ | Net mechanical power injection (Generation - Load). | p.u. |
| $B_{ij}$ | Line susceptance magnitude between bus $i$ and $j$. | p.u. |
| $D_i$ | Load-damping coefficient (frequency sensitivity). | p.u. |
| $u_i(t)$ | Total external control/probing injection. | p.u. |
| **Latent $\vartheta$** | **Uncertain Parameters** | |
| $M_i$ | **Effective Inertia Coefficient.** Related to inertia constant $H$ by $M = 2H/\omega_s$. | $s^2/\mathrm{rad}$ |
| $K_i$ | **Primary frequency response (droop) gain.** Aggregate governor/FFR gain; power response proportional to frequency deviation. | p.u. |

### 2.4 Latent Space Prior
The parameter vector $\vartheta = (M, K)$ is drawn from a uniform prior $p(\vartheta)$ over physically validated ranges for a 60 Hz system. **$M$ is the effective inertia coefficient** in the swing equation, related to the inertia constant $H$ (seconds) by $M = 2H/\omega_s$ with $\omega_s = 2\pi f_0$.
* **Inertia ($M$):** $[0.01,\, 0.06]$ $s^2/\mathrm{rad}$. This corresponds to $H \in [2.3,\, 5.0]$ s via $M = 2H/\omega_s$ at $\omega_s = 2\pi\times 60$ rad/s (Kundur; typical synchronous machine range).
* **Droop gain ($K$):** $[0.05,\, 0.50]$ p.u. (primary frequency response; literature often reports droop in %, e.g. 4–6%.)

---

## 3. Decision Objective: Minimal Safe Gain

We aim to estimate the **Minimal Safe Gain** $\gamma^*$, defined as the smallest control effort required to maintain system security under a reference contingency (e.g., load step).

$$
\gamma^*(\vartheta) = \inf \left\lbrace \gamma \in \mathbb{R}_+ \mid \forall t:\; \lvert \dot{f}(t) \rvert \le r_{\max} \land f(t) \ge f_{\min} \right\rbrace
$$

**Security Constraints:**
* **ROCOF Limit ($r_{\max}$):** 0.1 Hz/s. (Tightened for non-trivial control; standard withstand is higher, e.g. 0.5–2 Hz/s.)
* **Nadir Limit ($f_{\min}$):** 59.8 Hz. (60 Hz nominal; normal band 59.5–60.5 Hz; we use 59.8 for stricter constraint.)

---

## 4. The Experiment-to-Observation Pipeline

This section details how a probe $\xi$ is transformed into a scalar observation $y$. The process involves three mapping stages: **Dynamics ($\Phi$)**, **Sampling ($\mathcal{S}$)**, and **Feature Extraction ($\Psi$)**.

### 4.1 Pipeline Overview

```text
       [1. Dynamics]               [2. Sampling]              [3. Feature Extraction]
      (Continuous ODE)           (Discrete Measurement)           (Max-Pooling)

 u(t) ---> [ SYSTEM ] -- w(t) --> [ PMU SENSOR ] -- f[n] --> [ CALCULATE ROCOF ] -- y -->
             ^                         ^                            ^
             |                         |                            |
        Parameters (M,K)          Noise (eta)                  Max(|df/dt|)
```

### 4.2 Dynamics, Sampling, and Feature Extraction

**Stage 1: Dynamics (The Solution Map $\Phi$)**
Given parameters $\vartheta$ and probe $\xi$, we solve the ODE system to get the continuous angular frequency trajectory $\omega(t)$.
$$
\mathbf{x}(t) = \Phi_t(\mathbf{x}_0, \vartheta, u^{\mathrm{probe}}_{\xi}) \quad \mathrm{for}\; t \in [0, T_{\mathrm{obs}}]
$$

**Stage 2: Discrete Sampling (The Measurement Map $\mathcal{S}$)**
We sample the frequency deviation $\Delta f_i(t) = \omega_i(t)/2\pi$ at $f_s = 12$ Hz.
$$
\tilde{f}_i[n] = \Delta f_i(n \cdot \Delta t) + \eta_n, \quad \Delta t = 1/f_s
$$
* $\eta_n$: Measurement noise (negligible).

**Stage 3: Feature Extraction (The Reduction Map $\Psi$)**
We compute the discrete Rate of Change of Frequency (ROCOF) and pool it into a single scalar statistic $y$.
1.  **Finite Difference:** $\mathrm{ROCOF}_i[n] = (\tilde{f}_i[n] - \tilde{f}_i[n-1]) / \Delta t$
2.  **Max-Pooling:**

$$
y = \Psi(\tilde{\mathbf{f}}) = \max_{i, n} \lvert \mathrm{ROCOF}_i[n] \rvert
$$

### 4.3 The Forward Model $\mathcal{M}$
We define the composite forward model $\mathcal{M}(\vartheta, \xi)$ as the deterministic output of this entire pipeline (excluding noise). The final observation $y$ is:
$$
y = \mathcal{M}(\vartheta, \xi) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_{feat}^2)
$$
* **$\sigma_{feat}$:** 0.05 Hz/s (Aggregate uncertainty from numerical error and PMU noise).

---

## 5. Bayesian Inference and MOCU Objective

This section defines how the observation $y$ is used to update our belief about $\vartheta$ and how we quantify the uncertainty in the decision $\gamma^*$.

### 5.1 Likelihood Function
The likelihood describes the probability of observing a specific peak ROCOF value $y$ given a hypothesized parameter set $\vartheta$ and the applied probe $\xi$. Since we model the error as Gaussian:

$$
p(y \mid \vartheta, \xi) = \frac{1}{\sqrt{2\pi\sigma_{feat}^2}} \exp \left( -\frac{\bigl(y - \mathcal{M}(\vartheta, \xi)\bigr)^2}{2\sigma_{feat}^2} \right)
$$

### 5.2 Posterior Update
We maintain a belief state $p_t(\vartheta)$ (represented by a set of weighted particles). Upon collecting a new observation $y_{obs}$ from experiment $\xi$, the belief is updated via Bayes' rule:

$$
p_{t+1}(\vartheta) = \frac{p(y_{obs} \mid \vartheta, \xi) \cdot p_t(\vartheta)}{\int p(y_{obs} \mid \vartheta', \xi) p_t(\vartheta') \, d\vartheta'}
$$

### 5.3 Mean Objective Cost of Uncertainty (MOCU)

**1. The Bayes-Optimal Decision $\hat{\gamma}^*$**
Given a belief $p(\vartheta)$, the optimal estimator for the safe gain is the one that minimizes the expected loss. For absolute error loss ($L_1$), this is the **median** of the predicted safe gains:
$$
\hat{\gamma}^*(p) = \mathrm{median}_{\vartheta \sim p} [\gamma^*(\vartheta)]
$$

**2. MOCU (Current Uncertainty)**
The MOCU $J(p)$ quantifies the expected decision error we would incur if we stopped experimenting now.
$$
J(p) = \mathbb{E}_{\vartheta \sim p} \left[ \lvert \gamma^*(\vartheta) - \hat{\gamma}^*(p) \rvert \right]
$$

**3. Expected Remaining MOCU (The Design Objective)**
To select the optimal next probe $\xi^*$, we calculate the expected MOCU after the experiment. This is the **risk function** $\mathcal{R}(\xi)$ we minimize:

$$
\mathcal{R}(\xi; p_t) = \mathbb{E}_{y \sim p(y \mid p_t, \xi)} \left[ J\left( \mathrm{Posterior}(p_t, \xi, y) \right) \right]
$$

* **Inner term:** The MOCU of the hypothetical future posterior.
* **Outer expectation:** Averaged over all possible observation outcomes $y$ predicted by the current prior.



## 6. Deep Adaptive Design (DAD) Framework

We implement **Deep Adaptive Design** to amortize the cost of finding optimal experiments. Instead of optimizing $\xi$ via gradient descent at runtime (which is slow), we train a **policy network** $\pi_\phi$.

### 6.1 Design Policy
The policy $\pi_\phi$ maps the current experiment history $h_{t-1}$ to the next optimal design:
$$
\xi_t = \pi_\phi(h_{t-1})
$$
* **Input:** History embedding (encoding previous probes $\xi_{1:t-1}$ and observations $y_{1:t-1}$).
* **Output:** Distribution over candidate buses $\mathcal{B}$ and amplitudes $\mathcal{A}$.

### 6.2 Optimization Objective
The network is trained to minimize the **Terminal MOCU** over the entire experimental horizon $T$. We find parameters $\phi^*$ such that:
$$
\phi^* = \arg\min_\phi \mathbb{E}_{\vartheta \sim p(\vartheta),\; y_{1:T} \sim p(y \mid \vartheta, \pi_\phi)} \left[ J(p_T) \right]
$$
This end-to-end objective ensures the policy learns **non-myopic** strategies (e.g., probing different areas of the grid to disambiguate coupled parameters).

---

## 7. Fast MOCU Estimation via MPNN

Calculating the true MOCU $J(p)$ requires integrating over the expensive $\gamma^*(\vartheta)$ landscape (which involves binary search over ODE solutions). To accelerate training, we replace this with a neural surrogate.

### 7.1 The MPNN Estimator
We use a **Message Passing Neural Network (MPNN)** that leverages the graph structure of the IEEE-14 bus system ($B_{ij}$) to estimate MOCU directly.

$$
\hat{J}(p_t) \approx \mathrm{MPNN}_{\psi}(\mathrm{State}_t, \mathcal{G})
$$

* **Graph Input ($\mathcal{G}$):** Admittance matrix nodes and edges.
* **State Input ($\mathrm{State}_t$):** Summary statistics of the current belief $p_t$ (e.g., bounds or moments of marginal distributions for $M_i, K_i$).
* **Output:** Predicted scalar MOCU value.

### 7.2 Training the Surrogate
The MPNN is pre-trained or co-trained via supervised learning to match the ground-truth MOCU computed by the physics simulator:
$$
\mathcal{L}_{\psi} = \left\lVert \hat{J}_{\mathrm{MPNN}}(p) - J_{\mathrm{Physics}}(p) \right\rVert^2
$$

---

## 8. Sequential Execution Loop

The complete **sBOED** loop proceeds as follows for $t = 1$ to $T_{horizon}$:

1.  **Policy Step:** The DAD network observes history $h_{t-1}$ and outputs $\xi_t$.
2.  **Experiment:** Execute $\xi_t$ on the system, measure $y_t$.
3.  **Inference:** Update belief $p_t(\vartheta)$ using the likelihood $p(y_t \mid \vartheta, \xi_t)$.
4.  **Evaluation:** (Optional) Estimate current MOCU using the MPNN for progress tracking.

---

## 9. Probe and Observation Parameters

| Parameter | Symbol | Value / Set | Justification |
|-----------|--------|-------------|----------------|
| Probe location | $b_t$ | $\mathcal{B} \subset \lbrace 1,\dots,14 \rbrace$ | Buses with IBR actuation. |
| Probe amplitude | $A_t$ | $\lbrace 0.05, 0.1, 0.2 \rbrace$ | ROCOF above PMU noise. |
| Probe duration | $T_p$ | 2 s | Excites inertial dynamics. |
| Sampling rate | $f_s$ | 12 Hz | PMU/ROCOF reporting standards. |
| Observation window | $T_{\mathrm{obs}}$ | [0,10] s | Captures ROCOF peak. |

---

## References

[1] P. Kundur, *Power System Stability and Control*. McGraw-Hill, 1994.

[2] F. Dörfler and F. Bullo, *Automatica*, 2014.

[3] J. Peng et al., *NREL/CP-5D00-87925*, 2024.

[4] Foster et al., "Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design," *ICML*, 2021.

[5] ENTSO-E, "Inertia and rate of change of frequency (RoCoF)," 2020.