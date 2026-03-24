# DAD-MOCU: Design Document

**Sequential optimal experimental design for power systems** · *Gaoming Lin · Dr. Byung-Jun Yoon · Jan 2026*

Prototype for the project paper. **Keep `design_github.md` in sync.**

---

## 1. Goal

**sBOED + DAD** on IEEE-14 swing (second-order Kuramoto) dynamics. Reduce uncertainty in a **decision quantity**—the **minimum safe supplementary gain** $\gamma^\ast(\vartheta)$—not $(M,K)$ for their own sake. Utility: **MOCU**; policy: amortized **$\pi_\phi$** (Foster et al., 2021).

**Chain:** $\xi_t \to y_t \to p_t(\vartheta) \to$ uncertainty in $\gamma^\ast(\vartheta) \to \mathrm{MOCU}(p_t)$. Full math: `documents/pseudocode.tex`.

---

## 2. Pipeline

Grid → swing ODE → likelihood $\mathrm{Map}(\vartheta,\xi)$ → posterior → $\gamma^\ast$ → MOCU → (optional) MPNN surrogate → DAD policy.

```mermaid
flowchart LR
  G[IEEE-14 / Sim] --> ODE[Swing ODE]
  ODE --> L[p(y|ϑ,ξ)]
  ODE --> GS[γ* search]
  GS --> M[MOCU]
  M --> DAD[DAD π_φ]
  L --> DAD
```

---

## 3. Model (brief)

- **Topology:** Standard IEEE-14; susceptance $\mathbf{B}$; homogeneous uncertain $\vartheta=(M,K)$ with $M_i=M$, $K_i=K$.
- **Swing (per bus):** $\dot{\theta}_i=\omega_i$;  
  $M_i\dot{\omega}_i = P_{m,i}-P_{e,i}(\boldsymbol{\theta})-(D_i+K_i)\omega_i + u_i$ with  
  $P_{e,i}=\sum_{j\in\mathcal{N}_i} B_{ij}\sin(\theta_i-\theta_j)$.
- **Injection:** $u_i = u^{\mathrm{probe}}_i + u^{\mathrm{ctrl}}_i$. **Supplementary** control $u^{\mathrm{ctrl}}_i = \gamma\, g_i\, \omega_i$ (implementation sign/convention must match code).

**Terminology (aligned with `pseudocode.tex`):**

- **Droop $K$** (with $H$ in $\vartheta$): **latent plant parameters**—coefficients in the swing ODE, **not** control variables you set or learn exactly after finite data.
- **$\gamma$** and **$u^{\mathrm{ctrl}}$**: **supplementary** control for **evaluation** / security (minimum threshold $\gamma^\ast(\vartheta)$, MOCU on $\gamma^\ast$). This is the closest analogue to the **control-oscillator coupling** $a_{N+1}$ in uncertain Kuramoto OED (Chen et al., 2023; Hong et al., 2021), **not** $K$.
- **$\xi$**: **design of experiments** (probe)—**not** “primary control.”
- **Avoid** calling **$K$** “primary control”: that phrase suggests a precisely known or chosen input; **$K$** stays uncertain.

**Two uses (do not mix):**

| Mode | Probe | Control | Purpose |
|------|--------|---------|---------|
| **Learning (sBOED)** | From design $\xi_t=(b,A,T_p)$ | Off ($\gamma=0$) | $y_t=\mathrm{Map}(\vartheta,\xi_t)+\epsilon$ |
| **Evaluation ($\gamma^\ast$)** | **Not** the learning probe $\xi_t$; fixed **evaluation scenario** $d_{\mathrm{ref}}$ (contingency / reference disturbance in code) | Candidate $\gamma$ | Security limits on $[0,T_c]$ |

*Pseudocode note:* `pseudocode.tex` defines $\gamma^\ast$ from **evaluation dynamics under fixed $d_{\mathrm{ref}}$** without mixing in the identification probe. Implementation must apply a **non-trivial** contingency when computing $\gamma^\ast$ or MOCU degenerates.

**Prior:** Uniform on $M\in[0.01,0.06]$ $s^2$/rad, $K\in[0.05,0.50]$ p.u. (see `Parameter_references_table.md`).

---

## 4. Minimal safe gain $\gamma^\ast$

For $\vartheta$ fixed, $\gamma^\ast(\vartheta)$ is the smallest $\gamma\ge 0$ such that on $[0,T_c]$ (reference bus or worst over buses):

$$\sup_t |\dot f(t)| \le r_{\max}, \qquad f(t) \ge f_{\min}.$$

Typical: $r_{\max}=0.1$ Hz/s; $f_{\min}$ e.g. 49.8 Hz @ 50 Hz nominal. Under monotonicity (bisection in `pseudocode` Algorithm 2), $\gamma\in[\gamma^\ast(\vartheta),\infty)$ is safe; often $\gamma^\ast=\max\{\gamma_{\mathrm{ROCOF}},\gamma_{\mathrm{freq}}\}$ from the two constraints. **No closed form** in $(M,K,r_{\max},f_{\min})$—**simulate trajectory**, check constraints, bracket, bisect.

---

## 5. Observation → Bayes → MOCU

- **Feature:** Hann probe at $b$; solve ODE; sample $\Delta f$ at $f_s{=}12$ Hz; $y=\max_{i,n}|\mathrm{ROCOF}_i[n]|$.
- **Likelihood:** $y=\mathrm{Map}(\vartheta,\xi)+\epsilon$, $\epsilon\sim\mathcal{N}(0,\sigma_{\mathrm{feat}}^2)$ (default $\sigma_{\mathrm{feat}}=0.05$ Hz/s).
- **Posterior:** $p(\vartheta\mid y,\xi)\propto p(y\mid\vartheta,\xi)p(\vartheta)$. Sequential: $p_t\propto p(y_t\mid\vartheta,\xi_t)p_{t-1}$. Discrete grid: log-likelihood → log-sum-exp normalize (see pseudocode §Posterior).

- **Cost:** $J(\gamma,\vartheta)=|\gamma-\gamma^\ast(\vartheta)|$. Bayes action under $L_1$: **median** $\hat\gamma$ of $\gamma^\ast(\vartheta)$ under $p$.
- **MOCU:** $\mathrm{MOCU}(p)=\mathbb{E}_{\vartheta\sim p}[|\gamma^\ast(\vartheta)-\hat\gamma|]$ (Boluki et al., 2018; Imani et al., 2018).

**Myopic design risk** (aligned with `pseudocode.tex`): choose $\xi$ to minimize  
$\mathbb{E}_{y_t \mid p_{t-1},\xi}\big[\mathrm{MOCU}(p_t)\big]$  
where $p_t$ is the posterior **after** observing $y_t$ under $\xi$ (equivalently $\mathrm{MOCU}(p_{t+1})$ if you index the post-update belief as $p_{t+1}$).

**DAD training:** $\phi^\ast=\arg\min_\phi \mathbb{E}[\mathrm{MOCU}(p_T)]$ over episodes (terminal MOCU).

---

## 6. MPNN (optional acceleration)

$\widehat{\mathrm{MOCU}}(p)\approx \mathrm{MPNN}_\psi(\mathrm{state}_t,\mathcal{G})$ trained with MSE to physics $\mathrm{MOCU}(p)$. *Contribution:* value surrogate on graph; validate vs exact MOCU and policy quality.

---

## 7. Episode loop

For $t=1,\ldots,T$: $\xi_t\sim\pi_\phi(h_{t-1})$ → run probe → $y_t$ → update $p_t$ → (optional) track $\widehat{\mathrm{MOCU}}$. Algorithms: `pseudocode.tex` Alg. 1 (sBOED loop), Alg. 2 ($\gamma^\ast$).

---

## 8. Default probe / observation settings

| Item | Value |
|------|--------|
| $\xi$ | $(b,A,T_p)$, $T_p=2$ s |
| $f_s$ | 12 Hz |
| $T_{\mathrm{obs}}$ | e.g. [0,10] s |
| $\sigma_{\mathrm{feat}}$ | 0.05 Hz/s |

---

## 9. Standalone validation (subfolders under `tests/`)

**Math (canonical):** `src/core/discrete_bayes.py` — Gaussian likelihood, log-sum-exp posterior (pseudocode §Likelihood, §Posterior), **MOCU** = $\sum_n p^n|\gamma^\star_n-\hat\gamma|$ with $\hat\gamma$ a weighted median (eq.~mocu\_gamma). **`compute_mocu`** in `src/core/mocu_particles.py` uses the same definition. **Backends:** `swing_equation_mocu.py` (Torch MC), `mocu_pycuda.py` (PyCUDA + CUDA C++), `mocu_torchdiffeq.py` (helpers).

| Folder | Content |
|--------|---------|
| **`tests/posterior_inference/`** | Posterior / MOCU: tests `unit/`, `integration/`, `conftest.py`, `episode_helpers.py`; outputs under `tests/posterior_inference/output/`. |
| **`tests/simulink_reference/`** | Python swing ODE vs Simulink/MATLAB reference: `ode_validation.py`, `test_simulink_reference.py`; scratch under `tests/simulink_reference/output/`. |

**Run:** `pytest tests/posterior_inference tests/simulink_reference -v` · `python -m tests.simulink_reference.ode_validation`.

---

## References (short)

Boluki et al. (2018); Chen et al. (2023); Hong et al. (2021); Imani et al. (2018); Foster et al. (2021); Dörfler & Bullo (2012); Kundur (1994); Peng et al. (2024 NREL); ENTSO-E; NASPI; IEEE C37.118; Texas A&M IEEE-14 test case.
