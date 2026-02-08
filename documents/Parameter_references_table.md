# Final Parameter Reference Table (Model-Consistent)

## Modeling Note
The swing equation used in this project is implemented in **normalized form**. The inertia parameter **$M$** is therefore a *dimensionless effective inertia coefficient*, not the physical inertia constant **$H$** (seconds). A standard physical interpretation (when needed) uses:

$$
M = \frac{2H}{\omega_s}, \qquad \omega_s = 2\pi f_0.
$$

---

## Parameters

### $f_0$ (nominal frequency, Hz)
- **Value used in project:** $60$  
- **Reference interpretation:** $60$ Hz nominal (USA / North America). IEEE-14 is used as topology.  
- **Reference:** IEEE PES / Texas A\&M University (test case)  
- **Link:** https://electricgrids.engr.tamu.edu/electric-grid-test-cases/ieee-14-bus-system/

---

### $f_{\min}$ (minimum frequency, Hz)
- **Value used in project:** $59.8$  
- **Reference interpretation:** Normal operation band $59.5$–$60.5$ Hz (60 Hz systems); we use $59.8$ (stricter than $59.5$)  
- **Reference:** NERC/ERCOT (60 Hz); analogous to ENTSO-E band for 50 Hz  
- **Link:** https://www.entsoe.eu/Documents/Network%20codes%20documents/NC%20RfG/IGD_Frequency_ranges_final.pdf (50 Hz); NERC for 60 Hz

---

### ROCOF limit (Hz/s)
- **Value used in project:** $0.1$  
- **Reference interpretation:** ROCOF withstand capability $\pm 2.0$ Hz/s; minimum detectable ROCOF $\approx 0.1$–$0.2$ Hz/s  
- **Reference:** ENTSO-E IGD ROCOF Withstand Capability  
- **Link:** https://www.entsoe.eu/Documents/Network%20codes%20documents/NC%20RfG/IGD_RoCoF_withstand_capability_final.pdf

---

### $f_s$ (PMU sampling rate, Hz)
- **Value used in project:** $12$  
- **Reference interpretation:** PMU reporting rates typically $10$–$60$ Hz for 60-Hz systems  
- **Reference:** IEEE Standard C37.118.1  
- **Link:** https://standards.ieee.org/ieee/C37.118.1/4902

---

### $M$ (normalized inertia coefficient)
- **Value used in project:** $[0.3,\;2.0]$  
- **Reference interpretation:** Dimensionless inertia coefficient under normalized swing-equation formulation. Physical inertia constant $H$ (seconds) relates via $M = 2H/\omega_s$. Typical physical inertia constants:  
  - $H = 1$–$3$ s (low-inertia systems)  
  - $H = 2$–$7$ s (synchronous machines)  
  - $H = 2$–$9$ s (traditional grids)  
- **Reference:** Kundur (1994), *Power System Stability and Control*  
- **Link:** https://books.google.com/books?id=wOlSAAAAMAAJ

---

### $D$ (aggregate damping)
- **Value used in project:** $0.1$  
- **Reference interpretation:** Normalized aggregate damping representing load frequency sensitivity and primary control; commonly fixed in swing-equation models  
- **Reference:** Kundur (1994)  
- **Link:** https://books.google.com/books?id=wOlSAAAAMAAJ

---

### $K$ (electrical coupling gain)
- **Value used in project:** $[0.05,\;0.50]$  
- **Reference interpretation:** Dimensionless global scaling factor multiplying normalized network susceptance; treated as uncertain model parameter  
- **References:**  
  - Dörfler \& Bullo (2012)  
  - Rodrigues et al. (2016)  
- **Links:**  
  - https://arxiv.org/pdf/0910.5673  
  - https://arxiv.org/pdf/1511.07139

---

### $P_{m,i}$ (mechanical power injection, pu)
- **Value used in project:** Fixed (IEEE-14 operating point)  
- **Reference interpretation:** Generator and load injections obtained from IEEE-14 power-flow solution  
- **References:** Zimmerman et al. (2011); Texas A\&M University  
- **Links:**  
  - https://ieeexplore.ieee.org/document/5491276  
  - https://electricgrids.engr.tamu.edu/electric-grid-test-cases/ieee-14-bus-system/

---

### Probe duration $T_p$ (seconds)
- **Value used in project:** $2$  
- **Reference interpretation:** Typical probing windows of $1$–$3$ s validated via PHIL and PMU experiments  
- **Reference:** Peng et al. (2024), NREL  
- **Link:** https://docs.nrel.gov/docs/fy24osti/87925.pdf

---

### Probe shape
- **Value used in project:** Hann window  
- **Reference interpretation:** Smooth, band-limited probing signal to avoid spectral leakage  
- **Reference:** Peng et al. (2024)  
- **Link:** https://docs.nrel.gov/docs/fy24osti/87925.pdf

---

### Probe amplitude $A$
- **Value used in project:** $[0.05,\;0.1,\;0.2]$ (tests up to $0.5$)  
- **Reference interpretation:** Small active-power perturbations chosen above noise floor and below security limits; MW-level injections used in probing-based inertia estimation  
- **Reference:** Peng et al. (2024)  
- **Link:** https://docs.nrel.gov/docs/fy24osti/87925.pdf

---

### Observation window $T_{\mathrm{obs}}$ (seconds)
- **Value used in project:** $[0,\;10]$  
- **Reference interpretation:** ROCOF computed over $0.5$–$1$ s windows in grid codes; longer windows used for estimation robustness  
- **Reference:** ENTSO-E IGD ROCOF  
- **Link:** https://www.entsoe.eu/Documents/Network%20codes%20documents/NC%20RfG/IGD_RoCoF_withstand_capability_final.pdf

---

### Observation
- **Value used in project:** $\mathrm{ROCOF}_{\max}$  
- **Reference interpretation:** ROCOF is a primary observable for inertia and frequency-response assessment  
- **Reference:** ENTSO-E  
- **Link:** https://www.entsoe.eu/Documents/Network%20codes%20documents/NC%20RfG/IGD_RoCoF_withstand_capability_final.pdf

---

### Network
- **Value used in project:** IEEE-14  
- **Reference interpretation:** Standard IEEE-14 bus test system (14 buses, 5 generators)  
- **Reference:** IEEE PES / Texas A\&M University  
- **Link:** https://electricgrids.engr.tamu.edu/electric-grid-test-cases/ieee-14-bus-system/

---

### $\sigma$ (likelihood noise standard deviation)
- **Value used in project:** $0.01$ (configuration); $0.05$ (tests)  
- **Reference interpretation:** No standard ROCOF variance specified; represents PMU measurement and signal-processing uncertainty  
- **References:** IEEE C37.118.1; NASPI PMU performance guidance  
- **Links:**  
  - https://standards.ieee.org/ieee/C37.118.1/4902  
  - https://www.naspi.org/node/899

---

## Should I change my tests' and project's settings?

**60 Hz nominal (USA):**  
The project uses **$f_0 = 60$ Hz** and **$f_{\min} = 59.8$ Hz** throughout: configs (`f_min: 59.8`), `swing_equation_ode.py` (`f_nominal = 60.0`), and all script defaults (59.5 when config not loaded). Other values in this table are unchanged: $M \in [0.3, 2.0]$, $K \in [0.05, 0.50]$, $D = 0.1$, $r_{\max} = 0.1$ Hz/s, $f_s = 12$ Hz, $T_p = 2$ s, $T_{\mathrm{obs}} = [0, 10]$ s, $\sigma = 0.01$ (config) / $0.05$ (tests), probe amplitudes $[0.05, 0.1, 0.2]$.

---

