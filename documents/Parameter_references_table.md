# Final Parameter Reference Table (Implementation-Consistent)

## Modeling Note
**$M$ is derived from the inertia constant $H$ (seconds):** $M = 2H/\omega_s$ with $\omega_s = 2\pi f_0$. At 60 Hz, $\omega_s \approx 377$ rad/s, so $M \approx 2H/377$. Literature gives **$H$** (e.g. 2–7 s for synchronous machines); the **$M$ range used in the project** is the physical range $[0.01,\, 0.06]$ $s^2/\text{rad}$ corresponding to $H \in [2.3,\, 5.0]$ s (or slightly wider for $H \in [2,\, 11]$ s).

| Parameter | Symbol | Value Used | APA Citation & Reference |
|:----------|:---|:------------|:---|
| **Nominal Frequency** | $f_0$ | 60 Hz | Texas A&M University. (n.d.). *IEEE 14-bus system*. Electric Grid Test Cases. [https://electricgrids.engr.tamu.edu/electric-grid-test-cases/ieee-14-bus-system/](https://electricgrids.engr.tamu.edu/electric-grid-test-cases/ieee-14-bus-system/) |
| **Minimum Frequency** | $f_{\min}$ | 59.8 Hz | ENTSO-E (50 Hz band 49.5–50.5); NERC/60 Hz analogous. We use 59.8 (stricter than 59.5). [https://www.entsoe.eu/Documents/Network%20codes%20documents/NC%20RfG/IGD_Frequency_ranges_final.pdf](https://www.entsoe.eu/Documents/Network%20codes%20documents/NC%20RfG/IGD_Frequency_ranges_final.pdf) |
| **ROCOF Limit** | $r_{\max}$ | 0.1 Hz/s | ENTSO-E IGD RoCoF (withstand higher); we use 0.1 for non-trivial control. [https://www.entsoe.eu/Documents/Network%20codes%20documents/NC%20RfG/IGD_RoCoF_withstand_capability_final.pdf](https://www.entsoe.eu/Documents/Network%20codes%20documents/NC%20RfG/IGD_RoCoF_withstand_capability_final.pdf) |
| **Inertia Range** | $M$ | [0.01, 0.06] $s^2/\text{rad}$ | From $M = 2H/\omega_s$; $H \in [2.3,\, 5.0]$ s (Kundur, typical machines). Kundur, P. (1994). *Power System Stability and Control*. McGraw-Hill. |
| **Droop / primary frequency response gain** | $K$ | [0.05, 0.50] p.u. | Dörfler & Bullo (2012); NERC/ERCOT droop 4–6%. arXiv:0910.5673. [https://arxiv.org/pdf/0910.5673](https://arxiv.org/pdf/0910.5673) |
| **Damping** | $D_i$ | 0.1 p.u. | Per-bus load-damping (homogeneous). Kundur, P. (1994). *Power System Stability and Control*. McGraw-Hill. |
| **Sampling Rate** | $f_s$ | 12 Hz | IEEE C37.118.1 (60 Hz: reporting rates 10, 12, 15, 20, 30, 60 per second). [https://standards.ieee.org/ieee/C37.118.1/4902](https://standards.ieee.org/ieee/C37.118.1/4902) |
| **Probe Duration** | $T_p$ | 2.0 s | Peng, J., et al. (2024). NREL/CP-5D00-87925. [https://docs.nrel.gov/docs/fy24osti/87925.pdf](https://docs.nrel.gov/docs/fy24osti/87925.pdf) |
| **Probe Amplitude** | $A$ | [0.05, 0.1, 0.2] (tests up to 0.5) | Peng et al. (2024), NREL/CP-5D00-87925. [https://docs.nrel.gov/docs/fy24osti/87925.pdf](https://docs.nrel.gov/docs/fy24osti/87925.pdf) |
| **Observation Window** | $T_{\mathrm{obs}}$ | [0, 10] s | ENTSO-E RoCoF; config `T_obs_sec: 10.0`. Same link as ROCOF. |
| **Likelihood / observation noise** | $\sigma_{\mathrm{feat}}$ | 0.05 Hz/s (design); 0.01 in balanced config | Standard deviation of observation $y$ (ROCOF). Design §4.3. NASPI (2021). [https://www.naspi.org/node/899](https://www.naspi.org/node/899) |