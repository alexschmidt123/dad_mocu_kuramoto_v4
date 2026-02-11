# Final Parameter Reference Table (Implementation-Consistent)

## Modeling Note
**$M$ is derived from the inertia constant $H$ (seconds):** $M = 2H/\omega_s$ with $\omega_s = 2\pi f_0$. At 50 Hz (aligned with MATLAB .mdl / ENTSO-E), $\omega_s \approx 314$ rad/s, so $M \approx 2H/314$. Literature gives **$H$** (e.g. 2–7 s for synchronous machines); the **$M$ range used in the project** is the physical range $[0.01,\, 0.06]$ $s^2/\text{rad}$ corresponding to $H \in [2.3,\, 5.0]$ s (or slightly wider for $H \in [2,\, 11]$ s).

<table>
<colgroup>
<col style="width: 28%">
<col style="width: 10%">
<col style="width: 28%">
<col style="width: 34%">
</colgroup>
<thead>
<tr><th>Parameter</th><th>Symbol</th><th>Value Used</th><th>APA Citation & Reference</th></tr>
</thead>
<tbody>
<tr><td><strong>Nominal Frequency</strong></td><td>$f_0$</td><td>50 Hz</td><td>Aligned with MATLAB IEEE 14-bus .mdl and ENTSO-E (50 Hz systems). Texas A&M IEEE 14-bus test case; frequency choice per validation target.</td></tr>
<tr><td><strong>Minimum Frequency</strong></td><td>$f_{\min}$</td><td>49.8 Hz</td><td>ENTSO-E 50 Hz band 49.5–50.5; we use 49.8 (stricter than 49.5). <a href="https://www.entsoe.eu/Documents/Network%20codes%20documents/NC%20RfG/IGD_Frequency_ranges_final.pdf">link</a></td></tr>
<tr><td><strong>ROCOF Limit</strong></td><td>$r_{\max}$</td><td>0.1 Hz/s</td><td>ENTSO-E IGD RoCoF (withstand higher); we use 0.1 for non-trivial control. <a href="https://www.entsoe.eu/Documents/Network%20codes%20documents/NC%20RfG/IGD_RoCoF_withstand_capability_final.pdf">link</a></td></tr>
<tr><td><strong>Inertia Range</strong></td><td>$M$</td><td>[0.01, 0.06] $s^2/\text{rad}$</td><td>From $M = 2H/\omega_s$; $H \in [2.3,\, 5.0]$ s (Kundur, typical machines). Kundur, P. (1994). <em>Power System Stability and Control</em>. McGraw-Hill.</td></tr>
<tr><td><strong>Droop / primary frequency response gain</strong></td><td>$K$</td><td>[0.05, 0.50] p.u.</td><td>Dörfler & Bullo (2012); NERC/ERCOT droop 4–6%. arXiv:0910.5673. <a href="https://arxiv.org/pdf/0910.5673">link</a></td></tr>
<tr><td><strong>Damping</strong></td><td>$D_i$</td><td>0.1 p.u.</td><td>Per-bus load-damping (homogeneous). Kundur, P. (1994). <em>Power System Stability and Control</em>. McGraw-Hill.</td></tr>
<tr><td><strong>Sampling Rate</strong></td><td>$f_s$</td><td>12 Hz</td><td>IEEE C37.118.1 (reporting rates 10, 12, 15, 20, 30, 60 per second). <a href="https://standards.ieee.org/ieee/C37.118.1/4902">link</a></td></tr>
<tr><td><strong>Probe Duration</strong></td><td>$T_p$</td><td>2.0 s</td><td>Peng, J., et al. (2024). NREL/CP-5D00-87925. <a href="https://docs.nrel.gov/docs/fy24osti/87925.pdf">link</a></td></tr>
<tr><td><strong>Probe Amplitude</strong></td><td>$A$</td><td>[0.05, 0.1, 0.2] (tests up to 0.5)</td><td>Peng et al. (2024), NREL/CP-5D00-87925. <a href="https://docs.nrel.gov/docs/fy24osti/87925.pdf">link</a></td></tr>
<tr><td><strong>Observation Window</strong></td><td>$T_{\mathrm{obs}}$</td><td>[0, 10] s</td><td>ENTSO-E RoCOF; config <code>T_obs_sec: 10.0</code>. Same link as ROCOF.</td></tr>
<tr><td><strong>Likelihood / observation noise</strong></td><td>$\sigma_{\mathrm{feat}}$</td><td>0.05 Hz/s (design); 0.01 in balanced config</td><td>Standard deviation of observation $y$ (ROCOF). Design §4.3. NASPI (2021). <a href="https://www.naspi.org/node/899">link</a></td></tr>
</tbody>
</table>
