# U-bank and continuous u_ctrl validity audit

**U_n nature:** `B_discrete_grid_selected`

For each support particle θ_n, U_n = u_req(θ_n) is the smallest value on the discrete control candidate grid that satisfies ROCOF and frequency-nadir constraints under the configured contingency (see src/control/banks.py generate_control_bank_for_split / u_req_for_theta).

**Continuous u_ctrl status:** approximation_based_on_discrete_U_bank

U_n itself is a discrete-grid safe injection level. Continuous u_ctrl = Q_{1-α}(U|w) + margin interpolates between those banked discrete levels in posterior-quantile space. Intermediate continuous values are NOT individually re-simulated against ROCOF/nadir for this study. Safety thresholds (ROCOF, nadir) and calibrated margin are unchanged; only snap_up is removed from the terminal selector.

Safety thresholds are not weakened. Margin/α reused from the frozen historical rule; only snap_up is disabled for this experiment version.
