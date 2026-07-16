# DAD training-loop audit

Scope: existing random-initialized objective-based DAD implementation in
`src/neural/train.py`, plus the isolated corrected study runner in
`src/control/dad_rl_training_improvement.py`.

## Required-loop checks

| Check | Existing implementation | Evidence and disposition |
|---|---|---|
| Same `theta*` across all `T=3` steps | Pass | `_policy_rollout` receives one `sys` and reuses it for the entire loop. The caller samples `sys` once per trajectory. |
| Complete history supplied to policy | Pass | `act_h` and `obs_h` accumulate every prior `(action, observation)` pair and are passed together at each decision. `HistoryEncoder` encodes all pairs. |
| Terminal `u_ctrl` calculated after `T` | Pass | `_terminal_u_ctrl` is called only after `_policy_rollout` returns the complete sequence and observation list. The posterior likelihood multiplies contributions from every pair. |
| Every selected-action log probability included | Pass | Each step appends its selected-action log probability; training uses `log_ps.sum()` in the terminal REINFORCE loss. |
| Gradient reaches policy parameters | Pass | The selected-action log probabilities remain attached to the policy graph, `backward()` is called, and measured smoke-test gradient norms are nonzero. |
| No test/confirmation data enters training | Pass for the controlled runner | Training systems are support plus calibration systems, checkpoint selection uses validation systems, and confirmation systems are loaded only after the final configuration is frozen. |

## Material discrepancy found and fixed for this study

The existing core rollout used the clean bank response directly:

`y = lookup_action_y(sys, a_idx)`

That does not implement the required training observation model
`y = Y_bank[theta*, action] + epsilon`. The new isolated runner samples
Gaussian observation noise with the validated `sigma_y` for every training
observation while continuing to use only the offline bank. Validation and
confirmation use deterministic keyed Gaussian noise.

The corrected runner also explicitly seeds PyTorch per requested seed, saves
every periodically evaluated checkpoint, and writes `best_checkpoint.pt` only
when validation safety is exactly 1.0 and validation mean `u_ctrl` improves.

## Objective and leakage assertions

- Policy initialization is random for every seed.
- Fixed actions are never used as labels, initialization, forced actions, or
  policy inputs.
- The only policy objective is expected terminal posterior-safe `u_ctrl`.
- Entropy is an optimization regularizer for exploration, not a scientific
  reward.
- Baselines use only batch costs or complete past history; they do not receive
  true hidden `theta`, true `U_req`, future observations, or confirmation IDs.
- Potential rewards use posterior-bank `u_ctrl(h_{t-1}) - u_ctrl(h_t)` and are
  checked numerically for telescoping on every training batch.
- No physical simulation or ODE solve is performed during DAD, Fixed, Myopic,
  or Random execution. Safety uses the validated offline `u_req` bank.
