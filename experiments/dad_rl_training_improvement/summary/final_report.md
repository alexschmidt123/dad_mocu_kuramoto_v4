# Random-initialized DAD RL training improvement

The scientific objective remained `min E[u_ctrl(H_T)]`. Fixed, Myopic, and Random were comparison baselines only; no Fixed behavior cloning or Fixed initialization was used.

## Existing-loop audit

- One theta is retained across all three steps: yes.
- The policy receives the complete action-observation history: yes.
- Terminal u_ctrl is calculated after all three probes: yes.
- Every selected-action log probability enters the policy loss: yes.
- Gradients reach policy parameters: verified by nonzero measured gradient norms.
- Test/confirmation systems enter training or checkpoint selection: no.
- Original discrepancy: the prior core rollout used clean bank responses during training. This study uses bank response plus Gaussian observation noise.

## IEEE5 T=3

- Original training budget: 50 epochs.
- Selected duration: `B4_10x`.
- Selected entropy schedule: `E0_current`.
- Selected variance method: `R2_batch_baseline`.
- Selected credit method: `C1_potential`.
- Selected learning rate: `LR3_times3` (0.003).
- Gradient clipping threshold: 1.0; selected-LR clipping events across seeds: 2.
- Five-seed validation mean ± std: 0.828333 ± 0.006667 (min 0.816667, max 0.833333).
- Confirmation DAD mean u_ctrl: 0.849300.
- Confirmation Fixed mean u_ctrl: 0.843300.
- Confirmation Myopic mean u_ctrl: 0.857800.
- Confirmation Random mean u_ctrl: 0.875600.

### Duration study

- B1_current: five-seed best validation 0.861667 ± 0.011304; final 0.891667; mean best epoch 29.0; final-minus-best 0.030000.
- B2_2x: five-seed best validation 0.853333 ± 0.006667; final 0.873333; mean best epoch 45.8; final-minus-best 0.020000.
- B3_5x: five-seed best validation 0.855000 ± 0.006667; final 0.883333; mean best epoch 59.0; final-minus-best 0.028333.
- B4_10x: five-seed best validation 0.845000 ± 0.011304; final 0.880000; mean best epoch 128.0; final-minus-best 0.035000.

### Exploration study

- E0_current: best validation 0.845000 ± 0.011304.
- E1_slow_decay: best validation 0.850000 ± 0.009129.
- E2_constant_small: best validation 0.856667 ± 0.003333.
- E3_warmup_decay: best validation 0.855000 ± 0.006667.
- E0 quarter-training entropy: 3.308175 of theoretical feasible-action mean maximum 3.366899; early collapse classification: False.

### Variance reduction

- R0_current: best validation 0.845000 ± 0.011304.
- R1_advantage_normalization: best validation 0.853333 ± 0.015456.
- R2_batch_baseline: best validation 0.843333 ± 0.008165.

### Credit assignment

- Terminal-cost quantization: mean 3.724 unique values/batch, modal fraction 0.451, tied-trajectory fraction 0.771.
- C0_terminal: best validation 0.843333 ± 0.008165.
- C1_potential: best validation 0.838333 ± 0.013540.

### Learning rate

- LR0_current: best validation 0.838333 ± 0.013540.
- LR1_div3: best validation 0.853333 ± 0.011304.
- LR2_div10: best validation 0.853333 ± 0.017951.
- LR3_times3: best validation 0.828333 ± 0.006667.

### Final random-init seed checkpoints

- seed 101: best epoch 65 of 499; best/final validation 0.816667/0.891667; dominant sequence [28, 11, 13] (1.000).
- seed 202: best epoch 0 of 499; best/final validation 0.833333/0.908333; dominant sequence [16, 13, 24] (1.000).
- seed 303: best epoch 100 of 499; best/final validation 0.833333/0.900000; dominant sequence [8, 13, 6] (1.000).
- seed 404: best epoch 25 of 499; best/final validation 0.825000/0.883333; dominant sequence [23, 6, 8] (0.750).
- seed 505: best epoch 30 of 499; best/final validation 0.833333/0.866667; dominant sequence [11, 28, 3] (1.000).

### Per-seed adaptivity

- seed 101: effectively nonadaptive; MI=0; dominant fraction=1.000; unique sequences=1.
- seed 202: effectively nonadaptive; MI=0; dominant fraction=1.000; unique sequences=1.
- seed 303: effectively nonadaptive; MI=0; dominant fraction=1.000; unique sequences=1.
- seed 404: observation-dependent adaptive; MI=0.446032; dominant fraction=0.799; unique sequences=2.
- seed 505: effectively nonadaptive; MI=0; dominant fraction=1.000; unique sequences=1.

### Paired confirmation comparisons

- dad_minus_fixed: mean=0.006000, 95% CI [-0.006760, 0.018740], first lower=0.342, tied=0.302, second lower=0.356; no reliable difference (CI contains zero).
- dad_minus_myopic: mean=-0.008500, 95% CI [-0.013400, -0.003520], first lower=0.339, tied=0.419, second lower=0.242; first method reliably lower.
- dad_minus_random: mean=-0.026300, 95% CI [-0.034760, -0.017720], first lower=0.417, tied=0.347, second lower=0.236; first method reliably lower.
- myopic_minus_fixed: mean=0.014500, 95% CI [0.000900, 0.028100], first lower=0.330, tied=0.242, second lower=0.428; second method reliably lower.
- fixed_minus_random: mean=-0.032300, 95% CI [-0.043400, -0.020800], first lower=0.396, tied=0.335, second lower=0.269; first method reliably lower.
- myopic_minus_random: mean=-0.017800, 95% CI [-0.027200, -0.008200], first lower=0.319, tied=0.451, second lower=0.230; first method reliably lower.

### System conclusion

- DAD vs Fixed: no reliable difference (CI contains zero).
- DAD vs Myopic: first method reliably lower.
- Observation-dependent policies: 1/5.
- All policies fixed-like by dominant-sequence threshold: False.
- Seed stability (validation range): 0.016667.

## IEEE9 T=3

- Original training budget: 50 epochs.
- Selected duration: `B4_10x`.
- Selected entropy schedule: `E2_constant_small`.
- Selected variance method: `R0_current`.
- Selected credit method: `C0_terminal`.
- Selected learning rate: `LR0_current` (0.001).
- Gradient clipping threshold: 1.0; selected-LR clipping events across seeds: 0.
- Five-seed validation mean ± std: 0.967333 ± 0.003266 (min 0.963333, max 0.970000).
- Confirmation DAD mean u_ctrl: 0.974512.
- Confirmation Fixed mean u_ctrl: 0.979120.
- Confirmation Myopic mean u_ctrl: 0.964320.
- Confirmation Random mean u_ctrl: 0.982160.

### Duration study

- B1_current: five-seed best validation 0.970667 ± 0.005333; final 0.982000; mean best epoch 6.0; final-minus-best 0.011333.
- B2_2x: five-seed best validation 0.970000 ± 0.004216; final 0.978667; mean best epoch 9.0; final-minus-best 0.008667.
- B3_5x: five-seed best validation 0.969333 ± 0.003266; final 0.981333; mean best epoch 24.0; final-minus-best 0.012000.
- B4_10x: five-seed best validation 0.968000 ± 0.002667; final 0.974000; mean best epoch 137.0; final-minus-best 0.006000.

### Exploration study

- E0_current: best validation 0.968000 ± 0.002667.
- E1_slow_decay: best validation 0.968667 ± 0.002667.
- E2_constant_small: best validation 0.967333 ± 0.003266.
- E3_warmup_decay: best validation 0.968667 ± 0.002667.
- E0 quarter-training entropy: 3.931286 of theoretical feasible-action mean maximum 3.970173; early collapse classification: False.

### Variance reduction

- R0_current: best validation 0.967333 ± 0.003266.
- R1_advantage_normalization: best validation 0.968000 ± 0.002667.
- R2_batch_baseline: best validation 0.968667 ± 0.002667.
- R3_history_value: best validation 0.967333 ± 0.003266.

### Credit assignment

- Terminal-cost quantization: mean 1.742 unique values/batch, modal fraction 0.894, tied-trajectory fraction 0.919.
- C0_terminal: best validation 0.967333 ± 0.003266.
- C1_potential: best validation 0.968667 ± 0.002667.

### Learning rate

- LR0_current: best validation 0.967333 ± 0.003266.
- LR1_div3: best validation 0.970667 ± 0.001333.
- LR2_div10: best validation 0.969333 ± 0.003266.
- LR3_times3: best validation 0.967333 ± 0.003887.

### Final random-init seed checkpoints

- seed 101: best epoch 25 of 499; best/final validation 0.970000/0.976667; dominant sequence [30, 3, 38] (1.000).
- seed 202: best epoch 30 of 499; best/final validation 0.963333/0.980000; dominant sequence [15, 46, 45] (1.000).
- seed 303: best epoch 45 of 499; best/final validation 0.970000/0.993333; dominant sequence [39, 1, 35] (1.000).
- seed 404: best epoch 0 of 499; best/final validation 0.970000/0.980000; dominant sequence [18, 26, 13] (1.000).
- seed 505: best epoch 5 of 499; best/final validation 0.963333/0.986667; dominant sequence [0, 28, 11] (1.000).

### Per-seed adaptivity

- seed 101: effectively nonadaptive; MI=0; dominant fraction=1.000; unique sequences=1.
- seed 202: effectively nonadaptive; MI=0; dominant fraction=1.000; unique sequences=1.
- seed 303: effectively nonadaptive; MI=0; dominant fraction=1.000; unique sequences=1.
- seed 404: effectively nonadaptive; MI=0; dominant fraction=1.000; unique sequences=1.
- seed 505: effectively nonadaptive; MI=0; dominant fraction=1.000; unique sequences=1.

### Paired confirmation comparisons

- dad_minus_fixed: mean=-0.004608, 95% CI [-0.008016, -0.001376], first lower=0.131, tied=0.666, second lower=0.203; first method reliably lower.
- dad_minus_myopic: mean=0.010192, 95% CI [0.007792, 0.012528], first lower=0.027, tied=0.697, second lower=0.276; second method reliably lower.
- dad_minus_random: mean=-0.007648, 95% CI [-0.011472, -0.003936], first lower=0.140, tied=0.633, second lower=0.227; first method reliably lower.
- myopic_minus_fixed: mean=-0.014800, 95% CI [-0.018480, -0.011200], first lower=0.135, tied=0.839, second lower=0.026; first method reliably lower.
- fixed_minus_random: mean=-0.003040, 95% CI [-0.007840, 0.001680], first lower=0.117, tied=0.769, second lower=0.114; no reliable difference (CI contains zero).
- myopic_minus_random: mean=-0.017840, 95% CI [-0.022000, -0.013840], first lower=0.143, tied=0.832, second lower=0.025; first method reliably lower.

### System conclusion

- DAD vs Fixed: first method reliably lower.
- DAD vs Myopic: second method reliably lower.
- Observation-dependent policies: 0/5.
- All policies fixed-like by dominant-sequence threshold: True.
- Seed stability (validation range): 0.006667.

## Interpretation rule

A method is not declared better when its paired 95% confidence interval contains zero. Observation-driven sequence changes, rather than stochastic sequence diversity alone, determine whether DAD is classified as adaptive.
