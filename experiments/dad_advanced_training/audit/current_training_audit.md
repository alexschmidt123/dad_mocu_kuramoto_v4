# Current advanced REINFORCE training audit

## Scope and invariants

The previous five-seed, validation-selected REINFORCE policies were reloaded
without modifying their checkpoints. The audited rollout samples one training
system once, preserves it for all three decisions, uses complete histories,
masks used actions, reads observations only from the offline bank, and evaluates
the frozen terminal-control rule. Fixed and Myopic actions are not training
targets.

## Representation and optimizer audit

1. **Observation normalization:** the old encoder does not use training-split
   normalization. It applies `tanh(y/10)`. Training-bank statistics are:
   IEEE5 `{'mean': 2.8723770406715174, 'std': 1.6474632456924478, 'min': 0.8925621915625576, 'q05': 1.14189922256284, 'q50': 2.4836208066565426, 'q95': 6.524445509348955, 'max': 10.286181770237627}` and IEEE9
   `{'mean': 2.5175576876363848, 'std': 2.1188923202076637, 'min': 0.35974365614205506, 'q05': 0.5729824430393415, 'q50': 1.8152868927018901, 'q95': 7.409014970892286, 'max': 12.478535218778067}`. This fixed scale can make realistic
   observation differences unnecessarily small.
2. **Action encoding:** each past action is one-hot encoded and concatenated
   with one scalar observation.
3. **Complete history:** all accumulated `(action, observation)` pairs are
   supplied at every decision. Attention pooling is permutation-sensitive only
   through pair contents; there is no explicit within-history step embedding.
4. **Action masking:** used actions receive a pre-softmax logit of `-1e9` and
   are renormalized out of the feasible distribution.
5. **Logits:** raw and masked logits for every perturbation are recorded in
   `observation_sensitivity.csv`.
6. **Entropy:** feasible-action entropy is recorded at t=2 and t=3 in the same
   file; historical t=1/t=2/t=3 values remain in each seed's training metrics.
7. **Gradient flow:** component gradient norms and exact nonzero-gradient
   parameter percentages are in `gradient_flow.csv`.
8. **Terminal resolution:** IEEE5 averages
   `5.00` unique costs per
   64-trajectory audit batch with modal fraction
   `0.331`; IEEE9 averages
   `3.60` with modal fraction
   `0.875`.
9. **Advantages:** batch-centered terminal-cost advantage mean/std are recorded
   beside each component's gradient diagnostics.

## Direct observation perturbation test

For a fixed action history, the latest observation was swept through nine
training-only values spanning q05--q95. Mean L1 changes in the feasible action
distribution were:

- IEEE5: t=2 `0.00625831`, t=3
  `0.0028154`.
- IEEE9: t=2 `0.00957975`, t=3
  `0.00309071`.

Mean history-embedding L2 changes were:

- IEEE5: t=2 `0.0300422`, t=3
  `0.0151519`.
- IEEE9: t=2 `0.0585906`, t=3
  `0.0244083`.

## Failure attribution

The representation has a numerical path from `y` to the history embedding and
policy logits, and the gradient audit tests whether that path receives
nonzero gradients. However, one-hot action identity is unscaled while `y` is
compressed by a fixed `/10` transform, there is no explicit temporal feature,
and one quantized terminal cost is shared by all sampled actions in a
trajectory. Thus the old update provides weak action-conditional evidence for
learning observation branches and readily learns dominant action sequences.

This audit alone cannot conclude that IEEE5/IEEE9 have substantial adaptive
value. The three hypotheses are separated as follows:

- **Low adaptive value:** must be assessed by exhaustive/near-exhaustive
  adaptive-tree comparisons, not by policy nonadaptivity alone. Existing
  diagnoses indicate little measurable adaptive value under the frozen
  objective.
- **Representation ignores observations:** supported when embedding/logit
  perturbations and their gradients are negligible despite realistic changes
  in `y`.
- **Optimizer fails to train sensitivity:** supported if a belief-aware
  counterfactual trainer passes a known adaptive-positive diagnostic but the
  old REINFORCE trainer does not.

The adaptive-positive diagnostic is therefore a mandatory gate before drawing
the final causal interpretation.
