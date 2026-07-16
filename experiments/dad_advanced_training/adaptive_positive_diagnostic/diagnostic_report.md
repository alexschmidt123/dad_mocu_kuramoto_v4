# Adaptive-value-positive training diagnostic

This reduced T=2 problem was constructed exclusively from the ieee5
training bank. It uses 9 posterior particles and
6 actions. No validation, confirmation, test, Myopic label,
Fixed label, physical ODE call, or alternate scientific objective is used.

- Best Fixed cost: **0.864722**
- Reference adaptive-tree cost: **0.835205**
- Adaptive advantage (Fixed - adaptive): **0.029517**
- 95% paired bootstrap CI: **[0.026514, 0.032593]**
- First action: `15`
- Observation-bin edges: `[1.4958852064070338, 1.982521296993255]`
- Second actions by bin: `[22, 9, 9]`
- Verified positive adaptive advantage: **True**

The training algorithm passes this diagnostic only if a random-initialized
policy learns observation-dependent branching and recovers a meaningful,
positive fraction of this reference advantage.


## Random-initialized advanced DAD result

- Learned DAD cost: **0.831250**
- Fraction of known adaptive advantage recovered: **1.134**
- Branching accuracy: **0.135**
- Unique deterministic second actions: **3**
- Observation-dependent: **True**
- Trainer passed: **True**


## Random-initialized advanced DAD result

- Learned DAD cost: **0.838721**
- Fraction of known adaptive advantage recovered: **0.881**
- Branching accuracy: **0.000**
- Unique deterministic second actions: **4**
- Observation-dependent: **True**
- Trainer passed: **True**
