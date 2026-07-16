# Information redundancy report — ieee5

**Result:** FAIL

FAIL: The reset-based scenario may remain physically valid, but it is not a valid DAD-superiority benchmark under the current prior, action catalogue, noise level, and horizon. Do not proceed to claim that DAD should beat myopic in information gain.

## Per-seed summary
### Seed 0
- V* = 4.4351, V_myopic = 4.2245, G_full = 0.2105
- V_adaptive = 4.3662, V_fixed = 4.2663, G_adapt = 0.1000
- Oracle first = 28, myopic first = 34
- G_full CI [0.0485, 0.2395] pass=False
- G_adapt CI [0.0264, 0.1685] pass=False

### Seed 1
- V* = 4.3936, V_myopic = 4.2245, G_full = 0.1691
- V_adaptive = 4.3636, V_fixed = 4.2344, G_adapt = 0.1292
- Oracle first = 28, myopic first = 34
- G_full CI [0.0485, 0.2373] pass=False
- G_adapt CI [0.0537, 0.2068] pass=False

### Seed 2
- V* = 4.4083, V_myopic = 4.2245, G_full = 0.1838
- V_adaptive = 4.3129, V_fixed = 4.2344, G_adapt = 0.0785
- Oracle first = 33, myopic first = 34
- G_full CI [-0.0090, 0.1862] pass=False
- G_adapt CI [0.0092, 0.1561] pass=False

