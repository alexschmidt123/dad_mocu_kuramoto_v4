"""
OED Methods Package

All experimental design methods follow the OEDMethod base class interface.

Baseline methods:
- RANDOM_Method: Random probe selection
- ENTROPY_Method: Variance/entropy-based heuristic
- ODE_Method: Myopic MOCU optimization
- NN_Method: Static neural network predictor
- iNN_Method: Iterative neural network predictor

Policy methods:
- DAD_MOCU_Method: Deep Adaptive Design with MOCU objective
"""

from .base import OEDMethod
from .random_probe import RANDOM_Method
from .variance_heuristic import ENTROPY_Method
from .myopic_mocu import ODE_Method
from .static_nn import NN_Method
from .iterative_nn import iNN_Method
from .dad_policy import DAD_MOCU_Method

__all__ = [
    'OEDMethod',
    'RANDOM_Method',
    'ENTROPY_Method',
    'ODE_Method',
    'NN_Method',
    'iNN_Method',
    'DAD_MOCU_Method',
]
