"""
Swing equation simulator.

Based on documents/design_part1.tex Section 1:
- Second-order Kuramoto (swing equation) for IEEE-14 network
- State: [θ, ω] where θ is phase and ω is frequency
- Dynamics: M dω/dt = P_m - Σ B_ij sin(θ_i - θ_j) - D ω - K ω + u_probe + u_ctrl
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Re-export from existing implementation
from src.core.swing_equation_ode import (
    SwingEquationODE,
    solve_swing_equation_ode,
    check_frequency_synchronization
)

__all__ = ['SwingEquationODE', 'solve_swing_equation_ode', 'check_frequency_synchronization']
