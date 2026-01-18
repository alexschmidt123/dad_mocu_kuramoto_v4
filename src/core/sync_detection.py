"""
Synchronization detection for second-order Kuramoto (swing equation) systems.

This module is part of the MOCU-OED project for optimal experimental design
in coupled oscillator systems using the second-order Kuramoto model (swing equation).

Based on new_plan.tex: We check frequency synchronization (ω convergence) instead of phase.
"""

import numpy as np
import time

# Import second-order model functions
try:
    from .swing_equation_ode import (
        solve_swing_equation_ode,
        check_frequency_synchronization
    )
    from .swing_equation_params import get_default_swing_equation_params
    SWING_EQUATION_AVAILABLE = True
except ImportError:
    SWING_EQUATION_AVAILABLE = False
    print("[WARNING] Swing equation modules not available. Sync detection disabled.")


def mocu_comp(B, P_m, D, M, K, g, h, N, M_steps, gamma=None, 
              probe_bus=None, probe_amplitude=None, probe_duration=2.0):
    """
    Check if second-order Kuramoto system is frequency-synchronized.
    
    Replaces first-order mocu_comp function.
    
    Args:
        B: Coupling matrix [N, N] (numpy array)
        P_m: Mechanical power [N] (numpy array)
        D: Damping coefficient (scalar)
        M: Inertia (scalar)
        K: Control gain (scalar)
        g: Control allocation [N] (numpy array)
        h: Time step (float)
        N: Number of buses (int)
        M_steps: Number of time steps (int)
        gamma: Control capacity (scalar, optional)
        probe_bus: Bus index for probing (int, optional)
        probe_amplitude: Probe amplitude (float, optional)
        probe_duration: Probe duration (float, default 2.0s)
    
    Returns:
        D: 1 if synchronized, 0 if not
    """
    if not SWING_EQUATION_AVAILABLE:
        raise RuntimeError("Swing equation modules not available")
    
    # Solve swing equation
    T = h * M_steps
    state_traj = solve_swing_equation_ode(
        B, P_m, D, M, K, g, gamma=gamma,
        probe_bus=probe_bus, probe_amplitude=probe_amplitude, probe_duration=probe_duration,
        h=h, M_steps=M_steps, T=T, device='cpu'
    )
    
    # Check frequency synchronization
    D = check_frequency_synchronization(state_traj, M_steps)
    
    return D


def determineSyncN(B, P_m, D, M, K, g, h, N, M_steps, gamma=None):
    """
    Determine if N-bus system is frequency-synchronized.
    
    Replaces first-order determineSyncN function.
    
    Args:
        B: Coupling matrix [N, N] (numpy array)
        P_m: Mechanical power [N] (numpy array)
        D: Damping coefficient (scalar)
        M: Inertia (scalar)
        K: Control gain (scalar)
        g: Control allocation [N] (numpy array)
        h: Time step (float)
        N: Number of buses (int)
        M_steps: Number of time steps (int)
        gamma: Control capacity (scalar, optional)
    
    Returns:
        D: 1 if synchronized, 0 if not
    """
    return mocu_comp(B, P_m, D, M, K, g, h, N, M_steps, gamma=gamma)


def determineSyncTwo(B, P_m_i, P_m_j, D, M, K, g_i, g_j, h, N, M_steps, B_ij, gamma=None):
    """
    Determine if 2-bus system is frequency-synchronized.
    
    Replaces first-order determineSyncTwo function.
    
    Args:
        B: Coupling matrix [2, 2] (numpy array)
        P_m_i, P_m_j: Mechanical power for buses i and j (scalars)
        D: Damping coefficient (scalar)
        M: Inertia (scalar)
        K: Control gain (scalar)
        g_i, g_j: Control allocation for buses i and j (scalars)
        h: Time step (float)
        N: Number of buses (should be 2)
        M_steps: Number of time steps (int)
        B_ij: Coupling strength between buses i and j (scalar)
        gamma: Control capacity (scalar, optional)
    
    Returns:
        D: 1 if synchronized, 0 if not
    """
    if N != 2:
        print("Warning: determineSyncTwo expects N=2")
    
    # Create 2-bus system
    P_m = np.array([P_m_i, P_m_j])
    g = np.array([g_i, g_j])
    B_2bus = np.array([[0, B_ij], [B_ij, 0]])
    
    return mocu_comp(B_2bus, P_m, D, M, K, g, h, 2, M_steps, gamma=gamma)
