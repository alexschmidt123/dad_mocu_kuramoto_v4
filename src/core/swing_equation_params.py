"""
Helper functions to generate system parameters for second-order Kuramoto (swing equation).

Based on new_plan.tex, we need:
- B: Coupling matrix [N, N] (known, fixed)
- P_m: Mechanical power [N] (known, fixed)
- D: Damping coefficient (scalar, known, fixed)
- g: Control allocation [N] (known, fixed, sum to 1)
- M bounds: [M_lower, M_upper] (uncertain)
- K bounds: [K_lower, K_upper] (uncertain)
"""

import numpy as np
from typing import Tuple, Optional


def generate_ieee14_coupling_matrix(coupling_strength: float = 1.0) -> np.ndarray:
    """
    Generate coupling matrix B for IEEE-14 bus system.
    
    IEEE-14 bus system topology (based on standard test case):
    - 14 buses total (buses 1-14, indexed as 0-13)
    - Transmission lines connect buses as defined in IEEE-14 standard
    
    Standard IEEE-14 bus connections:
    - Bus 1 (slack): connected to 2, 5
    - Bus 2 (gen): connected to 1, 3, 4, 5
    - Bus 3 (gen): connected to 2, 4
    - Bus 4: connected to 2, 3, 5, 7, 9
    - Bus 5: connected to 1, 2, 4, 6
    - Bus 6 (gen): connected to 5, 11, 12, 13
    - Bus 7: connected to 4, 8, 9
    - Bus 8 (gen): connected to 7
    - Bus 9: connected to 4, 7, 10, 14
    - Bus 10: connected to 9, 11
    - Bus 11: connected to 6, 10
    - Bus 12: connected to 6, 13
    - Bus 13: connected to 6, 12, 14
    - Bus 14: connected to 9, 13
    
    Args:
        coupling_strength: Base coupling strength (float)
    
    Returns:
        B: Coupling matrix [14, 14] (numpy array, symmetric)
    """
    N = 14
    B = np.zeros((N, N))
    
    # IEEE-14 bus system transmission line connections
    # Bus indices are 0-indexed (bus 1 -> index 0, bus 14 -> index 13)
    connections = [
        (0, 1),   # Bus 1-2
        (0, 4),   # Bus 1-5
        (1, 2),   # Bus 2-3
        (1, 3),   # Bus 2-4
        (1, 4),   # Bus 2-5
        (2, 3),   # Bus 3-4
        (3, 4),   # Bus 4-5
        (3, 6),   # Bus 4-7
        (3, 8),   # Bus 4-9
        (4, 5),   # Bus 5-6
        (5, 10),  # Bus 6-11
        (5, 11),  # Bus 6-12
        (5, 12),  # Bus 6-13
        (6, 7),   # Bus 7-8
        (6, 8),   # Bus 7-9
        (8, 9),   # Bus 9-10
        (8, 13),  # Bus 9-14
        (9, 10),  # Bus 10-11
        (10, 11), # Bus 11-12
        (11, 12), # Bus 12-13
        (12, 13), # Bus 13-14
    ]
    
    # Set coupling strengths (can be made non-uniform based on line parameters)
    for i, j in connections:
        if i < N and j < N:
            B[i, j] = coupling_strength
            B[j, i] = coupling_strength
    
    return B


def generate_default_coupling_matrix(N: int, topology: str = 'fully_connected', 
                                     coupling_strength: float = 1.0) -> np.ndarray:
    """
    Generate coupling matrix B for swing equation.
    
    Args:
        N: Number of buses/oscillators
        topology: Network topology ('fully_connected', 'ring', 'star', 'random', 'ieee14')
        coupling_strength: Base coupling strength (float)
    
    Returns:
        B: Coupling matrix [N, N] (numpy array, symmetric)
    """
    if topology == 'ieee14':
        if N != 14:
            raise ValueError(f"IEEE-14 topology requires N=14, got N={N}")
        return generate_ieee14_coupling_matrix(coupling_strength)
    
    B = np.zeros((N, N))
    
    if topology == 'fully_connected':
        # All-to-all coupling (similar to first-order model)
        for i in range(N):
            for j in range(i + 1, N):
                B[i, j] = coupling_strength
                B[j, i] = coupling_strength
    
    elif topology == 'ring':
        # Ring topology: each bus connected to neighbors
        for i in range(N):
            j = (i + 1) % N
            B[i, j] = coupling_strength
            B[j, i] = coupling_strength
    
    elif topology == 'star':
        # Star topology: bus 0 is hub
        for i in range(1, N):
            B[0, i] = coupling_strength
            B[i, 0] = coupling_strength
    
    elif topology == 'random':
        # Random topology (Erdős–Rényi-like)
        np.random.seed(42)  # For reproducibility
        p = 0.5  # Connection probability
        for i in range(N):
            for j in range(i + 1, N):
                if np.random.random() < p:
                    B[i, j] = coupling_strength
                    B[j, i] = coupling_strength
    
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    return B


def generate_default_mechanical_power(N: int, method: str = 'uniform',
                                       base_power: float = 1.0) -> np.ndarray:
    """
    Generate mechanical power P_m for each bus.
    
    Args:
        N: Number of buses
        method: Generation method ('uniform', 'random', 'degree_based')
        base_power: Base power value (float)
    
    Returns:
        P_m: Mechanical power [N] (numpy array)
    """
    if method == 'uniform':
        P_m = np.ones(N) * base_power
    
    elif method == 'random':
        np.random.seed(42)
        P_m = base_power * (0.5 + np.random.random(N))
    
    elif method == 'degree_based':
        # Power proportional to node degree (if using degree-based coupling)
        # For now, use uniform as placeholder
        P_m = np.ones(N) * base_power
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return P_m


def generate_default_control_allocation(N: int, method: str = 'uniform') -> np.ndarray:
    """
    Generate control allocation g (spatial allocation of control across buses).
    
    Based on new_plan.tex: sum_i g_i = 1, g_i >= 0
    
    Args:
        N: Number of buses
        method: Allocation method ('uniform', 'random', 'hub_based')
    
    Returns:
        g: Control allocation [N] (numpy array, sum to 1)
    """
    if method == 'uniform':
        g = np.ones(N) / N
    
    elif method == 'random':
        np.random.seed(42)
        g = np.random.random(N)
        g = g / np.sum(g)  # Normalize to sum to 1
    
    elif method == 'hub_based':
        # More control at hub (bus 0)
        g = np.ones(N) * 0.5 / (N - 1)
        g[0] = 0.5
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return g


def get_default_swing_equation_params(N: int, 
                                     topology: str = 'fully_connected',
                                     coupling_strength: float = 1.0,
                                     damping: float = 0.1,
                                     base_power: float = 1.0,
                                     M_lower: float = 0.3,
                                     M_upper: float = 2.0,
                                     K_lower: float = 0.05,
                                     K_upper: float = 0.50) -> dict:
    """
    Generate default system parameters for swing equation.
    
    Args:
        N: Number of buses/oscillators
        topology: Network topology ('fully_connected', 'ring', 'star', 'random')
        coupling_strength: Base coupling strength (float)
        damping: Damping coefficient D (float)
        base_power: Base mechanical power (float)
        M_lower, M_upper: Inertia bounds (floats)
        K_lower, K_upper: Control gain bounds (floats)
    
    Returns:
        params: Dictionary with all system parameters
    """
    B = generate_default_coupling_matrix(N, topology, coupling_strength)
    P_m = generate_default_mechanical_power(N, method='uniform', base_power=base_power)
    g = generate_default_control_allocation(N, method='uniform')
    
    params = {
        'B': B,
        'P_m': P_m,
        'D': damping,
        'g': g,
        'M_lower': M_lower,
        'M_upper': M_upper,
        'K_lower': K_lower,
        'K_upper': K_upper,
        'N': N,
    }
    
    return params


def sample_uncertain_parameters(M_lower: float, M_upper: float,
                               K_lower: float, K_upper: float,
                               seed: Optional[int] = None) -> Tuple[float, float]:
    """
    Sample uncertain parameters (M, K) from uniform distribution.
    
    Args:
        M_lower, M_upper: Inertia bounds
        K_lower, K_upper: Control gain bounds
        seed: Random seed (optional)
    
    Returns:
        (M, K): Sampled inertia and control gain
    """
    if seed is not None:
        np.random.seed(seed)
    
    M = np.random.uniform(M_lower, M_upper)
    K = np.random.uniform(K_lower, K_upper)
    
    return M, K
