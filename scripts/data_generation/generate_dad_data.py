"""
Generate trajectory data for training DAD (Deep Adaptive Design) policy network.

For second-order Kuramoto (swing equation) with active probing.

Based on documents/design_part1.tex:
- Actions: Probe design ξ_t = (b_t, A_t, T_p) where b_t is bus, A_t is amplitude, T_p = 2 s fixed
- Observations: ROCOF-only y_t = ROCOF_max (design Section 4); full features used for heuristics
- State: Uncertainty bounds (M_lower, M_upper, K_lower, K_upper)
- True parameters: (M_true, K_true)
"""

import sys
from pathlib import Path
import time
import argparse
import random
import json
import os
import yaml

# Add project root to path
# File in scripts/data_generation/ -> project root = parent.parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm

# Import swing equation modules
try:
    from src.core.swing_equation_ode import (
        solve_swing_equation_ode,
        extract_frequency_features,
    )
    from src.core.swing_equation_params import (
        get_default_swing_equation_params,
        sample_uncertain_parameters
    )
    from src.models.predictors.swing_predictor_utils import (
        load_swing_mocu_predictor,
        predict_swing_mocu
    )
    SWING_EQUATION_AVAILABLE = True
except ImportError as e:
    SWING_EQUATION_AVAILABLE = False
    print(f"[ERROR] Swing equation modules not available: {e}")
    sys.exit(1)


def generate_random_system(N, B, P_m, D, g, M_lower_base, M_upper_base,
                          K_lower_base, K_upper_base, seed=None, uncertainty_ratio_min=0.3):
    """
    Generate a random system with initial uncertainty bounds and true parameters.
    
    Args:
        N: Number of buses
        B, P_m, D, g: Fixed system parameters
        M_lower_base, M_upper_base: Base inertia bounds
        K_lower_base, K_upper_base: Base control gain bounds
        seed: Random seed
        uncertainty_ratio_min: Min ratio of base range for initial bounds. 1.0 = full bounds
                              (harder problem, method differences matter); 0.3-0.7 = narrowed
                              (easier, all methods converge quickly to same curve).
    
    Returns:
        (M_lower_0, M_upper_0, K_lower_0, K_upper_0, M_true, K_true, init_sync)
        init_sync is always 0 (sync check removed; target is (M,K) uncertainty).
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Generate initial uncertainty bounds
    if uncertainty_ratio_min >= 1.0:
        # Full base bounds: harder problem, more initial uncertainty
        M_lower_0, M_upper_0 = M_lower_base, M_upper_base
        K_lower_0, K_upper_0 = K_lower_base, K_upper_base
    else:
        # Random subset of base range (original behavior)
        uncertainty_ratio = uncertainty_ratio_min + (1.0 - uncertainty_ratio_min) * random.random()
        M_center = np.random.uniform(M_lower_base, M_upper_base)
        K_center = np.random.uniform(K_lower_base, K_upper_base)
        M_range = M_upper_base - M_lower_base
        K_range = K_upper_base - K_lower_base
        M_half_range = M_range * uncertainty_ratio / 2.0
        K_half_range = K_range * uncertainty_ratio / 2.0
        M_lower_0 = max(M_lower_base, M_center - M_half_range)
        M_upper_0 = min(M_upper_base, M_center + M_half_range)
        K_lower_0 = max(K_lower_base, K_center - K_half_range)
        K_upper_0 = min(K_upper_base, K_center + K_half_range)
        if M_lower_0 >= M_upper_0:
            M_lower_0, M_upper_0 = M_lower_base, M_upper_base
        if K_lower_0 >= K_upper_0:
            K_lower_0, K_upper_0 = K_lower_base, K_upper_base
    
    # Sample true parameters from initial bounds
    M_true = np.random.uniform(M_lower_0, M_upper_0)
    K_true = np.random.uniform(K_lower_0, K_upper_0)
    
    init_sync = 0  # Unused; kept for API compatibility (sync check removed)
    return M_lower_0, M_upper_0, K_lower_0, K_upper_0, M_true, K_true, init_sync


def perform_probe_experiment(B, P_m, D, M_true, K_true, g, probe_bus, probe_amplitude,
                            probe_duration, h, T, M_steps, device='cuda', timeout=5.0,
                            sigma=0.0):
    """
    Perform probe experiment and extract frequency features.
    
    Args:
        B, P_m, D, g: Fixed system parameters
        M_true, K_true: True parameters (unknown to agent)
        probe_bus: Bus index to probe (0-13 for IEEE-14)
        probe_amplitude: Probe amplitude A
        probe_duration: Probe duration T
        h: Time step
        T: Time horizon
        M_steps: Number of time steps
        device: 'cuda' or 'cpu'
        timeout: Maximum time for ODE solving
        sigma: Observation noise std (Hz/s); if > 0, add N(0,sigma^2) to ROCOF_max (design §4.3)
    
    Returns:
        observation: Dictionary with frequency features {'ROCOF_max', 'f_min', 't_settle', ...}
    """
    try:
        # Solve swing equation with probe
        state_traj = solve_swing_equation_ode(
            B, P_m, D, M_true, K_true, g,
            probe_bus=probe_bus,
            probe_amplitude=probe_amplitude,
            probe_duration=probe_duration,
            h=h, M_steps=M_steps, T=T,
            device=device, timeout=timeout
        )
        
        # Extract frequency trajectory (last N columns are ω)
        N = len(P_m)
        omega_traj = state_traj[:, N:]  # [M_steps, N]
        
        # Extract frequency features (downsampled to fs=12 Hz, design §4)
        # Use probe_bus so observation depends on probe location (different probes → different y)
        features = extract_frequency_features(omega_traj, h, fs=12.0, probe_bus=probe_bus)
        # Add observation noise (design §4.3: y = Map(θ,ξ) + ε, ε ~ N(0,σ²))
        if sigma > 0:
            features['ROCOF_max'] = float(features['ROCOF_max']) + float(np.random.normal(0, sigma))
            features['ROCOF_max'] = max(0.0, features['ROCOF_max'])  # ROCOF cannot be negative
        return features
    except Exception as e:
        # Return default features if simulation fails
        print(f"[WARNING] Probe experiment failed: {e}")
        return {
            'ROCOF_max': 0.0,
            'f_min': 50.0,
            't_settle': T
        }


def update_bounds(M_lower, M_upper, K_lower, K_upper, observation, probe_bus,
                 probe_amplitude, M_lower_base, M_upper_base, K_lower_base, K_upper_base,
                 update_strength=0.1, min_relative_width=0.02):
    """
    Update uncertainty bounds based on observation.
    
    Simple heuristic: Use observation features to narrow bounds.
    Enforces a minimum relative width (fraction of base range) so bounds never
    collapse to a point, avoiding trivial MOCU=0 and keeping the problem non-trivial.
    
    Args:
        M_lower, M_upper, K_lower, K_upper: Current bounds
        observation: Frequency features from probe
        probe_bus: Bus where probe was applied
        probe_amplitude: Probe amplitude used
        M_lower_base, M_upper_base, K_lower_base, K_upper_base: Base bounds
        update_strength: How much to update (0.0 = no update, 1.0 = full update)
        min_relative_width: Minimum width as fraction of base range (default 0.02 = 2%)
    
    Returns:
        (M_lower_new, M_upper_new, K_lower_new, K_upper_new): Updated bounds
    """
    M_lower_new = float(M_lower)
    M_upper_new = float(M_upper)
    K_lower_new = float(K_lower)
    K_upper_new = float(K_upper)
    
    rocof_max = observation.get('ROCOF_max', 0.0)
    f_min = observation.get('f_min', 50.0)
    rocof_normalized = min(rocof_max / 1.0, 1.0)
    f_min_normalized = max((49.5 - f_min) / 0.5, 0.0)  # 50 Hz nominal: band 49.5-50.5 Hz (aligned with MATLAB .mdl)
    
    M_base_range = M_upper_base - M_lower_base
    K_base_range = K_upper_base - K_lower_base
    min_M_width = max(min_relative_width * M_base_range, 1e-6)
    min_K_width = max(min_relative_width * K_base_range, 1e-6)
    
    # Update M bounds
    M_range = M_upper_new - M_lower_new
    if rocof_normalized > 0.5 or f_min_normalized > 0.5:
        M_upper_new = M_upper_new - update_strength * M_range * rocof_normalized
        M_upper_new = max(M_lower_new, min(M_upper_base, M_upper_new))
    else:
        M_lower_new = M_lower_new + update_strength * M_range * (1.0 - rocof_normalized)
        M_lower_new = min(M_upper_new, max(M_lower_base, M_lower_new))
    if M_upper_new - M_lower_new < min_M_width:
        mid = (M_lower_new + M_upper_new) / 2
        M_lower_new = max(M_lower_base, mid - min_M_width / 2)
        M_upper_new = min(M_upper_base, M_lower_new + min_M_width)
        M_lower_new = max(M_lower_base, M_upper_new - min_M_width)
    
    # Update K bounds
    K_range = K_upper_new - K_lower_new
    if rocof_normalized > 0.5 or f_min_normalized > 0.5:
        K_upper_new = K_upper_new - update_strength * K_range * rocof_normalized
        K_upper_new = max(K_lower_new, min(K_upper_base, K_upper_new))
    else:
        K_lower_new = K_lower_new + update_strength * K_range * (1.0 - rocof_normalized)
        K_lower_new = min(K_upper_new, max(K_lower_base, K_lower_new))
    if K_upper_new - K_lower_new < min_K_width:
        mid = (K_lower_new + K_upper_new) / 2
        K_lower_new = max(K_lower_base, mid - min_K_width / 2)
        K_upper_new = min(K_upper_base, K_lower_new + min_K_width)
        K_lower_new = max(K_lower_base, K_upper_new - min_K_width)
    
    return M_lower_new, M_upper_new, K_lower_new, K_upper_new


def update_bounds_bayesian(M_lower, M_upper, K_lower, K_upper, history,
                          M_lower_base, M_upper_base, K_lower_base, K_upper_base,
                          B, P_m, D, g, h, T, M_steps, sigma=0.05,
                          n_particles=256, credible_alpha=0.05, min_relative_width=0.02,
                          device='cuda'):
    """
    Update uncertainty bounds using Bayesian posterior (design §5).
    Uses particle weights from likelihood p(y|θ,ξ) so different (design, observation)
    sequences yield different posteriors — order matters.

    Args:
        M_lower, M_upper, K_lower, K_upper: Current bounds (prior support)
        history: List of ((probe_bus, probe_amplitude, probe_duration), observation)
                 where observation has 'ROCOF_max' key
        B, P_m, D, g, h, T, M_steps: System params for likelihood
        sigma: Observation noise std (Hz/s)
        n_particles: Number of particles for posterior
        credible_alpha: Tail mass for credible interval (e.g. 0.05 → 2.5%-97.5%)
        min_relative_width: Minimum bound width as fraction of base range

    Returns:
        (M_lower_new, M_upper_new, K_lower_new, K_upper_new): Updated bounds
    """
    from src.core.posterior_particles import posterior_weights

    # Sample particles from uniform over current bounds
    particles = np.zeros((n_particles, 2))
    particles[:, 0] = np.random.uniform(M_lower, M_upper, n_particles)
    particles[:, 1] = np.random.uniform(K_lower, K_upper, n_particles)

    # Build history for posterior: xi = (b+1, A, T_p) 1-based bus, y = ROCOF_max
    ll_history = []
    for (probe_bus, probe_amplitude, probe_duration), obs in history:
        xi = (int(probe_bus) + 1, float(probe_amplitude), float(probe_duration))
        y = float(obs.get('ROCOF_max', 0.0))
        ll_history.append((xi, y))

    if not ll_history:
        return M_lower, M_upper, K_lower, K_upper

    # Compute posterior weights
    weights = posterior_weights(
        particles, ll_history, sigma, B, P_m, D, g,
        M_lower, M_upper, K_lower, K_upper,
        h=h, T=T, M_steps=M_steps, device=device
    )

    # Credible interval from weighted particles (2.5%-97.5% quantiles)
    M_vals = particles[:, 0]
    K_vals = particles[:, 1]
    order_M = np.argsort(M_vals)
    order_K = np.argsort(K_vals)
    cumw_M = np.cumsum(weights[order_M])
    cumw_K = np.cumsum(weights[order_K])
    lo_M = np.searchsorted(cumw_M, credible_alpha / 2)
    hi_M = np.searchsorted(cumw_M, 1.0 - credible_alpha / 2)
    M_lower_new = float(M_vals[order_M[min(lo_M, n_particles - 1)]])
    M_upper_new = float(M_vals[order_M[min(hi_M, n_particles - 1)]])
    lo_K = np.searchsorted(cumw_K, credible_alpha / 2)
    hi_K = np.searchsorted(cumw_K, 1.0 - credible_alpha / 2)
    K_lower_new = float(K_vals[order_K[min(lo_K, n_particles - 1)]])
    K_upper_new = float(K_vals[order_K[min(hi_K, n_particles - 1)]])

    # Enforce min width and clamp to base bounds
    M_base_range = M_upper_base - M_lower_base
    K_base_range = K_upper_base - K_lower_base
    min_M_width = max(min_relative_width * M_base_range, 1e-6)
    min_K_width = max(min_relative_width * K_base_range, 1e-6)
    if M_upper_new - M_lower_new < min_M_width:
        mid = (M_lower_new + M_upper_new) / 2
        M_lower_new = max(M_lower_base, mid - min_M_width / 2)
        M_upper_new = min(M_upper_base, M_lower_new + min_M_width)
        M_lower_new = max(M_lower_base, M_upper_new - min_M_width)
    if K_upper_new - K_lower_new < min_K_width:
        mid = (K_lower_new + K_upper_new) / 2
        K_lower_new = max(K_lower_base, mid - min_K_width / 2)
        K_upper_new = min(K_upper_base, K_lower_new + min_K_width)
        K_lower_new = max(K_lower_base, K_upper_new - min_K_width)

    M_lower_new = max(M_lower_base, min(M_upper_base, M_lower_new))
    M_upper_new = max(M_lower_new, min(M_upper_base, M_upper_new))
    K_lower_new = max(K_lower_base, min(K_upper_base, K_lower_new))
    K_upper_new = max(K_lower_new, min(K_upper_base, K_upper_new))

    return M_lower_new, M_upper_new, K_lower_new, K_upper_new


def generate_trajectory(N, K, B, P_m, D, g, M_lower_base, M_upper_base,
                       K_lower_base, K_upper_base, probe_amplitudes, probe_duration,
                       h, T, M_steps, device='cuda', verbose=False,
                       mocu_predictor=None, mocu_mean=None, mocu_std=None, sigma=0.0,
                       uncertainty_ratio_min=0.3, update_strength=0.05):
    """
    Generate a single trajectory using random policy.

    Args:
        mocu_predictor: Optional MPNN MOCU predictor for terminal MOCU.
        mocu_mean, mocu_std: Normalization statistics (for predictor).
    """
    # Generate random system (sync check removed; target is (M,K) uncertainty)
    (M_lower_0, M_upper_0, K_lower_0, K_upper_0, M_true, K_true, _) = \
        generate_random_system(
            N, B, P_m, D, g, M_lower_base, M_upper_base,
            K_lower_base, K_upper_base, seed=None, uncertainty_ratio_min=uncertainty_ratio_min
        )
    
    # Initialize trajectory
    trajectory = {
        'M_true': float(M_true),
        'K_true': float(K_true),
        'states': [(M_lower_0, M_upper_0, K_lower_0, K_upper_0)],  # Initial bounds
        'designs': [],  # Probe designs ξ = (b, A, T_p)
        'observations': []  # Frequency features
    }
    
    M_lower = M_lower_0
    M_upper = M_upper_0
    K_lower = K_lower_0
    K_upper = K_upper_0
    
    # Run K steps
    for step in range(K):
        # Random policy: select random bus and amplitude
        probe_bus = random.randint(0, N - 1)
        probe_amplitude = random.choice(probe_amplitudes)
        
        # Perform probe experiment
        observation = perform_probe_experiment(
            B, P_m, D, M_true, K_true, g,
            probe_bus, probe_amplitude, probe_duration,
            h, T, M_steps, device=device, sigma=sigma
        )
        
        # Record design and observation before update (history for Bayesian)
        trajectory['designs'].append((probe_bus, probe_amplitude, probe_duration))
        trajectory['observations'].append(observation)
        # Bayesian update: history = list of ((probe_bus, A, T_p), obs) so order matters
        history = list(zip(trajectory['designs'], trajectory['observations']))
        (M_lower, M_upper, K_lower, K_upper) = update_bounds_bayesian(
            M_lower, M_upper, K_lower, K_upper, history,
            M_lower_base, M_upper_base, K_lower_base, K_upper_base,
            B=B, P_m=P_m, D=D, g=g, h=h, T=T, M_steps=M_steps, sigma=sigma,
            n_particles=128, device=device
        )
        trajectory['states'].append((M_lower, M_upper, K_lower, K_upper))
        
        if verbose:
            print(f"Step {step+1}: Probe bus {probe_bus}, A={probe_amplitude}, "
                  f"ROCOF_max={observation.get('ROCOF_max', 0.0):.4f}")
    
    # Compute terminal MOCU if MPNN predictor provided
    terminal_MOCU = None
    if mocu_predictor is not None:
        try:
            terminal_MOCU = predict_swing_mocu(
                mocu_predictor, mocu_mean, mocu_std,
                M_lower, M_upper, K_lower, K_upper, device=device
            )
            if hasattr(terminal_MOCU, 'item'):
                terminal_MOCU = terminal_MOCU.item()
            terminal_MOCU = float(terminal_MOCU)
        except Exception as e:
            print(f"Warning: Failed to compute terminal MOCU: {e}")
            terminal_MOCU = None
    
    if terminal_MOCU is not None:
        trajectory['terminal_MOCU'] = terminal_MOCU
    
    return trajectory


def main():
    parser = argparse.ArgumentParser(description='Generate DAD policy training data for swing equation')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--num-episodes', type=int, default=None,
                        help='Number of trajectories to generate (overrides config)')
    parser.add_argument('--K', type=int, default=None,
                        help='Number of sequential experiments (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-mocu-predictor', action='store_true',
                       help='Pre-compute terminal MOCU using MPNN MOCU predictor')
    parser.add_argument('--mocu-model-name', type=str, default=None,
                       help='MOCU predictor model name (required if --use-mocu-predictor)')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters
    N = config['N']
    model_type = config.get('model_type', 'second_order')
    
    if model_type != 'second_order':
        raise ValueError(f"Expected model_type='second_order', got '{model_type}'")
    
    swing_params = config['swing_equation']
    dad_data_params = config.get('dad_data', {})
    paths = config.get('paths', {})
    
    # System parameters
    topology = swing_params.get('topology', 'ieee14')
    coupling_strength = swing_params.get('coupling_strength', 1.0)
    damping = swing_params.get('damping', 0.1)
    base_power = swing_params.get('base_power', 1.0)
    M_lower_base = swing_params.get('M_lower', 0.01)
    M_upper_base = swing_params.get('M_upper', 0.06)
    K_lower_base = swing_params.get('K_lower', 0.1)
    K_upper_base = swing_params.get('K_upper', 1.0)
    probe_duration = swing_params.get('probe_duration', 2.0)
    probe_amplitudes = swing_params.get('probe_amplitudes', [0.5, 1.0, 2.0])
    sigma = swing_params.get('sigma', 0.05)
    experiment_params = config.get('experiment', {})
    uncertainty_ratio_min = experiment_params.get('uncertainty_ratio_min', 0.3)
    update_strength = experiment_params.get('update_strength', 0.05)
    
    # DAD data parameters
    num_episodes = args.num_episodes if args.num_episodes is not None else dad_data_params.get('num_episodes', 1000)
    K = args.K if args.K is not None else dad_data_params.get('K', 4)
    
    # Time parameters
    T = 10.0  # Time horizon (seconds)
    h = 1.0 / 160.0  # Time step (seconds)
    M_steps = int(T / h)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(paths.get('data_dir', 'data'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Generate system parameters (fixed, known)
    system_params = get_default_swing_equation_params(
        N=N,
        topology=topology,
        coupling_strength=coupling_strength,
        damping=damping,
        base_power=base_power,
        M_lower=M_lower_base,
        M_upper=M_upper_base,
        K_lower=K_lower_base,
        K_upper=K_upper_base
    )
    
    B = system_params['B']
    P_m = system_params['P_m']
    D = system_params['D']
    g = system_params['g']
    
    print("=" * 80)
    print("DAD Policy Training Data Generation (Swing Equation)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Number of buses: {N}")
    print(f"  - Topology: {topology}")
    print(f"  - Design steps (K): {K}")
    print(f"  - Number of episodes: {num_episodes}")
    print(f"  - Probe amplitudes: {probe_amplitudes}")
    print(f"  - Probe duration: {probe_duration}s")
    print(f"  - Pre-compute MOCU: {'Yes' if args.use_mocu_predictor else 'No'}")
    print("=" * 80)
    
    mocu_predictor = None
    mocu_mean = None
    mocu_std = None
    
    if args.use_mocu_predictor:
        if args.mocu_model_name is None:
            args.mocu_model_name = os.getenv('MOCU_MODEL_NAME')
            if args.mocu_model_name is None:
                raise ValueError("--mocu-model-name required when using --use-mocu-predictor")
        try:
            print(f"[INFO] Loading MPNN MOCU predictor: {args.mocu_model_name}")
            mocu_predictor, mocu_mean, mocu_std = load_swing_mocu_predictor(
                model_name=args.mocu_model_name, device=device, B=B, N=N
            )
            print(f"[INFO] ✓ MPNN MOCU predictor loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load MPNN MOCU predictor: {e}")
            return
    
    trajectories = []
    start_time = time.time()
    
    for episode in tqdm(range(num_episodes), desc="Generating trajectories"):
        trajectory = generate_trajectory(
            N=N, K=K, B=B, P_m=P_m, D=D, g=g,
            M_lower_base=M_lower_base, M_upper_base=M_upper_base,
            K_lower_base=K_lower_base, K_upper_base=K_upper_base,
            probe_amplitudes=probe_amplitudes, probe_duration=probe_duration,
            sigma=sigma, uncertainty_ratio_min=uncertainty_ratio_min,
            update_strength=update_strength,
            h=h, T=T, M_steps=M_steps, device=device,
            verbose=False,
            mocu_predictor=mocu_predictor,
            mocu_mean=mocu_mean, mocu_std=mocu_std
        )
        
        if trajectory is not None:
            trajectories.append(trajectory)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (episode + 1)
            eta = avg_time * (num_episodes - episode - 1)
            print(f"\n  Progress: {episode+1}/{num_episodes} | "
                  f"Valid: {len(trajectories)} | "
                  f"Avg: {avg_time:.2f}s/episode | ETA: {eta/60:.1f} min")
    
    print(f"\n✓ Generated {len(trajectories)} valid trajectories")
    
    # Save data
    output_file = output_dir / f'swing_dad_trajectories_N{N}_K{K}.pth'
    
    has_mocu = any('terminal_MOCU' in traj for traj in trajectories)
    
    torch.save({
        'trajectories': trajectories,
        'config': {
            'N': N,
            'K': K,
            'num_episodes': len(trajectories),
            'has_precomputed_mocu': has_mocu,
            'mocu_model_used': args.mocu_model_name if args.use_mocu_predictor else None,
            'model_type': 'swing_equation',
            'probe_amplitudes': probe_amplitudes,
        }
    }, output_file)
    
    if has_mocu:
        mocu_values = [traj.get('terminal_MOCU', None) for traj in trajectories if 'terminal_MOCU' in traj]
        if mocu_values:
            mocu_array = np.array(mocu_values)
            print(f"\nTerminal MOCU statistics:")
            print(f"  Mean: {np.mean(mocu_array):.6f}")
            print(f"  Std:  {np.std(mocu_array):.6f}")
            print(f"  Min:  {np.min(mocu_array):.6f}")
            print(f"  Max:  {np.max(mocu_array):.6f}")
    
    print(f"✓ Saved to: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
