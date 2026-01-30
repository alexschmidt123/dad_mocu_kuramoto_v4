"""
Generate training data for Swing MLP predictor (second-order Kuramoto / swing equation).

Based on documents/design_part1.tex:
- Uncertainty: (M, K) with bounds [M_lower, M_upper, K_lower, K_upper]
- MOCU computation: E_{(M,K)~p_t}[γ*(A_t) - γ*(M,K)]
- Output: (M_lower, M_upper, K_lower, K_upper, MOCU) for MLP training
"""

import sys
from pathlib import Path
import time
import argparse
import os
from tqdm import tqdm
import multiprocessing as mp
import yaml

# Get absolute path to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import random
import torch

# Import swing equation MOCU computation
try:
    from src.core.swing_equation_mocu import MOCU_swing_equation
    from src.core.swing_equation_params import (
        get_default_swing_equation_params,
        sample_uncertain_parameters
    )
    SWING_EQUATION_AVAILABLE = True
except ImportError as e:
    SWING_EQUATION_AVAILABLE = False
    print(f"[ERROR] Swing equation modules not available: {e}")
    sys.exit(1)


def generate_uncertainty_bounds(M_lower_base, M_upper_base, K_lower_base, K_upper_base,
                                uncertainty_ratio=0.3, seed=None):
    """
    Generate a random uncertainty set (bounds) for (M, K).
    
    Args:
        M_lower_base, M_upper_base: Base inertia bounds
        K_lower_base, K_upper_base: Base control gain bounds
        uncertainty_ratio: Ratio of uncertainty (0-1)
        seed: Random seed
    
    Returns:
        (M_lower, M_upper, K_lower, K_upper): Random bounds within base ranges
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Sample a center point
    M_center = np.random.uniform(M_lower_base, M_upper_base)
    K_center = np.random.uniform(K_lower_base, K_upper_base)
    
    # Generate bounds around center with uncertainty_ratio
    M_range = M_upper_base - M_lower_base
    K_range = K_upper_base - K_lower_base
    
    M_half_range = M_range * uncertainty_ratio / 2.0
    K_half_range = K_range * uncertainty_ratio / 2.0
    
    M_lower = max(M_lower_base, M_center - M_half_range)
    M_upper = min(M_upper_base, M_center + M_half_range)
    K_lower = max(K_lower_base, K_center - K_half_range)
    K_upper = min(K_upper_base, K_center + K_half_range)
    
    # Ensure bounds are valid
    if M_lower >= M_upper:
        M_lower = M_lower_base
        M_upper = M_upper_base
    if K_lower >= K_upper:
        K_lower = K_lower_base
        K_upper = K_upper_base
    
    return M_lower, M_upper, K_lower, K_upper


def generate_single_sample(args_tuple):
    """
    Generate a single training sample: (M_lower, M_upper, K_lower, K_upper, MOCU).
    
    Args:
        args_tuple: (N, K_max, B, P_m, D, g, M_lower_base, M_upper_base, 
                     K_lower_base, K_upper_base, r_max, f_min, h, T, M_steps,
                     device, worker_id)
    
    Returns:
        dict with keys: 'M_lower', 'M_upper', 'K_lower', 'K_upper', 'MOCU'
        or None if generation failed
    """
    (N, K_max, B, P_m, D, g, M_lower_base, M_upper_base,
     K_lower_base, K_upper_base, r_max, f_min, h, T, M_steps,
     device, worker_id) = args_tuple
    
    # Set worker-specific random seed
    if worker_id is not None:
        seed = worker_id * 12345 + int(time.time()) % 10000
        np.random.seed(seed)
        random.seed(seed)
    else:
        seed = None
    
    # Generate random uncertainty bounds
    M_lower, M_upper, K_lower, K_upper = generate_uncertainty_bounds(
        M_lower_base, M_upper_base, K_lower_base, K_upper_base,
        uncertainty_ratio=0.3, seed=seed
    )
    
    # Compute MOCU for this set of bounds
    try:
        MOCU_val = MOCU_swing_equation(
            K_max=K_max,
            B=B,
            P_m=P_m,
            D=D,
            M_lower=M_lower,
            M_upper=M_upper,
            K_lower=K_lower,
            K_upper=K_upper,
            g=g,
            r_max=r_max,
            f_min=f_min,
            h=h,
            T=T,
            M_steps=M_steps,
            seed=seed if seed is not None else 0,
            device=device
        )
        
        return {
            'M_lower': float(M_lower),
            'M_upper': float(M_upper),
            'K_lower': float(K_lower),
            'K_upper': float(K_upper),
            'MOCU': float(MOCU_val)
        }
    except Exception as e:
        # Skip samples that fail (e.g., numerical issues)
        if worker_id == 0:  # Only print from first worker
            print(f"[WARNING] Sample generation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate training dataset for Swing MLP predictor')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to generate (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (overrides config)')
    parser.add_argument('--chunk_size', type=int, default=10,
                        help='Chunk size for multiprocessing')
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
    dataset_params = config.get('dataset', {})
    data_gen_params = config.get('data_generation', {})
    paths = config.get('paths', {})
    
    # System parameters
    topology = swing_params.get('topology', 'ieee14')
    coupling_strength = swing_params.get('coupling_strength', 1.0)
    damping = swing_params.get('damping', 0.1)
    base_power = swing_params.get('base_power', 1.0)
    M_lower_base = swing_params.get('M_lower', 0.5)
    M_upper_base = swing_params.get('M_upper', 2.0)
    K_lower_base = swing_params.get('K_lower', 0.1)
    K_upper_base = swing_params.get('K_upper', 1.0)
    r_max = swing_params.get('r_max', 0.5)
    f_min = swing_params.get('f_min', 49.5)
    
    # Dataset parameters
    num_samples = args.samples if args.samples is not None else dataset_params.get('samples_per_type', 1000)
    K_max = dataset_params.get('K_max', 20480)
    
    # Time parameters
    T = 10.0  # Time horizon (seconds)
    h = 1.0 / 160.0  # Time step (seconds)
    M_steps = int(T / h)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Number of workers
    num_workers = args.num_workers if args.num_workers is not None else data_gen_params.get('num_workers', max(1, mp.cpu_count() - 1))
    if num_workers == 0:
        num_workers = 1
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(paths.get('data_dir', 'data'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    print("Swing Equation MOCU Dataset Generation")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Number of buses: {N}")
    print(f"  - Topology: {topology}")
    print(f"  - Samples to generate: {num_samples}")
    print(f"  - Monte Carlo samples (K_max): {K_max}")
    print(f"  - M bounds: [{M_lower_base}, {M_upper_base}]")
    print(f"  - K bounds: [{K_lower_base}, {K_upper_base}]")
    print(f"  - Frequency constraints: r_max={r_max} Hz/s, f_min={f_min} Hz")
    print(f"  - Time: T={T}s, h={h}s, M_steps={M_steps}")
    print(f"  - Device: {device}")
    print(f"  - Workers: {num_workers}")
    print("=" * 80)
    
    # Set multiprocessing start method
    # CUDA + multiprocessing can cause hangs - use sequential for CUDA
    if device == 'cuda' and torch.cuda.is_available():
        if num_workers > 1:
            print(f"\n  ⚠ Warning: Multiprocessing with CUDA can cause hangs.")
            print(f"  ⚠ Switching to sequential processing (num_workers=1) for stability.")
            num_workers = 1
    
    # Prepare arguments for multiprocessing
    args_list = [
        (N, K_max, B, P_m, D, g, M_lower_base, M_upper_base,
         K_lower_base, K_upper_base, r_max, f_min, h, T, M_steps,
         device, i % num_workers if num_workers > 1 else None)
        for i in range(num_samples)
    ]
    
    # Test CUDA if using GPU
    if device == 'cuda':
        print(f"\n[INFO] Testing CUDA availability...")
        try:
            test_tensor = torch.randn(10, device=device)
            print(f"✓ CUDA is working (device: {torch.cuda.get_device_name(0)})")
        except Exception as e:
            print(f"⚠ CUDA test failed: {e}")
            print(f"  Falling back to CPU...")
            device = 'cpu'
            # Update device in args_list
            args_list = [
                (N, K_max, B, P_m, D, g, M_lower_base, M_upper_base,
                 K_lower_base, K_upper_base, r_max, f_min, h, T, M_steps,
                 device, i % num_workers if num_workers > 1 else None)
                for i in range(num_samples)
            ]
    
    # Generate samples
    print(f"\nGenerating {num_samples} samples...")
    print(f"[INFO] First sample may take 30-60 seconds (MOCU with K_max={K_max})...")
    start_time = time.time()
    
    if num_workers <= 1:
        # Sequential processing
        data = []
        with tqdm(total=num_samples, desc="Generating", unit="sample", ncols=100) as pbar:
            for idx, args_tuple in enumerate(args_list):
                sample_start_time = time.time()
                try:
                    if idx == 0:
                        print(f"\n[INFO] Computing first sample (this may take a while)...")
                        import sys
                        sys.stdout.flush()
                    sample = generate_single_sample(args_tuple)
                    sample_time = time.time() - sample_start_time
                    if sample is not None:
                        data.append(sample)
                        pbar.set_postfix({'Valid': len(data), 'Time': f'{sample_time:.1f}s'})
                        if idx == 0:
                            print(f"\n✓ First sample completed in {sample_time:.1f}s")
                    else:
                        pbar.set_postfix({'Valid': len(data), 'Skipped': idx + 1 - len(data)})
                except Exception as e:
                    print(f"\n[ERROR] Sample {idx+1} failed: {e}")
                    import traceback
                    traceback.print_exc()
                pbar.update(1)
    else:
        # Multiprocessing
        data = []
        with tqdm(total=num_samples, desc="Generating", unit="sample", ncols=100) as pbar:
            with mp.Pool(processes=num_workers) as pool:
                chunk_size = args.chunk_size
                for chunk_start in range(0, num_samples, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_samples)
                    chunk_args = args_list[chunk_start:chunk_end]
                    
                    chunk_results = pool.map(generate_single_sample, chunk_args)
                    
                    for sample in chunk_results:
                        if sample is not None:
                            data.append(sample)
                    
                    pbar.update(len(chunk_results))
    
    elapsed = time.time() - start_time
    print(f"\n✓ Generated {len(data)} valid samples in {elapsed:.1f}s")
    if len(data) > 0:
        print(f"  Average: {elapsed/len(data):.2f}s/sample, {len(data)/elapsed:.2f} samples/s")
    
    if len(data) == 0:
        print("\n⚠️  ERROR: No valid samples generated!")
        sys.exit(1)
    
    # Convert to numpy arrays
    M_lower_arr = np.array([d['M_lower'] for d in data], dtype=np.float32)
    M_upper_arr = np.array([d['M_upper'] for d in data], dtype=np.float32)
    K_lower_arr = np.array([d['K_lower'] for d in data], dtype=np.float32)
    K_upper_arr = np.array([d['K_upper'] for d in data], dtype=np.float32)
    MOCU_arr = np.array([d['MOCU'] for d in data], dtype=np.float32)
    
    # Save as .npz file (format expected by train_swing_mlp_predictor.py)
    output_file = output_dir / f'swing_mocu_data_{N}o.npz'
    np.savez(
        output_file,
        M_lower=M_lower_arr,
        M_upper=M_upper_arr,
        K_lower=K_lower_arr,
        K_upper=K_upper_arr,
        MOCU=MOCU_arr
    )
    
    print("\n" + "=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)
    print(f"Output file: {output_file}")
    print(f"Total samples: {len(data)}")
    print(f"\nMOCU Statistics:")
    print(f"  Mean: {np.mean(MOCU_arr):.6f}")
    print(f"  Std:  {np.std(MOCU_arr):.6f}")
    print(f"  Min:  {np.min(MOCU_arr):.6f}")
    print(f"  Max:  {np.max(MOCU_arr):.6f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
