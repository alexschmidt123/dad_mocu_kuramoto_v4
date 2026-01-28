"""
Evaluation Script for Baseline Methods (Swing Equation Model)

This script evaluates baseline OED methods (iNN, NN, ODE, ENTROPY, RANDOM) using torchdiffeq
for MOCU computation with the swing equation model.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --methods "ODE,iNN,NN"
"""

import sys
import time
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import yaml

# Suppress verbose warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*DataLoader.*deprecated.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import swing equation modules
try:
    from src.core.swing_equation_mocu import MOCU_swing_equation
    from src.core.swing_equation_params import (
        get_default_swing_equation_params,
        sample_uncertain_parameters
    )
    from src.core.swing_equation_ode import (
        solve_swing_equation_ode,
        check_frequency_synchronization
    )
    from scripts.training.generate_dad_data import (
        generate_random_system,
        perform_probe_experiment,
        update_bounds
    )
    SWING_EQUATION_AVAILABLE = True
except ImportError as e:
    SWING_EQUATION_AVAILABLE = False
    print(f"[ERROR] Swing equation modules not available: {e}")
    sys.exit(1)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate OED methods with swing equation')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated list of methods to evaluate (e.g., "ODE,iNN,NN")')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file (optional, uses env vars if not provided)')
    args = parser.parse_args()
    
    # ========== Configuration ==========
    def safe_getenv_int(key, default):
        """Get environment variable as int, handling empty strings."""
        val = os.getenv(key, default)
        return int(val) if val else int(default)
    
    # Load config if provided
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
    
    # Get parameters from config or environment
    if config:
        N = config.get('N', 14)
        swing_params = config.get('swing_equation', {})
        experiment_params = config.get('experiment', {})
        it_idx = experiment_params.get('it_idx', 10)
        update_cnt = experiment_params.get('update_count', 10)
        K_max = experiment_params.get('K_max', 20480)
        numberOfSimulationsPerMethod = experiment_params.get('num_simulations', 10)
        
        # Swing equation parameters
        topology = swing_params.get('topology', 'ieee14')
        coupling_strength = swing_params.get('coupling_strength', 1.0)
        damping = swing_params.get('damping', 0.1)
        base_power = swing_params.get('base_power', 1.0)
        M_lower_base = swing_params.get('M_lower', 0.3)
        M_upper_base = swing_params.get('M_upper', 2.0)
        K_lower_base = swing_params.get('K_lower', 0.05)
        K_upper_base = swing_params.get('K_upper', 0.50)
        r_max = swing_params.get('r_max', 0.5)
        f_min = swing_params.get('f_min', 49.5)
        probe_duration = swing_params.get('probe_duration', 2.0)
        probe_amplitudes = swing_params.get('probe_amplitudes', [0.5, 1.0, 2.0])
    else:
        # Fallback to environment variables
        it_idx = safe_getenv_int('EVAL_IT_IDX', '10')
        update_cnt = safe_getenv_int('EVAL_UPDATE_CNT', '10')
        N = safe_getenv_int('EVAL_N', '14')
        K_max = safe_getenv_int('EVAL_K_MAX', '20480')
        numberOfSimulationsPerMethod = safe_getenv_int('EVAL_NUM_SIMULATIONS', '10')
        
        # Default swing equation parameters
        topology = 'ieee14'
        coupling_strength = 1.0
        damping = 0.1
        base_power = 1.0
        M_lower_base = 0.3
        M_upper_base = 2.0
        K_lower_base = 0.05
        K_upper_base = 0.50
        r_max = 0.5
        f_min = 49.5
        probe_duration = 2.0
        probe_amplitudes = [0.5, 1.0, 2.0]
    
    result_folder = os.getenv('RESULT_FOLDER', str(PROJECT_ROOT / 'results' / 'default'))
    os.makedirs(result_folder, exist_ok=True)
    
    # Time parameters
    deltaT = 1.0 / 160.0
    TVirtual = 5
    MVirtual = int(TVirtual / deltaT)
    TReal = 10.0  # Swing equation uses longer time horizon
    MReal = int(TReal / deltaT)
    h = deltaT
    T = TReal
    
    # ========== Method Selection ==========
    if args.methods:
        method_names = [m.strip() for m in args.methods.split(',')]
    else:
        # Default: baseline methods only (DAD/iDAD not yet updated for swing equation)
        method_names = ['iNN', 'NN', 'ODE', 'ENTROPY', 'RANDOM']
    
    # Print configuration
    print(f"\n{'='*80}")
    print(f"Evaluation Configuration (Swing Equation)")
    print(f"{'='*80}")
    print(f"  N={N}, update_cnt={update_cnt}, it_idx={it_idx}, K_max={K_max}")
    print(f"  num_simulations={numberOfSimulationsPerMethod}")
    print(f"  methods={method_names}")
    print(f"  result_folder={result_folder}")
    print(f"  M bounds: [{M_lower_base}, {M_upper_base}]")
    print(f"  K bounds: [{K_lower_base}, {K_upper_base}]")
    print(f"{'='*80}\n")
    
    # ========== Generate system parameters (fixed, known) ==========
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
    
    # ========== Choose MOCU backend ==========
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using torchdiffeq for MOCU computation (device: {device})")
    
    # Clean up any old environment variables
    if 'USE_PYCUDA_FOR_BASELINES' in os.environ:
        del os.environ['USE_PYCUDA_FOR_BASELINES']
    
    numberOfVaildSimulations = 0
    numberOfSimulations = 0
    
    # ========== Results storage ==========
    save_MOCU_matrix = np.zeros([update_cnt + 1, len(method_names), numberOfSimulationsPerMethod])
    
    # ========== Main simulation loop ==========
    sim_pbar = tqdm(total=numberOfSimulationsPerMethod, desc="Simulations", unit="sim", ncols=100, mininterval=1.0)
    
    while numberOfVaildSimulations < numberOfSimulationsPerMethod:
        sim_pbar.set_description(f"Simulation {numberOfVaildSimulations + 1}/{numberOfSimulationsPerMethod}")
        
        # Generate random system with initial uncertainty bounds and true parameters
        random_seed = int(numberOfSimulations)
        (M_lower_init, M_upper_init, K_lower_init, K_upper_init, M_true, K_true, init_sync) = \
            generate_random_system(
                N, B, P_m, D, g,
                M_lower_base, M_upper_base, K_lower_base, K_upper_base,
                seed=random_seed
            )
        
        numberOfSimulations += 1
        
        # Check if system is already synchronized (optional - can skip for speed)
        # For swing equation, we check frequency synchronization
        try:
            state_traj = solve_swing_equation_ode(
                B, P_m, D, M_true, K_true, g,
                h=h, M_steps=MReal, T=T,
                device=device, timeout=5.0
            )
            N_buses = len(P_m)
            omega_traj = state_traj[:, N_buses:]  # Extract frequency part
            is_synced = check_frequency_synchronization(omega_traj, MReal)
            
            if is_synced:
                sim_pbar.write(f'  ⚠️  System {numberOfSimulations}: Already synchronized (skipping - no learning needed)')
                continue
            else:
                sim_pbar.write(f'  ✓ System {numberOfSimulations}: Not synchronized (good for OED evaluation)')
        except Exception as e:
            # If sync check fails, continue anyway
            sim_pbar.write(f'  ⚠️  System {numberOfSimulations}: Sync check failed ({e}), continuing...')
        
        # Save true parameters
        true_params_file = os.path.join(result_folder, f'paramTrue_M_K_{numberOfVaildSimulations}.txt')
        np.savetxt(true_params_file, [M_true, K_true], fmt='%.64e')
        
        # Save initial bounds
        init_bounds_file = os.path.join(result_folder, f'paramInitialBounds_{numberOfVaildSimulations}.txt')
        np.savetxt(init_bounds_file, [M_lower_init, M_upper_init, K_lower_init, K_upper_init], fmt='%.64e')
        
        # ========== Compute initial MOCU ==========
        timeMOCU = time.time()
        it_temp_val = np.zeros(it_idx)
        
        # Initial MOCU computation using swing equation MOCU
        with tqdm(total=it_idx, desc="  Initial MOCU", leave=False, unit="iter", ncols=80, mininterval=0.5) as pbar:
            for l in range(it_idx):
                it_temp_val[l] = MOCU_swing_equation(
                    K_max=K_max,
                    B=B,
                    P_m=P_m,
                    D=D,
                    M_lower=M_lower_init,
                    M_upper=M_upper_init,
                    K_lower=K_lower_init,
                    K_upper=K_upper_init,
                    g=g,
                    r_max=r_max,
                    f_min=f_min,
                    h=h,
                    T=T,
                    M_steps=MReal,
                    seed=l,
                    device=device
                )
                pbar.update(1)
        
        MOCUInitial = np.mean(it_temp_val)
        elapsed = time.time() - timeMOCU
        sim_pbar.write(f'  Initial MOCU: {MOCUInitial:.6f} ({elapsed:.1f}s)')
        
        # Save initial MOCU for this simulation (for DAD/iDAD to use same value)
        initial_mocu_file = os.path.join(result_folder, f'initial_MOCU_{numberOfVaildSimulations}.txt')
        np.savetxt(initial_mocu_file, [MOCUInitial], fmt='%.64e')
        
        # ========== Evaluate each method ==========
        method_pbar = tqdm(method_names, desc="  Methods", leave=False, unit="method", ncols=80, mininterval=1.0)
        
        # Monkey-patch print() to redirect to tqdm.write() during method execution
        original_print = print
        def redirect_print(*args, **kwargs):
            """Redirect print to tqdm.write() to avoid interfering with progress bars."""
            msg = ' '.join(str(arg) for arg in args)
            if any(marker in msg for marker in ['[iNN]', '[NN]', '[ODE]', '[iODE]', '[ENTROPY]', '[RANDOM]']):
                if any(important in msg for important in ['Warning:', 'Error:', 'ERROR']):
                    method_pbar.write(f'  {msg}')
            else:
                original_print(*args, **kwargs)
        
        for method_idx, method_name in enumerate(method_pbar):
            method_pbar.set_postfix({'method': method_name})
            
            method_start_time = time.time()
            
            try:
                # Temporarily redirect print for method initialization and execution
                import builtins
                builtins.print = redirect_print
                
                # Lazy import methods from src.methods
                if method_name == 'iNN':
                    from src.methods import iNN_Method
                    method = iNN_Method(N, K_max, deltaT, MReal, TReal, it_idx, 
                                       model_name=os.getenv('MOCU_MODEL_NAME', f'cons{N}'),
                                       probe_amplitudes=probe_amplitudes,
                                       probe_duration=probe_duration)
                
                elif method_name == 'NN':
                    from src.methods import NN_Method
                    method = NN_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                      model_name=os.getenv('MOCU_MODEL_NAME', f'cons{N}'),
                                      probe_amplitudes=probe_amplitudes,
                                      probe_duration=probe_duration)
                
                elif method_name == 'ODE':
                    from src.methods import ODE_Method
                    method = ODE_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                       B=B, P_m=P_m, D=D, g=g,
                                       probe_amplitudes=probe_amplitudes,
                                       probe_duration=probe_duration,
                                       r_max=r_max, f_min=f_min)
                
                elif method_name == 'ENTROPY':
                    from src.methods import ENTROPY_Method
                    method = ENTROPY_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                           probe_amplitudes=probe_amplitudes,
                                           probe_duration=probe_duration,
                                           B=B)
                
                elif method_name == 'RANDOM':
                    from src.methods import RANDOM_Method
                    method = RANDOM_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                          probe_amplitudes=probe_amplitudes,
                                          probe_duration=probe_duration,
                                          seed=numberOfVaildSimulations)
                
                elif method_name in ['DAD', 'DAD_MOCU', 'IDAD_MOCU']:
                    # DAD/iDAD methods not yet updated for swing equation
                    method_pbar.write(f'  ⚠️  Skipping {method_name}: Not yet updated for swing equation model')
                    method_pbar.write(f'  ⚠️  DAD/iDAD requires train_dad_policy.py and dad_eval.py updates')
                    continue
                
                else:
                    print(f"Unknown method: {method_name}")
                    continue
                
                # Run the method (with print redirection active)
                MOCUCurve, experimentSequence, timeComplexity = method.run_episode(
                    M_lower_init=M_lower_init,
                    M_upper_init=M_upper_init,
                    K_lower_init=K_lower_init,
                    K_upper_init=K_upper_init,
                    M_true=M_true,
                    K_true=K_true,
                    B=B,
                    P_m=P_m,
                    D=D,
                    g=g,
                    probe_amplitudes=probe_amplitudes,
                    probe_duration=probe_duration,
                    r_max=r_max,
                    f_min=f_min,
                    update_cnt=update_cnt,
                    initial_mocu=MOCUInitial
                )
                
                # Restore original print
                import builtins
                builtins.print = original_print
                
                total_time = time.time() - method_start_time
                method_pbar.set_postfix({
                    'Time': f'{total_time:.1f}s',
                    'Final MOCU': f'{MOCUCurve[-1]:.6f}'
                })
                
                # Save results
                outMOCUFile = open(os.path.join(result_folder, f'{method_name}_MOCU.txt'), 'a')
                outTimeFile = open(os.path.join(result_folder, f'{method_name}_timeComplexity.txt'), 'a')
                outSequenceFile = open(os.path.join(result_folder, f'{method_name}_sequence.txt'), 'a')
                
                np.savetxt(outMOCUFile, MOCUCurve.reshape(1, MOCUCurve.shape[0]), delimiter="\t")
                np.savetxt(outTimeFile, timeComplexity.reshape(1, timeComplexity.shape[0]), delimiter="\t")
                np.savetxt(outSequenceFile, experimentSequence, delimiter="\t")
                
                outMOCUFile.close()
                outTimeFile.close()
                outSequenceFile.close()
                
                save_MOCU_matrix[:, method_idx, numberOfVaildSimulations] = MOCUCurve
            
            except Exception as e:
                # Restore original print before error handling
                import builtins
                builtins.print = original_print
                method_pbar.write(f'  ✗ Error running {method_name}: {e}')
                import traceback
                traceback.print_exc()
                continue
        
        numberOfVaildSimulations += 1
        sim_pbar.update(1)
        sim_pbar.set_postfix({'Completed': f'{numberOfVaildSimulations}/{numberOfSimulationsPerMethod}'})
    
    sim_pbar.close()
    
    # ========== Final summary ==========
    print(f"\n{'='*80}")
    print("All simulations completed!")
    print(f"{'='*80}")
    
    mean_MOCU_matrix = np.mean(save_MOCU_matrix, axis=2)
    print("\nMean MOCU values across all simulations:")
    print(mean_MOCU_matrix)
    
    outMOCUFile = open(os.path.join(result_folder, 'mean_MOCU.txt'), 'w')
    np.savetxt(outMOCUFile, mean_MOCU_matrix, delimiter="\t")
    outMOCUFile.close()
    
    print(f"\n✓ Results saved to: {result_folder}")
