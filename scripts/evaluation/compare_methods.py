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

# File in scripts/evaluation/ -> project root = parent.parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import swing equation modules
try:
    from src.core.swing_equation_mocu import MOCU_swing_equation
    from src.core.swing_equation_params import (
        get_default_swing_equation_params,
        sample_uncertain_parameters
    )
    from scripts.data_generation.generate_dad_data import (
        generate_random_system,
        perform_probe_experiment,
        update_bounds,
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
    
    # Load config if provided (resolve relative paths against project root)
    config = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            print(f"[WARNING] Config file not found: {config_path} (config will not be loaded)")
    
    # Get parameters from config or environment
    mocu_model_name_from_config = None
    if config:
        N = config.get('N', 14)
        swing_params = config.get('swing_equation', {})
        experiment_params = config.get('experiment', {})
        training_params = config.get('training', {})
        mocu_model_name_from_config = training_params.get('model_name')
        it_idx = experiment_params.get('it_idx', 10)
        update_cnt = experiment_params.get('update_count', 4)  # T=4 (5 MOCU points: 0..4)
        K_max = experiment_params.get('K_max', 20480)
        numberOfSimulationsPerMethod = experiment_params.get('num_simulations', 10)
        
        # Swing equation parameters
        topology = swing_params.get('topology', 'ieee14')
        coupling_strength = swing_params.get('coupling_strength', 1.0)
        damping = swing_params.get('damping', 0.1)
        base_power = swing_params.get('base_power', 1.0)
        M_lower_base = swing_params.get('M_lower', 0.01)
        M_upper_base = swing_params.get('M_upper', 0.06)
        K_lower_base = swing_params.get('K_lower', 0.05)
        K_upper_base = swing_params.get('K_upper', 0.50)
        r_max = swing_params.get('r_max', 0.1)
        f_min = swing_params.get('f_min', 49.8)
        probe_duration = swing_params.get('probe_duration', 2.0)
        probe_amplitudes = swing_params.get('probe_amplitudes', [0.5, 1.0, 2.0])
        reference_probe_bus = swing_params.get('reference_probe_bus')
        reference_probe_amplitude = swing_params.get('reference_probe_amplitude')
        reference_probe_duration = swing_params.get('reference_probe_duration', 2.0)
        sigma = swing_params.get('sigma', 0.05)
        uncertainty_ratio_min = experiment_params.get('uncertainty_ratio_min', 0.3)
        update_strength = experiment_params.get('update_strength', 0.05)
    else:
        mocu_model_name_from_config = None
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
        M_lower_base = 0.01
        M_upper_base = 0.06
        K_lower_base = 0.05
        K_upper_base = 0.50
        r_max = 0.1
        f_min = 49.8
        probe_duration = 2.0
        probe_amplitudes = [0.5, 1.0, 2.0]
        reference_probe_bus = None
        reference_probe_amplitude = None
        reference_probe_duration = 2.0
        sigma = 0.05
        uncertainty_ratio_min = 0.3
        update_strength = 0.05
    
    # All results go under experiments/ (run.sh sets RESULT_FOLDER via EXP_EVAL_DIR)
    result_folder = os.getenv('RESULT_FOLDER')
    if not result_folder:
        result_folder = str(PROJECT_ROOT / 'experiments' / 'standalone_default' / 'eval')
        import warnings
        warnings.warn(f'RESULT_FOLDER not set; using {result_folder}')
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
        method_names = [m.strip() for m in args.methods.split(',') if m.strip()]
    elif config:
        methods_cfg = config.get('experiment', {}).get('methods')
        if methods_cfg:
            method_names = list(methods_cfg) if isinstance(methods_cfg, (list, tuple)) else [m.strip() for m in str(methods_cfg).split(',') if m.strip()]
        else:
            method_names = ['RANDOM', 'ENTROPY', 'ODE']
    else:
        method_names = ['RANDOM', 'ENTROPY', 'ODE', 'iNN', 'NN', 'DAD']
    
    # Design space: N buses × len(probe_amplitudes) = number of (bus, amplitude) designs ξ
    num_designs = N * len(probe_amplitudes)
    
    # Print configuration
    print(f"\n{'='*80}")
    print(f"Evaluation Configuration (Swing Equation)")
    print(f"{'='*80}")
    print(f"  N={N}, update_cnt={update_cnt}, it_idx={it_idx}, K_max={K_max}")
    print(f"  num_simulations={numberOfSimulationsPerMethod}")
    print(f"  methods={method_names}")
    print(f"  result_folder={result_folder}")
    print(f"  Design space: N_buses × num_amplitudes = {N} × {len(probe_amplitudes)} = {num_designs} designs ξ")
    if os.environ.get("DEBUG_OED_DESIGNS") == "1":
        print("  [DEBUG_OED_DESIGNS=1] Will log each method's design and bounds per step to stderr (to diagnose same-MOCU).")
    print(f"  M bounds: [{M_lower_base}, {M_upper_base}]")
    print(f"  K bounds: [{K_lower_base}, {K_upper_base}]")
    print(f"  probe_amplitudes={probe_amplitudes}")
    print(f"  reference_probe: bus={reference_probe_bus}, amplitude={reference_probe_amplitude}, duration={reference_probe_duration}")
    print(f"  observation_noise sigma={sigma} Hz/s")
    if reference_probe_bus is None or reference_probe_amplitude is None or (isinstance(reference_probe_amplitude, (int, float)) and reference_probe_amplitude <= 0):
        raise ValueError(
            "swing_equation.reference_probe_bus and reference_probe_amplitude (positive) must be set in config "
            "so γ* is non-trivial and MOCU differs across methods (design §3)."
        )
    mocu_model = os.getenv('MOCU_MODEL_NAME', '')
    if mocu_model:
        print(f"  MOCU model (iNN/NN): {mocu_model}")
    else:
        print(f"  MOCU model (iNN/NN): not set; iNN/NN will use config training.model_name or fail if missing")
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
    use_pycuda = os.environ.get('USE_PYCUDA', '1') in ('1', 'true', 'yes')
    from src.core.swing_equation_mocu import get_mocu_swing_computer
    _mocu_fn, backend = get_mocu_swing_computer(use_pycuda=use_pycuda)
    print(f"Using {backend} for MOCU computation (device: {device})")
    
    # Clean up any old environment variables
    if 'USE_PYCUDA_FOR_BASELINES' in os.environ:
        del os.environ['USE_PYCUDA_FOR_BASELINES']
    
    numberOfVaildSimulations = 0
    numberOfSimulations = 0
    
    # ========== Results storage ==========
    save_MOCU_matrix = np.zeros([update_cnt + 1, len(method_names), numberOfSimulationsPerMethod])
    method_status = {m: {'status': 'OK', 'reason': ''} for m in method_names}
    
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
                seed=random_seed, uncertainty_ratio_min=uncertainty_ratio_min
            )
        
        numberOfSimulations += 1
        sim_pbar.write(f'  ✓ System {numberOfSimulations}: Using system')
        
        # Save true parameters
        true_params_file = os.path.join(result_folder, f'paramTrue_M_K_{numberOfVaildSimulations}.txt')
        np.savetxt(true_params_file, [M_true, K_true], fmt='%.64e')
        
        # Save initial bounds
        init_bounds_file = os.path.join(result_folder, f'paramInitialBounds_{numberOfVaildSimulations}.txt')
        np.savetxt(init_bounds_file, [M_lower_init, M_upper_init, K_lower_init, K_upper_init], fmt='%.64e')
        
        # ========== Compute initial MOCU (batched: one call with K_max*it_idx) ==========
        # Use SAME bounds as methods (M_lower_init, etc.) for a self-consistent curve.
        # Step 0 = MOCU before any probes; steps 1+ = MOCU after each probe.
        # (Using full base bounds made step 0 artificially large and caused a misleading one-step drop.)
        timeMOCU = time.time()
        K_total = K_max * it_idx
        MOCUInitial = _mocu_fn(
            K_max=K_total,
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
            reference_probe_bus=reference_probe_bus,
            reference_probe_amplitude=reference_probe_amplitude,
            reference_probe_duration=reference_probe_duration,
            seed=random_seed,
            device=device
        )
        elapsed = time.time() - timeMOCU
        sim_pbar.write(f'  Initial MOCU: {MOCUInitial:.6f} ({elapsed:.1f}s)')
        if MOCUInitial < 1e-6:
            sim_pbar.write('  WARNING: Initial MOCU ≈ 0. Set swing_equation.reference_probe_amplitude larger so γ* is non-trivial (design §3).')
        
        # Save initial MOCU for this simulation (for DAD to use same value)
        initial_mocu_file = os.path.join(result_folder, f'initial_MOCU_{numberOfVaildSimulations}.txt')
        np.savetxt(initial_mocu_file, [MOCUInitial], fmt='%.64e')
        
        # ========== Evaluate each method ==========
        method_pbar = tqdm(method_names, desc="  Methods", leave=False, unit="method", ncols=80, mininterval=1.0)
        
        # Monkey-patch print() to redirect to tqdm.write() during method execution
        original_print = print
        def redirect_print(*args, **kwargs):
            """Redirect print to tqdm.write() to avoid interfering with progress bars."""
            msg = ' '.join(str(arg) for arg in args)
            if any(marker in msg for marker in ['[iNN]', '[NN]', '[ODE]', '[iODE]', '[ENTROPY]', '[RANDOM]', '[DAD']):
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
                
                # Resolve MOCU predictor model name: env MOCU_MODEL_NAME, else config training.model_name, else cons{N}
                _mocu_model = os.getenv('MOCU_MODEL_NAME') or mocu_model_name_from_config or f'cons{N}'
                
                # Lazy import methods from src.methods
                if method_name == 'iNN':
                    from src.methods import iNN_Method
                    method = iNN_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                       model_name=_mocu_model,
                                       probe_amplitudes=probe_amplitudes,
                                       probe_duration=probe_duration,
                                       B=B)
                
                elif method_name == 'NN':
                    from src.methods import NN_Method
                    method = NN_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                      model_name=_mocu_model,
                                      probe_amplitudes=probe_amplitudes,
                                      probe_duration=probe_duration,
                                      B=B)
                
                elif method_name == 'ODE':
                    from src.methods import ODE_Method
                    _ref_bus = 0 if reference_probe_bus is None else reference_probe_bus
                    _ref_amp = 0.5 if reference_probe_amplitude is None else reference_probe_amplitude
                    _ref_dur = reference_probe_duration or 2.0
                    method = ODE_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                       B=B, P_m=P_m, D=D, g=g,
                                       probe_amplitudes=probe_amplitudes,
                                       probe_duration=probe_duration,
                                       r_max=r_max, f_min=f_min, sigma=sigma,
                                       reference_probe_bus=_ref_bus,
                                       reference_probe_amplitude=_ref_amp,
                                       reference_probe_duration=_ref_dur,
                                       M_lower_base=M_lower_base, M_upper_base=M_upper_base,
                                       K_lower_base=K_lower_base, K_upper_base=K_upper_base)
                
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
                
                elif method_name == 'DAD':
                    from src.methods.dad_policy import DAD_MOCU_Method
                    policy_path = os.environ.get('DAD_POLICY_PATH') or None
                    method = DAD_MOCU_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                            policy_model_path=policy_path,
                                            probe_amplitudes=probe_amplitudes,
                                            probe_duration=probe_duration,
                                            B=B)
                
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
                    initial_mocu=MOCUInitial,
                    reference_probe_bus=reference_probe_bus,
                    reference_probe_amplitude=reference_probe_amplitude,
                    reference_probe_duration=reference_probe_duration,
                    sigma=sigma,
                    update_strength=update_strength
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
                
                # Diagnostic: print design sequence for first sim so we can see if methods differ
                if os.environ.get("DEBUG_OED_DESIGNS") == "1" and numberOfVaildSimulations == 0:
                    method_pbar.write(f"  [DEBUG] {method_name} design_sequence={experimentSequence}")
            
            except Exception as e:
                import builtins
                builtins.print = original_print
                reason = f"{type(e).__name__}: {str(e)[:200]}"
                method_status[method_name] = {'status': 'SKIPPED', 'reason': reason}
                method_pbar.write(f'  ✗ {method_name} skipped: {reason}')
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

    # Primary metrics: Terminal MOCU (most important) and AUC (area under MOCU curve)
    # mean_MOCU_matrix shape: (steps, methods)
    terminal_MOCU = mean_MOCU_matrix[-1, :]  # final-step mean per method
    x_steps = np.arange(update_cnt + 1, dtype=float)
    _trapz = getattr(np, 'trapezoid', np.trapz)
    auc_per_method = np.array([_trapz(mean_MOCU_matrix[:, j], x_steps) for j in range(len(method_names))])
    print("\n--- Primary metrics ---")
    print("Terminal MOCU (main metric, lower=better):")
    for j, m in enumerate(method_names):
        print(f"  {m}: {terminal_MOCU[j]:.6f}")
    print("AUC (area under MOCU curve, lower=better):")
    for j, m in enumerate(method_names):
        print(f"  {m}: {auc_per_method[j]:.6f}")

    outMOCUFile = open(os.path.join(result_folder, 'mean_MOCU.txt'), 'w')
    np.savetxt(outMOCUFile, mean_MOCU_matrix, delimiter="\t")
    outMOCUFile.close()

    # Save metrics for easy parsing
    metrics_lines = ["method\tterminal_MOCU\tAUC"]
    for j, m in enumerate(method_names):
        metrics_lines.append(f"{m}\t{terminal_MOCU[j]:.6f}\t{auc_per_method[j]:.6f}")
    with open(os.path.join(result_folder, 'metrics.txt'), 'w') as f:
        f.write('\n'.join(metrics_lines))

    # ========== Generate report ==========
    exp_root = Path(result_folder).parent
    config_path_str = str(args.config) if args.config else 'N/A'
    report_lines = [
        "",
        "--- Evaluation (Step 3) ---",
        f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"N={N}, update_cnt={update_cnt}, it_idx={it_idx}, K_max={K_max}",
        f"Simulations: {numberOfSimulationsPerMethod}",
        "",
        "Method Status:",
    ]
    for m in method_names:
        st = method_status[m]
        report_lines.append(f"  {m}: {st['status']}" + (f" - {st['reason']}" if st['reason'] else ""))
    report_lines.extend([
        "",
        "Mean MOCU (rows=steps, cols=methods):",
        np.array2string(mean_MOCU_matrix, precision=6),
        "",
        "--- Primary metrics ---",
        "Terminal MOCU (main metric, lower=better):",
    ])
    for j, m in enumerate(method_names):
        report_lines.append(f"  {m}: {terminal_MOCU[j]:.6f}")
    report_lines.extend([
        "AUC (area under MOCU curve, lower=better):",
    ])
    for j, m in enumerate(method_names):
        report_lines.append(f"  {m}: {auc_per_method[j]:.6f}")
    report_path = exp_root / 'report.txt'
    mode = 'a' if report_path.exists() else 'w'
    if mode == 'w':
        report_lines = [
            "=" * 60,
            "MOCU-OED Experiment Report",
            "=" * 60,
            f"Config: {config_path_str}",
            f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ] + report_lines
    with open(report_path, mode) as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✓ Results saved to: {result_folder}")
    print(f"✓ Report saved to: {report_path}")
