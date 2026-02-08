"""
DAD Method Evaluation Script

This script evaluates the DAD method using the same initial MOCU values
as the baseline methods. It loads the initial MOCU from baseline results
to ensure fair comparison.
"""

import sys
import time
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# File in scripts/evaluation/ -> project root = parent.parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Swing-equation only; no first-order Kuramoto path.


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate DAD method with baseline initial MOCU')
    parser.add_argument('--baseline_results', type=str, required=True,
                        help='Path to baseline results folder (contains initial MOCU info)')
    parser.add_argument('--result_folder', type=str, default=None,
                        help='Result folder for DAD results (default: baseline_results)')
    parser.add_argument('--method_name', type=str, default='DAD',
                        help='Method name for output files (default: DAD)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML (required for swing-equation evaluation)')
    args = parser.parse_args()
    
    # ========== Configuration ==========
    def safe_getenv_int(key, default):
        """Get environment variable as int, handling empty strings."""
        val = os.getenv(key, default)
        return int(val) if val else int(default)
    
    it_idx = safe_getenv_int('EVAL_IT_IDX', '10')
    update_cnt = safe_getenv_int('EVAL_UPDATE_CNT', '10')
    N = safe_getenv_int('EVAL_N', '5')
    K_max = safe_getenv_int('EVAL_K_MAX', '20480')
    numberOfSimulationsPerMethod = safe_getenv_int('EVAL_NUM_SIMULATIONS', '10')
    
    # Use baseline results folder if result_folder not specified
    baseline_results = Path(args.baseline_results)
    if args.result_folder:
        result_folder = Path(args.result_folder)
    else:
        result_folder = baseline_results
    
    result_folder.mkdir(parents=True, exist_ok=True)
    
    # Detect swing-equation baseline (compare_methods saves paramInitialBounds_* for swing)
    swing_bounds_file = baseline_results / 'paramInitialBounds_0.txt'
    is_swing = swing_bounds_file.exists()
    
    print(f"DAD Evaluation Configuration:")
    print(f"  Method: {args.method_name}")
    print(f"  N={N}, update_cnt={update_cnt}, it_idx={it_idx}, K_max={K_max}")
    print(f"  num_simulations={numberOfSimulationsPerMethod}")
    print(f"  result_folder={result_folder}")
    print(f"  baseline_results={baseline_results}")
    if is_swing:
        print(f"  Model: swing_equation")
    
    # Time parameters
    deltaT = 1.0 / 160.0
    TVirtual = 5
    MVirtual = int(TVirtual / deltaT)
    TReal = 10.0  # Swing equation uses 10s
    MReal = int(TReal / deltaT)
    
    if is_swing:
        # ========== Swing-equation DAD evaluation ==========
        import yaml
        import torch
        config_path = args.config or os.getenv('CONFIG_FILE')
        if not config_path or not Path(config_path).exists():
            raise FileNotFoundError(
                "Swing-equation DAD evaluation requires --config or CONFIG_FILE pointing to config YAML."
            )
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        swing_params = config.get('swing_equation', {})
        from src.core.swing_equation_params import get_default_swing_equation_params
        system_params = get_default_swing_equation_params(
            N=N,
            topology=swing_params.get('topology', 'ieee14'),
            coupling_strength=swing_params.get('coupling_strength', 1.0),
            damping=swing_params.get('damping', 0.1),
            base_power=swing_params.get('base_power', 1.0),
            M_lower=swing_params.get('M_lower', 0.3),
            M_upper=swing_params.get('M_upper', 2.0),
            K_lower=swing_params.get('K_lower', 0.05),
            K_upper=swing_params.get('K_upper', 0.50),
        )
        B, P_m, D, g = system_params['B'], system_params['P_m'], system_params['D'], system_params['g']
        r_max = swing_params.get('r_max', 0.5)
        f_min = swing_params.get('f_min', 59.5)
        probe_duration = swing_params.get('probe_duration', 2.0)
        probe_amplitudes = swing_params.get('probe_amplitudes', [0.5, 1.0, 2.0])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        policy_path = Path(os.environ.get('DAD_POLICY_PATH', ''))
        if not policy_path.exists():
            raise RuntimeError("DAD_POLICY_PATH must point to trained swing DAD policy (.pth)")
        from src.methods.dad_policy import DAD_MOCU_Method
        method = DAD_MOCU_Method(
            N, K_max, deltaT, MReal, TReal, it_idx,
            policy_model_path=str(policy_path),
            probe_amplitudes=probe_amplitudes,
            probe_duration=probe_duration,
            B=B,
        )
        save_MOCU_matrix = np.zeros([update_cnt + 1, 1, numberOfSimulationsPerMethod])
        for sim in range(numberOfSimulationsPerMethod):
            bounds_file = baseline_results / f'paramInitialBounds_{sim}.txt'
            true_file = baseline_results / f'paramTrue_M_K_{sim}.txt'
            mocu_file = baseline_results / f'initial_MOCU_{sim}.txt'
            if not bounds_file.exists() or not true_file.exists() or not mocu_file.exists():
                raise FileNotFoundError(f"Baseline files not found for sim {sim}")
            bounds = np.loadtxt(bounds_file)
            M_lower_init, M_upper_init, K_lower_init, K_upper_init = bounds[0], bounds[1], bounds[2], bounds[3]
            M_true, K_true = np.loadtxt(true_file)
            MOCUInitial = float(np.loadtxt(mocu_file))
            MOCUCurve, experimentSequence, timeComplexity = method.run_episode(
                M_lower_init, M_upper_init, K_lower_init, K_upper_init,
                M_true, K_true, B, P_m, D, g,
                probe_amplitudes, probe_duration,
                r_max=r_max, f_min=f_min,
                update_cnt=update_cnt, initial_mocu=MOCUInitial
            )
            save_MOCU_matrix[:, 0, sim] = MOCUCurve
            outMOCU = open(result_folder / f'{args.method_name}_MOCU.txt', 'a')
            outTime = open(result_folder / f'{args.method_name}_timeComplexity.txt', 'a')
            outSeq = open(result_folder / f'{args.method_name}_sequence.txt', 'a')
            np.savetxt(outMOCU, MOCUCurve.reshape(1, -1), delimiter='\t')
            np.savetxt(outTime, timeComplexity.reshape(1, -1), delimiter='\t')
            np.savetxt(outSeq, experimentSequence, delimiter='\t')
            outMOCU.close()
            outTime.close()
            outSeq.close()
        mean_MOCU = np.mean(save_MOCU_matrix, axis=2)
        with open(result_folder / 'mean_MOCU.txt', 'a') as f:
            np.savetxt(f, mean_MOCU, delimiter='\t')
        print(f"\n✓ DAD (swing) evaluation complete: {result_folder}")
        print(f"  Final mean MOCU: {mean_MOCU[-1, 0]:.6f}")
        sys.exit(0)
    
    # This project is swing-equation only; no first-order Kuramoto path.
    print("ERROR: Baseline results are not from swing-equation evaluation (paramInitialBounds_*.txt missing).")
    print("Run baseline evaluation (Step 3) with a swing config first.")
    sys.exit(1)

