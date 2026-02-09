"""
Validate MPNN MOCU predictor quality.

Reports MSE, MAE, R², Pearson correlation, and max absolute error on a test set.
Optionally runs a small ODE spot-check (true MOCU vs MPNN) for definitive quality.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictors.swing_predictor_utils import load_swing_mocu_predictor, predict_swing_mocu
from src.core.swing_equation_params import get_default_swing_equation_params


def load_test_data(data_file, test_fraction=0.2, seed=43):
    """Load .npz and return (M_lower, M_upper, K_lower, K_upper, MOCU) for test set."""
    data = np.load(data_file)
    n = len(data['MOCU'])
    np.random.seed(seed)
    idx = np.random.permutation(n)
    n_test = max(1, int(n * test_fraction))
    test_idx = idx[-n_test:]
    return (
        data['M_lower'].astype(np.float32)[test_idx],
        data['M_upper'].astype(np.float32)[test_idx],
        data['K_lower'].astype(np.float32)[test_idx],
        data['K_upper'].astype(np.float32)[test_idx],
        data['MOCU'].astype(np.float32)[test_idx],
    )


def compute_metrics(y_true, y_pred):
    """MSE, MAE, R², Pearson r, max abs error."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    if np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        r = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        r = 0.0
    max_ae = np.max(np.abs(y_true - y_pred))
    return {'mse': mse, 'mae': mae, 'r2': r2, 'pearson_r': r, 'max_abs_err': max_ae}


def main():
    parser = argparse.ArgumentParser(description='Validate MPNN MOCU predictor quality')
    parser.add_argument('--config', type=str, required=True, help='Config YAML (for B and N)')
    parser.add_argument('--model_name', type=str, default=None, help='Model name (default: from config training.model_name)')
    parser.add_argument('--data_file', type=str, default=None, help='.npz MOCU data (default: data/{model_name}_mocu_data.npz)')
    parser.add_argument('--test_fraction', type=float, default=0.2, help='Fraction of data used as test (default: 0.2)')
    parser.add_argument('--ode_spot_check', type=int, default=0, help='Number of samples to compare with true ODE MOCU (0=skip, slow)')
    parser.add_argument('--seed', type=int, default=43)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_name = args.model_name or config.get('training', {}).get('model_name', 'swing_mpnn')
    N = config.get('N', 14)
    swing_params = config.get('swing_equation', {})
    params = get_default_swing_equation_params(
        N=N,
        topology=swing_params.get('topology', 'ieee14'),
        coupling_strength=swing_params.get('coupling_strength', 1.0),
        damping=swing_params.get('damping', 0.1),
        base_power=swing_params.get('base_power', 1.0),
        M_lower=swing_params.get('M_lower', 0.01),
        M_upper=swing_params.get('M_upper', 0.06),
        K_lower=swing_params.get('K_lower', 0.05),
        K_upper=swing_params.get('K_upper', 0.50),
    )
    B = params['B']

    data_file = args.data_file or (PROJECT_ROOT / 'data' / f'{model_name}_mocu_data.npz')
    data_file = Path(data_file)
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, mean, std = load_swing_mocu_predictor(model_name=model_name, device=device, B=B, N=N)

    Ml, Mu, Kl, Ku, MOCU_true = load_test_data(data_file, test_fraction=args.test_fraction, seed=args.seed)
    n_test = len(MOCU_true)
    print(f"Test set: {n_test} samples from {data_file.name}")

    predictions = []
    for i in range(n_test):
        pred = predict_swing_mocu(model, mean, std, float(Ml[i]), float(Mu[i]), float(Kl[i]), float(Ku[i]), device=device)
        predictions.append(float(pred) if np.isscalar(pred) else float(pred.flat[0]))
    predictions = np.array(predictions, dtype=np.float32)

    metrics = compute_metrics(MOCU_true, predictions)
    print("\n--- MPNN vs data (ground-truth MOCU from same data generation) ---")
    print(f"  MSE:           {metrics['mse']:.6f}")
    print(f"  MAE:           {metrics['mae']:.6f}")
    print(f"  R²:            {metrics['r2']:.4f}")
    print(f"  Pearson r:     {metrics['pearson_r']:.4f}")
    print(f"  Max abs err:   {metrics['max_abs_err']:.6f}")

    if args.ode_spot_check > 0:
        try:
            from src.core.swing_equation_mocu import MOCU_swing_equation
            K_max = config.get('dataset', {}).get('K_max', config.get('experiment', {}).get('K_max', 512))
            r_max = swing_params.get('r_max', 0.5)
            f_min = swing_params.get('f_min', 59.5)
            n_spot = min(args.ode_spot_check, n_test)
            print(f"\n--- ODE spot-check ({n_spot} samples, K_max={K_max}) ---")
            true_ode = []
            for i in range(n_spot):
                mocu_ode = MOCU_swing_equation(
                    K_max=K_max, B=params['B'], P_m=params['P_m'], D=params['D'],
                    M_lower=float(Ml[i]), M_upper=float(Mu[i]),
                    K_lower=float(Kl[i]), K_upper=float(Ku[i]),
                    g=params['g'], r_max=r_max, f_min=f_min, seed=i, device=str(device),
                )
                true_ode.append(float(mocu_ode))
            true_ode = np.array(true_ode)
            pred_spot = predictions[:n_spot]
            spot_metrics = compute_metrics(true_ode, pred_spot)
            print(f"  MSE:           {spot_metrics['mse']:.6f}")
            print(f"  MAE:           {spot_metrics['mae']:.6f}")
            print(f"  R²:            {spot_metrics['r2']:.4f}")
            print(f"  Pearson r:     {spot_metrics['pearson_r']:.4f}")
            print(f"  Max abs err:   {spot_metrics['max_abs_err']:.6f}")
        except Exception as e:
            print(f"\nODE spot-check failed: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
