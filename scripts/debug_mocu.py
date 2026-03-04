#!/usr/bin/env python3
"""
Debug MOCU computation: verify γ* and MOCU are non-trivial.
Usage: python scripts/debug_mocu.py [--config config/fast_config.yaml]
"""
import sys
from pathlib import Path
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.swing_equation_mocu import binary_search_gamma_star_batch, MOCU_swing_equation
from src.core.swing_equation_params import get_default_swing_equation_params

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/fast_config.yaml')
    args = p.parse_args()
    config_path = PROJECT_ROOT / args.config
    config = None
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    sw = (config or {}).get('swing_equation', {})
    N = (config or {}).get('N', 14)
    M_lower, M_upper = sw.get('M_lower', 0.01), sw.get('M_upper', 0.06)
    K_lower, K_upper = sw.get('K_lower', 0.05), sw.get('K_upper', 0.50)
    r_max, f_min = sw.get('r_max', 0.15), sw.get('f_min', 49.8)
    ref_bus = sw.get('reference_probe_bus')
    ref_amp = sw.get('reference_probe_amplitude')
    ref_dur = sw.get('reference_probe_duration', 2.0)
    system = get_default_swing_equation_params(N=N, topology=sw.get('topology', 'ieee14'),
        coupling_strength=sw.get('coupling_strength', 1.0), damping=sw.get('damping', 0.1),
        base_power=sw.get('base_power', 1.0), M_lower=M_lower, M_upper=M_upper,
        K_lower=K_lower, K_upper=K_upper)
    B, P_m, D, g = system['B'], system['P_m'], system['D'], system['g']
    h, T, M_steps = 1.0/160.0, 10.0, int(10.0 / (1.0/160.0))
    K_test = 8
    print(f"Debug: r_max={r_max}, ref_amp={ref_amp}, K_test={K_test}")
    np.random.seed(42)
    M_batch = np.random.uniform(M_lower, M_upper, size=K_test).astype(np.float64)
    K_batch = np.random.uniform(K_lower, K_upper, size=K_test).astype(np.float64)
    dev = 'cpu'  # Use CPU to avoid CUDA numerical differences
    gamma_star = binary_search_gamma_star_batch(M_batch, K_batch, B, P_m, D, g,
        r_max=r_max, f_min=f_min, h=h, T=T, M_steps=M_steps,
        reference_probe_bus=ref_bus, reference_probe_amplitude=ref_amp, reference_probe_duration=ref_dur,
        device=dev)
    gamma_hat = np.median(gamma_star)
    mocu = np.mean(np.abs(gamma_star - gamma_hat))
    print(f"γ*: min={gamma_star.min():.2f}, max={gamma_star.max():.2f}, MOCU={mocu:.4f}")
    full_mocu = MOCU_swing_equation(64, B, P_m, D, M_lower, M_upper, K_lower, K_upper, g,
        r_max=r_max, f_min=f_min, h=h, T=T, M_steps=M_steps,
        reference_probe_bus=ref_bus, reference_probe_amplitude=ref_amp, reference_probe_duration=ref_dur,
        seed=42, device=dev)
    print(f"MOCU_swing_equation: {full_mocu:.4f}")

if __name__ == '__main__':
    main()
