"""
Utility functions for loading and using Swing MOCU predictor (MPNN only).

For second-order Kuramoto (swing equation) model.
Paper (first-order Kuramoto iNN/NN) used MPNN for MOCU estimation.
"""

import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


def load_swing_mpnn_predictor(B, model_name, device='cuda', N_probe_buses=14):
    """
    Load Swing MPNN predictor for MOCU estimation.
    Expects models/<model_name>/model_mpnn.pth and statistics.pth.
    Returns (model, mean, std).
    """
    try:
        from src.models.predictors.swing_mpnn_predictor import SwingMPNNPredictor, TORCH_GEOMETRIC_AVAILABLE
    except ImportError:
        raise ImportError("Swing MPNN requires torch_geometric. Install with: pip install torch-geometric")
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("Swing MPNN requires torch_geometric.")

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model_path = PROJECT_ROOT / 'models' / model_name / 'model_mpnn.pth'
    stats_path = PROJECT_ROOT / 'models' / model_name / 'statistics.pth'

    if not model_path.exists():
        raise FileNotFoundError(
            f"Swing MPNN model not found: {model_path}. "
            f"Train MPNN predictor and save as model_mpnn.pth."
        )

    B_np = np.asarray(B)
    model = SwingMPNNPredictor(B_np, use_probe=True, N_probe_buses=N_probe_buses).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    mean = torch.zeros(4, device=device)
    std = torch.ones(4, device=device)
    if stats_path.exists():
        stats = torch.load(stats_path, map_location=device, weights_only=False)
        mean = stats.get('mean', mean)
        std = stats.get('std', std)
        # MOCU output normalization (used by predict_mocu for denormalization)
        if 'mocu_mean' in stats and 'mocu_std' in stats:
            m = stats['mocu_mean']
            s = stats['mocu_std']
            if torch.is_tensor(m):
                m, s = m.to(device), s.to(device)
            else:
                m = torch.tensor(float(m), device=device, dtype=torch.float32)
                s = torch.tensor(float(s), device=device, dtype=torch.float32)
            model.register_buffer('mocu_mean', m.reshape(-1))
            model.register_buffer('mocu_std', s.reshape(-1))

    print(f"[SwingMPNN] Loaded model '{model_name}' on {device}")
    return model, mean, std


def load_swing_mocu_predictor(model_name, device='cuda', B=None, N=14):
    """
    Load MOCU predictor (MPNN). B (coupling matrix) is required.
    Returns (model, mean, std).
    """
    if B is None:
        raise ValueError("B (coupling matrix) required for MPNN MOCU predictor.")
    return load_swing_mpnn_predictor(B, model_name, device=device, N_probe_buses=N)


def predict_swing_mocu(model, mean, std, M_lower, M_upper, K_lower, K_upper, device='cuda',
                       probe_bus=None, probe_amplitude=None):
    """
    Predict MOCU using MPNN. probe_bus and probe_amplitude condition the prediction (optional).
    """
    model.eval()
    with torch.no_grad():
        out = model.predict_mocu(
            M_lower, M_upper, K_lower, K_upper,
            probe_bus=probe_bus, probe_amplitude=probe_amplitude, device=device
        )
        if isinstance(out, np.ndarray):
            return out
        return out.cpu().numpy().squeeze() if hasattr(out, 'cpu') else out
