"""
Utility functions for loading and using Swing MLP predictor.

For second-order Kuramoto (swing equation) model.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.predictors.swing_mlp_predictor import SwingMLPPredictor


def load_swing_mlp_predictor(model_name, device='cuda'):
    """
    Load Swing MLP predictor model and statistics.
    
    Args:
        model_name: Name of trained model (e.g., 'ieee14', 'fast_config')
        device: torch device
    
    Returns:
        model: Loaded SwingMLPPredictor model (in eval mode)
        mean: Normalization mean [4] for [M_lower, M_upper, K_lower, K_upper]
        std: Normalization std [4]
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Model and statistics paths
    model_path = PROJECT_ROOT / 'models' / model_name / 'model.pth'
    stats_path = PROJECT_ROOT / 'models' / model_name / 'statistics.pth'
    
    if not model_path.exists() or not stats_path.exists():
        raise FileNotFoundError(
            f"Swing MLP model or statistics not found for {model_name}.\n"
            f"Searched paths:\n"
            f"  - {model_path}\n"
            f"  - {stats_path}\n"
            f"Please train Swing MLP predictor first."
        )
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        state_dict = checkpoint
    
    # Create model (default architecture)
    model = SwingMLPPredictor().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # Load statistics
    stats = torch.load(stats_path, map_location=device, weights_only=False)
    mean = stats.get('mean', torch.zeros(4, device=device))
    std = stats.get('std', torch.ones(4, device=device))
    
    print(f"[SwingMLP] Loaded model '{model_name}' on {device}")
    
    return model, mean, std


def predict_swing_mocu(model, mean, std, M_lower, M_upper, K_lower, K_upper, device='cuda'):
    """
    Predict MOCU using Swing MLP predictor with normalization.
    
    Args:
        model: SwingMLPPredictor model
        mean: Normalization mean [4]
        std: Normalization std [4]
        M_lower, M_upper: Inertia bounds (scalars or arrays)
        K_lower, K_upper: Control gain bounds (scalars or arrays)
        device: torch device
    
    Returns:
        mocu_pred: Predicted MOCU (scalar or array)
    """
    model.eval()
    with torch.no_grad():
        # Convert to tensors
        if not isinstance(M_lower, torch.Tensor):
            M_lower = torch.tensor(M_lower, dtype=torch.float32, device=device)
        if not isinstance(M_upper, torch.Tensor):
            M_upper = torch.tensor(M_upper, dtype=torch.float32, device=device)
        if not isinstance(K_lower, torch.Tensor):
            K_lower = torch.tensor(K_lower, dtype=torch.float32, device=device)
        if not isinstance(K_upper, torch.Tensor):
            K_upper = torch.tensor(K_upper, dtype=torch.float32, device=device)
        
        # Stack into [batch, 4]
        x = torch.stack([M_lower, M_upper, K_lower, K_upper], dim=-1)
        x = x.to(device)
        
        # Normalize
        x_norm = (x - mean) / (std + 1e-8)
        
        # Predict
        pred = model(x_norm)
        
        # Return as numpy if input was numpy
        if isinstance(M_lower, (int, float)) or (isinstance(M_lower, np.ndarray) and not isinstance(M_lower, torch.Tensor)):
            return pred.cpu().numpy().squeeze()
        return pred.squeeze()
