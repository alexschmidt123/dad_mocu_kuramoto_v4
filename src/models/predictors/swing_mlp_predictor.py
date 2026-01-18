"""
Simple MLP Predictor for Swing Equation MOCU.

For second-order Kuramoto (swing equation), uncertainty is just (M, K) bounds:
- Input: [M_lower, M_upper, K_lower, K_upper] (4 scalars)
- Output: MOCU value (1 scalar)

This is much simpler than MPNN since uncertainty is not graph-structured.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SwingMLPPredictor(nn.Module):
    """
    Simple MLP for swing equation MOCU prediction.
    
    Input: [M_lower, M_upper, K_lower, K_upper] (4 scalars)
    Output: MOCU value (1 scalar)
    
    Architecture:
    - Input layer: 4 → 128
    - Hidden layer 1: 128 → 64
    - Hidden layer 2: 64 → 32
    - Output layer: 32 → 1
    """
    
    def __init__(self, n_hidden=[128, 64, 32], n_output=1, dropout=0.1):
        """
        Args:
            n_hidden: List of hidden layer sizes (default: [128, 64, 32])
            n_output: Output dimension (1 for MOCU)
            dropout: Dropout probability (default: 0.1)
        """
        super(SwingMLPPredictor, self).__init__()
        
        # Input layer: 4 features (M_lower, M_upper, K_lower, K_upper)
        self.input_dim = 4
        
        # Build layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in n_hidden:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, n_output))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, 4] where columns are [M_lower, M_upper, K_lower, K_upper]
        
        Returns:
            mocu_pred: Predicted MOCU [batch, 1]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [4] → [1, 4]
        
        # Ensure input has correct shape
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[-1]}")
        
        return self.network(x)  # [batch, 1]
    
    def predict_mocu(self, M_lower, M_upper, K_lower, K_upper, device='cuda'):
        """
        Convenience method to predict MOCU from bounds.
        
        Args:
            M_lower, M_upper: Inertia bounds (scalars or arrays)
            K_lower, K_upper: Control gain bounds (scalars or arrays)
            device: torch device
        
        Returns:
            mocu_pred: Predicted MOCU (scalar or array)
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensors if needed
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
            
            # Predict
            pred = self.forward(x)
            
            # Return as numpy if input was numpy
            if isinstance(M_lower, (int, float, np.ndarray)) and not isinstance(M_lower, torch.Tensor):
                return pred.cpu().numpy()
            return pred
