"""
MOCU Predictors Package for Swing Equation Model

This package contains neural network models for MOCU prediction for the 
second-order Kuramoto (swing equation) model.

Predictor:
- SwingMLPPredictor: Simple MLP for predicting MOCU from (M, K) bounds

Files:
- swing_mlp_predictor.py: MLP predictor for swing equation
- swing_predictor_utils.py: Utilities for loading/using Swing MLP predictor
- utils.py: Shared utility functions (minimal, for compatibility)

Based on: "Probing Signal-Based Inertia and Frequency Response Estimation 
for Power Systems with High Penetration of Inverter-Based Resources"
"""

# Import swing equation predictor
from .swing_mlp_predictor import SwingMLPPredictor

# Import utilities
from .swing_predictor_utils import (
    load_swing_mlp_predictor,
    predict_swing_mocu,
)

__all__ = [
    # Predictor
    'SwingMLPPredictor',
    # Utilities
    'load_swing_mlp_predictor',
    'predict_swing_mocu',
]
