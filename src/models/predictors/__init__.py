"""
MOCU Predictors Package for Swing Equation Model

Uses MPNN for MOCU estimation (paper first-order Kuramoto iNN/NN used MPNN).
- SwingMPNNPredictor: MPNN over graph B for MOCU from (M, K) bounds and optional probe
- load_swing_mpnn_predictor, load_swing_mocu_predictor, predict_swing_mocu
"""

try:
    from .swing_mpnn_predictor import SwingMPNNPredictor, TORCH_GEOMETRIC_AVAILABLE
except ImportError:
    SwingMPNNPredictor = None
    TORCH_GEOMETRIC_AVAILABLE = False

from .swing_predictor_utils import (
    load_swing_mpnn_predictor,
    load_swing_mocu_predictor,
    predict_swing_mocu,
)

__all__ = [
    'SwingMPNNPredictor',
    'TORCH_GEOMETRIC_AVAILABLE',
    'load_swing_mpnn_predictor',
    'load_swing_mocu_predictor',
    'predict_swing_mocu',
]
