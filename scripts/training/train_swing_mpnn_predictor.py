"""
Training script for Swing MPNN MOCU Predictor.

For second-order Kuramoto (swing equation) model. Paper (first-order iNN/NN) used MPNN.
Input: [M_lower, M_upper, K_lower, K_upper] (4 scalars), optional probe at inference
Output: MOCU value (1 scalar)
Uses same .npz data as MOCU data generation; B matrix from config.
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import yaml

# File lives in scripts/training/ -> project root is parent.parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictors.swing_mpnn_predictor import SwingMPNNPredictor, TORCH_GEOMETRIC_AVAILABLE
from src.core.swing_equation_params import get_default_swing_equation_params

if not TORCH_GEOMETRIC_AVAILABLE:
    raise ImportError("Swing MPNN requires torch_geometric. Install with: pip install torch-geometric")


class SwingMOCUDataset(Dataset):
    """Dataset for swing equation MOCU prediction (same format as MOCU data generation)."""
    def __init__(self, data_file):
        data = np.load(data_file)
        self.M_lower = data['M_lower'].astype(np.float32)
        self.M_upper = data['M_upper'].astype(np.float32)
        self.K_lower = data['K_lower'].astype(np.float32)
        self.K_upper = data['K_upper'].astype(np.float32)
        self.MOCU = data['MOCU'].astype(np.float32)
        print(f"Loaded dataset: {len(self.MOCU)} samples")

    def __len__(self):
        return len(self.MOCU)

    def __getitem__(self, idx):
        x = np.array([
            self.M_lower[idx], self.M_upper[idx],
            self.K_lower[idx], self.K_upper[idx]
        ], dtype=np.float32)
        y = np.array([self.MOCU[idx]], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def main():
    parser = argparse.ArgumentParser(description='Train Swing MPNN MOCU Predictor')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to .npz MOCU data (default: data/{model_name}_mocu_data.npz)')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--train_split', type=float, default=0.8)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_name = config.get('training', {}).get('model_name', 'swing_mpnn')
    N = config.get('N', 14)
    swing_params = config.get('swing_equation', {})
    topology = swing_params.get('topology', 'ieee14')
    coupling_strength = swing_params.get('coupling_strength', 1.0)
    damping = swing_params.get('damping', 0.1)
    base_power = swing_params.get('base_power', 1.0)
    M_lower_base = swing_params.get('M_lower', 0.3)
    M_upper_base = swing_params.get('M_upper', 2.0)
    K_lower_base = swing_params.get('K_lower', 0.05)
    K_upper_base = swing_params.get('K_upper', 0.50)

    params = get_default_swing_equation_params(
        N=N, topology=topology, coupling_strength=coupling_strength,
        damping=damping, base_power=base_power,
        M_lower=M_lower_base, M_upper=M_upper_base,
        K_lower=K_lower_base, K_upper=K_upper_base
    )
    B = params['B']

    if args.data_file is None:
        data_file = PROJECT_ROOT / 'data' / f'{model_name}_mocu_data.npz'
    else:
        data_file = Path(args.data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}. Run step1 (generate_mocu_data) first.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    full_dataset = SwingMOCUDataset(data_file)
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    B_np = np.asarray(B)
    model = SwingMPNNPredictor(B_np, use_probe=True, N_probe_buses=N).to(device)
    print(f"Model:\n{model}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model_dir = PROJECT_ROOT / 'models' / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    mean = torch.zeros(4, device=device)
    std = torch.ones(4, device=device)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, probe_bus=None, probe_amplitude=None)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x, probe_bus=None, probe_amplitude=None)
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch, 'val_loss': val_loss,
                'config': {'N': N, 'B_shape': B_np.shape}
            }, model_dir / 'model_mpnn.pth')
            torch.save({'mean': mean, 'std': std}, model_dir / 'statistics.pth')
            print(f"  ✓ Saved best model (val_loss = {val_loss:.6f})")

    # Final validation metrics (estimation quality)
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, probe_bus=None, probe_amplitude=None)
            val_preds.append(pred.cpu().numpy().ravel())
            val_true.append(y.cpu().numpy().ravel())
    val_preds = np.concatenate(val_preds)
    val_true = np.concatenate(val_true)
    val_mae = np.mean(np.abs(val_true - val_preds))
    ss_res = np.sum((val_true - val_preds) ** 2)
    ss_tot = np.sum((val_true - np.mean(val_true)) ** 2)
    val_r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    val_r = np.corrcoef(val_true, val_preds)[0, 1] if np.std(val_true) > 1e-12 else 0.0

    print(f"\nTraining complete. Best val loss (MSE): {best_val_loss:.6f}")
    print(f"Validation quality: MAE = {val_mae:.6f}, R² = {val_r2:.4f}, Pearson r = {val_r:.4f}")
    print(f"Model saved to: {model_dir / 'model_mpnn.pth'}")
    print(f"Run validation: python scripts/evaluation/validate_mpnn_mocu.py --config <config> [--ode_spot_check 5]")


if __name__ == '__main__':
    main()
