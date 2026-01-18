"""
Training script for Swing MLP Predictor.

For second-order Kuramoto (swing equation) model.
Input: [M_lower, M_upper, K_lower, K_upper] (4 scalars)
Output: MOCU value (1 scalar)
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.predictors.swing_mlp_predictor import SwingMLPPredictor


class SwingMOCUDataset(Dataset):
    """Dataset for swing equation MOCU prediction."""
    
    def __init__(self, data_file):
        """
        Args:
            data_file: Path to .npz file containing:
                - M_lower, M_upper, K_lower, K_upper: [N_samples]
                - MOCU: [N_samples]
        """
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
            self.M_lower[idx],
            self.M_upper[idx],
            self.K_lower[idx],
            self.K_upper[idx]
        ], dtype=np.float32)
        y = np.array([self.MOCU[idx]], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def compute_statistics(dataset):
    """Compute normalization statistics."""
    all_inputs = []
    for i in range(len(dataset)):
        x, _ = dataset[i]
        all_inputs.append(x.numpy())
    
    all_inputs = np.array(all_inputs)  # [N, 4]
    mean = torch.from_numpy(all_inputs.mean(axis=0)).float()
    std = torch.from_numpy(all_inputs.std(axis=0)).float()
    std = torch.clamp(std, min=1e-8)  # Avoid division by zero
    
    return mean, std


def train_epoch(model, dataloader, optimizer, criterion, device, mean, std):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)
        
        # Normalize inputs
        x_norm = (x - mean.to(device)) / (std.to(device) + 1e-8)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(x_norm)
        loss = criterion(pred, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device, mean, std):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating", leave=False):
            x = x.to(device)
            y = y.to(device)
            
            # Normalize inputs
            x_norm = (x - mean.to(device)) / (std.to(device) + 1e-8)
            
            # Forward pass
            pred = model(x_norm)
            loss = criterion(pred, y)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Train Swing MLP Predictor')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (e.g., configs/ieee14_config.yaml)')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to training data .npz file (default: data/{model_name}_mocu_data.npz)')
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Train/validation split ratio')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config.get('training', {}).get('model_name', 'swing_mlp')
    
    # Determine data file
    if args.data_file is None:
        data_file = PROJECT_ROOT / 'data' / f'{model_name}_mocu_data.npz'
    else:
        data_file = Path(args.data_file)
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            f"Please generate training data first using generate_mocu_data.py"
        )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    full_dataset = SwingMOCUDataset(data_file)
    
    # Compute statistics on full dataset
    print("Computing normalization statistics...")
    mean, std = compute_statistics(full_dataset)
    print(f"Mean: {mean.numpy()}")
    print(f"Std: {std.numpy()}")
    
    # Split dataset
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = SwingMLPPredictor().to(device)
    print(f"Model architecture:\n{model}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    model_dir = PROJECT_ROOT / 'models' / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Model will be saved to: {model_dir}")
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, mean, std)
        val_loss = validate(model, val_loader, criterion, device, mean, std)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'config': {
                    'n_hidden': [128, 64, 32],
                    'n_output': 1,
                }
            }, model_dir / 'model.pth')
            
            # Save statistics
            torch.save({
                'mean': mean,
                'std': std,
            }, model_dir / 'statistics.pth')
            
            print(f"  ✓ Saved best model (val_loss = {val_loss:.6f})")
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {model_dir / 'model.pth'}")


if __name__ == '__main__':
    main()
