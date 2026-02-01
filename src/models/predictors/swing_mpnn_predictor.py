"""
MPNN MOCU Predictor for Swing Equation (design_part1.tex Section 7.2).

Uses IEEE-14 graph topology B_ij to map latent bounds (M_low, M_up, K_low, K_up)
and optional probe (bus, amplitude) to MOCU. Provides a graph-aware surrogate
for fast, reliable MOCU estimation when MLP is insufficient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


def _build_edge_index_from_B(B, device):
    """Build edge_index [2, E] and edge_weight [E] from coupling matrix B (nonzero entries)."""
    B = np.asarray(B)
    nz = np.nonzero(B)
    if len(nz[0]) == 0:
        # Fallback: self-loops or empty
        N = B.shape[0]
        edge_index = torch.stack([
            torch.arange(N, device=device, dtype=torch.long),
            torch.arange(N, device=device, dtype=torch.long)
        ], dim=0)
        edge_weight = torch.ones(N, device=device, dtype=torch.float32)
    else:
        edge_index = torch.tensor(np.stack([nz[0], nz[1]], axis=0), dtype=torch.long, device=device)
        edge_weight = torch.tensor(B[nz].astype(np.float32), device=device)
    return edge_index, edge_weight


class SwingMPNNPredictor(nn.Module):
    """
    MPNN that leverages graph topology B to predict MOCU from (M,K) bounds.
    design_part1.tex Section 7.2: map (θ_low, θ_up) and probe ξ to MOCU.
    """

    def __init__(self, B, node_dim=1, hidden_dim=64, out_dim=32, dropout=0.1,
                 use_probe=True, N_probe_buses=14):
        """
        Args:
            B: Coupling matrix [N, N] (numpy or tensor); used to build graph.
            node_dim: Node feature dimension (default 1, e.g. degree or constant).
            hidden_dim: GNN hidden dimension.
            out_dim: Graph embedding dimension before MLP.
            dropout: Dropout probability.
            use_probe: If True, accept optional probe_bus, probe_amplitude in forward.
            N_probe_buses: Number of buses (for probe one-hot when use_probe=True).
        """
        super().__init__()
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("SwingMPNNPredictor requires torch_geometric. Install with: pip install torch-geometric")

        B_np = np.asarray(B)
        self.N = B_np.shape[0]
        self.use_probe = use_probe
        self.N_probe_buses = N_probe_buses

        # Build static edge_index and edge_weight from B (registered as buffers for same device as params)
        edge_index, edge_weight = _build_edge_index_from_B(B_np, device='cpu')
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)

        # Node features: constant 1 (or could use degree)
        self.node_lin = nn.Linear(node_dim, hidden_dim)

        # 2-layer GCN
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

        # Global features: [M_lower, M_upper, K_lower, K_upper] = 4; optional probe = 1 + N_probe_buses or 2
        self.global_in_dim = 4
        if use_probe:
            self.global_in_dim += 1 + N_probe_buses  # probe_amplitude + probe_bus one-hot
        self.mlp = nn.Sequential(
            nn.Linear(out_dim + self.global_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = dropout

    def _graph_embed(self, x_node, edge_index, edge_weight, batch=None):
        # x_node: [num_nodes, node_dim]
        h = self.node_lin(x_node)
        h = F.relu(self.conv1(h, edge_index, edge_weight))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_weight))
        if batch is not None:
            graph_emb = global_mean_pool(h, batch)
        else:
            graph_emb = h.mean(dim=0, keepdim=True)
        return graph_emb

    def forward(self, x_bounds, probe_bus=None, probe_amplitude=None):
        """
        Args:
            x_bounds: [batch, 4] = [M_lower, M_upper, K_lower, K_upper]
            probe_bus: Optional [batch] int tensor (bus index) or None.
            probe_amplitude: Optional [batch] float tensor or None.

        Returns:
            mocu: [batch, 1]
        """
        device = x_bounds.device
        self.edge_index = self.edge_index.to(device)
        self.edge_weight = self.edge_weight.to(device)

        batch_size = x_bounds.shape[0]
        num_nodes = self.N
        # Single graph, same for all samples
        x_node = torch.ones(num_nodes, 1, device=device, dtype=x_bounds.dtype)
        graph_emb = self._graph_embed(x_node, self.edge_index, self.edge_weight, batch=None)
        graph_emb = graph_emb.expand(batch_size, -1)

        global_feat = [x_bounds]
        if self.use_probe and (probe_bus is not None or probe_amplitude is not None):
            if probe_amplitude is None:
                probe_amplitude = torch.zeros(batch_size, device=device, dtype=x_bounds.dtype)
            else:
                probe_amplitude = torch.as_tensor(probe_amplitude, device=device, dtype=x_bounds.dtype)
                if probe_amplitude.dim() == 0:
                    probe_amplitude = probe_amplitude.unsqueeze(0).expand(batch_size)
            if probe_bus is None:
                probe_bus = torch.zeros(batch_size, dtype=torch.long, device=device)
            else:
                probe_bus = torch.as_tensor(probe_bus, device=device, dtype=torch.long)
                if probe_bus.dim() == 0:
                    probe_bus = probe_bus.unsqueeze(0).expand(batch_size)
            one_hot = F.one_hot(probe_bus.clamp(0, self.N_probe_buses - 1), self.N_probe_buses).float()
            global_feat.append(probe_amplitude.unsqueeze(1))
            global_feat.append(one_hot)
        else:
            global_feat.append(torch.zeros(batch_size, 1, device=device, dtype=x_bounds.dtype))
            global_feat.append(torch.zeros(batch_size, self.N_probe_buses, device=device, dtype=x_bounds.dtype))

        x_global = torch.cat(global_feat, dim=1)
        combined = torch.cat([graph_emb, x_global], dim=1)
        return self.mlp(combined)

    def predict_mocu(self, M_lower, M_upper, K_lower, K_upper, probe_bus=None, probe_amplitude=None, device='cuda'):
        """Predict MOCU from bounds; probe_bus and probe_amplitude optional."""
        self.eval()
        with torch.no_grad():
            x = torch.stack([
                torch.as_tensor(M_lower, dtype=torch.float32, device=device),
                torch.as_tensor(M_upper, dtype=torch.float32, device=device),
                torch.as_tensor(K_lower, dtype=torch.float32, device=device),
                torch.as_tensor(K_upper, dtype=torch.float32, device=device),
            ], dim=-1)
            # Single sample: x is (4,) -> need (1, 4) for forward which expects [batch, 4]
            if x.dim() == 1:
                x = x.unsqueeze(0)
            out = self.forward(x, probe_bus=probe_bus, probe_amplitude=probe_amplitude)
            out = out.squeeze(-1).cpu().numpy()
            return out.item() if out.size == 1 else out
