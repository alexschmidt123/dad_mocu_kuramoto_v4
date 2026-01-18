"""
Utility functions for MOCU predictors.

Shared utilities used by multiple predictor types.
"""

import torch
import numpy as np
from torch_geometric.data import Data


def get_edge_index(N):
    """
    Create edge indices for fully connected directed graph (excluding self-loops).
    
    Args:
        N: Number of nodes
    
    Returns:
        edge_index: [2, num_edges] tensor with edge connections
    """
    edge_index = []
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_index.append([i, j])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


def get_edge_attr_from_bounds(a_lower, a_upper, N):
    """
    Extract edge attributes from bound matrices.
    
    Args:
        a_lower: Lower bound matrix [N, N]
        a_upper: Upper bound matrix [N, N]
        N: Number of nodes
    
    Returns:
        edge_attr: [num_edges, 2] tensor with [a_lower, a_upper] for each edge
    """
    edge_attr = []
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_attr.append([a_lower[i, j], a_upper[i, j]])
    return torch.tensor(edge_attr, dtype=torch.float32)  # [num_edges, 2]


def get_node_features_with_degree(w, a_lower, a_upper, device='cpu'):
    """
    Create node features including frequency and degree (for Coutinho 2013 model).
    
    Args:
        w: Natural frequencies [N]
        a_lower: Lower bounds [N, N]
        a_upper: Upper bounds [N, N]
        device: torch device
    
    Returns:
        x: Node features [N, 2] with [frequency, normalized_degree]
    """
    N = len(w)
    
    # Compute degree from coupling bounds (effective degree)
    avg_coupling = (a_lower + a_upper) / 2.0
    degrees = np.sum(avg_coupling > 0, axis=1)  # Count non-zero connections
    # Normalize degrees
    degrees = degrees.astype(np.float32) / (N - 1) if N > 1 else degrees.astype(np.float32)
    
    # Stack frequency and degree
    x = torch.from_numpy(np.column_stack([
        w.astype(np.float32),
        degrees
    ])).to(device)  # [N, 2]: [frequency, normalized_degree]
    
    return x


def create_graph_data(w, a_lower, a_upper, device='cpu', include_degree=True):
    """
    Create PyTorch Geometric Data object from state.
    
    Args:
        w: Natural frequencies [N]
        a_lower: Lower bounds [N, N]
        a_upper: Upper bounds [N, N]
        device: torch device
        include_degree: If True, include degree features (for Coutinho 2013 model)
    
    Returns:
        data: PyG Data object for MPNN predictors
    """
    N = len(w)
    
    # Node features: [frequency, degree, ...]
    if include_degree:
        # Compute degree from coupling bounds (effective degree)
        # For Coutinho 2013: degree affects frequency, so we can infer it
        # Use average coupling strength as proxy for degree
        avg_coupling = (a_lower + a_upper) / 2.0
        degrees = np.sum(avg_coupling > 0, axis=1)  # Count non-zero connections
        # Normalize degrees
        degrees = degrees.astype(np.float32) / (N - 1) if N > 1 else degrees.astype(np.float32)
        
        x = torch.from_numpy(np.column_stack([
            w.astype(np.float32),
            degrees
        ]))  # [N, 2]: [frequency, normalized_degree]
    else:
        # Original: only frequency
        x = torch.from_numpy(w.astype(np.float32)).unsqueeze(-1)  # [N, 1]
    
    # Edge indices and attributes
    edge_index = get_edge_index(N)
    edge_attr = get_edge_attr_from_bounds(a_lower, a_upper, N)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = data.to(device)
    
    return data


def matrix_to_vector(x, N):
    """
    Convert N×N matrix to vector of lower triangular elements.
    
    Used for MLP input: [a_lower, a_upper] → flattened vector
    """
    x = np.tril(x, -1)  # Lower triangular
    x = x.ravel()[np.flatnonzero(x)]
    return x


def pre2R_mpnn(predictions, P_syn_list, N):
    """
    Convert MPNN predictions and probabilities to R matrix (expected remaining MOCU).
    
    Args:
        predictions: Array of predictions, pairs are [syn, non-syn] for each (i,j) pair
        P_syn_list: List of synchronization probabilities, one per (i,j) pair
        N: Number of nodes
    
    Returns:
        R_matrix: [N, N] matrix with expected remaining MOCU for each pair
    """
    R_matrix = np.zeros((N, N))
    pair_idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            # Each pair has 2 scenarios: syn (idx*2) and non-syn (idx*2+1)
            syn_idx = pair_idx * 2
            nonsyn_idx = pair_idx * 2 + 1
            
            if syn_idx < len(predictions) and nonsyn_idx < len(predictions):
                pred_syn = float(predictions[syn_idx])
                pred_nonsyn = float(predictions[nonsyn_idx])
                P_syn = float(P_syn_list[pair_idx]) if pair_idx < len(P_syn_list) else 0.5
                P_nonsyn = 1.0 - P_syn
                
                R = P_syn * pred_syn + P_nonsyn * pred_nonsyn
                R_matrix[i, j] = R
                R_matrix[j, i] = R
            pair_idx += 1
    return R_matrix

