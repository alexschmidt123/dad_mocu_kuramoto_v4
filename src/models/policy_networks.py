"""
Deep Adaptive Design Policy Network for MOCU-based Optimal Experimental Design

This module implements a policy network that learns to select optimal experiments
sequentially to minimize terminal MOCU (Model-based Objective-based Characterization of Uncertainty).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch.nn import Sequential, Linear, ReLU, GRU
import numpy as np
import os


class DADPolicyNetwork(nn.Module):
    """
    Policy network that outputs a probability distribution over available experiment pairs.
    
    Architecture:
    1. Graph encoder: Encodes current state (w, a_lower, a_upper) using GNN
    2. History encoder: Encodes past (action, observation) pairs using LSTM
    3. Action decoder: Outputs logits for each possible (i,j) pair
    """
    
    def __init__(self, N, hidden_dim=64, encoding_dim=32, num_message_passing=3, 
                 history_encoder_type='lstm', max_history_len=10):
        """
        Args:
            N: Number of oscillators
            hidden_dim: Hidden dimension for LSTM and MLPs
            encoding_dim: Dimension for graph embeddings
            num_message_passing: Number of message passing layers
            history_encoder_type: 'lstm' (order-dependent), 'sum' (order-independent), or 'cat' (concatenation)
            max_history_len: Maximum history length for 'cat' encoder (default: 10)
        """
        super(DADPolicyNetwork, self).__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.num_actions = N * (N - 1) // 2  # Number of unique pairs
        self.history_encoder_type = history_encoder_type
        
        # ========== Graph State Encoder (similar to iNN) ==========
        # Support both original (1 feature) and Coutinho 2013 (2 features: frequency + degree)
        self.lin0 = nn.Linear(2, encoding_dim)  # Node features: [frequency, degree] for Coutinho 2013
        
        # Message passing with edge features [a_lower, a_upper]
        edge_nn = Sequential(
            Linear(2, 128),
            ReLU(),
            Linear(128, encoding_dim * encoding_dim)
        )
        self.conv = NNConv(encoding_dim, encoding_dim, edge_nn, aggr='mean')
        self.gru = GRU(encoding_dim, encoding_dim)
        
        # Graph-level pooling
        self.set2set = Set2Set(encoding_dim, processing_steps=3)
        self.graph_mlp = nn.Sequential(
            Linear(2 * encoding_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )
        
        # ========== History Encoder (LSTM or Set-Equivariant Sum) ==========
        # Each history item: (i, j, observation) → embedding
        self.history_embed = nn.Embedding(N * N + 2, hidden_dim // 4)  # For i, j indices
        self.obs_embed = nn.Embedding(2, hidden_dim // 4)  # For sync/non-sync observation
        
        if history_encoder_type == 'lstm':
            # LSTM: Order-dependent (captures temporal dependencies)
            # Order-dependent (for time-dependent problems)
            self.history_lstm = nn.LSTM(
                input_size=hidden_dim // 2,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )
        elif history_encoder_type == 'sum':
            # Set-equivariant sum: Order-independent
            # Each (i, j, obs) pair encoded independently, then summed
            self.pair_encoder = nn.Sequential(
                Linear(hidden_dim // 2, encoding_dim),
                ReLU(),
                Linear(encoding_dim, encoding_dim)
            )
        elif history_encoder_type == 'cat':
            # Concatenation: Concatenates all encodings
            # Fixed-size concatenation of all history encodings
            self.max_history_len = max_history_len
            self.pair_encoder = nn.Sequential(
                Linear(hidden_dim // 2, encoding_dim),
                ReLU(),
                Linear(encoding_dim, encoding_dim)
            )
            # Output will be [batch, max_history_len * encoding_dim]
        else:
            raise ValueError(f"history_encoder_type must be 'lstm', 'sum', or 'cat', got {history_encoder_type}")
        
        # ========== Action Decoder ==========
        # Enhanced decoder that can use expected MOCU features if provided
        # Input: state_emb + history_emb + (optional) expected_mocu_features
        self.action_decoder = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),  # State + history (base)
            ReLU(),
            nn.Dropout(0.1),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, self.num_actions)  # Logits for each pair
        )
        
        # Alternative decoder that incorporates expected MOCU features
        # This is used when expected_mocu_features are provided (enhanced mode)
        self.action_decoder_with_mocu = nn.Sequential(
            Linear(hidden_dim * 2 + 1, hidden_dim),  # State + history + expected_mocu (per action)
            ReLU(),
            nn.Dropout(0.1),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)  # Single logit per action (will be stacked)
        )
    
    def encode_state(self, state_data):
        """
        Encode current state (w, a_lower, a_upper) using GNN.
        """
        device = next(self.parameters()).device
        state_data = state_data.to(device)
        
        x = state_data.x
        edge_index = state_data.edge_index
        edge_attr = state_data.edge_attr
        batch = state_data.batch if hasattr(state_data, 'batch') else None
        
        # Initial embedding
        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)
        
        # === FIX: Disable cuDNN for message passing ===
        cudnn_enabled = torch.backends.cudnn.enabled
        try:
            torch.backends.cudnn.enabled = False
            for _ in range(3):
                m = F.relu(self.conv(out, edge_index, edge_attr))
                m_input = m.unsqueeze(0)
                out, h = self.gru(m_input, h)
                out = out.squeeze(0)
        finally:
            torch.backends.cudnn.enabled = cudnn_enabled
        
        # === FIX: Handle pooling with cuDNN disabled ===
        if batch is not None:
            # Batched case - use Set2Set with cuDNN disabled
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            cudnn_enabled = torch.backends.cudnn.enabled
            cudnn_benchmark = torch.backends.cudnn.benchmark
            try:
                torch.backends.cudnn.enabled = False
                torch.backends.cudnn.benchmark = False
                
                if batch.device != device:
                    batch = batch.to(device)
                if not batch.is_contiguous():
                    batch = batch.contiguous()
                if not out.is_contiguous():
                    out = out.contiguous()
                
                out = self.set2set(out, batch)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            finally:
                torch.backends.cudnn.enabled = cudnn_enabled
                torch.backends.cudnn.benchmark = cudnn_benchmark
        else:
            # Single graph - use simpler pooling
            from torch_geometric.nn import global_mean_pool, global_max_pool
            batch_tensor = torch.zeros(out.size(0), dtype=torch.long, device=device)
            mean_pool = global_mean_pool(out, batch_tensor)
            max_pool = global_max_pool(out, batch_tensor)
            out = torch.cat([mean_pool, max_pool], dim=-1)
        
        state_embedding = self.graph_mlp(out)
        
        return state_embedding

    def encode_history(self, history_data):
        """
        Encode history of past (action, observation) pairs.
        
        Supports two modes:
        - 'lstm': Order-dependent sequential processing (for time-dependent problems)
        - 'sum': Order-independent set-equivariant sum
        """
        if history_data is None or len(history_data) == 0:
            batch_size = 1
            device = next(self.parameters()).device
            if self.history_encoder_type == 'lstm':
                return torch.zeros(batch_size, self.hidden_dim, device=device)
            else:  # sum
                return torch.zeros(batch_size, self.encoding_dim, device=device)
        
        # Convert history to tensor
        if isinstance(history_data, list):
            if len(history_data) == 0:
                device = next(self.parameters()).device
                if self.history_encoder_type == 'lstm':
                    return torch.zeros(1, self.hidden_dim, device=device)
                else:  # sum
                    return torch.zeros(1, self.encoding_dim, device=device)
            
            history_tensor = torch.tensor(history_data, dtype=torch.long)
            history_tensor = history_tensor.unsqueeze(0)
        else:
            history_tensor = history_data
        
        device = next(self.parameters()).device
        history_tensor = history_tensor.to(device)
        
        batch_size, seq_len, _ = history_tensor.shape
        
        # Embed i, j, obs separately
        i_indices = history_tensor[:, :, 0]
        j_indices = history_tensor[:, :, 1]
        obs_indices = history_tensor[:, :, 2]
        
        i_emb = self.history_embed(i_indices)
        j_emb = self.history_embed(j_indices)
        obs_emb = self.obs_embed(obs_indices)
        
        if self.history_encoder_type == 'lstm':
            # === LSTM: Order-dependent (for time-dependent problems) ===
            history_emb = torch.cat([i_emb, j_emb], dim=-1)
            
            if not history_emb.is_contiguous():
                history_emb = history_emb.contiguous()
            
            # Use cuDNN for LSTM if available (much faster!)
            USE_CUDNN_FOR_LSTM = os.getenv('USE_CUDNN_FOR_LSTM', '1') == '1'
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            cudnn_enabled = torch.backends.cudnn.enabled
            cudnn_benchmark = torch.backends.cudnn.benchmark
            
            try:
                if USE_CUDNN_FOR_LSTM:
                    pass  # cuDNN already enabled
                else:
                    torch.backends.cudnn.enabled = False
                    torch.backends.cudnn.benchmark = False
                
                lstm_out, (h_n, c_n) = self.history_lstm(history_emb)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            except RuntimeError as e:
                if 'CUDNN' in str(e) or 'cuDNN' in str(e):
                    if not hasattr(self, '_cudnn_lstm_warned'):
                        print(f"[WARNING] cuDNN LSTM failed, falling back to non-cuDNN: {e}")
                        self._cudnn_lstm_warned = True
                    torch.backends.cudnn.enabled = False
                    torch.backends.cudnn.benchmark = False
                    lstm_out, (h_n, c_n) = self.history_lstm(history_emb)
                else:
                    raise
            finally:
                torch.backends.cudnn.enabled = cudnn_enabled
                torch.backends.cudnn.benchmark = cudnn_benchmark
            
            history_embedding = h_n[-1]  # [batch, hidden_dim]
            return history_embedding
        
        elif self.history_encoder_type == 'sum':
            # === Set-Equivariant Sum: Order-independent ===
            # Encode each (i, j, obs) pair independently, then sum
            pair_emb = torch.cat([i_emb, j_emb], dim=-1)  # [batch, seq, hidden_dim//2]
            
            # Encode each pair
            pair_encodings = []
            for t in range(seq_len):
                pair_enc = self.pair_encoder(pair_emb[:, t, :])  # [batch, encoding_dim]
                pair_encodings.append(pair_enc)
            
            # Sum (set-equivariant operation - order-independent)
            sum_encoding = sum(pair_encodings)  # [batch, encoding_dim]
            return sum_encoding
        
        else:  # cat - concatenation
            # === Concatenation: Fixed-size concatenation ===
            # Encode each (i, j, obs) pair independently, then concatenate
            pair_emb = torch.cat([i_emb, j_emb], dim=-1)  # [batch, seq, hidden_dim//2]
            
            # Encode each pair
            pair_encodings = []
            for t in range(seq_len):
                pair_enc = self.pair_encoder(pair_emb[:, t, :])  # [batch, encoding_dim]
                pair_encodings.append(pair_enc)
            
            # Concatenate all encodings
            cat_encoding = torch.cat(pair_encodings, dim=-1)  # [batch, seq_len * encoding_dim]
            
            # Pad or truncate to fixed size
            target_size = self.max_history_len * self.encoding_dim
            current_size = cat_encoding.size(-1)
            
            if current_size < target_size:
                # Pad with zeros
                padding = torch.zeros(batch_size, target_size - current_size, device=cat_encoding.device)
                cat_encoding = torch.cat([cat_encoding, padding], dim=-1)
            elif current_size > target_size:
                # Truncate
                cat_encoding = cat_encoding[:, :target_size]
            
            return cat_encoding  # [batch, max_history_len * encoding_dim]


    def forward(self, state_data, history_data=None, available_actions_mask=None, 
                expected_mocu_features=None):
        """
        Forward pass: compute action logits.
        
        Args:
            state_data: PyTorch Geometric Data/Batch for current state
            history_data: History of (i, j, obs) tuples [batch_size, seq_len, 3]
            available_actions_mask: Binary mask for available actions [batch_size, num_actions]
                                   1 = available, 0 = already observed
            expected_mocu_features: Optional [batch_size, num_actions] tensor with expected MOCU
                                   for each action (like INN/NN R-matrix values)
                                   If provided, uses enhanced decoder that incorporates this info
        
        Returns:
            action_logits: [batch_size, num_actions]
            action_probs: [batch_size, num_actions] (softmax over available actions)
        """
        # Encode current state
        state_emb = self.encode_state(state_data)  # [batch_size, hidden_dim]
        
        # Encode history (LSTM, sum, or cat)
        history_emb = self.encode_history(history_data)  # [batch_size, hidden_dim] or [batch_size, encoding_dim] or [batch_size, max_history_len*encoding_dim]
        
        # Project history to hidden_dim if needed
        if self.history_encoder_type == 'sum':
            # history_emb is [batch, encoding_dim], need to project to hidden_dim
            if not hasattr(self, 'history_proj'):
                self.history_proj = nn.Linear(self.encoding_dim, self.hidden_dim).to(history_emb.device)
            history_emb = self.history_proj(history_emb)  # [batch, hidden_dim]
        elif self.history_encoder_type == 'cat':
            # history_emb is [batch, max_history_len * encoding_dim], project to hidden_dim
            if not hasattr(self, 'history_proj'):
                self.history_proj = nn.Linear(self.max_history_len * self.encoding_dim, self.hidden_dim).to(history_emb.device)
            history_emb = self.history_proj(history_emb)  # [batch, hidden_dim]
        
        # Combine state and history
        combined_base = torch.cat([state_emb, history_emb], dim=-1)  # [batch_size, hidden_dim*2]
        
        # Use enhanced decoder if expected MOCU features are provided
        if expected_mocu_features is not None:
            # Enhanced mode: incorporate expected MOCU for each action
            # For each action, combine base features with its expected MOCU
            batch_size = combined_base.size(0)
            num_actions = expected_mocu_features.size(1)
            
            # Expand base features for each action: [batch, num_actions, hidden_dim*2]
            combined_base_expanded = combined_base.unsqueeze(1).expand(-1, num_actions, -1)
            
            # Add expected MOCU feature: [batch, num_actions, hidden_dim*2+1]
            expected_mocu_expanded = expected_mocu_features.unsqueeze(-1)  # [batch, num_actions, 1]
            combined_with_mocu = torch.cat([combined_base_expanded, expected_mocu_expanded], dim=-1)
            
            # Reshape for batch processing: [batch*num_actions, hidden_dim*2+1]
            combined_flat = combined_with_mocu.view(batch_size * num_actions, -1)
            
            # Process through enhanced decoder: [batch*num_actions, 1]
            action_logits_flat = self.action_decoder_with_mocu(combined_flat)
            
            # Reshape back: [batch, num_actions]
            action_logits = action_logits_flat.view(batch_size, num_actions)
        else:
            # Standard mode: use base decoder
            action_logits = self.action_decoder(combined_base)  # [batch_size, num_actions]
        
        # Apply mask to prevent selecting already observed pairs
        if available_actions_mask is not None:
            # Set logits of unavailable actions to very negative value
            action_logits = action_logits.masked_fill(available_actions_mask == 0, -1e9)
        
        # Compute probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_logits, action_probs
    
    def select_action(self, state_data, history_data=None, available_actions_mask=None, 
                     deterministic=False, expected_mocu_features=None):
        """
        Select an action based on the current policy.
        
        Args:
            state_data: Current state
            history_data: History of decisions
            available_actions_mask: Mask for available actions
            deterministic: If True, select argmax; if False, sample from distribution
            expected_mocu_features: Optional expected MOCU features for each action
        
        Returns:
            action_idx: Selected action index
            log_prob: Log probability of the selected action
            action_pair: (i, j) tuple corresponding to action_idx
        """
        action_logits, action_probs = self.forward(
            state_data, history_data, available_actions_mask, 
            expected_mocu_features=expected_mocu_features
        )
        
        if deterministic:
            action_idx = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(probs=action_probs)
            action_idx = dist.sample()
        
        # Compute log probability
        log_prob = F.log_softmax(action_logits, dim=-1)
        log_prob_selected = log_prob.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
        
        # Convert action_idx to (i, j) pair
        action_pair = self.idx_to_pair(action_idx.item())
        
        return action_idx, log_prob_selected, action_pair
    
    def idx_to_pair(self, action_idx):
        """
        Convert action index to (i, j) oscillator pair.
        
        Maps: 0 → (0,1), 1 → (0,2), ..., to enumerate all N*(N-1)/2 pairs
        """
        # Enumerate pairs in order: (0,1), (0,2), ..., (0,N-1), (1,2), ...
        count = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if count == action_idx:
                    return (i, j)
                count += 1
        raise ValueError(f"Invalid action_idx: {action_idx}")
    
    def pair_to_idx(self, i, j):
        """Convert (i, j) pair to action index."""
        if i > j:
            i, j = j, i
        
        count = 0
        for ii in range(self.N):
            for jj in range(ii + 1, self.N):
                if ii == i and jj == j:
                    return count
                count += 1
        raise ValueError(f"Invalid pair: ({i}, {j})")


def create_state_data(w, a_lower, a_upper, device='cpu', include_degree=True):
    """
    Create PyTorch Geometric Data object from state.
    
    Args:
        w: Natural frequencies [N]
        a_lower: Lower bounds [N, N]
        a_upper: Upper bounds [N, N]
        device: torch device
    
    Returns:
        Data object for the state
    """
    N = len(w)
    
    # Node features: frequencies + degree (for Coutinho 2013 model)
    if include_degree:
        from src.models.predictors.utils import get_node_features_with_degree
        x = get_node_features_with_degree(w, a_lower, a_upper, device=device)  # [N, 2]
    else:
        x = torch.from_numpy(w.astype(np.float32)).unsqueeze(-1)  # [N, 1]
    
    # Edge indices: fully connected graph
    edge_index = []
    edge_attr = []
    
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_index.append([i, j])
                edge_attr.append([a_lower[i, j], a_upper[i, j]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # [num_edges, 2]
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = data.to(device)
    
    return data


class DADPolicyNetworkSwing(nn.Module):
    """
    Policy network for swing equation model.
    
    Architecture:
    1. State encoder: Encodes current state (M_lower, M_upper, K_lower, K_upper) using MLP
    2. History encoder: Encodes past (probe_action, observation) pairs using LSTM
    3. Action decoder: Outputs logits for each possible (bus, amplitude) combination
    """
    
    def __init__(self, N, num_probe_amplitudes=3, hidden_dim=64, encoding_dim=32, 
                 num_message_passing=3, history_encoder_type='lstm', max_history_len=10):
        """
        Args:
            N: Number of buses
            num_probe_amplitudes: Number of probe amplitude options
            hidden_dim: Hidden dimension for LSTM and MLPs
            encoding_dim: Dimension for state embeddings
            num_message_passing: Not used for swing equation (kept for compatibility)
            history_encoder_type: 'lstm' (order-dependent) or 'sum' (order-independent)
            max_history_len: Maximum history length
        """
        super(DADPolicyNetworkSwing, self).__init__()
        self.N = N
        self.num_probe_amplitudes = num_probe_amplitudes
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.num_actions = N * num_probe_amplitudes  # Number of (bus, amplitude) combinations
        self.history_encoder_type = history_encoder_type
        self.max_history_len = max_history_len
        
        # ========== State Encoder (MLP for M, K bounds) ==========
        # Input: [M_lower, M_upper, K_lower, K_upper] (4 scalars)
        self.state_encoder = nn.Sequential(
            nn.Linear(4, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, hidden_dim)
        )
        
        # ========== History Encoder ==========
        # Each history item: (bus, amplitude, observation) → embedding
        self.history_bus_embed = nn.Embedding(N, hidden_dim // 4)  # Bus index
        self.history_amp_embed = nn.Embedding(num_probe_amplitudes, hidden_dim // 4)  # Amplitude index
        self.obs_linear = nn.Linear(1, hidden_dim // 2)  # Observation (ROCOF_max, etc.)
        
        if history_encoder_type == 'lstm':
            self.history_lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )
        elif history_encoder_type == 'sum':
            self.pair_encoder = nn.Sequential(
                Linear(hidden_dim, encoding_dim),
                ReLU(),
                Linear(encoding_dim, encoding_dim)
            )
        elif history_encoder_type == 'cat':
            self.pair_encoder = nn.Sequential(
                Linear(hidden_dim, encoding_dim),
                ReLU(),
                Linear(encoding_dim, encoding_dim)
            )
        
        # ========== Action Decoder ==========
        # Combines state and history to output action logits
        if history_encoder_type == 'sum':
            combined_dim = hidden_dim + encoding_dim
        elif history_encoder_type == 'cat':
            combined_dim = hidden_dim + max_history_len * encoding_dim
        else:  # lstm
            combined_dim = hidden_dim + hidden_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions)
        )
        
        # Optional: Enhanced decoder with expected MOCU features
        self.use_expected_mocu = True
        if self.use_expected_mocu:
            self.enhanced_decoder = nn.Sequential(
                nn.Linear(combined_dim + self.num_actions, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_actions)
            )
    
    def encode_state(self, state_data):
        """Encode current state (M, K bounds)."""
        if isinstance(state_data, dict):
            # Extract bounds from dict
            M_lower = state_data['M_lower']
            M_upper = state_data['M_upper']
            K_lower = state_data['K_lower']
            K_upper = state_data['K_upper']
            state_vec = torch.cat([M_lower, M_upper, K_lower, K_upper], dim=-1)
        else:
            # Assume state_data is already a tensor [batch, 4]
            state_vec = state_data
        
        state_emb = self.state_encoder(state_vec)  # [batch, hidden_dim]
        return state_emb
    
    def encode_history(self, history_data):
        """Encode history of (bus, amplitude, observation) tuples."""
        if history_data is None or history_data.shape[1] == 0:
            # No history
            if self.history_encoder_type == 'lstm':
                batch_size = 1
                return torch.zeros(batch_size, self.hidden_dim, device=history_data.device if history_data is not None else 'cpu')
            elif self.history_encoder_type == 'sum':
                return torch.zeros(1, self.encoding_dim, device=history_data.device if history_data is not None else 'cpu')
            else:  # cat
                return torch.zeros(1, self.max_history_len * self.encoding_dim, device=history_data.device if history_data is not None else 'cpu')
        
        batch_size, seq_len, _ = history_data.shape
        device = history_data.device
        
        # Encode each history item: (bus, amplitude, observation)
        history_embeddings = []
        for t in range(seq_len):
            bus_idx = history_data[:, t, 0].long()  # [batch]
            amp_idx = history_data[:, t, 1].long()  # [batch]
            obs = history_data[:, t, 2:3]  # [batch, 1] - observation value
            
            bus_emb = self.history_bus_embed(bus_idx)  # [batch, hidden_dim//4]
            amp_emb = self.history_amp_embed(amp_idx)  # [batch, hidden_dim//4]
            obs_emb = self.obs_linear(obs)  # [batch, hidden_dim//2]
            
            pair_emb = torch.cat([bus_emb, amp_emb, obs_emb], dim=-1)  # [batch, hidden_dim]
            history_embeddings.append(pair_emb)
        
        if self.history_encoder_type == 'lstm':
            # LSTM: Order-dependent
            history_seq = torch.stack(history_embeddings, dim=1)  # [batch, seq_len, hidden_dim]
            h_n, (h_n, c_n) = self.history_lstm(history_seq)
            return h_n[-1]  # [batch, hidden_dim]
        elif self.history_encoder_type == 'sum':
            # Set-equivariant sum: Order-independent
            history_seq = torch.stack(history_embeddings, dim=1)  # [batch, seq_len, hidden_dim]
            pair_encodings = self.pair_encoder(history_seq)  # [batch, seq_len, encoding_dim]
            return torch.sum(pair_encodings, dim=1)  # [batch, encoding_dim]
        else:  # cat
            # Concatenation: Fixed-size
            history_seq = torch.stack(history_embeddings, dim=1)  # [batch, seq_len, hidden_dim]
            pair_encodings = self.pair_encoder(history_seq)  # [batch, seq_len, encoding_dim]
            
            # Pad or truncate to max_history_len
            if seq_len < self.max_history_len:
                padding = torch.zeros(batch_size, self.max_history_len - seq_len, self.encoding_dim, device=device)
                pair_encodings = torch.cat([pair_encodings, padding], dim=1)
            elif seq_len > self.max_history_len:
                pair_encodings = pair_encodings[:, :self.max_history_len, :]
            
            return pair_encodings.reshape(batch_size, -1)  # [batch, max_history_len * encoding_dim]
    
    def forward(self, state_data, history_data=None, available_actions_mask=None,
                expected_mocu_features=None):
        """
        Forward pass: compute action logits.
        
        Args:
            state_data: Dict with M_lower, M_upper, K_lower, K_upper or tensor [batch, 4]
            history_data: History of (bus, amp, obs) tuples [batch_size, seq_len, 3]
            available_actions_mask: Binary mask for available actions [batch_size, num_actions]
            expected_mocu_features: Optional [batch_size, num_actions] tensor with expected MOCU
        
        Returns:
            action_logits: [batch_size, num_actions]
            action_probs: [batch_size, num_actions] (softmax over available actions)
        """
        # Encode state
        state_emb = self.encode_state(state_data)  # [batch, hidden_dim]
        
        # Encode history
        history_emb = self.encode_history(history_data)  # [batch, hidden_dim or encoding_dim or ...]
        
        # Project history to hidden_dim if needed
        if self.history_encoder_type == 'sum':
            if not hasattr(self, 'history_proj'):
                self.history_proj = nn.Linear(self.encoding_dim, self.hidden_dim).to(history_emb.device)
            history_emb = self.history_proj(history_emb)
        elif self.history_encoder_type == 'cat':
            if not hasattr(self, 'history_proj'):
                self.history_proj = nn.Linear(self.max_history_len * self.encoding_dim, self.hidden_dim).to(history_emb.device)
            history_emb = self.history_proj(history_emb)
        
        # Combine state and history
        combined = torch.cat([state_emb, history_emb], dim=-1)  # [batch, hidden_dim * 2]
        
        # Decode to action logits
        if expected_mocu_features is not None and self.use_expected_mocu:
            # Enhanced decoder with expected MOCU features
            combined_with_mocu = torch.cat([combined, expected_mocu_features], dim=-1)
            action_logits = self.enhanced_decoder(combined_with_mocu)  # [batch, num_actions]
        else:
            action_logits = self.decoder(combined)  # [batch, num_actions]
        
        # Apply available actions mask
        if available_actions_mask is not None:
            action_logits = action_logits + (1.0 - available_actions_mask) * (-1e9)
        
        # Softmax over available actions
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_logits, action_probs
    
    def idx_to_action(self, action_idx):
        """Convert action index to (bus, amplitude)."""
        bus_idx = action_idx // self.num_probe_amplitudes
        amp_idx = action_idx % self.num_probe_amplitudes
        return (bus_idx, amp_idx)
    
    def action_to_idx(self, bus_idx, amp_idx):
        """Convert (bus, amplitude) to action index."""
        return bus_idx * self.num_probe_amplitudes + amp_idx


def create_swing_state_data(M_lower, M_upper, K_lower, K_upper, N, device='cpu'):
    """
    Create state data for swing equation policy network.
    
    Args:
        M_lower, M_upper, K_lower, K_upper: Uncertainty bounds (scalars)
        N: Number of buses
        device: torch device
    
    Returns:
        Dict with state tensors
    """
    return {
        'M_lower': torch.tensor([M_lower], dtype=torch.float32, device=device),
        'M_upper': torch.tensor([M_upper], dtype=torch.float32, device=device),
        'K_lower': torch.tensor([K_lower], dtype=torch.float32, device=device),
        'K_upper': torch.tensor([K_upper], dtype=torch.float32, device=device),
    }

