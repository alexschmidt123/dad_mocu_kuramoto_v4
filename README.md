# DAD-MOCU: Deep Adaptive Design for Optimal Experimental Design

This project implements Deep Adaptive Design (DAD) and Implicit Deep Adaptive Design (iDAD) methods for sequential optimal experimental design using MOCU (Mean Objective Cost of Uncertainty) as the objective. The framework uses neural message passing networks (MPNN) to accelerate MOCU prediction and reinforcement learning to learn optimal experimental selection policies.

## System Model

The project uses a **second-order Kuramoto model (swing equation)** for power system dynamics, applied to the **IEEE-14 bus system**:

### Dynamics

The network-coupled swing equation for bus $i$:

$$\dot{\theta}_i(t) = \omega_i(t)$$

$$M \dot{\omega}_i(t) = P_{m,i} - \sum_{j=1}^{N} B_{ij}\sin(\theta_i(t) - \theta_j(t)) - D \omega_i(t) - K \omega_i(t) + u^{\text{probe}}_{\xi,i}(t) + u^{\text{ctrl}}_{\gamma,i}(t)$$

where:
- $\theta_i$: phase angle of bus $i$
- $\omega_i$: frequency deviation of bus $i$
- $M$: equivalent inertia (uncertain parameter)
- $K$: fast frequency response / primary control gain (uncertain parameter)
- $B_{ij}$: coupling matrix (known, IEEE-14 network topology)
- $P_{m,i}$: mechanical power (known)
- $D$: damping coefficient (known)
- $u^{\text{probe}}_{\xi,i}$: probe signal at bus $i$ with action $\xi = (b, A, T)$
- $u^{\text{ctrl}}_{\gamma,i}$: control signal with capacity $\gamma$

### Uncertainty and MOCU

**Uncertain parameters**: $\vartheta = (M, K) \in \mathbb{R}_+^2$

**MOCU objective**: Compute $\gamma^*(M, K)$ - the minimum control capacity required to satisfy frequency security constraints:
- Maximum ROCOF: $\max_t |\dot{f}(t)| \le r_{\max}$
- Minimum frequency: $\min_t f(t) \ge f_{\min}$

**MOCU value**: Expected cost of uncertainty in $(M, K)$ for planning control capacity $\gamma$.

### Experiments and Observations

**Experiment action**: Probe signal $\xi = (b, A, T)$ where:
- $b$: bus location (0-13 for IEEE-14)
- $A$: probe amplitude
- $T$: probe duration

**Observation**: Frequency features extracted from system response:
- `ROCOF_max`: Maximum rate of change of frequency
- `f_min`: Minimum frequency during transient
- `t_settle`: Settling time to frequency synchronization

## Hardware and Environment Setup

**Hardware**:
- **Processor**: 13th Gen Intel® Core™ i7-13700F (24 cores)
- **Memory**: 64.0 GiB
- **Graphics**: NVIDIA GeForce RTX 4090 (24GB)
- **OS**: Ubuntu 22.04.5 LTS (64-bit)

**Create conda environment and install dependencies**:

```bash
# Create conda environment (Python 3.10)
conda create -n dad_mocu python=3.10 -y
conda activate dad_mocu

# Install core dependencies
conda install -y -c conda-forge numpy scipy matplotlib tqdm pyyaml pandas pip setuptools wheel

# Install PyTorch with CUDA 12.1 (via pip to avoid MKL issues)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
pip install torch-geometric

# Install PyG extensions
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Install CUDA Toolkit (required for PyCUDA kernel compilation)
conda install -y -c nvidia cuda-toolkit=12.1

# Install PyCUDA (required for MOCU computation)
pip install pycuda

# Install additional dependencies
pip install openpyxl torchdiffeq
```

## Project Structure

```
dad_mocu_kuramoto_v4/
├── configs/              # Configuration files
│   ├── fast_config.yaml  # Quick test configuration (IEEE-14)
│   └── ieee14_config.yaml # Full IEEE-14 bus system configuration
├── src/
│   ├── core/             # Core model implementations
│   │   ├── swing_equation_ode.py      # Second-order ODE solver
│   │   ├── swing_equation_mocu.py     # MOCU computation
│   │   ├── swing_equation_params.py   # Parameter generation
│   │   ├── mocu_torchdiffeq.py        # MOCU interface
│   │   └── sync_detection.py          # Frequency synchronization check
│   ├── models/           # Neural network models
│   │   ├── predictors/   # MPNN predictors for MOCU
│   │   └── policy_networks.py  # DAD policy networks
│   └── methods/          # OED methods
│       ├── ode.py        # ODE-based method
│       ├── inn.py        # iNN method
│       ├── nn.py         # NN method
│       └── ...
├── scripts/              # Data generation and training scripts
│   ├── generate_mocu_data.py    # Generate MPNN training data
│   ├── generate_dad_data.py    # Generate DAD training data
│   ├── evaluate.py              # Baseline evaluation
│   └── ...
├── run.sh                # Main pipeline script
└── README.md
```

## Running Experiments

### Using `run.sh` (Main Pipeline)

Run the complete pipeline (data generation, MPNN training, baseline evaluation, DAD training, DAD evaluation):

```bash
conda activate dad_mocu

# Quick test (3-5 minutes)
bash run.sh configs/fast_config.yaml

# Full IEEE-14 bus system experiment
bash run.sh configs/ieee14_config.yaml

# Override K (number of sequential experiments)
bash run.sh configs/ieee14_config.yaml 10  # Use K=10 instead of default
```

The script creates a self-contained experiment directory: `experiments/<config>_<timestamp>/` containing:
- `eval/`: Baseline method evaluation results and plots
- `dad_models/`: Trained DAD policy models and training curves
- `results/`: Final evaluation results comparing all methods

### Pipeline Steps

1. **Generate MPNN training data**: Sample $(M, K)$ parameters and compute $\gamma^*(M, K)$
2. **Train MPNN predictor**: Learn to predict MOCU from system state
3. **Evaluate baselines**: Compare RANDOM, ENTROPY, ODE, iNN, NN, REGRESSION_SCORER methods
4. **Generate DAD training data**: Generate trajectories for RL training
5. **Train DAD policy**: Learn optimal experimental selection policy
6. **Evaluate DAD methods**: Compare DAD_MOCU and iDAD_MOCU against baselines

### Using `run_sweepK.sh` (Optional)

Run experiments with multiple K values to study the effect of sequence length:

```bash
# Run with default K values: 4, 6, 8, 10
bash run_sweepK.sh configs/ieee14_config.yaml

# Run with custom K values
bash run_sweepK.sh configs/ieee14_config.yaml 2 4 6 8 10
```

This will run the complete pipeline for each K value and save results in separate experiment directories. Each run reuses shared MPNN data and models but generates fresh DAD training data and models for each K value.

## Methods

### Baseline Methods
- **RANDOM**: Random probe selection
- **ENTROPY**: Select probe that maximizes information entropy
- **ODE**: Use ODE-based MOCU computation for selection
- **iNN**: Implicit Neural Network method (uses MPNN predictor)
- **NN**: Neural Network method (uses MPNN predictor)
- **REGRESSION_SCORER**: Regression-based scoring method

### DAD Methods
- **DAD_MOCU**: Deep Adaptive Design with MOCU rewards (no critic)
- **iDAD_MOCU**: Implicit Deep Adaptive Design with MOCU rewards (with critic)

## Configuration Files

Configuration files define:
- System parameters (N=14 for IEEE-14)
- Swing equation parameters (topology, coupling, damping, etc.)
- Uncertainty bounds (M_lower, M_upper, K_lower, K_upper)
- Training parameters (epochs, batch size, learning rate)
- Experiment parameters (K, number of simulations)

See `configs/ieee14_config.yaml` for a complete example.

## References

- **Swing Equation Model**: Based on "Probing Signal-Based Inertia and Frequency Response Estimation for Power Systems with High Penetration of Inverter-Based Resources"
- **DAD/iDAD Methods**: Deep Adaptive Design for sequential optimal experimental design
- **MPNN Predictors**: Neural Message Passing for Objective-Based Uncertainty Quantification
