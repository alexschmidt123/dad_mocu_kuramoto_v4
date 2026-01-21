# DAD-MOCU: Deep Adaptive Design for Optimal Experimental Design

Sequential optimal experimental design for power systems using second-order Kuramoto (swing equation) model with active probing. The framework learns optimal probe selection policies to minimize MOCU (Mean Objective Cost of Uncertainty) for estimating uncertain parameters $(M, K)$ (inertia and control gain).

## Experimental Design

**System Model**: Second-order Kuramoto (swing equation) on IEEE-14 bus network

**Uncertain Parameters**: $(M, K)$ - inertia and control gain

**Actions**: Probe signals $\xi = (b, A, T)$ where $b$ is bus location, $A$ is amplitude, $T$ is duration

**Observations**: PMU-like frequency features (ROCOF_max, f_min, t_settle) extracted at 12 Hz

**Objective**: Minimize MOCU through sequential probe selection

**Methods**: RANDOM, ENTROPY, ODE, iNN, NN (baselines) and DAD_MOCU, IDAD_MOCU (learned policies)

## Installation

```bash
# Create conda environment
conda create -n dad_mocu python=3.10 -y
conda activate dad_mocu

# Install dependencies
conda install -y -c conda-forge numpy scipy matplotlib tqdm pyyaml pandas pip setuptools wheel
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install openpyxl torchdiffeq
```

## Running Experiments

```bash
conda activate dad_mocu

# Quick test (3-5 minutes)
bash run.sh configs/fast_config.yaml

# Full experiment
bash run.sh configs/ieee14_config.yaml
```

The pipeline generates training data, trains models, and evaluates all methods. Results are saved in `experiments/<config>_<timestamp>/`.
