# DAD-MOCU: Deep Adaptive Design for Optimal Experimental Design

Sequential optimal experimental design for power systems using second-order Kuramoto (swing equation) model with active probing. The framework learns optimal probe selection policies to minimize MOCU (Mean Objective Cost of Uncertainty) for estimating uncertain parameters $(M, K)$ (inertia and control gain).

## Experimental Design

**System Model**: Second-order Kuramoto (swing equation) on IEEE-14 bus network

**Uncertain Parameters**: $(M, K)$ - inertia and control gain

**Actions**: Probe signals $\xi = (b, A, T_p)$ where $b$ is bus location, $A$ is amplitude, $T_p = 2$ s (fixed)

**Observations**: ROCOF-only observation $y_t = \text{ROCOF}_{\max}$ (scalar) extracted at 12 Hz sampling rate

**Objective**: Minimize MOCU through sequential probe selection

**Methods**: RANDOM, ENTROPY, ODE, iNN, NN (baselines) and DAD (learned policy with explicit likelihood)


## Project Structure

Standard commercial software engineering layout:

```
project_root/
├── src/                    # All source code
│   ├── core/              # Core functionality (simulator, observation, inference, decision)
│   ├── methods/           # All methods (baselines + policies)
│   ├── models/            # Neural network models
│   └── utils/             # Utilities
├── scripts/               # All scripts
│   ├── bash/              # Pipeline orchestration (step1–step6)
│   ├── training/          # Data generation, MPNN & DAD training
│   ├── evaluation/        # Baseline comparison, DAD evaluation
│   └── visualization/    # MOCU/time plots
├── config/                # Configuration files
├── data/                  # Data storage
├── models/                # Trained models
└── experiments/           # Experiment results
```

**Key modules:**
- `src/core/`: Swing equation simulator, ROCOF observation, likelihood, posterior, MOCU computation
- `src/methods/`: All OED methods (RANDOM, ENTROPY, ODE, NN, iNN, DAD)
- `scripts/training/`: MOCU data generation, MPNN predictor training, DAD data generation, DAD policy training
- `scripts/evaluation/`: Baseline method comparison (`compare_methods.py`), DAD evaluation (`dad_eval.py`)
- `scripts/visualization/`: MOCU and time-complexity plots (`visualize.py`)
- `scripts/bash/`: Pipeline steps (step1–step6)

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
bash run.sh config/fast_config.yaml

# Full experiment
bash run.sh config/full_config.yaml
```

The pipeline generates training data, trains models, and evaluates all methods. Results are saved in `experiments/<config>_<timestamp>/`.

## Ensuring MPNN MOCU estimation quality

The MPNN surrogate is used by iNN, NN, and DAD; its quality directly affects method performance.

**1. Training data**
- Use enough samples: config `dataset.samples_per_type` (e.g. 250+ for fast, 1200 for full).
- Use a sufficiently accurate MOCU target: config `dataset.K_max` (e.g. 512–1024) so ground-truth MOCU in the data is reliable.

**2. Training**
- Training prints validation MSE; at the end it also prints **MAE**, **R²**, and **Pearson r** on the validation set.
- Aim for **R² > 0.9** and **Pearson r > 0.95** on validation; if not, increase data size or epochs.

**3. Validation script**
After training (Step 2), run the validator on a held-out test portion of the MOCU data:

```bash
python scripts/evaluation/validate_mpnn_mocu.py --config config/fast_config.yaml
```

This reports **MSE, MAE, R², Pearson r, max absolute error** (MPNN vs ground-truth MOCU in the data). Optionally compare against true ODE MOCU on a small subset (slow but definitive):

```bash
python scripts/evaluation/validate_mpnn_mocu.py --config config/fast_config.yaml --ode_spot_check 5
```

**4. If quality is poor**
- Increase `dataset.samples_per_type` and re-run Step 1 and Step 2.
- Increase `training.epochs` or check learning rate.
- Ensure `dataset.K_max` matches or exceeds the value used at evaluation time.
