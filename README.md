# DAD-MOCU: Deep Adaptive Design for Optimal Experimental Design

Sequential optimal experimental design for power systems using the second-order Kuramoto (swing equation) model with active probing. The framework learns probe-selection policies to minimize MOCU (Mean Objective Cost of Uncertainty) for estimating uncertain parameters \((M, K)\) (inertia and control gain).

## Experimental Design

- **Model**: Second-order Kuramoto (swing equation) on IEEE-14 bus network.
- **Uncertain parameters**: \((M, K)\) — inertia and control gain.
- **Actions**: Probe signals \(\xi = (b, A, T_p)\): bus \(b\), amplitude \(A\), \(T_p = 2\) s fixed.
- **Observations**: ROCOF-only \(y_t = \text{ROCOF}_{\max}\) (12 Hz).
- **Objective**: Minimize MOCU via sequential probe selection (reduce uncertainty in \((M,K)\)).
- **Methods**: Baselines RANDOM, ENTROPY, ODE, iNN, NN; learned policy DAD.

## Installation

```bash
conda create -n dad_mocu python=3.10 -y
conda activate dad_mocu

conda install -y -c conda-forge numpy scipy matplotlib tqdm pyyaml pandas pip setuptools wheel
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install openpyxl torchdiffeq
```

## Running

From the project root, run the full pipeline (data generation, training, evaluation) with a config file:

```bash
conda activate dad_mocu
bash run.sh config/fast_config.yaml
```

Results are written to `experiments/<config>_<timestamp>/`.
