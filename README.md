# Setup and Run

## Conda environment installation

```bash
conda create -n mocu_optimized python=3.11 -y
conda activate mocu_optimized
pip install -r requirements.txt
pip install pycuda
```

## Project setup

```bash
cd /path/to/dad_mocu_kuramoto_v4
export PYTHONPATH="$(pwd)"
```

## Run full pipeline with `run.sh`

```bash
# default T=3
./run.sh -config fast_config

# set horizon T explicitly (T≥5 uses on-demand PyCUDA sim; T≤4 pre-banks all sequences)
./run.sh -config fast_config -T 5

# reuse an existing experiment directory
./run.sh -config fast_config -T 1 -exp-dir experiments/<run_name>
```

## Run scripts in `scripts/` folder

```bash
# 1) data generation
./scripts/data_generation.sh -config fast_config -T 1

# 2) training
./scripts/dad_training.sh -exp-dir experiments/<run_name>

# 3) evaluation (all methods)
./scripts/evaluation.sh -exp-dir experiments/<run_name>

# evaluate one method only
./scripts/evaluation.sh -exp-dir experiments/<run_name> -method dad_spce
```
