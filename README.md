# Setup and Run

## Conda environment installation

```bash
conda create -n dad_mocu_kuramoto python=3.9 -y
conda activate dad_mocu_kuramoto
pip install -r requirements.txt
pip install pycuda
```

## Project setup

```bash
cd /path/to/dad_mocu_kuramoto_v4
export PYTHONPATH="$(pwd)"
```

## Five-phase workflow (`run.sh`)

```text
sweep_run.sh
    -> run.sh
        -> scripts/data_generation.sh             # Phase 1: probe + control U-banks
        -> scripts/control_safety_calibration.sh  # Phase 2: calibrate terminal rule
        -> scripts/objective_observability.sh     # Phase 3: gate (blocks training if FAIL)
        -> scripts/dad_training.sh                # Phase 4: objective-based DAD
        -> scripts/evaluation.sh                  # Phase 5: dad / myopic / fixed / random
```

| Phase | Script | Role |
|---|---|---|
| 1 | `scripts/data_generation.sh` | Build or reuse probe banks under `data/<slug>/`, then PyCUDA `generate-control-bank`. |
| 2 | `scripts/control_safety_calibration.sh` | Train-only calibration of `(α, margin)` so posterior `u_ctrl` is empirically safe; never uses the final test set. |
| 3 | `scripts/objective_observability.sh` | Diagnostic random probe histories; gate on whether posterior `u_ctrl` varies with observations. **Nonzero exit on failure — no DAD training.** |
| 4 | `scripts/dad_training.sh` | Train DAD minimizing terminal posterior-safe `u_ctrl`. |
| 5 | `scripts/evaluation.sh` | Evaluate DAD, Myopic, Fixed, Random. |

`sweep_run.sh` only loops `./run.sh` over configs and `T` (with `.sweep.lock`). It does **not** reimplement calibration or the gate.

### Examples

```bash
# One experiment (T=2)
./run.sh -config ieee5_config -T 2

# Resume on an existing run (still runs calibration + observability before training)
./run.sh -config ieee5_config -exp-dir experiments/<run_name>

# Sweep several systems / horizons
./sweep_run.sh -config ieee5_config,ieee9_config -from 1 -to 6
```

### Control-safety calibration CLI

```bash
./scripts/control_safety_calibration.sh -exp-dir experiments/<run>
./scripts/control_safety_calibration.sh -exp-dir experiments/<run> -num-rollouts 2000 -seed 2468

python -m src.cli calibrate-control-safety \
  --exp-dir experiments/<run> \
  --num-rollouts 2000 \
  --seed 2468
```

Outputs land in `<exp-dir>/diagnostics/control_safety_calibration/`. The selected rule is written into `run_config.yaml` (`control.alpha`, `control.safety_margin`) and used by all methods.

### Objective-observability CLI

```bash
./scripts/objective_observability.sh -exp-dir experiments/<run>
./scripts/objective_observability.sh -exp-dir experiments/<run> -num-rollouts 1000 -seed 1234

# equivalent
python -m src.cli check-objective-observability \
  --exp-dir experiments/<run> \
  --num-rollouts 1000 \
  --seed 1234
```

Outputs land in:

```text
<exp-dir>/diagnostics/objective_observability/
```

Training is blocked when the observability gate fails (unique final controls, std, fraction changed from prior, true safety, Spearman vs shuffled, etc.). Thresholds live under `objective_observability:` in the YAML configs.

### Other bank utilities

```bash
python -m src.cli generate-control-bank --config ieee5_config
python -m src.cli diagnose-control-objective --config ieee5_config
```

## Run scripts individually

```bash
./scripts/data_generation.sh -config ieee5_config -T 2
./scripts/control_safety_calibration.sh -exp-dir experiments/<run_name>
./scripts/objective_observability.sh -exp-dir experiments/<run_name>
./scripts/dad_training.sh -exp-dir experiments/<run_name>
./scripts/evaluation.sh -exp-dir experiments/<run_name>
./scripts/evaluation.sh -exp-dir experiments/<run_name> -method dad
```
