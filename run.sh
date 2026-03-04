#!/bin/bash
# Main pipeline script for second-order Kuramoto (swing equation) with active probing
# Based on documents/design.md and documents/pseudocode.tex (MOCU-based sBOED)

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file> [K_value]"
    echo ""
    echo "Examples:"
    echo "  $0 config/fast_config.yaml         # Fast (IEEE-14)"
    echo "  $0 config/balanced_config.yaml     # Deeper testing (balanced)"
    echo "  $0 config/full_config.yaml         # Full publication experiment"
    echo ""
    exit 1
fi

CONFIG_FILE=$1
K_OVERRIDE=$2
export CONFIG_FILE

# Support config/ directory
if [ ! -f "$CONFIG_FILE" ]; then
    # Try config/ directory if not found
    CONFIG_NAME=$(basename "$CONFIG_FILE")
    if [ -f "config/$CONFIG_NAME" ]; then
        CONFIG_FILE="config/$CONFIG_NAME"
    else
        echo "Error: Config file not found: $CONFIG_FILE"
        echo "  Tried: $CONFIG_FILE and config/$CONFIG_NAME"
        exit 1
    fi
fi

# Get absolute paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
BASE_CONFIG_NAME=$(echo "$CONFIG_NAME" | sed 's/_K[0-9]*$//')

# Parse config to get model type
MODEL_TYPE=$(grep "^model_type:" $CONFIG_FILE | awk '{print $2}' | tr -d "'\"")
if [ -z "$MODEL_TYPE" ]; then
    MODEL_TYPE="second_order"  # Default to second-order
fi

echo "=========================================="
echo "MOCU-OED Experiment Workflow"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
if [ -n "$K_OVERRIDE" ]; then
    echo "K value: K=$K_OVERRIDE (override)"
    # Update config temporarily
    sed -i.bak "s/update_count:.*/update_count: $K_OVERRIDE/" "$CONFIG_FILE"
    sed -i.bak "s/^  K:.*/  K: $K_OVERRIDE/" "$CONFIG_FILE"
fi
TIMESTAMP=$(date +"%m%d%Y_%H%M%S")
echo "Run timestamp: $TIMESTAMP"
echo "Model type: $MODEL_TYPE"
echo ""

# Create experiment directory and report
EXP_DIR="${PROJECT_ROOT}/experiments/${CONFIG_NAME}_${TIMESTAMP}"
mkdir -p "$EXP_DIR"
export EXP_DIR
export EXP_REPORT="${EXP_DIR}/report.txt"
{
  echo "=============================================="
  echo "MOCU-OED Experiment Report"
  echo "=============================================="
  echo "Config: $CONFIG_FILE"
  echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""
} > "$EXP_REPORT"
echo "Experiment directory: $EXP_DIR"
echo "Report: $EXP_REPORT"
echo ""

# Parse experiment parameters
N=$(grep "^N:" $CONFIG_FILE | awk '{print $2}')
UPDATE_CNT=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "update_count:" | awk '{print $2}')
IT_IDX=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "it_idx:" | awk '{print $2}')
K_MAX=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "K_max:" | awk '{print $2}')
NUM_SIMULATIONS=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "num_simulations:" | awk '{print $2}')

echo "Experiment Configuration:"
echo "  System size (N): $N"
echo "  N_global: $((N + 1))"
echo "  K: $UPDATE_CNT"
echo "  Methods to evaluate: RANDOM,ENTROPY,ODE,iNN,NN,DAD"
echo ""

# Export environment variables for scripts
export EXP_EVAL_DIR="$EXP_DIR/eval"
export EXP_DAD_DIR="$EXP_DIR/dad_models"
export EXP_DAD_DATA_DIR="$EXP_DIR/dad_data"
export EXP_RESULTS_DIR="$EXP_DIR/results"
mkdir -p "$EXP_DAD_DATA_DIR"

# Step 0: Verify configuration
echo "[Step 0/6] Verifying configuration..."
if [ "$MODEL_TYPE" != "second_order" ]; then
    echo "⚠️  Warning: model_type is '$MODEL_TYPE', expected 'second_order'"
    echo "   Continuing anyway, but scripts may fail if not properly configured"
fi
echo "✓ Configuration loaded: N=$N"
echo ""

# Step 1: Generate MOCU training data
echo "[Step 1/6] Generating MOCU training data..."
bash "${PROJECT_ROOT}/scripts/bash/step1_generate_mocu_data.sh" "$CONFIG_FILE"
echo ""

# Step 2: Train Swing MPNN MOCU predictor
echo "[Step 2/6] Training Swing MPNN MOCU predictor..."
bash "${PROJECT_ROOT}/scripts/bash/step2_train_swing_mpnn.sh" "$CONFIG_FILE"
echo ""

# Step 3: Run baseline evaluation and visualization
echo "[Step 3/6] Running baseline evaluation and visualization..."
bash "${PROJECT_ROOT}/scripts/bash/step3_evaluate_baselines.sh" "$CONFIG_FILE"
echo ""

# Step 4: Generate DAD training data (swing equation)
echo "[Step 4/6] Generating DAD training data..."
bash "${PROJECT_ROOT}/scripts/bash/step4_generate_dad_data.sh" "$CONFIG_FILE"
echo ""

# Step 5: Train DAD policy
echo "[Step 5/6] Training DAD policy..."
bash "${PROJECT_ROOT}/scripts/bash/step5_train_dad_policy.sh" "$CONFIG_FILE"
echo ""

# Step 6: Evaluate DAD methods
echo "[Step 6/6] Evaluating DAD methods..."
bash "${PROJECT_ROOT}/scripts/bash/step6_evaluate_dad.sh" "$CONFIG_FILE"
echo ""

# Restore config if modified
if [ -n "$K_OVERRIDE" ] && [ -f "${CONFIG_FILE}.bak" ]; then
    mv "${CONFIG_FILE}.bak" "$CONFIG_FILE"
fi

echo "=========================================="
echo "✓ Experiment completed successfully!"
echo "=========================================="
echo "Results saved to: $EXP_DIR"
echo "Report: $EXP_REPORT"
echo ""
{
  echo ""
  echo "--- Pipeline Complete ---"
  echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
} >> "$EXP_REPORT"
echo "To view results:"
echo "  - Report: $EXP_REPORT"
echo "  - Baseline evaluation: $EXP_DIR/eval/"
echo "  - DAD policies: $EXP_DIR/dad_models/"
echo "  - Final results: $EXP_DIR/results/"
echo ""