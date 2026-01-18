#!/bin/bash
# Main pipeline script for second-order Kuramoto (swing equation) with active probing
# Based on new_plan.tex and paper: "Probing Signal-Based Inertia and Frequency Response Estimation"

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file> [K_value]"
    echo ""
    echo "Examples:"
    echo "  $0 configs/fast_config.yaml        # Quick test (IEEE-14)"
    echo "  $0 configs/ieee14_config.yaml      # Full IEEE-14 experiment"
    echo "  $0 configs/ieee14_config.yaml 10   # Override K=10"
    echo ""
    exit 1
fi

CONFIG_FILE=$1
K_OVERRIDE=$2

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
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

# Create experiment directory
EXP_DIR="${PROJECT_ROOT}/experiments/${CONFIG_NAME}_${TIMESTAMP}"
mkdir -p "$EXP_DIR"
echo "Experiment directory: $EXP_DIR"
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
echo "  Methods to evaluate: RANDOM,ENTROPY,ODE,iNN,NN,REGRESSION_SCORER,DAD_MOCU,IDAD_MOCU"
echo ""

# Export environment variables for scripts
export EXP_EVAL_DIR="$EXP_DIR/eval"
export EXP_DAD_DIR="$EXP_DIR/dad_models"
export EXP_RESULTS_DIR="$EXP_DIR/results"

# Step 0: Verify configuration
echo "[Step 0/6] Verifying configuration..."
if [ "$MODEL_TYPE" != "second_order" ]; then
    echo "⚠️  Warning: model_type is '$MODEL_TYPE', expected 'second_order'"
    echo "   Continuing anyway, but scripts may fail if not properly configured"
fi
echo "✓ Configuration loaded: N=$N"
echo ""

# Step 1: Generate MPNN training data
echo "[Step 1/6] Generating MPNN training data..."
bash "${PROJECT_ROOT}/scripts/bash/step1_generate_mocu_data.sh" "$CONFIG_FILE"
echo ""

# Step 2: Train Swing MLP predictor
echo "[Step 2/6] Training Swing MLP predictor..."
bash "${PROJECT_ROOT}/scripts/bash/step2_train_swing_mlp.sh" "$CONFIG_FILE"
echo ""

# Step 3: Run baseline evaluation and visualization
echo "[Step 3/6] Running baseline evaluation and visualization..."
bash "${PROJECT_ROOT}/scripts/bash/step3_evaluate_baselines.sh" "$CONFIG_FILE"
echo ""

# Step 4: Generate DAD training data
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
echo ""
echo "To view results:"
echo "  - Evaluation plots: $EXP_DIR/eval/"
echo "  - DAD training curves: $EXP_DIR/dad_models/"
echo "  - Final results: $EXP_DIR/results/"
echo ""
