#!/bin/bash
# Step 3: Evaluate baseline methods first
# This runs baseline methods BEFORE DAD training, regardless of config
# Uses PyCUDA for MOCU computation (matches original paper 2023 workflow)

set -e

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
# Remove _K* suffix if present to get base config name (for fallback folder structure)
BASE_CONFIG_NAME=$(echo "$CONFIG_NAME" | sed 's/_K[0-9]*$//')

# Resolve absolute config path once (for Python and compare_methods.py)
if [[ "$CONFIG_FILE" = /* ]]; then
    ABS_CONFIG_FILE="$CONFIG_FILE"
else
    ABS_CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
fi
export ABS_CONFIG_FILE

# Get info from previous steps (remove quotes if present)
MOCU_MODEL_NAME=$(cat /tmp/mocu_model_name_${CONFIG_NAME}.txt 2>/dev/null | tr -d '"' | tr -d "'" || echo "")

# Parse config parameters
N=$(grep "^N:" "$CONFIG_FILE" | awk '{print $2}')
UPDATE_CNT=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "update_count:" | awk '{print $2}')
IT_IDX=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "it_idx:" | awk '{print $2}')
K_MAX=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "K_max:" | awk '{print $2}')
NUM_SIMULATIONS=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "num_simulations:" | awk '{print $2}')

# Validate and set defaults if empty
[ -z "$N" ] && N=5
[ -z "$UPDATE_CNT" ] && UPDATE_CNT=10
[ -z "$IT_IDX" ] && IT_IDX=10
[ -z "$K_MAX" ] && K_MAX=20480
[ -z "$NUM_SIMULATIONS" ] && NUM_SIMULATIONS=10

# Methods: from config experiment.methods (baseline-only) or default
# DAD is evaluated in step6, not here
BASELINE_METHODS=$(python3 -c "
import yaml
import sys
import os
config_path = os.environ.get('ABS_CONFIG_FILE', '$CONFIG_FILE')
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    methods = config.get('experiment', {}).get('methods', [])
    if isinstance(methods, list):
        # Keep only baseline methods (DAD is run in step6)
        baseline = [m for m in methods if m in ('RANDOM', 'ENTROPY', 'ODE', 'iNN', 'NN')]
        if baseline:
            print(','.join(baseline))
            sys.exit(0)
except Exception:
    pass
print('iNN,NN,ODE,ENTROPY,RANDOM')
" 2>/dev/null || echo "iNN,NN,ODE,ENTROPY,RANDOM")

# If MOCU_MODEL_NAME not set (e.g. /tmp cleared), resolve from config for iNN/NN
if [ -z "$MOCU_MODEL_NAME" ] || [ "$MOCU_MODEL_NAME" = "" ]; then
    MOCU_MODEL_NAME=$(grep -A 5 "^training:" "$CONFIG_FILE" 2>/dev/null | grep "model_name:" | awk '{print $2}' | tr -d '"' | tr -d "'" || echo "")
    [ -n "$MOCU_MODEL_NAME" ] && export MOCU_MODEL_NAME
fi

if [ -n "$EXP_EVAL_DIR" ]; then
    RESULT_RUN_FOLDER="$EXP_EVAL_DIR"
else
    TIMESTAMP=$(date +"%m%d%Y_%H%M%S")
    RESULT_RUN_FOLDER="${PROJECT_ROOT}/results/${BASE_CONFIG_NAME}/${TIMESTAMP}/"
fi
mkdir -p "$RESULT_RUN_FOLDER"

export MOCU_MODEL_NAME="$MOCU_MODEL_NAME"
export RESULT_FOLDER="$RESULT_RUN_FOLDER"
export EVAL_N="$N"
export EVAL_UPDATE_CNT="$UPDATE_CNT"
export EVAL_IT_IDX="$IT_IDX"
export EVAL_K_MAX="$K_MAX"
export EVAL_NUM_SIMULATIONS="$NUM_SIMULATIONS"

echo "Running baseline evaluation (Step 3/6)..."
echo "  Methods: $BASELINE_METHODS"
echo "  N=$N, update_cnt=$UPDATE_CNT, it_idx=$IT_IDX, K_max=$K_MAX, num_simulations=$NUM_SIMULATIONS"
echo "  Results: $RESULT_RUN_FOLDER"

# Use scripts/evaluation/ location
cd "${PROJECT_ROOT}/scripts/evaluation"
if [ ! -f "compare_methods.py" ]; then
    echo "Error: compare_methods.py not found in ${PROJECT_ROOT}/scripts/evaluation"
    exit 1
fi

python3 compare_methods.py --methods "$BASELINE_METHODS" --config "$ABS_CONFIG_FILE"

echo "✓ Baseline evaluation complete: $RESULT_RUN_FOLDER"
echo "$RESULT_RUN_FOLDER" > /tmp/baseline_results_folder_${CONFIG_NAME}.txt

# Step 3.5: Visualize baseline-only results
echo ""
echo "Generating baseline-only visualizations..."
ABS_RESULT_FOLDER=$(cd "$RESULT_RUN_FOLDER" && pwd)
if [ "${ABS_RESULT_FOLDER: -1}" != "/" ]; then
    ABS_RESULT_FOLDER="${ABS_RESULT_FOLDER}/"
fi

if [ -n "$UPDATE_CNT" ]; then
    python3 "${PROJECT_ROOT}/scripts/visualization/visualize.py" --N $N --update_cnt $UPDATE_CNT --result_folder "$ABS_RESULT_FOLDER" --baseline_only
else
    python3 "${PROJECT_ROOT}/scripts/visualization/visualize.py" --N $N --result_folder "$ABS_RESULT_FOLDER" --baseline_only
fi

echo "✓ Baseline-only visualizations generated"

