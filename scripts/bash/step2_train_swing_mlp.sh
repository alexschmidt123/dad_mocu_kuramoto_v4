#!/bin/bash
# Step 2: Train Swing MLP predictor (for second-order Kuramoto / swing equation)
# This script trains the simple MLP predictor for swing equation MOCU prediction

set -e

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file> [data_file]"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
BASE_CONFIG_NAME=$(echo "$CONFIG_NAME" | sed 's/_K[0-9]*$//')
EPOCHS=$(grep "epochs:" $CONFIG_FILE | awk '{print $2}')
BATCH_SIZE=$(grep "batch_size:" $CONFIG_FILE | awk '{print $2}')
LEARNING_RATE=$(grep "learning_rate:" $CONFIG_FILE | awk '{print $2}')

# Get model name from config (remove quotes if present)
MODEL_NAME=$(grep -A 5 "^training:" $CONFIG_FILE | grep "model_name:" | awk '{print $2}' | tr -d '"' | tr -d "'" || echo "$BASE_CONFIG_NAME")

# Use model name for model folder
MODEL_FOLDER="${PROJECT_ROOT}/models/${MODEL_NAME}/"
MODEL_FILE="${MODEL_FOLDER}model.pth"
STATS_FILE="${MODEL_FOLDER}statistics.pth"

# Get data file from argument or temp file or default location
if [ -n "$2" ]; then
    DATA_FILE="$2"
else
    DATA_FILE=$(cat /tmp/mocu_train_file_${CONFIG_NAME}.txt 2>/dev/null || echo "")
    if [ -z "$DATA_FILE" ]; then
        # Default location
        DATA_FILE="${PROJECT_ROOT}/data/${MODEL_NAME}_mocu_data.npz"
    fi
fi

if [ -z "$DATA_FILE" ] || [ ! -f "$DATA_FILE" ]; then
    echo "Error: Training data file not found: $DATA_FILE"
    echo "  Run step1_generate_mocu_data.sh first."
    exit 1
fi

# Check if model already exists - skip training if so
if [ -f "$MODEL_FILE" ] && [ -f "$STATS_FILE" ] && [ -f "$DATA_FILE" ]; then
    echo "✓ Swing MLP model already exists: $MODEL_FILE"
    echo "✓ Skipping training (model and data detected)"
    # Remove quotes from model name before saving
    MODEL_NAME_CLEAN=$(echo "$MODEL_NAME" | tr -d '"' | tr -d "'")
    echo "${MODEL_NAME_CLEAN}" > /tmp/mocu_model_name_${CONFIG_NAME}.txt
    echo "$MODEL_FOLDER" > /tmp/mocu_model_folder_${CONFIG_NAME}.txt
    exit 0
fi

echo "Training Swing MLP predictor (Step 2/6)..."
echo "  Epochs=$EPOCHS, Batch size=$BATCH_SIZE, LR=$LEARNING_RATE"
echo "  Output: $MODEL_FOLDER"
mkdir -p "$MODEL_FOLDER"

cd "${PROJECT_ROOT}/scripts"
ABS_DATA_FILE=$(cd "$(dirname "$DATA_FILE")" && pwd)/$(basename "$DATA_FILE")
ABS_MODEL_FOLDER=$(cd "$MODEL_FOLDER" && pwd)

# Resolve config file path to absolute (before changing to scripts directory)
if [[ "$CONFIG_FILE" = /* ]]; then
    # Already absolute path
    ABS_CONFIG_FILE="$CONFIG_FILE"
else
    # Relative path - resolve from PROJECT_ROOT
    ABS_CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
fi

python3 train_swing_mlp_predictor.py \
    --config "$ABS_CONFIG_FILE" \
    --data_file "$ABS_DATA_FILE" \
    --epochs ${EPOCHS:-400} \
    --batch_size ${BATCH_SIZE:-128} \
    --learning_rate ${LEARNING_RATE:-0.001}

echo "✓ Swing MLP predictor trained: ${MODEL_FILE}"
# Remove quotes from model name before saving
MODEL_NAME_CLEAN=$(echo "$MODEL_NAME" | tr -d '"' | tr -d "'")
echo "${MODEL_NAME_CLEAN}" > /tmp/mocu_model_name_${CONFIG_NAME}.txt
echo "$MODEL_FOLDER" > /tmp/mocu_model_folder_${CONFIG_NAME}.txt

