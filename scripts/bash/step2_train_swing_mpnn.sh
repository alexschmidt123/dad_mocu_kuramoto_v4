#!/bin/bash
# Step 2: Train Swing MPNN MOCU predictor (for second-order Kuramoto / swing equation)
# Paper (first-order iNN/NN) used MPNN for MOCU estimation.

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
N=$(grep "^N:" $CONFIG_FILE | awk '{print $2}')
EPOCHS=$(grep "epochs:" $CONFIG_FILE | awk '{print $2}')
BATCH_SIZE=$(grep "batch_size:" $CONFIG_FILE | awk '{print $2}')
LEARNING_RATE=$(grep "learning_rate:" $CONFIG_FILE | awk '{print $2}')

# Use config-based folder for all models: models/xxx_config/ (MPNN + DAD)
MODEL_FOLDER="${PROJECT_ROOT}/models/${BASE_CONFIG_NAME}/"
MODEL_FILE="${MODEL_FOLDER}model_mpnn.pth"
STATS_FILE="${MODEL_FOLDER}statistics.pth"
MODEL_NAME_CLEAN="$BASE_CONFIG_NAME"

if [ -n "$2" ]; then
    DATA_FILE="$2"
else
    DATA_FILE=$(cat /tmp/mocu_train_file_${CONFIG_NAME}.txt 2>/dev/null || echo "")
    if [ -z "$DATA_FILE" ]; then
        DATA_FILE="${PROJECT_ROOT}/data/${BASE_CONFIG_NAME}/mocu/swing_mocu_data_${N}.npz"
        [ ! -f "$DATA_FILE" ] && DATA_FILE="${PROJECT_ROOT}/data/${BASE_CONFIG_NAME}/mocu/swing_mocu_data_${N}.npz"
    fi
fi

if [ -z "$DATA_FILE" ] || [ ! -f "$DATA_FILE" ]; then
    echo "Error: Training data file not found: $DATA_FILE"
    echo "  Run step1_generate_mocu_data.sh first."
    exit 1
fi

# Always train/update MPNN when same config is run (models are overwritten)
if [ -f "$MODEL_FILE" ] && [ -f "$STATS_FILE" ]; then
    echo "Existing MPNN found; retraining to update: $MODEL_FILE"
fi

echo "Training Swing MPNN MOCU predictor (Step 2/6)..."
echo "  Epochs=$EPOCHS, Batch size=$BATCH_SIZE, LR=$LEARNING_RATE"
echo "  Output: $MODEL_FOLDER"
mkdir -p "$MODEL_FOLDER"

cd "${PROJECT_ROOT}/scripts/training"
ABS_DATA_FILE=$(cd "$(dirname "$DATA_FILE")" && pwd)/$(basename "$DATA_FILE")
if [[ "$CONFIG_FILE" != /* ]]; then
    ABS_CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
else
    ABS_CONFIG_FILE="$CONFIG_FILE"
fi

python3 train_swing_mpnn_predictor.py \
    --config "$ABS_CONFIG_FILE" \
    --data_file "$ABS_DATA_FILE" \
    --model_dir "$MODEL_FOLDER" \
    --epochs ${EPOCHS:-400} \
    --batch_size ${BATCH_SIZE:-128} \
    --learning_rate ${LEARNING_RATE:-0.001}

echo "✓ Swing MPNN MOCU predictor trained: ${MODEL_FILE}"
echo "${MODEL_NAME_CLEAN}" > /tmp/mocu_model_name_${CONFIG_NAME}.txt
echo "$MODEL_FOLDER" > /tmp/mocu_model_folder_${CONFIG_NAME}.txt
[ -n "$EXP_REPORT" ] && {
    echo "" >> "$EXP_REPORT"
    echo "--- MPNN Training (Step 2) ---" >> "$EXP_REPORT"
    echo "  Status: OK" >> "$EXP_REPORT"
    echo "  Model: $MODEL_FILE" >> "$EXP_REPORT"
    echo "  Epochs: ${EPOCHS:-400}, Batch: ${BATCH_SIZE:-128}" >> "$EXP_REPORT"
}
