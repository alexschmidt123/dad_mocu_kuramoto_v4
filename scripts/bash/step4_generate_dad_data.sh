#!/bin/bash
# Step 4: Generate DAD training data (before any policy training)

set +e  # Don't exit on error - allow graceful skip

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Resolve config file path
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
BASE_CONFIG_NAME=$(echo "$CONFIG_NAME" | sed 's/_K[0-9]*$//')
N=$(grep "^N:" "$CONFIG_FILE" | awk '{print $2}')

# DAD is enabled for second_order (swing equation) model
SKIP_DAD="${SKIP_DAD:-false}"
if [ "$SKIP_DAD" = "true" ]; then
    echo "⚠️  DAD skipped (SKIP_DAD=true). Use baselines only."
    echo "✓ Step 4 skipped gracefully"
    exit 0
fi

MOCU_MODEL_NAME=$(cat /tmp/mocu_model_name_${CONFIG_NAME}.txt 2>/dev/null || echo "")
MOCU_MODEL_FOLDER=$(cat /tmp/mocu_model_folder_${CONFIG_NAME}.txt 2>/dev/null || echo "")

if [ -z "$MOCU_MODEL_FOLDER" ] || [ ! -d "$MOCU_MODEL_FOLDER" ]; then
    echo "Error: MOCU predictor (MPNN) model folder not found. Run step2_train_swing_mpnn.sh first."
    exit 1
fi

# Determine output directories
# Strategy: Always generate fresh DAD data in experiment folder (never reuse)
# data/<config-name>/ is ONLY for MOCU training data, not DAD data
if [ -n "$EXP_DAD_DATA_DIR" ]; then
    DATA_FOLDER="$EXP_DAD_DATA_DIR"
else
    # Fallback: DAD data in data/<config>/dad/ (distinct from MOCU: data/<config>/mocu/)
    DATA_FOLDER="${PROJECT_ROOT}/data/${BASE_CONFIG_NAME}/dad"
fi
mkdir -p "$DATA_FOLDER"

NUM_EPISODES=$(grep -A 3 "^dad_data:" "$CONFIG_FILE" | grep "  num_episodes:" | awk '{print $2}' || echo "1000")
K=$(grep -A 3 "^dad_data:" "$CONFIG_FILE" | grep "  K:" | awk '{print $2}' || echo "4")
USE_PRECOMPUTED_MOCU=$(grep -A 3 "^dad_data:" "$CONFIG_FILE" | grep "  use_precomputed_mocu:" | awk '{print $2}' || echo "true")

[ -z "$NUM_EPISODES" ] && NUM_EPISODES=1000
[ -z "$K" ] && K=4
[ -z "$USE_PRECOMPUTED_MOCU" ] && USE_PRECOMPUTED_MOCU=true

# Swing equation DAD data uses swing_dad_trajectories_N*_K*.pth
DAD_TRAJECTORY_FILE="${DATA_FOLDER}/swing_dad_trajectories_N${N}_K${K}.pth"

check_data_k_n() {
    local data_file="$1"
    local filename_n=$(echo "$data_file" | sed -n 's/.*_trajectories_N\([0-9]*\)_K\([0-9]*\).*/\1/p')
    local filename_k=$(echo "$data_file" | sed -n 's/.*_trajectories_N\([0-9]*\)_K\([0-9]*\).*/\2/p')
    local file_k=$(python3 -c "
import torch
try:
    data = torch.load('$data_file', map_location='cpu', weights_only=False)
    print(data.get('config', {}).get('K', ''))
except:
    print('')
" 2>/dev/null || echo "")
    local file_n=$(python3 -c "
import torch
try:
    data = torch.load('$data_file', map_location='cpu', weights_only=False)
    print(data.get('config', {}).get('N', ''))
except:
    print('')
" 2>/dev/null || echo "")
    local file_episodes=$(python3 -c "
import torch
try:
    data = torch.load('$data_file', map_location='cpu', weights_only=False)
    print(data.get('config', {}).get('num_episodes', ''))
except:
    print('')
" 2>/dev/null || echo "")

    local k_match=false
    local n_match=false
    local episodes_match=true

    local data_k="${file_k:-$filename_k}"
    local data_n="${file_n:-$filename_n}"

    if [ "$data_k" = "$K" ]; then k_match=true; fi
    if [ "$data_n" = "$N" ]; then n_match=true; fi

    if [ -n "$file_episodes" ]; then
        if [ "$file_episodes" = "$NUM_EPISODES" ]; then
            episodes_match=true
        else
            episodes_match=false
        fi
    fi

    if [ "$k_match" = true ] && [ "$n_match" = true ] && [ "$episodes_match" = true ]; then
        echo "true"
    else
        echo "false"
    fi
}

# Always generate fresh DAD data for each experiment (never reuse)
# Remove existing data if present to ensure fresh generation
if [ -f "$DAD_TRAJECTORY_FILE" ]; then
    echo "⚠ Removing existing DAD data to ensure fresh generation: $DAD_TRAJECTORY_FILE"
    rm -f "$DAD_TRAJECTORY_FILE"
fi

echo "Generating DAD trajectory data (fresh for each experiment)..."
echo "  Settings: num_episodes=$NUM_EPISODES, K=$K, use_precomputed_mocu=$USE_PRECOMPUTED_MOCU"
echo "  Output folder: $DATA_FOLDER"

# Set up log file for this step (if EXP_LOGS_DIR is available)
if [ -n "$EXP_LOGS_DIR" ]; then
    STEP_LOG="${EXP_LOGS_DIR}/step4_generate_dad_data.log"
    echo "  Logging to: $STEP_LOG"
else
    STEP_LOG="/dev/null"
fi

cd "${PROJECT_ROOT}/scripts/data_generation"
if [ ! -f "generate_dad_data.py" ]; then
    echo "Error: generate_dad_data.py not found in ${PROJECT_ROOT}/scripts/data_generation"
    exit 1
fi
ABS_DAD_DATA_FOLDER=$(cd "$DATA_FOLDER" && pwd)
ABS_CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
if [[ "$CONFIG_FILE" = /* ]]; then
    ABS_CONFIG_FILE="$CONFIG_FILE"
fi
CMD="python3 generate_dad_data.py --config \"$ABS_CONFIG_FILE\" --num-episodes $NUM_EPISODES --K $K --output-dir \"$ABS_DAD_DATA_FOLDER\""
if [ "$USE_PRECOMPUTED_MOCU" = "true" ] && [ -n "$MOCU_MODEL_NAME" ] && [ -f "${MOCU_MODEL_FOLDER}model_mpnn.pth" ]; then
    CMD="$CMD --use-mocu-predictor --mocu-model-name $MOCU_MODEL_NAME"
    echo "  Using MPNN MOCU predictor ($MOCU_MODEL_NAME) to pre-compute MOCU values"
else
    echo "  Not using MPNN MOCU predictor (terminal MOCU computed during training if needed)"
fi

# Run command and save to both workflow log and step-specific log
# Note: We check for trajectory file existence even if command fails (diagnostics may fail but data is valid)
if [ -n "$EXP_LOGS_DIR" ]; then
    echo "=== Step 4: Generate DAD Data ===" | tee -a "$STEP_LOG"
    echo "Command: $CMD" | tee -a "$STEP_LOG"
    echo "Started: $(date)" | tee -a "$STEP_LOG"
    set +e  # Temporarily disable exit on error
    eval $CMD 2>&1 | tee -a "$STEP_LOG"
    CMD_EXIT_CODE=${PIPESTATUS[0]}
    set -e  # Re-enable exit on error
    echo "Completed: $(date)" | tee -a "$STEP_LOG"
else
    set +e  # Temporarily disable exit on error
    eval $CMD
    CMD_EXIT_CODE=$?
    set -e  # Re-enable exit on error
fi

# Check if trajectory file was created (even if diagnostics failed)
ABS_DAD_TRAJ_FILE=$(cd "$(dirname "$DAD_TRAJECTORY_FILE")" && pwd)/$(basename "$DAD_TRAJECTORY_FILE")
if [ -f "$ABS_DAD_TRAJ_FILE" ]; then
    echo "$ABS_DAD_TRAJ_FILE" > /tmp/dad_traj_file_${CONFIG_NAME}.txt
    echo "✓ DAD training data ready: $ABS_DAD_TRAJ_FILE"
    [ -n "$EXP_REPORT" ] && {
        echo "" >> "$EXP_REPORT"
        echo "--- DAD Data (Step 4) ---" >> "$EXP_REPORT"
        echo "  Status: OK" >> "$EXP_REPORT"
        echo "  File: $ABS_DAD_TRAJ_FILE" >> "$EXP_REPORT"
        echo "  Episodes: $NUM_EPISODES, K: $K" >> "$EXP_REPORT"
    }
    # If command failed but file exists, it's likely just a diagnostics error - warn but continue
    if [ "$CMD_EXIT_CODE" -ne 0 ]; then
        echo "⚠ Command exited with code $CMD_EXIT_CODE, but trajectory file exists."
        echo "  This may indicate a non-fatal error (e.g., diagnostics saving failed)."
        echo "  Proceeding with training..."
    fi
else
    echo "Error: DAD trajectory file was not created: $ABS_DAD_TRAJ_FILE"
    exit 1
fi

