#!/usr/bin/env bash
# Train a method into a stamped experiment folder (logs live under that folder).
#
# Usage:
#   ./scripts/training.sh --config configs/ieee9.yaml --method dad --T 5
#   ./scripts/training.sh --config configs/ieee9.yaml --method moe_sboed --exp-dir experiments/...
#   ./scripts/training.sh --config configs/ieee9.yaml --method all --seed 101
set -euo pipefail
# shellcheck source=../run.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run.sh"

CONFIG=""
METHOD="all"
SMOKE=""
EXPERIMENT_TYPE="$EXPERIMENT_TYPE_DEFAULT"
EXP_DIR=""
T=""
N_OBS="$DEFAULT_N_OBS"
NOISE_SIGMA="$DEFAULT_NOISE_SIGMA"
SEED=101

usage() {
    echo "Usage: $0 --config <config.yaml> [--method dad|rl_sboed|moe_sboed|matched_dense|all] [--T <horizon>] [--N_obs <count>] [--noise_sigma <sigma>] [--seed <int>] [--experiment_type objective_based|eig_based] [--exp-dir <path>] [--smoke]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|-config|-c) CONFIG="$2"; shift 2 ;;
        --method|-method|-m) METHOD="$2"; shift 2 ;;
        -T|--T|--step-number|--step_number) T="$2"; shift 2 ;;
        --N_obs|--n-obs|--n_obs) N_OBS="$2"; shift 2 ;;
        --noise_sigma|--noise-sigma) NOISE_SIGMA="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --experiment_type|--experiment-type)
            EXPERIMENT_TYPE="$(validate_experiment_type "$2")" || exit 1
            shift 2
            ;;
        --exp-dir|--exp_dir) EXP_DIR="$2"; shift 2 ;;
        --smoke) SMOKE="--smoke"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done

[[ -n "$CONFIG" ]] || { usage; exit 1; }

train_one() {
    local method="$1"
    echo "=== training method=$method (config=$CONFIG type=$EXPERIMENT_TYPE T=${T:-$DEFAULT_STEP_NUMBER} N_obs=$N_OBS noise_sigma=$NOISE_SIGMA seed=$SEED) ==="
    local args=(train --config "$CONFIG" --method "$method" --experiment-type "$EXPERIMENT_TYPE"
        --N_obs "$N_OBS" --noise_sigma "$NOISE_SIGMA" --seed "$SEED")
    [[ -n "$T" ]] && args+=(-T "$T")
    [[ -n "$EXP_DIR" ]] && args+=(--exp-dir "$EXP_DIR")
    python3 -m src.experiment "${args[@]}" ${SMOKE}
}

case "${METHOD,,}" in
    dad) train_one dad ;;
    rl_sboed|rl-sboed) train_one rl_sboed ;;
    moe_sboed|moe-sboed) train_one moe_sboed ;;
    matched_dense|matched-dense) train_one matched_dense ;;
    all)
        train_one dad
        train_one rl_sboed
        train_one moe_sboed
        ;;
    *)
        echo "Unknown --method: $METHOD (allowed: dad|rl_sboed|moe_sboed|matched_dense|all)" >&2
        exit 1
        ;;
esac
