#!/bin/bash
# Evaluate methods into a stamped result folder.
#
# Usage:
#   ./scripts/evaluation.sh --config configs/ieee5.yaml --T 8
#   ./scripts/evaluation.sh --config configs/ieee5.yaml --exp-dir experiments/...

set -euo pipefail
# shellcheck source=../run.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run.sh"

CONFIG=""
METHOD=""
SMOKE=""
EXPERIMENT_TYPE="$EXPERIMENT_TYPE_DEFAULT"
EXP_DIR=""
T=""
N_OBS="$DEFAULT_N_OBS"
NOISE_SIGMA="$DEFAULT_NOISE_SIGMA"

usage() {
    echo "Usage: $0 --config <config.yaml> [--T <horizon>] [--N_obs <count>] [--noise_sigma <sigma>] [--experiment_type objective_based|eig_based] [--method <method>] [--exp-dir <path>] [--smoke]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|-config|-c) CONFIG="$2"; shift 2 ;;
        -T|--T|--step-number|--step_number) T="$2"; shift 2 ;;
        --N_obs|--n-obs|--n_obs) N_OBS="$2"; shift 2 ;;
        --noise_sigma|--noise-sigma) NOISE_SIGMA="$2"; shift 2 ;;
        --method|-method|-m) METHOD="$2"; shift 2 ;;
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

echo "=== evaluation (config=$CONFIG type=$EXPERIMENT_TYPE T=${T:-$DEFAULT_STEP_NUMBER} N_obs=$N_OBS noise_sigma=$NOISE_SIGMA method=${METHOD:-ALL}) ==="
ARGS=(evaluate --config "$CONFIG" --experiment-type "$EXPERIMENT_TYPE" --N_obs "$N_OBS" --noise_sigma "$NOISE_SIGMA")
[[ -n "$T" ]] && ARGS+=(-T "$T")
[[ -n "$EXP_DIR" ]] && ARGS+=(--exp-dir "$EXP_DIR")
[[ -n "$METHOD" ]] && ARGS+=(--method "$METHOD")
exec python3 -m src.experiment "${ARGS[@]}" ${SMOKE}
