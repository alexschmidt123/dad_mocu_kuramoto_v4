#!/bin/bash
# Phase 2: calibrate the common posterior→u_ctrl rule (alpha, margin) for
# empirical true-system safety. Uses train-only calibration/validation splits;
# never touches the final test set. Does not train or evaluate named methods.
#
# Usage:
#   ./scripts/control_safety_calibration.sh -exp-dir experiments/<run>
#   ./scripts/control_safety_calibration.sh -exp-dir experiments/<run> -num-rollouts 2000 -seed 2468

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EXP_DIR=""
NUM_ROLLOUTS=""
SEED=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -exp-dir) EXP_DIR="$2"; shift 2 ;;
        -num-rollouts) NUM_ROLLOUTS="$2"; shift 2 ;;
        -seed) SEED="$2"; shift 2 ;;
        *)
            echo "Usage: $0 -exp-dir <experiment_folder> [-num-rollouts N] [-seed S]" >&2
            exit 1
            ;;
    esac
done

[[ -n "$EXP_DIR" ]] || {
    echo "Usage: $0 -exp-dir <experiment_folder> [-num-rollouts N] [-seed S]" >&2
    exit 1
}

ARGS=(calibrate-control-safety --exp-dir "$EXP_DIR")
[[ -n "$NUM_ROLLOUTS" ]] && ARGS+=(--num-rollouts "$NUM_ROLLOUTS")
[[ -n "$SEED" ]] && ARGS+=(--seed "$SEED")

exec python3 -m src.cli "${ARGS[@]}"
