#!/bin/bash
# Phase 3: objective-observability gate (blocks DAD training if probe histories
# cannot change the posterior terminal control). Does not train or evaluate methods.
#
# Usage:
#   ./scripts/objective_observability.sh -exp-dir experiments/<run>
#   ./scripts/objective_observability.sh -exp-dir experiments/<run> -num-rollouts 1000 -seed 1234

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

ARGS=(check-objective-observability --exp-dir "$EXP_DIR")
[[ -n "$NUM_ROLLOUTS" ]] && ARGS+=(--num-rollouts "$NUM_ROLLOUTS")
[[ -n "$SEED" ]] && ARGS+=(--seed "$SEED")

exec python3 -m src.cli "${ARGS[@]}"
