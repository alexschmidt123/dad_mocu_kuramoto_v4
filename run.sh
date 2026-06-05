#!/bin/bash
# Full sBOED workflow: data generation → DAD training → evaluation.
#
# Usage:
#   ./run.sh -config fast_config              # T=3 (default)
#   ./run.sh -config fast_config -T 1         # single probe step
#   ./run.sh -config fast_config -exp-dir <run>   # resume train/eval only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG=""
EXP_DIR=""
T=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -config) CONFIG="$2"; shift 2 ;;
        -exp-dir) EXP_DIR="$2"; shift 2 ;;
        -T) T="$2"; shift 2 ;;
        *)
            echo "Usage: $0 -config <name|path> [-T <horizon>] [-exp-dir <experiment_folder>]" >&2
            exit 1
            ;;
    esac
done

[[ -n "$CONFIG" ]] || {
    echo "Usage: $0 -config <name|path> [-T <horizon>] [-exp-dir <experiment_folder>]" >&2
    exit 1
}

DATA_ARGS=(-config "$CONFIG")
[[ -n "$T" ]] && DATA_ARGS+=(-T "$T")
[[ -n "$EXP_DIR" ]] && DATA_ARGS+=(-exp-dir "$EXP_DIR")

if [[ -z "$EXP_DIR" ]]; then
    echo "=== Phase 1: data generation ==="
    OUTPUT="$(./scripts/data_generation.sh "${DATA_ARGS[@]}" 2>&1 | tee /dev/stderr)"
    EXP_DIR="$(echo "$OUTPUT" | sed -n 's/^EXP_DIR=//p' | tail -1)"
    [[ -n "$EXP_DIR" ]] || { echo "Failed to resolve experiment directory" >&2; exit 1; }
else
    echo "=== Skipping data generation (using -exp-dir) ==="
fi

echo ""
echo "=== Phase 2: DAD training ==="
./scripts/dad_training.sh -exp-dir "$EXP_DIR"

echo ""
echo "=== Phase 3: evaluation ==="
./scripts/evaluation.sh -exp-dir "$EXP_DIR"

echo ""
echo "Done. Experiment → $EXP_DIR"
