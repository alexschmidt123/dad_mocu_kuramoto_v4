#!/bin/bash
# Phase 1: build or reuse shared probe banks (data/<run_slug>/), then build the
# PyCUDA control U-bank for the same data directory. Creates/links an experiment folder.
#
# Usage:
#   ./scripts/data_generation.sh -config ieee5_config [-T <horizon>] [-exp-dir <folder>]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

ARGS=(generate-data --config "$CONFIG")
[[ -n "$T" ]] && ARGS+=(-T "$T")
[[ -n "$EXP_DIR" ]] && ARGS+=(--exp-dir "$EXP_DIR")

python3 -m src.cli "${ARGS[@]}"

echo ""
echo "=== Phase 1b: control U-bank (PyCUDA) ==="
python3 -m src.cli generate-control-bank --config "$CONFIG"
