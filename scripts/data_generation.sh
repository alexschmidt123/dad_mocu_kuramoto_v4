#!/bin/bash
# Phase 1: build or reuse shared train/test data (data/<config>_T<T>/).

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

exec python3 -m src.cli "${ARGS[@]}"
