#!/bin/bash
# Phase 5: evaluate DAD, Myopic, Fixed, and Random → experiments/<run>/eval/
#
# Usage:
#   ./scripts/evaluation.sh -exp-dir <experiment_folder> [-method <method>]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EXP_DIR=""
METHOD=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -exp-dir) EXP_DIR="$2"; shift 2 ;;
        -method) METHOD="$2"; shift 2 ;;
        *)
            echo "Usage: $0 -exp-dir <experiment_folder> [-method <method>]" >&2
            exit 1
            ;;
    esac
done

[[ -n "$EXP_DIR" ]] || {
    echo "Usage: $0 -exp-dir <experiment_folder> [-method <method>]" >&2
    exit 1
}

ARGS=(evaluate --exp-dir "$EXP_DIR")
[[ -n "$METHOD" ]] && ARGS+=(--method "$METHOD")

exec python3 -m src.cli "${ARGS[@]}"
