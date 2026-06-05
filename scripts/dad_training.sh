#!/bin/bash
# Phase 2: train DAD policy (config read from experiment dir).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EXP_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -exp-dir) EXP_DIR="$2"; shift 2 ;;
        *)
            echo "Usage: $0 -exp-dir <experiment_folder>" >&2
            exit 1
            ;;
    esac
done

[[ -n "$EXP_DIR" ]] || {
    echo "Usage: $0 -exp-dir <experiment_folder>" >&2
    exit 1
}

exec python3 -m src.cli train --exp-dir "$EXP_DIR"
