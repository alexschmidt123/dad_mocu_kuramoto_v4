#!/bin/bash
# Sweep wrapper: call ./run.sh sequentially for one or more configs across T values.
# Each run.sh execution performs: banks → control-safety calibration → observability
# gate → DAD train → evaluate.
# Uses a file lock so only one sweep may run at a time (no parallel duplicate folders).
#
# Usage:
#   ./sweep_run.sh -config ieee14_config
#   ./sweep_run.sh -config ieee9_config -from 1 -to 5
#   ./sweep_run.sh -config ieee5_config,ieee9_config,ieee14_config -from 1 -to 6

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LOCK_FILE="$ROOT/.sweep.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "Another sweep is already running (lock: $LOCK_FILE). Wait for it to finish." >&2
    exit 1
fi

CONFIGS=()
T_FROM=1
T_TO=4

add_configs() {
    local raw="$1"
    local item
    IFS=',' read -ra parts <<< "$raw"
    for item in "${parts[@]}"; do
        item="${item//[[:space:]]/}"
        [[ -n "$item" ]] && CONFIGS+=("$item")
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -config) add_configs "$2"; shift 2 ;;
        -from) T_FROM="$2"; shift 2 ;;
        -to) T_TO="$2"; shift 2 ;;
        *)
            echo "Usage: $0 -config <name[,name...]|path> [-config <name> ...] [-from <T>] [-to <T>]" >&2
            exit 1
            ;;
    esac
done

[[ "${#CONFIGS[@]}" -gt 0 ]] || {
    echo "Usage: $0 -config <name[,name...]|path> [-config <name> ...] [-from <T>] [-to <T>]" >&2
    exit 1
}

if ! [[ "$T_FROM" =~ ^[0-9]+$ && "$T_TO" =~ ^[0-9]+$ && "$T_FROM" -le "$T_TO" ]]; then
    echo "Invalid T range: from=$T_FROM to=$T_TO" >&2
    exit 1
fi

echo "Sweep: configs=${CONFIGS[*]}  T=$T_FROM..$T_TO"
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    for T in $(seq "$T_FROM" "$T_TO"); do
        echo "========================================"
        echo "  config=$CONFIG  T=$T"
        echo "========================================"
        ./run.sh -config "$CONFIG" -T "$T"
        echo ""
    done
done

echo "Sweep complete: configs=${CONFIGS[*]}  T=$T_FROM..$T_TO"
