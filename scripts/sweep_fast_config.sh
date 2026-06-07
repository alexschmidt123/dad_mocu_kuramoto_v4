#!/bin/bash
# Full pipeline sweep: fast_config for T=1..4 (data → train → eval each).
#
# Usage:
#   ./scripts/sweep_fast_config.sh
#   ./scripts/sweep_fast_config.sh -from 1 -to 2

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="fast_config"
T_FROM=1
T_TO=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        -config) CONFIG="$2"; shift 2 ;;
        -from) T_FROM="$2"; shift 2 ;;
        -to) T_TO="$2"; shift 2 ;;
        *)
            echo "Usage: $0 [-config <name>] [-from <T>] [-to <T>]" >&2
            exit 1
            ;;
    esac
done

if ! [[ "$T_FROM" =~ ^[0-9]+$ && "$T_TO" =~ ^[0-9]+$ && "$T_FROM" -le "$T_TO" ]]; then
    echo "Invalid T range: from=$T_FROM to=$T_TO" >&2
    exit 1
fi

echo "Sweep: config=$CONFIG  T=$T_FROM..$T_TO"
echo ""

for T in $(seq "$T_FROM" "$T_TO"); do
    echo "========================================"
    echo "  T=$T"
    echo "========================================"
    ./run.sh -config "$CONFIG" -T "$T"
    echo ""
done

echo "Sweep complete (T=$T_FROM..$T_TO)."
