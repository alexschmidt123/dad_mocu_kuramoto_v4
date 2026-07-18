#!/bin/bash
# Bus-location + joint bus-amplitude adaptive-value diagnostic.
# Outputs: experiments/bus_joint_adaptive_value/
#
# Usage:
#   ./scripts/bus_joint_adaptive_value.sh -system both -stage run
#   ./scripts/bus_joint_adaptive_value.sh -system ieee5 -stage run --smoke

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -x "/home/grads/g/g.lin/miniconda3/envs/dad_mocu_kuramoto/bin/python3" ]]; then
    export PATH="/home/grads/g/g.lin/miniconda3/envs/dad_mocu_kuramoto/bin:${PATH}"
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SYSTEM="both"
STAGE="run"
SMOKE=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -system) SYSTEM="$2"; shift 2 ;;
        -stage) STAGE="$2"; shift 2 ;;
        --smoke) SMOKE="--smoke"; shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

echo "=== bus_joint_adaptive_value: stage=${STAGE} system=${SYSTEM} ==="

case "$STAGE" in
    run)
        exec python3 -m src.control.bus_joint_adaptive_value run --system "$SYSTEM" ${SMOKE} "${EXTRA[@]}"
        ;;
    report)
        exec python3 -m src.control.bus_joint_adaptive_value report
        ;;
    *)
        echo "Unknown stage: $STAGE (supported: run|report)" >&2
        exit 1
        ;;
esac
