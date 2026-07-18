#!/bin/bash
# Continuous u_ctrl + existing-amplitude adaptive-value diagnostic.
# Outputs: experiments/continuous_uctrl_amplitude_adaptive_value/
#
# Usage:
#   ./scripts/continuous_uctrl_amplitude.sh -system both -stage run
#   ./scripts/continuous_uctrl_amplitude.sh -system ieee5 -stage diagnose --smoke

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Prefer project conda env when available (PyTorch / banks tooling).
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

echo "=== continuous_uctrl_amplitude: stage=${STAGE} system=${SYSTEM} ==="

case "$STAGE" in
    audit)
        exec python3 -m src.control.continuous_uctrl_amplitude audit --system "$SYSTEM" "${EXTRA[@]}"
        ;;
    diagnose)
        exec python3 -m src.control.continuous_uctrl_amplitude diagnose --system "$SYSTEM" ${SMOKE} "${EXTRA[@]}"
        ;;
    report)
        exec python3 -m src.control.continuous_uctrl_amplitude report
        ;;
    run)
        exec python3 -m src.control.continuous_uctrl_amplitude run --system "$SYSTEM" ${SMOKE} "${EXTRA[@]}"
        ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        echo "Supported: audit|diagnose|report|run" >&2
        exit 1
        ;;
esac
