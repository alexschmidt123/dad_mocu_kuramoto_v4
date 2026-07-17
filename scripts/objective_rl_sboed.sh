#!/bin/bash
# Objective RL-sBOED study workflow (DAD vs RL-sBOED vs baselines).
# Official outputs: experiments/objective_rl_sboed/
# Layout: run_config.yaml, train/, model/, eval/, diagnostics/, logs/, summary/
#
# Usage:
#   ./scripts/objective_rl_sboed.sh -system ieee5 -stage migrate-layout
#   ./scripts/objective_rl_sboed.sh -system ieee5 -stage sensitivity
#   ./scripts/objective_rl_sboed.sh -system ieee5 -stage run-system --smoke
#   ./scripts/objective_rl_sboed.sh -system both -stage sensitivity

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SYSTEM="both"
STAGE="sensitivity"
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

echo "=== objective_rl_sboed: stage=${STAGE} system=${SYSTEM} ==="

case "$STAGE" in
    migrate-layout)
        exec python3 -m src.control.objective_rl_sboed migrate-layout
        ;;
    sensitivity)
        if [[ "$SYSTEM" == "both" ]]; then
            exec python3 -m src.control.objective_rl_sboed sensitivity --system both ${SMOKE} "${EXTRA[@]}"
        else
            exec python3 -m src.control.objective_rl_sboed sensitivity --system "$SYSTEM" ${SMOKE} "${EXTRA[@]}"
        fi
        ;;
    train)
        exec python3 -m src.control.objective_rl_sboed train "${EXTRA[@]}" ${SMOKE}
        ;;
    baselines)
        exec python3 -m src.control.objective_rl_sboed baselines --system "$SYSTEM" "${EXTRA[@]}"
        ;;
    compare)
        exec python3 -m src.control.objective_rl_sboed compare --system "$SYSTEM" "${EXTRA[@]}"
        ;;
    run-system)
        exec python3 -m src.control.objective_rl_sboed run-system --system "$SYSTEM" ${SMOKE} "${EXTRA[@]}"
        ;;
    report)
        exec python3 -m src.control.objective_rl_sboed report
        ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        echo "Supported: migrate-layout|sensitivity|train|baselines|compare|run-system|report" >&2
        exit 1
        ;;
esac
