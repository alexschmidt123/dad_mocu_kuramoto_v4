#!/bin/bash
# Particle posterior adequacy study: generate master 2048-θ banks (IEEE5/IEEE9).
# Physical data → data/<system>_particle_adequacy_master_2048/
# Experiment stubs → experiments/particle_posterior_adequacy/
#
# Usage:
#   ./scripts/particle_posterior_adequacy.sh -system both -stage generate-master
#   ./scripts/particle_posterior_adequacy.sh -system ieee5 -stage generate-master

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

echo "=== particle_posterior_adequacy: stage=${STAGE} system=${SYSTEM} ==="
echo "Historical banks preserved: data/ieee5 data/ieee9"
echo "IEEE14 is out of scope for this study."
echo "No DAD / RL-sBOED training in this study."

case "$STAGE" in
    generate-master)
        exec python3 -m src.control.particle_posterior_adequacy generate-master \
            --system "$SYSTEM" "${EXTRA[@]}"
        ;;
    run|run-system)
        exec python3 -m src.control.particle_posterior_adequacy run \
            --system "$SYSTEM" ${SMOKE} "${EXTRA[@]}"
        ;;
    report)
        exec python3 -m src.control.particle_posterior_adequacy report
        ;;
    *)
        echo "Unknown stage: $STAGE (supported: generate-master|run|run-system|report)" >&2
        exit 1
        ;;
esac
