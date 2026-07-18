#!/bin/bash
# Full sBOED workflow (five phases):
#   1) generate/reuse probe + control banks
#   2) install/verify frozen terminal-control rule OR calibrate (see config mode)
#   3) objective-observability gate (blocks training if it fails)
#   4) train objective-based DAD
#   5) evaluate DAD, Myopic, Fixed, Random
#
# Usage:
#   ./run.sh -config ieee5_config -T 2
#   ./run.sh -config ieee5_config -T 3
#   ./run.sh -config ieee5_config -exp-dir experiments/<run>   # skip data gen; still runs phase 2+
#   ./run.sh -study objective_rl_sboed -system ieee5 [-stage run-system|sensitivity] [--smoke]
#   ./run.sh -study continuous_uctrl_amplitude -system both [-stage run|diagnose|audit] [--smoke]
#   ./run.sh -study bus_joint_adaptive_value -system both [-stage run|report] [--smoke]
#   ./run.sh -study particle_posterior_adequacy -system both [-stage run|generate-master|report] [--smoke]
#   ./sweep_run.sh -study particle_posterior_adequacy -system both
#
# IEEE5 currently uses control_safety_calibration.mode=frozen with margin 0.55
# (policy-robust rule). Phase 2 verifies the rule and does not recalibrate.
#
# Controlled IEEE5 horizons (frozen margin 0.55):
#   ./run.sh -config ieee5_config -T 3
#   ./run.sh -config ieee5_config -T 4

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG=""
EXP_DIR=""
T=""
STUDY=""
SYSTEM="ieee5"
STAGE="run-system"
SMOKE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -config) CONFIG="$2"; shift 2 ;;
        -exp-dir) EXP_DIR="$2"; shift 2 ;;
        -T) T="$2"; shift 2 ;;
        -study) STUDY="$2"; shift 2 ;;
        -system) SYSTEM="$2"; shift 2 ;;
        -stage) STAGE="$2"; shift 2 ;;
        --smoke) SMOKE="--smoke"; shift ;;
        *)
            echo "Usage: $0 (-config <name|path> [-T <horizon>] [-exp-dir <dir>]) | (-study <name> -system <ieee5|ieee9|both> [-stage <stage>] [--smoke])" >&2
            exit 1
            ;;
    esac
done

if [[ -n "$STUDY" ]]; then
    case "$STUDY" in
        objective_rl_sboed)
            echo "=== Study: objective_rl_sboed (system=$SYSTEM stage=$STAGE) ==="
            exec ./scripts/objective_rl_sboed.sh -system "$SYSTEM" -stage "$STAGE" ${SMOKE}
            ;;
        continuous_uctrl_amplitude)
            echo "=== Study: continuous_uctrl_amplitude (system=$SYSTEM stage=$STAGE) ==="
            exec ./scripts/continuous_uctrl_amplitude.sh -system "$SYSTEM" -stage "$STAGE" ${SMOKE}
            ;;
        bus_joint_adaptive_value)
            echo "=== Study: bus_joint_adaptive_value (system=$SYSTEM stage=$STAGE) ==="
            exec ./scripts/bus_joint_adaptive_value.sh -system "$SYSTEM" -stage "$STAGE" ${SMOKE}
            ;;
        particle_posterior_adequacy)
            echo "=== Study: particle_posterior_adequacy (system=$SYSTEM stage=$STAGE) ==="
            exec ./scripts/particle_posterior_adequacy.sh -system "$SYSTEM" -stage "$STAGE" ${SMOKE}
            ;;
        *)
            echo "Unknown study: $STUDY (supported: objective_rl_sboed|continuous_uctrl_amplitude|bus_joint_adaptive_value|particle_posterior_adequacy)" >&2
            exit 1
            ;;
    esac
fi

[[ -n "$CONFIG" ]] || {
    echo "Usage: $0 -config <name|path> [-T <horizon>] [-exp-dir <experiment_folder>]" >&2
    echo "   or: $0 -study objective_rl_sboed|continuous_uctrl_amplitude|bus_joint_adaptive_value|particle_posterior_adequacy -system <ieee5|ieee9|both> [-stage <stage>] [--smoke]" >&2
    exit 1
}

# Dedicated controlled T=3/T=4 paths use the selected config's topology
# (frozen margins come from that system's calibrated/frozen rule).
if [[ "${T:-}" == "3" && -z "$EXP_DIR" ]]; then
    case "$CONFIG" in
        *ieee5*)
            echo "=== IEEE5 T=3 controlled experiment (frozen margin 0.55) ==="
            exec python3 -m src.cli run-ieee5-t3
            ;;
        *ieee9*)
            echo "=== IEEE9 T=3: use existing experiments/ieee9_T3 or -exp-dir ==="
            echo "Pass -exp-dir experiments/ieee9_T3 to continue from an existing run." >&2
            exit 1
            ;;
        *ieee14*)
            echo "IEEE14 T=3 is deferred until IEEE5/IEEE9 objective_rl_sboed completes." >&2
            exit 1
            ;;
    esac
fi
if [[ "${T:-}" == "4" && -z "$EXP_DIR" ]]; then
    case "$CONFIG" in
        *ieee5*)
            echo "=== IEEE5 T=4 controlled experiment (frozen margin 0.55) ==="
            exec python3 -m src.cli run-ieee5-t4
            ;;
        *)
            echo "T=4 controlled path is currently implemented for ieee5_config only." >&2
            exit 1
            ;;
    esac
fi

DATA_ARGS=(-config "$CONFIG")
[[ -n "$T" ]] && DATA_ARGS+=(-T "$T")
[[ -n "$EXP_DIR" ]] && DATA_ARGS+=(-exp-dir "$EXP_DIR")

if [[ -z "$EXP_DIR" ]]; then
    echo "=== Phase 1: probe + control banks ==="
    TMP_OUTPUT="$(mktemp)"
    ./scripts/data_generation.sh "${DATA_ARGS[@]}" 2>&1 | tee "$TMP_OUTPUT"
    OUTPUT="$(<"$TMP_OUTPUT")"
    rm -f "$TMP_OUTPUT"
    EXP_DIR="$(echo "$OUTPUT" | sed -n 's/^EXP_DIR=//p' | tail -1)"
    [[ -n "$EXP_DIR" ]] || { echo "Failed to resolve experiment directory" >&2; exit 1; }
else
    echo "=== Skipping Phase 1 data generation (using -exp-dir) ==="
fi

echo ""
echo "=== Phase 2: control-safety rule (frozen verify or calibrate) ==="
./scripts/control_safety_calibration.sh -exp-dir "$EXP_DIR"

echo ""
echo "=== Phase 3: objective-observability gate ==="
./scripts/objective_observability.sh -exp-dir "$EXP_DIR"

echo ""
echo "=== Phase 4: DAD training ==="
./scripts/dad_training.sh -exp-dir "$EXP_DIR"

echo ""
echo "=== Phase 5: evaluation (dad, myopic, fixed, random) ==="
./scripts/evaluation.sh -exp-dir "$EXP_DIR"

echo ""
echo "Done. Experiment → $EXP_DIR"
