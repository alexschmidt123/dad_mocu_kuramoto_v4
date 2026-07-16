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

# Dedicated controlled T=3/T=4 paths (frozen margin 0.55, full four-method reports).
if [[ "${T:-}" == "3" && -z "$EXP_DIR" ]]; then
    echo "=== IEEE5 T=3 controlled experiment (frozen margin 0.55) ==="
    exec python3 -m src.cli run-ieee5-t3
fi
if [[ "${T:-}" == "4" && -z "$EXP_DIR" ]]; then
    echo "=== IEEE5 T=4 controlled experiment (frozen margin 0.55) ==="
    exec python3 -m src.cli run-ieee5-t4
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
