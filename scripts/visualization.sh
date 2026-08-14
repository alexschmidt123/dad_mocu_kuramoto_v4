#!/usr/bin/env bash
# Visualization / diagnostics into an experiment folder (plots/, diagnostics/).
#
# Usage:
#   ./scripts/visualization.sh --mode moe-mechanism --config configs/ieee9.yaml --exp-dir experiments/...
#   ./scripts/visualization.sh --mode diagnose-collapse --config configs/ieee9.yaml --exp-dir experiments/...
#   ./scripts/visualization.sh --mode eig-plots --run-prefix ieee9 --out-dir documents/plots
set -euo pipefail
# shellcheck source=../run.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run.sh"

MODE=""
CONFIG=""
EXP_DIR=""
OUT_DIR=""
RUN_PREFIX="ieee9"
EXPERIMENT_TYPE="$EXPERIMENT_TYPE_DEFAULT"
T=""
N_OBS="$DEFAULT_N_OBS"
NOISE_SIGMA="$DEFAULT_NOISE_SIGMA"
SEED=101
ROLLOUTS=128
DEVICE="auto"

usage() {
    echo "Usage: $0 --mode moe-mechanism|diagnose-collapse|eig-plots [options]" >&2
    echo "  moe-mechanism / diagnose-collapse require --config and --exp-dir" >&2
    echo "  eig-plots writes under --out-dir (default: documents/plots); never experiments/_*" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --config|-config|-c) CONFIG="$2"; shift 2 ;;
        --exp-dir|--exp_dir) EXP_DIR="$2"; shift 2 ;;
        --out-dir|--out_dir) OUT_DIR="$2"; shift 2 ;;
        --run-prefix) RUN_PREFIX="$2"; shift 2 ;;
        -T|--T) T="$2"; shift 2 ;;
        --N_obs|--n-obs|--n_obs) N_OBS="$2"; shift 2 ;;
        --noise_sigma|--noise-sigma) NOISE_SIGMA="$2"; shift 2 ;;
        --experiment_type|--experiment-type)
            EXPERIMENT_TYPE="$(validate_experiment_type "$2")" || exit 1
            shift 2
            ;;
        --seed) SEED="$2"; shift 2 ;;
        --rollouts) ROLLOUTS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done

[[ -n "$MODE" ]] || { usage; exit 1; }

case "${MODE,,}" in
    moe-mechanism|moe_mechanism)
        [[ -n "$CONFIG" && -n "$EXP_DIR" ]] || { echo "--config and --exp-dir required" >&2; exit 1; }
        echo "=== visualization moe-mechanism exp-dir=$EXP_DIR ==="
        args=(moe-mechanism --config "$CONFIG" --exp-dir "$EXP_DIR" --experiment-type "$EXPERIMENT_TYPE"
            --N_obs "$N_OBS" --noise_sigma "$NOISE_SIGMA" --seed "$SEED" --rollouts "$ROLLOUTS" --device "$DEVICE")
        [[ -n "$T" ]] && args+=(-T "$T")
        exec python3 -m src.experiment "${args[@]}"
        ;;
    diagnose-collapse|diagnose_collapse)
        [[ -n "$CONFIG" && -n "$EXP_DIR" ]] || { echo "--config and --exp-dir required" >&2; exit 1; }
        echo "=== visualization diagnose-collapse exp-dir=$EXP_DIR ==="
        args=(diagnose-collapse --config "$CONFIG" --exp-dir "$EXP_DIR" --experiment-type "$EXPERIMENT_TYPE"
            --N_obs "$N_OBS" --noise_sigma "$NOISE_SIGMA")
        [[ -n "$T" ]] && args+=(-T "$T")
        exec python3 -m src.experiment "${args[@]}"
        ;;
    eig-plots|eig_plots)
        OUT_DIR="${OUT_DIR:-documents/plots}"
        case "$OUT_DIR" in
            experiments/_*|experiments/_*/*) echo "Refusing invalid out-dir under experiments/_* : $OUT_DIR" >&2; exit 1 ;;
        esac
        mkdir -p "$OUT_DIR"
        echo "=== visualization eig-plots run-prefix=$RUN_PREFIX out-dir=$OUT_DIR ==="
        exec python3 -m src.objectives.eig.cli plot --run-prefix "$RUN_PREFIX" --out-dir "$OUT_DIR"
        ;;
    *)
        echo "Unknown --mode: $MODE" >&2
        usage
        exit 1
        ;;
esac
