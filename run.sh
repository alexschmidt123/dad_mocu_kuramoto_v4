#!/bin/bash
# Full experiment: call core scripts in order.
#
#   bash run.sh --config configs/ieee5.yaml
#   bash run.sh --config configs/ieee5.yaml --T 8
#   bash run.sh --config configs/ieee5.yaml --experiment_type eig_based
#   bash run.sh --config configs/sir_ode.yaml
#   bash run.sh --config configs/ieee9.yaml --method dad --force
#
# Result folder (allocated once, reused for all steps):
#   experiments/date_time_configname_Uctrl|EIG_Tnum_NobsN_sigmaX
# Full terminal history is saved as <result_folder>/logs/run_log.log
#
# Nested scripts may ``source`` this file for shared env/helpers only
# (when sourced, the main pipeline below does not run).

# --- shared env / helpers (also used when this file is sourced) ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# Prefer the project conda env when present (activated env still wins if already first on PATH).
if [[ -x "/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python3" ]]; then
    export PATH="/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin:${PATH}"
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EXPERIMENT_TYPE_DEFAULT="objective_based"
DEFAULT_STEP_NUMBER=5
DEFAULT_N_OBS=0
DEFAULT_NOISE_SIGMA=0.005

validate_experiment_type() {
    local t="${1,,}"
    t="${t//-/_}"
    case "$t" in
        objective_based|eig_based)
            echo "$t"
            return 0
            ;;
        *)
            echo "Invalid --experiment_type: $1 (allowed: objective_based|eig_based)" >&2
            return 1
            ;;
    esac
}

# Tee all stdout/stderr to <result_dir>/logs/run_log.log (idempotent for nested scripts).
start_run_logging() {
    if [[ -n "${RUN_LOG_ACTIVE:-}" ]]; then
        return 0
    fi
    local result_dir="${1:-}"
    if [[ -z "$result_dir" ]]; then
        echo "start_run_logging: result directory required" >&2
        return 1
    fi
    result_dir="$(printf '%s' "$result_dir" | tr -d '\r' | sed 's/[[:space:]]*$//')"
    mkdir -p "${result_dir%/}/logs"
    export RUN_LOG_FILE="${result_dir%/}/logs/run_log.log"
    : > "${RUN_LOG_FILE}"
    export RUN_LOG_ACTIVE=1
    exec > >(tee -a "${RUN_LOG_FILE}") 2>&1
    echo "Log file: ${RUN_LOG_FILE}"
}

# Sourced by sweep_run.sh / scripts/*.sh — setup only.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
fi

set -euo pipefail

CONFIG=""
METHOD=""
SMOKE=""
FORCE=""
BANK_STRUCTURE_AUDIT=""
EXPERIMENT_TYPE="$EXPERIMENT_TYPE_DEFAULT"
EXPERIMENT_TYPE_SET=0
EXP_DIR=""
# Optional probe horizon (default 5). Do not encode T in config filenames.
T="$DEFAULT_STEP_NUMBER"
N_OBS="$DEFAULT_N_OBS"
N_OBS_SET=0
NOISE_SIGMA="$DEFAULT_NOISE_SIGMA"
NOISE_SIGMA_SET=0
SEED=101

usage() {
    echo "Usage: $0 --config <config.yaml> [--T <horizon>] [--N_obs <count>] [--noise_sigma <sigma>] [--seed <int>] [--experiment_type objective_based|eig_based] [--method <method>] [--exp-dir <path>] [--force] [--bank-structure-audit] [--smoke]" >&2
    echo "" >&2
    echo "  --T       probe horizon (default: ${DEFAULT_STEP_NUMBER})" >&2
    echo "  --N_obs   IEEE trajectory samples; 0 = scalar max-ROCOF (ignored for SIR ODE)" >&2
    echo "  --noise_sigma observation noise std (IEEE default: ${DEFAULT_NOISE_SIGMA}; SIR uses YAML)" >&2
    echo "  --force   regenerate data bank if one already exists (usually unnecessary)" >&2
    echo "  --bank-structure-audit  run Myopic-trap / redundancy audit after data gen; fail if not ready" >&2
    echo "Result folders: date_time_configname_Uctrl|EIG_Tnum_NobsN_sigmaX" >&2
    echo "Examples:" >&2
    echo "  bash run.sh --config configs/ieee5.yaml" >&2
    echo "  bash run.sh --config configs/ieee5.yaml --T 8" >&2
    echo "  bash run.sh --config configs/sir_ode.yaml" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|-config|-c) CONFIG="$2"; shift 2 ;;
        --method|-method|-m) METHOD="$2"; shift 2 ;;
        -T|--T|--step-number|--step_number) T="$2"; shift 2 ;;
        --N_obs|--n-obs|--n_obs) N_OBS="$2"; N_OBS_SET=1; shift 2 ;;
        --noise_sigma|--noise-sigma) NOISE_SIGMA="$2"; NOISE_SIGMA_SET=1; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --experiment_type|--experiment-type)
            EXPERIMENT_TYPE="$(validate_experiment_type "$2")" || exit 1
            EXPERIMENT_TYPE_SET=1
            shift 2
            ;;
        --exp-dir|--exp_dir) EXP_DIR="$2"; shift 2 ;;
        --force) FORCE="--force"; shift ;;
        --bank-structure-audit|--require-myopic-trap)
            BANK_STRUCTURE_AUDIT=1
            shift
            ;;
        --smoke) SMOKE="--smoke"; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            usage
            exit 1
            ;;
    esac
done

[[ -n "$CONFIG" ]] || { usage; exit 1; }
[[ "$T" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid --T: $T (positive integer required)" >&2; exit 1; }
[[ "$N_OBS" =~ ^[0-9]+$ ]] || { echo "Invalid --N_obs: $N_OBS (non-negative integer required)" >&2; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Invalid --seed: $SEED (non-negative integer required)" >&2; exit 1; }

# SIR ODE: EIG-only; design=time, observe infected count. Ignore IEEE N_obs.
IS_SIR_FLAG="$(python3 -c '
import sys, yaml
from pathlib import Path
raw = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
name = str((raw.get("system") or {}).get("name") or "").lower().replace("-", "_")
print("1" if name in {"sir_ode", "sir"} or raw.get("sir_ode") or raw.get("sir") else "0")
' "$CONFIG")"
if [[ "$IS_SIR_FLAG" == "1" ]]; then
    if [[ "$EXPERIMENT_TYPE_SET" -eq 1 && "$EXPERIMENT_TYPE" != "eig_based" ]]; then
        echo "SIR ODE supports eig_based only (got --experiment_type=$EXPERIMENT_TYPE)" >&2
        exit 1
    fi
    EXPERIMENT_TYPE="eig_based"
    if [[ "$N_OBS_SET" -eq 1 ]]; then
        echo "[run.sh] ignoring --N_obs for SIR ODE (scalar infected-count observation)" >&2
    fi
    N_OBS=1
    if [[ "$NOISE_SIGMA_SET" -eq 0 ]]; then
        NOISE_SIGMA="$(python3 -c '
import sys, yaml
from pathlib import Path
raw = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
obs = dict(raw.get("observation") or {})
sir = dict(raw.get("sir_ode") or raw.get("sir") or {})
print(float(obs.get("noise_sigma", sir.get("likelihood_sigma", 1.0))))
' "$CONFIG")"
    fi
    echo "[run.sh] SIR ODE mode: experiment_type=eig_based, chronological times, y=I(t) count"
fi

python3 -c 'import sys; x=float(sys.argv[1]); assert x > 0' "$NOISE_SIGMA" 2>/dev/null \
    || { echo "Invalid --noise_sigma: $NOISE_SIGMA (positive float required)" >&2; exit 1; }

need_dad=1
need_rl=1
need_moe=1
if [[ -n "$METHOD" ]]; then
    case "${METHOD,,}" in
        dad) need_dad=1; need_rl=0; need_moe=0 ;;
        rl_sboed|rl-sboed)
            need_dad=0
            need_rl=1
            need_moe=0
            ;;
        moe_sboed|moe-sboed)
            need_dad=1; need_rl=0; need_moe=1
            ;;
        matched_dense|matched-dense)
            need_dad=0; need_rl=0; need_moe=0; need_matched=1
            ;;
        step_dad|step-dad)
            # Step-DAD refines a trained DAD policy online.
            need_dad=1; need_rl=0; need_moe=0
            ;;
        myopic|fixed|random) need_dad=0; need_rl=0; need_moe=0 ;;
        *)
            echo "Unknown method: $METHOD (allowed: dad|rl_sboed|moe_sboed|matched_dense|step_dad|myopic|fixed|random)" >&2
            exit 1
            ;;
    esac
fi

need_matched=${need_matched:-0}

TYPE_ARGS=(--experiment_type "$EXPERIMENT_TYPE")
T_ARGS=(-T "$T")
OBS_ARGS=(--N_obs "$N_OBS" --noise_sigma "$NOISE_SIGMA")
if [[ -z "$EXP_DIR" ]]; then
    EXP_DIR="$(python3 -m src.experiment allocate-dir --config "$CONFIG" "${TYPE_ARGS[@]}" "${T_ARGS[@]}" "${OBS_ARGS[@]}")"
fi
# Keep only the path line (tolerate any stray stderr noise).
EXP_DIR="$(printf '%s' "$EXP_DIR" | tr -d '\r' | tail -n 1 | sed 's/[[:space:]]*$//')"
EXP_ARGS=(--exp-dir "$EXP_DIR")

# Log into the result folder once the path is known (nested scripts reuse this tee).
start_run_logging "$EXP_DIR"

echo "=== run.sh (config=$CONFIG type=$EXPERIMENT_TYPE T=$T N_obs=$N_OBS noise_sigma=$NOISE_SIGMA method=${METHOD:-ALL}) ==="
echo "Result folder: $EXP_DIR"

./scripts/data_generation.sh --config "$CONFIG" "${TYPE_ARGS[@]}" "${T_ARGS[@]}" "${OBS_ARGS[@]}" "${EXP_ARGS[@]}" ${FORCE} ${SMOKE}

if [[ -n "$BANK_STRUCTURE_AUDIT" ]]; then
    echo "=== bank-structure-audit (Myopic trap / adaptive-room gate) ==="
    python3 -m src.experiment bank-structure-audit --config "$CONFIG" \
        "${TYPE_ARGS[@]}" "${T_ARGS[@]}" "${OBS_ARGS[@]}" "${EXP_ARGS[@]}"
    audit_rc=$?
    echo "=== bank-structure-audit done (skipping train/eval; omit flag for full runs) ==="
    echo "Done → $EXP_DIR"
    echo "EXP_DIR=$EXP_DIR"
    exit "$audit_rc"
fi

if [[ "$need_dad" -eq 1 ]]; then
    ./scripts/training.sh --config "$CONFIG" --method dad --seed "$SEED" \
        "${TYPE_ARGS[@]}" "${T_ARGS[@]}" "${OBS_ARGS[@]}" "${EXP_ARGS[@]}" ${SMOKE}
fi
if [[ "$need_rl" -eq 1 ]]; then
    ./scripts/training.sh --config "$CONFIG" --method rl_sboed --seed "$SEED" \
        "${TYPE_ARGS[@]}" "${T_ARGS[@]}" "${OBS_ARGS[@]}" "${EXP_ARGS[@]}" ${SMOKE}
fi
if [[ "$need_moe" -eq 1 ]]; then
    ./scripts/training.sh --config "$CONFIG" --method moe_sboed --seed "$SEED" \
        "${TYPE_ARGS[@]}" "${T_ARGS[@]}" "${OBS_ARGS[@]}" "${EXP_ARGS[@]}" ${SMOKE}
fi
if [[ "$need_matched" -eq 1 ]]; then
    ./scripts/training.sh --config "$CONFIG" --method matched_dense --seed "$SEED" \
        "${TYPE_ARGS[@]}" "${T_ARGS[@]}" "${OBS_ARGS[@]}" "${EXP_ARGS[@]}" ${SMOKE}
fi

if [[ -n "$METHOD" ]]; then
    ./scripts/evaluation.sh --config "$CONFIG" "${TYPE_ARGS[@]}" "${T_ARGS[@]}" "${OBS_ARGS[@]}" "${EXP_ARGS[@]}" --method "$METHOD" ${SMOKE}
else
    ./scripts/evaluation.sh --config "$CONFIG" "${TYPE_ARGS[@]}" "${T_ARGS[@]}" "${OBS_ARGS[@]}" "${EXP_ARGS[@]}" ${SMOKE}
fi

echo ""
printf "Done config=%s type=%s T=%s.\n" "$CONFIG" "$EXPERIMENT_TYPE" "$T"
echo "EXP_DIR=$EXP_DIR"
echo "LOG_FILE=${RUN_LOG_FILE:-}"
