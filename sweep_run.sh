#!/bin/bash
# Sweep run.sh over configs and/or horizons (sequential calls to run.sh).
#
# Same T, multiple configs:
#   bash sweep_run.sh --configs ieee5,ieee9 --T 8
#
# Same config, multiple T:
#   bash sweep_run.sh --configs ieee5 --T 4,5,8
#
# Cartesian product (every config × every T):
#   bash sweep_run.sh --configs ieee5,ieee9 --T 4,8
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Shared PATH / PYTHONPATH / validate_experiment_type / defaults
# shellcheck source=run.sh
source "$ROOT/run.sh"

CONFIGS="ieee5,ieee9"
TS="$DEFAULT_STEP_NUMBER"
N_OBS_VALUES="$DEFAULT_N_OBS"
NOISE_SIGMAS="$DEFAULT_NOISE_SIGMA"
SEEDS="101"
FORCE=""
METHOD=""
SMOKE=""
BANK_STRUCTURE_AUDIT=""
EXPERIMENT_TYPE="$EXPERIMENT_TYPE_DEFAULT"

usage() {
    echo "Usage: $0 [--configs ieee5,ieee9] [--T 5|4,8] [--N_obs 0|120] [--noise_sigma 0.005|0.001,0.005] [--experiment_type objective_based|eig_based] [--method <method>] [--force] [--bank-structure-audit] [--smoke]" >&2
    echo "" >&2
    echo "  --configs   comma-separated config stems or paths under configs/ (default: ieee5,ieee9)" >&2
    echo "  --systems   alias for --configs" >&2
    echo "  --config    alias for --configs (also accepts full yaml paths)" >&2
    echo "  --T         one horizon or comma-separated list (default: ${DEFAULT_STEP_NUMBER})" >&2
    echo "  --N_obs     one count or comma-separated list (default: ${DEFAULT_N_OBS})" >&2
    echo "  --noise_sigma one std or comma-separated list (default: ${DEFAULT_NOISE_SIGMA})" >&2
    echo "  --force     regenerate physical banks" >&2
    echo "  --bank-structure-audit  forward to run.sh (Myopic-trap gate per cell)" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  $0 --configs ieee5,ieee9 --T 8          # same T, multiple yaml" >&2
    echo "  $0 --configs ieee5 --T 4,5,8            # same yaml, multiple T" >&2
    echo "  $0 --configs ieee5,ieee9 --T 4,8        # product of both" >&2
}

resolve_cfg() {
    local item="$1"
    if [[ -f "$item" ]]; then
        echo "$item"
        return 0
    fi
    if [[ -f "configs/${item}" ]]; then
        echo "configs/${item}"
        return 0
    fi
    if [[ -f "configs/${item}.yaml" ]]; then
        echo "configs/${item}.yaml"
        return 0
    fi
    echo "Missing config: $item (tried path, configs/${item}, configs/${item}.yaml)" >&2
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs|--systems|--config|-config|-c) CONFIGS="$2"; shift 2 ;;
        -T|--T|--step-number|--step_number) TS="$2"; shift 2 ;;
        --N_obs|--n-obs|--n_obs) N_OBS_VALUES="$2"; shift 2 ;;
        --noise_sigma|--noise-sigma) NOISE_SIGMAS="$2"; shift 2 ;;
        --seeds|--seed) SEEDS="$2"; shift 2 ;;
        --method|-method|-m) METHOD="$2"; shift 2 ;;
        --experiment_type|--experiment-type)
            EXPERIMENT_TYPE="$(validate_experiment_type "$2")" || exit 1
            shift 2
            ;;
        --force) FORCE="--force"; shift ;;
        --bank-structure-audit|--require-myopic-trap)
            BANK_STRUCTURE_AUDIT="--bank-structure-audit"
            shift
            ;;
        --smoke) SMOKE="--smoke"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done

CFG_ARR=()
IFS=',' read -r -a _raw_cfgs <<< "$CONFIGS"
for item in "${_raw_cfgs[@]}"; do
    item="$(echo "$item" | xargs)"
    [[ -n "$item" ]] || continue
    CFG_ARR+=("$(resolve_cfg "$item")")
done
[[ ${#CFG_ARR[@]} -gt 0 ]] || { echo "No configs given" >&2; usage; exit 1; }

T_ARR=()
IFS=',' read -r -a _raw_ts <<< "$TS"
for t in "${_raw_ts[@]}"; do
    t="$(echo "$t" | xargs)"
    [[ -n "$t" ]] || continue
    [[ "$t" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid --T value: $t" >&2; exit 1; }
    T_ARR+=("$t")
done
[[ ${#T_ARR[@]} -gt 0 ]] || { echo "No --T values given" >&2; usage; exit 1; }

NOBS_ARR=()
IFS=',' read -r -a _raw_nobs <<< "$N_OBS_VALUES"
for n_obs in "${_raw_nobs[@]}"; do
    n_obs="$(echo "$n_obs" | xargs)"
    [[ -n "$n_obs" ]] || continue
    [[ "$n_obs" =~ ^[0-9]+$ ]] || { echo "Invalid --N_obs value: $n_obs" >&2; exit 1; }
    NOBS_ARR+=("$n_obs")
done
[[ ${#NOBS_ARR[@]} -gt 0 ]] || { echo "No --N_obs values given" >&2; usage; exit 1; }

SIGMA_ARR=()
IFS=',' read -r -a _raw_sigmas <<< "$NOISE_SIGMAS"
for sigma in "${_raw_sigmas[@]}"; do
    sigma="$(echo "$sigma" | xargs)"
    [[ -n "$sigma" ]] || continue
    python3 -c 'import sys; x=float(sys.argv[1]); assert x > 0' "$sigma" 2>/dev/null \
        || { echo "Invalid --noise_sigma value: $sigma" >&2; exit 1; }
    SIGMA_ARR+=("$sigma")
done
[[ ${#SIGMA_ARR[@]} -gt 0 ]] || { echo "No --noise_sigma values given" >&2; usage; exit 1; }

SEED_ARR=()
IFS=',' read -r -a _raw_seeds <<< "$SEEDS"
for seed in "${_raw_seeds[@]}"; do
    seed="$(echo "$seed" | xargs)"
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid --seed value: $seed" >&2; exit 1; }
    SEED_ARR+=("$seed")
done

echo "=== sweep_run.sh configs=${CFG_ARR[*]} T=${T_ARR[*]} N_obs=${NOBS_ARR[*]} noise_sigma=${SIGMA_ARR[*]} type=$EXPERIMENT_TYPE ==="
for cfg in "${CFG_ARR[@]}"; do
    stem="$(basename "$cfg")"
    stem="${stem%.yml}"
    stem="${stem%.yaml}"
    for T in "${T_ARR[@]}"; do
      for N_OBS in "${NOBS_ARR[@]}"; do
       for NOISE_SIGMA in "${SIGMA_ARR[@]}"; do
        for SEED in "${SEED_ARR[@]}"; do
        extra=()
        if [[ -n "$FORCE" ]]; then
            extra=(--force)
        elif [[ ! -d "data/${stem}" ]]; then
            extra=(--force)
        fi
        echo "--- $cfg --T $T --N_obs $N_OBS --noise_sigma $NOISE_SIGMA ${extra[*]:-} ---"
        ARGS=(--config "$cfg" --experiment_type "$EXPERIMENT_TYPE" -T "$T" --N_obs "$N_OBS" --noise_sigma "$NOISE_SIGMA" --seed "$SEED")
        [[ -n "$METHOD" ]] && ARGS+=(--method "$METHOD")
        [[ -n "$BANK_STRUCTURE_AUDIT" ]] && ARGS+=(--bank-structure-audit)
        # shellcheck disable=SC2086
        bash run.sh "${ARGS[@]}" "${extra[@]+"${extra[@]}"}" ${SMOKE}
        done
       done
      done
    done
done
echo "=== sweep_run.sh complete ==="
