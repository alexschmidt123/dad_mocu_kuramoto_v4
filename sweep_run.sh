#!/bin/bash
# Sweep run.sh over configs and/or horizons (sequential calls to run.sh).
#
# Same T, multiple configs:
#   bash sweep_run.sh --configs ieee9,ieee14 --T 8
#
# Same config, multiple T:
#   bash sweep_run.sh --configs ieee9 --T 4,5,8
#
# Cartesian product (every config × every T × seed):
#   bash sweep_run.sh --configs ieee9 --T 3,4,5 --seed 101,202,303
#
# After a T sweep (at least two --T values), writes one stamped plots folder
# per (config, type, N_obs, sigma) group:
#   --T 3,4,5 → experiments/MMDDYYYY_HHMMSS_plots_ieee9_EIG_T3-5_Nobs10_sigma0p005
# Single-T sweeps do not write a plots folder. date_time is this sweep's start
# time. Each plots folder has exactly five files:
#   metric.md, time.md, metric_vs_T.png, time_vs_T.md, meta.json
#
# Rebuild plots from existing T-sweep result dirs (no train/eval):
#   bash sweep_run.sh --configs ieee9 --experiment_type eig_based \
#     --T 3,4,5 --N_obs 10 --noise_sigma 0.005 --seed 101,202,303 --plots-only
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Shared PATH / PYTHONPATH / validate_experiment_type / defaults
# shellcheck source=run.sh
source "$ROOT/run.sh"

CONFIGS="ieee9,ieee14"
TS="$DEFAULT_STEP_NUMBER"
N_OBS_VALUES="$DEFAULT_N_OBS"
NOISE_SIGMAS="$DEFAULT_NOISE_SIGMA"
SEEDS="$DEFAULT_SEEDS"
FORCE=""
METHOD=""
SMOKE=""
BANK_STRUCTURE_AUDIT=""
EXPERIMENT_TYPE="$EXPERIMENT_TYPE_DEFAULT"
PLOTS_ONLY=0

usage() {
    echo "Usage: $0 [--configs ieee9,ieee14] [--T 5|4,8] [--N_obs 0|120] [--noise_sigma 0.005|0.001,0.005] [--seed 101,202,303] [--experiment_type objective_based|eig_based] [--method <methods>] [--force] [--bank-structure-audit] [--smoke] [--plots-only]" >&2
    echo "" >&2
    echo "  --method    optional comma-separated evaluate/train subset (default: yaml methods)" >&2
    echo "              e.g. --method dad,rl_sboed,myopic  (no MoE; skips MoE training)" >&2
    echo "  --seed      one training seed or comma-separated list (default: ${DEFAULT_SEEDS})" >&2
    echo "  --seeds     alias for --seed" >&2
    echo "  --configs   comma-separated config stems or paths under configs/ (default: ieee9,ieee14)" >&2
    echo "              IEEE-5 catalogs are retired; use ieee9, ieee14, or sir_ode" >&2
    echo "  --systems   alias for --configs" >&2
    echo "  --config    alias for --configs (also accepts full yaml paths)" >&2
    echo "  --T         one horizon or comma-separated list (default: ${DEFAULT_STEP_NUMBER})" >&2
    echo "  --N_obs     one count or comma-separated list (default: ${DEFAULT_N_OBS})" >&2
    echo "  --noise_sigma one std or comma-separated list (default: ${DEFAULT_NOISE_SIGMA})" >&2
    echo "  --force     regenerate physical banks" >&2
    echo "  --bank-structure-audit  forward to run.sh (Myopic-trap gate per cell)" >&2
    echo "  --plots-only  write stamped plots from existing matching result dirs (no train/eval)" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  $0 --configs ieee9,ieee14 --T 8         # same T, multiple yaml" >&2
    echo "  $0 --configs ieee9 --T 4,5,8            # same yaml, multiple T" >&2
    echo "  $0 --configs ieee9,ieee14 --T 4,8       # product of both" >&2
    echo "  $0 --configs sir_ode --T 4,5 --experiment_type eig_based" >&2
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
        --plots-only) PLOTS_ONLY=1; shift ;;
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
    [[ -n "$seed" ]] || continue
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid --seed value: $seed" >&2; exit 1; }
    SEED_ARR+=("$seed")
done
[[ ${#SEED_ARR[@]} -gt 0 ]] || { echo "No --seed values given" >&2; usage; exit 1; }

SWEEP_STAMP="$(date +%m%d%Y_%H%M%S)"
STEM_ARR=()
for cfg in "${CFG_ARR[@]}"; do
    STEM_ARR+=("$(basename "$cfg" .yaml)")
done

write_sweep_plots() {
    local plot_args=(
        python3 -m src.results.plots
        --stamp "$SWEEP_STAMP"
        --experiment-type "$EXPERIMENT_TYPE"
    )
    if [[ -n "$METHOD" ]]; then
        plot_args+=(--methods "$METHOD")
    fi
    # Folder token is this sweep's --T range (T3-5). Single-T sweeps never plot.
    plot_args+=(--T "$TS" --N_obs "$N_OBS_VALUES" --noise_sigma "$NOISE_SIGMAS")
    if [[ "$1" == "discover" ]]; then
        plot_args+=(--discover --seeds "$SEEDS")
        local stem
        for stem in "${STEM_ARR[@]}"; do
            plot_args+=(--config-stem "$stem")
        done
    else
        local d
        for d in "${CELL_DIRS[@]}"; do
            plot_args+=(--exp-dir "$d")
        done
    fi
    echo "=== sweep_run.sh plots stamp=$SWEEP_STAMP ==="
    "${plot_args[@]}"
}

maybe_write_sweep_plots() {
    if [[ ${#T_ARR[@]} -lt 2 ]]; then
        echo "=== sweep_run.sh: skip plots (need a T sweep of at least two --T values; got T=${T_ARR[*]}) ==="
        return 0
    fi
    write_sweep_plots "$1"
}

echo "=== sweep_run.sh stamp=$SWEEP_STAMP configs=${CFG_ARR[*]} T=${T_ARR[*]} N_obs=${NOBS_ARR[*]} noise_sigma=${SIGMA_ARR[*]} seed=${SEED_ARR[*]} type=$EXPERIMENT_TYPE ==="

if [[ "$PLOTS_ONLY" -eq 1 ]]; then
    maybe_write_sweep_plots discover
    echo "=== sweep_run.sh plots-only complete ==="
    exit 0
fi

CELL_DIRS=()
for cfg in "${CFG_ARR[@]}"; do
    for T in "${T_ARR[@]}"; do
      for N_OBS in "${NOBS_ARR[@]}"; do
       for NOISE_SIGMA in "${SIGMA_ARR[@]}"; do
        for SEED in "${SEED_ARR[@]}"; do
        extra=()
        # Only forward explicit --force. Do not infer a missing bank from
        # data/<yaml-stem>: IEEE-9 lives at data/ieee9_duration_bus, and
        # generate-data refuses --force on existing physical banks.
        if [[ -n "$FORCE" ]]; then
            extra=(--force)
        fi
        echo "--- $cfg --T $T --N_obs $N_OBS --noise_sigma $NOISE_SIGMA --seed $SEED ${extra[*]:-} ---"
        ARGS=(--config "$cfg" --experiment_type "$EXPERIMENT_TYPE" -T "$T" --N_obs "$N_OBS" --noise_sigma "$NOISE_SIGMA" --seed "$SEED")
        [[ -n "$METHOD" ]] && ARGS+=(--method "$METHOD")
        [[ -n "$BANK_STRUCTURE_AUDIT" ]] && ARGS+=(--bank-structure-audit)
        cell_log="$(mktemp)"
        set +e
        # shellcheck disable=SC2086
        bash run.sh "${ARGS[@]}" "${extra[@]+"${extra[@]}"}" ${SMOKE} | tee "$cell_log"
        rc=${PIPESTATUS[0]}
        set -e
        cell="$(grep '^EXP_DIR=' "$cell_log" | tail -n 1 | sed 's/^EXP_DIR=//' | tr -d '\r' | sed 's/[[:space:]]*$//')"
        rm -f "$cell_log"
        if [[ $rc -ne 0 ]]; then
            echo "sweep cell failed (exit $rc): $cfg T=$T N_obs=$N_OBS noise_sigma=$NOISE_SIGMA seed=$SEED" >&2
            exit "$rc"
        fi
        if [[ -z "$cell" ]]; then
            echo "sweep cell succeeded but EXP_DIR= was not printed: $cfg T=$T seed=$SEED" >&2
            exit 1
        fi
        CELL_DIRS+=("$cell")
        done
       done
      done
    done
done

if [[ ${#CELL_DIRS[@]} -gt 0 ]]; then
    maybe_write_sweep_plots dirs
else
    echo "=== sweep_run.sh: no result dirs to plot ===" >&2
fi
echo "=== sweep_run.sh complete ==="
