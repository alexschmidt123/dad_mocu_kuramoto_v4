#!/bin/bash
# Shell syntax / CLI smoke checks for production entrypoints.
# Run: bash test/check_shell_scripts.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCRIPTS=(
    run.sh
    sweep_run.sh
    scripts/_env.sh
    scripts/data_generation.sh
    scripts/dad_training.sh
    scripts/rl_sboed_training.sh
    scripts/evaluation.sh
)

echo "=== bash -n (syntax) ==="
for s in "${SCRIPTS[@]}"; do
    [[ -f "$s" ]] || { echo "missing: $s" >&2; exit 1; }
    bash -n "$s"
    echo "  ok $s"
done

echo "=== --help smoke ==="
./run.sh --help >/dev/null
./sweep_run.sh --help >/dev/null
./scripts/data_generation.sh --help >/dev/null
./scripts/dad_training.sh --help >/dev/null
./scripts/rl_sboed_training.sh --help >/dev/null
./scripts/evaluation.sh --help >/dev/null

echo "=== experiment_type validation ==="
if ./run.sh --config configs/ieee5.yaml --experiment_type not_a_type >/dev/null 2>&1; then
    echo "expected invalid --experiment_type to fail" >&2
    exit 1
fi

echo "All shell checks passed."
