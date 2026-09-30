#!/usr/bin/env bash
# Driver for run_code.sh: runs a list of jobs, each one a set of env-var
# overrides for run_code.sh. Edit the JOBS array below to whatever
# combination of databases / GPUs / cardinality types / augmentation
# settings you need, then launch this in a tmux session.
#
# Each job runs run_code.sh once, so a job with CARDINALITY_TYPES="est act dd wj"
# already loops over all four inside that single run_code.sh call.
#
# Jobs run SEQUENTIALLY on purpose, even across different GPU_UUID values --
# this avoids racing on the shared results/*.xlsx files (they're not
# lock-protected, so two run_code.sh processes finishing at the same moment
# can silently clobber each other's row). If you want true parallelism across
# GPUs, open separate tmux sessions and call run_code.sh directly in each
# one instead of using this script -- just don't run two jobs that both
# finish around the same time and write to the same results file.
#
# Usage:
#   tmux new -s all_runs
#   GPU_UUID=GPU-... bash run_all_2.sh      # or fill in GPU_UUID below

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_code.sh"

# UUID of the GPU assigned to you (`nvidia-smi -L`), per the lab server manual.
# Never a GPU index. A single job can still override it with GPU_UUID=... in
# its entry below. Leave empty when running inside a Slurm allocation.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    export GPU_UUID="${GPU_UUID:-GPU-452b1bc7-d479-c54a-050e-e27eabf2d1de}"  # GPU 6
fi

# -----------------------------------------------------------------------
# Edit this list. Each entry is a set of VAR=value overrides for
# run_code.sh (anything not set here falls back to run_code.sh's own
# defaults). A short label (before the first space) is just for the
# summary printout below.
# -----------------------------------------------------------------------
JOBS=(
    "basketball3 TEST_DB=basketball AUGMENT_POOLING=max CARDINALITY_TYPES=act EPOCHS=100 LAMBDA_STRUCT=0.1 AUGMENT_COARSE_LAYERS=1 AUGMENT_REFINEMENT=residual_sum"
    # "cons-sum TEST_DB=consumer AUGMENT_POOLING=sum"
    # "cons-mean TEST_DB=consumer AUGMENT_POOLING=mean"
    # "cons-max TEST_DB=consumer AUGMENT_POOLING=max"
    # "cons-attention TEST_DB=consumer AUGMENT_POOLING=attention"
    # "cons-weighted-mean TEST_DB=consumer AUGMENT_POOLING=weighted_mean"
    # "cons-sum-residual TEST_DB=consumer AUGMENT_REFINEMENT=sum_residual"
)

declare -A exit_codes=()

for job in "${JOBS[@]}"; do
    label="${job%% *}"
    overrides="${job#* }"
    echo "=== $(date) Starting $label ($overrides) ==="
    env $overrides bash "$RUN_SCRIPT"
    exit_codes["$label"]="$?"
    echo "=== $(date) Finished $label (exit ${exit_codes[$label]}) ==="
done

echo
echo "=== SUMMARY ==="
overall_exit_code=0
for job in "${JOBS[@]}"; do
    label="${job%% *}"
    code="${exit_codes[$label]}"
    status="OK"
    if [[ "$code" -ne 0 ]]; then
        status="FAILED (exit $code)"
        overall_exit_code=1
    fi
    echo "  $label: $status"
done

exit "$overall_exit_code"
