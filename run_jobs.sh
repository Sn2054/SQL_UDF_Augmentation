#!/usr/bin/env bash
# Driver for run_code.sh: runs a list of jobs, each one a set of env-var
# overrides for run_code.sh. Edit the JOBS array below to whatever
# combination of databases / GPUs / cardinality types / augmentation
# settings you need, then launch this in a tmux session.
#
# This mirrors Ishana's run_all.sh pattern (GRACEFUL_Augmentation repo): a
# single JOBS array is the one place that documents what was run, so editing
# it and committing gives a readable git history of "what ran and why" --
# use a descriptive commit message per change to the list (e.g. "runs: fhnk
# pooling sweep pt2") rather than committing silently.
#
# Jobs run SEQUENTIALLY on purpose, even across different DEVICE values --
# this avoids racing on the shared results/*.xlsx files (they're not
# lock-protected, so two run_code.sh processes finishing at the same moment
# can silently clobber each other's row). If you want true parallelism across
# GPUs, use this script in one tmux pane and run_jobs_copy.sh (-> run_code_copy.sh)
# in another instead of running two jobs from the same list in parallel.
#
# Usage:
#   tmux new -s jobs
#   bash run_jobs.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_code.sh"

# -----------------------------------------------------------------------
# Edit this list. Each entry is a set of VAR=value overrides for
# run_code.sh (anything not set here falls back to run_code.sh's own
# defaults). A short label (before the first space) is just for the
# summary printout below.
# -----------------------------------------------------------------------
# EP multi-query pooling sweep on the new supernodes (2026-10-05): Ishana's region fix + SEQ regions +
# CFG coarse edges on, LeakyReLU, est, 100 epochs, coarse_layers=1, lambda_struct=0.1.
# Database by database (fhnk, employee, financial); per database: attention baseline, then
# multi_query_attention M=1, 8, 4, 16. fhnk's baseline already ran as job 1 of the first launch.
# -----------------------------------------------------------------------
COMMON="GPU_UUID=GPU-042f3a36-fa80-8994-33b1-37c79834d513 CARDINALITY_TYPE=est AUGMENT=True EPOCHS=100 AUGMENT_COARSE_LAYERS=1 LAMBDA_STRUCT=0.1 ACTIVATION_CLASS_NAME=LeakyReLU AUGMENT_SEQ_REGIONS=True AUGMENT_CFG_COARSE_EDGES=True"

JOBS=()
for db in fhnk employee financial; do
    [[ "$db" != fhnk ]] && JOBS+=("$db-base-attention TEST_DB=$db $COMMON AUGMENT_POOLING=attention")
    for m in 1 8 4 16; do
        JOBS+=("$db-mq$m TEST_DB=$db $COMMON AUGMENT_POOLING=multi_query_attention AUGMENT_MQ_QUERIES=$m")
    done
done

while pgrep -u "$USER" -f "bash .*run_code.sh" >/dev/null; do sleep 300; done

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
