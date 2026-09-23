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
# dkegpuserver2 weekend sweep, GPU 2 only (GPUs 0/1 are reserved by another
# lab member). 7 configs x 3 held-out DBs, run DB-by-DB (fhnk, employee,
# financial) so each finished DB is a complete comparison.
#   - act-*: model-wide activation sweep, attention pooling held fixed.
#     act-leakyrelu is also the reference row for the hybrid pooling runs.
#   - pool-*: the three hybrid pooling modes, LeakyReLU held fixed.
# All share coarse_layers=1 / lambda_struct=0.1 / est / 100 epochs so they
# land in one comparable cohort (see understanding/progress.md sec. 8).
# -----------------------------------------------------------------------
COMMON="DEVICE=2 CARDINALITY_TYPE=est AUGMENT=True EPOCHS=100 AUGMENT_COARSE_LAYERS=1 LAMBDA_STRUCT=0.1"

JOBS=()
for db in fhnk employee financial; do
    JOBS+=(
        "$db-act-leakyrelu TEST_DB=$db $COMMON AUGMENT_POOLING=attention ACTIVATION_CLASS_NAME=LeakyReLU"
        "$db-act-relu TEST_DB=$db $COMMON AUGMENT_POOLING=attention ACTIVATION_CLASS_NAME=ReLU"
        "$db-act-selu TEST_DB=$db $COMMON AUGMENT_POOLING=attention ACTIVATION_CLASS_NAME=SELU"
        "$db-act-celu TEST_DB=$db $COMMON AUGMENT_POOLING=attention ACTIVATION_CLASS_NAME=CELU"
        "$db-pool-hybrid_attn_max TEST_DB=$db $COMMON AUGMENT_POOLING=hybrid_attn_max ACTIVATION_CLASS_NAME=LeakyReLU"
        "$db-pool-hybrid_max_wmean TEST_DB=$db $COMMON AUGMENT_POOLING=hybrid_max_wmean ACTIVATION_CLASS_NAME=LeakyReLU"
        "$db-pool-hybrid_max_wmean_gated TEST_DB=$db $COMMON AUGMENT_POOLING=hybrid_max_wmean_gated ACTIVATION_CLASS_NAME=LeakyReLU"
    )
done

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
