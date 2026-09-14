#!/usr/bin/env bash
# Driver for run_code_copy.sh: runs a list of jobs, each one a set of env-var
# overrides for run_code_copy.sh. Edit the JOBS array below to whatever
# combination of databases / GPUs / cardinality types / augmentation
# settings you need, then launch this in a tmux session.
#
# This mirrors Ishana's run_all.sh pattern (GRACEFUL_Augmentation repo): a
# single JOBS array is the one place that documents what was run, so editing
# it and committing gives a readable git history of "what ran and why" --
# use a descriptive commit message per change to the list (e.g. "runs: fhnk
# pooling sweep pt2") rather than committing silently.
#
# Jobs run SEQUENTIALLY on purpose -- this avoids racing on the shared
# results/*.xlsx files (they're not lock-protected, so two run_code*.sh
# processes finishing at the same moment can silently clobber each other's
# row). If you want true parallelism across GPUs, use this script in one
# tmux pane and run_jobs.sh (-> run_code.sh) in another instead of running
# two jobs from the same list in parallel.
#
# Usage:
#   tmux new -s jobs_copy
#   bash run_jobs_copy.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_code_copy.sh"

# -----------------------------------------------------------------------
# Edit this list. Each entry is a set of VAR=value overrides for
# run_code_copy.sh (anything not set here falls back to run_code_copy.sh's
# own defaults, incl. DEVICE=1). A short label (before the first space) is
# just for the summary printout below.
#
# All jobs below hold coarse_layers=1, lambda_struct=0.1 fixed so they land
# in the same comparable cohort as the existing attention (push q50=1.546)
# and max (push q50=1.414) results already in augmented_cost_estimation.xlsx.
# -----------------------------------------------------------------------
JOBS=(
    "hybrid-attn-max TEST_DB=fhnk CARDINALITY_TYPE=est AUGMENT=True EPOCHS=100 AUGMENT_POOLING=hybrid_attn_max AUGMENT_COARSE_LAYERS=1 LAMBDA_STRUCT=0.1"
    "hybrid-max-wmean TEST_DB=fhnk CARDINALITY_TYPE=est AUGMENT=True EPOCHS=100 AUGMENT_POOLING=hybrid_max_wmean AUGMENT_COARSE_LAYERS=1 LAMBDA_STRUCT=0.1"
    "hybrid-max-wmean-gated TEST_DB=fhnk CARDINALITY_TYPE=est AUGMENT=True EPOCHS=100 AUGMENT_POOLING=hybrid_max_wmean_gated AUGMENT_COARSE_LAYERS=1 LAMBDA_STRUCT=0.1"
    "final-act-relu TEST_DB=fhnk CARDINALITY_TYPE=est AUGMENT=True EPOCHS=100 AUGMENT_POOLING=attention AUGMENT_COARSE_LAYERS=1 LAMBDA_STRUCT=0.1 FINAL_ACTIVATION_CLASS_NAME=ReLU"
    "final-act-celu TEST_DB=fhnk CARDINALITY_TYPE=est AUGMENT=True EPOCHS=100 AUGMENT_POOLING=attention AUGMENT_COARSE_LAYERS=1 LAMBDA_STRUCT=0.1 FINAL_ACTIVATION_CLASS_NAME=CELU"
    "final-act-selu TEST_DB=fhnk CARDINALITY_TYPE=est AUGMENT=True EPOCHS=100 AUGMENT_POOLING=attention AUGMENT_COARSE_LAYERS=1 LAMBDA_STRUCT=0.1 FINAL_ACTIVATION_CLASS_NAME=SELU"
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
