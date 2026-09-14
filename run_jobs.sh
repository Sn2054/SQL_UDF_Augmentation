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
JOBS=(
    "resume-mean-cl1-l0.001 TEST_DB=fhnk CARDINALITY_TYPE=est AUGMENT=True EPOCHS=100 AUGMENT_POOLING=mean AUGMENT_COARSE_LAYERS=1 LAMBDA_STRUCT=0.001 PRETRAINED_MODEL_ARTIFACT_DIR=$SCRIPT_DIR/saved/models/aug_est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_fhnk_bs512_ep100_maxr30_augpoolmean_augrefgated_residual_augcl1_augnoRET_cfl0.001 PRETRAINED_MODEL_FILENAME=aug_est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_fhnk_bs512_ep100_maxr30_augpoolmean_augrefgated_residual_augcl1_augnoRET_cfl0.001_20260913_045046_076"
    "attention-cl0-l0 TEST_DB=fhnk CARDINALITY_TYPE=est AUGMENT=True EPOCHS=100 AUGMENT_POOLING=attention AUGMENT_COARSE_LAYERS=0 LAMBDA_STRUCT=0.0"
    "wmean-cl1-l0.1 TEST_DB=fhnk CARDINALITY_TYPE=est AUGMENT=True EPOCHS=100 AUGMENT_POOLING=weighted_mean AUGMENT_COARSE_LAYERS=1 LAMBDA_STRUCT=0.1"
    "max-cl3-l0.1 TEST_DB=fhnk CARDINALITY_TYPE=est AUGMENT=True EPOCHS=100 AUGMENT_POOLING=max AUGMENT_COARSE_LAYERS=3 LAMBDA_STRUCT=0.1"
    "hybrid-cl2-l0.01 TEST_DB=fhnk CARDINALITY_TYPE=est AUGMENT=True EPOCHS=100 AUGMENT_POOLING=hybrid AUGMENT_COARSE_LAYERS=2 LAMBDA_STRUCT=0.01"
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
