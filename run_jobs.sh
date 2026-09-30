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
# Leftover sweep (2026-09-30). Pinned by GPU UUID, not index: index order can
# differ between nvidia-smi and CUDA and across reboots, the UUID can't.
# Same cohort as the first sweep (coarse_layers=1 / lambda_struct=0.1 / est /
# 100 epochs), see understanding/progress.md sec. 8.
#   - fhnk: only GELU and SiLU are new (LeakyReLU/ReLU/SELU/CELU and both
#     hybrids already have rows).
#   - employee, financial: activations ReLU/SELU/CELU/GELU/SiLU (attention
#     pooling) and the two hybrids (LeakyReLU). No LeakyReLU or gated runs.
# -----------------------------------------------------------------------
COMMON="GPU_UUID=GPU-042f3a36-fa80-8994-33b1-37c79834d513 CARDINALITY_TYPE=est AUGMENT=True EPOCHS=100 AUGMENT_COARSE_LAYERS=1 LAMBDA_STRUCT=0.1"

JOBS=(
    "fhnk-act-gelu TEST_DB=fhnk $COMMON AUGMENT_POOLING=attention ACTIVATION_CLASS_NAME=GELU"
    "fhnk-act-silu TEST_DB=fhnk $COMMON AUGMENT_POOLING=attention ACTIVATION_CLASS_NAME=SiLU"
)
for db in employee financial; do
    JOBS+=(
        "$db-act-relu TEST_DB=$db $COMMON AUGMENT_POOLING=attention ACTIVATION_CLASS_NAME=ReLU"
        "$db-act-selu TEST_DB=$db $COMMON AUGMENT_POOLING=attention ACTIVATION_CLASS_NAME=SELU"
        "$db-act-celu TEST_DB=$db $COMMON AUGMENT_POOLING=attention ACTIVATION_CLASS_NAME=CELU"
        "$db-act-gelu TEST_DB=$db $COMMON AUGMENT_POOLING=attention ACTIVATION_CLASS_NAME=GELU"
        "$db-act-silu TEST_DB=$db $COMMON AUGMENT_POOLING=attention ACTIVATION_CLASS_NAME=SiLU"
        "$db-pool-hybrid_attn_max TEST_DB=$db $COMMON AUGMENT_POOLING=hybrid_attn_max ACTIVATION_CLASS_NAME=LeakyReLU"
        "$db-pool-hybrid_max_wmean TEST_DB=$db $COMMON AUGMENT_POOLING=hybrid_max_wmean ACTIVATION_CLASS_NAME=LeakyReLU"
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
