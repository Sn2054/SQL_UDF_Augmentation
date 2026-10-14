#!/usr/bin/env bash
# Driver for run_code.sh: runs a list of jobs, each one a set of env-var
# overrides for run_code.sh. Jobs are grouped into WAVES; pick which waves to
# run with WAVES, then launch this in a tmux session.
#
# This mirrors Ishana's run_all.sh pattern (GRACEFUL_Augmentation repo): a
# single JOBS array is the one place that documents what was run, so editing
# it and committing gives a readable git history of "what ran and why" --
# use a descriptive commit message per change to the list rather than
# committing silently.
#
# Jobs run SEQUENTIALLY on purpose (they share the results/*.xlsx files).
#
# Usage (GPU_UUID = the UUID the lab assigned to you; never a GPU index):
#   GPU_UUID=GPU-... LIST_ONLY=1 WAVES="1 2" bash run_jobs.sh   # print the jobs, run nothing
#   tmux new -s jobs
#   GPU_UUID=GPU-... WAVES="1" bash run_jobs.sh                 # run wave 1
#
# Environment variables are single-valued per job (the driver word-splits the
# overrides), so est and act are separate jobs, never CARDINALITY_TYPES="est act".

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_code.sh"

: "${GPU_UUID:?set GPU_UUID=<your allocated GPU UUID> (the old UUID from before the 2026-10-12 reset is not yours any more)}"
WAVES="${WAVES:-1}"
LIST_ONLY="${LIST_ONLY:-0}"

# -----------------------------------------------------------------------
# What each wave answers (hours are rough, from measured job times: baseline ~12 h,
# augmented on the new supernodes ~20 h; est = DuckDB estimated cardinalities,
# act = actual cardinalities; dd and wj are left for later). The baseline is the
# non-augmented model with LeakyReLU, as in GRACEFUL; every augmented job uses SELU.
#
#  1  fhnk / employee / financial, est: non-augmented baseline at 100 epochs, then
#     augmented SELU + attention.  Does augmentation beat a fairly trained baseline?   ~96 h
#  2  same 3 DBs, est: SELU with max, EP M=4 and EP M=16 (9 jobs). Which pooling wins?  ~180 h
#  3  same 3 DBs, act: baseline + the winning config (edit BEST_ARGS first).            ~96 h
#
# Baselines are queued BEFORE the augmented job of the same DB on purpose: the
# augmented writer computes its diff columns against whatever baseline row
# exists when the job finishes. The baseline sheet is keyed on (db, card type)
# only, so the first 100-epoch baseline overwrites the old 50-epoch row; a copy
# of the old sheet is kept as results/baseline_cost_estimation_ep50.xlsx.
# -----------------------------------------------------------------------

# Settings shared by every augmented job (activation is passed per job).
COMMON_AUG="GPU_UUID=$GPU_UUID AUGMENT=True EPOCHS=100 AUGMENT_COARSE_LAYERS=1 LAMBDA_STRUCT=0.1 AUGMENT_SEQ_REGIONS=True AUGMENT_CFG_COARSE_EDGES=True"

# The config wave 3 runs. Edit after waves 1-2 (e.g. AUGMENT_POOLING=max, or
# AUGMENT_POOLING=multi_query_attention AUGMENT_MQ_QUERIES=4).
BEST_ARGS="ACTIVATION_CLASS_NAME=SELU AUGMENT_POOLING=attention"

THREE_DBS=(fhnk employee financial)

JOBS=()
base_job() {  # db card
    JOBS+=("$1-base-$2 TEST_DB=$1 GPU_UUID=$GPU_UUID CARDINALITY_TYPE=$2 AUGMENT=False EPOCHS=100")
}
aug_job() {   # db card label extra-overrides...
    local db="$1" card="$2" label="$3"; shift 3
    JOBS+=("$db-$label-$card TEST_DB=$db $COMMON_AUG CARDINALITY_TYPE=$card $*")
}

wave_1() {
    for db in "${THREE_DBS[@]}"; do base_job "$db" est; done
    for db in "${THREE_DBS[@]}"; do aug_job "$db" est selu-attn ACTIVATION_CLASS_NAME=SELU AUGMENT_POOLING=attention; done
}

wave_2() {
    for db in "${THREE_DBS[@]}"; do
        aug_job "$db" est selu-max  ACTIVATION_CLASS_NAME=SELU AUGMENT_POOLING=max
        aug_job "$db" est selu-ep4  ACTIVATION_CLASS_NAME=SELU AUGMENT_POOLING=multi_query_attention AUGMENT_MQ_QUERIES=4
        aug_job "$db" est selu-ep16 ACTIVATION_CLASS_NAME=SELU AUGMENT_POOLING=multi_query_attention AUGMENT_MQ_QUERIES=16
    done
}

wave_3() { for db in "${THREE_DBS[@]}"; do base_job "$db" act; done; for db in "${THREE_DBS[@]}"; do aug_job "$db" act best $BEST_ARGS; done; }

for w in $WAVES; do
    case "$w" in
        1) wave_1 ;; 2) wave_2 ;; 3) wave_3 ;;
        *) echo "Unknown wave '$w' (valid: 1 2 3)"; exit 2 ;;
    esac
done

if [[ "$LIST_ONLY" == "1" ]]; then
    echo "${#JOBS[@]} jobs for WAVES='$WAVES':"
    for job in "${JOBS[@]}"; do
        label="${job%% *}"; overrides="${job#* }"
        # Hide the long shared prefix and the resume path to keep the listing readable.
        echo "  $label  ${overrides//$COMMON_AUG/<common>}" | sed -E "s#PRETRAINED_MODEL_ARTIFACT_DIR=[^ ]*#PRETRAINED_MODEL_ARTIFACT_DIR=<...>#"
    done
    exit 0
fi

# Keep the old 50-epoch baseline rows before the first 100-epoch baseline overwrites them.
cp -n "$SCRIPT_DIR/results/baseline_cost_estimation.xlsx" "$SCRIPT_DIR/results/baseline_cost_estimation_ep50.xlsx" 2>/dev/null || true

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
