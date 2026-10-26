#!/usr/bin/env bash
# Driver for run_code.sh: runs a list of jobs, each one a set of env-var
# overrides for run_code.sh, sequentially. Two queues run in parallel on two
# GPUs, one per tmux session:
#   queue A: this file (run_jobs.sh)      -> run_code.sh,      GPU_UUID_A
#   queue B: run_jobs_copy.sh (same file, QUEUE defaults to B) -> run_code_copy.sh, GPU_UUID_B
# Each queue runs from its own pair of scripts, so one pair can be edited while the
# other runs (bash reads a running script incrementally: never edit a running one).
# The results writers lock the shared xlsx files, so the two queues can't clobber
# each other's rows.
#
# This mirrors Ishana's run_all.sh pattern (GRACEFUL_Augmentation repo): the JOBS
# lists below are the one place that documents what was run.
#
# Usage:
#   QUEUE=A LIST_ONLY=1 bash run_jobs.sh          # print a queue's jobs, run nothing
# After an interruption (e.g. a server reboot), restart a queue at the interrupted job and resume it:
#   START_AT=<label> RESUME_DIR=saved/models/<dir> RESUME_FILE=<checkpoint name without .pt> bash run_jobs.sh ...
# START_AT skips the jobs listed before <label>; the resume applies to that first job only.
#   tmux new -s jobsA   ->   bash run_jobs.sh      2>&1 | tee -a saved/training_logs/run_jobs_A_$(date +%Y%m%d_%H%M%S).log
#   tmux new -s jobsB   ->   bash run_jobs_copy.sh 2>&1 | tee -a saved/training_logs/run_jobs_B_$(date +%Y%m%d_%H%M%S).log
#
# Environment variables are single-valued per job (the driver word-splits the
# overrides), so est and act are separate jobs, never CARDINALITY_TYPES="est act".

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="${QUEUE:-A}"   # run_jobs_copy.sh defaults this to B
LIST_ONLY="${LIST_ONLY:-0}"
START_AT="${START_AT:-}"
RESUME_DIR="${RESUME_DIR:-}"
RESUME_FILE="${RESUME_FILE:-}"

# Both UUIDs are the user's allocations (given by the user; never index-based).
GPU_UUID_A="GPU-452b1bc7-d479-c54a-050e-e27eabf2d1de"
GPU_UUID_B="GPU-060f92dc-2611-318d-5454-1ce480e07298"
case "$QUEUE" in
    A) GPU_UUID="$GPU_UUID_A"; RUN_SCRIPT="$SCRIPT_DIR/run_code.sh" ;;
    B) GPU_UUID="$GPU_UUID_B"; RUN_SCRIPT="$SCRIPT_DIR/run_code_copy.sh" ;;
    *) echo "QUEUE must be A or B"; exit 2 ;;
esac

# -----------------------------------------------------------------------
# Plan (2026-10-26): finalize the pooling among attention, max and hybrid (EP dropped: no gain over
# attention). hybrid = concat(mean-pool, max-pool) -> linear back to d (semantic_graph_augmentor.py).
# hybrid est on the six databases that already have attention and/or max est, same settings as all
# augmented runs: est, 100 epochs, SELU, coarse_layers=1, lambda_struct=0.1, new supernodes (SEQ regions
# + CFG coarse edges). Plus a rerun of the financial act baseline (non-augmented, LeakyReLU = run_code.sh
# default, 100 epochs): its 50-epoch push q50 1.6225 is far above every other act baseline (1.11-1.23).
# The baseline sheet is keyed on (db, card), so the rerun replaces that row (backup:
# results/backup_20261026_act_rows/baseline_cost_estimation.before_financial_act_rerun.xlsx).
# The 2026-10-19 plan (attention/act/max/EP slots 1-9) finished; see git history and progress.md.
# Rough job time ~15-20 h per augmented job, ~6 h per baseline.
# -----------------------------------------------------------------------
COMMON="GPU_UUID=$GPU_UUID AUGMENT=True EPOCHS=100 AUGMENT_COARSE_LAYERS=1 LAMBDA_STRUCT=0.1 AUGMENT_SEQ_REGIONS=True AUGMENT_CFG_COARSE_EDGES=True ACTIVATION_CLASS_NAME=SELU"
ATTN="AUGMENT_POOLING=attention"
MAX="AUGMENT_POOLING=max"
HYBRID="AUGMENT_POOLING=hybrid"

JOBS=()
job() {  # label db card pooling-overrides...
    local label="$1" db="$2" card="$3"; shift 3
    JOBS+=("$label TEST_DB=$db $COMMON CARDINALITY_TYPE=$card $*")
}
baseline() {  # label db card -- non-augmented reference model (LeakyReLU default), 100 epochs
    JOBS+=("$1 TEST_DB=$2 GPU_UUID=$GPU_UUID AUGMENT=False EPOCHS=100 CARDINALITY_TYPE=$3")
}

# Slot by slot (both queues side by side):
#   1  financial act baseline | employee hybrid est
#   2  fhnk hybrid est        | credit hybrid est
#   3  financial hybrid est   | seznam hybrid est
#   4  tournament hybrid est  |
if [[ "$QUEUE" == A ]]; then
    baseline financial-base-act       financial  act
    job fhnk-selu-hybrid-est          fhnk       est $HYBRID
    job financial-selu-hybrid-est     financial  est $HYBRID
    job tournament-selu-hybrid-est    tournament est $HYBRID
else
    job employee-selu-hybrid-est      employee   est $HYBRID
    job credit-selu-hybrid-est        credit     est $HYBRID
    job seznam-selu-hybrid-est        seznam     est $HYBRID
fi

if [[ -n "$START_AT" ]]; then
    for i in "${!JOBS[@]}"; do
        [[ "${JOBS[$i]%% *}" == "$START_AT" ]] && break
        unset 'JOBS[i]'
    done
    JOBS=("${JOBS[@]}")
    if [[ ${#JOBS[@]} -eq 0 ]]; then echo "START_AT=$START_AT matches no job in queue $QUEUE"; exit 2; fi
fi
if [[ -n "$RESUME_DIR$RESUME_FILE" ]]; then
    if [[ -z "$RESUME_DIR" || -z "$RESUME_FILE" || ! -f "$SCRIPT_DIR/$RESUME_DIR/$RESUME_FILE.pt" && ! -f "$RESUME_DIR/$RESUME_FILE.pt" ]]; then
        echo "RESUME_DIR/RESUME_FILE must both be set and point at an existing <file>.pt"; exit 2
    fi
    JOBS[0]="${JOBS[0]} PRETRAINED_MODEL_ARTIFACT_DIR=$RESUME_DIR PRETRAINED_MODEL_FILENAME=$RESUME_FILE"
fi

if [[ "$LIST_ONLY" == "1" ]]; then
    echo "queue $QUEUE ($GPU_UUID, $(basename "$RUN_SCRIPT")): ${#JOBS[@]} jobs"
    for job in "${JOBS[@]}"; do
        echo "  ${job%% *}  $(echo "${job#* }" | sed "s#$COMMON#<common>#")"
    done
    exit 0
fi

# Queue A waits for any leftover run_code.sh job on its GPU (never two jobs on one GPU).
# (The pattern doesn't match run_code_copy.sh, so queue B never blocks queue A.)
if [[ "$QUEUE" == A ]]; then
    while pgrep -u "$USER" -f "bash .*run_code\.sh" >/dev/null; do sleep 300; done
fi

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
echo "=== SUMMARY (queue $QUEUE) ==="
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
