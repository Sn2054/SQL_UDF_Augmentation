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
#   tmux new -s jobsA   ->   bash run_jobs.sh      2>&1 | tee -a saved/training_logs/run_jobs_A_$(date +%Y%m%d_%H%M%S).log
#   tmux new -s jobsB   ->   bash run_jobs_copy.sh 2>&1 | tee -a saved/training_logs/run_jobs_B_$(date +%Y%m%d_%H%M%S).log
#
# Environment variables are single-valued per job (the driver word-splits the
# overrides), so est and act are separate jobs, never CARDINALITY_TYPES="est act".

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="${QUEUE:-A}"   # run_jobs_copy.sh defaults this to B
LIST_ONLY="${LIST_ONLY:-0}"

# Both UUIDs are the user's allocations (given by the user; never index-based).
GPU_UUID_A="GPU-452b1bc7-d479-c54a-050e-e27eabf2d1de"
GPU_UUID_B="GPU-060f92dc-2611-318d-5454-1ce480e07298"
case "$QUEUE" in
    A) GPU_UUID="$GPU_UUID_A"; RUN_SCRIPT="$SCRIPT_DIR/run_code.sh" ;;
    B) GPU_UUID="$GPU_UUID_B"; RUN_SCRIPT="$SCRIPT_DIR/run_code_copy.sh" ;;
    *) echo "QUEUE must be A or B"; exit 2 ;;
esac

# -----------------------------------------------------------------------
# Plan (2026-10-19). Baselines are not run here: the non-augmented baselines
# (50 epochs) come from Ishana's runs, merged into results/baseline_cost_estimation.xlsx;
# fhnk/employee/financial have our own. Every augmented job: est or act, 100 epochs,
# SELU, coarse_layers=1, lambda_struct=0.1, new supernodes (SEQ regions + CFG coarse edges).
# New DBs: credit, tournament, seznam (chosen 2026-10-19: both est and act baselines
# exist and Ishana is not augmenting them; Ishana's baselines merged into the baseline sheet).
# Rough job time ~15-20 h (one GPU-day per ~1.3 jobs).
# -----------------------------------------------------------------------
COMMON="GPU_UUID=$GPU_UUID AUGMENT=True EPOCHS=100 AUGMENT_COARSE_LAYERS=1 LAMBDA_STRUCT=0.1 AUGMENT_SEQ_REGIONS=True AUGMENT_CFG_COARSE_EDGES=True ACTIVATION_CLASS_NAME=SELU"
ATTN="AUGMENT_POOLING=attention"
MAX="AUGMENT_POOLING=max"
EP4="AUGMENT_POOLING=multi_query_attention AUGMENT_MQ_QUERIES=4"
EP16="AUGMENT_POOLING=multi_query_attention AUGMENT_MQ_QUERIES=16"

JOBS=()
job() {  # label db card pooling-overrides...
    local label="$1" db="$2" card="$3"; shift 3
    JOBS+=("$label TEST_DB=$db $COMMON CARDINALITY_TYPE=$card $*")
}

# Order (both queues run side by side, ~15-20 h per slot): first the result that can be shown for
# each new DB (attention est, then act, so each DB is complete early), then act for the original
# DBs, then the pooling ablations (max, then EP). Slot by slot:
#   1  credit est        | seznam est
#   2  tournament est    | credit act
#   3  tournament act    | seznam act        -> all three new DBs have est + act
#   4  fhnk act          | employee act
#   5  financial act     | employee max      -> original DBs have act
#   6  financial max     | employee EP4
#   7  credit EP4        | financial EP4
#   8  tournament EP4    | employee EP16
#   9  financial EP16    | seznam EP4
if [[ "$QUEUE" == A ]]; then
    job credit-selu-attn-est       credit     est $ATTN
    job tournament-selu-attn-est   tournament est $ATTN
    job tournament-selu-attn-act   tournament act $ATTN
    job fhnk-selu-attn-act         fhnk       act $ATTN
    job financial-selu-attn-act    financial  act $ATTN
    job financial-selu-max-est     financial  est $MAX
    job credit-selu-ep4-est        credit     est $EP4
    job tournament-selu-ep4-est    tournament est $EP4
    job financial-selu-ep16-est    financial  est $EP16
else
    job seznam-selu-attn-est       seznam     est $ATTN
    job credit-selu-attn-act       credit     act $ATTN
    job seznam-selu-attn-act       seznam     act $ATTN
    job employee-selu-attn-act     employee   act $ATTN
    job employee-selu-max-est      employee   est $MAX
    job employee-selu-ep4-est      employee   est $EP4
    job financial-selu-ep4-est     financial  est $EP4
    job employee-selu-ep16-est     employee   est $EP16
    job seznam-selu-ep4-est        seznam     est $EP4
fi

if [[ "$LIST_ONLY" == "1" ]]; then
    echo "queue $QUEUE ($GPU_UUID, $(basename "$RUN_SCRIPT")): ${#JOBS[@]} jobs"
    for job in "${JOBS[@]}"; do
        echo "  ${job%% *}  $(echo "${job#* }" | sed "s#$COMMON#<common>#")"
    done
    exit 0
fi

# Queue A shares its GPU with the job left over from the stopped wave 2: wait for it.
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
