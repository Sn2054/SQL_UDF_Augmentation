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
# Run one queue's job list on the other queue's GPU (e.g. when one GPU reservation has ended):
#   QUEUE=B JOB_LIST=A bash run_jobs.sh ...      # queue A's jobs on GPU B via run_code_copy.sh
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
JOB_LIST="${JOB_LIST:-$QUEUE}"   # which queue's job list to run; QUEUE still picks the GPU and run_code script
START_AT="${START_AT:-}"
RESUME_DIR="${RESUME_DIR:-}"
RESUME_FILE="${RESUME_FILE:-}"

# Both UUIDs are the user's allocations (given by the user; never index-based).
GPU_UUID_A="GPU-dfcf303a-7e8d-e2f6-b68c-c3d30682f80b"   # since 2026-11-02 (the earlier A, GPU-452b1bc7-..., ended 2026-10-28)
GPU_UUID_B="GPU-060f92dc-2611-318d-5454-1ce480e07298"
case "$QUEUE" in
    A) GPU_UUID="$GPU_UUID_A"; RUN_SCRIPT="$SCRIPT_DIR/run_code.sh" ;;
    B) GPU_UUID="$GPU_UUID_B"; RUN_SCRIPT="$SCRIPT_DIR/run_code_copy.sh" ;;
    *) echo "QUEUE must be A or B"; exit 2 ;;
esac

# -----------------------------------------------------------------------
# Plan (2026-10-30): pooling and activation stay fixed at hybrid + SELU (to be kept consistent across DBs);
# only the numeric hyperparameters (coarse_layers, lambda_struct) may differ per DB. Our DBs are now
# fhnk, employee, financial, credit, seznam, tournament, imdb, hepatitis, tpc_h.
# Two GPUs again from 2026-11-02, so the list is split into queues A and B (below). Jobs:
#   hybrid est on the three new DBs at the default settings (cl1, lambda 0.1). Their est
#        baselines already exist (50 epochs, same server and code). imdb and hepatitis are done; tpc_h was
#        interrupted at epoch 27 by the 2026-10-31 maintenance and resumes first in queue A.
#   tuning for the DBs where hybrid at the defaults is weakest vs the baseline: tournament
#        (push +0.4%, pull +4.4%), hepatitis (push -7.9%, pull -15.4%; added 2026-11-02), financial and
#        seznam pull (-3.5%, -3.8%). One setting changed at a time from cl1/lambda 0.1: coarse_layers 2,
#        or lambda_struct 0.25.
# Same settings otherwise as all augmented runs: est, 100 epochs, SELU, new supernodes (SEQ regions +
# CFG coarse edges). The model name carries augcl<N> and cfl<lambda>, so tuning runs get their own
# folders, and the augmented sheet keeps one row per config.
# The 2026-10-26 plan (hybrid est on 6 DBs, financial act baseline rerun) finished; see git history and
# progress.md. Rough job time ~15-20 h per augmented job, ~6 h per baseline.
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
baseline() {  # label db card -- non-augmented reference model (LeakyReLU default), 50 epochs (the baseline protocol)
    JOBS+=("$1 TEST_DB=$2 GPU_UUID=$GPU_UUID AUGMENT=False EPOCHS=50 CARDINALITY_TYPE=$3")
}

# The 2026-11-02 lists (tpc_h, tournament/hepatitis cl2 + lambda .25) finished; seznam cl2 and financial cl2 were
# stopped on 2026-11-06 (for the CPU-only per-query profile) and resume from their checkpoints below.
#
# Plan (2026-11-06): hybrid + SELU est at the defaults (cl1, lambda 0.1, patience 20) on DBs not yet run with it
# by us. accidents and ssb first (accidents has no augmented run at all; ssb only one, patience 10). baseball,
# movielens and airline have one hybrid run each from Ishana (new supernodes, patience 10), all below the
# baseline; geneea and carcinogenesis have none. imdb's defaults run (patience 10) is +38.6% push but -0.2%
# pull, so it gets coarse_layers 2, the setting that took hepatitis to +17.1%. Every DB has its est baseline
# (50 epochs).
# Slot by slot:
#   1  accidents hybrid est              | seznam hybrid est cl2 (resume, epoch 80 of 100)
#   2  baseball hybrid est               | ssb hybrid est
#   3  movielens hybrid est              | geneea hybrid est
#   4  imdb hybrid est cl2               | airline hybrid est
#   5  financial hybrid est cl2 (resume) | carcinogenesis hybrid est
# Queue B's 2026-11-06 list finished on 2026-11-10. Its next list (2026-11-10), tuning the near misses:
#   baseball (+10.9% push / -0.4% pull) and accidents (-0.9% / +12.9%) get coarse_layers 2, which fixed pull-up on
#   hepatitis and imdb; tournament gets the defaults with patience 20 (its only defaults run used patience 10, and
#   patience 20 helped hepatitis and movielens). airline waits until its collapse (push q50 3.01 vs 1.37) is
#   understood. (run_jobs.sh got the same list on 2026-11-11, once queue A was idle.)
# Queue A's 2026-11-06 list finished on 2026-11-11. Its next list (2026-11-11), one setting changed per run:
#   financial: cl2 made pull-up worse (-12.7% vs -3.5% at the defaults), so try cl3 and, separately, lambda 0.25;
#   ssb (pull -14.7%) and geneea (push -7.1%) get cl2; seznam gets lambda 0.25 (cl2 left its pull-up at -3.8%).
#   (Written into run_jobs.sh while queue B ran run_jobs_copy.sh; the copy gets it once queue B is idle.)
# Queue A's 2026-11-11 list finished on 2026-11-15 (geneea cl2 +0.4% / +8.2%, 4.3% overall; seznam lambda 0.25
#   +13.3% / -0.9%). Its next list (2026-11-16), one setting changed per run, following each DB's trend:
#   geneea cl3 (cl1 -3.4% -> cl2 +4.3% overall; cl3 fixed financial's pull-up); basketball cl2 (defaults +2.0% /
#   +6.0%; cl2 fixed pull-up on hepatitis, imdb and accidents); seznam lambda 0.5 (pull-up -3.8% at lambda 0.1,
#   -0.9% at 0.25). (Written into run_jobs.sh while queue B runs run_jobs_copy.sh; queue B's own list is in the copy.)
# Queue B's 2026-11-10 list finished on 2026-11-13 (accidents cl2 met the goal; baseball cl2 +0.9%, tournament defaults
#   at patience 20 +5.0% / +5.0%, 4.99% overall). Its next list (2026-11-13): every DB still short of the goal that
#   queue A doesn't cover (geneea, seznam), one setting changed from the defaults per run:
#   baseball lambda 0.25 (cl1 +10.9% / -0.4%; cl2 hurt pull-up); tournament cl3 (cl2 +8.2% / -1.2%); ssb cl3 (cl2
#   lifted pull-up from -14.7% to -3.1%, and cl3 fixed financial's pull-up); basketball and consumer at the defaults
#   (never run with hybrid + SELU); carcinogenesis cl2 (defaults -13.4%; cl2 fixed hepatitis and imdb); airline at the
#   defaults (our run collapsed on push-down, 3.01 vs 1.37). Order: the DBs never run with hybrid + SELU first
#   (basketball, consumer), then the rest; tournament last, since its 4.99% run is already in best_results.
# 2026-11-18: both queues stopped at about 17:16 / 17:30 (no error in the logs). basketball cl2 had finished
#   training and testing (push +19.4%, pull -3.9%) but its results row was not written, so queue A re-tests it from the
#   finished checkpoint (training is skipped). ssb cl3 ended +1.9% / +7.1% (cl1 -4.6%, cl2 -2.5% overall), so ssb gets
#   cl4; geneea cl3 was worse than cl2 (-8.5% / +2.0%), so geneea gets lambda 0.25 (it lifted the lagging workload on
#   baseball and seznam). Queue B resumes carcinogenesis cl2 from its epoch-27 checkpoint, then airline, tournament cl3.
# (later overrides win: env applies them in order, so CL2/L025 replace the values in $COMMON)
CL2="AUGMENT_COARSE_LAYERS=2"
CL3="AUGMENT_COARSE_LAYERS=3"
L025="LAMBDA_STRUCT=0.25"
L05="LAMBDA_STRUCT=0.5"
CL4="AUGMENT_COARSE_LAYERS=4"
M=saved/models
P=aug_est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge
S=bs512_ep100_esp20_maxr30_augpoolhybrid_augrefgated_residual_augcl2_augnoRET_augseq_augcfge_cfl0.1_actSELU
RESUME_SEZNAM="PRETRAINED_MODEL_ARTIFACT_DIR=$M/${P}_seznam_$S PRETRAINED_MODEL_FILENAME=${P}_seznam_${S}_20261104_234711_091"
RESUME_FINANCIAL="PRETRAINED_MODEL_ARTIFACT_DIR=$M/${P}_financial_$S PRETRAINED_MODEL_FILENAME=${P}_financial_${S}_20261105_161959_089"
# basketball cl2: finished checkpoint (re-test only); carcinogenesis cl2: stopped after epoch 27 (2026-11-18)
RESUME_BASKETBALL="PRETRAINED_MODEL_ARTIFACT_DIR=$M/${P}_basketball_$S PRETRAINED_MODEL_FILENAME=${P}_basketball_${S}_20261117_160430_018"
RESUME_CARCINOGENESIS="PRETRAINED_MODEL_ARTIFACT_DIR=$M/${P}_carcinogenesis_$S PRETRAINED_MODEL_FILENAME=${P}_carcinogenesis_${S}_20261118_102540_094"
if [[ "$JOB_LIST" == A ]]; then
    job basketball-selu-hybrid-est-cl2    basketball     est $HYBRID $CL2 $RESUME_BASKETBALL
    job seznam-selu-hybrid-est-l05        seznam         est $HYBRID $L05
    job geneea-selu-hybrid-est-l025       geneea         est $HYBRID $L025
    job ssb-selu-hybrid-est-cl4           ssb            est $HYBRID $CL4
else
    job carcinogenesis-selu-hybrid-est-cl2 carcinogenesis est $HYBRID $CL2 $RESUME_CARCINOGENESIS
    job airline-selu-hybrid-est           airline        est $HYBRID
    job tournament-selu-hybrid-est-cl3    tournament     est $HYBRID $CL3
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
    echo "queue $QUEUE ($GPU_UUID, $(basename "$RUN_SCRIPT")), job list $JOB_LIST: ${#JOBS[@]} jobs"
    for job in "${JOBS[@]}"; do
        echo "  ${job%% *}  $(echo "${job#* }" | sed "s#$COMMON#<common>#")"
    done
    exit 0
fi

# Queue A waits for any leftover run_code.sh job on its GPU (never two jobs on one GPU).
# (The pattern doesn't match run_code_copy.sh, so queue B never blocks queue A.)
if [[ "$QUEUE" == A ]]; then  # same GPU as a leftover run_code.sh job
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
