#!/usr/bin/env bash
# Paper-faithful leave-one-out training with DuckDB estimated cardinalities.
#
# Protocol (matches Graceful paper exactly, except card type):
#   - data_keyword: complex_dd  → trains on ALL 19 DBs (20 minus held-out), from scratch
#   - card_type: est            → DuckDB estimated cardinalities at training AND test time
#   - model_config: ddestfonudf → plan nodes use est_card; UDF internal nodes use in_rows_est
#   - N_RUNS sequential repetitions for the configured held-out database
#
# TEST_ALL_CARDINALITY selects either the configured card type or all four card types.
#
# Prerequisites:
#   1. setup_symlinks.sh has been run (all 20 DBs linked under graceful_results/)
#   2. You are inside a tmux session
#
# Usage:
#   tmux new -s est_training
#   cd /mnt/shared/development/shehryar/POLARIS/Graceful
#   bash ../pipeline_scripts/run_paper_est.sh
#   # Ctrl+B D to detach; tmux attach -t est_training to check in

set -euo pipefail

# =============================================================================
# 1. Input and output paths
# =============================================================================

DATASET_BASE="/mnt/shared/data/dataset/Graceful_data"
WL_BASE="$DATASET_BASE/workload_runs/"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS_OUT="$SCRIPT_DIR/saved/models"
LOG_DIR="$SCRIPT_DIR/saved/training_logs"
SUMMARY_DIR="$SCRIPT_DIR/saved/summary_xlsx"
BASELINE_RESULTS_XLSX="$SCRIPT_DIR/results/baseline_cost_estimation.xlsx"
AUGMENTED_RESULTS_XLSX="$SCRIPT_DIR/results/augmented_cost_estimation.xlsx"
AUGMENTED_PLOT_DIR="$SCRIPT_DIR/results/augmented_plots"

SUMMARY_SCRIPT="$SCRIPT_DIR/summarize_training_log.py"
REPEAT_SUMMARY_SCRIPT="$SCRIPT_DIR/summarize_repeated_runs.py"
BASELINE_RESULTS_SCRIPT="$SCRIPT_DIR/update_baseline_cost_estimation.py"
AUGMENTED_RESULTS_SCRIPT="$SCRIPT_DIR/update_augmented_cost_estimation.py"
AUGMENTED_PLOT_SCRIPT="$SCRIPT_DIR/plot_augmented_cost_estimation.py"
LOSS_CURVE_SCRIPT="$SCRIPT_DIR/plot_loss_curves.py"

# =============================================================================
# 2. Runtime and reproducibility
# =============================================================================

N_RUNS=1                 #? Number of sequential repetitions; all repetitions use SEED below.
SEED=42 
DEVICE=1
CUDA_DEVICE="cuda:${DEVICE}"  #? Any integer, e.g. 0, 1, 2, 42.
DETERMINISTIC=True       #? True, False

# =============================================================================
# 3. Model and training data
# =============================================================================

MODEL_CONFIG="ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge"
DATA_KEYWORD="complex_dd"
DATABASE="duckdb"

# =============================================================================
# 4. Held-out evaluation
# =============================================================================

TEST_DB="${TEST_DB:-employee}"
CARDINALITY_TYPE="${CARDINALITY_TYPE:-act}"       #? est, act, dd, wj
TEST_ALL_CARDINALITY=False   #? False: selected type only; True: est, act, dd, and wj.

# =============================================================================
# 5. Data loading and training
# =============================================================================

NUM_WORKERS=8
INCLUDE_PULLUP_DATA=True
INCLUDE_PUSHDOWN_DATA=True
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE=512
ACTIVATION_CLASS_NAME="${ACTIVATION_CLASS_NAME:-LeakyReLU}"  #? LeakyReLU, ReLU, CELU, SELU -- applies everywhere (final layer, message passing, node encoder). Why? Simpler than special-casing just the output layer.
PRETRAINED_MODEL_ARTIFACT_DIR="${PRETRAINED_MODEL_ARTIFACT_DIR:-}"  #? Set both this and PRETRAINED_MODEL_FILENAME to resume an interrupted run -- checkpoint filenames are freshly timestamped every invocation, so a plain re-run does NOT auto-resume; point these at the old saved/models/<dir>/ and its filename (no .pt) explicitly.
PRETRAINED_MODEL_FILENAME="${PRETRAINED_MODEL_FILENAME:-}"

# =============================================================================
# 6. Semantic graph augmfwentation
# =============================================================================

AUGMENT=True                          #? True, False
TEST_AUGMENT=True                     #? Effective only when AUGMENT=True; False disables augmentation for held-out testing.
AUGMENT_POOLING="${AUGMENT_POOLING:-attention}"  #? mean, sum, max, weighted_mean, attention, hybrid, hybrid_attn_max, hybrid_max_wmean, hybrid_max_wmean_gated
AUGMENT_REFINEMENT="gated_residual"   #? residual_sum, gated_residual
AUGMENT_COARSE_LAYERS="${AUGMENT_COARSE_LAYERS:-1}"  #? 0, 1, 2, ... (run 1 round of message passing between the coarse/region nodes created by the augmentation.)
AUGMENT_INCLUDE_INV=False             #? True, False
AUGMENT_REFINE_RET=False              #? True, False
LAMBDA_STRUCT="${LAMBDA_STRUCT:-0.001}"   #? 0.0, 0.001, 0.01, 0.05, 0.1 (total_loss = runtime_loss + LAMBDA_STRUCT * coarse_fine_loss)

# =============================================================================
# 7. Diagnostics
# =============================================================================

SAVE_LOSS_PLOTS="${SAVE_LOSS_PLOTS:-True}"   #? True, False -- generate train/val loss & accuracy plots after this run

# =============================================================================
# 8. Values derived for this execution
# =============================================================================

GROUP_RUN_TIME="$(date +%Y%m%d_%H%M%S)_$$"  #? PID suffix guards against collisions with a concurrent run_code.sh invocation that happens to start in the same second -- GROUP_RUN_TIME feeds LOG, AGGREGATE_LOG, and SUMMARY_XLSX, and a same-second collision means two unrelated runs silently share (and corrupt) those files.
SUMMARY_STEM="${GROUP_RUN_TIME}_${TEST_DB}_${CARDINALITY_TYPE}_${AUGMENT}"
AGGREGATE_XLSX="$SUMMARY_DIR/${SUMMARY_STEM}_aggregate.xlsx"
AGGREGATE_LOG="$LOG_DIR/paper_${CARDINALITY_TYPE}_${GROUP_RUN_TIME}_n${N_RUNS}_aggregate.log"

mkdir -p "$MODELS_OUT" "$LOG_DIR" "$SUMMARY_DIR"

cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
# Python hash randomization and cuBLAS must be configured before Python/CUDA start.
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# source polaris/bin/activate

# Keep this list in the same order as the configuration sections above.
COMMON_ARGS=(
    --wl_base_path "$WL_BASE"
    --out_base_path "$MODELS_OUT"
    --device "$CUDA_DEVICE"
    --seed "$SEED"
)

if [[ "${DETERMINISTIC,,}" == "true" ]]; then
    COMMON_ARGS+=(--deterministic)
fi

COMMON_ARGS+=(
    --model_config "$MODEL_CONFIG"
    --data_keyword "$DATA_KEYWORD"
    --database "$DATABASE"

    --card_type "$CARDINALITY_TYPE"
    --test_all_cardinality "$TEST_ALL_CARDINALITY"
    --test_against "$TEST_DB"

    --num_workers "$NUM_WORKERS"
    # --max_runtime 30
    # --stratify_per_database_by_runtimes True
    # --include_no_udf_data
    # --min_runtime_ms 50
)

if [[ "${INCLUDE_PULLUP_DATA,,}" == "true" ]]; then
    COMMON_ARGS+=(--include_pullup_data)
fi
if [[ "${INCLUDE_PUSHDOWN_DATA,,}" == "true" ]]; then
    COMMON_ARGS+=(--include_pushdown_data)
fi

if [[ -n "$PRETRAINED_MODEL_ARTIFACT_DIR" && -n "$PRETRAINED_MODEL_FILENAME" ]]; then
    COMMON_ARGS+=(
        --pretrained_model_artifact_dir "$PRETRAINED_MODEL_ARTIFACT_DIR"
        --pretrained_model_filename "$PRETRAINED_MODEL_FILENAME"
    )
fi

COMMON_ARGS+=(
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --activation_class_name "$ACTIVATION_CLASS_NAME"

    --augment "$AUGMENT"
    --test_augment "$TEST_AUGMENT"
    --augment_pooling "$AUGMENT_POOLING"
    --augment_refinement "$AUGMENT_REFINEMENT"
    --augment_coarse_layers "$AUGMENT_COARSE_LAYERS"
    --augment_include_inv "$AUGMENT_INCLUDE_INV"
    --augment_refine_ret "$AUGMENT_REFINE_RET"
    --lambda_struct "$LAMBDA_STRUCT"
)

tee_log() {
    tee -a "$LOG"
}

append_summary() {
    local exit_code="$1"

    if [[ -f "$SUMMARY_SCRIPT" ]]; then
        python "$SUMMARY_SCRIPT" "$LOG" \
            --exit-code "$exit_code" \
            --test-db "$TEST_DB" \
            --cardinality-type "$CARDINALITY_TYPE" \
            --graceful-dir "$SCRIPT_DIR" \
            --group-timestamp "$GROUP_RUN_TIME" \
            --requested-runs "$N_RUNS" \
            --run-index "$run_index" \
            --run-variable "DATASET_BASE=$DATASET_BASE" \
            --run-variable "WL_BASE=$WL_BASE" \
            --run-variable "MODELS_OUT=$MODELS_OUT" \
            --run-variable "N_RUNS=$N_RUNS" \
            --run-variable "DEVICE=$DEVICE" \
            --run-variable "CUDA_DEVICE=$CUDA_DEVICE" \
            --run-variable "SEED=$SEED" \
            --run-variable "DETERMINISTIC=$DETERMINISTIC" \
            --run-variable "MODEL_CONFIG=$MODEL_CONFIG" \
            --run-variable "DATA_KEYWORD=$DATA_KEYWORD" \
            --run-variable "DATABASE=$DATABASE" \
            --run-variable "TEST_DB=$TEST_DB" \
            --run-variable "CARDINALITY_TYPE=$CARDINALITY_TYPE" \
            --run-variable "TEST_ALL_CARDINALITY=$TEST_ALL_CARDINALITY" \
            --run-variable "NUM_WORKERS=$NUM_WORKERS" \
            --run-variable "INCLUDE_PULLUP_DATA=$INCLUDE_PULLUP_DATA" \
            --run-variable "INCLUDE_PUSHDOWN_DATA=$INCLUDE_PUSHDOWN_DATA" \
            --run-variable "EPOCHS=$EPOCHS" \
            --run-variable "BATCH_SIZE=$BATCH_SIZE" \
            --run-variable "ACTIVATION_CLASS_NAME=$ACTIVATION_CLASS_NAME" \
            --run-variable "AUGMENT=$AUGMENT" \
            --run-variable "TEST_AUGMENT=$TEST_AUGMENT" \
            --run-variable "AUGMENT_POOLING=$AUGMENT_POOLING" \
            --run-variable "AUGMENT_REFINEMENT=$AUGMENT_REFINEMENT" \
            --run-variable "AUGMENT_COARSE_LAYERS=$AUGMENT_COARSE_LAYERS" \
            --run-variable "AUGMENT_INCLUDE_INV=$AUGMENT_INCLUDE_INV" \
            --run-variable "AUGMENT_REFINE_RET=$AUGMENT_REFINE_RET" \
            --run-variable "LAMBDA_STRUCT=$LAMBDA_STRUCT" \
            --run-variable "GROUP_RUN_TIME=$GROUP_RUN_TIME" \
            --run-variable "PYTHONHASHSEED=$PYTHONHASHSEED" \
            --run-variable "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG" \
            --xlsx-path "$SUMMARY_XLSX" | tee_log
    else
        echo "Summary script missing: $SUMMARY_SCRIPT" | tee_log
    fi
}

if ! [[ "$N_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_RUNS must be a positive integer, got: $N_RUNS" >&2
    exit 2
fi

declare -a RUN_XLSXS=()
overall_exit_code=0

for ((run_index = 1; run_index <= N_RUNS; run_index++)); do
    printf -v run_number "%02d" "$run_index"
    LOG="$LOG_DIR/paper_${CARDINALITY_TYPE}_${GROUP_RUN_TIME}_run${run_number}.log"
    if (( N_RUNS == 1 )); then
        SUMMARY_XLSX="$SUMMARY_DIR/${SUMMARY_STEM}.xlsx"
    else
        SUMMARY_XLSX="$SUMMARY_DIR/${SUMMARY_STEM}_${run_number}.xlsx"
    fi

    echo "=== $(date) Starting experiment ===" | tee_log
    echo "Repeat run: $run_index/$N_RUNS" | tee_log
    echo "BASE DIRECTORY: $SCRIPT_DIR" | tee_log
    echo "Log: $LOG" | tee_log
    echo "Summary XLSX: $SUMMARY_XLSX" | tee_log
    echo "Work dir: $SCRIPT_DIR" | tee_log
    echo "Workload base: $WL_BASE" | tee_log
    echo "Models output: $MODELS_OUT" | tee_log

    echo "CUDA device: $CUDA_DEVICE" | tee_log
    echo "Seed: $SEED" | tee_log
    echo "Deterministic: $DETERMINISTIC" | tee_log
    echo "Python hash seed: $PYTHONHASHSEED" | tee_log

    echo "Model config: $MODEL_CONFIG" | tee_log
    echo "Data keyword: $DATA_KEYWORD" | tee_log
    echo "Database: $DATABASE" | tee_log

    echo "Test DB: $TEST_DB" | tee_log
    echo "Card type: $CARDINALITY_TYPE" | tee_log
    echo "Test all cardinalities: $TEST_ALL_CARDINALITY" | tee_log

    echo "Number of workers: $NUM_WORKERS" | tee_log
    echo "Include pull-up data: $INCLUDE_PULLUP_DATA" | tee_log
    echo "Include pushdown data: $INCLUDE_PUSHDOWN_DATA" | tee_log
    echo "Epochs: $EPOCHS" | tee_log
    echo "Batch size: $BATCH_SIZE" | tee_log
    echo "Activation: $ACTIVATION_CLASS_NAME" | tee_log
    if [[ -n "$PRETRAINED_MODEL_ARTIFACT_DIR" && -n "$PRETRAINED_MODEL_FILENAME" ]]; then
        echo "Resuming from: $PRETRAINED_MODEL_ARTIFACT_DIR/$PRETRAINED_MODEL_FILENAME" | tee_log
    fi

    echo "Augment: $AUGMENT" | tee_log
    echo "Test augment: $TEST_AUGMENT" | tee_log
    echo "Augment pooling: $AUGMENT_POOLING" | tee_log
    echo "Augment refinement: $AUGMENT_REFINEMENT" | tee_log
    echo "Augment coarse layers: $AUGMENT_COARSE_LAYERS" | tee_log
    echo "Augment include INV: $AUGMENT_INCLUDE_INV" | tee_log
    echo "Augment refine RET: $AUGMENT_REFINE_RET" | tee_log
    echo "Lambda struct: $LAMBDA_STRUCT" | tee_log
    echo "Protocol: $DATA_KEYWORD = train on 19 DBs, test on held-out database" | tee_log

    set +e
    python train.py "${COMMON_ARGS[@]}" 2>&1 | tee_log
    train_exit_code="${PIPESTATUS[0]}"
    set -e

    if [[ "$train_exit_code" -eq 0 ]]; then
        echo "=== $(date) DONE leave-out-$TEST_DB ===" | tee_log
    else
        echo "=== $(date) FAILED leave-out-$TEST_DB with exit code $train_exit_code ===" | tee_log
        overall_exit_code="$train_exit_code"
    fi

    if [[ -f "$LOSS_CURVE_SCRIPT" && "${SAVE_LOSS_PLOTS,,}" == "true" ]]; then
        last_checkpoint_line="$(grep "Saved checkpoint to" "$LOG" | tail -1)"
        checkpoint_pt_path="$(echo "$last_checkpoint_line" | sed -n "s/.*Saved checkpoint to \(.*\.pt\) in .*/\1/p")"
        if [[ -n "$checkpoint_pt_path" ]]; then
            stats_csv_path="${checkpoint_pt_path%.pt}.csv"
            if [[ -f "$stats_csv_path" ]]; then
                set +e
                python "$LOSS_CURVE_SCRIPT" \
                    --csv "$stats_csv_path" \
                    --test-db "$TEST_DB" \
                    --time-stamp "$GROUP_RUN_TIME" \
                    --output-dir "$AUGMENTED_PLOT_DIR" \
                    --seed "$SEED" \
                    --cardinality "$CARDINALITY_TYPE" \
                    --augment "$AUGMENT" \
                    --test-augment "$TEST_AUGMENT" \
                    --augment-pooling "$AUGMENT_POOLING" \
                    --augment-refinement "$AUGMENT_REFINEMENT" \
                    --augment-coarse-layers "$AUGMENT_COARSE_LAYERS" \
                    --augment-include-inv "$AUGMENT_INCLUDE_INV" \
                    --augment-refine-ret "$AUGMENT_REFINE_RET" \
                    --lambda-struct "$LAMBDA_STRUCT" | tee_log
                set -e
            else
                echo "Loss curve skipped: stats CSV not found at $stats_csv_path" | tee_log
            fi
        else
            echo "Loss curve skipped: no checkpoint line found in $LOG" | tee_log
        fi
    fi

    append_summary "$train_exit_code"
    RUN_XLSXS+=("$SUMMARY_XLSX")
done

if (( N_RUNS > 1 )); then
    if [[ -f "$REPEAT_SUMMARY_SCRIPT" ]]; then
        set +e
        python "$REPEAT_SUMMARY_SCRIPT" \
            --requested-runs "$N_RUNS" \
            --output-xlsx "$AGGREGATE_XLSX" \
            "${RUN_XLSXS[@]}" | tee "$AGGREGATE_LOG"
        aggregate_exit_code="${PIPESTATUS[0]}"
        set -e
        if [[ "$aggregate_exit_code" -ne 0 && "$overall_exit_code" -eq 0 ]]; then
            overall_exit_code="$aggregate_exit_code"
        fi
    else
        echo "Repeated-run summary script missing: $REPEAT_SUMMARY_SCRIPT" >&2
        overall_exit_code=1
    fi
fi

if [[ "${AUGMENT,,}" == "false" ]]; then
    if [[ -f "$BASELINE_RESULTS_SCRIPT" ]]; then
        set +e
        python "$BASELINE_RESULTS_SCRIPT" \
            --requested-runs "$N_RUNS" \
            --test-db "$TEST_DB" \
            --cardinality-type "$CARDINALITY_TYPE" \
            --time-stamp "$GROUP_RUN_TIME" \
            --epochs "$EPOCHS" \
            --output-xlsx "$BASELINE_RESULTS_XLSX" \
            "${RUN_XLSXS[@]}"
        baseline_exit_code="$?"
        set -e
        if [[ "$baseline_exit_code" -ne 0 && "$overall_exit_code" -eq 0 ]]; then
            overall_exit_code="$baseline_exit_code"
        fi
    else
        echo "Baseline results script missing: $BASELINE_RESULTS_SCRIPT" >&2
        overall_exit_code=1
    fi
fi

if [[ "${AUGMENT,,}" == "true" ]]; then
    if [[ -f "$AUGMENTED_RESULTS_SCRIPT" ]]; then
        set +e
        python "$AUGMENTED_RESULTS_SCRIPT" \
            --requested-runs "$N_RUNS" \
            --test-db "$TEST_DB" \
            --cardinality-type "$CARDINALITY_TYPE" \
            --time-stamp "$GROUP_RUN_TIME" \
            --epochs "$EPOCHS" \
            --baseline-xlsx "$BASELINE_RESULTS_XLSX" \
            --output-xlsx "$AUGMENTED_RESULTS_XLSX" \
            --test-augment "$TEST_AUGMENT" \
            --augment-pooling "$AUGMENT_POOLING" \
            --augment-refinement "$AUGMENT_REFINEMENT" \
            --augment-coarse-layers "$AUGMENT_COARSE_LAYERS" \
            --augment-include-inv "$AUGMENT_INCLUDE_INV" \
            --augment-refine-ret "$AUGMENT_REFINE_RET" \
            --lambda-struct "$LAMBDA_STRUCT" \
            "${RUN_XLSXS[@]}"
        augmented_results_exit_code="$?"
        set -e
        if [[ "$augmented_results_exit_code" -ne 0 && "$overall_exit_code" -eq 0 ]]; then
            overall_exit_code="$augmented_results_exit_code"
        fi
    else
        echo "Augmented results script missing: $AUGMENTED_RESULTS_SCRIPT" >&2
        overall_exit_code=1
    fi

    if [[ -f "$BASELINE_RESULTS_XLSX" && -f "$AUGMENTED_PLOT_SCRIPT" ]]; then
        set +e
        python "$AUGMENTED_PLOT_SCRIPT" \
            --requested-runs "$N_RUNS" \
            --test-db "$TEST_DB" \
            --cardinality "$CARDINALITY_TYPE" \
            --time-stamp "$GROUP_RUN_TIME" \
            --baseline-xlsx "$BASELINE_RESULTS_XLSX" \
            --output-dir "$AUGMENTED_PLOT_DIR" \
            --seed "$SEED" \
            --augment "$AUGMENT" \
            --test-augment "$TEST_AUGMENT" \
            --augment-pooling "$AUGMENT_POOLING" \
            --augment-refinement "$AUGMENT_REFINEMENT" \
            --augment-coarse-layers "$AUGMENT_COARSE_LAYERS" \
            --augment-include-inv "$AUGMENT_INCLUDE_INV" \
            --augment-refine-ret "$AUGMENT_REFINE_RET" \
            --lambda-struct "$LAMBDA_STRUCT" \
            "${RUN_XLSXS[@]}"
        plot_exit_code="$?"
        set -e
        if [[ "$plot_exit_code" -ne 0 && "$overall_exit_code" -eq 0 ]]; then
            overall_exit_code="$plot_exit_code"
        fi
    else
        echo "Augmented plot skipped: baseline workbook or plot script is missing."
    fi
fi

exit "$overall_exit_code"
