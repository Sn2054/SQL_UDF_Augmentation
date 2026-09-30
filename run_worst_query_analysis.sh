#!/usr/bin/env bash
# Edit these 6 lines for the run you want, then run:
#   GPU_UUID=GPU-... bash run_worst_query_analysis.sh
TEST_DB=basketball
CARD_TYPE=act
MODEL_DIR=saved/models/aug_act_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_basketball_bs512_ep100_maxr30_augpoolmax_augrefgated_residual_augcl1_augnoRET_cfl0.25
MODEL_NAME=aug_act_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_basketball_bs512_ep100_maxr30_augpoolmax_augrefgated_residual_augcl1_augnoRET_cfl0.25_20260909_185342_082
POOLING=max
COARSE_LAYERS=1

# GPU by UUID only (lab server manual sec. 3.3): pass GPU_UUID=GPU-... or run
# inside a Slurm GPU allocation. The assigned GPU is then local device cuda:0.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! [[ "${GPU_UUID:-}" =~ ^GPU-[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$ ]]; then
        echo "Set GPU_UUID to your assigned GPU's UUID (see \`nvidia-smi -L\`)." >&2
        exit 2
    fi
    export CUDA_VISIBLE_DEVICES="$GPU_UUID"
fi

# Same lookup order as run_code.sh; override with DATASET_BASE=/path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${DATASET_BASE:-}" ]]; then
    for candidate in "$SCRIPT_DIR/../data/Graceful_data" "/mnt/store5/ishana/data/Graceful_data"; do
        [[ -d "$candidate/workload_runs" ]] && { DATASET_BASE="$candidate"; break; }
    done
fi
WL_BASE="${DATASET_BASE:?Graceful_data not found, set DATASET_BASE}/workload_runs/"
cd "$SCRIPT_DIR"

python find_worst_queries.py --test_db $TEST_DB --card_type $CARD_TYPE --model_config ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge --model_dir $MODEL_DIR --model_name $MODEL_NAME --augment_pooling $POOLING --augment_coarse_layers $COARSE_LAYERS --wl_base_path $WL_BASE --device cuda:0 --top_n 50 --best_n 50

RUN_STAMP=$(grep -oE '[0-9]{8}_[0-9]{6}_[0-9]{3}$' <<<"$MODEL_NAME")
QUERIES_CSV=results/worst_queries/${TEST_DB}_${RUN_STAMP}_worst50_best50.csv
python analyze_refinement_similarity.py --test_db $TEST_DB --card_type $CARD_TYPE --model_config ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge --model_dir $MODEL_DIR --model_name $MODEL_NAME --augment_pooling $POOLING --augment_coarse_layers $COARSE_LAYERS --queries_csv $QUERIES_CSV --wl_base_path $WL_BASE --device cuda:0 && rm $QUERIES_CSV
