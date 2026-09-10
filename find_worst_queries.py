#!/usr/bin/env python3
"""Find the worst-predicted queries for a trained augmented cost-estimation model.

For a held-out test database, this runs inference with the *estimated*
(DuckDB) cardinalities -- the same card type used at test time by an
`aug_est_...` run produced by `run_code.sh` -- through the augmented
("this") model checkpoint, against the pullup and pushdown test workloads.
It then ranks every query by how wrong the model's prediction is (q-error
against the ground-truth actual, by default), and writes out the full
per-query table plus the worst N queries.

Pass --with_baseline (plus --baseline_model_dir) to additionally load the
matching non-augmented checkpoint and compare against it -- this adds
baseline_pred/baseline_qerror/qerror_delta columns, and enables
`--rank_by qerror_delta` (queries where augmentation hurt most relative to
baseline). Without --with_baseline, only the augmented model is loaded and
run.

This mirrors the inference machinery in `inference.py` /
`evaluate_pull_up_predictor.py` (same checkpoint loading, dataset creation,
and `run_inference` call), reduced to a per-query CSV instead of aggregate
q-error summaries.

Usage example (consumer, aug_est weighted_mean -- the run behind
results/augmented_plots/consumer_20260902_141126*.png), augmented model only:

    python find_worst_queries.py \\
        --test_db consumer \\
        --model_config ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge \\
        --model_dir saved/models/aug_est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_consumer_bs512_ep50_maxr30_augpoolweighted_mean_augrefgated_residual_augcl1_augnoRET \\
        --model_name aug_est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_consumer_bs512_ep50_maxr30_augpoolweighted_mean_augrefgated_residual_augcl1_augnoRET_20260902_141131_012 \\
        --augment_pooling weighted_mean --augment_refinement gated_residual --augment_coarse_layers 1 \\
        --device cuda:0 --top_n 10

Add the baseline comparison with:

    --with_baseline \\
    --baseline_model_dir saved/models/est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_consumer_bs512_ep50_maxr30

`--model_name` / `--baseline_model_name` may be omitted to auto-pick the most
recently modified checkpoint in the corresponding model dir.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/polaris_matplotlib")

import numpy as np
import pandas as pd

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.utils import load_json
from models.dataset.dataset_creation import create_dataloader, create_datasets, derive_label_normalizer, \
    read_workload_runs
from models.training.checkpoint import load_checkpoint
from models.training.metrics import MAPE, QError, RMSE
from models.training.train import run_inference
from models.training.utils import find_early_stopping_metric
from models.zero_shot_models.specific_models.model import zero_shot_models
from utils.hyperparams_utils import get_config

DATABASE = DatabaseSystem.DUCKDB
# order matches compile_train_test_filenames() in utils/hyperparams_utils.py:
# pushdown data is appended before pullup data.
WORKLOADS = ("pushdown", "pullup")
MIN_VAL = 0.01  # matches models/training/metrics.py::QError


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got {value}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--test_db', required=True, help='Held-out test database, e.g. consumer')
    parser.add_argument('--wl_base_path', default='/mnt/shared/data/dataset/Graceful_data/workload_runs/',
                        help='Root of duckdb_pushdown / duckdb_pullup workload runs (matches WL_BASE in run_code.sh)')
    parser.add_argument('--pushdown_plans_path', default=None,
                        help='Override: defaults to <wl_base_path>/duckdb_pushdown/parsed_plans/<test_db>/workload.json')
    parser.add_argument('--pullup_plans_path', default=None,
                        help='Override: defaults to <wl_base_path>/duckdb_pullup/parsed_plans/<test_db>/workload.json')
    parser.add_argument('--statistics_file', default=None,
                        help='Override: defaults to <wl_base_path>/duckdb_pushdown/parsed_plans/statistics_workload_combined.json')
    parser.add_argument('--data_keyword', default='complex_dd')
    parser.add_argument('--card_type', default='est', choices=['est', 'act', 'dd', 'wj'],
                        help='Cardinality type fed to both models at test time (default: est, matching an aug_est run)')
    parser.add_argument('--card_est_udf_sel', type=int, default=None,
                        help='Selectivity override (%%) for est-cardinality UDF filters; default: the actual DuckDB estimate')
    parser.add_argument('--max_runtime', type=int, default=30, help='Matches --max_runtime default in train.py')
    parser.add_argument('--min_runtime_ms', type=int, default=100, help='Matches the base min_runtime_ms default')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    ###
    # Augmented ("this") model
    ###
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--model_name', default=None, help='Omit to auto-pick the newest .pt in --model_dir')
    parser.add_argument('--model_config', required=True,
                        help='e.g. ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge')
    parser.add_argument('--mp_ignore_udf', type=str2bool, default=None)
    parser.add_argument('--work_with_udf_repr', default=False, action='store_true')
    parser.add_argument('--test_augment', type=str2bool, default=True)
    parser.add_argument('--augment_pooling', default='attention',
                        choices=['mean', 'sum', 'max', 'weighted_mean', 'attention', 'hybrid'])
    parser.add_argument('--augment_refinement', default='gated_residual', choices=['residual_sum', 'gated_residual'])
    parser.add_argument('--augment_coarse_layers', type=int, default=1)
    parser.add_argument('--augment_include_inv', type=str2bool, default=False)
    parser.add_argument('--augment_refine_ret', type=str2bool, default=True)

    ###
    # Baseline (non-augmented) model -- optional, off by default
    ###
    parser.add_argument('--with_baseline', action='store_true',
                        help='Also load --baseline_model_dir and compare against it (adds baseline_pred/'
                             'baseline_qerror/qerror_delta columns). Off by default: only the augmented model runs.')
    parser.add_argument('--baseline_model_dir', default=None, help='Required when --with_baseline is set')
    parser.add_argument('--baseline_model_name', default=None,
                        help='Omit to auto-pick the newest .pt in --baseline_model_dir')
    parser.add_argument('--baseline_model_config', default=None,
                        help='Defaults to --model_config (baseline architecture keywords are usually identical)')

    parser.add_argument('--top_n', type=int, default=50, help='Number of worst queries to keep')
    parser.add_argument('--best_n', type=int, default=50,
                        help='Number of best (lowest-qerror) queries to append to the same output CSV, '
                             'tagged via the "rank_group" column. 0 disables.')
    parser.add_argument('--rank_by', default='aug_qerror', choices=['aug_qerror', 'qerror_delta'],
                        help='aug_qerror: worst absolute predictions from the augmented model. '
                             'qerror_delta: queries where augmentation hurt most relative to baseline '
                             '(requires --with_baseline).')
    parser.add_argument('--output_dir', default='results/worst_queries')
    args = parser.parse_args()

    if args.with_baseline and not args.baseline_model_dir:
        parser.error('--baseline_model_dir is required when --with_baseline is set')
    if args.rank_by == 'qerror_delta' and not args.with_baseline:
        parser.error('--rank_by qerror_delta requires --with_baseline')
    return args


def resolve_model_name(model_dir: str, model_name: Optional[str]) -> str:
    if model_name:
        return model_name
    checkpoints = sorted(glob.glob(os.path.join(model_dir, '*.pt')), key=os.path.getmtime)
    if not checkpoints:
        raise SystemExit(f'No .pt checkpoint found in {model_dir}')
    chosen = os.path.basename(checkpoints[-1])[:-len('.pt')]
    if len(checkpoints) > 1:
        print(f'NOTE: {len(checkpoints)} checkpoints found in {model_dir}; using the most recently modified: {chosen}')
    return chosen


def build_config(model_config: str, data_keyword: str, mp_ignore_udf: Optional[bool], work_with_udf_repr: bool,
                 augment: bool, test_augment: bool, augment_pooling: str, augment_refinement: str,
                 augment_coarse_layers: int, augment_include_inv: bool, augment_refine_ret: bool) -> dict:
    args_config = {'model_config': model_config, 'data_keyword': data_keyword}
    if mp_ignore_udf is not None:
        args_config['mp_ignore_udf'] = mp_ignore_udf
    if work_with_udf_repr:
        args_config['work_with_udf_repr'] = work_with_udf_repr
    if augment:
        args_config['augment'] = augment
        args_config['test_augment'] = test_augment
        args_config['augment_pooling'] = augment_pooling
        args_config['augment_refinement'] = augment_refinement
        args_config['augment_coarse_layers'] = augment_coarse_layers
        args_config['augment_include_inv'] = augment_include_inv
        args_config['augment_refine_ret'] = augment_refine_ret
    config, _, _, _, _ = get_config(args_config, wl_base_path='', assemble_filenames=False)
    return config


def load_model(model_dir: str, model_name: str, config: dict, feature_statistics: dict, device: str):
    label_norm = derive_label_normalizer('QLoss', np.asarray([1, 10, 10]))
    model = zero_shot_models[DATABASE](
        device=device, final_mlp_kwargs=config['final_mlp_kwargs'], node_type_kwargs=config['node_type_kwargs'],
        output_dim=1, feature_statistics=feature_statistics, tree_layer_kwargs=config['tree_layer_kwargs'],
        featurization=config['featurization'], label_norm=label_norm, mp_ignore_udf=config['mp_ignore_udf'],
        return_graph_repr=True, return_udf_repr=True, plans_have_no_udf=False,
        train_udf_graph_against_udf_runtime=False, work_with_udf_repr=config['work_with_udf_repr'],
        test_with_count_edges_msg_aggr=False, augment=config['augment'], augment_pooling=config['augment_pooling'],
        augment_refinement=config['augment_refinement'], augment_coarse_layers=config['augment_coarse_layers'],
        augment_include_inv=config['augment_include_inv'], augment_refine_ret=config['augment_refine_ret'])
    model = model.to(model.device)

    metrics = [RMSE(), MAPE(), QError(percentile=50, early_stopping_metric=True), QError(percentile=95),
               QError(percentile=99), QError(percentile=100)]
    checkpoint_map_location = {'cuda:1': device, 'cuda:0': device, 'cpu': device}
    _, _, _, model, _, _, metrics, _ = load_checkpoint(
        model, model_dir, model_name, optimizer=None, lr_scheduler=None, metrics=metrics, filetype='.pt',
        zs_paper_model=False, map_location=checkpoint_map_location)

    best_model_state = find_early_stopping_metric(metrics).best_model
    model.load_state_dict(best_model_state)
    model.set_augmentation_enabled(config['augment'] and config['test_augment'])
    return model


def run_query_inference(model, config: dict, plans_path: str, statistics_file: str, feature_statistics: dict,
                        card_type: str, card_est_udf_sel: Optional[int], num_workers: int,
                        plans_override: Optional[list] = None):
    """plans_override lets a caller isolate a single already-matched plan (see
    analyze_refinement_similarity.py) so exactly one query's graph passes through the model per call --
    needed to unambiguously attribute a forward hook's captured node embeddings to one query."""
    plans, dataset_stats = read_workload_runs([plans_path], min_runtime_ms=config['min_runtime_ms'],
                                              max_runtime=config['max_runtime'])
    if plans_override is not None:
        plans = plans_override
    _, dataset, _, _, _, database_statistics = create_datasets(
        None, loss_class_name=None, val_ratio=0, shuffle_before_split=False, stratify_dataset_by_runtimes=False,
        max_runtime=config['max_runtime'], zs_paper_dataset=False, train_udf_graph_against_udf_runtime=False,
        min_runtime_ms=config['min_runtime_ms'], infuse_plans=plans, infuse_database_statistics=dataset_stats)

    create_dataset_fn_test_artefacts = {plans_path: (_, dataset, _, database_statistics)}

    _, _, _, _, _, _, data_loaders, _, _ = create_dataloader(
        workload_run_paths=[], test_workload_run_paths=[plans_path], statistics_file=statistics_file,
        database=DATABASE, val_ratio=0.15, finetune_ratio=0.0, batch_size=config['batch_size'], shuffle=False,
        num_workers=num_workers, pin_memory=False, limit_queries=False, limit_queries_affected_wl=None,
        loss_class_name=config['final_mlp_kwargs']['loss_class_name'], offset_np_import=config['offset_np_import'],
        stratify_dataset_by_runtimes=config['stratify_dataset_by_runtimes'],
        stratify_per_database_by_runtimes=config['stratify_per_database_by_runtimes'],
        max_runtime=config['max_runtime'], multi_label_keep_duplicates=config['multi_label_keep_duplicates'],
        zs_paper_dataset=config['zs_paper_dataset'],
        train_udf_graph_against_udf_runtime=config['train_udf_graph_against_udf_runtime'],
        w_loop_end_node=config['w_loop_end_node'], add_loop_loopend_edge=config['add_loop_loopend_edge'],
        card_est_assume_lazy_eval=config['card_est_assume_lazy_eval'], min_runtime_ms=config['min_runtime_ms'],
        create_dataset_fn_test_artefacts=create_dataset_fn_test_artefacts,
        separate_sql_udf_graphs=config['separate_sql_udf_graphs'],
        annotate_flat_vector_udf_preds=config['flat_vector_udf_est'],
        flat_vector_model_path=config.get('flat_vector_model_path'), featurization=config['featurization'],
        est_card_udf_sel=card_est_udf_sel, feature_statistics=feature_statistics, card_type_below_udf=card_type,
        card_type_in_udf=card_type, card_type_above_udf=card_type)

    if len(data_loaders) == 0:
        raise SystemExit(f'No test data loaded for {plans_path} (card_type={card_type})')

    labels, preds, _, _, _, _, _, _, stats, _ = run_inference(
        data_loaders[0], model, 100000, pt_profile=False,
        separate_sql_udf_graphs=config['separate_sql_udf_graphs'], flat_vector_udf_est=config['flat_vector_udf_est'])
    return labels, preds, stats


def q_error(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    preds = np.clip(preds, MIN_VAL, np.inf)
    errors = np.maximum(labels / preds, preds / labels)
    return np.nan_to_num(errors, nan=np.inf)


def safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_") or "unknown"


UDF_NAME_PATTERN = re.compile(r'func_\d+')
NO_UDF_METRICS = {
    'num_udfs': 0, 'udf_num_loops_src': 0, 'udf_num_branches_src': 0, 'udf_has_nested_loop': False,
    'udf_has_nested_branch': False, 'udf_max_nesting_depth': 0, 'udf_num_code_lines': 0,
}


def load_udf_source(plans_path: str, test_db: str) -> dict:
    """UDF Python source lives alongside the parsed plans at dbs/<db>/sql_scripts/udfs.json
    (three directories up from .../parsed_plans/<db>/workload.json), keyed by udf_name -- see
    pull_push_advisor/utils.py's col_col_stats_path for the same directory-layout convention."""
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(plans_path)))
    udfs_path = os.path.join(base_path, 'dbs', test_db, 'sql_scripts', 'udfs.json')
    if not os.path.exists(udfs_path):
        print(f'NOTE: no udfs.json found at {udfs_path}; UDF nesting/complexity columns will be blank.')
        return {}
    with open(udfs_path) as f:
        return json.load(f)


def analyze_udf_code(code_lines) -> dict:
    """Flat udf_num_loops/udf_num_branches (line-prefix counts) are already tracked elsewhere in
    this repo (see extract_udf_stats in cross_db_benchmark/benchmark_tools/dbms/parse_dd_plan.py),
    but nesting depth is not. Walk the tab-indented source (a stack of open for/while/if blocks by
    indent level) to find loops nested inside loops, branches nested inside branches, and how deep
    the UDF's control flow goes -- cheap proxies for "this UDF is legitimately hard to cost"."""
    stack = []
    max_depth = 0
    has_nested_loop = False
    has_nested_branch = False
    num_loops = 0
    num_branches = 0
    num_code_lines = 0
    for line in code_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(('def ', 'return', 'db_conn.create_function')):
            continue
        num_code_lines += 1
        depth = len(line) - len(line.lstrip('\t'))
        while len(stack) > depth:
            stack.pop()
        if stripped.startswith('for ') or stripped.startswith('while '):
            num_loops += 1
            has_nested_loop = has_nested_loop or 'loop' in stack
            max_depth = max(max_depth, depth)
            stack.append('loop')
        elif stripped.startswith('if '):
            num_branches += 1
            has_nested_branch = has_nested_branch or 'branch' in stack
            max_depth = max(max_depth, depth)
            stack.append('branch')
        elif stripped.startswith('elif ') or stripped.startswith('else'):
            # continues the enclosing if/elif at the same depth, reopening it for this arm's body
            stack.append('branch')
    return {
        'num_udfs': 1, 'udf_num_loops_src': num_loops, 'udf_num_branches_src': num_branches,
        'udf_has_nested_loop': has_nested_loop, 'udf_has_nested_branch': has_nested_branch,
        'udf_max_nesting_depth': max_depth, 'udf_num_code_lines': num_code_lines,
    }


def annotate_udf_metrics(results: pd.DataFrame, plans_paths: dict, test_db: str) -> pd.DataFrame:
    source_by_workload = {}
    analysis_cache = {}
    records = []
    for workload, sql in zip(results['workload'], results['sql']):
        match = UDF_NAME_PATTERN.search(sql)
        if match is None:
            records.append(NO_UDF_METRICS)
            continue

        udf_name = match.group(0)
        cache_key = (workload, udf_name)
        if cache_key not in analysis_cache:
            if workload not in source_by_workload:
                source_by_workload[workload] = load_udf_source(plans_paths[workload], test_db)
            code = source_by_workload[workload].get(udf_name)
            analysis_cache[cache_key] = NO_UDF_METRICS if code is None else analyze_udf_code(code)
        records.append(analysis_cache[cache_key])

    return pd.concat([results.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def main() -> int:
    args = parse_args()

    pushdown_plans_path = args.pushdown_plans_path or os.path.join(
        args.wl_base_path, 'duckdb_pushdown', 'parsed_plans', args.test_db, 'workload.json')
    pullup_plans_path = args.pullup_plans_path or os.path.join(
        args.wl_base_path, 'duckdb_pullup', 'parsed_plans', args.test_db, 'workload.json')
    statistics_file = args.statistics_file or os.path.join(
        args.wl_base_path, 'duckdb_pushdown', 'parsed_plans', 'statistics_workload_combined.json')
    plans_paths = {'pushdown': pushdown_plans_path, 'pullup': pullup_plans_path}

    feature_statistics = load_json(statistics_file, namespace=False)
    feature_statistics['on_udf'] = {"value_dict": {"True": 0, "False": 1}, "no_vals": 2, "type": "categorical"}

    aug_config = build_config(
        args.model_config, args.data_keyword, args.mp_ignore_udf, args.work_with_udf_repr, augment=True,
        test_augment=args.test_augment, augment_pooling=args.augment_pooling,
        augment_refinement=args.augment_refinement, augment_coarse_layers=args.augment_coarse_layers,
        augment_include_inv=args.augment_include_inv, augment_refine_ret=args.augment_refine_ret)
    aug_config['max_runtime'] = args.max_runtime
    aug_config['min_runtime_ms'] = args.min_runtime_ms

    baseline_config = None
    if args.with_baseline:
        baseline_config = build_config(
            args.baseline_model_config or args.model_config, args.data_keyword, args.mp_ignore_udf,
            args.work_with_udf_repr, augment=False, test_augment=False, augment_pooling=args.augment_pooling,
            augment_refinement=args.augment_refinement, augment_coarse_layers=args.augment_coarse_layers,
            augment_include_inv=args.augment_include_inv, augment_refine_ret=args.augment_refine_ret)
        baseline_config['max_runtime'] = args.max_runtime
        baseline_config['min_runtime_ms'] = args.min_runtime_ms

    model_name = resolve_model_name(args.model_dir, args.model_name)

    print(f'Loading augmented model: {args.model_dir}/{model_name}')
    aug_model = load_model(args.model_dir, model_name, aug_config, feature_statistics, args.device)

    baseline_model = None
    if args.with_baseline:
        baseline_model_name = resolve_model_name(args.baseline_model_dir, args.baseline_model_name)
        print(f'Loading baseline model: {args.baseline_model_dir}/{baseline_model_name}')
        baseline_model = load_model(args.baseline_model_dir, baseline_model_name, baseline_config,
                                    feature_statistics, args.device)

    frames = []
    for workload in WORKLOADS:
        plans_path = plans_paths[workload]
        if not os.path.exists(plans_path):
            print(f'Skipping {workload}: no plans file at {plans_path}')
            continue

        print(f'Running inference on {workload} ({plans_path}) ...')
        aug_labels, aug_preds, stats = run_query_inference(
            aug_model, aug_config, plans_path, statistics_file, feature_statistics, args.card_type,
            args.card_est_udf_sel, args.num_workers)

        n = len(aug_labels)
        actual = aug_labels[:n]
        row_data = {
            'workload': workload,
            'sql': stats['sql'][:n],
            'database_name': stats['database_name'][:n],
            'udf_pos_in_query': stats['udf_pos_in_query'][:n],
            'num_joins': stats['num_joins'][:n],
            'num_filters': stats['num_filters'][:n],
            'udf_filter_num_logicals': stats['udf_filter_num_logicals'][:n],
            'udf_filter_num_literals': stats['udf_filter_num_literals'][:n],
            'udf_in_card': stats['udf_in_card'][:n],
            'udf_num_loops': stats['udf_num_loops'][:n],
            'udf_num_branches': stats['udf_num_branches'][:n],
            'udf_num_np_calls': stats['udf_num_np_calls'][:n],
            'udf_num_math_calls': stats['udf_num_math_calls'][:n],
            'udf_num_comp_nodes': stats['udf_num_comp_nodes'][:n],
            'actual': actual,
            'predicted_est': aug_preds[:n],
            'aug_qerror': q_error(actual, aug_preds[:n]),
        }

        if args.with_baseline:
            baseline_labels, baseline_preds, _ = run_query_inference(
                baseline_model, baseline_config, plans_path, statistics_file, feature_statistics, args.card_type,
                args.card_est_udf_sel, args.num_workers)

            if len(aug_labels) != len(baseline_labels) or not np.allclose(aug_labels, baseline_labels):
                print(f'WARNING: {workload} label mismatch between augmented and baseline runs '
                      f'({len(aug_labels)} vs {len(baseline_labels)} queries) -- results may be misaligned.')

            n = min(n, len(baseline_labels))
            row_data = {key: value[:n] for key, value in row_data.items() if key != 'workload'}
            row_data['workload'] = workload
            baseline_qerr = q_error(actual[:n], baseline_preds[:n])
            row_data['baseline_pred'] = baseline_preds[:n]
            row_data['baseline_qerror'] = baseline_qerr
            row_data['qerror_delta'] = row_data['aug_qerror'] - baseline_qerr

        frames.append(pd.DataFrame(row_data))

    if not frames:
        raise SystemExit('No workloads produced results; nothing to rank.')

    results = pd.concat(frames, ignore_index=True)
    print('Annotating UDF nesting/complexity metrics from source ...')
    results = annotate_udf_metrics(results, plans_paths, args.test_db)
    results.sort_values(args.rank_by, ascending=False, inplace=True)

    worst = results.head(args.top_n).copy()
    worst['rank_group'] = 'worst'
    if args.best_n > 0:
        best = results.tail(args.best_n).iloc[::-1].copy()
        best['rank_group'] = 'best'
        worst_and_best = pd.concat([worst, best], ignore_index=True)
    else:
        worst_and_best = worst

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{safe_filename_part(args.test_db)}_{safe_filename_part(model_name)}'
    full_path = output_dir / f'{stem}_all_queries.csv'
    ranked_stem = f'{stem}_worst{args.top_n}'
    if args.best_n > 0:
        ranked_stem += f'_best{args.best_n}'
    worst_path = output_dir / f'{ranked_stem}.csv'
    results.to_csv(full_path, index=False)
    worst_and_best.to_csv(worst_path, index=False)

    print(f'\nWrote {len(results)} queries to {full_path}')
    print(f'Wrote worst {len(worst)} + best {len(worst_and_best) - len(worst)} queries '
          f'(ranked by {args.rank_by}) to {worst_path}\n')

    display_cols = ['rank_group', 'workload', 'actual', 'predicted_est', 'aug_qerror']
    if args.with_baseline:
        display_cols += ['baseline_pred', 'baseline_qerror', 'qerror_delta']
    display_cols += ['num_udfs', 'udf_num_loops', 'udf_has_nested_loop', 'udf_num_branches',
                     'udf_has_nested_branch', 'udf_max_nesting_depth']
    display_cols.append('sql')
    with pd.option_context('display.max_colwidth', 60, 'display.width', 200):
        print(worst_and_best[display_cols].to_string(index=False))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
