#!/usr/bin/env python3
"""Per-query profile for the UDF time vs SQL time experiment (2026-11-03): DuckDB timings, executed plans,
UDF graphs and the q-error of the baseline and the augmented model, query by query.

For each selected test query of a database, in both workloads (push-down, pull-up):
  1. Run it in DuckDB 0.10.1, one query at a time, with the JSON profiler on (EXPLAIN ANALYZE, --repetitions
     times), on GRACEFUL's database file opened read-only and with GRACEFUL's UDF definitions registered (as
     cross_db_benchmark/benchmark_tools/dbms/run_workload.py does, including `filter_pushdown` disabled for
     pull-up). Recorded per query: the total query time, the time of the plan operator(s) that call the UDF,
     SQL time = total - UDF, and the executed plan. With --threads 1 (default) operator times are real
     seconds, so UDF time <= total time. With more threads DuckDB adds operator time up across threads, so the
     UDF time is thread-seconds and can exceed the total.
  2. Save the executed plan as a networkx tree (`plans/<id>.gpickle`, edges parent -> child, one node per
     operator) and a drawing (`plans/<id>.png`, the UDF operator in orange). Copy the query's UDF graph, GRACEFUL's
     `<udf>.loopend.gpickle` (the file the model reads), to `udf_graphs/<id>.gpickle` with a drawing.
  3. Predict the runtime with the baseline model and with the augmented model. Both were trained with this
     database held out. The q-error is computed against GRACEFUL's recorded runtime, which is the label the models
     were trained on (measured on GRACEFUL's machine, so it is not the time measured here).

Output, under --output_dir/<db>/:
  - <db>_baseline.xlsx and <db>_augmented.xlsx: the same DuckDB columns, each with its own model's prediction
    and q-error; one sheet per workload plus an `about` sheet;
  - plans/ and udf_graphs/.

CPU only (lab rule: Claude never touches a GPU). Run with CUDA_VISIBLE_DEVICES="":
    CUDA_VISIBLE_DEVICES="" python profile_udf_queries.py --n_queries 100     # 100 per database per workload
    CUDA_VISIBLE_DEVICES="" python profile_udf_queries.py --n_queries 0       # every test query
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import math
import os
import random
import re
import shutil
import statistics
import threading
import time
from types import SimpleNamespace

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('Run with CUDA_VISIBLE_DEVICES="" (CPU only).')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/polaris_matplotlib')

import duckdb
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy
import pandas as pd

import find_worst_queries as fwq
from cross_db_benchmark.benchmark_tools.utils import load_json
from models.dataset.dataset_creation import read_workload_runs
from utils.hyperparams_utils import get_config

GRACEFUL = '/mnt/data/shared/Graceful_data'
WL_BASE = f'{GRACEFUL}/workload_runs'
STATISTICS_FILE = f'{WL_BASE}/duckdb_pushdown/parsed_plans/statistics_workload_combined.json'
WORKLOADS = {'pushdown': 'duckdb_pushdown', 'pullup': 'duckdb_pullup'}

# The checkpoints behind the result-sheet rows; the run log holds each run's exact training config.
MODELS = {
    'employee': {
        'baseline': ('saved/models/est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_'
                     'employee_bs512_ep100_maxr30',
                     'est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_employee_'
                     'bs512_ep100_maxr30_20261015_021247_014',
                     'saved/training_logs/paper_est_20261015_021244_3186075_run01.log'),
        'augmented': ('saved/models/aug_est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_'
                      'loopedge_employee_bs512_ep100_maxr30_augpoolhybrid_augrefgated_residual_augcl1_augnoRET_augseq_'
                      'augcfge_cfl0.1_actSELU',
                      'aug_est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_employee_'
                      'bs512_ep100_maxr30_augpoolhybrid_augrefgated_residual_augcl1_augnoRET_augseq_augcfge_cfl0.1_'
                      'actSELU_20261026_202734_018',
                      'saved/training_logs/paper_est_20261026_202730_523164_run01.log'),
    },
    'financial': {
        'baseline': ('saved/models/est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_'
                     'financial_bs512_ep100_maxr30',
                     'est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_financial_'
                     'bs512_ep100_maxr30_20261015_070416_058',
                     'saved/training_logs/paper_est_20261015_070413_3374010_run01.log'),
        'augmented': ('saved/models/aug_est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_'
                      'loopedge_financial_bs512_ep100_maxr30_augpoolhybrid_augrefgated_residual_augcl1_augnoRET_augseq_'
                      'augcfge_cfl0.1_actSELU',
                      'aug_est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_financial_'
                      'bs512_ep100_maxr30_augpoolhybrid_augrefgated_residual_augcl1_augnoRET_augseq_augcfge_cfl0.1_'
                      'actSELU_20261027_213157_021',
                      'saved/training_logs/paper_est_20261027_213153_2064398_run01.log'),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n_queries', type=int, default=100,
                        help='queries per database per workload; 0 = every test query')
    parser.add_argument('--dbs', nargs='+', default=['financial', 'employee'], choices=sorted(MODELS))
    parser.add_argument('--workloads', nargs='+', default=list(WORKLOADS), choices=list(WORKLOADS))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--timeout_s', type=float, default=300.0, help='per execution; the query is interrupted after')
    parser.add_argument('--num_workers', type=int, default=2, help='data loader workers for model inference')
    parser.add_argument('--output_dir', default='results/udf_query_profile')
    return parser.parse_args()


# ---------------------------------------------------------------- models

def model_config(run_log: str) -> dict:
    """The exact training config of a run, rebuilt from its log's `Running with config:` line."""
    text = open(run_log, errors='replace').read()
    args = ast.literal_eval(re.search(r'Running with config: (\{.*\})', text).group(1))
    for key in ('test_against', 'include_pullup_data', 'include_pushdown_data'):
        args.pop(key, None)
    config, *_ = get_config(args, wl_base_path='', assemble_filenames=False)
    config['max_runtime'] = args.get('max_runtime', 30)
    return config


def load_model(model_dir: str, model_name: str, config: dict, feature_statistics: dict):
    """find_worst_queries.load_model, plus the augmentation options it lacks (SEQ regions, CFG coarse edges,
    multi-query count), built the way models/training/train.py builds the model."""
    label_norm = fwq.derive_label_normalizer('QLoss', numpy.asarray([1, 10, 10]))
    model = fwq.zero_shot_models[fwq.DATABASE](
        device='cpu', final_mlp_kwargs=config['final_mlp_kwargs'], node_type_kwargs=config['node_type_kwargs'],
        output_dim=1, feature_statistics=feature_statistics, tree_layer_kwargs=config['tree_layer_kwargs'],
        featurization=config['featurization'], label_norm=label_norm, mp_ignore_udf=config['mp_ignore_udf'],
        return_graph_repr=True, return_udf_repr=True, plans_have_no_udf=False,
        train_udf_graph_against_udf_runtime=False, work_with_udf_repr=config['work_with_udf_repr'],
        test_with_count_edges_msg_aggr=False, augment=config['augment'], augment_pooling=config['augment_pooling'],
        augment_refinement=config['augment_refinement'], augment_coarse_layers=config['augment_coarse_layers'],
        augment_include_inv=config['augment_include_inv'], augment_refine_ret=config['augment_refine_ret'],
        augment_seq_regions=config['augment_seq_regions'],
        augment_cfg_coarse_edges=config['augment_cfg_coarse_edges'],
        augment_mq_queries=config['augment_mq_queries'])
    metrics = [fwq.RMSE(), fwq.MAPE(), fwq.QError(percentile=50, early_stopping_metric=True),
               fwq.QError(percentile=95), fwq.QError(percentile=99), fwq.QError(percentile=100)]
    _, _, _, model, _, _, metrics, _ = fwq.load_checkpoint(
        model, model_dir, model_name, optimizer=None, lr_scheduler=None, metrics=metrics, filetype='.pt',
        zs_paper_model=False, map_location={'cuda:1': 'cpu', 'cuda:0': 'cpu', 'cpu': 'cpu'})
    model.load_state_dict(fwq.find_early_stopping_metric(metrics).best_model)
    model.set_augmentation_enabled(config['augment'] and config.get('test_augment', True))
    model.eval()
    return model


def predict(db: str, kind: str, plans_path: str, selected: list, feature_statistics: dict, num_workers: int):
    """{sql: (label_s, predicted_s, stats row)} for the selected plans, from one model."""
    model_dir, model_name, run_log = MODELS[db][kind]
    config = model_config(run_log)
    model = load_model(model_dir, model_name, config, feature_statistics)
    labels, preds, stats = fwq.run_query_inference(model, config, plans_path, STATISTICS_FILE, feature_statistics,
                                                   'est', None, num_workers, plans_override=selected)
    keys = ('udf_pos_in_query', 'num_joins', 'num_filters', 'udf_in_card', 'udf_num_loops', 'udf_num_branches',
            'udf_num_np_calls', 'udf_num_math_calls', 'udf_num_comp_nodes')
    out = {}
    for i, sql in enumerate(stats['sql'][:len(labels)]):
        out[sql] = (float(labels[i]), float(preds[i]), {k: stats[k][i] for k in keys})
    return out, model_name


# ---------------------------------------------------------------- DuckDB

def connect(db: str, workload: str, threads: int):
    con = duckdb.connect(f'{GRACEFUL}/datasets/{db}_10_1.db', read_only=True)
    con.execute(f'SET threads TO {threads};')
    udf_code = open(f'{WL_BASE}/{WORKLOADS[workload]}/dbs/{db}/sql_scripts/udfs.sql').read()
    exec(udf_code, {'math': math, 'numpy': numpy, 'db_conn': con, 'date': datetime.date, 'time': datetime.time,
                    'VARCHAR': duckdb.typing.VARCHAR, 'INTEGER': duckdb.typing.INTEGER,
                    'DOUBLE': duckdb.typing.DOUBLE, 'TIMESTAMP': duckdb.typing.TIMESTAMP,
                    'DATE': duckdb.typing.DATE, 'TIME': duckdb.typing.TIME, 'BOOLEAN': duckdb.typing.BOOLEAN,
                    'HUGEINT': duckdb.typing.HUGEINT, 'FLOAT': duckdb.typing.FLOAT})
    con.execute("PRAGMA enable_profiling = 'json';")
    if workload == 'pullup':  # as run_workload.py does for pulled-up UDF filters
        con.execute("SET disabled_optimizers = 'filter_pushdown';")
    return con


def run_profiled(con, sql: str, timeout_s: float):
    """(plan json, wall seconds) of one EXPLAIN ANALYZE, or (None, None) on timeout / error."""
    timer = threading.Timer(timeout_s, con.interrupt)
    timer.start()
    start = time.perf_counter()
    try:
        rows = con.execute(f'EXPLAIN ANALYZE {sql}').fetchall()
        return json.loads(rows[0][1]), time.perf_counter() - start
    except Exception as e:  # interrupted (timeout) or a query error
        print(f'    execution failed: {type(e).__name__}: {str(e)[:120]}', flush=True)
        return None, None
    finally:
        timer.cancel()


def plan_nodes(plan: dict):
    """Pre-order list of (node, parent index) for the operator tree under the profiler's Query/EXPLAIN nodes."""
    out = []

    def walk(node, parent):
        out.append((node, parent))
        me = len(out) - 1
        for child in node.get('children', []):
            walk(child, me)

    roots = plan.get('children', [])
    while len(roots) == 1 and roots[0].get('name', '').strip() in ('RESULT_COLLECTOR', 'EXPLAIN_ANALYZE', 'Query'):
        roots = roots[0].get('children', [])  # profiler wrappers, not part of the query
    for root in roots:
        walk(root, None)
    return out


def udf_operator_time(plan: dict, udf_name: str):
    """Time of the operators whose expression calls the UDF, their names, and the rows entering them."""
    nodes = plan_nodes(plan)
    seconds, names, rows_in = 0.0, [], 0
    for node, _ in nodes:
        if re.search(rf'\b{udf_name}\(', str(node.get('extra_info', ''))):
            seconds += float(node.get('timing', 0.0))
            names.append(node.get('name', '').strip())
            rows_in += sum(int(c.get('cardinality', 0)) for c in node.get('children', []))
    return seconds, names, rows_in


def plan_graph(plan: dict, udf_name: str) -> nx.DiGraph:
    g = nx.DiGraph()
    for i, (node, parent) in enumerate(plan_nodes(plan)):
        info = str(node.get('extra_info', ''))
        est = re.search(r'EC:\s*(\d+)', info)
        g.add_node(i, operator=node.get('name', '').strip(), time_s=float(node.get('timing', 0.0)),
                   rows=int(node.get('cardinality', 0)), est_rows=int(est.group(1)) if est else None,
                   details=info.replace('[INFOSEPARATOR]', '|').strip(),
                   calls_udf=bool(re.search(rf'\b{udf_name}\(', info)))
        if parent is not None:
            g.add_edge(parent, i)
    g.graph.update(total_time_s=float(plan.get('result', 0.0)), udf=udf_name,
                   note='edges parent -> child; time_s = operator time, rows = actual output rows, '
                        'est_rows = DuckDB estimate (EC)')
    return g


def tree_positions(g: nx.DiGraph):
    """Top-down tree layout: leaves on consecutive x slots, parents centred over their children."""
    pos, next_x = {}, [0]

    def place(n, depth):
        children = list(g.successors(n))
        if not children:
            pos[n] = (next_x[0], -depth)
            next_x[0] += 1
        else:
            for c in children:
                place(c, depth + 1)
            pos[n] = (sum(pos[c][0] for c in children) / len(children), -depth)

    for root in [n for n in g.nodes if g.in_degree(n) == 0]:
        place(root, 0)
    return pos


def draw_plan(g: nx.DiGraph, path: str, title: str):
    pos = tree_positions(g)
    width = max(6, 2.6 * (max(x for x, _ in pos.values()) + 1))
    height = max(4, 1.5 * (1 - min(y for _, y in pos.values())) + 1)
    fig, ax = plt.subplots(figsize=(width, height))
    for u, v in g.edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='#98A6B3', zorder=1)
    for n, d in g.nodes(data=True):
        label = f"{d['operator']}\nrows {d['rows']:,}" + (f" (est {d['est_rows']:,})" if d['est_rows'] is not None else '') \
                + f"\n{d['time_s']:.3f} s"
        ax.text(pos[n][0], pos[n][1], label, ha='center', va='center', fontsize=8, zorder=2,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0DD' if d['calls_udf'] else 'white',
                          edgecolor='#9D1912' if d['calls_udf'] else '#D3DADE'))
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    ax.margins(0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


UDF_NODE_COLOURS = {'VAR': '#E9ECF4', 'INVOCATION': '#E9ECF4', 'COMP': 'white', 'BRANCH': '#FFF0DD',
                    'LOOP_HEAD': '#FFE0B2', 'LOOP_END': '#FFE0B2', 'RETURN': '#E4F2E7'}


def udf_node_label(n, d) -> str:
    ntype = str(d.get('type', '?'))
    extra = ''
    if ntype == 'VAR':
        extra = str(d.get('var_id', ''))
    elif ntype == 'BRANCH':
        extra = str(d.get('cmops', ''))
    elif ntype == 'LOOP_HEAD':
        extra = f"{d.get('loop_type', '')} x{d.get('no_iter', '?')}"
    elif ntype == 'COMP':
        ops = d.get('ops') or []
        extra = ','.join(map(str, ops))[:18] if ops else ''
    return f'{ntype} {n}' + (f'\n{extra}' if extra else '')


def draw_udf_graph(g: nx.DiGraph, path: str, title: str):
    """Top-down control flow from the INVOCATION node (BFS depth, back edges ignored); the input VAR nodes sit in
    the top row and their data edges are drawn faintly, so the control structure stays readable."""
    var_nodes = [n for n, d in g.nodes(data=True) if d.get('type') == 'VAR']
    control = g.subgraph([n for n in g.nodes if n not in var_nodes])
    starts = [n for n, d in control.nodes(data=True) if d.get('type') == 'INVOCATION'] or \
             [n for n in control.nodes if control.in_degree(n) == 0] or list(control.nodes)[:1]
    depth = {s: 1 for s in starts}
    queue = list(starts)
    while queue:
        n = queue.pop(0)
        for m in sorted(control.successors(n)):
            if m not in depth:
                depth[m] = depth[n] + 1
                queue.append(m)
    for n in control.nodes:
        depth.setdefault(n, max(depth.values(), default=0) + 1)
    for n in var_nodes:
        depth[n] = 0
    layers = {}
    for n in sorted(depth, key=lambda x: (depth[x], x)):
        layers.setdefault(depth[n], []).append(n)
    width_slots = max(len(v) for v in layers.values())
    pos = {n: (i - (len(ns) - 1) / 2, -d) for d, ns in layers.items() for i, n in enumerate(ns)}
    fig, ax = plt.subplots(figsize=(max(6, 2.0 * width_slots + 2), max(4, 0.75 * len(layers) + 1.5)))
    for u, v in g.edges:
        if u in var_nodes or v in var_nodes:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='#D3DADE', lw=0.6, ls=':', zorder=0)
        else:
            ax.annotate('', xy=pos[v], xytext=pos[u], zorder=1,
                        arrowprops=dict(arrowstyle='->', color='#98A6B3', lw=1,
                                        connectionstyle='arc3,rad=0.25' if depth[v] <= depth[u] else 'arc3'))
    for n, d in g.nodes(data=True):
        ntype = str(d.get('type', '?'))
        ax.text(pos[n][0], pos[n][1], udf_node_label(n, d), ha='center', va='center', fontsize=7, zorder=2,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=UDF_NODE_COLOURS.get(ntype, 'white'),
                          edgecolor='#9D1912' if ntype in ('BRANCH', 'LOOP_HEAD', 'LOOP_END') else '#D3DADE'))
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    ax.margins(0.08)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main

def select_plans(plans_path: str, n: int, seed: int):
    plans, _ = read_workload_runs([plans_path], min_runtime_ms=100, max_runtime=30)
    indices = list(range(len(plans)))
    if n and n < len(plans):
        indices = sorted(random.Random(seed).sample(indices, n))
    return [(i, plans[i]) for i in indices], len(plans)


def profile_workload(db: str, workload: str, args, feature_statistics: dict, out_dir: str):
    plans_path = f'{WL_BASE}/{WORKLOADS[workload]}/parsed_plans/{db}/workload.json'
    selected, n_total = select_plans(plans_path, args.n_queries, args.seed)
    print(f'== {db} {workload}: {len(selected)} of {n_total} test queries', flush=True)

    predictions, model_names = {}, {}
    for kind in ('baseline', 'augmented'):
        print(f'  {kind} model inference ...', flush=True)
        predictions[kind], model_names[kind] = predict(db, kind, plans_path, [p for _, p in selected],
                                                       feature_statistics, args.num_workers)

    for sub in ('plans', 'udf_graphs'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    con = connect(db, workload, args.threads)
    graphs_dir = f'{WL_BASE}/{WORKLOADS[workload]}/dbs/{db}/created_graphs'
    rows = []
    for k, (idx, plan) in enumerate(selected):
        sql, udf_name = plan.query, plan.udf.udf_name
        qid = f'{db}_{workload}_{idx:05d}'
        print(f'  [{k + 1}/{len(selected)}] {qid} {udf_name}', flush=True)
        runs = [run_profiled(con, sql, args.timeout_s) for _ in range(args.repetitions)]
        runs = [(p, w) for p, w in runs if p is not None]
        row = dict(query_id=qid, workload=workload, udf=udf_name, sql=sql, threads=args.threads,
                   repetitions_ok=len(runs))
        if runs:
            totals = [float(p.get('result', 0.0)) for p, _ in runs]
            median_run = sorted(zip(totals, range(len(runs))))[len(runs) // 2][1]
            plan_json = runs[median_run][0]
            udf_times = [udf_operator_time(p, udf_name)[0] for p, _ in runs]
            _, udf_ops, udf_rows_in = udf_operator_time(plan_json, udf_name)
            total_s, udf_s = statistics.median(totals), statistics.median(udf_times)
            row.update(total_time_s=total_s, udf_time_s=udf_s, sql_time_s=total_s - udf_s,
                       udf_share=udf_s / total_s if total_s > 0 else None,
                       udf_operator=' + '.join(udf_ops), rows_into_udf=udf_rows_in,
                       total_time_runs_s=', '.join(f'{t:.4f}' for t in totals),
                       udf_time_runs_s=', '.join(f'{t:.4f}' for t in udf_times))
            g = plan_graph(plan_json, udf_name)
            nx.write_gpickle(g, os.path.join(out_dir, 'plans', f'{qid}.gpickle'))
            draw_plan(g, os.path.join(out_dir, 'plans', f'{qid}.png'),
                      f'{qid}: total {total_s:.3f} s, UDF ({udf_name}) {udf_s:.3f} s')
            row.update(plan_gpickle=f'plans/{qid}.gpickle', plan_png=f'plans/{qid}.png')
        src = os.path.join(graphs_dir, f'{udf_name}.loopend.gpickle')
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out_dir, 'udf_graphs', f'{qid}.gpickle'))
            try:
                draw_udf_graph(nx.read_gpickle(src), os.path.join(out_dir, 'udf_graphs', f'{qid}.png'),
                               f'{qid}: UDF graph of {udf_name}')
                row['udf_graph_png'] = f'udf_graphs/{qid}.png'
            except Exception as e:
                print(f'    UDF graph drawing failed: {e}', flush=True)
            row['udf_graph_gpickle'] = f'udf_graphs/{qid}.gpickle'
        rows.append(row)
    con.close()

    sheets = {}
    for kind in ('baseline', 'augmented'):
        recs = []
        for row in rows:
            rec = dict(row)
            hit = predictions[kind].get(row['sql'])
            if hit is not None:
                label, pred, stats = hit
                rec.update(stats)
                rec.update(graceful_runtime_s=label, predicted_runtime_s=pred,
                           q_error=float(fwq.q_error(numpy.array([label]), numpy.array([pred]))[0]))
            recs.append(rec)
        sheets[kind] = pd.DataFrame(recs)
    return sheets, model_names


COLUMNS = ['query_id', 'workload', 'udf', 'udf_pos_in_query', 'num_joins', 'num_filters', 'udf_num_loops',
           'udf_num_branches', 'udf_num_comp_nodes', 'udf_num_np_calls', 'udf_num_math_calls', 'udf_in_card',
           'threads', 'repetitions_ok', 'total_time_s', 'udf_time_s', 'sql_time_s', 'udf_share', 'udf_operator',
           'rows_into_udf', 'graceful_runtime_s', 'predicted_runtime_s', 'q_error', 'total_time_runs_s',
           'udf_time_runs_s', 'plan_gpickle', 'plan_png', 'udf_graph_gpickle', 'udf_graph_png', 'sql']

ABOUT = [
    ('total_time_s', 'Whole query time measured here (median of the repetitions), DuckDB 0.10.1, read-only GRACEFUL db file.'),
    ('udf_time_s', 'Time of the plan operator(s) that call the UDF (udf_operator), median of the repetitions.'),
    ('sql_time_s', 'total_time_s - udf_time_s.'),
    ('udf_share', 'udf_time_s / total_time_s: the fraction of the query spent in the UDF.'),
    ('threads', 'DuckDB threads. With 1, operator times are real seconds. With more, operator times are summed '
                'over threads, so udf_time_s is thread-seconds.'),
    ('rows_into_udf', 'Actual rows entering the UDF operator(s) in the executed plan.'),
    ('graceful_runtime_s', "GRACEFUL's recorded runtime: the label the model was trained on (GRACEFUL's machine)."),
    ('predicted_runtime_s / q_error', "This spreadsheet's model's prediction; q_error = max(pred/label, label/pred)."),
    ('udf_in_card / num_joins / ...', "Query and UDF features as the model's data loader reports them."),
    ('plan_gpickle', 'Executed plan as a networkx DiGraph (nx.read_gpickle): nodes = operators with operator, time_s, '
                     'rows, est_rows, details, calls_udf; edges parent -> child.'),
    ('udf_graph_gpickle', "GRACEFUL's UDF graph for this query's UDF (the .loopend file the model reads)."),
]


def main() -> int:
    args = parse_args()
    feature_statistics = load_json(STATISTICS_FILE, namespace=False)
    feature_statistics['on_udf'] = {'value_dict': {'True': 0, 'False': 1}, 'no_vals': 2, 'type': 'categorical'}
    for db in args.dbs:
        out_dir = os.path.join(args.output_dir, db)
        os.makedirs(out_dir, exist_ok=True)
        per_kind = {'baseline': {}, 'augmented': {}}
        names = {}
        for workload in args.workloads:
            sheets, names = profile_workload(db, workload, args, feature_statistics, out_dir)
            for kind, frame in sheets.items():
                per_kind[kind][workload] = frame[[c for c in COLUMNS if c in frame.columns]]
            for kind in per_kind:  # write after every workload, so a later failure keeps what finished
                path = os.path.join(out_dir, f'{db}_{kind}.xlsx')
                with pd.ExcelWriter(path) as writer:
                    about = pd.DataFrame(ABOUT, columns=['column', 'meaning'])
                    about.loc[len(about)] = ['model', names[kind]]
                    about.loc[len(about)] = ['settings', f'n_queries={args.n_queries or "all"} per workload, '
                                                         f'seed={args.seed}, threads={args.threads}, '
                                                         f'repetitions={args.repetitions}']
                    about.to_excel(writer, sheet_name='about', index=False)
                    for wl, frame in per_kind[kind].items():
                        frame.to_excel(writer, sheet_name=wl, index=False)
                print(f'  wrote {path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
