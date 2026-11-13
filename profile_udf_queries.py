#!/usr/bin/env python3
"""Per-query profile for the UDF time vs SQL time experiment: DuckDB timings, executed plans, UDF graphs and the
q-error of the baseline and the augmented model, query by query.

Three steps, so timing never runs next to model inference and a missing checkpoint can be added later:
  --step time     DuckDB only (no checkpoint needed). Times the selected test queries of each database and
                  workload and appends one line per finished execution to <db>_<workload>_progress.csv. A restart
                  (Ctrl-C, reboot, end of a reservation) skips what is done, so it loses at most the query that was
                  running. Two procedures:
                    --procedure graceful  GRACEFUL's own procedure (cross_db_benchmark/benchmark_tools/dbms/
                        run_workload.py with duckdb_connection.py): a writable copy of the db file, VACUUM ANALYZE,
                        one connection with DuckDB's default threads (or --threads), `filter_pushdown` disabled for
                        pull-up, queries in file order, first EXPLAIN, then --repetitions repetitions, each a child
                        process that runs the query (timeout check, not timed) followed by EXPLAIN ANALYZE in the
                        main process (timed). The time is the profiler's `result`, and GRACEFUL's label is the mean
                        over the repetitions (parse_dd_plan.py:151-156). 30 s timeout.
                    --procedure passes    Our procedure (2026-11-05): the read-only db file, --threads (default 1),
                        pinned to one logical CPU, --passes passes over the selected set (1 in order, 2 reversed,
                        3+ seeded random), times = mean over the passes.
                  With more than one thread DuckDB adds operator time up over threads, so the UDF operator time is
                  thread-seconds (udf_time_unit), not elapsed time.
  --step predict  Models only (no DuckDB timing). The baseline and the augmented model predict the selected
                  queries; one line per query in <db>_<workload>_pred_<kind>.csv. A model whose checkpoint is not
                  here (MODELS entry None) is skipped and can be added later.
  --step capcheck Checks that the clock cap holds: a CPU-bound DuckDB query on all threads for --capcheck_s s
                  while typeperf logs the real clock; prints the effective MHz under load.
  --step merge    Joins the progress and prediction files by query_id into <db>_baseline.xlsx and
                  <db>_augmented.xlsx (one sheet per workload plus `about` and `machine`), and joins the Windows-side
                  clock log to every execution. --draw also writes the plan and UDF graph files and drawings.

Query selection: a seeded permutation (--seed) of each workload's test queries; --n_queries N takes its first N,
so a later run with a larger N (or 0 = all) adds only the queries that are not done yet and the smaller sample is
always a prefix of the larger one. --n_pushdown / --n_pullup override N per workload.

Machine state: a sampler records the system-wide CPU busy share and the load average for every execution, plus the
CPU clock and temperature where Linux exposes them (not in WSL). In WSL the real clock comes from the Windows side:
the time step runs typeperf.exe (processor frequency and % processor performance, once per second) into
clock_<timestamp>.csv, and every execution stores its absolute start and end time to join it. machine_<timestamp>.json
records the host, CPU, cores, clock cap, Windows power plan, .wslconfig, DuckDB version, threads and procedure. Every
output row carries a `host` column.

CPU only (lab rule: Claude never touches a GPU). Run with CUDA_VISIBLE_DEVICES="", e.g. on the lab PC:
    export GRACEFUL_DATA=~/polaris/data/Graceful_data
    CUDA_VISIBLE_DEVICES="" python profile_udf_queries.py --step time --procedure graceful --dbs financial \\
        --n_pushdown 1000 --n_pullup 0 --clock_cap "2.2 GHz, Windows max processor state 59%"
    CUDA_VISIBLE_DEVICES="" python profile_udf_queries.py --step predict --dbs financial --n_pushdown 1000 --n_pullup 0
    CUDA_VISIBLE_DEVICES="" python profile_udf_queries.py --step merge --procedure graceful --dbs financial
Pause other programs during a timing run: they compete for CPU and memory bandwidth.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime
import glob
import gzip
import json
import math
import os
import platform
import random
import re
import shutil
import socket
import statistics
import subprocess
import threading
import time

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('Run with CUDA_VISIBLE_DEVICES="" (CPU only).')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/polaris_matplotlib')
# flat_vector_baseline.py (imported by the data loader) creates wandb.Api() at import; on a terminal without a wandb
# login that asks interactively and would stall an unattended run. This script never logs to wandb.
os.environ.setdefault('WANDB_MODE', 'disabled')

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

DEFAULT_GRACEFUL = os.environ.get('GRACEFUL_DATA', '/mnt/data/shared/Graceful_data')
WORKLOADS = {'pushdown': 'duckdb_pushdown', 'pullup': 'duckdb_pullup'}
HOST = socket.gethostname()

# The checkpoints behind the result-sheet rows, as (checkpoint timestamp suffix, training log). The checkpoint is
# found under saved/models/ by its suffix; the log holds the run's exact training config, so it must be in
# saved/training_logs/. None = checkpoint not available here (the db is timed, that model's predictions come later).
MODELS = {
    'employee': {'baseline': ('20261015_021247_014', 'paper_est_20261015_021244_3186075_run01.log'),
                 'augmented': ('20261026_202734_018', 'paper_est_20261026_202730_523164_run01.log')},
    'financial': {'baseline': ('20261015_070416_058', 'paper_est_20261015_070413_3374010_run01.log'),
                  'augmented': ('20261027_213157_021', 'paper_est_20261027_213153_2064398_run01.log')},
    'fhnk': {'baseline': ('20261014_200658_030', 'paper_est_20261014_200654_2889800_run01.log'),
             'augmented': ('20261026_212002_085', 'paper_est_20261026_211958_598907_run01.log')},
    # seznam baseline: Ishana's run (est, 50 epochs, time_stamp 20261017_211513_743446), from the second transfer zip.
    'seznam': {'baseline': ('20261017_211517_086', 'paper_est_20261017_211513_743446_run01.log'),
               'augmented': ('20261028_120105_053', 'paper_est_20261028_120101_2917524_run01.log')},
}

# Progress file columns: one line per finished execution (a repetition or a pass).
PROGRESS_FIELDS = ['host', 'query_id', 'plan_index', 'udf', 'procedure', 'threads', 'run_no', 'status',
                   'total_time_s', 'udf_time_s', 'wall_s', 'start_epoch', 'end_epoch', 'pin_cpu',
                   'cpu_freq_mean_mhz', 'cpu_freq_min_mhz', 'core_temp_max_c', 'package_temp_max_c',
                   'system_cpu_busy_pct', 'loadavg_1min']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--step', required=True, choices=['time', 'predict', 'merge', 'capcheck'])
    parser.add_argument('--procedure', default='graceful', choices=['graceful', 'passes'],
                        help='timing procedure (time and merge steps)')
    parser.add_argument('--graceful_dir', default=DEFAULT_GRACEFUL,
                        help='Graceful_data root (default: $GRACEFUL_DATA or the server path)')
    parser.add_argument('--dbs', nargs='+', default=['financial', 'employee'], choices=sorted(MODELS))
    parser.add_argument('--workloads', nargs='+', default=list(WORKLOADS), choices=list(WORKLOADS))
    parser.add_argument('--n_queries', type=int, default=100,
                        help='queries per database per workload (first N of the seeded permutation); 0 = all')
    parser.add_argument('--n_pushdown', type=int, default=None, help='overrides --n_queries for push-down')
    parser.add_argument('--n_pullup', type=int, default=None, help='overrides --n_queries for pull-up')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=None,
                        help='DuckDB threads; default: DuckDB default (all logical CPUs) for graceful, 1 for passes')
    parser.add_argument('--repetitions', type=int, default=2, help='graceful: timed repetitions per query')
    parser.add_argument('--passes', type=int, default=3,
                        help='passes: passes over the whole set (1 in order, 2 reversed, 3+ random); times = mean')
    parser.add_argument('--pin_cpu', type=int, default=2, help='passes: logical CPU the DuckDB runs are pinned to')
    parser.add_argument('--timeout_s', type=float, default=None,
                        help='per execution; default 30 for graceful (as GRACEFUL), 300 for passes')
    parser.add_argument('--clock_cap', default='not set',
                        help='time step: how the CPU clock is capped, recorded in machine_*.json')
    parser.add_argument('--no_clock_log', action='store_true', help='time step: do not start typeperf.exe')
    parser.add_argument('--capcheck_s', type=float, default=30, help='capcheck: seconds of load')
    parser.add_argument('--num_workers', type=int, default=2, help='predict: data loader workers')
    parser.add_argument('--draw', action='store_true', help='merge: write plan / UDF graph files and drawings')
    parser.add_argument('--run_name', default=None,
                        help='output subfolder (default <procedure>_threads<N>, N = effective threads)')
    parser.add_argument('--output_dir', default='results/udf_query_profile_labpc')
    args = parser.parse_args()
    if args.timeout_s is None:
        args.timeout_s = 30.0 if args.procedure == 'graceful' else 300.0
    if args.threads is None and args.procedure == 'passes':
        args.threads = 1
    return args


def paths(graceful_dir: str) -> dict:
    wl_base = f'{graceful_dir}/workload_runs'
    return dict(graceful=graceful_dir, wl_base=wl_base,
                statistics=f'{wl_base}/duckdb_pushdown/parsed_plans/statistics_workload_combined.json')


def n_for(args, workload: str) -> int:
    override = args.n_pushdown if workload == 'pushdown' else args.n_pullup
    return args.n_queries if override is None else override


def effective_threads(args) -> int:
    """Threads the run will use: --threads, or DuckDB's default (a fresh connection's setting)."""
    if args.threads:
        return args.threads
    con = duckdb.connect()
    try:
        return int(con.execute("SELECT current_setting('threads')").fetchone()[0])
    finally:
        con.close()


def run_dir(args) -> str:
    name = args.run_name or f'{args.procedure}_threads{effective_threads(args)}'
    return os.path.join(args.output_dir, name)


# ---------------------------------------------------------------- query selection

def test_plans(p: dict, db: str, workload: str):
    """(workload.json path, the test plans as the data loader reads them)."""
    plans_path = f"{p['wl_base']}/{WORKLOADS[workload]}/parsed_plans/{db}/workload.json"
    plans, _ = read_workload_runs([plans_path], min_runtime_ms=100, max_runtime=30)
    return plans_path, plans


def select(plans: list, db: str, workload: str, n: int, seed: int):
    """[(query_id, plan index, plan)]: the first n of a seeded permutation (all if n is 0)."""
    order = list(range(len(plans)))
    random.Random(seed).shuffle(order)
    if n:
        order = order[:n]
    return [(f'{db}_{workload}_{i:05d}', i, plans[i]) for i in order]


# ---------------------------------------------------------------- progress files

def read_progress(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=PROGRESS_FIELDS)
    return pd.read_csv(path)


class ProgressWriter:
    """Appends progress lines and flushes them to disk at once."""

    def __init__(self, path: str):
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.f = open(path, 'a', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=PROGRESS_FIELDS, extrasaction='ignore')
        if new:
            self.w.writeheader()

    def write(self, rows: list):
        for row in rows:
            self.w.writerow(row)
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()


def save_plan(out_dir: str, qid: str, plan_json: dict):
    """Keep the first successful executed plan of a query (gzipped profiler JSON)."""
    path = os.path.join(out_dir, 'plans', f'{qid}.json.gz')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        with gzip.open(tmp, 'wt') as f:
            json.dump(plan_json, f)
        os.replace(tmp, path)


def load_plan(out_dir: str, qid: str):
    path = os.path.join(out_dir, 'plans', f'{qid}.json.gz')
    if not os.path.exists(path):
        return None
    with gzip.open(path, 'rt') as f:
        return json.load(f)


# ---------------------------------------------------------------- models

def resolve_model(db: str, kind: str):
    """(model dir, model name, run log) of a MODELS entry, or None if it has no checkpoint here."""
    entry = MODELS[db][kind]
    if entry is None:
        return None
    suffix, log_name = entry
    hits = glob.glob(f'saved/models/*_{db}_*/*_{suffix}.pt')
    if len(hits) != 1:
        raise SystemExit(f'{db} {kind}: expected one checkpoint *_{suffix}.pt under saved/models/, found {hits}')
    run_log = os.path.join('saved/training_logs', log_name)
    if not os.path.exists(run_log):
        raise SystemExit(f'{db} {kind}: training log {run_log} is missing (it holds the training config)')
    return os.path.dirname(hits[0]), os.path.basename(hits[0])[:-len('.pt')], run_log


def model_config(run_log: str) -> dict:
    """The exact training config of a run, rebuilt from its log's `Running with config:` line."""
    text = open(run_log, errors='replace').read()
    args = ast.literal_eval(re.search(r'Running with config: (\{.*\})', text).group(1))
    for key in ('test_against', 'include_pullup_data', 'include_pushdown_data'):
        args.pop(key, None)
    # Runs from older code (the seznam baseline) name the activation `activation`; get_config expects
    # `activation_class_name` and asserts that every key is used.
    if 'activation' in args:
        args['activation_class_name'] = args.pop('activation')
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


STAT_KEYS = ('udf_pos_in_query', 'num_joins', 'num_filters', 'udf_in_card', 'udf_num_loops', 'udf_num_branches',
             'udf_num_np_calls', 'udf_num_math_calls', 'udf_num_comp_nodes')


def predict(p: dict, db: str, kind: str, plans_path: str, plans: list, feature_statistics: dict, num_workers: int):
    """{sql: (label_s, predicted_s, stats row)} for the given plans, from one model; and the model name."""
    model_dir, model_name, run_log = resolve_model(db, kind)
    config = model_config(run_log)
    model = load_model(model_dir, model_name, config, feature_statistics)
    labels, preds, stats = fwq.run_query_inference(model, config, plans_path, p['statistics'], feature_statistics,
                                                   'est', None, num_workers, plans_override=plans)
    out = {}
    for i, sql in enumerate(stats['sql'][:len(labels)]):
        out[sql] = (float(labels[i]), float(preds[i]), {k: stats[k][i] for k in STAT_KEYS})
    return out, model_name


# ---------------------------------------------------------------- DuckDB plans

UDF_SCOPE_NOTE = ('only the UDFs of the selected queries are registered. GRACEFUL registers the whole udfs.sql, '
                  '100,000 UDFs per database and workload, which takes about 5 GB here and does not fit in the '
                  'lab PC WSL VM; functions are looked up by name, so the unused ones are not expected to change '
                  'query times')


def udf_source(path: str, names: set) -> str:
    """The `def` blocks (each followed by its own db_conn.create_function line) of the named UDFs in a udfs.sql."""
    keep, chunks, found = False, [], set()
    with open(path) as f:
        for line in f:
            if line.startswith('def '):
                name = re.match(r'def (\w+)\(', line).group(1)
                keep = name in names
                if keep:
                    found.add(name)
            if keep:
                chunks.append(line)
    missing = names - found
    if missing:
        raise SystemExit(f'{path}: no definition for {sorted(missing)[:5]} ({len(missing)} missing)')
    return ''.join(chunks)


def register_udfs(con, p: dict, db: str, workload: str, names: set):
    """Register GRACEFUL's definitions of the named UDFs, with run_workload.py's globals (see UDF_SCOPE_NOTE)."""
    udf_code = udf_source(f"{p['wl_base']}/{WORKLOADS[workload]}/dbs/{db}/sql_scripts/udfs.sql", names)
    exec(udf_code, {'math': math, 'numpy': numpy, 'db_conn': con, 'date': datetime.date, 'time': datetime.time,
                    'VARCHAR': duckdb.typing.VARCHAR, 'INTEGER': duckdb.typing.INTEGER,
                    'DOUBLE': duckdb.typing.DOUBLE, 'TIMESTAMP': duckdb.typing.TIMESTAMP,
                    'DATE': duckdb.typing.DATE, 'TIME': duckdb.typing.TIME, 'BOOLEAN': duckdb.typing.BOOLEAN,
                    'HUGEINT': duckdb.typing.HUGEINT, 'FLOAT': duckdb.typing.FLOAT})


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


def estimated_rows(node: dict):
    """DuckDB's cardinality estimate (EC) of a node; through single-child nodes that carry none (e.g. a
    projection), the first estimate below, since those nodes pass their child's rows on."""
    while node is not None:
        est = re.search(r'EC:\s*(\d+)', str(node.get('extra_info', '')))
        if est:
            return int(est.group(1))
        children = node.get('children', [])
        node = children[0] if len(children) == 1 else None
    return None


def udf_operator_info(plan: dict, udf_name: str):
    """Time of the operators whose expression calls the UDF, their names, the actual rows entering them, and
    DuckDB's estimate of those rows (None if a feeding operator has no estimate)."""
    seconds, names, rows_in, est_in = 0.0, [], 0, 0
    for node, _ in plan_nodes(plan):
        if re.search(rf'\b{udf_name}\(', str(node.get('extra_info', ''))):
            seconds += float(node.get('timing', 0.0))
            names.append(node.get('name', '').strip())
            for child in node.get('children', []):
                rows_in += int(child.get('cardinality', 0))
                est = estimated_rows(child)
                est_in = None if est is None or est_in is None else est_in + est
    return seconds, names, rows_in, (est_in if names else None)


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


# ---------------------------------------------------------------- machine state (throttling / interference)

def _read_number(path):
    try:
        return float(open(path).read().strip())
    except (OSError, ValueError, TypeError):
        return None


def coretemp_inputs(cpu: int):
    """(this CPU's core temperature file, [package temperature files]) from the coretemp hwmon sensors; (None, [])
    where there are none (WSL has no hwmon sensors)."""
    pkg = int(_read_number(f'/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id') or 0)
    core = int(_read_number(f'/sys/devices/system/cpu/cpu{cpu}/topology/core_id') or 0)
    core_file, package_files = None, []
    try:
        hwmons = sorted(os.listdir('/sys/class/hwmon'))
    except OSError:
        return None, []
    for hw in hwmons:
        base = f'/sys/class/hwmon/{hw}'
        try:
            if open(f'{base}/name').read().strip() != 'coretemp':
                continue
            labels = {}
            for f in os.listdir(base):
                if f.endswith('_label'):
                    labels[open(f'{base}/{f}').read().strip()] = f'{base}/{f[:-len("_label")]}_input'
        except OSError:
            continue
        package_files += [v for k, v in labels.items() if k.startswith('Package id')]
        if f'Package id {pkg}' in labels and f'Core {core}' in labels:
            core_file = labels[f'Core {core}']
    return core_file, package_files


def cpu_busy_ticks():
    """(busy, total) jiffies over all CPUs from /proc/stat."""
    fields = [int(x) for x in open('/proc/stat').readline().split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    return sum(fields) - idle, sum(fields)


class MachineSampler(threading.Thread):
    """Samples a CPU's clock and core temperature and the hottest package temperature every `period` s, where
    Linux exposes them (None in WSL). Started before the main thread pins itself, so it does not run on the
    measured CPU."""

    def __init__(self, cpu: int, period: float = 0.25):
        super().__init__(daemon=True)
        self.cpu, self.period = cpu, period
        self.freq_file = f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq'
        self.core_file, self.package_files = coretemp_inputs(cpu)
        self.samples = []  # (time, MHz, core C, package C)
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            freq = _read_number(self.freq_file)
            core = _read_number(self.core_file) if self.core_file else None
            pkgs = [v for v in (_read_number(f) for f in self.package_files) if v is not None]
            with self.lock:
                self.samples.append((time.monotonic(), freq / 1000 if freq else None, core / 1000 if core else None,
                                     max(pkgs) / 1000 if pkgs else None))
                if len(self.samples) > 20000:
                    del self.samples[:10000]
            time.sleep(self.period)

    def window(self, start: float, end: float) -> dict:
        with self.lock:
            win = [x for x in self.samples if start <= x[0] <= end] or self.samples[-1:]
        freqs = [x[1] for x in win if x[1] is not None]
        cores = [x[2] for x in win if x[2] is not None]
        pkgs = [x[3] for x in win if x[3] is not None]
        return dict(cpu_freq_mean_mhz=statistics.mean(freqs) if freqs else None,
                    cpu_freq_min_mhz=min(freqs) if freqs else None,
                    core_temp_max_c=max(cores) if cores else None,
                    package_temp_max_c=max(pkgs) if pkgs else None)


class StateWindow:
    """Machine state over one execution: absolute start/end time (to join the Windows clock log), wall time,
    system-wide CPU busy share, load average and the sampler's window."""

    def __init__(self, sampler: MachineSampler):
        self.sampler = sampler

    def __enter__(self):
        self.load1 = float(open('/proc/loadavg').read().split()[0])
        self.busy0, self.total0 = cpu_busy_ticks()
        self.start_epoch, self.start = time.time(), time.monotonic()
        return self

    def __exit__(self, *exc):
        self.end, self.end_epoch = time.monotonic(), time.time()
        busy1, total1 = cpu_busy_ticks()
        self.state = self.sampler.window(self.start, self.end)
        self.state.update(loadavg_1min=self.load1, wall_s=self.end - self.start,
                          start_epoch=round(self.start_epoch, 3), end_epoch=round(self.end_epoch, 3),
                          system_cpu_busy_pct=100 * (busy1 - self.busy0) / (total1 - self.total0)
                          if total1 > self.total0 else None)
        return False


# ---------------------------------------------------------------- Windows side (WSL): clock log, machine facts

WINDOWS = '/mnt/c/Windows/System32'
TYPEPERF = f'{WINDOWS}/typeperf.exe'
POWERSHELL = f'{WINDOWS}/WindowsPowerShell/v1.0/powershell.exe'
CLOCK_COUNTERS = [r'\Processor Information(_Total)\% Processor Performance',
                  r'\Processor Information(_Total)\Processor Frequency']


def powershell(command: str) -> str | None:
    if not os.path.exists(POWERSHELL):
        return None
    try:
        # stdin from /dev/null: powershell.exe otherwise waits on the terminal and stalls a background run
        out = subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-Command', command], capture_output=True,
                             text=True, timeout=60, stdin=subprocess.DEVNULL)
        return out.stdout.replace('\r', '').strip()
    except (OSError, subprocess.SubprocessError):
        return None


class ClockLogger:
    """typeperf.exe on the Windows host, once per second, into a CSV. Effective MHz = Processor Frequency x
    % Processor Performance / 100. typeperf writes Windows local time; the UTC offset is kept to convert it."""

    def __init__(self, path: str):
        self.path, self.proc = path, None
        if os.path.exists(TYPEPERF):
            self.f = open(path, 'w')
            self.proc = subprocess.Popen([TYPEPERF, *CLOCK_COUNTERS, '-si', '1'], stdout=self.f,
                                         stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.f.close()


def windows_utc_offset_s():
    out = powershell('[System.TimeZoneInfo]::Local.GetUtcOffset([datetime]::Now).TotalSeconds')
    try:
        return float(out)
    except (TypeError, ValueError):
        return None


def read_clock_logs(run_root: str) -> pd.DataFrame:
    """All clock_*.csv of a run as (epoch, effective_mhz), using each log's UTC offset from its machine json."""
    frames = []
    for path in sorted(glob.glob(os.path.join(run_root, 'clock_*.csv'))):
        stamp = os.path.basename(path)[len('clock_'):-len('.csv')]
        meta_path = os.path.join(run_root, f'machine_{stamp}.json')
        offset = json.load(open(meta_path)).get('windows_utc_offset_s') if os.path.exists(meta_path) else None
        if offset is None:
            continue
        try:
            df = pd.read_csv(path, skipinitialspace=True, on_bad_lines='skip')
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if df.shape[1] < 3 or df.empty:
            continue
        df.columns = ['time', 'perf_pct', 'freq_mhz'] + list(df.columns[3:])
        df['perf_pct'] = pd.to_numeric(df['perf_pct'], errors='coerce')
        df['freq_mhz'] = pd.to_numeric(df['freq_mhz'], errors='coerce')
        local = pd.to_datetime(df['time'], format='%m/%d/%Y %H:%M:%S.%f', errors='coerce')
        df['epoch'] = (local - pd.Timestamp('1970-01-01')).dt.total_seconds() - offset
        df['effective_mhz'] = df['freq_mhz'] * df['perf_pct'] / 100
        df = df[df['perf_pct'] >= 0]  # typeperf writes -1 for a sample it could not read
        frames.append(df[['epoch', 'effective_mhz']].dropna())
    return pd.concat(frames, ignore_index=True).sort_values('epoch') if frames else pd.DataFrame()


def machine_facts(args, threads: int, stamp: str) -> dict:
    """Host, CPU, cores, clock cap and Windows power plan, WSL config, DuckDB version and run settings."""
    cpuinfo = open('/proc/cpuinfo').read()
    model = re.search(r'model name\s*:\s*(.*)', cpuinfo)
    facts = dict(host=HOST, recorded=stamp, platform=platform.platform(),
                 linux_cpu_model=model.group(1).strip() if model else None, linux_logical_cpus=os.cpu_count(),
                 wsl='microsoft' in platform.release().lower(), duckdb_version=duckdb.__version__,
                 procedure=args.procedure, threads=threads, threads_set_explicitly=bool(args.threads),
                 repetitions=args.repetitions if args.procedure == 'graceful' else None,
                 passes=args.passes if args.procedure == 'passes' else None,
                 pin_cpu=args.pin_cpu if args.procedure == 'passes' else None,
                 timeout_s=args.timeout_s, seed=args.seed, clock_cap=args.clock_cap, udf_registration=UDF_SCOPE_NOTE,
                 graceful_dir=args.graceful_dir)
    try:
        facts['git_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                             text=True).stdout.strip()
    except OSError:
        pass
    if facts['wsl']:
        cpu = powershell('Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,'
                         'NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json')
        try:
            facts['windows_cpu'] = json.loads(cpu)
        except (TypeError, ValueError):
            facts['windows_cpu'] = cpu
        facts['windows_power_scheme'] = powershell('powercfg /getactivescheme')
        facts['windows_max_processor_state'] = powershell(
            'powercfg /q SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX | Select-String "Current AC|Current DC" | '
            'ForEach-Object { $_.Line.Trim() }')
        facts['windows_utc_offset_s'] = windows_utc_offset_s()
        profile = powershell('$env:USERPROFILE')
        wslconfig = None
        if profile:
            path = '/mnt/' + profile[0].lower() + profile[2:].replace('\\', '/') + '/.wslconfig'
            wslconfig = open(path).read() if os.path.exists(path) else f'none ({path} does not exist)'
        facts['wslconfig'] = wslconfig
        if args.procedure == 'passes':
            facts['pinning_note'] = ('pin_cpu is a WSL virtual CPU; Windows decides which physical core runs it')
    return facts


# ---------------------------------------------------------------- time step

class GracefulConnection:
    """GRACEFUL's own DuckDB connection class (cross_db_benchmark/benchmark_tools/duckdb_connection.py) on a copy
    of the db file, so VACUUM ANALYZE can write to it."""

    def __init__(self, p: dict, db: str, workload: str, work_dir: str, threads: int | None, timeout_s: float,
                 udf_names: set):
        from cross_db_benchmark.benchmark_tools.duckdb_connection import DuckdbDatabaseConnection
        os.makedirs(work_dir, exist_ok=True)
        shutil.copyfile(f"{p['graceful']}/datasets/{db}_10_1.db", os.path.join(work_dir, f'{db}_10_1.db'))
        self.dbms = DuckdbDatabaseConnection(version='0.10.1', db_name=db, database_kwargs={'dir': work_dir})
        con = self.dbms.get_connection()  # duckdb.connect(db_path), default threads, as GRACEFUL
        if threads:
            con.execute(f'SET threads TO {threads};')
        register_udfs(con, p, db, workload, udf_names)
        # Read the setting only after registering: an open result keeps a transaction active, and DuckDB refuses
        # create_function inside one. fetchall() consumes the result.
        self.threads = int(con.execute("SELECT current_setting('threads')").fetchall()[0][0])
        print(f'  VACUUM ANALYZE {db} ({workload}) ...', flush=True)
        self.dbms.submit_query('VACUUM ANALYZE;')  # before the timeout is set, as run_workload.py
        self.dbms.set_statement_timeout(timeout_s)
        self.pullup = workload == 'pullup'

    def run_query(self, sql: str, repetitions: int, sampler: MachineSampler):
        """run_query_collect_statistics (duckdb_connection.py:86-123), unrolled so each repetition gets its own
        machine-state window: EXPLAIN, then per repetition get_result(EXPLAIN ANALYZE), which runs the query in a
        child process first (timeout check) and then in this process (timed by the profiler). Returns
        [(plan json or None, status, state)] per repetition attempted."""
        import func_timeout
        if self.pullup:  # run_workload.py sets this before every pulled-up query
            self.dbms.submit_query("set disabled_optimizers = 'filter_pushdown';")
        if not self.dbms.profiling_enabled:
            self.dbms.submit_query("PRAGMA enable_profiling = 'json';")
            self.dbms.profiling_enabled = True
        out = []
        try:
            self.dbms.get_result(f'EXPLAIN {sql}', as_df=False, verbose=False)
            for _ in range(repetitions):
                window = StateWindow(sampler)
                try:
                    with window:
                        rows = self.dbms.get_result(f'EXPLAIN ANALYZE {sql}', as_df=False, verbose=False)
                except BaseException:
                    out.append((None, 'failed', window.state))
                    raise
                out.append((json.loads(rows[0][1]), 'ok', window.state))
        except func_timeout.exceptions.FunctionTimedOut:
            status = 'timeout'
        except duckdb.InvalidInputException as e:
            status = f'invalid input: {str(e)[:80]}'
        except duckdb.BinderException as e:
            status = f'binder error: {str(e)[:80]}'
        except Exception as e:
            status = f'error: {type(e).__name__}: {str(e)[:80]}'
        else:
            return out
        if out and out[-1][1] == 'failed':
            out[-1] = (None, status, out[-1][2])
        else:
            out.append((None, status, {}))
        return out

    def close(self):
        self.dbms.close_conn()


def progress_row(qid, idx, udf, args, threads, run_no, status, plan_json, state):
    total = float(plan_json['result']) if plan_json else None
    udf_s = udf_operator_info(plan_json, udf)[0] if plan_json else None
    return dict(host=HOST, query_id=qid, plan_index=idx, udf=udf, procedure=args.procedure, threads=threads,
                run_no=run_no, status=status, total_time_s=total, udf_time_s=udf_s,
                pin_cpu=args.pin_cpu if args.procedure == 'passes' else None, **state)


def time_graceful(db, workload, args, p, out_dir, selected, done, writer, sampler):
    con = GracefulConnection(p, db, workload, os.path.join(out_dir, 'work'), args.threads, args.timeout_s,
                             {q[2].udf.udf_name for q in selected})
    todo = sorted([q for q in selected if q[0] not in done], key=lambda q: q[1])  # file order
    print(f'  {len(todo)} queries to time ({len(selected) - len(todo)} done), threads={con.threads}', flush=True)
    try:
        for pos, (qid, idx, plan) in enumerate(todo, start=1):
            udf = plan.udf.udf_name
            results = con.run_query(plan.query, args.repetitions, sampler)
            rows = [progress_row(qid, idx, udf, args, con.threads, k, status, plan_json, state)
                    for k, (plan_json, status, state) in enumerate(results, start=1)]
            first_ok = next((r[0] for r in results if r[0] is not None), None)
            if first_ok is not None:
                save_plan(out_dir, qid, first_ok)
            writer.write(rows)  # all repetitions of a query at once: a stopped query reruns whole
            if pos % 50 == 0 or pos == len(todo) or results[-1][1] != 'ok':
                times = [r['total_time_s'] for r in rows if r['total_time_s'] is not None]
                print(f'    [{pos}/{len(todo)}] {qid} {udf}: {results[-1][1]}, '
                      f'{statistics.mean(times) if times else float("nan"):.3f} s', flush=True)
    finally:
        con.close()


def time_passes(db, workload, args, p, out_dir, selected, done, writer, sampler):
    con = duckdb.connect(f"{p['graceful']}/datasets/{db}_10_1.db", read_only=True)
    con.execute(f'SET threads TO {args.threads};')
    register_udfs(con, p, db, workload, {q[2].udf.udf_name for q in selected})
    con.execute("PRAGMA enable_profiling = 'json';")
    if workload == 'pullup':  # as run_workload.py does for pulled-up UDF filters
        con.execute("SET disabled_optimizers = 'filter_pushdown';")
    # Pass orders over the selected set: 1 = in order, 2 = reversed, 3+ = seeded random.
    base = sorted(selected, key=lambda q: q[1])
    orders = [list(range(len(base))), list(range(len(base)))[::-1]]
    for k in range(2, args.passes):
        order = list(range(len(base)))
        random.Random(args.seed + k).shuffle(order)
        orders.append(order)
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {args.pin_cpu})
    try:
        for pass_no, order in enumerate(orders[:args.passes], start=1):
            todo = [base[k] for k in order if (base[k][0], pass_no) not in done]
            print(f'  pass {pass_no}/{args.passes}: {len(todo)} to time', flush=True)
            for pos, (qid, idx, plan) in enumerate(todo, start=1):
                udf = plan.udf.udf_name
                window = StateWindow(sampler)
                timer = threading.Timer(args.timeout_s, con.interrupt)
                timer.start()
                plan_json, status = None, 'ok'
                try:
                    with window:
                        plan_json = json.loads(con.execute(f'EXPLAIN ANALYZE {plan.query}').fetchall()[0][1])
                except Exception as e:  # interrupted (timeout) or a query error
                    status = f'error: {type(e).__name__}: {str(e)[:80]}'
                finally:
                    timer.cancel()
                if plan_json is not None:
                    save_plan(out_dir, qid, plan_json)
                writer.write([progress_row(qid, idx, udf, args, args.threads, pass_no, status, plan_json,
                                           window.state)])
                if pos % 100 == 0 or pos == len(todo):
                    print(f'    [{pos}/{len(todo)}] {qid} {udf}', flush=True)
    finally:
        con.close()
        os.sched_setaffinity(0, affinity)


def step_time(args, p):
    root = run_dir(args)
    os.makedirs(root, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    threads = effective_threads(args)
    facts = machine_facts(args, threads, stamp)
    clock = None if args.no_clock_log or not facts['wsl'] else ClockLogger(os.path.join(root, f'clock_{stamp}.csv'))
    facts['clock_log'] = os.path.basename(clock.path) if clock and clock.proc else None
    json.dump(facts, open(os.path.join(root, f'machine_{stamp}.json'), 'w'), indent=2)
    print(f'Timing run {root} ({args.procedure}, {threads} threads), host {HOST}', flush=True)
    sampler = MachineSampler(args.pin_cpu)
    sampler.start()  # before pinning, so the sampler thread keeps the full CPU set
    try:
        for db in args.dbs:
            out_dir = os.path.join(root, db)
            os.makedirs(out_dir, exist_ok=True)
            for workload in args.workloads:
                _, plans = test_plans(p, db, workload)
                selected = select(plans, db, workload, n_for(args, workload), args.seed)
                path = os.path.join(out_dir, f'{db}_{workload}_progress.csv')
                prog = read_progress(path)
                print(f'== {db} {workload}: {len(selected)} of {len(plans)} test queries selected', flush=True)
                writer = ProgressWriter(path)
                try:
                    if args.procedure == 'graceful':
                        done = set(prog['query_id'])
                        time_graceful(db, workload, args, p, out_dir, selected, done, writer, sampler)
                    else:
                        done = set(zip(prog['query_id'], prog['run_no'].astype(int)))
                        time_passes(db, workload, args, p, out_dir, selected, done, writer, sampler)
                finally:
                    writer.close()
                    shutil.rmtree(os.path.join(out_dir, 'work'), ignore_errors=True)
    finally:
        sampler.running = False
        if clock:
            clock.stop()
    return 0


# ---------------------------------------------------------------- predict step

def step_predict(args, p):
    feature_statistics = load_json(p['statistics'], namespace=False)
    feature_statistics['on_udf'] = {'value_dict': {'True': 0, 'False': 1}, 'no_vals': 2, 'type': 'categorical'}
    out_root = os.path.join(args.output_dir, 'predictions')
    for db in args.dbs:
        out_dir = os.path.join(out_root, db)
        os.makedirs(out_dir, exist_ok=True)
        for workload in args.workloads:
            plans_path, plans = test_plans(p, db, workload)
            selected = select(plans, db, workload, n_for(args, workload), args.seed)
            for kind in ('baseline', 'augmented'):
                if MODELS[db][kind] is None:
                    print(f'== {db} {workload} {kind}: no checkpoint here, skipped', flush=True)
                    continue
                path = os.path.join(out_dir, f'{db}_{workload}_pred_{kind}.csv')
                done = set(pd.read_csv(path)['query_id']) if os.path.exists(path) else set()
                todo = [q for q in selected if q[0] not in done]
                print(f'== {db} {workload} {kind}: {len(todo)} to predict ({len(done)} done)', flush=True)
                if not todo:
                    continue
                preds, model_name = predict(p, db, kind, plans_path, [q[2] for q in todo], feature_statistics,
                                            args.num_workers)
                rows = []
                for qid, idx, plan in todo:
                    hit = preds.get(plan.query)
                    row = dict(host=HOST, query_id=qid, plan_index=idx, model=model_name)
                    if hit is not None:  # the data loader drops some plans (column-to-column UDF comparisons)
                        label, pred, stats = hit
                        row.update(stats, graceful_runtime_s=label, predicted_runtime_s=pred,
                                   q_error=float(fwq.q_error(numpy.array([label]), numpy.array([pred]))[0]))
                    rows.append(row)
                frame = pd.DataFrame(rows)
                frame.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
                print(f'  wrote {len(rows)} rows to {path}', flush=True)
    return 0


# ---------------------------------------------------------------- merge step

def clock_window(clock: pd.DataFrame, start: float, end: float):
    """(mean, min) effective MHz of the Windows clock samples inside [start, end], or the nearest sample."""
    if clock.empty or pd.isna(start) or pd.isna(end):
        return None, None
    epochs = clock['epoch'].values
    lo, hi = numpy.searchsorted(epochs, start - 0.5), numpy.searchsorted(epochs, end + 0.5)
    window = clock['effective_mhz'].values[lo:hi]
    if len(window) == 0:
        k = min(max(numpy.searchsorted(epochs, (start + end) / 2), 0), len(epochs) - 1)
        window = clock['effective_mhz'].values[k:k + 1]
    return float(window.mean()), float(window.min())


def merge_workload(db, workload, args, p, run_root, out_dir, clock, threads):
    prog = read_progress(os.path.join(out_dir, f'{db}_{workload}_progress.csv'))
    if prog.empty:
        return None
    if not clock.empty:
        wins = [clock_window(clock, s, e) for s, e in zip(prog['start_epoch'], prog['end_epoch'])]
        prog['effective_mhz_mean'] = [w[0] for w in wins]
        prog['effective_mhz_min'] = [w[1] for w in wins]
    _, plans = test_plans(p, db, workload)
    unit = 'seconds' if threads == 1 else 'thread-seconds'
    graphs_dir = f"{p['wl_base']}/{WORKLOADS[workload]}/dbs/{db}/created_graphs"
    rows = []
    for qid, group in prog.groupby('query_id', sort=False):
        idx = int(group['plan_index'].iloc[0])
        plan = plans[idx]
        udf = plan.udf.udf_name
        ok = group[group['status'] == 'ok']
        row = dict(host=HOST, query_id=qid, workload=workload, udf=udf, sql=plan.query, procedure=args.procedure,
                   threads=threads, udf_time_unit=unit, runs_ok=len(ok), runs_total=len(group),
                   status='ok' if len(ok) == len(group) else ';'.join(sorted(set(group['status']) - {'ok'})))
        for _, r in group.iterrows():
            row[f'total_time_r{int(r["run_no"])}_s'] = r['total_time_s']
            row[f'udf_time_r{int(r["run_no"])}_s'] = r['udf_time_s']
        if len(ok):
            total_s, udf_s = ok['total_time_s'].mean(), ok['udf_time_s'].mean()
            row.update(total_time_s=total_s, udf_time_s=udf_s,
                       total_time_cv=ok['total_time_s'].std(ddof=0) / total_s if len(ok) > 1 and total_s > 0 else None,
                       system_cpu_busy_max_pct=ok['system_cpu_busy_pct'].max())
            if unit == 'seconds':  # operator times are real seconds only with one thread
                row.update(sql_time_s=total_s - udf_s, udf_share=udf_s / total_s if total_s > 0 else None)
            if 'effective_mhz_min' in ok:
                row.update(effective_mhz_mean=ok['effective_mhz_mean'].mean(),
                           effective_mhz_min=ok['effective_mhz_min'].min())
            plan_json = load_plan(out_dir, qid)
            if plan_json is not None:
                _, ops, rows_in, est_in = udf_operator_info(plan_json, udf)
                row.update(udf_operator=' + '.join(ops), rows_into_udf=rows_in, udf_in_est_rows=est_in)
                if args.draw:
                    g = plan_graph(plan_json, udf)
                    nx.write_gpickle(g, os.path.join(out_dir, 'plans', f'{qid}.gpickle'))
                    draw_plan(g, os.path.join(out_dir, 'plans', f'{qid}.png'),
                              f'{qid}: total {total_s:.3f} s, UDF ({udf}) {udf_s:.3f} {unit}')
                    row.update(plan_gpickle=f'plans/{qid}.gpickle', plan_png=f'plans/{qid}.png')
        src = os.path.join(graphs_dir, f'{udf}.loopend.gpickle')
        if args.draw and os.path.exists(src):
            os.makedirs(os.path.join(out_dir, 'udf_graphs'), exist_ok=True)
            shutil.copyfile(src, os.path.join(out_dir, 'udf_graphs', f'{qid}.gpickle'))
            try:
                draw_udf_graph(nx.read_gpickle(src), os.path.join(out_dir, 'udf_graphs', f'{qid}.png'),
                               f'{qid}: UDF graph of {udf}')
                row['udf_graph_png'] = f'udf_graphs/{qid}.png'
            except Exception as e:
                print(f'    UDF graph drawing failed: {e}', flush=True)
            row['udf_graph_gpickle'] = f'udf_graphs/{qid}.gpickle'
        rows.append(row)
    prog.to_csv(os.path.join(out_dir, f'{db}_{workload}_runs.csv'), index=False)
    return pd.DataFrame(rows)


COLUMNS = ['host', 'query_id', 'workload', 'udf', 'udf_pos_in_query', 'num_joins', 'num_filters', 'udf_num_loops',
           'udf_num_branches', 'udf_num_comp_nodes', 'udf_num_np_calls', 'udf_num_math_calls', 'udf_in_card',
           'procedure', 'threads', 'status', 'runs_ok', 'runs_total', 'total_time_s', 'udf_time_s', 'udf_time_unit',
           'sql_time_s', 'udf_share', 'total_time_cv', 'udf_operator', 'rows_into_udf', 'udf_in_est_rows',
           'graceful_runtime_s', 'predicted_runtime_s', 'q_error', 'effective_mhz_mean', 'effective_mhz_min',
           'system_cpu_busy_max_pct', 'plan_gpickle', 'plan_png', 'udf_graph_gpickle', 'udf_graph_png', 'sql']

ABOUT = [
    ('host', 'Machine the timings were measured on.'),
    ('procedure', 'graceful = GRACEFUL\'s procedure (VACUUM ANALYZE on a copy, default threads, file order, per '
                  'repetition a hidden child run then a timed EXPLAIN ANALYZE, 30 s timeout); passes = ours '
                  '(read-only db, one pinned CPU, pass 1 in order, 2 reversed, 3+ random).'),
    ('total_time_s', 'Whole query time (DuckDB profiler `result`), mean over the successful repetitions or passes. '
                     'For graceful this is how GRACEFUL builds its label.'),
    ('total_time_rN_s / udf_time_rN_s', 'The single measurement of repetition or pass N.'),
    ('udf_time_s / udf_time_unit', 'Time of the plan operator(s) that call the UDF (udf_operator). With one thread '
                                   'it is seconds; with more, DuckDB sums operator time over threads, so it is '
                                   'thread-seconds and can exceed the total. sql_time_s and udf_share only exist '
                                   'for seconds.'),
    ('sql_time_s / udf_share', 'total_time_s - udf_time_s, and udf_time_s / total_time_s (one thread only).'),
    ('total_time_cv', 'Standard deviation / mean of total_time over the runs: how stable the measurement was.'),
    ('rows_into_udf / udf_in_est_rows', 'Actual rows entering the UDF operator(s), and DuckDB\'s own estimate of '
                                        'them (EC of the feeding operators), from the executed plan.'),
    ('effective_mhz_mean / _min', 'Real CPU clock from the Windows side (typeperf: frequency x % performance / '
                                  '100) during the runs of this query: mean, and the lowest sample.'),
    ('system_cpu_busy_max_pct', 'Highest system-wide CPU busy share (WSL /proc/stat) over the runs.'),
    ('graceful_runtime_s', "GRACEFUL's recorded runtime: the label the model was trained on (GRACEFUL's machine)."),
    ('predicted_runtime_s / q_error', "This spreadsheet's model's prediction; q_error = max(pred/label, label/pred)."),
    ('udf_in_card / num_joins / ...', "Query and UDF features as the model's data loader reports them."),
    ('plan_gpickle', 'Executed plan as a networkx DiGraph (nx.read_gpickle): nodes = operators with operator, time_s, '
                     'rows, est_rows, details, calls_udf; edges parent -> child. Raw plan: plans/<id>.json.gz.'),
    ('udf_graph_gpickle', "GRACEFUL's UDF graph for this query's UDF (the .loopend file the model reads)."),
]


def step_merge(args, p):
    root = run_dir(args)
    if not os.path.isdir(root):
        raise SystemExit(f'No timing run at {root}')
    clock = read_clock_logs(root)
    machine = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, 'machine_*.json')))]
    threads = machine[-1]['threads'] if machine else effective_threads(args)
    for db in args.dbs:
        out_dir = os.path.join(root, db)
        timed = {}
        for workload in args.workloads:
            frame = merge_workload(db, workload, args, p, root, out_dir, clock, threads)
            if frame is not None:
                timed[workload] = frame
        if not timed:
            print(f'== {db}: nothing timed yet', flush=True)
            continue
        for kind in ('baseline', 'augmented'):
            sheets = {}
            for workload, frame in timed.items():
                path = os.path.join(args.output_dir, 'predictions', db, f'{db}_{workload}_pred_{kind}.csv')
                if os.path.exists(path):
                    pred = pd.read_csv(path).drop(columns=['host', 'plan_index'], errors='ignore')
                    frame = frame.merge(pred.drop_duplicates('query_id', keep='last'), on='query_id', how='left')
                sheets[workload] = frame[[c for c in COLUMNS + [c for c in frame.columns if re.match(
                    r'(total|udf)_time_r\d+_s', c)] if c in frame.columns]]
            path = os.path.join(out_dir, f'{db}_{kind}.xlsx')
            with pd.ExcelWriter(path) as writer:
                about = pd.DataFrame(ABOUT, columns=['column', 'meaning'])
                model = MODELS[db][kind]
                about.loc[len(about)] = ['model', f'checkpoint *_{model[0]}' if model else 'no checkpoint here yet']
                about.loc[len(about)] = ['settings', f'procedure={args.procedure}, threads={threads}, seed={args.seed}, '
                                                     f'host={HOST}, clock log samples={len(clock)}']
                about.to_excel(writer, sheet_name='about', index=False)
                # as text: a transposed mixed column otherwise turns 1 into True (bool and int share a column)
                pd.DataFrame([{k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in m.items()}
                              for m in machine]).T.to_excel(writer, sheet_name='machine', header=False)
                for wl, frame in sheets.items():
                    frame.to_excel(writer, sheet_name=wl, index=False)
            print(f'  wrote {path}', flush=True)
    return 0


# ---------------------------------------------------------------- cap check

def step_capcheck(args, p):
    """Does the clock cap hold under load? Runs a CPU-bound DuckDB query on all threads for --capcheck_s seconds
    while typeperf logs the real clock on the Windows side, then prints the effective MHz during the load."""
    root = os.path.join(args.output_dir, 'capcheck')
    os.makedirs(root, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    threads = effective_threads(args)
    json.dump(dict(windows_utc_offset_s=windows_utc_offset_s(), clock_cap=args.clock_cap, threads=threads,
                   windows_max_processor_state=powershell(
                       'powercfg /q SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX | Select-String "Current AC|Current DC" '
                       '| ForEach-Object { $_.Line.Trim() }')),
              open(os.path.join(root, f'machine_{stamp}.json'), 'w'), indent=2)
    clock = ClockLogger(os.path.join(root, f'clock_{stamp}.csv'))
    if clock.proc is None:
        raise SystemExit('typeperf.exe not found: the cap check needs the Windows side (WSL)')
    time.sleep(3)  # idle samples before the load
    con = duckdb.connect()
    con.execute(f'SET threads TO {threads};')
    start = time.time()
    while time.time() - start < args.capcheck_s:
        con.execute('SELECT sum(hash(i)) FROM range(300000000) t(i)').fetchall()
    end = time.time()
    con.close()
    time.sleep(2)
    clock.stop()
    samples = read_clock_logs(root)
    samples = samples[samples['epoch'].between(start + 2, end)]  # skip the ramp-up second
    if samples.empty:
        raise SystemExit(f'No clock samples during the load (see {clock.path})')
    mhz = samples['effective_mhz']
    print(f'{threads} threads, {end - start:.0f} s load, cap "{args.clock_cap}": effective clock mean {mhz.mean():.0f} MHz, '
          f'min {mhz.min():.0f}, max {mhz.max():.0f} ({len(mhz)} samples)')
    return 0


def main() -> int:
    args = parse_args()
    p = paths(os.path.expanduser(args.graceful_dir))
    if not os.path.isdir(p['wl_base']):
        raise SystemExit(f"No workload_runs under {p['graceful']} (set --graceful_dir or GRACEFUL_DATA)")
    steps = {'time': step_time, 'predict': step_predict, 'merge': step_merge, 'capcheck': step_capcheck}
    return steps[args.step](args, p)


if __name__ == '__main__':
    raise SystemExit(main())
