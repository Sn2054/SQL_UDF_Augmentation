#!/usr/bin/env python3
"""Inspect the augmentation module's coarse-to-fine "refinement" step per query.

`models/graph_augmentor/semantic_graph_augmentor.py::SemanticGraphAugmentor` finds every
LOOP/BRANCH node's local neighborhood ("region"), pools each region into one coarse embedding
(--augment_pooling), optionally message-passes between coarse regions (--augment_coarse_layers),
then writes that coarse context back into each region member's *original* node embedding via a
gated residual (--augment_refinement): `updated = original + gate * projected_context`.

This script measures, per UDF-graph node, the cosine similarity between that node's embedding
*before* this refinement and *after* it. Similarity near 1 means the node was barely touched by
the global/coarse context; low similarity means it absorbed a lot of it. Comparing this across the
worst-predicted and best-predicted queries for a run (see find_worst_queries.py) answers: does the
refinement step actually pay more attention to complexity-relevant nodes (BRANCH/LOOP) on queries
the model gets right than on ones it gets wrong?

It works by registering a forward hook on `model.graph_augmentor` (a plain nn.Module, called once
per forward as `feat_dict = self.graph_augmentor(g, feat_dict)`), then running inference one query
at a time -- so the hook's captured before/after tensors are unambiguously about a single query's
graph, not a batch mixing several queries' nodes together.

Usage (reading the worst/best CSV find_worst_queries.py already wrote):

    python analyze_refinement_similarity.py \\
        --test_db consumer \\
        --model_config ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge \\
        --model_dir saved/models/aug_est_..._augpoolweighted_mean_..._augnoRET \\
        --model_name aug_est_..._augpoolweighted_mean_..._20260902_141131_012 \\
        --augment_pooling weighted_mean --augment_refinement gated_residual --augment_coarse_layers 1 \\
        --queries_csv results/worst_queries/consumer_..._worst10_best10.csv \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/polaris_matplotlib")

import pandas as pd
import torch
import torch.nn.functional as F

import models.dataset.plan_graph_batching.dd_plan_batching as dd_plan_batching
from find_worst_queries import (
    UDF_NAME_PATTERN, build_config, load_model, load_udf_source, resolve_model_name, run_query_inference,
    safe_filename_part, str2bool,
)
from models.dataset.dataset_creation import read_workload_runs
from cross_db_benchmark.benchmark_tools.utils import load_json

REFINED_NODE_TYPES = ("INV", "COMP", "BRANCH", "LOOP", "LOOPEND", "RET")

# Matches dd_plan_batching.py's own `lookup` dict: translates the UDF-source AST graph's node
# type names (as written by udf_graph/create_graph.py, e.g. from a gpickle file) to the
# abbreviated DGL node type used everywhere else in this script and in feat_dict.
NX_TYPE_TO_DGL_TYPE = {
    'RETURN': 'RET', 'COMP': 'COMP', 'BRANCH': 'BRANCH', 'LOOP_HEAD': 'LOOP',
    'LOOP_END': 'LOOPEND', 'INVOCATION': 'INV',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--test_db', required=True)
    parser.add_argument('--wl_base_path', default='/mnt/shared/data/dataset/Graceful_data/workload_runs/')
    parser.add_argument('--pushdown_plans_path', default=None)
    parser.add_argument('--pullup_plans_path', default=None)
    parser.add_argument('--statistics_file', default=None)
    parser.add_argument('--data_keyword', default='complex_dd')
    parser.add_argument('--card_type', default='est', choices=['est', 'act', 'dd', 'wj'])
    parser.add_argument('--card_est_udf_sel', type=int, default=None)
    parser.add_argument('--max_runtime', type=int, default=30)
    parser.add_argument('--min_runtime_ms', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--model_config', required=True)
    parser.add_argument('--mp_ignore_udf', type=str2bool, default=None)
    parser.add_argument('--work_with_udf_repr', default=False, action='store_true')
    parser.add_argument('--test_augment', type=str2bool, default=True)
    parser.add_argument('--augment_pooling', default='attention',
                        choices=['mean', 'sum', 'max', 'weighted_mean', 'attention', 'hybrid'])
    parser.add_argument('--augment_refinement', default='gated_residual', choices=['residual_sum', 'gated_residual'])
    parser.add_argument('--augment_coarse_layers', type=int, default=1)
    parser.add_argument('--augment_include_inv', type=str2bool, default=False)
    parser.add_argument('--augment_refine_ret', type=str2bool, default=True)

    parser.add_argument('--queries_csv', required=True,
                        help='A worst/best CSV from find_worst_queries.py (needs workload, sql, rank_group columns)')
    parser.add_argument('--output_dir', default='results/worst_queries')
    return parser.parse_args()


def find_matching_plan(plans, sql: str):
    for plan in plans:
        if plan.query == sql:
            return plan
    return None


def capture_refinement(model, config: dict, plan, plans_path: str, statistics_file: str,
                       feature_statistics: dict, card_type: str, card_est_udf_sel: Optional[int]):
    """Besides the augmentor's own before/after feat_dicts, also intercept two more things during
    the same single-query inference call:

    - _coarse_message_passing -- the last step before _refine_nodes -- to recover its two inputs
      (region_embeddings: each region's raw pooled embedding, straight out of --augment_pooling,
      before any cross-region communication; region_members: which (node_type, node_id) nodes make
      up each region) and its output (the same regions' embeddings after --augment_coarse_layers
      rounds of message-passing with neighboring regions -- this is the actual "context" vector
      _refine_nodes injects into nodes). Diffing input vs. output is what lets the CSV report how
      much a region's own representation moved *because it talked to its neighbors*, separately
      from how much a node moved because it absorbed that (already-settled) context. Both are local
      variables inside SemanticGraphAugmentor.forward, not otherwise exposed; a forward hook only
      works on nn.Module instances, but _coarse_message_passing is a plain bound method, so it's
      intercepted by temporarily shadowing the instance attribute instead.

    - dd_plan_batching.create_udf_feat_list -- called once per UDF-graph node while the dataset is
      built (well before the model even runs), always immediately followed by
      `udf_node_features[dgl_type].append(...)`. So the order these calls happen in, per DGL type,
      is exactly the per-type index (`node_id`) used everywhere else (feat_dict[node_type][node_id],
      the CSV's node_id column, ...). Each call receives the *original* AST node (`src`) and the
      full parsed UDF graph (`udf_graph`, loaded from the .gpickle udf_graph/create_graph.py wrote),
      which carries that node's real source line number as its `lineno` attribute -- recovering it
      here, in call order, is what lets the CSV report an actual line instead of an opaque index."""
    augmentor = model.graph_augmentor
    captured: Dict[str, object] = {}
    node_source_by_type: Dict[str, list] = {}
    original_coarse_mp = augmentor._coarse_message_passing
    original_feat_list_fn = dd_plan_batching.create_udf_feat_list

    def patched_coarse_mp(region_embeddings, region_members):
        result = original_coarse_mp(region_embeddings, region_members)
        captured['region_members'] = region_members
        captured['region_embeddings_before_mp'] = region_embeddings.detach().clone()
        captured['region_embeddings'] = result.detach().clone()
        return result

    def patched_feat_list(feature_statistics_arg, plan_featurization_arg, src, udf_graph, *args, **kwargs):
        result = original_feat_list_fn(feature_statistics_arg, plan_featurization_arg, src, udf_graph,
                                       *args, **kwargs)
        node_attrs = udf_graph.nodes[src]
        node_type = NX_TYPE_TO_DGL_TYPE.get(node_attrs.get('type'))
        if node_type is not None:
            # udf_graph/create_graph.py splits one source line's expression into several COMP nodes
            # -- one per library call (lib_onehot), plus one more for whatever combines their
            # results (ops), so e.g. `numpy.mod(a,b)-numpy.power(c,d)` becomes 3 distinct nodes all
            # sharing the same lineno. lib_onehot/ops/cmops are what actually tell those apart.
            node_source_by_type.setdefault(node_type, []).append({
                'lineno': node_attrs.get('lineno'),
                'lib_onehot': node_attrs.get('lib_onehot'),
                'ops': node_attrs.get('ops'),
                'cmops': node_attrs.get('cmops'),
            })
        return result

    def hook(_module, inputs, output):
        _graph, feat_dict = inputs
        captured['before'] = {k: v.detach().clone() for k, v in feat_dict.items()}
        captured['after'] = {k: v.detach().clone() for k, v in output.items()}

    augmentor._coarse_message_passing = patched_coarse_mp
    dd_plan_batching.create_udf_feat_list = patched_feat_list
    handle = augmentor.register_forward_hook(hook)
    try:
        # create_udf_feat_list runs during dataset/graph construction, which DataLoader offloads to
        # separate worker processes whenever num_workers > 0 -- and a monkeypatch applied only in
        # this (main) process is invisible there, so create_udf_feat_list calls in a worker would
        # silently use the *original* function and node_source_by_type would stay empty. Force 0:
        # there's exactly one query in this batch anyway, so parallelism buys nothing here.
        run_query_inference(model, config, plans_path, statistics_file, feature_statistics, card_type,
                            card_est_udf_sel, num_workers=0, plans_override=[plan])
    finally:
        handle.remove()
        augmentor._coarse_message_passing = original_coarse_mp
        dd_plan_batching.create_udf_feat_list = original_feat_list_fn

    node_source = {(node_type, node_id): attrs
                  for node_type, attrs_list in node_source_by_type.items()
                  for node_id, attrs in enumerate(attrs_list)}

    return (captured.get('before'), captured.get('after'), captured.get('region_members'),
            captured.get('region_embeddings_before_mp'), captured.get('region_embeddings'), node_source)


def explain_refinement(augmentor, before: dict, region_members, region_embeddings_before_mp,
                       region_embeddings) -> dict:
    """Recompute _refine_nodes's own context/gate/projection math (using the model's real,
    already-trained context_projection/gate layers) per refined node, so the aggregate
    cosine-similarity numbers can be attributed to a concrete cause: how many regions a node
    belongs to, how wide open its gate was, and how large the injected update was relative to
    the node's own prior embedding. Also reports how much the *context itself* moved during
    coarse message-passing (region_embeddings_before_mp -> region_embeddings), averaged over
    whichever region(s) this node belongs to -- the same averaging _refine_nodes itself does to
    build `context` in the first place."""
    if region_members is None or region_embeddings is None:
        return {}

    member_regions: Dict[tuple, list] = {}
    for region_idx, members in enumerate(region_members):
        for member in members:
            member_regions.setdefault(member, []).append(region_idx)

    details = {}
    with torch.no_grad():
        for (node_type, node_id), region_idxs in member_regions.items():
            if node_type not in before:
                continue
            original = before[node_type][node_id]
            context = region_embeddings[region_idxs].mean(dim=0)
            projected = augmentor.context_projection(context)
            gate = torch.sigmoid(augmentor.gate(torch.cat([original, context], dim=-1)))
            injected = gate * projected

            context_before_mp = region_embeddings_before_mp[region_idxs].mean(dim=0)
            context_change_cos = F.cosine_similarity(context_before_mp, context, dim=0)
            context_change_delta_norm = (context - context_before_mp).norm()

            details[(node_type, node_id)] = {
                'num_regions': len(region_idxs),
                'gate_mean': gate.mean().item(),
                'original_norm': original.norm().item(),
                'context_norm': context.norm().item(),
                'injected_norm': injected.norm().item(),
                'relative_injection': (injected.norm() / original.norm().clamp(min=1e-8)).item(),
                'context_change_cosine_similarity': context_change_cos.item(),
                'context_change_delta_norm': context_change_delta_norm.item(),
            }
    return details


def resolve_code_line(lineno: Optional[int], udf_source_lines: Optional[list]) -> Optional[str]:
    """udf_graph/create_graph.py assigns `lineno` from Python's ast module, which is 1-indexed and
    counts the `def ...:` line itself as line 1 -- exactly udf_source_lines[0] in the raw source
    list loaded from dbs/<db>/sql_scripts/udfs.json (see find_worst_queries.py::load_udf_source)."""
    if lineno is None or udf_source_lines is None or not (1 <= lineno <= len(udf_source_lines)):
        return None
    return udf_source_lines[lineno - 1].strip()


def summarize_node_detail(attrs: Optional[dict]) -> Optional[str]:
    """Several COMP (and LOOP_HEAD) nodes can share one udf_lineno/udf_code_line -- create_graph.py
    gives each library call on a line its own node (lib_onehot), plus one more node for whatever
    combines their results (ops); BRANCH nodes carry their comparison operator (cmops) instead.
    This turns those attributes into a short label so otherwise-identical-looking rows for the same
    line are distinguishable."""
    if not attrs:
        return None
    parts = []
    lib = attrs.get('lib_onehot')
    if lib and lib != 'null':
        parts.append(f'call={lib}')
    ops = attrs.get('ops')
    if ops:
        parts.append(f'ops={",".join(ops) if isinstance(ops, (list, tuple)) else ops}')
    cmops = attrs.get('cmops')
    if cmops:
        parts.append(f'cmp={cmops}')
    return '; '.join(parts) if parts else None


def per_node_cosine(before: dict, after: dict, explain: dict, node_source: dict,
                    udf_source_lines: Optional[list]) -> pd.DataFrame:
    rows = []
    for node_type in REFINED_NODE_TYPES:
        if node_type not in before or node_type not in after:
            continue
        b, a = before[node_type], after[node_type]
        if b.shape != a.shape:
            continue
        cosine = F.cosine_similarity(b, a, dim=-1)
        # Cosine similarity only captures direction: a node could be rotated a lot while barely
        # changing size, or rescaled a lot while barely changing direction. This is the raw
        # Euclidean distance between the before/after vectors -- how far the embedding actually
        # moved, in the same units as original_norm/context_norm/injected_norm below.
        delta_norm = (a - b).norm(dim=-1)
        was_refined = ~torch.all(torch.isclose(b, a, atol=1e-7, rtol=1e-5), dim=-1)
        for node_id in range(b.shape[0]):
            attrs = node_source.get((node_type, node_id))
            lineno = attrs.get('lineno') if attrs else None
            row = {
                'node_type': node_type,
                'node_id': node_id,
                'udf_lineno': lineno,
                'udf_code_line': resolve_code_line(lineno, udf_source_lines),
                'udf_node_detail': summarize_node_detail(attrs),
                'cosine_similarity': cosine[node_id].item(),
                'embedding_delta_norm': delta_norm[node_id].item(),
                'was_refined': bool(was_refined[node_id].item()),
            }
            row.update(explain.get((node_type, node_id), {}))
            rows.append(row)
    return pd.DataFrame(rows)


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

    config = build_config(
        args.model_config, args.data_keyword, args.mp_ignore_udf, args.work_with_udf_repr, augment=True,
        test_augment=args.test_augment, augment_pooling=args.augment_pooling,
        augment_refinement=args.augment_refinement, augment_coarse_layers=args.augment_coarse_layers,
        augment_include_inv=args.augment_include_inv, augment_refine_ret=args.augment_refine_ret)
    config['max_runtime'] = args.max_runtime
    config['min_runtime_ms'] = args.min_runtime_ms

    model_name = resolve_model_name(args.model_dir, args.model_name)
    print(f'Loading augmented model: {args.model_dir}/{model_name}')
    model = load_model(args.model_dir, model_name, config, feature_statistics, args.device)
    if model.graph_augmentor is None:
        raise SystemExit('This checkpoint has no graph_augmentor (augment=False) -- nothing to inspect.')

    queries = pd.read_csv(args.queries_csv)
    plans_cache = {}
    udf_source_cache = {}
    node_frames = []
    skipped_no_region = 0
    skipped_not_found = 0

    for _, row in queries.iterrows():
        workload = row['workload']
        if workload not in plans_cache:
            plans, _ = read_workload_runs([plans_paths[workload]], min_runtime_ms=config['min_runtime_ms'],
                                          max_runtime=config['max_runtime'])
            plans_cache[workload] = plans
        if workload not in udf_source_cache:
            udf_source_cache[workload] = load_udf_source(plans_paths[workload], args.test_db)

        plan = find_matching_plan(plans_cache[workload], row['sql'])
        if plan is None:
            skipped_not_found += 1
            continue

        udf_name_match = UDF_NAME_PATTERN.search(row['sql'])
        udf_source_lines = (udf_source_cache[workload].get(udf_name_match.group(0))
                            if udf_name_match else None)

        before, after, region_members, region_embeddings_before_mp, region_embeddings, node_source = \
            capture_refinement(model, config, plan, plans_paths[workload], statistics_file, feature_statistics,
                               args.card_type, args.card_est_udf_sel)
        if before is None:
            # no LOOP/BRANCH region found for this UDF -- the augmentor short-circuits and never
            # calls its own forward internals that the hook would see meaningful tensors from.
            skipped_no_region += 1
            continue

        explain = explain_refinement(model.graph_augmentor, before, region_members, region_embeddings_before_mp,
                                     region_embeddings)
        node_df = per_node_cosine(before, after, explain, node_source, udf_source_lines)
        if node_df.empty:
            skipped_no_region += 1
            continue
        node_df['rank_group'] = row['rank_group']
        node_df['workload'] = workload
        node_df['sql'] = row['sql']
        node_df['aug_qerror'] = row.get('aug_qerror')
        # how many of this query's nodes the augmentor actually rewrote, out of all nodes in its UDF
        # graph -- i.e. how big a fraction of the query the refinement step touched at all.
        node_df['num_refined_nodes'] = int(node_df['was_refined'].sum())
        node_df['num_graph_nodes'] = len(node_df)
        # keeps queries in their original (worst-to-best) order after the source-line sort below,
        # which only reorders rows *within* one query.
        node_df['_query_order'] = len(node_frames)
        node_frames.append(node_df)

    if skipped_not_found:
        print(f'NOTE: {skipped_not_found} queries in {args.queries_csv} had no matching plan (sql text mismatch).')
    if skipped_no_region:
        print(f'NOTE: {skipped_no_region} queries had no LOOP/BRANCH region for the augmentor to refine.')

    if not node_frames:
        raise SystemExit('No queries produced refinement data; nothing to report.')

    detail = pd.concat(node_frames, ignore_index=True)
    # Sort by source line within each query so a loop's/branch's body -- COMP/BRANCH rows that are
    # already linked to it as a region member -- physically lands between its LOOP and LOOPEND
    # rows, instead of being grouped away under a separate node_type block.
    detail = detail.sort_values(['_query_order', 'udf_lineno'], na_position='last').drop(columns='_query_order')
    detail = detail.reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{safe_filename_part(args.test_db)}_{safe_filename_part(model_name)}_refinement_similarity'
    detail_path = output_dir / f'{stem}_per_node.csv'
    detail.to_csv(detail_path, index=False)

    print(f'\nWrote {len(detail)} per-node rows to {detail_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
