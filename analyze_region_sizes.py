"""Measure supernode (region) sizes on the real UDF graphs, with SEQ regions off and on.

Reads every UDF graph (`<root>/**/created_graphs/*<suffix>.gpickle`), rebuilds the typed DGL graph the
way `udf_to_graph` does (VAR/column nodes dropped, node types mapped through `lookup`), and calls the
real `SemanticGraphAugmentor._extract_regions`. CPU only: it never touches CUDA or any feature tensors.

    python analyze_region_sizes.py --root <DATASET_BASE> [--suffix .loopend.lazy] [--no-loop-edge]

Counts are per unique UDF graph (not weighted by how many plans use each UDF).
"""
import argparse
import glob
import os
from collections import Counter, defaultdict

import dgl
import networkx as nx
import numpy as np
import torch

from models.dataset.plan_graph_batching.dd_plan_batching import lookup
from models.graph_augmentor.semantic_graph_augmentor import SemanticGraphAugmentor

BUCKETS = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5-6", 5, 6), ("7-10", 7, 10), ("11+", 11, 10 ** 9)]


def to_dgl(udf_graph: nx.DiGraph, add_loop_loopend_edge: bool = True):
    #? Same node/edge selection as udf_to_graph: ids are assigned per node type in encounter order, VAR
    #? nodes (column inputs) are not part of the UDF graph, and without the loop->loopend edge the
    #? shortcut edge is dropped whenever the loop head has another successor.
    node_ids, counts, edges = {}, Counter(), defaultdict(list)

    def node_id(n):
        if n not in node_ids:
            t = lookup[udf_graph.nodes[n]["type"]]
            node_ids[n] = counts[t]
            counts[t] += 1
        return node_ids[n]

    for src, dst in udf_graph.edges:
        if udf_graph.nodes[src]["type"] == "VAR":
            continue
        s_type, d_type = lookup[udf_graph.nodes[src]["type"]], lookup[udf_graph.nodes[dst]["type"]]
        if not add_loop_loopend_edge and (s_type, d_type) == ("LOOP", "LOOPEND") \
                and len(list(udf_graph.successors(src))) > 1:
            continue
        edges[(s_type, f"{s_type}_{d_type}", d_type)].append((node_id(src), node_id(dst)))
    if not edges:
        return None
    return dgl.heterograph(dict(edges), num_nodes_dict={t: n for t, n in counts.items()})


def bucket_table(sizes):
    n = max(len(sizes), 1)
    return {name: 100.0 * sum(lo <= s <= hi for s in sizes) / n for name, lo, hi in BUCKETS}


def summarize(label, sizes_by_kind):
    allsizes = [s for v in sizes_by_kind.values() for s in v]
    print(f"\n[{label}] regions: {len(allsizes)}  mean {np.mean(allsizes):.2f}  median {np.median(allsizes):.0f}  "
          f"min {min(allsizes)}  max {max(allsizes)}")
    print("  share of regions by member count (%):  " + "  ".join(f"{k}:{v:5.1f}" for k, v in bucket_table(allsizes).items()))
    for kind in ("LOOP", "BRANCH", "SEQ"):
        s = sizes_by_kind.get(kind)
        if s:
            print(f"  {kind:6s} n={len(s):6d} ({100 * len(s) / len(allsizes):4.1f}%)  mean {np.mean(s):.2f}  "
                  f"median {np.median(s):.0f}  max {max(s)}  | " +
                  "  ".join(f"{k}:{v:4.1f}" for k, v in bucket_table(s).items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root (e.g. Graceful_data) searched recursively")
    ap.add_argument("--suffix", default=".loopend.lazy", help="gpickle suffix of the config under test")
    ap.add_argument("--no-loop-edge", action="store_true", help="drop LOOP->LOOPEND edges (config without 'loopedge')")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "**", "created_graphs", f"*{args.suffix}.gpickle"), recursive=True))
    print(f"{len(files)} UDF graphs matching *{args.suffix}.gpickle under {args.root}")
    assert files, "no graphs found: check --root and --suffix"

    modes = {"loop/branch only (seq off)": SemanticGraphAugmentor(hidden_dim=4, seq_regions=False),
             "loop/branch + SEQ (as in the EP sweep)": SemanticGraphAugmentor(hidden_dim=4, seq_regions=True)}
    sizes = {m: defaultdict(list) for m in modes}
    per_db = defaultdict(lambda: defaultdict(list))
    udf_nodes, regions_per_udf, skipped = [], {m: [] for m in modes}, 0
    for path in files:
        g = to_dgl(nx.read_gpickle(path), add_loop_loopend_edge=not args.no_loop_edge)
        if g is None:
            skipped += 1
            continue
        udf_nodes.append(sum(g.num_nodes(t) for t in g.ntypes))
        db = os.path.basename(os.path.dirname(os.path.dirname(path)))
        for mode, aug in modes.items():
            regions, members = aug._extract_regions(g)
            regions_per_udf[mode].append(len(regions))
            for (kind, _), m in zip(regions, members):
                sizes[mode][kind].append(len(m))
                if "SEQ" in mode:
                    per_db[db][kind].append(len(m))
    print(f"graphs used {len(udf_nodes)} (skipped {skipped} without edges); UDF nodes per graph: mean "
          f"{np.mean(udf_nodes):.1f}, median {np.median(udf_nodes):.0f}, max {max(udf_nodes)}")
    for mode in modes:
        summarize(mode, sizes[mode])
        print(f"  regions per UDF graph: mean {np.mean(regions_per_udf[mode]):.1f}, "
              f"median {np.median(regions_per_udf[mode]):.0f}, max {max(regions_per_udf[mode])}")
    print("\nper database, SEQ on (all kinds): n regions / mean size / % of regions with exactly 4 members")
    for db, by_kind in sorted(per_db.items()):
        s = [x for v in by_kind.values() for x in v]
        print(f"  {db:24s} {len(s):6d}  {np.mean(s):5.2f}  {100 * sum(x == 4 for x in s) / len(s):5.1f}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
