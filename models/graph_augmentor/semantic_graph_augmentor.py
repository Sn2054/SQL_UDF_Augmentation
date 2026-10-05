from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import torch
from torch import nn
import torch.nn.functional as F

from models.zero_shot_models.utils import activations


UDF_NODE_TYPES = ("INV", "COMP", "BRANCH", "LOOP", "LOOPEND", "RET")
DEFAULT_REFINED_NODE_TYPES = ("COMP", "BRANCH", "LOOP", "LOOPEND", "RET")
REGION_KINDS = ("LOOP", "BRANCH", "SEQ")


class SemanticGraphAugmentor(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            pooling: str = "attention",
            refinement: str = "gated_residual",
            coarse_layers: int = 1,
            include_inv: bool = False,
            refine_ret: bool = True,
            seq_regions: bool = False,
            cfg_coarse_edges: bool = False,
            mq_num_queries: int = 8,
            activation_class_name: str = "LeakyReLU"):
        super().__init__()
        valid_pooling = {"mean", "sum", "max", "weighted_mean", "attention", "hybrid",
                         "hybrid_attn_max", "hybrid_max_wmean", "hybrid_max_wmean_gated",
                         "multi_query_attention"}
        valid_refinement = {"residual_sum", "gated_residual"}
        if pooling not in valid_pooling:
            raise ValueError(f"Unknown augment pooling {pooling}. Expected one of {sorted(valid_pooling)}")
        if refinement not in valid_refinement:
            raise ValueError(f"Unknown augment refinement {refinement}. Expected one of {sorted(valid_refinement)}")
        if pooling == "multi_query_attention" and (mq_num_queries <= 0 or hidden_dim % mq_num_queries != 0):
            raise ValueError(f"augment mq_num_queries ({mq_num_queries}) must be positive and divide "
                             f"hidden_dim ({hidden_dim})")

        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.refinement = refinement
        self.coarse_layers = coarse_layers
        self.include_inv = include_inv
        self.refine_ret = refine_ret
        self.seq_regions = seq_regions
        self.cfg_coarse_edges = cfg_coarse_edges

        self.attention_score = nn.Linear(hidden_dim, 1)
        if seq_regions:
            #? Pooling weights are shared across LOOP/BRANCH/SEQ supernodes; this tells them apart.
            #? Only created when enabled so checkpoints trained without SEQ regions still load.
            self.region_kind_embedding = nn.Embedding(len(REGION_KINDS), hidden_dim)
        if pooling == "hybrid":
            # Preserve both the region-wide signal and its strongest activations,
            # then restore the hidden size expected by downstream layers.
            self.hybrid_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        if pooling == "hybrid_attn_max":
            # Reuses self.attention_score for the attention branch; only the projection is new.
            self.hybrid_attn_max_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        if pooling == "hybrid_max_wmean":
            self.hybrid_max_wmean_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        if pooling == "hybrid_max_wmean_gated":
            #? The gate must map to a hidden_dim-sized [0,1] mixing vector, not reduce
            #? dimensionality like the concat+project hybrids do, so it needs its own layer.
            self.hybrid_max_wmean_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        if pooling == "multi_query_attention":
            #? Efficient Probing (Psomas et al., ICLR 2026): M learned queries, each with its own softmax
            #? over members, each pooling its own hidden_dim/M slice of a value projection (no key projection).
            #? Small init + 1/sqrt(d) scaling (as in EP) keeps the initial attention near-uniform (~ mean),
            #? since member embeddings are not normalized.
            self.mq_num_queries = mq_num_queries
            self.mq_queries = nn.Parameter(torch.randn(mq_num_queries, hidden_dim) * 0.02)
            self.mq_value = nn.Linear(hidden_dim, hidden_dim)
        self.coarse_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            activations.__dict__[activation_class_name](inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.last_coarse_fine_loss = None

    def forward(self, graph, feat_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        #? The augmentor only enriches encoded UDF node embeddings and leaves the original graph unchanged.
        self.last_coarse_fine_loss = None
        control_flow_graph = self._build_control_flow_graph(graph)
        regions, region_members = self._extract_regions(graph, control_flow_graph)
        if len(regions) == 0:
            return feat_dict

        region_embeddings = self._coarsen_regions(graph, feat_dict, regions, region_members)
        if region_embeddings is None:
            return feat_dict

        region_embeddings = self._coarse_message_passing(region_embeddings, region_members, control_flow_graph)
        refined = self._refine_nodes(feat_dict, regions, region_members, region_embeddings)
        self.last_coarse_fine_loss = self._coarse_fine_consistency_loss(
            refined,
            region_members,
            region_embeddings)
        return refined

    def _extract_regions(
            self, graph, control_flow_graph: Optional[nx.DiGraph] = None
    ) -> Tuple[List[Tuple[str, int]], List[List[Tuple[str, int]]]]:
        #? Each region is the full single-entry-single-exit (SESE) body of a LOOP or BRANCH node:
        #? every node forward-dominated by the head, up to its immediate post-dominator (where the
        #? loop body / branch arms reconverge) -- not just its direct one-hop neighbors.
        regions = []
        region_members = []
        if control_flow_graph is None:
            control_flow_graph = self._build_control_flow_graph(graph)
        if control_flow_graph is None:
            return regions, region_members

        children_by_dominator, immediate_post_dominator = self._compute_region_dominance(control_flow_graph)
        for region_type in ("LOOP", "BRANCH"):
            if region_type not in graph.ntypes:
                continue
            for region_node_id in range(graph.num_nodes(region_type)):
                head = (region_type, region_node_id)
                if head not in control_flow_graph:
                    continue
                members = self._extract_region_members(
                    control_flow_graph, children_by_dominator, immediate_post_dominator, head)
                if len(members) > 1:
                    regions.append((region_type, region_node_id))
                    region_members.append(sorted(members))

        if self.seq_regions:
            covered = {member for members in region_members for member in members}
            for segment_idx, segment in enumerate(self._extract_sequence_regions(control_flow_graph, covered)):
                regions.append(("SEQ", segment_idx))
                region_members.append(segment)
        return regions, region_members

    def _extract_sequence_regions(self, control_flow_graph: nx.DiGraph, covered) -> List[List[Tuple[str, int]]]:
        #? Nodes outside every LOOP/BRANCH region are the "block/sequence" regions of structural
        #? CFG analysis. Region members (incl. LOOPEND/join boundaries) act as separators, so each
        #? maximal straight-line run between, before or after constructs becomes its own component.
        uncovered = [node for node in control_flow_graph if node not in covered]
        segments = [sorted(component) for component in
                    nx.weakly_connected_components(control_flow_graph.subgraph(uncovered))]
        return sorted(segments)

    def _build_control_flow_graph(self, graph) -> Optional[nx.DiGraph]:
        #? Flatten the heterogeneous UDF node types into a single control-flow graph so region
        #? membership can be derived from actual CFG topology rather than DGL edge-type bookkeeping.
        control_flow_graph = nx.DiGraph()
        for node_type in UDF_NODE_TYPES:
            if node_type not in graph.ntypes:
                continue
            control_flow_graph.add_nodes_from((node_type, node_id) for node_id in range(graph.num_nodes(node_type)))

        if control_flow_graph.number_of_nodes() == 0:
            return None

        for src_type, edge_type, dst_type in graph.canonical_etypes:
            if src_type not in UDF_NODE_TYPES or dst_type not in UDF_NODE_TYPES:
                continue
            src_ids, dst_ids = graph.edges(etype=(src_type, edge_type, dst_type))
            src_list = src_ids.detach().cpu().tolist()
            dst_list = dst_ids.detach().cpu().tolist()
            for src_id, dst_id in zip(src_list, dst_list):
                control_flow_graph.add_edge((src_type, src_id), (dst_type, dst_id))
        return control_flow_graph

    def _compute_region_dominance(self, control_flow_graph: nx.DiGraph):
        #? A batched graph holds several UDF instances as disjoint components, each with its own
        #? entry (INV) node, so dominance is computed per component. Post-dominance is computed by
        #? reversing the component and routing every terminal node through one virtual exit, since a
        #? UDF can have multiple RET/dead-end nodes and post-dominance needs a single sink.
        children_by_dominator = defaultdict(list)
        immediate_post_dominator = {}
        for component in nx.weakly_connected_components(control_flow_graph):
            subgraph = control_flow_graph.subgraph(component)
            entries = [node for node in subgraph if subgraph.in_degree(node) == 0]
            if len(entries) != 1:
                # Ambiguous entry point for this component; skip rather than guess.
                continue
            entry = entries[0]

            immediate_dominator = nx.immediate_dominators(subgraph, entry)
            for node, dominator in immediate_dominator.items():
                if node != dominator:
                    children_by_dominator[dominator].append(node)

            reverse_subgraph = subgraph.reverse(copy=True)
            virtual_exit = object()
            reverse_subgraph.add_node(virtual_exit)
            for node in subgraph:
                if subgraph.out_degree(node) == 0:
                    reverse_subgraph.add_edge(virtual_exit, node)

            immediate_post_dom = nx.immediate_dominators(reverse_subgraph, virtual_exit)
            for node, post_dominator in immediate_post_dom.items():
                if node != virtual_exit:
                    immediate_post_dominator[node] = post_dominator

        return children_by_dominator, immediate_post_dominator

    def _dominated_set(self, children_by_dominator, root) -> Set[Tuple[str, int]]:
        dominated = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if node in dominated:
                continue
            dominated.add(node)
            stack.extend(children_by_dominator.get(node, ()))
        return dominated

    def _find_matching_loop_end(self, control_flow_graph: nx.DiGraph, loop_head) -> Optional[Tuple[str, int]]:
        #? A LOOP node has a single successor (its body), so post-dominance of the head only ever
        #? gives that immediate next node, not the loop's true end. Instead, walk the body forward
        #? and bracket-match LOOP/LOOPEND nesting depth to find the LOOPEND that actually closes
        #? this loop (as opposed to a nested loop's own end).
        stack = [(successor, 0) for successor in control_flow_graph.successors(loop_head)]
        seen = set()
        matches = set()
        while stack:
            node, depth = stack.pop()
            if node[0] == "LOOP":
                depth += 1
            elif node[0] == "LOOPEND":
                if depth == 0:
                    matches.add(node)
                    continue  # don't walk past this loop's own end
                depth -= 1

            state = (node, depth)
            if state in seen:
                continue
            seen.add(state)
            stack.extend((successor, depth) for successor in control_flow_graph.successors(node))

        if len(matches) == 1:
            return next(iter(matches))
        return None

    def _extract_region_members(
            self, control_flow_graph, children_by_dominator, immediate_post_dominator, head) -> Set[Tuple[str, int]]:
        members = self._dominated_set(children_by_dominator, head)
        if head[0] == "LOOP":
            boundary = self._find_matching_loop_end(control_flow_graph, head)
        else:
            boundary = immediate_post_dominator.get(head)
            if not isinstance(boundary, tuple):
                boundary = None

        if boundary is not None:
            # Everything from the reconvergence point onward is no longer exclusive to this
            # region; drop it, then keep the boundary node itself as the region's end marker
            # (the loop's matching LOOP_END, or the branch's join node).
            members -= self._dominated_set(children_by_dominator, boundary)
            members.add(boundary)
        return members

    def _coarsen_regions(self, graph, feat_dict, regions, region_members):
        #? Coarsening pools heterogeneous UDF node embeddings because all encoders emit the same hidden dimension.
        pooled_regions = []
        for (region_kind, _), members in zip(regions, region_members):
            member_embeddings = []
            member_weights = []
            for node_type, node_id in members:
                if node_type not in feat_dict or feat_dict[node_type].shape[0] <= node_id:
                    continue
                member_embeddings.append(feat_dict[node_type][node_id])
                member_weights.append(self._member_weight(graph, feat_dict[node_type], node_type, node_id))

            if len(member_embeddings) == 0:
                continue

            stacked = torch.stack(member_embeddings, dim=0)
            weights = torch.stack(member_weights, dim=0).to(stacked.device).reshape(-1, 1)
            pooled = self._pool_members(stacked, weights)
            if self.seq_regions:
                kind_idx = torch.tensor(REGION_KINDS.index(region_kind), device=pooled.device)
                pooled = pooled + self.region_kind_embedding(kind_idx)
            pooled_regions.append(pooled)

        if len(pooled_regions) != len(regions):
            return None
        return torch.stack(pooled_regions, dim=0)

    def _member_weight(self, graph, feature_tensor, node_type: str, node_id: int) -> torch.Tensor:
        if "out_degree" not in graph.nodes[node_type].data:
            return torch.ones((), device=feature_tensor.device)
        #? Match feature dtype so weighted pooling works even when graph degree data is integer-valued.
        out_degree = graph.nodes[node_type].data["out_degree"][node_id].to(
            device=feature_tensor.device,
            dtype=feature_tensor.dtype)
        return torch.log1p(torch.clamp(out_degree, min=0.0))

    def _pool_members(self, stacked: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return stacked.mean(dim=0)
        if self.pooling == "sum":
            return stacked.sum(dim=0)
        if self.pooling == "max":
            return stacked.max(dim=0).values
        if self.pooling == "weighted_mean":
            safe_weights = torch.clamp(weights, min=1.0)
            return (stacked * safe_weights).sum(dim=0) / safe_weights.sum(dim=0).clamp(min=1.0)
        if self.pooling == "attention":
            scores = self.attention_score(stacked)
            attn = torch.softmax(scores, dim=0)
            return (stacked * attn).sum(dim=0)
        if self.pooling == "hybrid":
            mean_pooled = stacked.mean(dim=0)
            max_pooled = stacked.max(dim=0).values
            return self.hybrid_projection(torch.cat([mean_pooled, max_pooled], dim=-1))
        if self.pooling == "hybrid_attn_max":
            scores = self.attention_score(stacked)
            attn = torch.softmax(scores, dim=0)
            attn_pooled = (stacked * attn).sum(dim=0)
            max_pooled = stacked.max(dim=0).values
            return self.hybrid_attn_max_projection(torch.cat([attn_pooled, max_pooled], dim=-1))
        if self.pooling == "hybrid_max_wmean":
            max_pooled = stacked.max(dim=0).values
            safe_weights = torch.clamp(weights, min=1.0)
            wmean_pooled = (stacked * safe_weights).sum(dim=0) / safe_weights.sum(dim=0).clamp(min=1.0)
            return self.hybrid_max_wmean_projection(torch.cat([max_pooled, wmean_pooled], dim=-1))
        if self.pooling == "hybrid_max_wmean_gated":
            max_pooled = stacked.max(dim=0).values
            safe_weights = torch.clamp(weights, min=1.0)
            wmean_pooled = (stacked * safe_weights).sum(dim=0) / safe_weights.sum(dim=0).clamp(min=1.0)
            gate = torch.sigmoid(self.hybrid_max_wmean_gate(torch.cat([max_pooled, wmean_pooled], dim=-1)))
            return gate * max_pooled + (1 - gate) * wmean_pooled
        if self.pooling == "multi_query_attention":
            num_members = stacked.shape[0]
            scores = (stacked @ self.mq_queries.t()) * self.hidden_dim ** -0.5  # [n, M]
            attn = torch.softmax(scores, dim=0)  # per query, over members
            values = self.mq_value(stacked).view(num_members, self.mq_num_queries, -1)  # [n, M, d/M]
            return (attn.unsqueeze(-1) * values).sum(dim=0).reshape(self.hidden_dim)
        raise ValueError(f"Unknown augment pooling {self.pooling}")

    def _coarse_message_passing(
            self, region_embeddings: torch.Tensor, region_members,
            control_flow_graph: Optional[nx.DiGraph] = None) -> torch.Tensor:
        if self.coarse_layers <= 0 or region_embeddings.shape[0] <= 1:
            return region_embeddings

        neighbors = self._region_neighbors(region_members, control_flow_graph)
        hidden = region_embeddings
        for _ in range(self.coarse_layers):
            messages = []
            for region_idx, region_neighbors in enumerate(neighbors):
                if len(region_neighbors) == 0:
                    messages.append(torch.zeros_like(hidden[region_idx]))
                    continue
                neighbor_ids = torch.tensor(sorted(region_neighbors), device=hidden.device)
                messages.append(hidden.index_select(0, neighbor_ids).mean(dim=0))
            message_tensor = torch.stack(messages, dim=0)
            hidden = self.layer_norm(hidden + self.coarse_update(torch.cat([hidden, message_tensor], dim=-1)))
        return hidden

    def _region_neighbors(self, region_members, control_flow_graph: Optional[nx.DiGraph] = None) -> List[Set[int]]:
        #? Regions are connected when they overlap through at least one fine UDF node (S^T S).
        shared_member_regions = defaultdict(list)
        for region_idx, members in enumerate(region_members):
            for member in members:
                shared_member_regions[member].append(region_idx)

        neighbors = [set() for _ in range(len(region_members))]
        for region_ids in shared_member_regions.values():
            for src_id in region_ids:
                for dst_id in region_ids:
                    if src_id != dst_id:
                        neighbors[dst_id].add(src_id)

        if self.cfg_coarse_edges and control_flow_graph is not None:
            #? SESE regions that run one after another share no members, so overlap alone only yields
            #? nesting edges. Also link regions joined by a CFG edge (the S^T A S term of coarsening).
            for src_node, dst_node in control_flow_graph.edges:
                for src_id in shared_member_regions.get(src_node, ()):
                    for dst_id in shared_member_regions.get(dst_node, ()):
                        if src_id != dst_id:
                            neighbors[dst_id].add(src_id)
                            neighbors[src_id].add(dst_id)
        return neighbors

    def _refine_nodes(self, feat_dict, regions, region_members, region_embeddings):
        #? Refinement writes the coarse context back into the original UDF node tensors without changing shapes.
        refined = dict(feat_dict)
        context_by_type = {}
        count_by_type = {}
        refined_types = set(DEFAULT_REFINED_NODE_TYPES)
        if self.include_inv:
            refined_types.add("INV")
        if not self.refine_ret:
            refined_types.discard("RET")

        for node_type in refined_types:
            if node_type in feat_dict:
                context_by_type[node_type] = torch.zeros_like(feat_dict[node_type])
                count_by_type[node_type] = torch.zeros(
                    (feat_dict[node_type].shape[0], 1),
                    dtype=feat_dict[node_type].dtype,
                    device=feat_dict[node_type].device,
                )

        for region_idx, members in enumerate(region_members):
            for node_type, node_id in members:
                if node_type not in context_by_type or context_by_type[node_type].shape[0] <= node_id:
                    continue
                context_by_type[node_type][node_id] += region_embeddings[region_idx]
                count_by_type[node_type][node_id] += 1

        for node_type, context in context_by_type.items():
            mask = count_by_type[node_type] > 0
            if not mask.any():
                continue
            context = context / count_by_type[node_type].clamp(min=1.0)
            projected = self.context_projection(context)
            original = feat_dict[node_type]
            if self.refinement == "residual_sum":
                updated = self.layer_norm(original + projected)
            elif self.refinement == "gated_residual":
                gate = torch.sigmoid(self.gate(torch.cat([original, context], dim=-1)))
                updated = self.layer_norm(original + gate * projected)
            else:
                raise ValueError(f"Unknown augment refinement {self.refinement}")
            refined[node_type] = torch.where(mask, updated, original)

        return refined

    def _coarse_fine_consistency_loss(self, feat_dict, region_members, region_embeddings):
        #? Keep refined fine nodes directionally consistent with their coarse semantic region embeddings.
        fine_embeddings = []
        coarse_embeddings = []
        for region_idx, members in enumerate(region_members):
            for node_type, node_id in members:
                if node_type not in feat_dict or feat_dict[node_type].shape[0] <= node_id:
                    continue
                fine_embeddings.append(feat_dict[node_type][node_id])
                coarse_embeddings.append(region_embeddings[region_idx])

        if len(fine_embeddings) == 0:
            return None

        fine_tensor = F.normalize(torch.stack(fine_embeddings, dim=0), dim=-1)
        coarse_tensor = F.normalize(torch.stack(coarse_embeddings, dim=0), dim=-1)
        return F.mse_loss(fine_tensor, coarse_tensor)
