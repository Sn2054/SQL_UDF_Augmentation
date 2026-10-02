import unittest

import dgl
import torch

from models.graph_augmentor.semantic_graph_augmentor import SemanticGraphAugmentor


def _udf_graph(edges, num_nodes):
    #? edges: list of ((src_type, src_id), (dst_type, dst_id)), grouped into typed DGL relations.
    data_dict = {}
    for (src_type, src_id), (dst_type, dst_id) in edges:
        data_dict.setdefault((src_type, f"{src_type}_{dst_type}", dst_type), []).append((src_id, dst_id))
    return dgl.heterograph(data_dict, num_nodes_dict=num_nodes)


def _loop_then_branch_graph():
    # INV0 -> C0 -> LOOP0 -> C1 -> LOOPEND0 -> C2 -> C3 -> BRANCH0 -> {C4 | C5} -> C6 -> C7 -> RET0
    edges = [
        (("INV", 0), ("COMP", 0)), (("COMP", 0), ("LOOP", 0)), (("LOOP", 0), ("COMP", 1)),
        (("COMP", 1), ("LOOPEND", 0)), (("LOOPEND", 0), ("COMP", 2)), (("COMP", 2), ("COMP", 3)),
        (("COMP", 3), ("BRANCH", 0)), (("BRANCH", 0), ("COMP", 4)), (("BRANCH", 0), ("COMP", 5)),
        (("COMP", 4), ("COMP", 6)), (("COMP", 5), ("COMP", 6)), (("COMP", 6), ("COMP", 7)),
        (("COMP", 7), ("RET", 0)),
    ]
    return _udf_graph(edges, {"INV": 1, "COMP": 8, "LOOP": 1, "LOOPEND": 1, "BRANCH": 1, "RET": 1})


def _sibling_loops_graph(gap_between: bool):
    # INV0 -> LOOP0 -> C0 -> LOOPEND0 -> [C1 ->] LOOP1 -> C2 -> LOOPEND1 -> RET0
    edges = [
        (("INV", 0), ("LOOP", 0)), (("LOOP", 0), ("COMP", 0)), (("COMP", 0), ("LOOPEND", 0)),
        (("LOOP", 1), ("COMP", 2)), (("COMP", 2), ("LOOPEND", 1)), (("LOOPEND", 1), ("RET", 0)),
    ]
    if gap_between:
        edges += [(("LOOPEND", 0), ("COMP", 1)), (("COMP", 1), ("LOOP", 1))]
    else:
        edges += [(("LOOPEND", 0), ("LOOP", 1))]
    return _udf_graph(edges, {"INV": 1, "COMP": 3, "LOOP": 2, "LOOPEND": 2, "RET": 1})


class SemanticGraphAugmentorSequenceRegionTest(unittest.TestCase):
    def test_seq_regions_cover_prefix_middle_and_tail_gaps(self):
        augmentor = SemanticGraphAugmentor(hidden_dim=4, seq_regions=True)

        regions, region_members = augmentor._extract_regions(_loop_then_branch_graph())

        by_region = dict(zip(regions, region_members))
        self.assertEqual(by_region[("LOOP", 0)], [("COMP", 1), ("LOOP", 0), ("LOOPEND", 0)])
        self.assertEqual(by_region[("BRANCH", 0)], [("BRANCH", 0), ("COMP", 4), ("COMP", 5), ("COMP", 6)])
        seq_segments = [members for (kind, _), members in zip(regions, region_members) if kind == "SEQ"]
        self.assertEqual(seq_segments, [
            [("COMP", 0), ("INV", 0)],
            [("COMP", 2), ("COMP", 3)],
            [("COMP", 7), ("RET", 0)],
        ])

    def test_every_udf_node_is_covered_when_seq_regions_enabled(self):
        graph = _loop_then_branch_graph()
        augmentor = SemanticGraphAugmentor(hidden_dim=4, seq_regions=True)

        _, region_members = augmentor._extract_regions(graph)

        covered = {member for members in region_members for member in members}
        all_nodes = {(ntype, i) for ntype in graph.ntypes for i in range(graph.num_nodes(ntype))}
        self.assertEqual(covered, all_nodes)

    def test_disabled_flags_keep_only_loop_and_branch_regions(self):
        augmentor = SemanticGraphAugmentor(hidden_dim=4)

        regions, _ = augmentor._extract_regions(_loop_then_branch_graph())

        self.assertEqual(regions, [("LOOP", 0), ("BRANCH", 0)])
        self.assertFalse(hasattr(augmentor, "region_kind_embedding"))

    def test_sibling_loops_connect_through_seq_segment_only_with_cfg_edges(self):
        graph = _sibling_loops_graph(gap_between=True)
        for cfg_coarse_edges, expect_connected in ((False, False), (True, True)):
            augmentor = SemanticGraphAugmentor(hidden_dim=4, seq_regions=True, cfg_coarse_edges=cfg_coarse_edges)
            control_flow_graph = augmentor._build_control_flow_graph(graph)
            regions, region_members = augmentor._extract_regions(graph, control_flow_graph)

            neighbors = augmentor._region_neighbors(region_members, control_flow_graph)

            loop0, loop1 = regions.index(("LOOP", 0)), regions.index(("LOOP", 1))
            middle_seq = region_members.index([("COMP", 1)])
            self.assertEqual(middle_seq in neighbors[loop0] and loop1 in neighbors[middle_seq], expect_connected)

    def test_directly_adjacent_sibling_loops_connect_with_cfg_edges(self):
        graph = _sibling_loops_graph(gap_between=False)
        augmentor = SemanticGraphAugmentor(hidden_dim=4, cfg_coarse_edges=True)
        control_flow_graph = augmentor._build_control_flow_graph(graph)
        regions, region_members = augmentor._extract_regions(graph, control_flow_graph)

        neighbors = augmentor._region_neighbors(region_members, control_flow_graph)

        self.assertIn(regions.index(("LOOP", 1)), neighbors[regions.index(("LOOP", 0))])

    def test_forward_backward_with_seq_regions_for_all_poolings(self):
        graph = _loop_then_branch_graph()
        for pooling in ["mean", "sum", "max", "weighted_mean", "attention", "hybrid", "multi_query_attention"]:
            augmentor = SemanticGraphAugmentor(
                hidden_dim=4, pooling=pooling, seq_regions=True, cfg_coarse_edges=True, mq_num_queries=2)
            feat_dict = {ntype: torch.randn(graph.num_nodes(ntype), 4, requires_grad=True)
                         for ntype in graph.ntypes}

            refined = augmentor(graph, feat_dict)
            (sum(value.sum() for value in refined.values()) + augmentor.last_coarse_fine_loss).backward()

            self.assertIsNotNone(augmentor.region_kind_embedding.weight.grad)
            # C2 sits only in a SEQ segment, so it is now refined instead of passing through unchanged.
            self.assertFalse(torch.equal(refined["COMP"][2], feat_dict["COMP"][2]))


class SemanticGraphAugmentorPoolingTest(unittest.TestCase):
    def test_max_pooling_returns_featurewise_maximum(self):
        augmentor = SemanticGraphAugmentor(hidden_dim=3, pooling="max")
        stacked = torch.tensor([
            [1.0, 5.0, -2.0],
            [3.0, 2.0, -1.0],
            [2.0, 4.0, -3.0],
        ])
        weights = torch.ones(3, 1)

        actual = augmentor._pool_members(stacked, weights)

        torch.testing.assert_close(actual, torch.tensor([3.0, 5.0, -1.0]))

    def test_hybrid_pooling_fuses_mean_and_max_and_propagates_gradients(self):
        augmentor = SemanticGraphAugmentor(hidden_dim=2, pooling="hybrid")
        with torch.no_grad():
            augmentor.hybrid_projection.weight.copy_(torch.tensor([
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ]))
            augmentor.hybrid_projection.bias.zero_()

        stacked = torch.tensor([
            [1.0, 4.0],
            [3.0, 2.0],
        ], requires_grad=True)
        weights = torch.ones(2, 1)

        actual = augmentor._pool_members(stacked, weights)

        # The test projection adds each feature's mean and max values.
        torch.testing.assert_close(actual, torch.tensor([5.0, 7.0]))
        actual.sum().backward()
        self.assertIsNotNone(stacked.grad)
        self.assertTrue(torch.all(stacked.grad > 0))

    def test_hybrid_projection_is_only_created_for_hybrid_pooling(self):
        hybrid = SemanticGraphAugmentor(hidden_dim=4, pooling="hybrid")
        mean = SemanticGraphAugmentor(hidden_dim=4, pooling="mean")

        self.assertTrue(hasattr(hybrid, "hybrid_projection"))
        self.assertFalse(hasattr(mean, "hybrid_projection"))
        self.assertEqual(hybrid.hybrid_projection.in_features, 8)
        self.assertEqual(hybrid.hybrid_projection.out_features, 4)



class SemanticGraphAugmentorMultiQueryPoolingTest(unittest.TestCase):
    def test_output_shape_and_gradients_for_query_counts(self):
        torch.manual_seed(0)
        for num_queries in (1, 4, 8, 16):
            augmentor = SemanticGraphAugmentor(
                hidden_dim=128, pooling="multi_query_attention", mq_num_queries=num_queries)
            for num_members in (1, 4, 7):
                with self.subTest(num_queries=num_queries, num_members=num_members):
                    augmentor.zero_grad()
                    stacked = torch.randn(num_members, 128, requires_grad=True)

                    pooled = augmentor._pool_members(stacked, torch.ones(num_members, 1))
                    pooled.pow(2).sum().backward()

                    self.assertEqual(pooled.shape, (128,))
                    for grad in (augmentor.mq_value.weight.grad, stacked.grad):
                        self.assertTrue(torch.isfinite(grad).all())
                    self.assertTrue(torch.isfinite(augmentor.mq_queries.grad).all())
                    if num_members > 1:
                        # A single member gets softmax weight 1 regardless of the queries.
                        self.assertGreater(augmentor.mq_queries.grad.abs().sum().item(), 0.0)

    def test_query_count_must_divide_hidden_dim(self):
        for num_queries in (0, 3, 256):
            with self.assertRaises(ValueError):
                SemanticGraphAugmentor(hidden_dim=128, pooling="multi_query_attention", mq_num_queries=num_queries)
        # Other poolings ignore the query count.
        SemanticGraphAugmentor(hidden_dim=128, pooling="attention", mq_num_queries=3)

    def test_each_query_pools_its_own_value_slice(self):
        augmentor = SemanticGraphAugmentor(hidden_dim=4, pooling="multi_query_attention", mq_num_queries=2)
        with torch.no_grad():
            augmentor.mq_value.weight.copy_(torch.eye(4))
            augmentor.mq_value.bias.zero_()
            # Query 0 sharply selects the member with the largest feature 0, query 1 feature 3.
            augmentor.mq_queries.copy_(torch.tensor([[100.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 100.0]]))
        stacked = torch.tensor([
            [1.0, 2.0, 3.0, 0.0],
            [0.0, 5.0, 6.0, 1.0],
        ])

        actual = augmentor._pool_members(stacked, torch.ones(2, 1))

        # Dims 0-1 come from member 0 (query 0's winner), dims 2-3 from member 1 (query 1's winner).
        torch.testing.assert_close(actual, torch.tensor([1.0, 2.0, 6.0, 1.0]))

    def test_attention_starts_near_uniform(self):
        torch.manual_seed(0)
        augmentor = SemanticGraphAugmentor(hidden_dim=128, pooling="multi_query_attention")
        stacked = torch.randn(4, 128)

        actual = augmentor._pool_members(stacked, torch.ones(4, 1))

        # Small query init + 1/sqrt(d) scaling: initially ~ mean pooling of the projected values.
        torch.testing.assert_close(actual, augmentor.mq_value(stacked).mean(dim=0), atol=0.02, rtol=0.0)

    def test_mq_parameters_are_only_created_for_mq_pooling(self):
        mq = SemanticGraphAugmentor(hidden_dim=8, pooling="multi_query_attention", mq_num_queries=4)
        attention = SemanticGraphAugmentor(hidden_dim=8, pooling="attention")

        self.assertEqual(tuple(mq.mq_queries.shape), (4, 8))
        self.assertFalse(hasattr(attention, "mq_queries"))
        self.assertFalse(hasattr(attention, "mq_value"))


if __name__ == "__main__":
    unittest.main()
