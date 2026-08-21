import unittest

import torch

from models.graph_augmentor.semantic_graph_augmentor import SemanticGraphAugmentor


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


if __name__ == "__main__":
    unittest.main()
