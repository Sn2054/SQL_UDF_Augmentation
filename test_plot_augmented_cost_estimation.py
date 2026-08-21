import argparse
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from plot_augmented_cost_estimation import format_settings, read_baseline_workloads


class PlotAugmentedCostEstimationTest(unittest.TestCase):
    def test_baseline_workloads_match_only_database_and_cardinality(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.xlsx"
            workbook = Workbook()
            worksheet = workbook.active
            worksheet.title = "Baseline"
            worksheet.append([
                "test_db", "cardinality_type",
                "pull_up_q50_mean", "pull_up_q95_mean", "pull_up_q99_mean",
                "push_down_q50_mean", "push_down_q95_mean", "push_down_q99_mean",
                "time_stamp", "epochs",
            ])
            worksheet.append([
                "accidents", "est",
                1.4, 9.3, 23.5, 1.7, 21.6, 57.6, "old", 50,
            ])
            workbook.save(path)
            workbook.close()

            actual = read_baseline_workloads(path, "ACCIDENTS", "EST")

            self.assertEqual(actual["pullup"]["q50"], 1.4)
            self.assertEqual(actual["pushdown"]["q99"], 57.6)

    def test_settings_include_test_augment(self):
        args = argparse.Namespace(
            seed=42,
            augment="True",
            test_augment="False",
            augment_pooling="hybrid",
            augment_refinement="gated_residual",
            augment_coarse_layers=1,
            augment_include_inv="False",
            augment_refine_ret="False",
            lambda_struct=0.0,
        )

        self.assertIn("TEST_AUGMENT=False", format_settings(args))


if __name__ == "__main__":
    unittest.main()
