import math
import unittest

import pandas as pd

from src.repeated_significance import _markdown_table, holm_adjust_pvalues, paired_wilcoxon_tests


class RepeatedSignificanceTests(unittest.TestCase):
    def test_holm_adjust_pvalues_is_monotonic_and_capped(self):
        adjusted = holm_adjust_pvalues([0.01, 0.04, 0.03, 0.90])

        self.assertEqual(len(adjusted), 4)
        self.assertEqual(adjusted[0], 0.04)
        self.assertEqual(adjusted[2], 0.09)
        self.assertEqual(adjusted[1], 0.09)
        self.assertEqual(adjusted[3], 0.90)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))

    def test_paired_wilcoxon_tests_uses_matched_seeds(self):
        frame = pd.DataFrame(
            [
                {"seed": 1, "model": "XGBoost", "pr_auc": 0.90, "f1": 0.80},
                {"seed": 1, "model": "Random Forest", "pr_auc": 0.85, "f1": 0.76},
                {"seed": 2, "model": "XGBoost", "pr_auc": 0.91, "f1": 0.82},
                {"seed": 2, "model": "Random Forest", "pr_auc": 0.86, "f1": 0.77},
                {"seed": 3, "model": "XGBoost", "pr_auc": 0.92, "f1": 0.81},
                {"seed": 3, "model": "Random Forest", "pr_auc": 0.87, "f1": 0.78},
            ]
        )

        result = paired_wilcoxon_tests(
            frame=frame,
            item_column="model",
            baseline_item="XGBoost",
            comparison_items=["Random Forest"],
            metrics=["pr_auc", "f1"],
            family="model_comparison",
        )

        self.assertEqual(set(result["metric"]), {"pr_auc", "f1"})
        self.assertEqual(set(result["comparison"]), {"XGBoost vs Random Forest"})
        self.assertEqual(set(result["n_pairs"]), {3})
        self.assertTrue((result["mean_delta"] > 0).all())
        self.assertTrue(result["p_value"].notna().all())
        self.assertTrue(result["p_value_holm"].notna().all())
        self.assertTrue(result["family"].eq("model_comparison").all())
        self.assertTrue(
            math.isclose(
                float(result.loc[result["metric"] == "pr_auc", "baseline_mean"].iloc[0]),
                0.91,
            )
        )

    def test_markdown_table_has_no_optional_tabulate_dependency(self):
        frame = pd.DataFrame(
            [
                {"model": "XGBoost", "pr_auc_mean": 0.91234},
                {"model": "LightGBM", "pr_auc_mean": 0.90123},
            ]
        )

        table = _markdown_table(frame, ["model", "pr_auc_mean"], max_rows=1)

        self.assertIn("| model | pr_auc_mean |", table)
        self.assertIn("| XGBoost | 0.91234 |", table)
        self.assertNotIn("LightGBM", table)


if __name__ == "__main__":
    unittest.main()
