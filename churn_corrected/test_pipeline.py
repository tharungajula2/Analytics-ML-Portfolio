"""Synthetic software checks; these are not bank performance estimates."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from churn_corrected import pipeline as p


def synthetic(n=120):
    rng = np.random.default_rng(2026)
    x, y = [], []
    for i in range(n):
        age = float(25 + i % 45)
        geography = ("France", "Germany", "Spain")[i % 3]
        x.append([
            float(450 + i % 300), age, float(i % 10), float(3000 + i * 101),
            float(1 + i % 3), float(i % 2), float((i // 2) % 2),
            float(25000 + i * 117), geography, ("Female", "Male")[i % 2],
        ])
        # Explicit synthetic signal with stochastic noise, both classes in each split.
        y.append(int(rng.random() < (0.35 if age > 45 else 0.15)))
    return np.asarray(x, dtype=object), np.asarray(y, dtype=int)


class ChurnPipelineTests(unittest.TestCase):
    def test_split_is_disjoint_stratified_and_repeatable(self):
        _, y = synthetic()
        first = p.split_indices(y)
        second = p.split_indices(y)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
            self.assertIn(0, y[a])
            self.assertIn(1, y[a])
        self.assertEqual(sum(map(len, first)), len(y))
        self.assertEqual(len(set(np.concatenate(first))), len(y))
        self.assertEqual(tuple(map(len, first)), (72, 24, 24))

    def test_preprocessor_fits_only_training_rows(self):
        x, y = synthetic()
        train, validation, test = p.split_indices(y)
        model, _, _ = p.develop(x[train], y[train], x[validation], y[validation])
        mean = model.named_steps["preprocess"].named_transformers_["numeric"].mean_
        np.testing.assert_allclose(mean, np.asarray(x[train, :len(p.NUMERIC)], dtype=float).mean(axis=0))
        x[test, 0] = 1e12
        np.testing.assert_allclose(mean, np.asarray(x[train, :len(p.NUMERIC)], dtype=float).mean(axis=0))

    def test_test_rows_do_not_control_selection(self):
        x, y = synthetic()
        train, validation, test = p.split_indices(y)
        _, threshold_a, report_a = p.develop(x[train], y[train], x[validation], y[validation])
        x[test, 0] = 1e12
        y[test] = 1 - y[test]
        _, threshold_b, report_b = p.develop(x[train], y[train], x[validation], y[validation])
        self.assertEqual((threshold_a, report_a), (threshold_b, report_b))

    def test_unseen_category_does_not_refit_encoder(self):
        x, y = synthetic()
        train, validation, _ = p.split_indices(y)
        model, _, _ = p.develop(x[train], y[train], x[validation], y[validation])
        new = x[validation[:1]].copy()
        new[0, len(p.NUMERIC)] = "Unseen country"
        self.assertTrue(np.isfinite(model.predict_proba(new)).all())

    def test_metric_arithmetic_and_pr_auc(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.7, 0.4, 0.9])
        result = p.metrics(y, scores, 0.5)
        self.assertEqual(result["confusion_matrix_tn_fp_fn_tp"], [1, 1, 1, 1])
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["f1"], 0.5)
        self.assertAlmostEqual(result["roc_auc"], 0.75)
        self.assertAlmostEqual(result["pr_auc_average_precision"], (1 + 2 / 3) / 2)

    def test_threshold_comes_from_validation_grid(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.2, 0.4, 0.6, 0.8])
        threshold, f1 = p.choose_threshold(y, scores)
        self.assertIn(threshold, p.THRESHOLDS)
        self.assertEqual(f1, 1.0)
        self.assertEqual(threshold, 0.6)

    def test_csv_contract_excludes_identifiers(self):
        row = dict(zip(p.FEATURES, ["650", "40", "2", "1000", "2", "1", "0", "50000", "France", "Female"]))
        row.update({"RowNumber": "1", "CustomerId": "123", "Surname": "Example", "Exited": "1"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(p.EXPECTED))
                writer.writeheader()
                writer.writerow(row)
                row["CustomerId"] = "456"
                row["Exited"] = "0"
                writer.writerow(row)
            x, y, digest = p.load_rows(path)
            self.assertEqual(x.shape, (2, len(p.FEATURES)))
            self.assertEqual(list(y), [1, 0])
            self.assertEqual(len(digest), 64)
            row["HasCrCard"] = "0.5"
            with path.open("a", newline="") as handle:
                csv.DictWriter(handle, fieldnames=sorted(p.EXPECTED)).writerow(row)
            with self.assertRaisesRegex(ValueError, "must be binary"):
                p.load_rows(path)

    def test_development_is_repeatable(self):
        x, y = synthetic()
        train, validation, test = p.split_indices(y)
        a, threshold_a, report_a = p.develop(x[train], y[train], x[validation], y[validation])
        b, threshold_b, report_b = p.develop(x[train], y[train], x[validation], y[validation])
        self.assertEqual((threshold_a, report_a), (threshold_b, report_b))
        np.testing.assert_allclose(a.predict_proba(x[test]), b.predict_proba(x[test]))

    def test_end_to_end_synthetic_csv_and_baseline(self):
        x, y = synthetic()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(p.EXPECTED))
                writer.writeheader()
                for i, (features, target) in enumerate(zip(x, y)):
                    row = dict(zip(p.FEATURES, features))
                    row.update({"RowNumber": i + 1, "CustomerId": 1000 + i, "Surname": "Synthetic", "Exited": target})
                    writer.writerow(row)
            result = p.run(path)
            self.assertEqual(result["partition_sizes"], {"train": 72, "validation": 24, "test": 24})
            self.assertEqual(sum(result["test_model"]["confusion_matrix_tn_fp_fn_tp"]), 24)
            self.assertEqual(sum(bin_["n"] for bin_ in result["test_model_calibration"]), 24)
            self.assertAlmostEqual(result["test_prevalence_baseline"]["roc_auc"], 0.5)


if __name__ == "__main__":
    unittest.main()
