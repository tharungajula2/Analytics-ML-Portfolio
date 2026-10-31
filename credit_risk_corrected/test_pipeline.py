"""Software-only checks with generated synthetic loans; no real performance claims."""

import csv
import hashlib
import json
import math
import tempfile
import unittest
from copy import deepcopy
from datetime import date
from pathlib import Path

from .contract import (FIELDS, add_months, label_at, load_rows, parse_date,
                       split_labeled, validate_config)
from .model import (OTHER, auc_bad, encode, fit_logistic, fit_woe, ks_bad,
                    pd_from_score, predict_pd, score_components, score_from_pd)
from .pipeline import fit_and_evaluate, run_verified
from .synthetic_fixture import synthetic_rows


HERE = Path(__file__).resolve().parent


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class ScorecardChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        cls.temp = tempfile.TemporaryDirectory()
        cls.file = Path(cls.temp.name) / "synthetic_loans.csv"
        write_csv(cls.file, synthetic_rows())
        cls.rows = load_rows(cls.file)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_config_has_label_gaps_and_strict_dates(self):
        validate_config(self.config)
        self.assertEqual(add_months(date(2012, 2, 29), 12), date(2013, 2, 28))
        with self.assertRaises(ValueError):
            parse_date("2012/02/29")
        changed = deepcopy(self.config)
        changed["test_start"] = "2013-06-01"
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_label_horizon_payoff_and_censoring(self):
        base = self.rows[0].copy()
        base["_decision"] = date(2010, 1, 15)
        base["_observed"] = date(2011, 2, 1)
        base["_bad"] = date(2011, 1, 15)
        base["_paid"] = None
        self.assertEqual(label_at(base, date(2012, 1, 1)), 1)
        base["_bad"] = date(2011, 1, 16)
        self.assertEqual(label_at(base, date(2012, 1, 1)), 0)
        base["_bad"] = None
        base["_observed"] = date(2010, 9, 1)
        self.assertIsNone(label_at(base, date(2012, 1, 1)))
        base["_paid"] = date(2010, 8, 15)
        self.assertEqual(label_at(base, date(2012, 1, 1)), 0)
        base["_paid"] = None
        base["_bad"] = date(2010, 8, 15)
        self.assertIsNone(label_at(base, date(2010, 8, 1)))

    def test_contract_rejects_outcome_and_feature_leakage_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=(*FIELDS, "loan_status"))
                writer.writeheader()
                writer.writerow({**synthetic_rows()[:1][0], "loan_status": "Charged Off"})
            with self.assertRaises(ValueError):
                load_rows(path)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicate.csv"
            row = synthetic_rows()[:1][0]
            write_csv(path, [row, row])
            with self.assertRaises(ValueError):
                load_rows(path)

    def test_chronological_cohorts_and_exclusions(self):
        cohorts, excluded = split_labeled(self.rows, self.config)
        self.assertGreater(excluded["outside_windows"], 0)
        self.assertEqual(excluded["censored"], 0)
        for name, start, end in (("train", date(2007, 1, 1), date(2011, 1, 1)),
                                 ("validation", date(2012, 1, 1), date(2013, 1, 1)),
                                 ("test", date(2014, 1, 1), date(2015, 1, 1))):
            self.assertTrue(all(start <= row["_decision"] < end for row, _ in cohorts[name]))
            self.assertEqual({target for _, target in cohorts[name]}, {0, 1})
        ids = [row["loan_id"] for cohort in cohorts.values() for row, _ in cohort]
        self.assertEqual(len(ids), len(set(ids)))

    def test_training_only_woe_and_unknown_category(self):
        cohorts, _ = split_labeled(self.rows, self.config)
        spec = fit_woe(cohorts["train"], self.config)
        beta = fit_logistic(cohorts["train"], spec, self.config["logistic_l2"])
        changed = deepcopy(cohorts)
        row, target = changed["validation"][0]
        row["home_ownership"] = "UNSEEN_FUTURE_CATEGORY"
        changed["validation"][0] = (row, 1 - target)
        self.assertEqual(spec, fit_woe(changed["train"], self.config))
        self.assertEqual(beta, fit_logistic(changed["train"], spec, self.config["logistic_l2"]))
        self.assertEqual(encode(row, spec)[5], spec["woe"]["home_ownership"][OTHER])
        self.assertTrue(all(math.isfinite(value) for weights in spec["woe"].values()
                            for value in weights.values()))

    def test_score_odds_points_and_components(self):
        base_pd = 1 / (1 + self.config["good_to_bad_odds_at_base"])
        self.assertAlmostEqual(score_from_pd(base_pd, self.config), self.config["score_base"])
        doubled = 1 / (1 + 2 * self.config["good_to_bad_odds_at_base"])
        self.assertAlmostEqual(score_from_pd(doubled, self.config),
                               self.config["score_base"] + self.config["points_to_double_odds"])
        cohorts, _ = split_labeled(self.rows, self.config)
        spec = fit_woe(cohorts["train"], self.config)
        beta = fit_logistic(cohorts["train"], spec, self.config["logistic_l2"])
        row = cohorts["test"][0][0]
        pd_bad = predict_pd(row, spec, beta)
        points = score_from_pd(pd_bad, self.config)
        self.assertAlmostEqual(pd_from_score(points, self.config), pd_bad)
        self.assertAlmostEqual(sum(score_components(row, spec, beta, self.config).values()), points)

    def test_discrimination_arithmetic(self):
        self.assertAlmostEqual(auc_bad([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]), 1.0)
        self.assertAlmostEqual(ks_bad([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]), 1.0)
        self.assertAlmostEqual(auc_bad([0, 1], [0.5, 0.5]), 0.5)

    def test_synthetic_end_to_end_without_reporting_metrics(self):
        result = fit_and_evaluate(self.rows, self.config)
        self.assertEqual(set(result["metrics"]), {"train", "validation", "test"})
        self.assertTrue(all(0 <= result["metrics"][name]["auc"] <= 1
                            for name in result["metrics"]))
        self.assertTrue(all(math.isfinite(v) for v in result["logistic_coefficients_bad_logit"]))
        changed = deepcopy(self.rows)
        for row in changed:
            if row["_decision"] >= date(2012, 1, 1):
                row["annual_inc"] = "999999"
                row["home_ownership"] = "UNSEEN_FUTURE_CATEGORY"
        altered = fit_and_evaluate(changed, self.config)
        self.assertEqual(result["woe"], altered["woe"])
        self.assertEqual(result["logistic_coefficients_bad_logit"],
                         altered["logistic_coefficients_bad_logit"])
        self.assertEqual(result["calibration_bin_boundaries_training_only"],
                         altered["calibration_bin_boundaries_training_only"])

    def test_real_cli_refuses_unverified_source(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            with self.assertRaisesRegex(ValueError, "verified canonical input"):
                run_verified(self.file, HERE / "config.json", output)
            self.assertFalse(output.exists())

    def test_cli_serializes_explicit_synthetic_fixture_only(self):
        with tempfile.TemporaryDirectory() as folder:
            config_file = Path(folder) / "synthetic-config.json"
            output = Path(folder) / "synthetic-result.json"
            config = deepcopy(self.config)
            config["input_sha256"] = hashlib.sha256(self.file.read_bytes()).hexdigest()
            config["upstream_source"] = "SYNTHETIC SOFTWARE TEST ONLY"
            config_file.write_text(json.dumps(config), encoding="utf-8")
            result = run_verified(self.file, config_file, output)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved, result)
            self.assertEqual(saved["upstream_source"], "SYNTHETIC SOFTWARE TEST ONLY")


if __name__ == "__main__":
    unittest.main()
