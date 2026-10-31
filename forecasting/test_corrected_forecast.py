"""Focused mathematical and information-boundary checks; no third-party packages."""

import json
import unittest
from pathlib import Path

from corrected_forecast import (CANDIDATES, TRAIN, evaluate, fit, forecast,
                                load_series, metrics, seasonal_first_difference)


class ForecastChecks(unittest.TestCase):
    def test_lag_twelve_differencing(self):
        y = [float(t) + 3.0 * (t % 12) for t in range(40)]
        z = seasonal_first_difference(y)
        self.assertEqual(z[:13], [None] * 13)
        self.assertTrue(all(abs(v) < 1e-10 for v in z[13:]))
        # Adjacent 12th-order differencing would not express this operator.

    def test_equal_aic_sample_and_forecast_length(self):
        _, y = load_series()
        fits = [fit(y[:TRAIN], p) for p in CANDIDATES]
        self.assertEqual({m["n"] for m in fits}, {153})
        for model in fits:
            self.assertEqual(len(forecast(y[:TRAIN], model)), 12)

    def test_future_values_do_not_change_any_origin(self):
        _, y = load_series()
        original = evaluate(y)
        for index, origin in enumerate((168, 180, 192)):
            changed = y[:origin] + [v + 500.0 for v in y[origin:]]
            altered = evaluate(changed)
            self.assertEqual(original["selected_p"], altered["selected_p"])
            self.assertEqual(original["candidate_fits"], altered["candidate_fits"])
            self.assertEqual(original["windows"][index]["predictions"],
                             altered["windows"][index]["predictions"])

    def test_inverse_difference_forecast(self):
        y = [float(t) + 3.0 * (t % 12) for t in range(40)]
        predicted = forecast(y, {"p": 0, "coefficients": [0.0]}, 12)
        expected = [float(t) + 3.0 * (t % 12) for t in range(40, 52)]
        for a, b in zip(predicted, expected):
            self.assertAlmostEqual(a, b)

    def test_metric_arithmetic(self):
        result = metrics([10.0, 20.0], [8.0, 24.0])
        self.assertAlmostEqual(result["mae"], 3.0)
        self.assertAlmostEqual(result["rmse"], (10.0) ** 0.5)
        self.assertAlmostEqual(result["mape_pct"], 20.0)

    def test_saved_result_matches_fresh_calculation(self):
        _, y = load_series()
        saved = json.loads((Path(__file__).resolve().parent / "corrected_results.json").read_text())
        fresh = evaluate(y)
        for key in ("candidate_fits", "selected_p", "overall_metrics",
                    "actual", "predictions", "windows"):
            self.assertEqual(saved[key], fresh[key])


if __name__ == "__main__":
    unittest.main()
