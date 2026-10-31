"""Strict dated-outcome and application-time input contract for a 12-month PD.

No real LendingClub file in this repository satisfies this event-date contract.
Unknown columns are rejected instead of accidentally becoming model features.
"""

from __future__ import annotations

import calendar
import csv
import math
import re
from datetime import date
from pathlib import Path


FEATURES = ("annual_inc", "dti", "delinq_2yrs", "inq_last_6mths",
            "home_ownership", "purpose")
FIELDS = ("loan_id", "decision_date", "observed_through", "first_bad_date",
          "first_bad_type", "paid_off_date", *FEATURES)
BAD_TYPES = {"charge_off", "90_plus_dpd"}


def parse_date(value: str, *, optional: bool = False) -> date | None:
    if optional and value == "":
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"Expected ISO date YYYY-MM-DD, got {value!r}")
    return date.fromisoformat(value)


def add_months(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 + months
    year, month0 = divmod(index, 12)
    month = month0 + 1
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


def validate_config(config: dict) -> dict:
    required = {"horizon_months", "train_start", "train_end", "validation_start",
                "validation_end", "test_start", "test_end", "as_of_date",
                "numeric_cuts", "categorical_features", "min_category_count",
                "woe_smoothing", "logistic_l2", "score_base",
                "good_to_bad_odds_at_base", "points_to_double_odds",
                "input_sha256", "upstream_source"}
    if set(config) != required:
        raise ValueError(f"Config keys must exactly match contract: {sorted(required)}")
    if config["horizon_months"] != 12:
        raise ValueError("This study defines a fixed 12-calendar-month outcome")
    keys = ("train_start", "train_end", "validation_start", "validation_end",
            "test_start", "test_end", "as_of_date")
    dates = {key: parse_date(config[key]) for key in keys}
    if not (dates["train_start"] < dates["train_end"] <= dates["validation_start"]
            < dates["validation_end"] <= dates["test_start"]
            < dates["test_end"] <= dates["as_of_date"]):
        raise ValueError("Cohort dates must be ordered without overlap")
    h = config["horizon_months"]
    if (add_months(dates["train_end"], h) > dates["validation_start"]
            or add_months(dates["validation_end"], h) > dates["test_start"]
            or add_months(dates["test_end"], h) > dates["as_of_date"]):
        raise ValueError("Each cohort needs a full label-observation gap")
    if set(config["numeric_cuts"]) != set(FEATURES[:4]):
        raise ValueError("Numeric feature contract differs from the declared list")
    for name, cuts in config["numeric_cuts"].items():
        if not cuts or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                           for v in cuts) or cuts != sorted(set(cuts)):
            raise ValueError(f"Invalid predeclared cuts for {name}")
    if tuple(config["categorical_features"]) != FEATURES[4:]:
        raise ValueError("Categorical feature contract differs from the declared list")
    if (config["min_category_count"] < 1 or config["woe_smoothing"] <= 0
            or config["logistic_l2"] <= 0 or config["good_to_bad_odds_at_base"] <= 0
            or config["points_to_double_odds"] <= 0):
        raise ValueError("Invalid binning, regularization or score scale parameter")
    return dates


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if (reader.fieldnames is None or len(reader.fieldnames) != len(FIELDS)
                or set(reader.fieldnames) != set(FIELDS)):
            raise ValueError(f"Expected exactly these canonical columns: {FIELDS}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty input")
    ids = set()
    for row in rows:
        if set(row) != set(FIELDS) or any(value is None for value in row.values()):
            raise ValueError("Malformed CSV row or unexpected extra fields")
        if not row["loan_id"] or row["loan_id"] in ids:
            raise ValueError("loan_id must be present and unique")
        ids.add(row["loan_id"])
        decision = parse_date(row["decision_date"])
        observed = parse_date(row["observed_through"])
        bad = parse_date(row["first_bad_date"], optional=True)
        paid = parse_date(row["paid_off_date"], optional=True)
        kind = row["first_bad_type"]
        if (bad is None) != (kind == "") or kind not in BAD_TYPES | {""}:
            raise ValueError("Bad event date/type must both be present or absent")
        if observed < decision or any(d is not None and not (decision < d <= observed)
                                      for d in (bad, paid)):
            raise ValueError("Outcome dates contradict decision or observation dates")
        if bad is not None and paid is not None and paid < bad:
            raise ValueError("A paid-off loan cannot subsequently first default")
        for name in FEATURES[:4]:
            value = row[name].strip()
            if value != "":
                number = float(value)
                if not math.isfinite(number) or number < 0:
                    raise ValueError(f"Invalid {name}")
                if name in ("delinq_2yrs", "inq_last_6mths") and not number.is_integer():
                    raise ValueError(f"{name} must be an integer count")
        row["_decision"] = decision
        row["_observed"] = observed
        row["_bad"] = bad
        row["_paid"] = paid
    return rows


def label_at(row: dict, freeze: date, months: int = 12) -> int | None:
    """1=first 90+ DPD/charge-off by horizon, 0=fully observed nonbad.

    None means censored. Performance information after freeze is ignored.
    """
    horizon = add_months(row["_decision"], months)
    if row["_bad"] is not None and row["_bad"] <= min(horizon, freeze):
        return 1
    if row["_paid"] is not None and row["_paid"] <= min(horizon, freeze):
        return 0
    if min(row["_observed"], freeze) >= horizon:
        return 0
    return None


def split_labeled(rows: list[dict], config: dict) -> tuple[dict, dict]:
    dates = validate_config(config)
    cohorts = {name: [] for name in ("train", "validation", "test")}
    exclusions = {"outside_windows": 0, "censored": 0}
    windows = (("train", dates["train_start"], dates["train_end"], dates["validation_start"]),
               ("validation", dates["validation_start"], dates["validation_end"], dates["test_start"]),
               ("test", dates["test_start"], dates["test_end"], dates["as_of_date"]))
    for row in rows:
        match = next(((name, freeze) for name, start, end, freeze in windows
                      if start <= row["_decision"] < end), None)
        if match is None:
            exclusions["outside_windows"] += 1
            continue
        name, freeze = match
        target = label_at(row, freeze, config["horizon_months"])
        if target is None:
            exclusions["censored"] += 1
            continue
        cohorts[name].append((row, target))
    for name, cohort in cohorts.items():
        if {target for _, target in cohort} != {0, 1}:
            raise ValueError(f"{name} needs both classes for evaluation")
    return cohorts, exclusions
