"""Leakage-controlled reference analysis for the historical bank churn schema.

Requires NumPy and scikit-learn. No dataset is bundled or downloaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
NUMERIC = (
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
)
CATEGORICAL = ("Geography", "Gender")
IGNORED = ("RowNumber", "CustomerId", "Surname")
TARGET = "Exited"
FEATURES = NUMERIC + CATEGORICAL
EXPECTED = set(FEATURES + IGNORED + (TARGET,))
BINARY = ("HasCrCard", "IsActiveMember")
THRESHOLDS = tuple(round(i / 100, 2) for i in range(10, 91, 5))
CANDIDATES = ((0.1, None), (1.0, None), (0.1, "balanced"), (1.0, "balanced"))


def load_rows(path: str | Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Enforce the 14-column notebook schema and return only decision-time candidates."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Missing or duplicate CSV header")
        if set(reader.fieldnames) != EXPECTED:
            raise ValueError(f"Expected exactly {sorted(EXPECTED)}; got {sorted(reader.fieldnames)}")
        features, labels = [], []
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {line}")
            if row[TARGET] not in ("0", "1"):
                raise ValueError(f"Row {line}: Exited must be 0 or 1")
            values = []
            for name in NUMERIC:
                try:
                    number = float(row[name])
                except ValueError as exc:
                    raise ValueError(f"Row {line}: invalid {name}") from exc
                if not math.isfinite(number):
                    raise ValueError(f"Row {line}: nonfinite {name}")
                if name in BINARY and number not in (0.0, 1.0):
                    raise ValueError(f"Row {line}: {name} must be binary")
                values.append(number)
            for name in CATEGORICAL:
                value = row[name].strip()
                if not value:
                    raise ValueError(f"Row {line}: empty {name}")
                values.append(value)
            features.append(values)
            labels.append(int(row[TARGET]))
    if not features or len(set(labels)) != 2:
        raise ValueError("A nonempty, two-class dataset is required")
    return np.asarray(features, dtype=object), np.asarray(labels, dtype=int), digest


def split_indices(y: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed stratified 60/20/20 row split; no event dates exist for a temporal split."""
    indices = np.arange(len(y))
    development, test = train_test_split(indices, test_size=0.2, stratify=y, random_state=seed)
    train, validation = train_test_split(
        development, test_size=0.25, stratify=y[development], random_state=seed + 1
    )
    return train, validation, test


def estimator(c: float, class_weight: str | None) -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(range(len(NUMERIC)))),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(range(len(NUMERIC), len(FEATURES)))),
        ]
    )
    return Pipeline([
        ("preprocess", transformer),
        ("model", LogisticRegression(C=c, class_weight=class_weight, max_iter=1000, random_state=SEED)),
    ])


def choose_threshold(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Choose a predeclared grid point by validation F1; prefer higher threshold on ties."""
    scores = [(f1_score(y, probabilities >= t, zero_division=0), t) for t in THRESHOLDS]
    score, threshold = max(scores)
    return threshold, float(score)


def develop(x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray):
    """Fit and select using train/validation only. Never receives test rows."""
    if len(set(y_train)) != 2 or len(set(y_validation)) != 2:
        raise ValueError("Both development partitions need both classes")
    ranked = []
    for c, weight in CANDIDATES:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = estimator(c, weight).fit(x_train, y_train)
        probabilities = model.predict_proba(x_validation)[:, 1]
        ap = float(average_precision_score(y_validation, probabilities))
        brier = float(brier_score_loss(y_validation, probabilities))
        # Fixed candidate order breaks exact ties; lower Brier is secondary.
        ranked.append((ap, -brier, model, c, weight, probabilities))
    best = max(ranked, key=lambda item: (item[0], item[1]))
    threshold, validation_f1 = choose_threshold(y_validation, best[5])
    return best[2], threshold, {
        "C": best[3], "class_weight": best[4], "validation_average_precision": best[0],
        "validation_brier": -best[1], "validation_f1_at_selected_threshold": validation_f1,
        "threshold": threshold,
    }


def metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """PR-AUC is average precision (step-area), calculated from probabilities."""
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(y, probabilities)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "brier": float(brier_score_loss(y, probabilities)),
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        "threshold": threshold,
    }


def calibration_bins(y: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    """Descriptive test-only reliability table in five fixed probability intervals."""
    result = []
    for lower in np.linspace(0, 0.8, 5):
        upper = round(float(lower + 0.2), 10)
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= 1)
        if np.any(mask):
            result.append({
                "probability_interval": [round(float(lower), 2), upper],
                "n": int(mask.sum()), "mean_predicted": float(np.mean(probabilities[mask])),
                "observed_churn_rate": float(np.mean(y[mask])),
            })
    return result


def run(path: str | Path) -> dict:
    x, y, digest = load_rows(path)
    train, validation, test = split_indices(y)
    model, threshold, selection = develop(x[train], y[train], x[validation], y[validation])
    # Final test boundary: one call after all training and threshold decisions.
    probabilities = model.predict_proba(x[test])[:, 1]
    prevalence = float(np.mean(y[train]))
    baseline = np.full(len(test), prevalence)
    return {
        "input_sha256": digest, "seed": SEED, "rows": len(y),
        "partition_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "train_prevalence": prevalence, "selection": selection,
        "test_model": metrics(y[test], probabilities, threshold),
        "test_model_calibration": calibration_bins(y[test], probabilities),
        "test_prevalence_baseline": metrics(y[test], baseline, 0.5),
        "warning": "A static, undated dataset cannot establish prospective churn prediction or a retention policy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Local, authorized copy of the exact 14-column churn.csv")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.csv), indent=2))
