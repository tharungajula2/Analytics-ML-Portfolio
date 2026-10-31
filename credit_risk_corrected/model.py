"""Training-only WoE, regularized logistic PD, score scaling, diagnostics."""

from __future__ import annotations

import bisect
import math
from collections import Counter

from .contract import FEATURES


MISSING = "__MISSING__"
OTHER = "__OTHER__"


def feature_bin(row: dict, name: str, spec: dict) -> str:
    raw = row[name].strip()
    if raw == "":
        return MISSING
    if name in spec["numeric_cuts"]:
        index = bisect.bisect_right(spec["numeric_cuts"][name], float(raw))
        return f"bin_{index}"
    value = raw.upper()
    return value if value in spec["categories"][name] else OTHER


def fit_woe(train: list[tuple[dict, int]], config: dict) -> dict:
    """Only training labels and training category vocabulary influence WoE/IV."""
    if {y for _, y in train} != {0, 1}:
        raise ValueError("Training requires both bad and good outcomes")
    spec = {"numeric_cuts": config["numeric_cuts"], "categories": {},
            "woe": {}, "iv": {}}
    for name in config["categorical_features"]:
        counts = Counter(row[name].strip().upper() for row, _ in train if row[name].strip())
        spec["categories"][name] = sorted(key for key, n in counts.items()
                                            if n >= config["min_category_count"])
    good_total = sum(y == 0 for _, y in train)
    bad_total = len(train) - good_total
    alpha = config["woe_smoothing"]
    for name in FEATURES:
        if name in config["numeric_cuts"]:
            bins = [f"bin_{i}" for i in range(len(config["numeric_cuts"][name]) + 1)]
        else:
            bins = spec["categories"][name] + [OTHER]
        bins += [MISSING]
        counts = {key: [0, 0] for key in bins}  # [good, bad]
        for row, target in train:
            counts[feature_bin(row, name, spec)][target] += 1
        k = len(bins)
        values = {}
        iv = 0.0
        for key in bins:
            good_share = (counts[key][0] + alpha) / (good_total + alpha * k)
            bad_share = (counts[key][1] + alpha) / (bad_total + alpha * k)
            weight = math.log(good_share / bad_share)
            values[key] = weight
            iv += (good_share - bad_share) * weight
        spec["woe"][name] = values
        spec["iv"][name] = iv
    return spec


def encode(row: dict, spec: dict) -> list[float]:
    return [1.0] + [spec["woe"][name][feature_bin(row, name, spec)] for name in FEATURES]


def sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1 / (1 + inverse)
    exponent = math.exp(value)
    return exponent / (1 + exponent)


def linear_solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Pivoted elimination for the seven-parameter Newton system."""
    size = len(vector)
    work = [matrix[i][:] + [vector[i]] for i in range(size)]
    for k in range(size):
        pivot = max(range(k, size), key=lambda i: abs(work[i][k]))
        if abs(work[pivot][k]) < 1e-12:
            raise ValueError("Singular logistic Hessian")
        work[k], work[pivot] = work[pivot], work[k]
        scale = work[k][k]
        work[k] = [v / scale for v in work[k]]
        for i in range(size):
            if i != k:
                factor = work[i][k]
                work[i] = [work[i][j] - factor * work[k][j]
                           for j in range(size + 1)]
    return [work[i][-1] for i in range(size)]


def fit_logistic(train: list[tuple[dict, int]], spec: dict, l2: float) -> list[float]:
    """Deterministic Newton minimization of Bernoulli loss + L2 (slopes only)."""
    x = [encode(row, spec) for row, _ in train]
    y = [target for _, target in train]
    rate = sum(y) / len(y)
    if not 0 < rate < 1 or l2 <= 0:
        raise ValueError("Both outcomes and positive ridge penalty are required")
    size = len(FEATURES) + 1
    beta = [math.log(rate / (1 - rate))] + [0.0] * (size - 1)

    def loss(coef: list[float]) -> float:
        value = 0.5 * l2 * sum(v * v for v in coef[1:])
        for row, target in zip(x, y):
            z = sum(a * b for a, b in zip(coef, row))
            value += max(z, 0) + math.log1p(math.exp(-abs(z))) - target * z
        return value

    for _ in range(100):
        gradient = [0.0] * size
        hessian = [[0.0] * size for _ in range(size)]
        for row, target in zip(x, y):
            probability = sigmoid(sum(a * b for a, b in zip(beta, row)))
            residual = probability - target
            weight = probability * (1 - probability)
            for j in range(size):
                gradient[j] += residual * row[j]
                for k in range(size):
                    hessian[j][k] += weight * row[j] * row[k]
        for j in range(1, size):
            gradient[j] += l2 * beta[j]
            hessian[j][j] += l2
        step = linear_solve(hessian, gradient)
        if max(abs(v) for v in step) < 1e-8:
            return beta
        before = loss(beta)
        scale = 1.0
        while scale > 1e-8:
            candidate = [b - scale * s for b, s in zip(beta, step)]
            if loss(candidate) < before:
                beta = candidate
                break
            scale /= 2
        else:
            if max(abs(v) for v in gradient) < 1e-6:
                return beta
            raise ValueError("Logistic line search did not converge")
    raise ValueError("Logistic fit exceeded 100 Newton iterations")


def predict_pd(row: dict, spec: dict, beta: list[float]) -> float:
    return sigmoid(sum(a * b for a, b in zip(beta, encode(row, spec))))


def score_from_pd(pd_bad: float, config: dict) -> float:
    if not 0 < pd_bad < 1:
        raise ValueError("PD must be strictly between zero and one")
    factor = config["points_to_double_odds"] / math.log(2)
    log_odds_bad = math.log(pd_bad) - math.log1p(-pd_bad)
    return config["score_base"] - factor * (
        log_odds_bad + math.log(config["good_to_bad_odds_at_base"]))


def pd_from_score(points: float, config: dict) -> float:
    factor = config["points_to_double_odds"] / math.log(2)
    log_odds_bad = -(points - config["score_base"]) / factor - math.log(
        config["good_to_bad_odds_at_base"])
    return sigmoid(log_odds_bad)


def score_components(row: dict, spec: dict, beta: list[float], config: dict) -> dict:
    factor = config["points_to_double_odds"] / math.log(2)
    components = {"base_and_intercept": config["score_base"] - factor *
                  (beta[0] + math.log(config["good_to_bad_odds_at_base"]))}
    for name, coefficient, woe in zip(FEATURES, beta[1:], encode(row, spec)[1:]):
        components[name] = -factor * coefficient * woe
    return components


def auc_bad(y: list[int], p: list[float]) -> float:
    positives, negatives = sum(y), len(y) - sum(y)
    if not positives or not negatives or len(y) != len(p):
        raise ValueError("AUC requires paired predictions and both classes")
    ordered = sorted(zip(p, y))
    concordant = 0.0
    earlier_good = 0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        bad = sum(label for _, label in ordered[index:end])
        good = end - index - bad
        concordant += bad * (earlier_good + good / 2)
        earlier_good += good
        index = end
    return concordant / (positives * negatives)


def ks_bad(y: list[int], p: list[float]) -> float:
    positives, negatives = sum(y), len(y) - sum(y)
    if not positives or not negatives or len(y) != len(p):
        raise ValueError("KS requires paired predictions and both classes")
    ordered = sorted(zip(p, y))
    bad_seen = good_seen = 0
    maximum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        bad_seen += sum(label for _, label in ordered[index:end])
        good_seen += end - index - sum(label for _, label in ordered[index:end])
        maximum = max(maximum, abs(bad_seen / positives - good_seen / negatives))
        index = end
    return maximum


def score_boundaries(train_predictions: list[float]) -> list[float]:
    ordered = sorted(train_predictions)
    return sorted(set(ordered[max(0, math.ceil(q * len(ordered) / 10) - 1)]
                      for q in range(1, 10)))


def calibration(y: list[int], p: list[float], boundaries: list[float]) -> list[dict]:
    groups = [[] for _ in range(len(boundaries) + 1)]
    for label, probability in zip(y, p):
        groups[bisect.bisect_right(boundaries, probability)].append((label, probability))
    return [{"bin": index, "n": len(group),
             "observed_bad_rate": sum(v for v, _ in group) / len(group),
             "mean_predicted_pd": sum(v for _, v in group) / len(group)}
            for index, group in enumerate(groups) if group]


def psi(train_p: list[float], other_p: list[float], boundaries: list[float]) -> float:
    """Population stability index on train-defined PD bins, with smoothing."""
    k = len(boundaries) + 1
    def shares(values: list[float]) -> list[float]:
        counts = [0] * k
        for value in values:
            counts[bisect.bisect_right(boundaries, value)] += 1
        return [(n + 0.5) / (len(values) + 0.5 * k) for n in counts]
    reference, current = shares(train_p), shares(other_p)
    return sum((q - p) * math.log(q / p) for p, q in zip(reference, current))


def evaluate(cohort: list[tuple[dict, int]], spec: dict, beta: list[float],
             boundaries: list[float], train_predictions: list[float]) -> dict:
    y = [target for _, target in cohort]
    p = [predict_pd(row, spec, beta) for row, _ in cohort]
    if not p or {0, 1} != set(y):
        raise ValueError("Evaluation needs nonempty data and both classes")
    return {"n": len(y), "bad_rate": sum(y) / len(y),
            "auc": auc_bad(y, p), "gini": 2 * auc_bad(y, p) - 1,
            "ks": ks_bad(y, p),
            "brier": sum((a - b) ** 2 for a, b in zip(y, p)) / len(y),
            "log_loss": -sum(a * math.log(max(b, 1e-15)) +
                             (1 - a) * math.log(max(1 - b, 1e-15))
                             for a, b in zip(y, p)) / len(y),
            "calibration": calibration(y, p, boundaries),
            "psi_vs_train": psi(train_predictions, p, boundaries)}
