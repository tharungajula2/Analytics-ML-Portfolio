"""Reproducible, dependency-free companion to the historical prescription notebook.

Run: python forecasting/corrected_forecast.py
The small, predeclared models are SARIMA(p,1,0)(0,1,0)[12], p=0,1,2,
estimated by conditional least squares. This is not statsmodels exact MLE.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "AusAntidiabeticDrug.csv"
SHA256 = "d6f32528dbeabdad405b578d351375aca9eb6f0eee7b632c9a49b909f80741c5"
PERIOD = 12
TRAIN = 168
HORIZON = 12
ORIGINS = (168, 180, 192)
CANDIDATES = (0, 1, 2)
MAX_P = max(CANDIDATES)


def load_series(path: Path = DATA) -> tuple[list[str], list[float]]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Input SHA-256 differs from the audited source snapshot")
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    if len(rows) != 204 or set(rows[0]) != {"ds", "y"}:
        raise ValueError("Expected 204 rows with exactly ds,y columns")
    dates = [row["ds"] for row in rows]
    values = [float(row["y"]) for row in rows]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Expected finite positive prescriptions")
    months = [int(d[:4]) * 12 + int(d[5:7]) for d in dates]
    if months != list(range(months[0], months[0] + 204)):
        raise ValueError("Expected uninterrupted monthly dates")
    if dates[0] != "1991-07-01" or dates[-1] != "2008-06-01":
        raise ValueError("Unexpected date span")
    return dates, values


def seasonal_first_difference(y: list[float]) -> list[float | None]:
    """(1-B)(1-B^12)y; z[t] exists for t >= 13."""
    z: list[float | None] = [None] * len(y)
    for t in range(PERIOD + 1, len(y)):
        z[t] = y[t] - y[t - 1] - y[t - PERIOD] + y[t - PERIOD - 1]
    return z


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Small normal-equation solve with pivoting; fail visibly on singularity."""
    n = len(b)
    m = [a[i][:] + [b[i]] for i in range(n)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(m[i][k]))
        if abs(m[pivot][k]) < 1e-12:
            raise ValueError("Singular conditional regression")
        m[k], m[pivot] = m[pivot], m[k]
        scale = m[k][k]
        m[k] = [v / scale for v in m[k]]
        for i in range(n):
            if i != k:
                factor = m[i][k]
                m[i] = [m[i][j] - factor * m[k][j] for j in range(n + 1)]
    return [m[i][-1] for i in range(n)]


def fit(y: list[float], p: int) -> dict:
    """Same conditional sample t=15,...,n-1 for all p, for comparable AIC."""
    if p not in CANDIDATES or len(y) <= PERIOD + MAX_P + 3:
        raise ValueError("Invalid order or too little history")
    z = seasonal_first_difference(y)
    start = PERIOD + 1 + MAX_P
    x = [[1.0] + [float(z[t - lag]) for lag in range(1, p + 1)]
         for t in range(start, len(y))]
    target = [float(z[t]) for t in range(start, len(y))]
    k = p + 1
    xtx = [[sum(row[i] * row[j] for row in x) for j in range(k)] for i in range(k)]
    xty = [sum(row[i] * v for row, v in zip(x, target)) for i in range(k)]
    beta = solve(xtx, xty)
    if (p == 1 and abs(beta[1]) >= 1) or (p == 2 and not
            (-1 < beta[2] < 1 and beta[1] + beta[2] < 1
             and beta[2] - beta[1] < 1)):
        raise ValueError("Conditional AR coefficients are outside the stationary region")
    residuals = [v - sum(a * b for a, b in zip(row, beta))
                 for row, v in zip(x, target)]
    rss = sum(e * e for e in residuals)
    n = len(target)
    # The Gaussian likelihood's n*(1+log(2*pi)) constant is omitted for every fit.
    aic = n * math.log(rss / n) + 2 * (k + 1)  # coefficients + variance
    return {"p": p, "coefficients": beta, "n": n, "rss": rss, "aic_relative": aic}


def forecast(y: list[float], model: dict, horizon: int = HORIZON) -> list[float]:
    history = y[:]
    z = seasonal_first_difference(history)
    beta = model["coefficients"]
    for _ in range(horizon):
        t = len(history)
        next_z = beta[0] + sum(beta[lag] * float(z[t - lag])
                               for lag in range(1, model["p"] + 1))
        next_y = next_z + history[t - 1] + history[t - PERIOD] - history[t - PERIOD - 1]
        z.append(next_z)
        history.append(next_y)
    return history[-horizon:]


def metrics(actual: list[float], predicted: list[float]) -> dict:
    if len(actual) != len(predicted) or not actual or any(v <= 0 for v in actual):
        raise ValueError("Metric inputs must align and actuals must be positive")
    errors = [a - f for a, f in zip(actual, predicted)]
    return {"mae": sum(abs(e) for e in errors) / len(errors),
            "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "mape_pct": 100 * sum(abs(e) / a for e, a in zip(errors, actual)) / len(errors)}


def evaluate(y: list[float]) -> dict:
    # Select only once, before opening the 36-month holdout.
    initial_fits = [fit(y[:TRAIN], p) for p in CANDIDATES]
    selected = min(initial_fits, key=lambda m: (m["aic_relative"], m["p"]))["p"]
    predictions = {name: [] for name in ("naive", "seasonal_naive", "selected_sarima")}
    windows = []
    for origin in ORIGINS:
        history = y[:origin]  # the last two origins may use only newly observed past values
        actual = y[origin:origin + HORIZON]
        block = {"naive": [history[-1]] * HORIZON,
                 "seasonal_naive": history[-PERIOD:],
                 "selected_sarima": forecast(history, fit(history, selected))}
        for name, values in block.items():
            predictions[name].extend(values)
        windows.append({"origin": origin, "actual": actual, "predictions": block,
                        "metrics": {name: metrics(actual, values) for name, values in block.items()}})
    actual = y[TRAIN:]
    return {"train_rows": TRAIN, "test_rows": len(actual), "origins": ORIGINS,
            "candidate_fits": initial_fits, "selected_p": selected,
            "overall_metrics": {name: metrics(actual, values)
                                for name, values in predictions.items()},
            "actual": actual, "predictions": predictions, "windows": windows}


def render_svg(dates: list[str], y: list[float], result: dict) -> str:
    """Two compact, deterministic plots requiring no graphics dependency."""
    width, height = 940, 540
    colors = {"actual": "#172554", "naive": "#b45309",
              "seasonal_naive": "#0f766e", "selected_sarima": "#7c3aed"}
    def line(values, start, x0, y0, w, h, lo, hi, color, dash=""):
        pts = " ".join(f"{x0 + w * (start + i) / 203:.1f},{y0 + h * (1 - (v-lo)/(hi-lo)):.1f}"
                       for i, v in enumerate(values))
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}" points="{pts}"/>'
    lo, hi = min(y) - 2, max(y) + 2
    top = line(y, 0, 55, 52, 830, 185, lo, hi, colors["actual"])
    split_x = 55 + 830 * TRAIN / 203
    top += f'<line x1="{split_x:.1f}" y1="52" x2="{split_x:.1f}" y2="237" stroke="#dc2626" stroke-dasharray="5 4"/>'
    holdout = y[TRAIN:]
    hlow = min(holdout + sum(result["predictions"].values(), [])) - 2
    hhigh = max(holdout + sum(result["predictions"].values(), [])) + 2
    def hold_line(values, color, dash=""):
        pts = " ".join(f"{55 + 830*i/35:.1f},{302 + 178*(1-(v-hlow)/(hhigh-hlow)):.1f}"
                       for i, v in enumerate(values))
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}" points="{pts}"/>'
    bottom = hold_line(holdout, colors["actual"])
    for name in ("naive", "seasonal_naive", "selected_sarima"):
        bottom += hold_line(result["predictions"][name], colors[name], "4 3")
    legend = "".join(f'<text x="{55+i*205}" y="521" fill="{colors[name]}" font-size="14">{name.replace("_", " ")}</text>'
                     for i, name in enumerate(colors))
    axes = ('<path d="M55 52 V237 H885 M55 302 V480 H885" fill="none" stroke="#64748b"/>'
            f'<text x="5" y="60" font-size="12">{hi:.0f}</text>'
            f'<text x="5" y="237" font-size="12">{lo:.0f}</text>'
            f'<text x="5" y="310" font-size="12">{hhigh:.0f}</text>'
            f'<text x="5" y="480" font-size="12">{hlow:.0f}</text>'
            '<text x="739" y="48" fill="#dc2626" font-size="12">test begins</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Monthly prescription series and rolling holdout forecasts">'
            '<rect width="100%" height="100%" fill="white"/>'
            '<text x="55" y="27" font-size="18">Australian antidiabetic prescriptions, 1991–2008</text>'
            '<text x="55" y="274" font-size="18">36-month rolling holdout (three annual origins)</text>'
            '<text x="55" y="495" font-size="12">Jul 2005</text><text x="820" y="495" font-size="12">Jun 2008</text>'
            + axes + top + bottom + legend + '</svg>')


def main() -> None:
    dates, y = load_series()
    result = evaluate(y)
    result["dataset_sha256"] = SHA256
    result["date_span"] = [dates[0], dates[-1]]
    (HERE / "corrected_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (HERE / "rolling_forecasts.svg").write_text(render_svg(dates, y, result), encoding="utf-8")
    print("Candidate conditional AIC:")
    for m in result["candidate_fits"]:
        print(f"  p={m['p']}: n={m['n']}, AIC*={m['aic_relative']:.3f}")
    print(f"Selected SARIMA({result['selected_p']},1,0)(0,1,0)[12]")
    print("36-month rolling holdout:")
    for name, values in result["overall_metrics"].items():
        print(f"  {name}: MAE={values['mae']:.3f}, RMSE={values['rmse']:.3f}, MAPE={values['mape_pct']:.3f}%")


if __name__ == "__main__":
    main()
