"""CLI for a verified canonical loan-performance CSV; never downloads data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .contract import FEATURES, load_rows, split_labeled, validate_config
from .model import (evaluate, fit_logistic, fit_woe, predict_pd,
                    score_boundaries, score_components, score_from_pd)


def fit_and_evaluate(rows: list[dict], config: dict) -> dict:
    cohorts, exclusions = split_labeled(rows, config)
    spec = fit_woe(cohorts["train"], config)
    beta = fit_logistic(cohorts["train"], spec, config["logistic_l2"])
    train_pd = [predict_pd(row, spec, beta) for row, _ in cohorts["train"]]
    boundaries = score_boundaries(train_pd)
    return {
        "label": "first charge-off or 90+ DPD within 12 calendar months; 1=bad",
        "cohort_counts": {name: len(cohort) for name, cohort in cohorts.items()},
        "exclusions": exclusions,
        "features": list(FEATURES),
        "woe": spec,
        "logistic_coefficients_bad_logit": beta,
        "iv_training_only": spec["iv"],
        "calibration_bin_boundaries_training_only": boundaries,
        "score_scale": {key: config[key] for key in (
            "score_base", "good_to_bad_odds_at_base", "points_to_double_odds")},
        "score_component_example_synthetic_or_supplied": score_components(
            cohorts["train"][0][0], spec, beta, config),
        "example_score": score_from_pd(train_pd[0], config),
        "metrics": {name: evaluate(cohort, spec, beta, boundaries, train_pd)
                    for name, cohort in cohorts.items()},
    }


def run_verified(input_path: Path, config_path: Path, output_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if not config["input_sha256"] or not config["upstream_source"]:
        raise ValueError("Set verified canonical input SHA-256 and upstream source first")
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if digest.lower() != config["input_sha256"].lower():
        raise ValueError("Input checksum does not match the reviewed configuration")
    rows = load_rows(input_path)
    result = fit_and_evaluate(rows, config)
    result["input_sha256"] = digest
    result["upstream_source"] = config["upstream_source"]
    result["config"] = {key: value for key, value in config.items()
                        if key not in ("input_sha256", "upstream_source")}
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Verified canonical CSV")
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"), type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Results JSON destination")
    args = parser.parse_args()
    run_verified(args.input, args.config, args.output)
    print(f"Wrote {args.output} from checksum-verified input")


if __name__ == "__main__":
    main()
