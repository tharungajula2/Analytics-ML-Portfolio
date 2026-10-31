# Corrected 12-month credit-risk scorecard — reference pipeline

This is a separate, **synthetic-tested modelling framework, not a validated production PD model**. It preserves the historical [data-preparation notebook](<../Capstone_NIBM_Credit Risk Modeling (PD Scorecard)/1_Data-Preparation.ipynb>) and [PD/scorecard notebook](<../Capstone_NIBM_Credit Risk Modeling (PD Scorecard)/2_PD Model & Scorecard.ipynb>) unchanged. **No authentic LendingClub model was fitted and no real-data metric is claimed.** The missing source file and, more importantly, the missing dated performance history block that step.

The original notebook names `loan_data_2007_2014.csv`, loaded from an author's `D:\2.1_codebase\1_Datasets` path. Its saved output has 466,285 loans and 75 raw columns, including `issue_d`, `loan_status`, `annual_inc`, `dti`, `delinq_2yrs`, `inq_last_6mths`, `home_ownership` and `purpose`. The exact file and intermediate exports are absent from this repository and the workspace filename search. The notebook has no URL, checksum or source license. Same-named public mirrors cannot establish byte identity. No large file was downloaded for this phase.

## Decision and outcome contract

The intended prediction time is **immediately before the lending decision**, with `decision_date` recorded as an exact ISO date. This field is **not** in the saved 75-column original schema; `issue_d` is only an issue month and cannot silently substitute for it. The bad event is the first **90+ days-past-due or charge-off**, supported by a dated servicing event. For each loan, bad = 1 if the first event occurs after decision and on or before the 12-calendar-month anniversary. Good = 0 if no bad event occurs in that window and performance is observed through the horizon, or the loan is fully paid earlier. Otherwise the loan is censored and excluded. A later charge-off does not turn an already fully observed 12-month good into a bad.

The canonical CSV accepted by [the pipeline](pipeline.py) has **exactly** these fields:

| Field | Contract and use |
| --- | --- |
| `loan_id` | Unique nonempty loan key; never a predictor. |
| `decision_date` | Exact decision timestamp as `YYYY-MM-DD`; chronology and horizon, not a predictor. |
| `observed_through` | Last date with complete servicing observation; outcome eligibility only. |
| `first_bad_date`, `first_bad_type` | Earliest 90+ DPD or charge-off date and `90_plus_dpd`/`charge_off`; both blank when absent. Outcome only. |
| `paid_off_date` | Terminal full-payoff date, blank if absent; outcome eligibility only. |
| `annual_inc`, `dti` | Nonnegative decision-time income and debt-to-income percentage; numeric, blank for missing. |
| `delinq_2yrs`, `inq_last_6mths` | Nonnegative integer bureau counts measured at decision time; blank for missing. |
| `home_ownership`, `purpose` | Decision-time application categories; blank for missing. |

Only the six listed borrower/application features enter the model. An adapter from a verified raw dataset must prove their timestamps and units before a real run. The pipeline rejects extra columns, including a raw `loan_status` or any servicing amount, so a post-outcome field cannot accidentally become a predictor. It also rejects malformed dates, inconsistent event sequences, duplicates and invalid numeric values. `loan_status`, `recoveries`, `total_pymnt`, `last_pymnt_d`, `out_prncp`, later credit pulls and 2017-derived loan age are excluded. Grade, interest rate, installment, funded amount, term and listing status are excluded because their availability before the chosen decision is unproven.

## Chronological design

[Configuration](config.json) fixes a 12-month horizon and illustrative cohort windows before modelling:

| Cohort | Decision dates | Label information frozen at |
| --- | --- | --- |
| Train | 2007-01-01 through 2010-12-31 | 2012-01-01 |
| Validation | 2012-01-01 through 2012-12-31 | 2014-01-01 |
| Test | 2014-01-01 through 2014-12-31 | 2016-01-01 |

The 2011 and 2013 issue cohorts are intentionally unused gaps: they allow full 12-month label observation before the next cohort begins. Rows outside windows and censored rows are counted separately. The validation and test periods are **design defaults**, not evidence that the unavailable dataset supports all required fields or class counts. Do not retune bins or coefficients after seeing the test. A later verified dataset may justify different dates; update and review the config before fitting.

## Binning, model, score and diagnostics

Numeric cut points are declared in config; categorical vocabulary and counts come only from training rows. Unseen categories map to `__OTHER__`, and missing values to `__MISSING__`. Every bin receives 0.5-count smoothing. For bin `b`,

```text
WoE_b = ln[(good_b + α)/(G + Kα) ÷ (bad_b + α)/(B + Kα)]
IV = Σ_b (good_share_b − bad_share_b) × WoE_b
logit(P(bad|x)) = β0 + Σ_j βj × WoE_j(x)
fit loss = Σ_i BernoulliLogLoss_i + (λ/2) × Σ_{j>0} βj²
```

Here positive WoE means a bin is relatively more common among **good** loans, while the logistic target is **bad = 1**. Coefficient signs are estimated, not imposed. The deterministic Newton fit uses an L2 penalty and rejects nonconvergence. These choices are train-only; validation and test use frozen maps and coefficients.

Score increases as risk decreases. With `PDO = 20`, base score 600 and base **good:bad** odds 50:1:

```text
factor = PDO / ln(2)
score = 600 + factor × [ln((1 − PD)/PD) − ln(50)]
      = 600 − factor × [logit(PD) + ln(50)]
```

Thus 600 points corresponds to PD `1/51`, and doubling good:bad odds adds 20 points. The output keeps unrounded floating-point scores and an exact component sum; there is no manual point adjustment. Display rounding, if used later, must be treated separately from model PD.

For each chronological cohort, the pipeline calculates **bad-class AUROC**, Gini, KS, Brier score, log loss and predicted-versus-observed calibration bins. Calibration and PSI use boundaries fixed from training predictions. PSI measures score-distribution shift; it is not a fairness, causal or calibration test. No decision threshold or approval policy is inferred from these metrics.

## Run and evidence status

The runner and tests need **Python 3.10 or later, standard library only**. Run the synthetic software tests from the repository root:

```powershell
python -m unittest credit_risk_corrected.test_pipeline -v
```

The [deterministic fixture generator](synthetic_fixture.py) creates artificial loans in temporary test files. Its fitted coefficients and metrics are **not** recorded as LendingClub evidence.

The real-data CLI remains deliberately gated. After a legitimate, timestamp-verified event-history source and canonical export are available, copy `config.json` outside the repository, fill its `input_sha256` with the canonical CSV SHA-256 and `upstream_source` with the verified source/version, review cohort dates, then run:

```powershell
python -m credit_risk_corrected.pipeline --input path/to/canonical.csv --config path/to/reviewed-config.json --output path/to/results.json
```

The default config has null provenance fields, so this command refuses to model an unverified file. It never downloads data. A future real study needs event-definition and timestamp checks, a documented upstream license, a frozen input hash, out-of-time sample sizes, uncertainty intervals, calibration drift, rejection/selection bias and fairness review before a credit decision use case. This reference implementation is **not a lending policy**.
