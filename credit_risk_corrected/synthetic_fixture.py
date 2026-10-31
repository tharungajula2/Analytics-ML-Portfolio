"""Explicitly synthetic records for software tests, never historical evidence."""

from __future__ import annotations

import random
from datetime import date

from .contract import add_months


def synthetic_rows(seed: int = 47) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for year in range(2008, 2015):
        for month in range(1, 13):
            decision = date(year, month, 15)
            for number in range(5):
                income = rng.choice((25000, 45000, 70000, 110000, 170000))
                dti = rng.choice((6, 12, 19, 28, 40))
                delinquency = rng.choice((0, 0, 0, 1, 2, 4))
                inquiries = rng.choice((0, 0, 1, 2, 4))
                risk = (0.04 + 0.07 * (income < 45000) + 0.06 * (dti > 25)
                        + 0.08 * (delinquency > 0) + 0.04 * (inquiries > 2))
                bad = rng.random() < risk
                paid = not bad and rng.random() < 0.15
                rows.append({
                    "loan_id": f"SYN-{year}-{month:02d}-{number}",
                    "decision_date": decision.isoformat(),
                    "observed_through": "2016-01-01",
                    "first_bad_date": add_months(decision, 4).isoformat() if bad else "",
                    "first_bad_type": "90_plus_dpd" if bad else "",
                    "paid_off_date": add_months(decision, 8).isoformat() if paid else "",
                    "annual_inc": str(income), "dti": str(dti),
                    "delinq_2yrs": str(delinquency), "inq_last_6mths": str(inquiries),
                    "home_ownership": rng.choice(("RENT", "OWN", "MORTGAGE")),
                    "purpose": rng.choice(("credit_card", "debt_consolidation", "home_improvement")),
                })
    return rows
