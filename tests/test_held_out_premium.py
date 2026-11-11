"""The rules that read and score the premium on the held-out cities.

Fixed by docs/registrations/2026-11-11-premium-on-held-out-cities.md before any
interval or swing was computed, so a change here is a change to a registered
design and should fail loudly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]

import held_out_premium as premium  # noqa: E402


def test_a_city_pays_only_when_its_interval_is_above_zero():
    assert premium.outcome(20, 1.0, 9.0) == "pays"
    assert premium.outcome(20, -9.0, -1.0) == "harms"
    assert premium.outcome(20, -1.0, 9.0) == "not shown"


def test_too_few_retrained_weeks_read_as_not_shown_whatever_the_interval():
    assert premium.outcome(4, 5.0, 9.0) == "not shown"
    assert premium.outcome(4, -9.0, -5.0) == "not shown"
    assert premium.outcome(5, 5.0, 9.0) == "pays"


def _year(monthly: list[float], drop_month: int | None = None) -> pd.Series:
    hours = pd.date_range(premium.SWING_START, premium.SWING_END + pd.Timedelta(23, unit="h"),
                          freq="h")
    values = np.array([monthly[(h.month - 5) % 12] for h in hours], dtype=float)
    if drop_month is not None:
        values[hours.month == drop_month] = np.nan
    return pd.Series(values, index=hours)


def test_the_swing_is_the_highest_month_over_the_lowest():
    monthly = [10.0] * 12
    monthly[3], monthly[8] = 40.0, 5.0  # August and January
    got = premium.swing(_year(monthly))
    assert got["swing"] == pytest.approx(8.0)
    assert (got["swing_low"], got["swing_high"], got["swing_months"]) == (5.0, 40.0, 12)


def test_a_month_without_enough_hours_is_left_out_of_the_swing():
    monthly = [10.0] * 12
    monthly[8] = 1.0  # January, then emptied
    got = premium.swing(_year(monthly, drop_month=1))
    assert got["swing_months"] == 11 and got["swing"] == pytest.approx(1.0)


def test_the_swing_needs_ten_of_twelve_months():
    series = _year([10.0] * 12)
    series[series.index.month.isin([1, 2, 3])] = np.nan
    assert math.isnan(premium.swing(series)["swing"])


def test_the_swing_year_ends_before_any_held_out_city_trains():
    assert premium.SWING_END < pd.Timestamp("2025-05-19")
    assert (premium.SWING_END - premium.SWING_START).days == 364


def test_the_threshold_is_the_smallest_swing_among_the_paying_six():
    six = [
        {"city": "A", "swing": 3.0, "outcome": "pays"},
        {"city": "B", "swing": 5.0, "outcome": "pays"},
        {"city": "C", "swing": 1.5, "outcome": "not shown"},
        {"city": "D", "swing": 7.0, "outcome": "not shown"},
    ]
    assert premium.threshold(six) == (3.0, "A")


def test_the_chance_of_a_sort_at_random_is_counted_exactly():
    # Nine cities, three pay, three predicted to: all nine right needs the one
    # matching choice of three among 84.
    assert premium.chance_of_at_least(9, 9, 3, 3) == pytest.approx(1 / math.comb(9, 3))
    # Some sort always gets at least the floor right.
    assert premium.chance_of_at_least(0, 9, 3, 3) == 1.0
    # Predicting nobody pays is right about everyone who does not, every time.
    assert premium.chance_of_at_least(6, 9, 3, 0) == 1.0
    assert premium.chance_of_at_least(7, 9, 3, 0) == 0.0


def _city(swing: float, outcome: str, predicted: str) -> dict:
    return {"city": f"c{swing}", "swing": swing, "outcome": outcome, "predicted": predicted,
            "premium": float(swing), "correct": (predicted == "pays") == (outcome == "pays")}


def test_p1_fails_at_six_of_nine_and_holds_at_seven():
    right = [_city(i, "pays", "pays") for i in range(6)]
    wrong = [_city(10 + i, "not shown", "pays") for i in range(3)]
    assert not premium.score(right + wrong)["p1"]["holds"]
    nearly = right + [_city(20, "not shown", "not")] + wrong[:2]
    assert premium.score(nearly)["p1"]["holds"]


def test_p2_fails_on_one_harmed_city():
    nine = [_city(i, "not shown", "not") for i in range(8)] + [_city(9, "harms", "not")]
    result = premium.score(nine)
    assert not result["p2"]["holds"] and result["p2"]["harms"] == ["c9"]
    # A harm is not a pay, so the sort still counts it right.
    assert result["p1"]["correct"] == 9
