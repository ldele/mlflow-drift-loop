"""The rules and arms of the calendar-retraining test.

All of them are fixed by docs/registrations/2026-11-15-calendar-retraining-baseline.md
before the test is run on any city, so a change here is a change to a registered
design and should fail loudly. The arms are checked on the synthetic source only.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import mlflow
import pandas as pd
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]

import calendar_baseline as cal  # noqa: E402

from driftloop.config import PROFILES, SHIPPED, LoopConfig  # noqa: E402
from driftloop.data import SyntheticSource  # noqa: E402
from driftloop.loop import bootstrap_champion, run_simulation  # noqa: E402

NAN = float("nan")


@pytest.fixture()
def isolated_mlflow(tmp_path):
    """A throwaway sqlite backend, as in test_loop.py, so no real backend is touched."""
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    mlflow.create_experiment("test-calendar", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("test-calendar")
    yield LoopConfig(experiment_name="test-calendar", registered_model_name="test-calendar")


# --------------------------------------------------------------------------- #
# How one city is read                                                         #
# --------------------------------------------------------------------------- #


def test_a_city_reads_better_or_worse_only_when_the_interval_clears_zero():
    assert cal.reading(10, 0.5, 4.0) == "better"
    assert cal.reading(10, -4.0, -0.5) == "worse"
    assert cal.reading(10, -1.0, 3.0) == "not shown"


def test_too_few_differing_weeks_reads_not_shown_whatever_the_interval():
    assert cal.reading(cal.MIN_DIFFERING - 1, 2.0, 9.0) == "not shown"
    assert cal.reading(cal.MIN_DIFFERING, 2.0, 9.0) == "better"
    assert cal.reading(10, NAN, NAN) == "not shown"


# --------------------------------------------------------------------------- #
# The reading rule across cities                                               #
# --------------------------------------------------------------------------- #


def _cities(better: int, worse: int, total: int = 15) -> list[str]:
    return ["better"] * better + ["worse"] * worse + ["not shown"] * (total - better - worse)


def test_three_cities_one_way_and_none_the_other_is_a_verdict():
    assert cal.verdict(_cities(3, 0)) == "better"
    assert cal.verdict(_cities(0, 3)) == "worse"


def test_two_cities_is_not_enough_out_of_fifteen():
    assert cal.verdict(_cities(2, 0)) == "not detectable"
    assert cal.verdict(_cities(0, 2)) == "not detectable"


def test_a_city_against_the_direction_makes_it_mixed():
    assert cal.verdict(_cities(4, 1)) == "mixed"
    assert cal.verdict(_cities(1, 2)) == "mixed"
    assert cal.verdict(_cities(1, 1)) == "not detectable"


def test_the_six_city_rule_is_the_one_the_five_fixes_were_read_under():
    assert cal.two_city_rule(_cities(2, 0, 6)) == "better"
    assert cal.two_city_rule(_cities(1, 0, 6)) == "not detectable"
    assert cal.two_city_rule(_cities(2, 2, 6)) == "mixed"


# --------------------------------------------------------------------------- #
# Comparisons                                                                  #
# --------------------------------------------------------------------------- #


def _arm(rmse: list[float]) -> dict:
    return {"_rmse": rmse, "_as_of": [f"2025-08-{d:02d}" for d in range(1, len(rmse) + 1)]}


def test_identical_arms_differ_in_no_week_and_read_not_shown():
    row = cal.compare(_arm([10.0] * 20), _arm([10.0] * 20))
    assert row["differing"] == 0 and row["reading"] == "not shown"
    assert math.isnan(row["acted"])


def test_a_comparison_is_positive_when_x_forecasts_better():
    row = cal.compare(_arm([9.0] * 20), _arm([10.0] * 20))
    assert row["differing"] == 20
    assert row["acted"] > 0 and row["reading"] == "better"


def test_a_comparison_is_paired_on_the_date_not_the_position():
    x = _arm([9.0] * 20)
    y = {"_rmse": [10.0] * 19, "_as_of": x["_as_of"][1:]}
    assert cal.compare(x, y)["weeks"] == 19


# --------------------------------------------------------------------------- #
# The arms                                                                     #
# --------------------------------------------------------------------------- #


def test_the_arms_change_only_what_the_registration_says():
    base = PROFILES["openmeteo_delhi"]
    shipped = cal.arm_profile(base, cal.SHIPPED_ARM).loop
    exam = cal.arm_profile(base, cal.WEEKLY_EXAM).loop
    refit = cal.arm_profile(base, cal.WEEKLY_REFIT).loop
    names = {"registered_model_name": "x"}
    assert replace(shipped, **names) == replace(base.loop, **names)
    assert replace(exam, **names) == replace(base.loop, recertify_days=1, **names)
    assert replace(refit, **names) == replace(
        base.loop, recertify_days=1, promotion_margin=float("-inf"), **names)
    for key, value in SHIPPED.items():
        assert getattr(exam, key) == value and getattr(refit, key) == value


def test_each_arm_registers_under_a_name_of_its_own():
    base = PROFILES["openmeteo_delhi"]
    keys = {cal.arm_profile(base, arm).key for arm in cal.ARMS}
    names = {cal.arm_profile(base, arm).loop.registered_model_name for arm in cal.ARMS}
    assert len(keys) == len(names) == len(cal.ARMS)


def _replay(cfg) -> pd.DataFrame:
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    return run_simulation(
        src, cfg, pd.Timestamp("2025-07-15"), pd.Timestamp("2025-08-12"), step_days=7,
    )


def test_the_weekly_exam_trains_a_challenger_on_every_run(isolated_mlflow):
    df = _replay(replace(isolated_mlflow, **cal.ARMS[cal.WEEKLY_EXAM]))
    assert df["retrain_triggered"].all()
    row = {"runs": len(df), "retrains": int(df["retrain_triggered"].sum()),
           "promotions": int((df["promotion_decision"] == "promoted").sum())}
    assert cal.mechanism_ok(cal.WEEKLY_EXAM, row)


def test_the_weekly_refit_ships_every_challenger(isolated_mlflow):
    df = _replay(replace(isolated_mlflow, **cal.ARMS[cal.WEEKLY_REFIT]))
    assert df["retrain_triggered"].all()
    assert (df["promotion_decision"] == "promoted").all()


def test_the_mechanism_check_catches_an_arm_that_skipped_a_week():
    short = {"runs": 40, "retrains": 39, "promotions": 39}
    assert not cal.mechanism_ok(cal.WEEKLY_EXAM, short)
    assert not cal.mechanism_ok(cal.WEEKLY_REFIT, short)
    assert cal.mechanism_ok(cal.SHIPPED_ARM, short)


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #


def _rows(readings: dict[tuple[str, str], list[str]], exam_share: float) -> list[dict]:
    rows = []
    for (x, y), found in readings.items():
        rows += [{"kind": "comparison", "group": "six" if i < 6 else "nine", "x": x, "y": y,
                  "reading": r} for i, r in enumerate(found)]
    rows += [{"kind": "arm", "arm": cal.WEEKLY_EXAM, "runs": 40,
              "promotions": int(40 * exam_share)} for _ in range(15)]
    return rows


def test_the_predictions_are_scored_as_registered():
    quiet = _cities(0, 0)
    readings = {pair: quiet for pair in cal.COMPARISONS}
    scores = cal.score(_rows(readings, 0.25), controls_ok=True)
    assert scores["predictions"] == {"P1": "held", "P2": "held", "P3": "held", "P4": "held"}

    readings[(cal.WEEKLY_REFIT, cal.SHIPPED_ARM)] = _cities(0, 3)
    scores = cal.score(_rows(readings, 0.6), controls_ok=True)
    assert scores["predictions"]["P2"] == "failed"
    assert scores["predictions"]["P4"] == "failed"


def test_a_failed_control_voids_every_prediction():
    readings = {pair: _cities(0, 0) for pair in cal.COMPARISONS}
    scores = cal.score(_rows(readings, 0.25), controls_ok=False)
    assert scores["void"] and set(scores["predictions"].values()) == {"void"}
