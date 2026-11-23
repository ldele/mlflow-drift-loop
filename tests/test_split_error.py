"""The split of each week's error, and its gate.

Fixed by docs/registrations/2026-11-23-why-a-forecast-was-wrong.md, so a change
here is a change to a registered design and should fail loudly. No network: the
end-to-end check runs on the simulated city.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]

import split_error as se  # noqa: E402

from driftloop import tracking  # noqa: E402
from driftloop.config import FEATURES, TARGET, LoopConfig, OpenMeteoConfig, ReplayWindows  # noqa: E402
from driftloop.data import OpenMeteoSource  # noqa: E402

T0 = pd.Timestamp("2026-01-05")


# --------------------------------------------------------------------------- #
# The registered constants                                                     #
# --------------------------------------------------------------------------- #


def test_the_registered_thresholds() -> None:
    assert se.MIN_COVERAGE == 0.95
    assert se.MIN_CITIES == 12
    assert se.REFIT_DAYS == 28
    assert se.SENSITIVITY_DAYS == (14, 56)
    assert se.PERMUTATIONS == 10_000
    assert se.P3_ALPHA == 0.05
    assert se.P4_FAILS_AT == 0.5
    assert se.SYNTHETIC_DIALS == (0.0, 2.0)


def test_two_thirds_rounds_up() -> None:
    assert se.two_thirds(15) == 10
    assert se.two_thirds(12) == 8
    assert se.two_thirds(13) == 9


# --------------------------------------------------------------------------- #
# The parts                                                                    #
# --------------------------------------------------------------------------- #


def test_the_three_parts_add_up_to_the_served_error() -> None:
    a, b, c, d = 100.0, 70.0, 60.0, 40.0
    assert sum(se.parts(a, b, c, d)) == pytest.approx(a)


def test_each_factor_is_averaged_over_both_orders() -> None:
    weather, relation, unexplained = se.parts(100.0, 70.0, 60.0, 40.0)
    # Weather first: 100 - 70 = 30; after the refit: 60 - 40 = 20.
    assert weather == pytest.approx(25.0)
    # Relation first: 100 - 60 = 40; on day-0 weather: 70 - 40 = 30.
    assert relation == pytest.approx(35.0)
    assert unexplained == 40.0


def test_identical_weather_gives_a_weather_part_of_exactly_zero() -> None:
    """The inert control's arithmetic: B equal to A and D equal to C."""
    weather, relation, _ = se.parts(100.0, 100.0, 60.0, 60.0)
    assert weather == 0.0
    assert relation == pytest.approx(40.0)


def test_the_largest_part_is_named() -> None:
    assert se.largest_part(100.0, 70.0, 60.0, 40.0) == "unexplained"
    assert se.largest_part(100.0, 20.0, 90.0, 10.0) == "weather"


# --------------------------------------------------------------------------- #
# Weeks and windows                                                            #
# --------------------------------------------------------------------------- #


def test_weeks_tile_the_replay_and_take_the_model_that_served_them() -> None:
    retro = SimpleNamespace(as_of=[T0 + pd.Timedelta(7 * i, unit="D") for i in range(3)],
                            serving_version=[1, 1, 2])
    weeks = se.weeks_of(retro, [False, True, False], 7)
    assert [w.version for w in weeks] == [1, 1, 2]
    assert [w.retrained for w in weeks] == [False, True, False]
    assert all(w.end - w.start == pd.Timedelta(7, unit="D") for w in weeks)
    # No gap and no overlap between one week and the next.
    assert all(weeks[i].end == weeks[i + 1].start for i in range(len(weeks) - 1))
    assert weeks[-1].end == retro.as_of[-1]


def test_the_refit_window_centres_the_week_and_stays_inside_the_span() -> None:
    span = (T0 - pd.Timedelta(100, unit="D"), T0 + pd.Timedelta(100, unit="D"))
    lo, hi = se.refit_window(T0, T0 + pd.Timedelta(7, unit="D"), 28, span)
    assert hi - lo == pd.Timedelta(28, unit="D")
    assert T0 - lo == pd.Timedelta(10.5, unit="D")
    # Near the end of the data the window shifts back, keeping its length.
    end = span[1] - pd.Timedelta(2, unit="D")
    lo, hi = se.refit_window(end - pd.Timedelta(7, unit="D"), end, 28, span)
    assert hi == span[1] and hi - lo == pd.Timedelta(28, unit="D")
    lo, hi = se.refit_window(span[0], span[0] + pd.Timedelta(7, unit="D"), 28, span)
    assert lo == span[0] and hi - lo == pd.Timedelta(28, unit="D")


def _frame(days: int, start: pd.Timestamp = T0 - pd.Timedelta(30, unit="D")) -> pd.DataFrame:
    stamps = pd.date_range(start, periods=24 * days, freq="h")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"timestamp": stamps})
    for feature in FEATURES:
        frame[feature] = rng.normal(size=len(stamps))
    frame[TARGET] = 10 + frame[FEATURES[0]] + rng.normal(size=len(stamps))
    return frame


@dataclass
class _Fit:
    """Stands in for fit_for, recording the hours each model was trained on."""

    seen: list

    def __call__(self, rows, cfg):
        self.seen.append(set(rows["timestamp"]))
        mean = float(rows[TARGET].mean())
        return SimpleNamespace(pipeline=SimpleNamespace(predict=lambda x: np.full(len(x), mean)))


def test_no_hour_is_scored_by_a_model_that_saw_it() -> None:
    frame = _frame(80)
    week = se.Week(T0, T0 + pd.Timedelta(7, unit="D"), 1, False)
    span = (frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1] + se.HOUR)
    fit = _Fit([])
    se.cross_fitted_sse(frame, week, 28, span, LoopConfig(), fit=fit)
    assert len(fit.seen) == 7, "one model per day of the week"
    for i, trained_on in enumerate(fit.seen):
        day = set(pd.date_range(T0 + pd.Timedelta(i, unit="D"), periods=24, freq="h"))
        assert not trained_on & day, f"day {i} was predicted by a model that saw it"
        # The other 27 days of the window, and nothing outside it.
        assert len(trained_on) == 27 * 24


def test_the_twin_trains_on_the_models_own_rows() -> None:
    frame = _frame(80)
    model = SimpleNamespace(train_start=T0, train_end=T0 + pd.Timedelta(10, unit="D") - se.HOUR)
    fit = _Fit([])
    se.twin(model, frame, LoopConfig(), fit=fit)
    assert min(fit.seen[0]) == model.train_start and max(fit.seen[0]) == model.train_end
    assert len(fit.seen[0]) == 10 * 24


def test_the_reach_refit_trains_on_the_loops_challenger_window() -> None:
    frame = _frame(400, start=T0 - pd.Timedelta(300, unit="D"))
    week = se.Week(T0, T0 + pd.Timedelta(7, unit="D"), 1, False)
    cfg = LoopConfig()
    fit = _Fit([])
    se.reach_sse(frame, week, cfg, fit=fit)
    assert min(fit.seen[0]) == T0 - pd.Timedelta(cfg.challenger_train_days, unit="D")
    assert max(fit.seen[0]) == T0 - pd.Timedelta(cfg.holdout_days, unit="D") - se.HOUR


# --------------------------------------------------------------------------- #
# The predictions                                                              #
# --------------------------------------------------------------------------- #


def _city(largest: str, smallest: str, relation: float, premium: float,
          retrain_weeks: list[str], enters: bool = True) -> dict:
    return {"enters": enters, "largest": largest, "smallest": smallest,
            "shares": {"relation": relation}, "premium": premium, "retrain_weeks": retrain_weeks}


def test_the_predictions_are_scored_as_registered() -> None:
    cities = [_city("unexplained", "relation", 0.01 * i, float(i), ["weather", "unexplained"])
              for i in range(15)]
    scores = se.score(cities, controls_ok=True)
    assert scores["entering"] == 15 and scores["two_thirds"] == 10
    assert scores["P1"]["verdict"] == "held" and scores["P2"]["verdict"] == "held"
    assert scores["P3"]["rho"] == pytest.approx(1.0)
    assert scores["P3"]["verdict"] == "held"
    assert scores["P4"]["share"] == 0.0 and scores["P4"]["verdict"] == "held"


def test_p1_fails_below_two_thirds_and_p4_fails_at_one_half() -> None:
    cities = ([_city("unexplained", "relation", 0.1, 1.0, ["relation"]) for _ in range(9)]
              + [_city("weather", "relation", 0.1, 1.0, ["relation", "weather"]) for _ in range(6)])
    scores = se.score(cities, controls_ok=True)
    assert scores["P1"]["unexplained_largest"] == 9
    assert scores["P1"]["verdict"] == "failed"
    # 15 relation weeks of 21: past one half.
    assert scores["P4"]["share"] == pytest.approx(15 / 21)
    assert scores["P4"]["verdict"] == "failed"


def test_a_city_left_out_by_the_gate_is_not_counted() -> None:
    cities = [_city("unexplained", "relation", 0.1, 1.0, []) for _ in range(12)]
    cities.append(_city("weather", "weather", 0.1, 1.0, [], enters=False))
    scores = se.score(cities, controls_ok=True)
    assert scores["entering"] == 12 and scores["two_thirds"] == 8
    assert scores["P1"]["unexplained_largest"] == 12


def test_too_few_cities_or_a_failed_control_voids_every_prediction() -> None:
    few = [_city("unexplained", "relation", 0.1 * i, float(i), []) for i in range(11)]
    assert all(se.score(few, True)[k]["verdict"] == "void" for k in ("P1", "P2", "P3", "P4"))
    enough = [_city("unexplained", "relation", 0.1 * i, float(i), []) for i in range(15)]
    assert all(se.score(enough, False)[k]["verdict"] == "void" for k in ("P1", "P2", "P3", "P4"))


def test_the_permutation_test_is_one_sided() -> None:
    x = list(range(15))
    rho, p = se.spearman_permutation(x, x, permutations=2000)
    assert rho == pytest.approx(1.0) and p < 0.01
    rho, p = se.spearman_permutation(x, x[::-1], permutations=2000)
    assert rho == pytest.approx(-1.0) and p > 0.99


def _described_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Four cities, the last kept out by the gate, and their weeks as weeks.csv holds them."""
    shares = [(0.1, 0.3, 0.6), (0.3, 0.1, 0.6), (-0.1, 0.4, 0.7), (0.9, 0.05, 0.05)]
    cities = pd.DataFrame([
        {"group": "six" if i < 2 else "nine", "city": f"c{i}", "enters": i < 3,
         "premium": float(i), "reach": 0.1 * i, "reproduction_error": 0.5, "inert_error": 0.4 + i / 10,
         **{f"{part}{suffix}": s for part, s in zip(se.PARTS, row, strict=True)
            for suffix in ("", "_14", "_56")}}
        for i, row in enumerate(shares)
    ])
    weeks = pd.DataFrame([
        {"group": "six", "city": "c0", "retrained": True, "largest": "relation"},
        {"group": "six", "city": "c0", "retrained": False, "largest": "unexplained"},
        {"group": "six", "city": "c1", "retrained": True, "largest": "unexplained"},
        {"group": "nine", "city": "c2", "retrained": False, "largest": "relation"},
        {"group": "nine", "city": "c3", "retrained": True, "largest": "relation"},
        {"group": "synthetic", "city": "Synthetic, concept dial 2", "retrained": True,
         "largest": "relation"},
    ])
    return weeks, cities


def test_the_readings_beside_the_predictions_use_only_the_cities_that_entered() -> None:
    weeks, cities = _described_tables()
    out = se.describe(weeks, cities)
    assert out["cities"] == 3
    weather = out["means"]["weather"]
    assert weather["point"] == pytest.approx(0.1)
    # A 95% t interval on three values: mean ± t(0.975, 2) × sd / √3.
    assert weather["hi"] - weather["point"] == pytest.approx(4.302653 * 0.2 / np.sqrt(3), rel=1e-5)
    assert out["reach"]["median"] == pytest.approx(0.1)
    assert out["windows"]["28"]["largest"] == {"weather": 0, "relation": 0, "unexplained": 3}
    assert out["windows"]["28"]["smallest"] == {"weather": 2, "relation": 1, "unexplained": 0}
    # Neither the gated-out city's weeks nor the simulated city's are read.
    assert out["retrain_weeks"] == {"weeks": 2, "relation_largest": 1, "share": 0.5}
    assert out["other_weeks"] == {"weeks": 2, "relation_largest": 1, "share": 0.5}
    assert out["groups"]["six"]["cities"] == 2 and out["groups"]["nine"]["cities"] == 1
    assert out["inert_error"] == {"min": pytest.approx(0.4), "max": pytest.approx(0.6)}


def test_the_readings_count_the_same_retrain_weeks_as_p4() -> None:
    weeks, cities = _described_tables()
    entering = cities[cities["enters"]]
    per_city = [list(weeks[(weeks.city == c) & weeks.retrained]["largest"]) for c in entering.city]
    scored = se.score([
        {"enters": True, "largest": "unexplained", "smallest": "weather",
         "shares": {"relation": r}, "premium": p, "retrain_weeks": w}
        for r, p, w in zip(entering["relation"], entering["premium"], per_city, strict=True)
    ], controls_ok=True)
    described = se.describe(weeks, cities)["retrain_weeks"]
    assert described["weeks"] == scored["P4"]["retrain_weeks"]
    assert described["relation_largest"] == scored["P4"]["relation_largest"]


# --------------------------------------------------------------------------- #
# The gate                                                                     #
# --------------------------------------------------------------------------- #


def test_coverage_is_the_share_of_week_ahead_hours_with_a_same_day_row() -> None:
    hours = pd.date_range("2026-01-01", periods=20, freq="h")
    ahead = pd.DataFrame({"timestamp": hours})
    same_day = pd.DataFrame({"timestamp": hours[:19].append(pd.DatetimeIndex(["2027-01-01"]))})
    # 19 of the 20 week-ahead hours; the extra same-day hour counts for nothing.
    assert se.coverage(ahead, same_day) == 19 / 20
    assert se.coverage(ahead.iloc[:0], same_day) == 0.0


def test_the_same_day_source_keeps_place_and_span_and_reads_the_forecast_archive(tmp_path) -> None:
    week_ahead = OpenMeteoSource(OpenMeteoConfig(forecast_lead_days=7), cache_dir=tmp_path)
    same_day = se.day0_source(week_ahead)
    cfg = same_day.config
    assert cfg.forecast_lead_days == 0 and cfg.day0_from_forecast_archive
    assert (cfg.latitude, cfg.longitude, cfg.origin, cfg.horizon) == (
        week_ahead.config.latitude, week_ahead.config.longitude,
        week_ahead.config.origin, week_ahead.config.horizon)
    assert same_day.cache_dir == tmp_path
    assert same_day._cache_path().name.endswith("_lead0d_forecast.parquet")


# --------------------------------------------------------------------------- #
# End to end, on the simulated city                                            #
# --------------------------------------------------------------------------- #


SHORT_PLAN = ReplayWindows(
    champion_train_start=pd.Timestamp("2025-04-01"),
    champion_train_end=pd.Timestamp("2025-07-01"),
    first_run=pd.Timestamp("2025-07-15"),
    last_run=pd.Timestamp("2025-08-12"),
)


def test_on_the_simulated_city_cell_a_reproduces_the_loop_and_the_inert_check_holds(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tracking, "REPO_ROOT", tmp_path)
    rows, reproduction = se.synthetic_weeks(2.0, SHORT_PLAN)
    assert len(rows) == 5
    # Cell A, rebuilt from the tags, is what the loop logged at every run, within
    # what rounding the tags can explain.
    assert reproduction <= 1.0
    # The twin trained on the same weather is the champion refitted.
    assert se.inert_error(rows) <= 1.0
    summary = se.summarise(rows)
    assert sum(summary["shares"].values()) == pytest.approx(1.0)
    for row in rows:
        assert row["hours"] == 7 * 24
        assert sum(se.parts(row["A"], row["B"], row["C"], row["D"])) == pytest.approx(row["A"])
        # No forecast here, so the weather part is zero up to tag rounding.
        assert row["C"] == row["D"] and abs(row["A"] - row["B"]) <= row["slack"]


def test_a_model_off_by_the_tag_rounding_stays_inside_the_slack() -> None:
    """The worst rounding of six-decimal tags moves the squared error by no more than the slack."""
    frame = _frame(7)
    x = frame[FEATURES].to_numpy(dtype=float)
    fitted = 10 + x[:, 0]
    # Every slope and the intercept off by the full half-digit, all the same way.
    rebuilt = fitted + se.TAG_STEP * (1 + np.abs(x).sum(axis=1))
    gap = abs(se._sse(frame[TARGET], fitted) - se._sse(frame[TARGET], rebuilt))
    assert 0 < gap <= se.tag_slack(frame, rebuilt)
    # And a model off by more than rounding is caught.
    wrong = fitted + 0.01
    assert abs(se._sse(frame[TARGET], fitted) - se._sse(frame[TARGET], wrong)) > se.tag_slack(frame, wrong)


def test_the_pages_split_each_week_with_the_registered_formula() -> None:
    from driftloop import split

    rng = np.random.default_rng(0)
    a, b, c, d = rng.uniform(0, 100, (4, 20))
    for mine, registered in zip(split.parts(a, b, c, d), se.parts(a, b, c, d), strict=True):
        np.testing.assert_allclose(mine, registered)
    assert split.PARTS == se.PARTS


def test_a_citys_weeks_are_drawn_per_hour_and_in_order() -> None:
    from driftloop import split

    weeks = pd.DataFrame([
        {"city": "x", "week_end": "2026-01-12", "hours": 84, "retrained": "True",
         "A": 84.0, "B": 84.0, "C": 42.0, "D": 42.0, "largest": "relation"},
        {"city": "x", "week_end": "2026-01-05", "hours": 168, "retrained": False,
         "A": 336.0, "B": 168.0, "C": 336.0, "D": 168.0, "largest": "weather"},
        {"city": "y", "week_end": "2026-01-05", "hours": 168, "retrained": False,
         "A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "largest": "unexplained"},
    ])
    out = split.city_weeks(weeks, "x")
    assert list(out["week_end"].dt.day) == [5, 12]
    assert list(out["retrained"]) == [False, True]
    assert out.loc[0, "weather"] == 1.0 and out.loc[0, "relation"] == 0.0
    assert out.loc[1, "relation"] == 0.5 and out.loc[1, "unexplained"] == 0.5
    np.testing.assert_allclose(out[list(split.PARTS)].sum(axis=1), out["error"])
