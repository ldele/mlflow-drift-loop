"""The retrospective scoring, and the one claim everything in it rests on.

Every decay curve, skill number and gate-calibration point comes from models
rebuilt out of their registry coefficient tags rather than from refitting. If
that reconstruction is not exact, every chart built on it is quietly wrong, so
it is asserted against the loop's own independently logged RMSE rather than
against another copy of the same arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftloop.config import FEATURES, LoopConfig, SyntheticConfig, TARGET
from driftloop.data.synthetic import SyntheticSource
from driftloop.model import effective_coefficients, train
from driftloop.retrospect import (
    CLIMATOLOGY_DAYS,
    RegisteredModel,
    build,
    climatology_prediction,
    training_window_stats,
)


@pytest.fixture(scope="module")
def source() -> SyntheticSource:
    return SyntheticSource(SyntheticConfig())


def _as_registered(trained, version: int) -> RegisteredModel:
    """The same trip through coefficient tags the registry round-trips a model on."""
    coefficients = effective_coefficients(trained.pipeline)
    return RegisteredModel(
        version=version,
        # Through str() and back, because that is what the registry stores: the
        # tags are text, so any precision lost there is lost in production too.
        coefficients={f: float(f"{coefficients[f]:.6f}") for f in FEATURES},
        intercept=float(f"{coefficients['intercept']:.6f}"),
        train_start=trained.train_start,
        train_end=trained.train_end,
        baseline_rmse=trained.baseline_rmse,
    )


def test_coefficient_reconstruction_matches_the_fitted_pipeline(source):
    """A version rebuilt from its tags predicts what the pipeline it came from does.

    This is the load-bearing assumption of the whole module: it lets old versions
    be scored on windows they never served without keeping a pickled model for
    each. Six-decimal tags leave a little rounding, so this is tight rather than
    exact.
    """
    window = source.get_data(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01"))
    trained = train(window)
    rebuilt = _as_registered(trained, 1)

    later = source.get_data(pd.Timestamp("2025-04-01"), pd.Timestamp("2025-05-01"))
    from_pipeline = trained.pipeline.predict(later[FEATURES])
    from_tags = rebuilt.predict(later)

    assert np.allclose(from_pipeline, from_tags, atol=1e-3)


def test_climatology_never_reads_past_the_forecast_issue_time(source, monkeypatch):
    """The baseline may only average hours observable when the forecast went out.

    A baseline that peeks at the window it is scoring would beat any real model
    and make the skill number meaningless, so the causality is asserted rather
    than trusted to the arithmetic.
    """
    window_start = pd.Timestamp("2025-06-01")
    window = source.get_data(window_start, window_start + pd.Timedelta(14, "D"))

    requested: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    original = source.get_data

    def spy(start, end):
        requested.append((start, end))
        return original(start, end)

    monkeypatch.setattr(source, "get_data", spy)
    climatology_prediction(source, window, window_start, lead_days=7)

    (start, end), = requested
    assert end == window_start - pd.Timedelta(7, "D"), "reference must stop a lead before the window"
    assert start == end - pd.Timedelta(CLIMATOLOGY_DAYS, "D")
    assert end <= window_start


def test_climatology_is_a_beatable_but_real_baseline(source):
    """Sanity: the baseline predicts, rather than being trivially awful or perfect."""
    window_start = pd.Timestamp("2025-06-01")
    window = source.get_data(window_start, window_start + pd.Timedelta(14, "D"))
    predicted = climatology_prediction(source, window, window_start, lead_days=0)
    actual = window[TARGET].to_numpy(dtype=float)

    assert np.isfinite(predicted).all()
    error = float(np.sqrt(np.nanmean((actual - predicted) ** 2)))
    assert 0 < error < float(np.std(actual)) * 3


def test_build_scores_every_version_on_every_window(source):
    """Each version gets a full-length curve, including windows it never served."""
    early = train(source.get_data(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01")))
    late = train(source.get_data(pd.Timestamp("2025-05-01"), pd.Timestamp("2025-07-01")))
    models = {1: _as_registered(early, 1), 2: _as_registered(late, 2)}

    as_of = pd.date_range("2025-07-15", periods=6, freq="7D")
    runs = pd.DataFrame({"as_of": as_of, "champion_version": [1, 1, 1, 2, 2, 2]})

    result = build(source, runs, models, monitor_days=14, lead_days=0, promotion_margin=0.05)

    assert len(result.as_of) == len(as_of)
    for version in models:
        assert len(result.version_rmse[version]) == len(as_of)
        assert len(result.version_skill[version]) == len(as_of)
    # The champion series must agree with the per-version series it is drawn from,
    # or the headline number and the decay curves would be telling different
    # stories about the same model.
    for i, version in enumerate(runs["champion_version"]):
        assert result.champion_rmse[i] == result.version_rmse[version][i]
    assert result.promoted_at[2] == as_of[3]


def test_skill_is_positive_exactly_when_the_model_beats_the_baseline(source):
    trained = train(source.get_data(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01")))
    models = {1: _as_registered(trained, 1)}
    as_of = pd.date_range("2025-03-15", periods=4, freq="7D")
    runs = pd.DataFrame({"as_of": as_of, "champion_version": [1] * 4})

    result = build(source, runs, models, monitor_days=14, lead_days=0, promotion_margin=0.05)

    for rmse, climatology, skill in zip(
        result.champion_rmse, result.climatology_rmse, result.champion_skill
    ):
        assert skill == pytest.approx(1 - rmse / climatology)
        assert (skill > 0) == (rmse < climatology)


def test_the_champion_series_uses_the_loop_s_own_logged_error(source):
    """The serving champion's error is read, not reconstructed.

    A run that promotes monitors with the *outgoing* champion and then writes the
    winner onto its ``champion_version`` tag. Reconstructing from the tag would
    credit the new model with a window it never served, and that window overlaps
    the challenger's training data, so the credit flatters it.
    """
    early = train(source.get_data(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01")))
    late = train(source.get_data(pd.Timestamp("2025-05-01"), pd.Timestamp("2025-07-01")))
    models = {1: _as_registered(early, 1), 2: _as_registered(late, 2)}

    as_of = pd.date_range("2025-07-15", periods=4, freq="7D")
    sentinel = 123.456
    runs = pd.DataFrame(
        {
            "as_of": as_of,
            "champion_version": [1, 2, 2, 2],
            "tags.promotion_decision": ["none", "promoted", "none", "none"],
            "metrics.champion_rmse": [sentinel, sentinel, sentinel, sentinel],
        }
    )

    result = build(source, runs, models, monitor_days=14, lead_days=0, promotion_margin=0.05)

    assert result.champion_rmse == [sentinel] * 4, "logged error should win over reconstruction"
    # The promoting run was served by version 1, whatever its tag now says.
    assert result.serving_version == [1, 1, 2, 2]


def test_paired_value_counts_windows_by_who_served_them(source):
    """"Did retraining help" must not be decided by comparing two floats.

    The served figure is logged and the frozen one is reconstructed, so for the
    identical model they agree only to the precision of the coefficient tags.
    Testing ``served != frozen`` therefore marks every window as retrained and
    averages in the ones from before anything was promoted.
    """
    from driftloop.retrospect import retraining_value

    early = train(source.get_data(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01")))
    late = train(source.get_data(pd.Timestamp("2025-05-01"), pd.Timestamp("2025-07-01")))
    models = {1: _as_registered(early, 1), 2: _as_registered(late, 2)}

    as_of = pd.date_range("2025-07-15", periods=6, freq="7D")
    runs = pd.DataFrame(
        {
            "as_of": as_of,
            "champion_version": [1, 1, 1, 1, 2, 2],
            "tags.promotion_decision": ["none"] * 4 + ["promoted", "none"],
        }
    )

    value = build(source, runs, models, 14, 0, 0.05)
    scored = retraining_value(value)

    # Promotion lands on window 4, and that window was still served by v1, so
    # only the last window counts.
    assert scored["acted_windows"] == 1
    assert scored["windows"] == 6


def test_gate_calibration_excludes_the_promotion_window_itself(source):
    """The delivered margin must not be measured on the challenger's own training data.

    The monitor window at the promotion runs from 14 days before ``as_of``, and a
    challenger trains up to 7 days before it -- so the first half of that window
    is inside its training set. Scoring there would flatter every promotion.
    """
    early = train(source.get_data(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01")))
    late = train(source.get_data(pd.Timestamp("2025-05-01"), pd.Timestamp("2025-07-01")))
    models = {1: _as_registered(early, 1), 2: _as_registered(late, 2)}

    as_of = pd.date_range("2025-07-15", periods=6, freq="7D")
    runs = pd.DataFrame(
        {
            "as_of": as_of,
            "champion_version": [1, 1, 1, 2, 2, 2],
            "metrics.champion_rmse_holdout": [np.nan, np.nan, np.nan, 10.0, np.nan, np.nan],
            "metrics.challenger_rmse": [np.nan, np.nan, np.nan, 8.0, np.nan, np.nan],
        }
    )

    result = build(source, runs, models, monitor_days=14, lead_days=0, promotion_margin=0.05)

    (gate,) = result.gate
    assert gate["version"] == 2 and gate["replaced"] == 1
    # 1 - 8/10, straight off the exam the loop logged.
    assert gate["exam_margin"] == pytest.approx(0.2)

    promotion_index = 3
    served = slice(promotion_index + 1, 6)
    expected = 1 - np.median(result.version_rmse[2][served]) / np.median(result.version_rmse[1][served])
    assert gate["delivered_margin"] == pytest.approx(expected)


def test_training_window_band_covers_the_bulk_of_what_the_model_saw(source):
    """The band is a 10th-90th percentile range, so at least 80% of hours sit in it.

    At least, not exactly: precipitation is zero for ~89% of hours, so both the
    10th percentile and the band's lower edge land inside that atom and the band
    sweeps up every zero with them. Weather features are lumpy and the assertion
    has to survive that.

    And not "the band contains the mean": a feature that is zero most of the time
    with occasional downpours has a mean above its own 90th percentile, which is
    a fact about rain rather than a bug in the band.
    """
    window = source.get_data(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01"))
    stats = training_window_stats(source, _as_registered(train(window), 1))

    assert stats, "expected a band per drift feature"
    for feature, entry in stats.items():
        assert entry["lo"] <= entry["hi"]
        values = window[feature].to_numpy(dtype=float)
        inside = float(np.mean((values >= entry["lo"]) & (values <= entry["hi"])))
        assert 0.78 <= inside <= 1.0, f"{feature} band covers {inside:.0%}"


def test_versions_without_coefficient_tags_are_skipped():
    """A version we cannot reconstruct must be dropped, never guessed at."""
    from driftloop.retrospect import registered_models

    class Version:
        def __init__(self, version, tags):
            self.version = version
            self.tags = tags

    class Client:
        def search_model_versions(self, _filter):
            return [
                Version("1", {}),  # predates coefficient logging
                Version("2", {**{f"coef_{f}": "0.5" for f in FEATURES}, "coef_intercept": "1.0"}),
            ]

    models = registered_models(Client(), "any")
    assert list(models) == [2]


def test_empty_inputs_degrade_rather_than_raise(source):
    """A profile with no runs or no registered versions returns an empty result."""
    empty = build(source, pd.DataFrame(), {}, 14, 0, 0.05)
    assert empty.as_of == [] and empty.gate == []

    runs = pd.DataFrame({"as_of": pd.date_range("2025-07-15", periods=2, freq="7D"),
                         "champion_version": [1, 1]})
    assert build(source, runs, {}, 14, 0, LoopConfig().promotion_margin).as_of == []
