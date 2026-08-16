"""Baseline fairness at a forecast horizon.

The autoregressive baselines are the only predictors allowed to see past PM2.5,
which makes them the place a leak would hide. A forecaster issuing at T for
T+lead may use observations up to T and no later, so persistence has to repeat
the reading from issue time rather than the one just before the target. Getting
this wrong would hand the baseline a week of hindsight and make the model look
worse than it is -- silently, and in the direction that flatters nobody."""

import numpy as np
import pandas as pd

from driftloop.benchmark import autoregressive_lags, detail_map, predictor_columns
from driftloop.config import FEATURES, TARGET


def _timeline(hours: int = 500) -> pd.DataFrame:
    stamps = pd.date_range("2025-06-01", periods=hours, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp": stamps,
            **{f: rng.normal(size=hours) for f in FEATURES},
            # A ramp, so a lagged column is trivially distinguishable from the
            # actual: value == row index.
            TARGET: np.arange(hours, dtype=float),
        }
    )


def test_nowcast_lags_are_one_hour_and_one_day():
    assert autoregressive_lags(0) == (1, 24)


def test_forecast_lags_step_back_to_issue_time():
    persistence, seasonal = autoregressive_lags(7)
    assert persistence == 7 * 24
    # Seasonal naive must not collapse onto persistence at a whole-day lead,
    # or the table shows the same column twice under two names.
    assert seasonal != persistence
    assert seasonal == 7 * 24 + 24


def test_persistence_never_sees_past_issue_time():
    """The value used for a target row must be the one from `lead` hours before."""
    lead_days = 7
    timeline = _timeline()
    columns = predictor_columns(timeline, timeline.iloc[:200], lead_days=lead_days)

    lead_hours = lead_days * 24
    actual = columns["actual"].to_numpy(dtype=float)
    persistence = columns["persistence"].to_numpy(dtype=float)

    observed = ~np.isnan(persistence)
    # TARGET is the row index, so the gap is exactly the lag in rows.
    assert np.all(actual[observed] - persistence[observed] == lead_hours)
    # And the first `lead_hours` rows have no issue-time observation at all.
    assert np.all(np.isnan(persistence[:lead_hours]))


def test_details_describe_the_horizon_they_were_scored_at():
    assert "issued" in detail_map(7)["persistence"]
    assert "issued" not in detail_map(0)["persistence"]


# --------------------------------------------------------------------------- #
# Tuning, and the plumbing that lets a tuned model reach the loop               #
# --------------------------------------------------------------------------- #


def test_hyper_parameters_reach_the_estimator():
    """`params` has to override, or a tuned arm silently replays the default.

    The ablation's whole claim is that both model classes were tuned the same
    way. If these did not reach the estimator the script would report a tuned
    comparison and run a defaulted one, and nothing downstream would notice.
    """
    from driftloop.model import GBM, build_pipeline

    ridge = build_pipeline(params={"alpha": 42.0})
    assert ridge.named_steps["ridge"].alpha == 42.0
    # The positional default must still work, since it is what ships.
    assert build_pipeline().named_steps["ridge"].alpha == 1.0

    gbm = build_pipeline(kind=GBM, params={"max_leaf_nodes": 2, "learning_rate": 0.01})
    assert gbm.named_steps["gbm"].max_leaf_nodes == 2
    assert gbm.named_steps["gbm"].learning_rate == 0.01
    # Early stopping must stay off: its automatic form holds out a random slice,
    # which on autocorrelated hourly data is a validation set the model has
    # effectively already seen.
    assert gbm.named_steps["gbm"].early_stopping is False


def test_a_tuned_model_is_what_the_loop_actually_trains():
    """`LoopConfig.model_params` has to survive the trip into `train`."""
    from driftloop.config import LoopConfig
    from driftloop.model import train

    df = _timeline(400)
    cfg = LoopConfig(model_params={"alpha": 500.0})
    fitted = train(df, kind=cfg.model_kind, params=cfg.model_params)
    assert fitted.pipeline.named_steps["ridge"].alpha == 500.0


def test_tuning_picks_from_the_grid_it_was_given():
    """And reports what the default scored, which is the number that says
    whether tuning was worth doing at all."""
    from driftloop.benchmark import tune_gbm

    # Two candidates, so the test costs seconds rather than minutes.
    grid = {"max_leaf_nodes": (2, 7), "learning_rate": (0.1,)}
    sweep = tune_gbm(_timeline(400), grid=grid, n_splits=3)

    assert sweep.best["max_leaf_nodes"] in (2, 7)
    assert sweep.best["learning_rate"] == 0.1
    assert sweep.n_candidates == 2
    assert sweep.best_rmse <= sweep.default_rmse or np.isnan(sweep.default_rmse)


def test_tuning_degrades_rather_than_raising_on_a_window_too_short_to_split():
    from driftloop.benchmark import tune_gbm

    sweep = tune_gbm(_timeline(3), n_splits=5)
    assert sweep.best == {}
    assert np.isnan(sweep.gain_pct)
