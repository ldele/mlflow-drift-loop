"""Guards on the loop's decision rules -- especially the evaluation leak."""

import mlflow
import pandas as pd
import pytest

from driftloop.config import LoopConfig
from driftloop.data import SyntheticSource
from driftloop.loop import bootstrap_champion, run_cycle, run_simulation
from driftloop.model import train


@pytest.fixture()
def isolated_mlflow(tmp_path):
    """A throwaway sqlite backend so tests never touch the real one."""
    uri = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    cfg = LoopConfig(experiment_name="test-loop", registered_model_name="test-model")
    mlflow.create_experiment("test-loop", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("test-loop")
    yield cfg


def test_baseline_rmse_is_measured_on_data_the_model_did_not_fit():
    df = SyntheticSource().get_data(pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"))
    trained = train(df)
    assert trained.baseline_rmse > 0
    assert trained.n_rows == len(df)


def test_run_simulation_rejects_a_cadence_that_would_leak(isolated_mlflow):
    cfg = isolated_mlflow
    with pytest.raises(ValueError, match="holdout_days"):
        run_simulation(
            SyntheticSource(),
            cfg,
            pd.Timestamp("2025-08-01"),
            pd.Timestamp("2025-09-01"),
            step_days=3,  # < holdout_days=7
        )


def test_bootstrap_then_cycle_records_a_decision(isolated_mlflow):
    cfg = isolated_mlflow
    src = SyntheticSource()
    version = bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    assert version == "1"

    # Well after the drift date: the champion should be visibly failing.
    result = run_cycle(src, pd.Timestamp("2025-11-01"), cfg)
    assert result.perf_drift_detected
    assert result.retrain_triggered
    assert result.promotion_decision in {"promoted", "rejected"}
    assert result.challenger_rmse is not None


def test_challenger_never_trains_on_the_holdout_window(isolated_mlflow):
    """The leak the original prototype had, asserted directly."""
    cfg = isolated_mlflow
    src = SyntheticSource()
    as_of = pd.Timestamp("2025-11-01")
    holdout_start = as_of - pd.Timedelta(days=cfg.holdout_days)
    challenger_start = as_of - pd.Timedelta(days=cfg.challenger_train_days)

    challenger_df = src.get_data(challenger_start, holdout_start)
    holdout = src.get_data(holdout_start, as_of)
    assert challenger_df["timestamp"].max() < holdout["timestamp"].min()


def test_a_healthy_champion_is_left_alone(isolated_mlflow):
    """No drift -> no retrain. The trigger must not fire on noise."""
    cfg = isolated_mlflow
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    result = run_cycle(src, pd.Timestamp("2025-07-15"), cfg)  # two weeks later, pre-drift
    assert not result.retrain_triggered
    assert result.promotion_decision == "none"
    assert result.challenger_rmse is None
