import numpy as np
import pandas as pd

from driftloop.config import FEATURES, LoopConfig, SyntheticConfig
from driftloop.data import SyntheticSource
from driftloop.drift import compute_data_drift, compute_perf_drift, psi
from driftloop.model import rmse, train

TRAIN = (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"))
MONITOR = (pd.Timestamp("2025-10-15"), pd.Timestamp("2025-10-29"))


def test_psi_of_a_sample_against_itself_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x) < 1e-9


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    values = [psi(ref, rng.normal(loc=shift, size=5000)) for shift in (0.0, 0.5, 1.0, 2.0)]
    assert values == sorted(values)


def _measure(drift_strength: float, feature_shift: float):
    src = SyntheticSource(
        SyntheticConfig(drift_strength=drift_strength, feature_shift=feature_shift)
    )
    train_df = src.get_data(*TRAIN)
    monitor_df = src.get_data(*MONITOR)
    champion = train(train_df)
    data = compute_data_drift(train_df, monitor_df, FEATURES)
    perf = compute_perf_drift(
        champion.baseline_rmse, rmse(champion.pipeline, monitor_df), LoopConfig().perf_drift_threshold
    )
    return data.max_psi, perf.ratio


def test_feature_shift_moves_psi_monotonically():
    psis = [_measure(0.0, level)[0] for level in (0.0, 0.5, 1.0, 2.0)]
    assert psis == sorted(psis)


def test_concept_drift_moves_the_performance_signal_but_not_psi():
    """The two signals are independent -- that is the whole point of having both."""
    psi_low, ratio_low = _measure(0.0, 0.0)
    psi_high, ratio_high = _measure(2.0, 0.0)
    assert ratio_high > ratio_low * 1.5
    assert abs(psi_high - psi_low) < 1e-9


def test_perf_drift_threshold():
    assert not compute_perf_drift(10.0, 11.0, 1.25).detected
    assert compute_perf_drift(10.0, 15.0, 1.25).detected
