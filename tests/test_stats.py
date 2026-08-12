"""Guards on the uncertainty machinery.

The point of this module is to stop the project reporting numbers it cannot
support, so its own failure mode is an interval that is too *narrow*, one
that manufactures confidence rather than measuring it. Most of these tests are aimed
at that direction specifically.
"""

from __future__ import annotations

import numpy as np
import pytest

from driftloop import stats


def ar1(n: int, rho: float, seed: int = 0) -> np.ndarray:
    """An AR(1) series: each value is ``rho`` times the last plus fresh noise."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + rng.normal()
    return out


def test_the_interval_contains_its_own_point_estimate():
    """The invariant that catches a bootstrap resampling a different statistic.

    If the resampled statistic and the reported point estimate are computed by
    different code, they drift apart and the interval stops bracketing the
    number it is attached to. That is silent and it is fatal, so it is asserted
    directly rather than trusted.
    """
    values = ar1(60, 0.4)
    interval = stats.block_bootstrap((values,), stats.mean_stat)
    assert interval.lo <= interval.point <= interval.hi


def test_autocorrelation_widens_the_interval():
    """The whole reason this module exists rather than calling a plain bootstrap.

    On a strongly autocorrelated series the IID bootstrap (block length 1)
    reports a narrower interval than the block bootstrap, because it destroys
    the dependence and so believes it has more independent data than it does.
    """
    values = ar1(120, 0.85, seed=3)
    iid = stats.block_bootstrap((values,), stats.mean_stat, block=1)
    blocked = stats.block_bootstrap((values,), stats.mean_stat, block=8)
    assert (blocked.hi - blocked.lo) > (iid.hi - iid.lo)


def test_independent_data_makes_the_two_agree():
    """The converse: with no autocorrelation, blocking should cost little.

    A block bootstrap that is much wider even on IID data would be adding
    uncertainty rather than measuring it.
    """
    rng = np.random.default_rng(11)
    values = rng.normal(size=200)
    iid = stats.block_bootstrap((values,), stats.mean_stat, block=1)
    blocked = stats.block_bootstrap((values,), stats.mean_stat, block=6)
    assert (blocked.hi - blocked.lo) < 1.6 * (iid.hi - iid.lo)


def test_the_interval_covers_a_known_answer():
    """Coverage on data whose true mean is known to be zero."""
    covered = 0
    trials = 40
    for seed in range(trials):
        values = ar1(150, 0.5, seed=seed)
        interval = stats.block_bootstrap((values,), stats.mean_stat, resamples=800, seed=seed)
        covered += interval.lo <= 0 <= interval.hi
    # Nominal coverage is 95%. Block bootstrap under-covers on short series, so
    # this asserts the loose property that matters -- it is not badly broken --
    # rather than a precise rate the method does not promise at n=150.
    assert covered >= int(0.8 * trials)


def test_pairs_are_resampled_together():
    """Paired statistics are meaningless if the pairing is broken by resampling.

    Two identical series have a ratio of 1 in every window, so the paired
    improvement is 0 in every resample. If the columns were
    resampled independently the ratio would scatter and the interval would open.
    """
    rng = np.random.default_rng(5)
    served = rng.uniform(5, 50, size=40)
    interval = stats.block_bootstrap(
        (served, served.copy()), stats.pct_improvement_paired, resamples=500
    )
    assert interval.point == pytest.approx(0.0)
    assert interval.lo == pytest.approx(0.0)
    assert interval.hi == pytest.approx(0.0)


def test_nan_rows_are_dropped_jointly():
    """Dropping NaNs column by column would silently misalign the pairs."""
    served = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    frozen = np.array([2.0, 2.0, np.nan, 8.0, 10.0, 12.0])
    interval = stats.block_bootstrap((served, frozen), stats.pct_improvement_paired, resamples=200)
    assert interval.n == 4  # rows 0, 3, 4, 5
    assert interval.point == pytest.approx(50.0)


def test_the_bootstrap_degenerates_at_a_boundary():
    """A documented failure, asserted so it cannot be forgotten.

    When every observation is a win, every resample is also all wins and the
    interval collapses to a point. That is not certainty. ``_win_cell`` in
    ``scripts/uncertainty.py`` reports Wilson instead for this reason,
    and this test is what stops someone 'simplifying' that back to the bootstrap.
    """
    served = np.arange(1.0, 17.0)
    frozen = served + 1.0  # served always wins
    interval = stats.block_bootstrap((served, frozen), stats.win_rate, resamples=500)
    assert interval.lo == interval.hi == 100.0

    lo, hi = stats.wilson_interval(16, 16)
    assert lo < 100.0, "Wilson must express uncertainty where the bootstrap cannot"
    assert hi == pytest.approx(100.0)


def test_effective_sample_size_falls_as_autocorrelation_rises():
    independent = np.random.default_rng(1).normal(size=200)
    correlated = ar1(200, 0.8, seed=1)
    assert stats.effective_sample_size(independent) > 100
    assert stats.effective_sample_size(correlated) < 60


def test_block_length_cannot_swallow_the_series():
    """A block as long as the series would resample it unchanged every time.

    That reports zero width, which is the most dangerous output this module
    could produce, so the default is capped at half the series.
    """
    for n in (4, 8, 19, 48, 200):
        assert stats.default_block_length(n) <= max(1, n // 2)
    assert stats.default_block_length(48) == 4


def test_intervals_are_reproducible():
    """A number nobody can re-derive is not evidence."""
    values = ar1(50, 0.6)
    first = stats.block_bootstrap((values,), stats.mean_stat)
    second = stats.block_bootstrap((values,), stats.mean_stat)
    assert (first.lo, first.hi) == (second.lo, second.hi)


def test_too_few_observations_refuses_to_invent_an_interval():
    """Three points cannot support an interval, and must not be given one.

    The gate calibration's long-serving group has three promotions, so
    this is the live case rather than a hypothetical.
    """
    interval = stats.block_bootstrap((np.array([1.0, 2.0, 3.0]),), stats.mean_stat)
    assert interval.point == pytest.approx(2.0)
    assert interval.lo == float("-inf")
    assert interval.hi == float("inf")
    assert not interval.excludes_zero


def test_sensitivity_sweep_reports_every_requested_block():
    values = ar1(48, 0.7)
    sweep = stats.sensitivity_to_block_length((values,), stats.mean_stat, resamples=300)
    assert set(sweep) == {1, 2, 3, 4, 6, 8}
    assert all(s.lo <= s.point <= s.hi for s in sweep.values())


def test_excludes_zero_is_not_fooled_by_a_straddling_interval():
    assert not stats.Interval(1.0, -0.5, 2.0, 10, 10.0, 2, 100).excludes_zero
    assert stats.Interval(1.0, 0.5, 2.0, 10, 10.0, 2, 100).excludes_zero
    assert stats.Interval(-1.0, -2.0, -0.5, 10, 10.0, 2, 100).excludes_zero


def test_the_median_of_ratios_is_blind_to_a_minority_of_bad_weeks():
    """Why ``differing`` exists, asserted rather than left in a comment.

    Two replay arms agree in most weeks because a changed trigger usually leaves
    the serving model alone. Those weeks are exact ties, the median ratio lands
    on 1.0, and the comparison reports no difference however badly the arm
    behaves in the weeks it does act.
    """
    off = np.full(20, 10.0)
    arm = off.copy()
    arm[:6] = 15.0  # a minority of weeks, 50% worse

    everything = stats.pct_improvement_paired(arm, off)
    assert everything == pytest.approx(0.0), "ties dominate, as designed"

    acted = stats.differing(arm, off)
    assert acted.sum() == 6
    only_acted = stats.pct_improvement_paired(arm[acted], off[acted])
    assert only_acted == pytest.approx(-50.0)


def test_differing_tolerates_float_noise_but_not_real_gaps():
    """Two arms that served the identical model must count as tied.

    The two error values come from separate replays, so they agree to floating
    point rather than bitwise. A strict `!=` would call every week a difference
    and the conditioned comparison would silently become the unconditioned one.
    """
    base = np.array([10.0, 20.0, 30.0])
    assert not stats.differing(base, base * (1 + 1e-15)).any()
    assert stats.differing(base, base + 0.01).all()


# --------------------------------------------------------------------------- #
# The model-class ablation needs a non-linear champion, which breaks two        #
# assumptions the rest of the project is built on: that a version serialises    #
# into nine numbers, and that the registry is enough to score one.              #
# --------------------------------------------------------------------------- #


def test_a_tree_is_not_mistaken_for_something_with_slopes():
    from driftloop.model import GBM, RIDGE, build_pipeline, is_linear

    assert is_linear(build_pipeline(kind=RIDGE))
    assert not is_linear(build_pipeline(kind=GBM))


def test_an_unknown_model_kind_fails_loudly():
    """Silently falling back to Ridge would make an ablation compare it to itself."""
    from driftloop.model import build_pipeline

    with pytest.raises(ValueError, match="unknown model kind"):
        build_pipeline(kind="randomforest")


def test_a_tree_registers_without_fake_coefficients():
    """Writing a coefficient for a tree would make an unscoreable version look scoreable.

    ``retrospect.registered_models`` decides whether it can rebuild a version by
    looking for coefficient tags. A placeholder there would be silently wrong
    rather than loudly missing.
    """
    import pandas as pd

    from driftloop.data import SyntheticSource
    from driftloop.model import GBM, RIDGE, train
    from driftloop.tracking import _version_tags

    df = SyntheticSource().get_data(pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-01"))
    assert any(k.startswith("coef_") for k in _version_tags(train(df, kind=RIDGE)))
    tree_tags = _version_tags(train(df, kind=GBM))
    assert not any(k.startswith("coef_") for k in tree_tags)
    assert "baseline_rmse" in tree_tags, "the non-coefficient tags still have to be written"
