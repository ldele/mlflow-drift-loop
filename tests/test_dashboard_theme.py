"""Guards on the chart theme's per-feature assumptions.

These exist because widening the model from three features to eight broke every
dashboard chart that looked a feature up by name or by palette index, and
nothing caught it: the palette had four colours and the feature-colour map had
three entries, both silently encoding "there are three features". The site's
JavaScript was updated at the same time and the Streamlit side was not.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import theme  # noqa: E402

from driftloop.config import DRIFT_FEATURES  # noqa: E402


def test_palette_has_a_slot_for_every_drift_feature():
    """The IndexError that took out the PSI and coefficient charts."""
    assert len(theme.SERIES) >= len(DRIFT_FEATURES)


def test_every_drift_feature_has_a_colour():
    """The KeyError that took out the distribution histograms."""
    missing = [f for f in DRIFT_FEATURES if f not in theme.FEATURE_COLOR]
    assert not missing, f"no colour for {missing}"


def test_feature_colours_are_distinct():
    """Two features sharing a hue is a chart you cannot read."""
    used = [theme.FEATURE_COLOR[f] for f in DRIFT_FEATURES]
    assert len(set(used)) == len(used)


def test_status_colours_are_never_used_as_a_series():
    """Thresholds and decisions must not be mistakable for a feature line."""
    reserved = {theme.GOOD, theme.WARNING, theme.CRITICAL}
    assert reserved.isdisjoint(set(theme.SERIES))


@pytest.fixture()
def versions():
    return pd.DataFrame(
        {
            "version": ["1", "2", "3"],
            "train_end": pd.to_datetime(["2025-08-01", "2025-09-01", "2025-10-01"]),
            **{f"coef_{f}": [0.1, -0.2, 0.3] for f in DRIFT_FEATURES},
        }
    )


def test_coefficient_panels_render_one_per_feature(versions):
    fig = theme.coef_small_multiples(versions, DRIFT_FEATURES)
    assert len(fig.data) == len(DRIFT_FEATURES)
    titles = {a.text for a in fig.layout.annotations}
    assert set(DRIFT_FEATURES) <= titles


def test_each_coefficient_panel_gets_its_own_axis(versions):
    """The whole point of the small multiples: no shared y-axis.

    Effective coefficients are in original feature units and differ by ~500x in
    span, so one shared axis pins the small-unit features flat against zero.
    """
    fig = theme.coef_small_multiples(versions, DRIFT_FEATURES)
    axes = {trace.yaxis for trace in fig.data}
    assert len(axes) == len(DRIFT_FEATURES)
