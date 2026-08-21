"""Smoke tests that actually run the Streamlit app.

The chart crashes these guard against are runtime lookups: a palette index, a
colour-map key, an artifact key. Nothing short of executing the script catches
them, and Streamlit runs the whole file top to bottom, including every
``with tab:`` block, so one clean run exercises all of them.

Every profile with a backend is run, not just the default. The live schedule's
backend was populated when the model had three features, and its drift reports
are written once and never rewritten, so indexing today's six-feature list into
one raised a ``KeyError`` while every other test stayed green.

Skipped where the backend has not been generated. Backends are gitignored and CI
does not build them, so requiring one here would fail on a missing artifact
rather than on a defect.
"""

from pathlib import Path

import pytest

from driftloop.config import PROFILES

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT = PROFILES["openmeteo"]

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / DEFAULT.db_filename).exists(),
    reason=f"{DEFAULT.db_filename} not generated -- run scripts/run_openmeteo.py --fresh",
)


def _run(profile_key: str | None = None):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(REPO_ROOT / "dashboard" / "app.py"), default_timeout=300)
    app.run()
    if profile_key is not None:
        app.sidebar.radio[0].set_value(profile_key)
        app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert not app.error, [str(e.value) for e in app.error]
    return app


def test_the_dashboard_runs_without_raising():
    app = _run()
    # If the script had died partway through, the later tabs would be missing.
    assert len(app.tabs) >= 5


@pytest.mark.parametrize(
    "profile_key",
    [k for k, p in PROFILES.items() if (REPO_ROOT / p.db_filename).exists()],
)
def test_every_generated_profile_renders(profile_key):
    """Every entry in the sidebar, not just the one that opens by default."""
    _run(profile_key)


def test_every_chart_gets_its_reading_rendered():
    """The "How to read this" panels reach the page, not just the source.

    `tests/test_guides.py` checks statically that every chart names a reading.
    This checks the other half: that Streamlit actually renders them, which a
    grep cannot see and which a mistyped key would silently fail.
    """
    app = _run()
    panels = [e for e in app.expander if e.label == "How to read this"]
    assert panels, "no readings rendered at all"
    # Six charts on the drift-loop tab alone, and the tabs below it add more.
    assert len(panels) >= 6, f"only {len(panels)} readings rendered"


def test_the_synthetic_knob_charts_get_theirs_too():
    """The one place a reading is passed through a loop variable.

    Every other panel names its key at the call site, so a typo there is caught
    by the static check. The knob sweep carries the key in the tuple it iterates,
    and that tab only renders under the synthetic profile, so it needs running.
    """
    app = _run("synthetic")  # the radio's options are keys, not labels
    if any("No data" in w.value for w in app.warning):
        pytest.skip("synthetic backend not generated -- run scripts/run_simulation.py --fresh")
    bodies = " ".join(m.value for m in app.markdown)
    # Each knob's reading names the dial it belongs to.
    assert "This dial moves the weather" in bodies, "the covariate knob has no reading"
    assert "changes how weather turns into pollution" in bodies, "the concept knob has no reading"
