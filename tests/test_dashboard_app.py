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
