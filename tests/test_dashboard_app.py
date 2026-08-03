"""A smoke test that actually runs the Streamlit app.

The four chart crashes this guards against were all runtime lookups -- a palette
index, a colour-map key, an artifact key -- so nothing short of executing the
script would have caught them. Streamlit runs the whole file top to bottom,
including every ``with tab:`` block, so one clean run exercises all of them.

Skipped when the city backend has not been generated. It is gitignored, and CI
runs only `uv sync` + ruff + pytest without building it, so requiring it here
would fail CI for a missing artifact rather than a real defect.
"""

from pathlib import Path

import pytest

from driftloop.config import PROFILES

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = PROFILES["openmeteo"]
BACKEND = REPO_ROOT / PROFILE.db_filename

pytestmark = pytest.mark.skipif(
    not BACKEND.exists(),
    reason=f"{PROFILE.db_filename} not generated -- run scripts/run_openmeteo.py --fresh",
)


def test_the_dashboard_runs_without_raising():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(REPO_ROOT / "dashboard" / "app.py"), default_timeout=300)
    app.run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert not app.error, [str(e.value) for e in app.error]
    # If the script had died partway through, the later tabs would be missing.
    assert len(app.tabs) >= 5
