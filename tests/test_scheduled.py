"""The weekly runner's catch-up arithmetic.

The Action failed every Monday from 2026-08-03 to 2026-09-07. When it came back
it ran the current week and nothing else, so six weekly decisions were never
taken at all: six windows nobody monitored, and a champion that went on serving
because no evidence against it was ever collected. A loop that skips the weeks it
was down is not running weekly, it is running whenever it happens to work.

These are on the date arithmetic alone, which is where the defect was. Running a
cycle is `test_loop.py`'s subject.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]

from run_scheduled import STEP_DAYS, missed_cycles  # noqa: E402


def test_a_loop_that_ran_last_week_has_nothing_to_catch_up():
    """The ordinary case, and it has to stay free: no extra cycles, no extra cost."""
    last = pd.Timestamp("2026-09-14")
    assert missed_cycles(last, last + pd.Timedelta(STEP_DAYS, unit="D"), 12) == []


def test_the_six_weeks_the_action_was_down_are_the_six_it_runs():
    """The outage, exactly as it happened: last good cycle 07-20, resumed at 09-07."""
    dates = missed_cycles(pd.Timestamp("2026-07-20"), pd.Timestamp("2026-09-07"), 12)

    assert [str(d.date()) for d in dates] == [
        "2026-07-27", "2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24", "2026-08-31",
    ]


def test_a_long_outage_is_truncated_from_the_oldest_end_forward():
    """Bounded so one CI run cannot become unbounded, and in order so state holds.

    Oldest first matters: the loop is stateful, and a promotion in the second
    missed week is what the third one monitors. Taking the most recent N instead
    would monitor a champion this backend never promoted.
    """
    dates = missed_cycles(pd.Timestamp("2026-07-20"), pd.Timestamp("2026-09-07"), 3)

    assert [str(d.date()) for d in dates] == ["2026-07-27", "2026-08-03", "2026-08-10"]
    assert dates == sorted(dates)


def test_a_backend_with_no_cycles_yet_catches_up_nothing():
    """A fresh deployment bootstraps and starts from there; there is no history to replay."""
    assert missed_cycles(None, pd.Timestamp("2026-09-07"), 12) == []
