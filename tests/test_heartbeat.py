"""The live loop's alarm on its own silence.

The weekly job failed every Monday from 2026-08-03 to 2026-09-07 and nothing
noticed. These pin the rule that would have: a store whose newest cycle was
logged more than eight days ago is stale, and a store with no cycle at all is
stale too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from driftloop import heartbeat
from driftloop.heartbeat import MAX_SILENCE_DAYS, check, last_logged

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]

LAST = pd.Timestamp("2026-07-27 06:12", tz="UTC")


def test_a_loop_that_ran_this_week_is_fresh():
    beat = check(LAST, LAST + pd.Timedelta(7, unit="D"))
    assert not beat.stale
    assert beat.days_since == 7


def test_one_day_of_slack_is_allowed_and_no_more():
    """A Monday run that lands on Tuesday passes; a whole skipped Monday does not."""
    assert check(LAST, pd.Timestamp("2026-08-04 23:59", tz="UTC")).days_since == MAX_SILENCE_DAYS
    assert not check(LAST, pd.Timestamp("2026-08-04 23:59", tz="UTC")).stale
    assert check(LAST, pd.Timestamp("2026-08-05 00:01", tz="UTC")).stale


def test_the_outage_would_have_failed_on_its_second_monday():
    """Last good cycle logged 07-27; the 08-03 run failed, so 08-10 is 14 days on."""
    beat = check(LAST, pd.Timestamp("2026-08-10 06:00", tz="UTC"))
    assert beat.stale
    assert beat.days_since == 14
    assert "14 days ago" in beat.message()
    assert "skipped" in beat.message()


def test_an_empty_store_is_stale_not_exempt():
    """A check that passes on nothing is how a wrong store path stays green."""
    beat = check(None, LAST)
    assert beat.stale
    assert beat.days_since is None
    assert "No cycle" in beat.message()


def test_naive_times_are_read_as_utc():
    naive = check(pd.Timestamp("2026-07-27 06:12"), pd.Timestamp("2026-07-28 06:12"))
    aware = check(LAST, LAST + pd.Timedelta(1, unit="D"))
    assert naive == aware


def test_the_message_counts_calendar_days_so_it_agrees_with_the_date():
    """Logged on the 9th, read on the 11th: two days ago, whatever the hour."""
    logged = pd.Timestamp("2026-11-09 15:16", tz="UTC")
    assert "today" in check(logged, pd.Timestamp("2026-11-09 23:00", tz="UTC")).message()
    assert "yesterday" in check(logged, pd.Timestamp("2026-11-10 01:00", tz="UTC")).message()
    assert "2 days ago" in check(logged, pd.Timestamp("2026-11-11 11:00", tz="UTC")).message()


# --------------------------------------------------------------------------- #
# Reading the store                                                            #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def store(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'hb.db').as_posix()}")
    yield "hb"
    mlflow.set_tracking_uri("")


def _log(experiment: str, cycle_type: str) -> pd.Timestamp:
    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(experiment).experiment_id) as run:
        mlflow.set_tags({"cycle_type": cycle_type})
    return pd.Timestamp(run.info.start_time, unit="ms", tz="UTC")


def test_a_missing_experiment_reads_as_no_cycles_and_is_not_created(store):
    assert last_logged(store) is None
    assert mlflow.get_experiment_by_name(store) is None


def test_the_newest_cycle_of_either_kind_is_the_heartbeat(store):
    mlflow.create_experiment(store)
    _log(store, "bootstrap")
    newest = _log(store, "monitor")
    assert last_logged(store) == newest


def test_a_bootstrap_alone_counts_as_alive(store):
    mlflow.create_experiment(store)
    first = _log(store, "bootstrap")
    assert last_logged(store) == first


def test_runs_that_are_not_cycles_are_ignored(store):
    """Fixture or registration runs without a cycle tag are not a sign of life."""
    mlflow.create_experiment(store)
    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(store).experiment_id):
        pass
    assert last_logged(store) is None


def test_the_script_fails_on_a_stale_store_and_warns_when_asked(store, monkeypatch, capsys):
    import check_heartbeat

    mlflow.create_experiment(store)
    _log(store, "monitor")
    uri = mlflow.get_tracking_uri()
    # Point the script at this test's store instead of the committed one. The
    # profile's config is frozen, so the reader is redirected rather than the name.
    monkeypatch.setattr(check_heartbeat.tracking, "tracking_uri", lambda _db: uri)
    monkeypatch.setattr(check_heartbeat.heartbeat, "last_logged", lambda _name: last_logged(store))
    monkeypatch.setattr(sys, "argv", ["check_heartbeat.py"])
    assert check_heartbeat.main() == 0

    later = pd.Timestamp.now(tz="UTC") + pd.Timedelta(MAX_SILENCE_DAYS + 1, unit="D")
    real_check = heartbeat.check
    monkeypatch.setattr(check_heartbeat.heartbeat, "check", lambda last: real_check(last, later))
    assert check_heartbeat.main() == 1
    monkeypatch.setattr(sys, "argv", ["check_heartbeat.py", "--warn-only"])
    assert check_heartbeat.main() == 0
    assert "skipped" in capsys.readouterr().out


def test_the_site_publishes_the_limit_the_job_fails_on():
    """The page ages the date in the reader's browser; its limit must be this one."""
    import json

    data = Path(__file__).resolve().parents[1] / "site" / "data.json"
    if not data.is_file():
        pytest.skip("site/data.json is a build output; run scripts/build_site.py")
    schedule = json.loads(data.read_text(encoding="utf-8")).get("schedule")
    if not schedule:
        pytest.skip("no scheduled cycles in this build")
    assert schedule["max_silence_days"] == MAX_SILENCE_DAYS
    assert heartbeat.check(schedule["logged_at_utc"]).last_logged is not None
