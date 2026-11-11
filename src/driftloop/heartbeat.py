"""Is the live loop still running? A check on its own silence.

The weekly job failed every Monday from 2026-08-03 to 2026-09-07 and nothing
noticed: a loop built to notice that a model has gone stale had no alarm for
going stale itself. This is that alarm. It asks one question of the tracking
store, *when was a cycle last logged*, and calls the loop stale once that is
more than ``MAX_SILENCE_DAYS`` ago.

It reads the wall-clock time a run was logged, not the run's ``as_of``. The
weekly runner deliberately targets a date a week behind today (the weather and
air-quality reanalysis lag), so ``as_of`` is a week old the moment a cycle
finishes. When the cycle *ran* is the heartbeat; what week it looked at is not.

What it can and cannot see: a cycle run by hand and one run by the weekly
Action leave the same record, so "fresh" means "somebody ran a cycle", not "the
cron is healthy". It runs in three places: inside the weekly job after each
cycle, in a daily workflow of its own (``heartbeat.yml``) for the job that never
starts, and in the reader's browser on the site's schedule card. A fourth
alarm lives off GitHub: after a cycle passes this check, the weekly job pings an
outside dead-man's switch, which alerts when the pings stop.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# A weekly job, plus one day of slack for a queued or slow runner, in calendar
# days. Past this, at least one Monday has gone by without a cycle.
MAX_SILENCE_DAYS = 8

# Both kinds of run the loop logs. The bootstrap counts: a deployment that has
# only just registered its first champion is alive, not silent.
CYCLE_TYPES = ("monitor", "bootstrap")


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


@dataclass(frozen=True)
class Heartbeat:
    """The age of the newest logged cycle, and whether that is too old."""

    last_logged: pd.Timestamp | None
    checked_at: pd.Timestamp
    limit_days: int = MAX_SILENCE_DAYS

    @property
    def days_since(self) -> int | None:
        """Calendar days (UTC) from the last cycle's date to the check's date.

        Calendar days rather than elapsed hours, so the count agrees with the
        date printed beside it: a cycle logged on the 9th is "2 days ago" on the
        11th whatever the hour.
        """
        if self.last_logged is None:
            return None
        return (self.checked_at.normalize() - self.last_logged.normalize()).days

    @property
    def stale(self) -> bool:
        """True when no cycle is on record, or the newest is more than the limit ago.

        An empty store is stale rather than exempt: a heartbeat that passes when
        there is nothing to read is how a broken backend path stays green.
        """
        return self.days_since is None or self.days_since > self.limit_days

    def message(self) -> str:
        if self.last_logged is None:
            return "No cycle has been logged in this tracking store."
        when = f"The last cycle was logged on {self.last_logged:%Y-%m-%d %H:%M} UTC"
        days = self.days_since
        ago = "today" if days == 0 else "yesterday" if days == 1 else f"{days} days ago"
        verdict = (
            f"more than the {self.limit_days} days a weekly job should ever leave, "
            f"so at least one week was skipped"
            if self.stale
            else f"within the {self.limit_days}-day limit"
        )
        return f"{when}, {ago}: {verdict}."


def check(last_logged, now=None, limit_days: int = MAX_SILENCE_DAYS) -> Heartbeat:
    """Judge a last-logged time against *now* (default: the current UTC time).

    Naive timestamps are taken as UTC, which is what MLflow stores.
    """
    checked_at = _utc(pd.Timestamp.now(tz="UTC") if now is None else now)
    last = None if last_logged is None or pd.isna(last_logged) else _utc(last_logged)
    return Heartbeat(last_logged=last, checked_at=checked_at, limit_days=limit_days)


def last_logged(experiment_name: str) -> pd.Timestamp | None:
    """When the newest cycle in the current tracking store was logged, in UTC.

    Reads whatever store ``mlflow`` currently points at and creates nothing:
    a missing experiment is reported as no cycles, not quietly made.
    """
    if MlflowClient().get_experiment_by_name(experiment_name) is None:
        return None
    # One query per kind: MLflow's filter grammar has no IN for tags.
    newest = []
    for cycle_type in CYCLE_TYPES:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=f"tags.cycle_type = '{cycle_type}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs.empty:
            newest.append(_utc(runs["start_time"].iloc[0]))
    return max(newest) if newest else None
