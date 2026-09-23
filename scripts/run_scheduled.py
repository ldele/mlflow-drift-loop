"""One incremental scheduled cycle against a persistent backend.

What the weekly GitHub Action calls. Where the replay scripts walk a whole
timeline in one process, each invocation here does one thing and exits:

  * no champion registered yet  -> bootstrap one on the trailing window,
  * champion exists             -> run a single monitoring cycle at ``as_of``.

State lives in a persistent MLflow backend (``mlflow_scheduled.db``): locally
that is just the file, and in CI the workflow commits it back so the next run
continues where this one left off. Nothing is ever reset.

    python scripts/run_scheduled.py [--as-of YYYY-MM-DD] [--lag-days N]

``--as-of`` pins the run date (useful for backfills and for local testing across
several "weeks"); without it, the run targets ``today - lag-days`` -- the lag
covers the reanalysis delay in the weather / air-quality feeds.

**Missed weeks are run, not skipped.** The Action failed every Monday from
2026-08-03 to 2026-09-07 and then resumed at the current week, so six weekly
decisions were never taken: the loop cannot notice drift in a window it never
looked at, and the champion of the day kept serving on the strength of no
evidence at all. Each invocation now walks forward one step at a time from the
last cycle it recorded, which is what "runs every week" was always supposed to
mean. ``--no-catch-up`` restores the old behaviour, and ``--max-catch-up``
bounds a very long outage so one CI run cannot become unbounded; whatever is
left is picked up by the next invocation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import pandas as pd

from driftloop import tracking
from driftloop.config import PROFILES, OpenMeteoConfig
from driftloop.data import OpenMeteoSource
from driftloop.loop import bootstrap_champion, run_cycle
from driftloop.tracking import load_champion

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"
PROFILE = PROFILES["scheduled"]

# The first champion is trained on the trailing ~2.5 months before deployment.
BOOTSTRAP_TRAIN_DAYS = 75
# Fetch a generous trailing span so any recent champion's training window and the
# monitor/challenger windows are all covered by one cached pull.
TRAILING_FETCH_DAYS = 400
# The cadence the Action runs at, and the step catch-up walks in.
STEP_DAYS = 7


def _resolve_as_of(args: argparse.Namespace) -> pd.Timestamp:
    if args.as_of:
        return pd.Timestamp(args.as_of).normalize()
    # ERA5 / air-quality reanalysis lags real time; step back to a safe date.
    return pd.Timestamp.now().normalize() - pd.Timedelta(args.lag_days, unit="D")


def last_recorded_cycle(cfg) -> pd.Timestamp | None:
    """The ``as_of`` of the newest cycle in this backend, or None if there is none.

    Read from the runs rather than from the champion, because the question is
    when the loop last *looked*, not when it last acted. A champion that has
    served through six unexamined weeks looks identical to one examined weekly
    and left alone.
    """
    runs = mlflow.search_runs(
        experiment_names=[cfg.experiment_name],
        filter_string="tags.cycle_type = 'monitor'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty or "params.as_of" not in runs:
        return None
    return pd.Timestamp(runs["params.as_of"].iloc[0]).normalize()


def missed_cycles(last: pd.Timestamp | None, target: pd.Timestamp, limit: int) -> list[pd.Timestamp]:
    """Every cycle date between the last one recorded and *target*, oldest first.

    Empty when the loop is up to date, which is the ordinary case: a weekly run
    that ran last week has nothing to catch up on.
    """
    if last is None:
        return []
    dates = []
    at = last + pd.Timedelta(STEP_DAYS, unit="D")
    while at < target:
        dates.append(at)
        at = at + pd.Timedelta(STEP_DAYS, unit="D")
    # Oldest first, and truncated from the front rather than the back. The loop
    # is stateful: a promotion in week 2 is what week 3 monitors. Running the
    # most recent N would step over that and monitor a champion the backend
    # never promoted.
    return dates[:limit] if limit and len(dates) > limit else dates


def _emit_ci(*, promotion: bool, headline: str) -> None:
    """Expose a ``promotion`` step-output and a summary line under GitHub
    Actions, for the notice step to key on. A no-op locally."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as fh:
            fh.write(f"promotion={'true' if promotion else 'false'}\n")
            fh.write(f"headline={headline}\n")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        icon = "🔺" if promotion else "•"
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write(f"{icon} {headline}\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=str, default=None, help="YYYY-MM-DD; default = today - lag")
    parser.add_argument("--lag-days", type=int, default=7)
    parser.add_argument(
        "--no-catch-up",
        dest="catch_up",
        action="store_false",
        help="run only the target cycle, leaving any missed weeks unexamined",
    )
    parser.add_argument(
        "--max-catch-up",
        type=int,
        default=12,
        help="most missed weeks to run in one invocation; the rest wait for the next",
    )
    args = parser.parse_args()

    as_of = _resolve_as_of(args)
    cfg = PROFILE.loop
    # setup only, never reset(): this backend accrues history across runs.
    tracking.setup(cfg.experiment_name, PROFILE.db_filename)

    om_cfg = OpenMeteoConfig(
        origin=(as_of - pd.Timedelta(TRAILING_FETCH_DAYS, unit="D")).normalize(),
        horizon=(as_of + pd.Timedelta(1, unit="D")).normalize(),
    )
    source = OpenMeteoSource(om_cfg)

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / PROFILE.meta_filename).write_text(
        json.dumps(
            {
                "drift_date": None,
                "location": om_cfg.name,
                "latitude": om_cfg.latitude,
                "longitude": om_cfg.longitude,
            }
        ),
        encoding="utf-8",
    )

    champion = load_champion(cfg.registered_model_name)
    if champion is None:
        train_start = as_of - pd.Timedelta(BOOTSTRAP_TRAIN_DAYS, unit="D")
        print(f"[{as_of.date()}] no champion yet -> bootstrapping on "
              f"{train_start.date()} .. {as_of.date()} (first deploy)")
        version = bootstrap_champion(source, train_start, as_of, cfg)
        print(f"  registered {cfg.registered_model_name} v{version} as @champion")
        _emit_ci(promotion=False, headline=f"Bootstrapped champion v{version} on {as_of.date()} (first deploy)")
        return

    backlog = (
        missed_cycles(last_recorded_cycle(cfg), as_of, args.max_catch_up)
        if args.catch_up
        else []
    )
    if backlog:
        print(
            f"  catching up {len(backlog)} missed cycle(s): "
            f"{backlog[0].date()} .. {backlog[-1].date()}"
        )

    promotions: list[pd.Timestamp] = []
    for at in [*backlog, as_of]:
        result = run_cycle(source, at, cfg)
        if result.promotion_decision == "promoted":
            promotions.append(at)
        line = (
            f"[{at.date()}] champion v{result.champion_version}  "
            f"psi={result.data_drift_psi:5.2f}  perf_ratio={result.perf_drift_ratio:4.2f}  "
            f"rmse={result.champion_rmse:6.2f}  -> {result.promotion_decision}"
        )
        if result.challenger_rmse is not None:
            line += (f"  (challenger {result.challenger_rmse:.2f} vs "
                     f"champion {result.champion_rmse_holdout:.2f} on holdout)")
        print(line)

    # Any promotion in the batch counts, not just the target week's: a catch-up
    # that promoted in week 2 of six changed which model serves, and a notice
    # keyed on the last cycle alone would not say so.
    promoted = bool(promotions)
    caught_up = f" (after catching up {len(backlog)} missed week(s))" if backlog else ""
    if promoted:
        when = ", ".join(str(d.date()) for d in promotions)
        headline = (
            f"Champion promoted on {when}{caught_up} — latest challenger "
            f"{result.challenger_rmse:.2f} against {result.champion_rmse_holdout:.2f} RMSE "
            f"on the held-out window"
            if result.challenger_rmse is not None
            else f"Champion promoted on {when}{caught_up}"
        )
    else:
        headline = (
            f"{as_of.date()}: no promotion ({result.promotion_decision}){caught_up} — "
            f"champion v{result.champion_version}, PSI {result.data_drift_psi:.2f}, "
            f"perf x{result.perf_drift_ratio:.2f}"
        )
    _emit_ci(promotion=promoted, headline=headline)


if __name__ == "__main__":
    main()
