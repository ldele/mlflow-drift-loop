"""Fail when the live loop has gone more than a week without logging a cycle.

The weekly Action runs this after it has persisted its store, so a job that
finishes green without leaving a recent cycle on record turns red instead, and
the failure step files it as an issue.

    python scripts/check_heartbeat.py [--warn-only]

``--warn-only`` reports the same finding and exits 0. The Action uses it before
the cycle runs: a stale store at that point is what the runner's catch-up is
about to repair, so it is worth saying but not worth stopping the repair for.

Exit status: 0 when fresh (or with ``--warn-only``), 1 when stale.
"""

from __future__ import annotations

import argparse
import os
import sys

import mlflow

from driftloop import heartbeat, tracking
from driftloop.config import PROFILES

PROFILE = PROFILES["scheduled"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warn-only", action="store_true", help="report staleness but exit 0")
    args = parser.parse_args()

    # Point at the store without tracking.setup(), which would create the
    # experiment: an empty store has to read as empty.
    mlflow.set_tracking_uri(tracking.tracking_uri(PROFILE.db_filename))
    beat = heartbeat.check(heartbeat.last_logged(PROFILE.loop.experiment_name))
    text = beat.message()
    print(text)

    if os.environ.get("GITHUB_ACTIONS") == "true" and beat.stale:
        level = "warning" if args.warn_only else "error"
        print(f"::{level} title=Drift loop: no recent cycle::{text}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"{'⚠️' if beat.stale else '•'} Heartbeat: {text}\n\n")

    return 1 if beat.stale and not args.warn_only else 0


if __name__ == "__main__":
    sys.exit(main())
