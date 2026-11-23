"""Why was a forecast wrong? Split each week's error three ways.

    python scripts/split_error.py --gate     # fetch day-0 weather, report coverage

Every rule here is fixed by
docs/registrations/2026-11-23-why-a-forecast-was-wrong.md, committed as
``f9306b3`` before this file existed. This file implements that document and
decides nothing.

Only the gate is written so far. It fetches, for each of the fifteen cities of
the calendar test, the forecast archive's same-day weather over the same span as
the week-ahead features, and measures how many of those hours it covers. A city
enters the split only at 95% or more. The gate's other condition, that the
city's shipped replay reproduces its published premium, needs the replays and is
checked by the split itself before any week is split.

Writes outputs/split/gate.json: per city, the coverage, the day-0 file and its
hash, which the registration's addendum pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path[:0] = [str(Path(__file__).resolve().parent)]

import calendar_baseline as cal  # noqa: E402

from driftloop.data import OpenMeteoSource  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs" / "split"

# The registration's thresholds.
MIN_COVERAGE = 0.95
MIN_CITIES = 12


def day0_source(source: OpenMeteoSource) -> OpenMeteoSource:
    """The same place and span, with the same-day forecast in place of the week-ahead one."""
    return OpenMeteoSource(
        replace(source.config, forecast_lead_days=0, day0_from_forecast_archive=True),
        cache_dir=source.cache_dir,
    )


def coverage(week_ahead, same_day) -> float:
    """Share of the week-ahead hours that have a complete same-day row.

    Both frames are already complete rows: the source drops any hour missing a
    feature or the target. Measured over the whole data span, which holds every
    training window as well as every week, so it is at least as strict as
    measuring over the replayed weeks alone.
    """
    hours = set(week_ahead["timestamp"])
    if not hours:
        return 0.0
    return len(hours & set(same_day["timestamp"])) / len(hours)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def gate() -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, profile, source, _want in cal.units():
        ahead = source.timeline()
        same_day_source = day0_source(source)
        same_day = same_day_source.timeline()
        share = coverage(ahead, same_day)
        path = same_day_source._cache_path()
        rows.append({
            "group": group, "city": profile.label, "hours": int(len(ahead)),
            "covered": round(share, 4), "enters": share >= MIN_COVERAGE,
            "file": path.name, "sha256": _sha256(path),
        })
        print(f"  {profile.label:<14} {share:7.2%} of {len(ahead):>6} hours  "
              f"{'enters' if share >= MIN_COVERAGE else 'OUT'}  {path.name}", flush=True)

    entering = sum(r["enters"] for r in rows)
    result = {"min_coverage": MIN_COVERAGE, "min_cities": MIN_CITIES,
              "entering": entering, "void": entering < MIN_CITIES, "cities": rows}
    (OUTPUTS / "gate.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n{entering} of {len(rows)} cities enter; "
          f"{'VOID' if result['void'] else 'not void'} (needs {MIN_CITIES})")
    print(f"wrote {OUTPUTS / 'gate.json'}")
    return 1 if result["void"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate", action="store_true", help="fetch day-0 weather and report coverage")
    args = parser.parse_args()
    if args.gate:
        return gate()
    parser.error("only --gate is implemented so far")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
