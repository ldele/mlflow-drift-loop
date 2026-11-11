"""Test the second retrain alarm on cities the study has never used.

    python scripts/held_out_cities.py --check   # the gate on two known cities, no network
    python scripts/held_out_cities.py           # walk the candidates, run both arms

Every rule here is fixed by
docs/registrations/2026-11-10-second-alarm-on-held-out-cities.md, which was
written before any candidate's data was fetched: the twelve candidates and their
order, the span, how the replay windows are chosen, which cities are walked, and
what is compared. This file implements that document and decides nothing itself.

The walk runs only the alarm-off arm first. From it two numbers say whether a
city's first alarm goes deaf, which is the condition under which the second
alarm has anything to do. The walk stops at the second such city, the selection
is written to disk, and only then does any city get the alarm-on arm.

The new cities are a held-out test and not new profiles: the site, the six-city
tables and `config.PROFILES` are unchanged by this.

Writes outputs/held_out/candidates.json and outputs/held_out/alarm.csv, and
caches each fetched span under data_cache/ so the run repeats without a network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path[:0] = [str(Path(__file__).resolve().parent)]

import sweep_skill_floor as sweep  # noqa: E402

from driftloop import tracking  # noqa: E402
from driftloop.config import (  # noqa: E402
    PROFILES,
    SHIPPED,
    TARGET,
    TIMESTAMP,
    LoopConfig,
    OpenMeteoConfig,
    Profile,
    ReplayWindows,
)
from driftloop.data import OpenMeteoSource  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs" / "held_out"


@dataclass(frozen=True)
class Candidate:
    slug: str
    name: str
    country: str
    latitude: float
    longitude: float


#: In the registered order. One per country, none in a country the six use.
CANDIDATES: tuple[Candidate, ...] = (
    Candidate("tokyo", "Tokyo", "JP", 35.6762, 139.6503),
    Candidate("shanghai", "Shanghai", "CN", 31.2304, 121.4737),
    Candidate("sao_paulo", "São Paulo", "BR", -23.5505, -46.6333),
    Candidate("mexico_city", "Mexico City", "MX", 19.4326, -99.1332),
    Candidate("cairo", "Cairo", "EG", 30.0444, 31.2357),
    Candidate("dhaka", "Dhaka", "BD", 23.8103, 90.4125),
    Candidate("karachi", "Karachi", "PK", 24.8607, 67.0011),
    Candidate("buenos_aires", "Buenos Aires", "AR", -34.6037, -58.3816),
    Candidate("istanbul", "Istanbul", "TR", 41.0082, 28.9784),
    Candidate("lagos", "Lagos", "NG", 6.5244, 3.3792),
    Candidate("manila", "Manila", "PH", 14.5995, 120.9842),
    Candidate("kinshasa", "Kinshasa", "CD", -4.4419, 15.2663),
)

# The span Kraków and Delhi use, and the fewest clean hourly rows a candidate may
# return over it: 90% of the span.
SPAN_ORIGIN = pd.Timestamp("2025-05-01")
SPAN_HORIZON = pd.Timestamp("2026-07-15")
MIN_ROWS = 9_500

# The replay windows, by rule: train on the cleanest 75 days, then walk forward.
TRAIN_DAYS = 75
TRAIN_END_FIRST = pd.Timestamp("2025-07-15")
TRAIN_END_LAST = pd.Timestamp("2025-11-27")
GAP_DAYS = 15
LAST_RUN = pd.Timestamp("2026-07-10")

# The one setting under test, and what counts as a deaf first alarm.
FLOOR = -0.5
DEAF_SILENCE_WEEKS = 10
DEAF_ELIGIBLE_WEEKS = 3
CITIES_WANTED = 2
# Fewer changed weeks than this and there is no interval worth reading.
MIN_ACTED_WEEKS = 5

COLUMNS = [
    "city", "floor", "deaf", "runs", "retrains", "promotions", "fired_by_skill",
    "longest_silence", "median_rmse", "paired", "differing_weeks",
    "vs_off_acted", "vs_off_acted_lo", "vs_off_acted_hi", "vs_off_acted_real",
]


def replay_windows(timeline: pd.DataFrame) -> ReplayWindows:
    """The registered rule: train on the cleanest 75 days, start 15 days later.

    Among every 75-day window ending between `TRAIN_END_FIRST` and
    `TRAIN_END_LAST`, the one with the lowest mean PM2.5; the earliest wins a
    tie. Nobody picks a window after seeing what the loop does in it.
    """
    pm25 = timeline.set_index(TIMESTAMP)[TARGET].sort_index()
    length = pd.Timedelta(TRAIN_DAYS, unit="D")
    best_end, best_mean = None, float("inf")
    for end in pd.date_range(TRAIN_END_FIRST, TRAIN_END_LAST, freq="D"):
        window = pm25[(pm25.index >= end - length) & (pm25.index < end)]
        if window.empty:
            continue
        mean = float(window.mean())
        if mean < best_mean:  # strict, so the earliest of equals is kept
            best_end, best_mean = end, mean
    if best_end is None:
        raise ValueError("no PM2.5 in any candidate training window")
    return ReplayWindows(
        champion_train_start=best_end - length,
        champion_train_end=best_end,
        first_run=best_end + pd.Timedelta(GAP_DAYS, unit="D"),
        last_run=LAST_RUN,
    )


def deafness(off_arm: dict) -> dict:
    """Whether the first alarm goes deaf, read from the arm without the second.

    Two numbers: the longest run of weeks in which nothing fired, and how many
    quiet weeks had the model's skill below the floor. Both come from the
    alarm-off replay, so they say where the second alarm would have something to
    do and nothing about what it achieves.
    """
    quiet_below = sum(
        1 for fired, skill in zip(off_arm["_fired"], off_arm["_skill"], strict=True)
        if not fired and skill == skill and skill < FLOOR
    )
    silence = int(off_arm["longest_silence"])
    return {
        "longest_silence": silence,
        "quiet_weeks_below_floor": quiet_below,
        "deaf": silence >= DEAF_SILENCE_WEEKS and quiet_below >= DEAF_ELIGIBLE_WEEKS,
    }


def held_out_profile(candidate: Candidate, replay: ReplayWindows) -> Profile:
    """A profile like the six, built for this run and added to nothing."""
    return Profile(
        key=f"heldout_{candidate.slug}",
        label=candidate.name,
        loop=LoopConfig(
            experiment_name=f"drift-loop-heldout-{candidate.slug}",
            registered_model_name=f"pm25-ridge-heldout-{candidate.slug}",
            **SHIPPED,
        ),
        db_filename=f"mlflow_heldout_{candidate.slug}.db",
        meta_filename=f"run_meta_heldout_{candidate.slug}.json",
        location=OpenMeteoConfig(
            name=candidate.name, country=candidate.country,
            latitude=candidate.latitude, longitude=candidate.longitude,
            origin=SPAN_ORIGIN, horizon=SPAN_HORIZON,
        ),
        replay=replay,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check() -> int:
    """The gate on two cities whose answers are known. No candidate is touched."""
    published = {
        city["label"]: city["interval"]["when_it_acted"]["point"]
        for city in json.loads(
            (REPO_ROOT / "outputs" / "uncertainty.json").read_text(encoding="utf-8")
        )["cities"]
    }
    failures = 0
    for key, expect_deaf in (("openmeteo_la", True), ("openmeteo_santiago", False)):
        profile = PROFILES[key]
        source = OpenMeteoSource(profile.location)
        source.timeline()
        off = sweep.run_arm(profile, source, None)
        verdict = deafness(off)
        ok = verdict["deaf"] == expect_deaf
        failures += not ok
        print(f"  {profile.label:<12} silence {verdict['longest_silence']:>2}w, "
              f"{verdict['quiet_weeks_below_floor']:>2} quiet weeks below the floor -> "
              f"{'deaf' if verdict['deaf'] else 'not deaf'}  {'ok' if ok else 'WRONG'}")
        if expect_deaf:
            on = sweep.run_arm(profile, source, FLOOR)
            got, want = round(on["paired"], 1), round(published[profile.label], 1)
            ok = abs(got - want) <= 0.1
            failures += not ok
            print(f"  {profile.label:<12} the -0.5 arm reads {got:+.1f}%, published "
                  f"{want:+.1f}%  {'ok' if ok else 'WRONG'}")
    return failures


def walk() -> tuple[list[dict], list[tuple[Profile, OpenMeteoSource, dict]]]:
    """Take candidates in order, alarm off only, until two have a deaf first alarm."""
    record, walked, found = [], [], 0
    for candidate in CANDIDATES:
        entry: dict = {"city": candidate.name, "slug": candidate.slug}
        record.append(entry)
        if found >= CITIES_WANTED:
            entry["status"] = "not reached"
            continue
        location = OpenMeteoConfig(
            name=candidate.name, country=candidate.country, latitude=candidate.latitude,
            longitude=candidate.longitude, origin=SPAN_ORIGIN, horizon=SPAN_HORIZON,
        )
        source = OpenMeteoSource(location)
        try:
            timeline = source.timeline()
        except Exception as error:  # a candidate the source cannot serve is dropped, by rule
            entry.update(status="dropped", reason=f"fetch failed: {type(error).__name__}: {error}")
            print(f"  {candidate.name:<13} dropped: fetch failed", flush=True)
            continue
        entry.update(rows=len(timeline), sha256=sha256(source._cache_path())[:16],
                     file=source._cache_path().name)
        if len(timeline) < MIN_ROWS:
            entry.update(status="dropped", reason=f"{len(timeline)} rows, fewer than {MIN_ROWS}")
            print(f"  {candidate.name:<13} dropped: {len(timeline)} rows", flush=True)
            continue
        replay = replay_windows(timeline)
        profile = held_out_profile(candidate, replay)
        off = sweep.run_arm(profile, source, None)
        verdict = deafness(off)
        found += verdict["deaf"]
        entry.update(
            status="walked", **verdict,
            train_start=str(replay.champion_train_start.date()),
            train_end=str(replay.champion_train_end.date()),
            first_run=str(replay.first_run.date()), runs=off["runs"],
        )
        walked.append((profile, source, off))
        print(f"  {candidate.name:<13} trained to {entry['train_end']}, {off['runs']} runs, "
              f"silence {verdict['longest_silence']:>2}w, {verdict['quiet_weeks_below_floor']:>2} "
              f"quiet weeks below the floor -> {'DEAF' if verdict['deaf'] else 'not deaf'}",
              flush=True)
    return record, walked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="only run the gate on Los Angeles and Santiago; no network")
    args = parser.parse_args()

    original_root = tracking.REPO_ROOT
    with tempfile.TemporaryDirectory(prefix="driftloop-heldout-", ignore_cleanup_errors=True) as tmp:
        tracking.REPO_ROOT = Path(tmp)
        try:
            print("=== instrument check ===", flush=True)
            if check():
                print("the instrument check failed; nothing else was run")
                return 1
            if args.check:
                return 0

            print("\n=== the walk, alarm off ===", flush=True)
            record, walked = walk()
            OUTPUTS.mkdir(parents=True, exist_ok=True)
            # Written before any city gets the alarm-on arm, so the selection is
            # on disk before there is an outcome it could have been bent toward.
            (OUTPUTS / "candidates.json").write_text(
                json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8"
            )

            print("\n=== both arms ===", flush=True)
            rows = []
            for (profile, source, off), entry in zip(
                walked, [e for e in record if e["status"] == "walked"], strict=True
            ):
                on = sweep.run_arm(profile, source, FLOOR)
                pair = [off, on]
                sweep.compare_against_off(pair)
                for row in pair:
                    row["deaf"] = entry["deaf"]
                rows.extend(pair)
                acted = on.get("differing_weeks", 0)
                interval = (
                    f"{on['vs_off_acted']:+.1f}% [{on['vs_off_acted_lo']:+.1f}, "
                    f"{on['vs_off_acted_hi']:+.1f}]"
                    if acted >= MIN_ACTED_WEEKS and "vs_off_acted" in on else "no interval"
                )
                print(f"  {profile.label:<13} {'deaf' if entry['deaf'] else '    '}  silence "
                      f"{off['longest_silence']:>2}w -> {on['longest_silence']:>2}w, "
                      f"{acted:>2} changed weeks, {interval}", flush=True)
        finally:
            tracking.REPO_ROOT = original_root

    pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(OUTPUTS / "alarm.csv", index=False)
    print(f"\nwrote {OUTPUTS / 'candidates.json'} and {OUTPUTS / 'alarm.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
