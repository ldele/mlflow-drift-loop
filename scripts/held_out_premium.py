"""What retraining is worth on the nine held-out cities, and whether the season says where.

    python scripts/held_out_premium.py

Every rule here is fixed by
docs/registrations/2026-11-11-premium-on-held-out-cities.md, committed as
`b69c755` before any interval or swing below was computed: the nine cities and
their replays, the statistic and how each city is read, the swing and the year it
is measured on, the threshold and the six cities it comes from, the controls and
the two predictions. This file implements that document and decides nothing.

In order, each step stopping the run if it fails:

1. Controls: Delhi and Los Angeles, replayed through the same path, must
   reproduce their published week-by-week premium and interval to one decimal.
2. The nine replays of the loop as it ships, each of which must reproduce the
   point premium already stored in outputs/held_out/alarm.csv to 0.01.
3. The swing of all fifteen cities over 2024-05-01 to 2025-04-30, a year that
   ends before any of them trains or is scored. Fetched once and cached under
   data_cache/, so the run repeats without a network.
4. The threshold from the six, the prediction for each of the nine, and the score.

Writes outputs/held_out/premium.csv (one row per city) and premium.json (the
threshold, the controls and the two scored predictions).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import pandas as pd
from scipy import stats as scipy_stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path[:0] = [str(Path(__file__).resolve().parent)]

import held_out_cities as held  # noqa: E402
import sweep_skill_floor as sweep  # noqa: E402

from driftloop import retrospect, tracking  # noqa: E402
from driftloop.config import PROFILES, OpenMeteoConfig  # noqa: E402
from driftloop.data import OpenMeteoSource  # noqa: E402
from driftloop.data import openmeteo  # noqa: E402
from driftloop.figures import _city_slug  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs" / "held_out"

# The year the swing is measured on: before any city's first model trains.
SWING_START = pd.Timestamp("2024-05-01")
SWING_END = pd.Timestamp("2025-04-30")
# A month counts if at least half its hours have a value; the swing needs 10 of 12.
MONTH_COVERAGE = 0.5
MIN_MONTHS = 10

# Fewer windows than this served by a retrained model, and no interval is read.
MIN_ACTED = held.MIN_ACTED_WEEKS
# P1 fails at this many correct or fewer, of the nine.
P1_FAILS_AT = 6

# The six, in the order of the README's table. The controls are two of them.
SIX = ("openmeteo_delhi", "openmeteo_santiago", "openmeteo", "openmeteo_joburg",
       "openmeteo_melbourne", "openmeteo_la")
CONTROLS = {"openmeteo_delhi": "pays", "openmeteo_la": "not shown"}

COLUMNS = [
    "group", "city", "slug", "swing", "swing_low", "swing_high", "swing_months",
    "predicted", "weeks", "acted", "n_eff", "premium", "premium_lo", "premium_hi",
    "outcome", "correct", "stored_paired",
]


# --------------------------------------------------------------------------- #
# The outcome in one city                                                      #
# --------------------------------------------------------------------------- #


def outcome(acted: int, lo: float, hi: float) -> str:
    """The registered reading of one city's week-by-week premium."""
    if acted < MIN_ACTED:
        return "not shown"
    if lo > 0:
        return "pays"
    if hi < 0:
        return "harms"
    return "not shown"


def premium_of(retro) -> dict:
    """The six-city table's week-by-week premium, and its interval, from one replay."""
    interval = retrospect.retraining_uncertainty(retro).get("when_it_acted")
    weeks = len(retro.as_of)
    if interval is None or interval.n == 0:
        return {"weeks": weeks, "acted": 0, "n_eff": float("nan"),
                "premium": float("nan"), "premium_lo": float("nan"), "premium_hi": float("nan")}
    return {"weeks": weeks, "acted": interval.n, "n_eff": interval.n_effective,
            "premium": interval.point, "premium_lo": interval.lo, "premium_hi": interval.hi}


# --------------------------------------------------------------------------- #
# The swing                                                                    #
# --------------------------------------------------------------------------- #


def _swing_cache(latitude: float, longitude: float) -> Path:
    stem = f"pm25_{latitude}_{longitude}_{SWING_START.date()}_{SWING_END.date()}".replace(".", "p")
    return openmeteo.CACHE_DIR / f"{stem}.parquet"


def swing_hours(latitude: float, longitude: float, timezone: str) -> pd.Series:
    """Hourly PM2.5 over the swing year, from the cache or the air-quality endpoint."""
    path = _swing_cache(latitude, longitude)
    if not path.exists():
        hourly = openmeteo._get_json(openmeteo.AIR_QUALITY_URL, {
            "latitude": latitude, "longitude": longitude,
            "start_date": SWING_START.strftime("%Y-%m-%d"),
            "end_date": SWING_END.strftime("%Y-%m-%d"),
            "timezone": timezone, "hourly": "pm2_5",
        })["hourly"]
        frame = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"]),
                              "pm25": pd.to_numeric(pd.Series(hourly["pm2_5"]), errors="coerce")})
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    frame = pd.read_parquet(path)
    return frame.set_index("timestamp")["pm25"].sort_index()


def swing(pm25: pd.Series) -> dict:
    """Highest calendar-month mean over the lowest, over the months with enough hours."""
    months = pm25.groupby(pm25.index.to_period("M"))
    means, counted = {}, 0
    for month, values in months:
        if values.notna().sum() >= MONTH_COVERAGE * month.days_in_month * 24:
            means[str(month)] = float(values.mean())
            counted += 1
    if counted < MIN_MONTHS:
        return {"swing": float("nan"), "swing_low": float("nan"),
                "swing_high": float("nan"), "swing_months": counted}
    low, high = min(means.values()), max(means.values())
    return {"swing": high / low, "swing_low": low, "swing_high": high, "swing_months": counted}


# --------------------------------------------------------------------------- #
# The score                                                                    #
# --------------------------------------------------------------------------- #


def threshold(six: list[dict]) -> tuple[float, str]:
    """The smallest swing among the six cities where the premium clears zero."""
    paying = [row for row in six if row["outcome"] == "pays"]
    lowest = min(paying, key=lambda row: row["swing"])
    return lowest["swing"], lowest["city"]


def chance_of_at_least(correct: int, units: int, observed_pays: int, predicted_pays: int) -> float:
    """The chance that the same number of "pays" labels, put on the cities at random,
    sorts at least ``correct`` of them right. Exact, by counting.

    With ``tp`` of the predicted cities truly paying, the sort gets ``tp`` right among
    them and ``units - predicted - (observed - tp)`` right among the rest.
    """
    total = math.comb(units, predicted_pays)
    hits = 0
    for tp in range(0, min(observed_pays, predicted_pays) + 1):
        right = 2 * tp + units - predicted_pays - observed_pays
        if right >= correct:
            hits += (math.comb(observed_pays, tp)
                     * math.comb(units - observed_pays, predicted_pays - tp))
    return hits / total


def score(nine: list[dict]) -> dict:
    """P1 (the swing sorts the nine) and P2 (retraining harms none)."""
    correct = sum(row["correct"] for row in nine)
    observed = sum(row["outcome"] == "pays" for row in nine)
    predicted = sum(row["predicted"] == "pays" for row in nine)
    harms = [row["city"] for row in nine if row["outcome"] == "harms"]
    points = [(row["swing"], row["premium"]) for row in nine if row["premium"] == row["premium"]]
    rho = scipy_stats.spearmanr([p[0] for p in points], [p[1] for p in points]).statistic
    return {
        "p1": {"correct": correct, "of": len(nine), "holds": correct > P1_FAILS_AT,
               "observed_pays": observed, "predicted_pays": predicted,
               "chance": chance_of_at_least(correct, len(nine), observed, predicted)},
        "p2": {"harms": harms, "holds": not harms},
        "spearman": float(rho),
    }


# --------------------------------------------------------------------------- #
# The run                                                                      #
# --------------------------------------------------------------------------- #


def _published() -> dict[str, dict]:
    data = json.loads((REPO_ROOT / "outputs" / "uncertainty.json").read_text(encoding="utf-8"))
    return {city["key"]: city["interval"]["when_it_acted"] for city in data["cities"]}


def run_controls(published: dict[str, dict]) -> list[dict]:
    """Delhi and Los Angeles through the same path; any mismatch stops the run."""
    out = []
    for key, expected in CONTROLS.items():
        profile = PROFILES[key]
        source = OpenMeteoSource(profile.location)
        arm = sweep.run_arm(profile, source, held.FLOOR)
        got = premium_of(arm["_retro"])
        want = published[key]
        same = all(abs(round(got[g], 1) - round(want[w], 1)) <= 0.1
                   for g, w in (("premium", "point"), ("premium_lo", "lo"), ("premium_hi", "hi")))
        reading = outcome(got["acted"], got["premium_lo"], got["premium_hi"])
        ok = same and reading == expected
        print(f"  {profile.label:<12} {got['premium']:+.1f}% [{got['premium_lo']:+.1f}, "
              f"{got['premium_hi']:+.1f}], published {want['point']:+.1f}% [{want['lo']:+.1f}, "
              f"{want['hi']:+.1f}], reads {reading}  {'ok' if ok else 'WRONG'}", flush=True)
        out.append({"city": profile.label, "ok": ok, "reading": reading,
                    **{k: round(v, 2) for k, v in got.items() if k != "weeks"}})
    return out


def run_nine() -> list[dict]:
    """Each walked city's shipped replay, checked against the stored point premium."""
    stored = pd.read_csv(OUTPUTS / "alarm.csv")
    stored = stored[stored["floor"] != "off"].set_index("city")["paired"]
    walked = [e for e in json.loads((OUTPUTS / "candidates.json").read_text(encoding="utf-8"))
              if e["status"] == "walked"]
    by_slug = {c.slug: c for c in held.CANDIDATES}
    rows = []
    for entry in walked:
        candidate = by_slug[entry["slug"]]
        location = OpenMeteoConfig(
            name=candidate.name, country=candidate.country, latitude=candidate.latitude,
            longitude=candidate.longitude, origin=held.SPAN_ORIGIN, horizon=held.SPAN_HORIZON,
        )
        source = OpenMeteoSource(location)
        replay = held.replay_windows(source.timeline())
        arm = sweep.run_arm(held.held_out_profile(candidate, replay), source, held.FLOOR)
        got = premium_of(arm["_retro"])
        want = float(stored[candidate.name])
        reproduces = abs(round(arm["paired"], 2) - want) <= 0.01
        print(f"  {candidate.name:<13} paired {arm['paired']:+.2f}, stored {want:+.2f}  "
              f"{'ok' if reproduces else 'WRONG'}", flush=True)
        rows.append({"group": "held out", "city": candidate.name, "slug": candidate.slug,
                     "stored_paired": want, "_reproduces": reproduces,
                     "_coords": (candidate.latitude, candidate.longitude, location.timezone),
                     **got})
    return rows


def six_rows(published: dict[str, dict]) -> list[dict]:
    """The six, from their published intervals: the threshold and the in-sample read."""
    data = json.loads((REPO_ROOT / "outputs" / "uncertainty.json").read_text(encoding="utf-8"))
    weeks = {city["key"]: city["weeks"] for city in data["cities"]}
    rows = []
    for key in SIX:
        profile, block = PROFILES[key], published[key]
        loc = profile.location
        rows.append({
            "group": "six", "city": profile.label, "slug": _city_slug(profile.label),
            "weeks": weeks[key], "acted": block["n"], "n_eff": block["n_effective"],
            "premium": block["point"], "premium_lo": block["lo"], "premium_hi": block["hi"],
            "_coords": (loc.latitude, loc.longitude, loc.timezone),
        })
    return rows


def add_swing(rows: list[dict]) -> None:
    for row in rows:
        lat, lon, tz = row["_coords"]
        row.update(swing(swing_hours(lat, lon, tz)))
        print(f"  {row['city']:<13} swing {row['swing']:.2f}x "
              f"({row['swing_low']:.1f} to {row['swing_high']:.1f} µg/m³, "
              f"{row['swing_months']} months)", flush=True)


def markdown(nine: list[dict], cut: float) -> str:
    out = ["| | swing, 2024-25 | predicted | weeks served by a retrained model | "
           "week by week, against the first model | read |",
           "|---|---|---|---|---|---|"]
    for row in sorted(nine, key=lambda r: -r["swing"]):
        cell = (f"{row['premium']:+.1f}% [{row['premium_lo']:+.1f}, {row['premium_hi']:+.1f}]"
                if row["acted"] >= MIN_ACTED else f"{row['premium']:+.1f}%, no interval")
        out.append(f"| {row['city']} | {row['swing']:.1f}× | {row['predicted']} | {row['acted']} | "
                   f"{cell} | {row['outcome']}{'' if row['correct'] else ' (missed)'} |")
    out.append(f"\nthreshold {cut:.2f}×")
    return "\n".join(out)


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    published = _published()

    original_root = tracking.REPO_ROOT
    with tempfile.TemporaryDirectory(prefix="driftloop-premium-", ignore_cleanup_errors=True) as tmp:
        tracking.REPO_ROOT = Path(tmp)
        try:
            print("=== controls ===", flush=True)
            controls = run_controls(published)
            if not all(c["ok"] for c in controls):
                print("a control failed: P1 and P2 are void, and nothing else was run")
                return 1
            print("\n=== the nine replays ===", flush=True)
            nine = run_nine()
        finally:
            tracking.REPO_ROOT = original_root
    if not all(row["_reproduces"] for row in nine):
        print("a city did not reproduce its stored premium: P1 and P2 are void")
        return 1
    for row in nine:
        row["outcome"] = outcome(row["acted"], row["premium_lo"], row["premium_hi"])

    print("\n=== the swing, 2024-05-01 to 2025-04-30 ===", flush=True)
    six = six_rows(published)
    for row in six:
        row["outcome"] = outcome(row["acted"], row["premium_lo"], row["premium_hi"])
    add_swing(six + nine)
    void = [r["city"] for r in nine + [s for s in six if s["outcome"] == "pays"]
            if r["swing"] != r["swing"]]

    result: dict = {"registration": "docs/registrations/2026-11-11-premium-on-held-out-cities.md",
                    "controls": controls, "void": void}
    cut, cut_city = threshold(six) if not void else (float("nan"), "")
    for row in six + nine:
        row["predicted"] = "pays" if row["swing"] >= cut else "not"
        row["correct"] = (row["predicted"] == "pays") == (row["outcome"] == "pays")
    result.update(threshold=cut, threshold_city=cut_city, **(score(nine) if not void else {}))
    result["six_in_sample_correct"] = sum(row["correct"] for row in six)

    frame = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in six + nine])
    frame.reindex(columns=COLUMNS).round(4).to_csv(OUTPUTS / "premium.csv", index=False)
    (OUTPUTS / "premium.json").write_text(json.dumps(result, indent=1, ensure_ascii=False),
                                          encoding="utf-8")

    print("\n" + markdown(nine, cut))
    if void:
        print(f"\nP1 void: no swing for {', '.join(void)}")
    else:
        p1, p2 = result["p1"], result["p2"]
        print(f"\nP1 {'held' if p1['holds'] else 'FAILED'}: {p1['correct']} of {p1['of']} sorted "
              f"right (chance of as many or more at random {p1['chance']:.3f})")
        print(f"P2 {'held' if p2['holds'] else 'FAILED'}: harms in {p2['harms'] or 'no city'}")
        print(f"rank correlation, swing against point premium: {result['spearman']:+.2f}")
    print(f"six, in sample: {result['six_in_sample_correct']} of 6 sorted right at {cut:.2f}x")
    print(f"\nwrote {OUTPUTS / 'premium.csv'} and {OUTPUTS / 'premium.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
