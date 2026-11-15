"""Does the loop beat retraining on a calendar?

    python scripts/calendar_baseline.py

Every rule here is fixed by
docs/registrations/2026-11-15-calendar-retraining-baseline.md, which is to be
committed before this script is run on any city: the fifteen cities and their
replays, the three arms, the statistic, how each city is read, the reading rule
across cities, the controls and the four predictions. This file implements that
document and decides nothing.

The three arms are the loop as it ships and two calendars built from settings it
already has, so `src/` is unchanged:

- ``shipped``: retrain when an alarm fires, ship if the exam is passed by 5%.
- ``weekly_exam``: ``recertify_days=1``, so a challenger is trained on every run
  and sits the usual exam.
- ``weekly_refit``: the same, plus ``promotion_margin=-inf``, so every challenger
  is shipped.

In order:

1. Every city's shipped arm, which must reproduce what is already published:
   the six against outputs/uncertainty.json to one decimal, the nine against
   outputs/held_out/alarm.csv to 0.01.
2. Both weekly arms in every city, which must do what they claim: a challenger
   on every run, and for the refit a promotion on every run.
3. Three paired comparisons per city, the reading rule across the fifteen, and
   the four predictions, each marked void if step 1 or 2 failed anywhere.

Writes outputs/calendar/arms.csv (one row per city and arm),
comparisons.csv (one row per city and comparison), series.json (the per-window
errors behind both) and scores.json (controls, verdicts and predictions).
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path[:0] = [str(Path(__file__).resolve().parent)]

import held_out_cities as held  # noqa: E402
import held_out_premium as prem  # noqa: E402
import sweep_skill_floor as sweep  # noqa: E402

from driftloop import stats, tracking  # noqa: E402
from driftloop.config import PROFILES, OpenMeteoConfig, Profile  # noqa: E402
from driftloop.data import OpenMeteoSource  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs" / "calendar"

# The arms, as overrides on each city's shipped LoopConfig. At a seven-day step a
# certificate is at least seven days old on every run, so a one-day cadence
# re-sits the exam every time. An infinitely negative margin passes every
# challenger: `chal < champ * (1 - margin)` is `chal < inf`.
SHIPPED_ARM = "shipped"
WEEKLY_EXAM = "weekly_exam"
WEEKLY_REFIT = "weekly_refit"
ARMS: dict[str, dict] = {
    SHIPPED_ARM: {},
    WEEKLY_EXAM: {"recertify_days": 1},
    WEEKLY_REFIT: {"recertify_days": 1, "promotion_margin": float("-inf")},
}

# (X, Y): X is read against Y, and positive means X forecast better.
COMPARISONS: tuple[tuple[str, str], ...] = (
    (WEEKLY_EXAM, SHIPPED_ARM),
    (WEEKLY_REFIT, SHIPPED_ARM),
    (WEEKLY_EXAM, WEEKLY_REFIT),
)

# Fewer differing windows than this and a comparison reads "not shown".
MIN_DIFFERING = 5
# The reading rule's threshold: this many cities clear in one direction.
CITIES_FOR_A_VERDICT = 3
# P4 fails when this many cities or more promote in half the weekly exam's runs.
P4_FAILS_AT = 4

ARM_COLUMNS = [
    "group", "city", "arm", "runs", "retrains", "promotions", "longest_silence",
    "median_rmse", "premium", "premium_lo", "premium_hi", "premium_acted",
]
COMPARISON_COLUMNS = [
    "group", "city", "x", "y", "weeks", "differing", "acted", "acted_lo", "acted_hi",
    "reading", "all_weeks", "all_weeks_lo", "all_weeks_hi",
]


# --------------------------------------------------------------------------- #
# The rules                                                                    #
# --------------------------------------------------------------------------- #


def reading(differing: int, lo: float, hi: float) -> str:
    """How one comparison in one city is read."""
    if differing < MIN_DIFFERING or math.isnan(lo) or math.isnan(hi):
        return "not shown"
    if lo > 0:
        return "better"
    if hi < 0:
        return "worse"
    return "not shown"


def verdict(readings: list[str]) -> str:
    """The reading rule across cities, for one comparison."""
    better = readings.count("better")
    worse = readings.count("worse")
    if better >= CITIES_FOR_A_VERDICT and worse == 0:
        return "better"
    if worse >= CITIES_FOR_A_VERDICT and better == 0:
        return "worse"
    if better >= 1 and worse >= 1 and better + worse >= 3:
        return "mixed"
    return "not detectable"


def two_city_rule(readings: list[str]) -> str:
    """The 2026-11-10 rule, applied to the six for comparison. Not scored."""
    better, worse = readings.count("better"), readings.count("worse")
    if better >= 2 and worse < 2:
        return "better"
    if worse >= 2 and better < 2:
        return "worse"
    if better >= 2 and worse >= 2:
        return "mixed"
    return "not detectable"


def compare(x: dict, y: dict) -> dict:
    """X against Y in one city, paired on the run date."""
    y_by_date = dict(zip(y["_as_of"], y["_rmse"]))
    pairs = [(a, y_by_date[d]) for a, d in zip(x["_rmse"], x["_as_of"]) if d in y_by_date]
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    nan = float("nan")
    row = {"weeks": len(pairs), "differing": 0, "acted": nan, "acted_lo": nan,
           "acted_hi": nan, "all_weeks": nan, "all_weeks_lo": nan, "all_weeks_hi": nan}
    if not pairs:
        row["reading"] = "not shown"
        return row
    every = stats.block_bootstrap((xs, ys), stats.pct_improvement_paired)
    row.update(all_weeks=every.point, all_weeks_lo=every.lo, all_weeks_hi=every.hi)
    mask = stats.differing(xs, ys)
    row["differing"] = int(mask.sum())
    if row["differing"] >= MIN_DIFFERING:
        acted = stats.block_bootstrap((xs[mask], ys[mask]), stats.pct_improvement_paired)
        row.update(acted=acted.point, acted_lo=acted.lo, acted_hi=acted.hi)
    row["reading"] = reading(row["differing"], row["acted_lo"], row["acted_hi"])
    return row


def mechanism_ok(arm: str, row: dict) -> bool:
    """Whether a weekly arm did what the registration says it does."""
    if arm == SHIPPED_ARM:
        return True
    trains_every_run = row["retrains"] == row["runs"]
    if arm == WEEKLY_REFIT:
        return trains_every_run and row["promotions"] == row["runs"]
    return trains_every_run


def arm_profile(profile: Profile, arm: str) -> Profile:
    """The city's profile with one arm's settings, under names of its own."""
    loop = replace(
        profile.loop,
        **ARMS[arm],
        registered_model_name=f"{profile.loop.registered_model_name}-cal-{arm}",
    )
    return replace(profile, key=f"{profile.key}_cal_{arm}", loop=loop)


def score(rows: list[dict], controls_ok: bool) -> dict:
    """The verdicts and the four predictions from the per-city comparisons."""
    void = not controls_ok
    verdicts = {}
    for x, y in COMPARISONS:
        found = [r for r in rows if r["kind"] == "comparison" and r["x"] == x and r["y"] == y]
        readings = [r["reading"] for r in found]
        six = [r["reading"] for r in found if r["group"] == "six"]
        verdicts[f"{x}_vs_{y}"] = {
            "verdict": verdict(readings), "better": readings.count("better"),
            "worse": readings.count("worse"), "cities": len(readings),
            "six_two_city_rule": two_city_rule(six),
        }

    exam = [r for r in rows if r["kind"] == "arm" and r["arm"] == WEEKLY_EXAM]
    half_or_more = sum(1 for r in exam if r["runs"] and r["promotions"] / r["runs"] >= 0.5)

    def held_if(ok: bool) -> str:
        return "void" if void else ("held" if ok else "failed")

    v = {k: d["verdict"] for k, d in verdicts.items()}
    predictions = {
        "P1": held_if(v[f"{WEEKLY_EXAM}_vs_{SHIPPED_ARM}"] != "better"),
        "P2": held_if(v[f"{WEEKLY_REFIT}_vs_{SHIPPED_ARM}"] != "worse"),
        "P3": held_if(v[f"{WEEKLY_EXAM}_vs_{WEEKLY_REFIT}"] != "worse"),
        "P4": held_if(half_or_more < P4_FAILS_AT),
    }
    return {"void": void, "verdicts": verdicts, "p4_cities_half_or_more": half_or_more,
            "predictions": predictions}


# --------------------------------------------------------------------------- #
# The units                                                                    #
# --------------------------------------------------------------------------- #


def units() -> list[tuple[str, Profile, OpenMeteoSource, dict]]:
    """The fifteen cities, each with what its shipped arm must reproduce."""
    published = prem._published()
    out = []
    for key in prem.SIX:
        profile = PROFILES[key]
        want = published[key]
        out.append(("six", profile, OpenMeteoSource(profile.location),
                    {"premium": want["point"], "premium_lo": want["lo"],
                     "premium_hi": want["hi"]}))

    stored = pd.read_csv(REPO_ROOT / "outputs" / "held_out" / "alarm.csv")
    stored = stored[stored["floor"] != "off"].set_index("city")["paired"]
    walked = [e for e in json.loads(
        (REPO_ROOT / "outputs" / "held_out" / "candidates.json").read_text(encoding="utf-8"))
        if e["status"] == "walked"]
    by_slug = {c.slug: c for c in held.CANDIDATES}
    for entry in walked:
        candidate = by_slug[entry["slug"]]
        location = OpenMeteoConfig(
            name=candidate.name, country=candidate.country, latitude=candidate.latitude,
            longitude=candidate.longitude, origin=held.SPAN_ORIGIN, horizon=held.SPAN_HORIZON,
        )
        source = OpenMeteoSource(location)
        replay = held.replay_windows(source.timeline())
        out.append(("nine", held.held_out_profile(candidate, replay), source,
                    {"paired": float(stored[candidate.name])}))
    return out


def reproduces(group: str, arm: dict, premium: dict, want: dict) -> bool:
    if group == "six":
        return all(abs(round(premium[k], 1) - round(want[k], 1)) <= 0.1 for k in want)
    return abs(round(arm["paired"], 2) - want["paired"]) <= 0.01


# --------------------------------------------------------------------------- #
# The run                                                                      #
# --------------------------------------------------------------------------- #


def run() -> tuple[list[dict], list[dict], bool]:
    rows: list[dict] = []
    series: list[dict] = []
    controls_ok = True
    for group, profile, source, want in units():
        source.timeline()  # warm the parquet cache once, not once per arm
        print(f"\n=== {profile.label} ({group}) ===", flush=True)
        arms = {}
        for arm in ARMS:
            result = sweep.run_arm(arm_profile(profile, arm), source, held.FLOOR)
            premium = prem.premium_of(result["_retro"])
            ok = mechanism_ok(arm, result)
            if arm == SHIPPED_ARM:
                ok = ok and reproduces(group, result, premium, want)
            controls_ok = controls_ok and ok
            arms[arm] = result
            rows.append({
                "kind": "arm", "group": group, "city": profile.label, "arm": arm,
                "runs": result["runs"], "retrains": result["retrains"],
                "promotions": result["promotions"],
                "longest_silence": result["longest_silence"],
                "median_rmse": result["median_rmse"], "premium": premium["premium"],
                "premium_lo": premium["premium_lo"], "premium_hi": premium["premium_hi"],
                "premium_acted": premium["acted"], "control_ok": ok,
            })
            series.append({"city": profile.label, "arm": arm,
                           "as_of": result["_as_of"], "rmse": result["_rmse"]})
            print(f"  {arm:<13} {result['retrains']:>3} trained, {result['promotions']:>3} "
                  f"promoted of {result['runs']} runs, premium {premium['premium']:+6.1f}% "
                  f"[{premium['premium_lo']:+6.1f}, {premium['premium_hi']:+6.1f}]  "
                  f"{'ok' if ok else 'CONTROL FAILED'}", flush=True)
        for x, y in COMPARISONS:
            c = compare(arms[x], arms[y])
            rows.append({"kind": "comparison", "group": group, "city": profile.label,
                         "x": x, "y": y, **c})
            print(f"  {x} against {y}: {c['acted']:+6.2f}% [{c['acted_lo']:+6.2f}, "
                  f"{c['acted_hi']:+6.2f}] over {c['differing']} differing weeks, "
                  f"{c['reading']}", flush=True)
    return rows, series, controls_ok


def main() -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    # Every backend is a throwaway, as in the sweeps. ignore_cleanup_errors because
    # MLflow keeps its sqlite handles open and Windows will not unlink them.
    original_root = tracking.REPO_ROOT
    with tempfile.TemporaryDirectory(prefix="driftloop-calendar-",
                                     ignore_cleanup_errors=True) as tmp:
        tracking.REPO_ROOT = Path(tmp)
        try:
            rows, series, controls_ok = run()
        finally:
            tracking.REPO_ROOT = original_root

    arms = [r for r in rows if r["kind"] == "arm"]
    comparisons = [r for r in rows if r["kind"] == "comparison"]
    pd.DataFrame(arms).reindex(columns=[*ARM_COLUMNS, "control_ok"]).to_csv(
        OUTPUTS / "arms.csv", index=False)
    pd.DataFrame(comparisons).reindex(columns=COMPARISON_COLUMNS).to_csv(
        OUTPUTS / "comparisons.csv", index=False)
    (OUTPUTS / "series.json").write_text(json.dumps(series, indent=1), encoding="utf-8")
    scores = {"controls_ok": controls_ok, **score(rows, controls_ok)}
    (OUTPUTS / "scores.json").write_text(json.dumps(scores, indent=1), encoding="utf-8")

    print("\n=== Verdicts ===")
    for name, v in scores["verdicts"].items():
        print(f"  {name}: {v['verdict']} ({v['better']} better, {v['worse']} worse "
              f"of {v['cities']}); the six by the two-city rule: {v['six_two_city_rule']}")
    print("  " + ", ".join(f"{k} {v}" for k, v in scores["predictions"].items()))
    print(f"\nwrote {OUTPUTS}")
    return 0 if controls_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
