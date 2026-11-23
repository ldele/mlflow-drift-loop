"""Why was a forecast wrong? Split each week's error three ways.

    python scripts/split_error.py --gate             # fetch day-0 weather, report coverage
    python scripts/split_error.py --split --jobs 4   # the split, its controls, the predictions
    python scripts/split_error.py --describe         # the readings beside them, from the tables

Every rule here is fixed by
docs/registrations/2026-11-23-why-a-forecast-was-wrong.md, committed as
``f9306b3`` before this file existed, and by its addenda. This file implements
that document and decides nothing.

Each week is scored four times, two models by two kinds of weather:

- A: the model in service, rebuilt from its registry tags, on the week-ahead
  forecast. What the loop measured.
- B: its twin, the same pipeline fitted on the same rows with day-0 weather, on
  day-0 weather.
- C, D: the same pipeline refitted through the week, on week-ahead (C) and day-0
  (D) weather: fitted on the 28 days with the week in the middle, each day of the
  week predicted by a model fitted on the other 27.

Squared errors summed over a city's hours give three parts that add up to A:
weather = ((A-B) + (C-D)) / 2, relation = ((A-C) + (B-D)) / 2, unexplained = D.

What ``--split`` does, in order:

1. The inputs: each city's day-0 file has to hash to what gate.json and the
   addendum pin, and cover the gate's 95%.
2. Each city's shipped replay, exactly as the calendar test ran it. It has to
   reproduce the city's published premium, the gate's second half. For the six,
   cell A has to equal the loop's logged error at every run.
3. The four cells for every week, the inert-weather control (the twin trained on
   week-ahead weather reproduces A), and the reported extras: the refit window at
   14 and 56 days, and the retraining reach.
4. The positive control: the simulated city with the concept dial at 2.0 against
   0.0, on the replay plan below.
5. The three parts per city and the four predictions, each void if a control
   failed or fewer than 12 cities entered.

Writes outputs/split/weeks.csv (one row per city and week), cities.csv (one row
per city), scores.json (the controls, the gate and the predictions) and
described.json (what the registration reports beside them and does not score).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from scipy.stats import rankdata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path[:0] = [str(Path(__file__).resolve().parent)]

import calendar_baseline as cal  # noqa: E402
import held_out_cities as held  # noqa: E402
import held_out_premium as prem  # noqa: E402
import sweep_skill_floor as sweep  # noqa: E402

from driftloop import retrospect, stats, tracking  # noqa: E402
from driftloop.config import (  # noqa: E402
    FEATURES, PROFILES, SHIPPED, TARGET, Profile, ReplayWindows, SyntheticConfig,
)
from driftloop.data import OpenMeteoSource, SyntheticSource  # noqa: E402
from driftloop.loop import fit_for  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs" / "split"

# The registration's thresholds and choices.
MIN_COVERAGE = 0.95
MIN_CITIES = 12
REFIT_DAYS = 28
SENSITIVITY_DAYS = (14, 56)
PERMUTATIONS = 10_000
P3_ALPHA = 0.05
P4_FAILS_AT = 0.5
PREMIUM_TOLERANCE = 0.1
PARTS = ("weather", "relation", "unexplained")

# Coefficient tags carry six decimals, so each slope and the intercept of a model
# rebuilt from them can be off by half the last digit. See ``tag_slack``.
TAG_STEP = 5e-7

# The positive control. The registration fixes the dials. The simulated city has
# no replay plan of its own, so this one takes scripts/sweep_knobs.py's training
# window and runs weekly from the first date the loop allows (training end plus
# the 14-day monitor window) to the last week inside the simulated year.
SYNTHETIC_DIALS = (0.0, 2.0)
SYNTHETIC_PLAN = ReplayWindows(
    champion_train_start=pd.Timestamp("2025-04-01"),
    champion_train_end=pd.Timestamp("2025-07-01"),
    first_run=pd.Timestamp("2025-07-15"),
    last_run=pd.Timestamp("2025-12-23"),
)

HOUR = pd.Timedelta(1, unit="h")
DAY = pd.Timedelta(1, unit="D")


# --------------------------------------------------------------------------- #
# The parts                                                                    #
# --------------------------------------------------------------------------- #


def parts(a, b, c, d):
    """Weather, relation and unexplained, from the four cells. They sum to ``a``.

    Each factor's effect is averaged over the two orders it can be taken in, the
    Shapley split of two factors (Shapley, 1953), so neither goes first.
    """
    return 0.5 * ((a - b) + (c - d)), 0.5 * ((a - c) + (b - d)), d


def largest_part(a: float, b: float, c: float, d: float) -> str:
    return PARTS[int(np.argmax(parts(a, b, c, d)))]


# --------------------------------------------------------------------------- #
# Frames and weeks                                                             #
# --------------------------------------------------------------------------- #


def day0_source(source: OpenMeteoSource) -> OpenMeteoSource:
    """The same place and span, with the same-day forecast in place of the week-ahead one."""
    return OpenMeteoSource(
        replace(source.config, forecast_lead_days=0, day0_from_forecast_archive=True),
        cache_dir=source.cache_dir,
    )


def frame_of(source) -> pd.DataFrame:
    """A source's whole span, sorted by time."""
    if hasattr(source, "timeline"):
        return source.timeline()
    cfg = source.config  # the simulated city has no cache, only its span
    return source.get_data(cfg.origin, cfg.horizon + HOUR).reset_index(drop=True)


def _slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Rows in ``[start, end)``. ``frame`` is sorted by timestamp."""
    stamps = frame["timestamp"].to_numpy()
    lo = np.searchsorted(stamps, start.to_datetime64(), side="left")
    hi = np.searchsorted(stamps, end.to_datetime64(), side="left")
    return frame.iloc[lo:hi]


def _sse(actual, predicted) -> float:
    return float(np.sum((np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)) ** 2))


def tag_slack(hours: pd.DataFrame, predicted) -> float:
    """How far rounding the tags can move a sum of squared errors over ``hours``.

    Each of the slopes and the intercept is off by at most ``TAG_STEP``, so an
    hour's prediction moves by at most ``b = TAG_STEP * (1 + sum |x|)``. Surface
    pressure near 1000 hPa makes that about 1e-3 µg/m³. With ``e`` the residual
    of the rebuilt model, the fitted model's squared error differs from it by at
    most ``2|e|b + 3b²`` an hour. A gap within this is rounding; past it, the two
    are different models.
    """
    b = TAG_STEP * (1.0 + np.abs(hours[FEATURES].to_numpy(dtype=float)).sum(axis=1))
    e = np.abs(hours[TARGET].to_numpy(dtype=float) - np.asarray(predicted, dtype=float))
    return float(np.sum(2.0 * e * b + 3.0 * b * b))


@dataclass(frozen=True)
class Week:
    """The seven days before a run, and the model that served them."""

    start: pd.Timestamp
    end: pd.Timestamp
    version: int
    retrained: bool


def weeks_of(retro, retrained: list[bool], step_days: int) -> list[Week]:
    """One week per run: ``[as_of - step, as_of)``, served by ``serving_version``.

    The model that served a run's monitor window is the one in service after the
    run before it, which is ``serving_version`` (the outgoing champion on a run
    that promotes). At a seven-day step the weeks tile the replay without overlap.
    """
    step = pd.Timedelta(step_days, unit="D")
    return [
        Week(pd.Timestamp(as_of) - step, pd.Timestamp(as_of), int(version), bool(fired))
        for as_of, version, fired in zip(retro.as_of, retro.serving_version, retrained, strict=True)
    ]


# --------------------------------------------------------------------------- #
# The four cells                                                               #
# --------------------------------------------------------------------------- #


def twin(model, frame: pd.DataFrame, cfg, fit=fit_for):
    """The same pipeline, fitted the loop's way on the same rows of ``frame``."""
    return fit(_slice(frame, model.train_start, model.train_end + HOUR), cfg)


def refit_window(start, end, days: int, span: tuple[pd.Timestamp, pd.Timestamp]):
    """``days`` days with the week in the middle, shifted to stay inside ``span``."""
    length = pd.Timedelta(days, unit="D")
    pad = (length - (end - start)) / 2
    lo, hi = start - pad, end + pad
    if lo < span[0]:
        lo, hi = span[0], span[0] + length
    if hi > span[1]:
        lo, hi = span[1] - length, span[1]
    return lo, hi


def cross_fitted_sse(frame, week: Week, days: int, span, cfg, fit=fit_for) -> float:
    """The refit through the week: each day predicted by a model fitted on the rest."""
    lo, hi = refit_window(week.start, week.end, days, span)
    window = _slice(frame, lo, hi)
    stamps = window["timestamp"]
    total = 0.0
    day = week.start
    while day < week.end:
        held_out = ((stamps >= day) & (stamps < day + DAY)).to_numpy()
        if held_out.any():
            model = fit(window.loc[~held_out], cfg)
            total += _sse(window.loc[held_out, TARGET],
                          model.pipeline.predict(window.loc[held_out, FEATURES]))
        day += DAY
    return total


def reach_sse(frame, week: Week, cfg, fit=fit_for) -> float:
    """What a refit at the run before the week gets: the loop's challenger window."""
    rows = _slice(frame, week.start - pd.Timedelta(cfg.challenger_train_days, unit="D"),
                  week.start - pd.Timedelta(cfg.holdout_days, unit="D"))
    if len(rows) < 50:
        return float("nan")
    hours = _slice(frame, week.start, week.end)
    return _sse(hours[TARGET], fit(rows, cfg).pipeline.predict(hours[FEATURES]))


def city_weeks(label: str, group: str, ahead: pd.DataFrame, same_day: pd.DataFrame,
               weeks: list[Week], models: dict, cfg, extras: bool = True) -> list[dict]:
    """Every cell of every week in one city. Top level so a process pool can run it."""
    common = set(ahead["timestamp"]) & set(same_day["timestamp"])
    ahead = ahead[ahead["timestamp"].isin(common)].reset_index(drop=True)
    same_day = same_day[same_day["timestamp"].isin(common)].reset_index(drop=True)
    span = (ahead["timestamp"].iloc[0], ahead["timestamp"].iloc[-1] + HOUR)

    versions = sorted({w.version for w in weeks})
    twins = {v: twin(models[v], same_day, cfg) for v in versions}
    # The inert control: the twin trained on week-ahead weather is the champion
    # refitted, so it has to score what the tags score.
    inert = {v: twin(models[v], ahead, cfg) for v in versions}

    rows = []
    for week in weeks:
        hours_ahead = _slice(ahead, week.start, week.end)
        hours_same = _slice(same_day, week.start, week.end)
        actual = hours_ahead[TARGET]
        served = models[week.version].predict(hours_ahead)
        row = {
            "group": group, "city": label,
            "week_start": week.start.isoformat(), "week_end": week.end.isoformat(),
            "version": week.version, "retrained": week.retrained, "hours": int(len(hours_ahead)),
            "A": _sse(actual, served),
            "slack": tag_slack(hours_ahead, served),
            "B": _sse(hours_same[TARGET], twins[week.version].pipeline.predict(hours_same[FEATURES])),
            "B_inert": _sse(actual, inert[week.version].pipeline.predict(hours_ahead[FEATURES])),
            "C": cross_fitted_sse(ahead, week, REFIT_DAYS, span, cfg),
            "D": cross_fitted_sse(same_day, week, REFIT_DAYS, span, cfg),
        }
        if extras:
            for days in SENSITIVITY_DAYS:
                row[f"C{days}"] = cross_fitted_sse(ahead, week, days, span, cfg)
                row[f"D{days}"] = cross_fitted_sse(same_day, week, days, span, cfg)
            row["reach"] = reach_sse(ahead, week, cfg)
        row["largest"] = largest_part(row["A"], row["B"], row["C"], row["D"])
        rows.append(row)
    return rows


def _city_weeks_job(args: tuple) -> list[dict]:
    return city_weeks(*args)


# --------------------------------------------------------------------------- #
# Replays and controls                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class Replay:
    retro: object
    runs: pd.DataFrame
    models: dict
    cfg: object


def replay(profile: Profile, source) -> Replay:
    """The shipped replay as the calendar test ran it, with its runs and models.

    ``sweep.run_arm`` leaves the tracking URI on the backend it wrote, and these
    are the names it registered under, so the registry is read before the next
    replay moves the URI.
    """
    result = sweep.run_arm(profile, source, held.FLOOR)
    slug = sweep._slug(held.FLOOR)
    runs = mlflow.search_runs(experiment_names=[f"skill-floor-{slug}"],
                              order_by=["attributes.start_time ASC"])
    runs = runs[runs["tags.cycle_type"] == "monitor"].copy()
    runs["as_of"] = pd.to_datetime(runs["params.as_of"])
    runs = runs.sort_values("as_of").reset_index(drop=True)
    models = retrospect.registered_models(
        MlflowClient(), f"{profile.loop.registered_model_name}-floor-{slug}")
    cfg = replace(profile.loop, skill_floor=held.FLOOR)
    return Replay(result["_retro"], runs, models, cfg)


def reproduction_error(rep: Replay, frame: pd.DataFrame) -> float:
    """Cell A's model against the loop's logged error, over every run.

    The largest gap in squared error over a monitor window, as a multiple of what
    rounding the tags can explain there (``tag_slack``). Up to 1 is a match.
    """
    worst = 0.0
    monitor = pd.Timedelta(rep.cfg.monitor_days, unit="D")
    for i, as_of in enumerate(rep.retro.as_of):
        window = _slice(frame, pd.Timestamp(as_of) - monitor, pd.Timestamp(as_of))
        predicted = rep.models[int(rep.retro.serving_version[i])].predict(window)
        logged = float(rep.runs["metrics.champion_rmse"].iloc[i])
        gap = abs(_sse(window[TARGET], predicted) - len(window) * logged ** 2)
        worst = max(worst, gap / tag_slack(window, predicted))
    return worst


def inert_error(rows: list[dict]) -> float:
    """The week-ahead twin against the tags: the largest gap as a multiple of the slack.

    Up to 1 is a match, so the twin is the champion refitted.
    """
    return max((abs(r["B_inert"] - r["A"]) / r["slack"] for r in rows if r["slack"] > 0),
               default=0.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


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


# --------------------------------------------------------------------------- #
# Summaries and predictions                                                    #
# --------------------------------------------------------------------------- #


def _share(i: int):
    def statistic(a, b, c, d):
        return parts(a.sum(), b.sum(), c.sum(), d.sum())[i] / a.sum()
    return statistic


def summarise(rows: list[dict]) -> dict:
    """One city's three parts as shares of its served error, with intervals and extras."""
    def col(key: str) -> np.ndarray:
        return np.array([r.get(key, np.nan) for r in rows], dtype=float)

    a, b, c, d = col("A"), col("B"), col("C"), col("D")
    total = a.sum()
    split = parts(total, b.sum(), c.sum(), d.sum())
    shares = {name: float(split[i] / total) for i, name in enumerate(PARTS)}
    intervals = {}
    for i, name in enumerate(PARTS):
        iv = stats.block_bootstrap([a, b, c, d], _share(i))
        intervals[name] = {"lo": iv.lo, "hi": iv.hi}
    out = {
        "weeks": len(rows),
        "mse_served": float(total / sum(r["hours"] for r in rows)),
        "shares": shares,
        "intervals": intervals,
        "largest": max(PARTS, key=shares.get),
        "smallest": min(PARTS, key=shares.get),
        "interaction": float(((a - b) - (c - d)).sum() / total),
        "weather_first": {"weather": float((a - b).sum() / total),
                          "relation": float((b - d).sum() / total)},
        "relation_first": {"relation": float((a - c).sum() / total),
                           "weather": float((c - d).sum() / total)},
        "inert_error": inert_error(rows),
    }
    for days in SENSITIVITY_DAYS:
        cd, dd = col(f"C{days}"), col(f"D{days}")
        if np.isfinite(cd).all() and np.isfinite(dd).all():
            p = parts(total, b.sum(), cd.sum(), dd.sum())
            out[f"refit_{days}"] = {name: float(p[i] / total) for i, name in enumerate(PARTS)}
    reach = col("reach")
    ok = np.isfinite(reach)
    if ok.any():
        relation = parts(a[ok].sum(), b[ok].sum(), c[ok].sum(), d[ok].sum())[1]
        out["reach"] = float((a[ok].sum() - reach[ok].sum()) / relation) if relation else float("nan")
        out["reach_weeks"] = int(ok.sum())
    return out


def spearman_permutation(x, y, permutations: int = PERMUTATIONS,
                         seed: int = stats.DEFAULT_SEED) -> tuple[float, float]:
    """Spearman's rho and its one-sided permutation p-value, upper tail.

    p is (hits + 1) / (permutations + 1), so it is never zero (Phipson and Smyth,
    2010). Where either side is constant there is no ranking to correlate: rho is
    undefined and p is 1.
    """
    rx, ry = rankdata(x), rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan"), 1.0
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    hits = sum(float(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= rho - 1e-12
               for _ in range(permutations))
    return rho, (hits + 1) / (permutations + 1)


def two_thirds(n: int) -> int:
    return math.ceil(2 * n / 3)


def score(cities: list[dict], controls_ok: bool) -> dict:
    """The four predictions over the cities that entered, as registered."""
    entering = [c for c in cities if c["enters"]]
    n = len(entering)
    need = two_thirds(n)
    void = n < MIN_CITIES or not controls_ok

    p1 = sum(c["largest"] == "unexplained" for c in entering)
    p2 = sum(c["smallest"] == "relation" for c in entering)
    rho, p_value = (spearman_permutation([c["shares"]["relation"] for c in entering],
                                         [c["premium"] for c in entering])
                    if n >= 3 else (float("nan"), float("nan")))

    per_city = [[part for part in c["retrain_weeks"]] for c in entering]
    pooled = [part for weeks in per_city for part in weeks]
    k = sum(part == "relation" for part in pooled)
    share = k / len(pooled) if pooled else float("nan")
    rng = np.random.default_rng(stats.DEFAULT_SEED)
    draws = []
    for _ in range(stats.DEFAULT_RESAMPLES):
        picked = rng.integers(0, n, n) if n else []
        hits = sum(sum(part == "relation" for part in per_city[i]) for i in picked)
        total = sum(len(per_city[i]) for i in picked)
        if total:
            draws.append(hits / total)
    cluster = (np.percentile(draws, [2.5, 97.5]).tolist() if draws
               else [float("nan"), float("nan")])

    def verdict(holds: bool) -> str:
        return "void" if void else ("held" if holds else "failed")

    def wilson_of(k: int, trials: int) -> list[float]:
        return list(stats.wilson_interval(k, trials)) if trials else [float("nan")] * 2

    return {
        "entering": n, "two_thirds": need, "void": void,
        "P1": {"unexplained_largest": p1, "of": n, "wilson": wilson_of(p1, n),
               "verdict": verdict(p1 >= need)},
        "P2": {"relation_smallest": p2, "of": n, "wilson": wilson_of(p2, n),
               "verdict": verdict(p2 >= need)},
        "P3": {"rho": rho, "p": p_value, "permutations": PERMUTATIONS,
               "verdict": verdict(p_value < P3_ALPHA)},
        "P4": {"retrain_weeks": len(pooled), "relation_largest": k, "share": share,
               "wilson": wilson_of(k, len(pooled)), "city_bootstrap": cluster,
               "verdict": verdict(share < P4_FAILS_AT)},
    }


# --------------------------------------------------------------------------- #
# Reported, not scored                                                         #
# --------------------------------------------------------------------------- #


def _rank_correlation(x, y) -> float:
    """Spearman's rho alone, nan where either side is constant."""
    rx, ry = rankdata(x), rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def describe(weeks: pd.DataFrame, cities: pd.DataFrame) -> dict:
    """What is read beside the predictions and decides nothing, from the saved tables.

    Over the cities that entered:

    - the mean share of each part with a 95% t interval, which the registration's
      simulation found narrow at fifteen units, so it is a description;
    - the retraining reach across cities;
    - how often each part is the largest and the smallest at each refit window;
    - the weeks the loop did not retrain, read as P4 reads the weeks it did;
    - P3's rank correlation within the six and within the nine;
    - the range of both control gaps, as multiples of the tag slack.

    Reads only ``weeks.csv`` and ``cities.csv``, so ``--describe`` repeats it
    without a replay.
    """
    from scipy.stats import t as student

    entering = cities[cities["enters"].astype(bool)]
    n = len(entering)
    out: dict = {"cities": n, "means": {}, "windows": {}}
    for part in PARTS:
        values = entering[part].to_numpy(dtype=float)
        mean = float(values.mean())
        half = (float(student.ppf(0.975, n - 1) * values.std(ddof=1) / math.sqrt(n))
                if n > 1 else float("nan"))
        out["means"][part] = {"point": mean, "lo": mean - half, "hi": mean + half,
                              "median": float(np.median(values))}

    reach = entering["reach"].to_numpy(dtype=float)
    out["reach"] = {"mean": float(reach.mean()), "median": float(np.median(reach)),
                    "min": float(reach.min()), "max": float(reach.max())}

    for days, suffix in ((REFIT_DAYS, ""), *((d, f"_{d}") for d in SENSITIVITY_DAYS)):
        shares = entering[[f"{part}{suffix}" for part in PARTS]].to_numpy(dtype=float)
        largest = [PARTS[i] for i in shares.argmax(axis=1)]
        smallest = [PARTS[i] for i in shares.argmin(axis=1)]
        out["windows"][str(days)] = {
            "largest": {part: largest.count(part) for part in PARTS},
            "smallest": {part: smallest.count(part) for part in PARTS},
            "means": {part: float(shares[:, i].mean()) for i, part in enumerate(PARTS)},
        }

    real = weeks[weeks["city"].isin(entering["city"])]
    retrained = real["retrained"].astype(str) == "True"
    for name, picked in (("retrain_weeks", real[retrained]), ("other_weeks", real[~retrained])):
        k = int((picked["largest"] == "relation").sum())
        out[name] = {"weeks": len(picked), "relation_largest": k,
                     "share": k / len(picked) if len(picked) else float("nan")}

    out["groups"] = {}
    for group in ("six", "nine"):
        rows = entering[entering["group"] == group]
        out["groups"][group] = {"cities": len(rows),
                                "rho": _rank_correlation(rows["relation"], rows["premium"])}

    for control in ("reproduction_error", "inert_error"):
        values = entering[control].to_numpy(dtype=float)
        out[control] = {"min": float(values.min()), "max": float(values.max())}
    return out


def write_described() -> dict:
    """``described.json`` from the tables ``--split`` wrote."""
    described = describe(pd.read_csv(OUTPUTS / "weeks.csv"), pd.read_csv(OUTPUTS / "cities.csv"))
    (OUTPUTS / "described.json").write_text(json.dumps(described, indent=1), encoding="utf-8")
    return described


# --------------------------------------------------------------------------- #
# The positive control                                                         #
# --------------------------------------------------------------------------- #


def synthetic_profile(dial: float, plan: ReplayWindows = SYNTHETIC_PLAN) -> Profile:
    """The simulated city on the shipped loop, under names of its own per dial."""
    base = PROFILES["synthetic"]
    tag = f"{dial:g}".replace(".", "p")
    loop = replace(base.loop, **SHIPPED,
                   registered_model_name=f"{base.loop.registered_model_name}-split-{tag}")
    return replace(base, key=f"synthetic_split_{tag}", label=f"Synthetic, concept dial {dial:g}",
                   loop=loop, replay=plan)


def synthetic_weeks(dial: float, plan: ReplayWindows = SYNTHETIC_PLAN) -> tuple[list[dict], float]:
    """Every week of the simulated city at one dial, and cell A's reproduction gap.

    The simulated features carry no forecast, so its day-0 weather is the same
    frame: B is the twin on the same rows and the weather part is zero by the
    inert check.
    """
    source = SyntheticSource(SyntheticConfig(drift_strength=dial, feature_shift=0.0))
    profile = synthetic_profile(dial, plan)
    rep = replay(profile, source)
    frame = frame_of(source)
    fired = [str(v) == "True" for v in rep.runs["tags.retrain_triggered"]]
    weeks = weeks_of(rep.retro, fired, plan.step_days)
    rows = city_weeks(profile.label, "synthetic", frame, frame, weeks, rep.models, rep.cfg,
                      extras=False)
    return rows, reproduction_error(rep, frame)


def positive_control(low: list[dict], high: list[dict]) -> dict:
    """The relation share at the high dial minus the low, paired by week."""
    def cols(rows):
        return [np.array([r[k] for r in rows], dtype=float) for k in "ABCD"]

    relation = _share(1)

    def difference(*c):
        return relation(*c[4:]) - relation(*c[:4])

    iv = stats.block_bootstrap(cols(low) + cols(high), difference)
    low_s, high_s = summarise(low), summarise(high)
    # No forecast in the simulated city: day-0 weather is the same frame, so D is C
    # exactly and B differs from A by tag rounding only. The weather part is zero.
    weather_zero = all(r["C"] == r["D"] and abs(r["A"] - r["B"]) <= r["slack"] for r in low + high)
    return {
        "relation_low": low_s["shares"]["relation"], "relation_high": high_s["shares"]["relation"],
        "difference": iv.point, "lo": iv.lo, "hi": iv.hi,
        "weather_low": low_s["shares"]["weather"], "weather_high": high_s["shares"]["weather"],
        "passed": bool(iv.lo > 0 and weather_zero),
    }


# --------------------------------------------------------------------------- #
# The run                                                                      #
# --------------------------------------------------------------------------- #


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


def _premiums() -> dict[str, float]:
    rows = pd.read_csv(REPO_ROOT / "outputs" / "held_out" / "premium.csv")
    return dict(zip(rows["city"], rows["premium"].astype(float), strict=True))


def split(jobs: int) -> int:
    pinned = {c["city"]: c for c in json.loads(
        (OUTPUTS / "gate.json").read_text(encoding="utf-8"))["cities"]}
    premiums = _premiums()

    prepared, cities = [], []
    original_root = tracking.REPO_ROOT
    with tempfile.TemporaryDirectory(prefix="driftloop-split-", ignore_cleanup_errors=True) as tmp:
        tracking.REPO_ROOT = Path(tmp)
        try:
            for group, profile, source, _want in cal.units():
                print(f"\n=== {profile.label} ({group}) ===", flush=True)
                ahead = source.timeline()
                same_source = day0_source(source)
                same_day = same_source.timeline()
                pin = pinned[profile.label]
                hash_ok = _sha256(same_source._cache_path()) == pin["sha256"]
                share = coverage(ahead, same_day)

                rep = replay(cal.arm_profile(profile, cal.SHIPPED_ARM), source)
                got = prem.premium_of(rep.retro)["premium"]
                premium_ok = abs(round(got, 1) - round(premiums[profile.label], 1)) <= PREMIUM_TOLERANCE
                repro = reproduction_error(rep, ahead)
                fired = [str(v) == "True" for v in rep.runs["tags.retrain_triggered"]]
                weeks = weeks_of(rep.retro, fired, profile.replay.step_days)
                cities.append({
                    "group": group, "city": profile.label, "premium": premiums[profile.label],
                    "replay_premium": got, "premium_ok": premium_ok, "hash_ok": hash_ok,
                    "coverage": share, "reproduction_error": repro,
                    "reproduction_ok": group != "six" or repro <= 1.0,
                    "enters": share >= MIN_COVERAGE and premium_ok and hash_ok,
                })
                prepared.append((profile.label, group, ahead, same_day, weeks, rep.models, rep.cfg))
                print(f"  premium {got:+.1f}% (published {premiums[profile.label]:+.1f}%), "
                      f"A against logged {repro:.2f} of the tag slack, {len(weeks)} weeks, "
                      f"hash {'ok' if hash_ok else 'CHANGED'}", flush=True)

            print("\n=== the simulated city ===", flush=True)
            synthetic = {dial: synthetic_weeks(dial) for dial in SYNTHETIC_DIALS}
        finally:
            tracking.REPO_ROOT = original_root

    print(f"\nscoring every week of {len(prepared)} cities with {jobs} process(es)...", flush=True)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_city_weeks_job, prepared))
    else:
        results = [city_weeks(*args) for args in prepared]

    week_rows = [row for rows in results for row in rows]
    for city, rows in zip(cities, results, strict=True):
        city.update(summarise(rows))
        city["retrain_weeks"] = [r["largest"] for r in rows if r["retrained"]]
        city["inert_ok"] = city["inert_error"] <= 1.0

    low_rows, low_repro = synthetic[SYNTHETIC_DIALS[0]]
    high_rows, high_repro = synthetic[SYNTHETIC_DIALS[1]]
    positive = positive_control(low_rows, high_rows)
    positive["reproduction_error"] = max(low_repro, high_repro)
    week_rows += low_rows + high_rows

    controls = {
        "reproduction": all(c["reproduction_ok"] for c in cities)
        and positive["reproduction_error"] <= 1.0,
        "inert": all(c["inert_ok"] for c in cities),
        "positive": positive["passed"],
    }
    scores = {"controls": controls, "positive_control": positive,
              **score(cities, all(controls.values()))}

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(week_rows).to_csv(OUTPUTS / "weeks.csv", index=False)
    flat = []
    for c in cities:
        row = {k: v for k, v in c.items() if not isinstance(v, (dict, list))}
        for name in PARTS:
            row[name] = c["shares"][name]
            row[f"{name}_lo"] = c["intervals"][name]["lo"]
            row[f"{name}_hi"] = c["intervals"][name]["hi"]
        for days in SENSITIVITY_DAYS:
            for name in PARTS:
                row[f"{name}_{days}"] = c.get(f"refit_{days}", {}).get(name, float("nan"))
        row["retrain_weeks"] = len(c["retrain_weeks"])
        row["retrain_weeks_relation"] = sum(p == "relation" for p in c["retrain_weeks"])
        flat.append(row)
    pd.DataFrame(flat).to_csv(OUTPUTS / "cities.csv", index=False)
    (OUTPUTS / "scores.json").write_text(json.dumps(scores, indent=1, default=float),
                                         encoding="utf-8")
    write_described()

    print("\n=== Controls ===")
    print("  " + ", ".join(f"{k} {'ok' if v else 'FAILED'}" for k, v in controls.items()))
    print(f"  simulated relation share {positive['relation_low']:.1%} at dial "
          f"{SYNTHETIC_DIALS[0]:g}, {positive['relation_high']:.1%} at {SYNTHETIC_DIALS[1]:g}, "
          f"difference [{positive['lo']:+.1%}, {positive['hi']:+.1%}]")
    print(f"\n=== Predictions, {scores['entering']} cities entered ===")
    for key in ("P1", "P2", "P3", "P4"):
        print(f"  {key}: {scores[key]['verdict']}")
    print(f"\nwrote {OUTPUTS}")
    return 1 if scores["void"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate", action="store_true", help="fetch day-0 weather and report coverage")
    parser.add_argument("--split", action="store_true", help="the split, its controls, the predictions")
    parser.add_argument("--describe", action="store_true",
                        help="the readings reported beside the predictions, from the tables")
    parser.add_argument("--jobs", type=int, default=1, help="processes for scoring the weeks")
    args = parser.parse_args()
    if args.gate:
        return gate()
    if args.split:
        return split(args.jobs)
    if args.describe:
        print(json.dumps(write_described(), indent=1))
        print(f"wrote {OUTPUTS / 'described.json'}")
        return 0
    parser.error("pass --gate, --split or --describe")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
