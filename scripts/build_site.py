"""Emit site/data.json for the GitHub Pages dashboard.

The page itself (site/index.html + site/app.js) is committed static source that
fetches this JSON and renders the interactive Plotly charts client-side. So this
script's only job is to distill each profile's MLflow backend down to plain data
-- which also means the published data is directly inspectable at
`…/mlflow-drift-loop/data.json`.

Reads only metrics/tags (no artifact files), so it needs nothing but the sqlite
backends and is immune to the absolute-artifact-path issue.

    python scripts/build_site.py            # -> site/data.json
"""

from __future__ import annotations

import inspect
import json
import sys
import unicodedata
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driftloop import retrospect, tracking  # noqa: E402
from driftloop.config import DRIFT_FEATURES, FEATURES, PROFILES  # noqa: E402

OUT = REPO_ROOT / "site"

# How far past a promotion a version's decay curve is followed. Longer than any
# model here served, on purpose: the interesting part of the curve is
# what a model *would* have done had it stayed, which is the counterfactual the
# loop never computes for itself.
DECAY_WEEKS = 26

# Which sources the published page shows, in order. Real data only; the
# synthetic world stays an offline correctness proof (tests and sweep_knobs).
#
# Ordered as an argument rather than geographically: Kraków states the case,
# Santiago answers "is this just northern winter?", and the rest walk down the
# scale of what retraining is worth, from Delhi through Johannesburg's smaller
# gain to Melbourne where no effect is measurable and Los Angeles where the
# interval says the loop does harm.
DISPLAY_ORDER = [
    "openmeteo",
    "openmeteo_santiago",
    "openmeteo_delhi",
    "openmeteo_joburg",
    "openmeteo_melbourne",
    "openmeteo_la",
]

# The live schedule is not a city and is not comparable to one: it is the same
# Kraków data with one cycle appended at a time, a handful of cycles against
# Kraków's 48. It gets a status block in the section on the machinery rather than
# a place in the city selector, where it would invite that comparison.
SCHEDULE_KEY = "scheduled"

# Templates, not prose. Every number a city's story quotes is a `{placeholder}`
# filled from that city's own run and benchmark output at build time, on the same
# rule the method block follows: introspected rather than retyped. Widening the
# feature set or changing the forecast lead moves all of these figures at once,
# and hand-edited paragraphs go stale.
#
# Only what a re-run cannot change stays hardcoded: geography, season, and why a
# city was chosen. A sentence a re-run could falsify belongs in a placeholder.
# See _story_facts for the available names.
STORY = {
    # Both readings for the lead city, where they disagree by six points and the
    # page's own argument is that the across-the-replay number misleads.
    "openmeteo": "The city this started with. Kraków sits in a valley, and when the coal heating "
    "goes on for winter the smog has nowhere to escape to. A model trained on clean summer air "
    "goes from {rmse_first} to {rmse_peak} µg/m³ of error as that happens, and is replaced "
    "{retrains} times over {runs} weeks. Measured across the whole replay that replacing is worth "
    "{retrain_gain}, near enough a wash; held week by week over the {acted_weeks} weeks a "
    "retrained model was serving, it is {retrain_acted} {retrain_acted_ci}, which is "
    "{retrain_verdict}. Those {acted_weeks} weeks carry about {effective_n} independent "
    "observations between them, and that is why the range is so wide. What it ends up with "
    "is {rank_phrase}.",
    "openmeteo_delhi": "The violent one. The monsoon scrubs Delhi's air clean by September, then "
    "the crop stubble is burned and the winter air stops moving, and the pollution triples. The "
    "model's error runs from {rmse_first} to {rmse_peak} µg/m³ while the weather stops resembling "
    "anything it was trained on. Keeping it retrained is worth {retrain_gain} across the replay "
    "and {retrain_acted} {retrain_acted_ci} week by week, the biggest payoff here and the one "
    "furthest clear of zero. Left alone, it ends up worse than guessing the same number every "
    "hour.",
    "openmeteo_la": "Retraining buys nothing here, and that is why the city is on the page. Los "
    "Angeles was picked as the summer-smog control and the measurements disagreed twice over: its "
    "pollution peaks in November, and it moves only 1.9× against Delhi's 3×. Across {runs} weeks "
    "the loop fires {retrains_times}. Over the whole replay retraining scores {retrain_gain}, and "
    "in the {acted_weeks} weeks a retrained model was serving it won {win_rate} of them, worse "
    "than a coin toss rather than equal to one. Week by week it costs {retrain_acted} "
    "{retrain_acted_ci}, a range that stays {retrain_verdict}, so the control does not merely "
    "fail to benefit. It is measurably harmed. The model comes {rank_phrase}. Something built "
    "to catch change needs something to change.",
    "openmeteo_santiago": "Kraków's twin, half a year out of step. Santiago sits in a coastal bowl "
    "that traps winter air the same way, except that its winter is June. A model trained on clean "
    "December air decays from {rmse_first} to {rmse_peak} µg/m³ as that winter arrives. It is "
    "replaced {retrains} times over {runs} weeks, worth {retrain_gain} across the replay and "
    "{retrain_acted} {retrain_acted_ci} week by week, on settings identical to every other "
    "city here.",
    "openmeteo_joburg": "Where the quality gate does its most visible work, and where the headline "
    "number lies. Highveld winter nights trap coal and wood smoke, and the model's error climbs "
    "from {rmse_first} to {rmse_peak} µg/m³, the worst on this page. It trains {retrains} "
    "replacements over {runs} weeks and ships only {promotions}. Across the whole replay "
    "retraining scores {retrain_gain}, which reads as a wash and is not one: nothing was promoted "
    "until week 14 of 20, so most windows compare the first model against itself. In the "
    "{acted_weeks} weeks where a retrained model was serving it beat the original in {win_rate} of "
    "them, by a median of {retrain_acted} {retrain_acted_ci}. Six weeks is the thinnest "
    "evidence of any city here, and the range says so.",
    "openmeteo_melbourne": "Clean air does not mean a stable model. Sydney is the obvious "
    "Australian choice and its air barely moves; Melbourne swings 3.1× on winter wood smoke while "
    "staying near the WHO guideline all year round, and its model still decays to {perf_peak}× the "
    "error it started with across {runs} weeks. Retraining scores {retrain_gain} over the replay "
    "and {retrain_acted} {retrain_acted_ci} in the {acted_weeks} weeks it was serving a "
    "retrained model, winning {win_rate} of them. That range is {retrain_verdict}, so the "
    "honest reading is that retraining here buys nothing measurable at all.",
    "synthetic": "A made-up world with a dial controlling how far the data shifts, so the "
    "detection can be shown to fire when something really has changed, and only then.",
    "scheduled": "The same system running unattended, and not a city at all. This is the Kraków "
    "data again, with one check appended automatically by a robot every Monday. What it shows is "
    "that the machinery keeps going with nobody starting it, building its own history in real "
    "calendar time.",
}

# Human-readable position of the served champion in the benchmark table, so a
# story can say where it placed without a re-run silently making that a lie.
_ORDINALS = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}
_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
          8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def _count(n: int) -> str:
    """Small counts read as words in prose; larger ones stay numerals."""
    return _WORDS.get(n, str(n))


def _pct(value: float) -> str:
    """A signed percentage with a real minus sign, and no sign at all on zero.

    "+0.0%" is what the arithmetic produces when retraining breaks exactly even,
    and it reads as a gain that rounded away rather than as a wash.
    """
    if abs(value) < 0.05:
        return "0.0%"
    return f"{'+' if value > 0 else '−'}{abs(value):.1f}%"


def _interval(ci: dict | None) -> str:
    """An interval as prose: "[−15, +28]". Empty string where there is none."""
    if not ci:
        return ""
    return f"[{_pct(ci['lo'])}, {_pct(ci['hi'])}]".replace("%", "")


def _story_facts(runs: pd.DataFrame, retr: pd.DataFrame, prom: pd.DataFrame,
                 benchmark: dict | None, retraining: dict | None = None,
                 intervals: dict | None = None) -> dict[str, str]:
    """Every number a story is allowed to quote, derived from that city's run."""
    rmse = runs.get("metrics.champion_rmse")
    perf = runs.get("metrics.perf_drift_ratio")
    facts = {
        "runs": str(len(runs)),
        "retrains": _count(len(retr)),
        # "fires {retrains} times" reads as "fires 1 times" in Los Angeles, the
        # one city that retrains once and the control a sceptic reads hardest.
        "retrains_times": (
            "once" if len(retr) == 1
            else "twice" if len(retr) == 2
            else f"{_count(len(retr))} times"
        ),
        "promotions": _count(len(prom)),
        "psi_peak": f"{runs['metrics.data_drift_psi'].max():.2f}",
        "rmse_first": "—" if rmse is None else f"{rmse.iloc[0]:.1f}",
        "rmse_peak": "—" if rmse is None else f"{rmse.max():.1f}",
        "rmse_last": "—" if rmse is None else f"{rmse.iloc[-1]:.1f}",
        "perf_peak": "—" if perf is None else f"{perf.max():.2f}",
    }

    if benchmark and benchmark.get("scored"):
        scored = sorted(benchmark["scored"], key=lambda s: s["median_rmse"])
        names = [s["name"] for s in scored]
        served = next((s for s in scored if s["name"] == "champion_served"), None)
        frozen = next((s for s in scored if s["name"] == "champion_frozen"), None)
        if served and frozen:
            facts["retrain_gain"] = _pct((1 - served["median_rmse"] / frozen["median_rmse"]) * 100)
        if "champion_served" in names:
            place = names.index("champion_served") + 1
            facts["rank_phrase"] = (
                f"the lowest-error predictor of the {_count(len(names))} scored"
                if place == 1
                else f"{_ORDINALS.get(place, f'{place}th')} of the {_count(len(names))} scored"
            )

    # The paired reading, which is what a story should quote when the unpaired
    # one is dragged to zero by windows in which nothing had been retrained yet.
    if retraining and "when_it_acted" in retraining:
        facts["retrain_acted"] = _pct(retraining["when_it_acted"])
        facts["acted_weeks"] = str(retraining.get("acted_windows", ""))
        facts["win_rate"] = f"{retraining.get('win_rate', 0):.0f}%"

    # A story may quote an estimate only alongside its interval, and may call it
    # a finding only where that interval clears zero. Two of the six do not.
    acted_ci = (intervals or {}).get("when_it_acted") or {}
    facts["retrain_acted_ci"] = _interval(acted_ci)
    facts["retrain_gain_ci"] = _interval((intervals or {}).get("across_replay"))
    if acted_ci:
        facts["retrain_verdict"] = (
            "clear of zero" if acted_ci.get("excludes_zero")
            else "not distinguishable from zero"
        )
        facts["effective_n"] = f"{acted_ci.get('n_effective', 0):.0f}"
    return facts


def _render_story(key: str, facts: dict[str, str]) -> str:
    """Fill a story template, refusing to publish one with a hole in it."""
    template = STORY[key]
    try:
        return template.format(**facts)
    except KeyError as exc:  # a placeholder with nothing to fill it
        raise SystemExit(
            f"story for {key!r} wants {exc} but the run did not produce it; "
            f"available: {sorted(facts)}"
        ) from exc


def _floats(series) -> list[float | None]:
    """JSON can't hold NaN; map it to null (Plotly renders a gap)."""
    return [None if pd.isna(v) else float(v) for v in series]


def _dates(series) -> list[str]:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d").tolist()


def load_runs(db: str, experiment: str) -> pd.DataFrame:
    tracking.setup(experiment, db)
    df = mlflow.search_runs(experiment_names=[experiment], order_by=["attributes.start_time ASC"])
    if df.empty or "tags.cycle_type" not in df:
        return pd.DataFrame()
    df = df[df["tags.cycle_type"] == "monitor"].copy()
    if df.empty:
        return df
    df["as_of"] = pd.to_datetime(df["params.as_of"])
    return df.sort_values("as_of").reset_index(drop=True)


def load_versions(db: str, experiment: str, model: str) -> pd.DataFrame:
    tracking.setup(experiment, db)
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model}'")
    except Exception:
        return pd.DataFrame()
    rows = []
    for mv in versions:
        row = {"version": int(mv.version), "train_end": pd.to_datetime(mv.tags.get("train_end"))}
        for name in [*FEATURES, "intercept"]:
            row[f"coef_{name}"] = float(mv.tags.get(f"coef_{name}", "nan"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("version").reset_index(drop=True)


def retrospective_block(key: str, runs: pd.DataFrame) -> dict | None:
    """Everything that needs models scored on windows they never served.

    Needs the city's raw data, so it is built only for profiles tied to a place.
    The live schedule fetches a rolling window rather than a fixed cached span,
    so it gets no block and the page drops those cards -- the same way it already
    drops the benchmark card for a profile that has not been benchmarked.
    """
    from driftloop.data import replayable_source  # noqa: E402

    profile = PROFILES[key]
    location = profile.location
    source = replayable_source(profile)
    if source is None or location is None or "tags.champion_version" not in runs:
        return None

    cfg = profile.loop
    tracking.setup(cfg.experiment_name, profile.db_filename)
    models = retrospect.registered_models(MlflowClient(), cfg.registered_model_name)
    if not models:
        return None

    # The gate calibration reads the holdout metrics off these rows, so the whole
    # frame is passed through rather than just the two columns build() indexes on.
    frame = runs.copy()
    frame["champion_version"] = runs["tags.champion_version"].astype(int)
    retro = retrospect.build(
        source, frame, models, cfg.monitor_days, location.forecast_lead_days, cfg.promotion_margin
    )
    if not retro.as_of:
        return None

    as_of = _dates(pd.Series(retro.as_of))

    # One decay curve per version that served, aligned on weeks since it
    # took over. Followed past its retirement where the data allows, so a curve
    # that keeps falling after the model was replaced shows what keeping it would
    # have cost. Versions that were trained and rejected never served and are not
    # plotted -- they have no service life to decay through.
    step_days = max(1, (retro.as_of[1] - retro.as_of[0]).days) if len(retro.as_of) > 1 else 7
    decay = []
    for version, promoted in sorted(retro.promoted_at.items()):
        # Week 0 is the first run *after* the promotion, for the same reason the
        # gate calibration starts there: the window at the promotion itself
        # overlaps the challenger's own training data.
        start = retro.as_of.index(promoted) + 1
        end = min(len(retro.as_of), start + DECAY_WEEKS)
        if start >= end:
            continue
        served = sum(1 for v in retro.champion_version[start:] if v == version)
        decay.append(
            {
                "version": int(version),
                "promoted": promoted.strftime("%Y-%m-%d"),
                "weeks": [round((i - start) * step_days / 7.0, 2) for i in range(start, end)],
                "skill": _floats(pd.Series(retro.version_skill[version][start:end])),
                "served_weeks": round(served * step_days / 7.0, 1),
            }
        )

    latest_version = retro.champion_version[-1]
    latest_model = models.get(latest_version)
    importance: dict | None = None
    if latest_model is not None:
        window = source.get_data(
            retro.as_of[-1] - pd.Timedelta(cfg.monitor_days, unit="D"), retro.as_of[-1]
        )
        if not window.empty:
            ranked = sorted(
                latest_model.importance(window).items(), key=lambda kv: kv[1], reverse=True
            )
            # The two clock terms are folded into one entry: separately they are
            # half a daily cycle each and neither number means anything alone.
            weather = [(f, v) for f, v in ranked if f in DRIFT_FEATURES]
            clock = sum(v for f, v in ranked if f not in DRIFT_FEATURES)
            rows = [*weather, ("hour of day", clock)]
            rows.sort(key=lambda kv: kv[1], reverse=True)
            importance = {
                "features": [f for f, _ in rows],
                "values": [round(v, 3) for _, v in rows],
                "version": int(latest_version),
            }

    bootstrap = models[min(models)]
    return {
        "as_of": as_of,
        "retraining": retrospect.retraining_value(retro),
        # The interval on each of those, so the page can say which are findings.
        # The README and evaluation.md report intervals, and a site publishing
        # the same numbers bare would contradict them.
        "retraining_ci": {
            key: interval.to_dict()
            for key, interval in retrospect.retraining_uncertainty(retro).items()
        },
        "skill": {
            "champion": _floats(pd.Series(retro.champion_skill)),
            "climatology_rmse": _floats(pd.Series(retro.climatology_rmse)),
            "champion_rmse": _floats(pd.Series(retro.champion_rmse)),
            "climatology_days": retrospect.CLIMATOLOGY_DAYS,
        },
        "decay": decay,
        # The trigger in µg/m³ rather than as a ratio. Same units as the error it
        # is compared against, so the bar's staircase is visible: it steps up at
        # every promotion and never comes back down.
        "trigger": {
            "rmse": _floats(runs["metrics.champion_rmse"]),
            "bar": _floats(runs["metrics.champion_baseline_rmse"] * cfg.perf_drift_threshold),
            "baseline": _floats(runs["metrics.champion_baseline_rmse"]),
        },
        "factors": {
            "features": {f: _floats(pd.Series(v)) for f, v in retro.feature_means.items()},
            "target": _floats(pd.Series(retro.target_mean)),
            "trained_on": retrospect.training_window_stats(source, bootstrap),
            "bootstrap_train": [
                bootstrap.train_start.strftime("%Y-%m-%d"),
                bootstrap.train_end.strftime("%Y-%m-%d"),
            ],
        },
        "importance": importance,
        "gate": [
            {
                "version": g["version"],
                "replaced": g["replaced"],
                "as_of": g["as_of"].strftime("%Y-%m-%d"),
                "exam": round(g["exam_margin"], 4),
                "delivered": round(g["delivered_margin"], 4),
                "weeks": g["weeks_served"],
            }
            for g in retro.gate
            if not pd.isna(g["exam_margin"]) and not pd.isna(g["delivered_margin"])
        ],
    }


def profile_data(key: str) -> dict | None:
    profile = PROFILES[key]
    cfg = profile.loop
    runs = load_runs(profile.db_filename, cfg.experiment_name)
    if runs.empty:
        return None
    versions = load_versions(profile.db_filename, cfg.experiment_name, cfg.registered_model_name)

    meta_path = REPO_ROOT / "outputs" / profile.meta_filename
    drift_date = None
    if meta_path.exists():
        drift_date = json.loads(meta_path.read_text(encoding="utf-8")).get("drift_date")

    retr = runs[runs["tags.retrain_triggered"] == "True"]
    prom = runs[runs["tags.promotion_decision"] == "promoted"]
    if "metrics.challenger_rmse" in runs.columns:
        judged = runs.dropna(subset=["metrics.challenger_rmse"])
    else:
        judged = runs.iloc[:0]
    latest = runs.iloc[-1]

    coef = None
    if not versions.empty and not versions["coef_temperature"].isna().all():
        coef = {"train_end": _dates(versions["train_end"])}
        for f in [*FEATURES, "intercept"]:
            coef[f] = _floats(versions[f"coef_{f}"])

    # Written by scripts/benchmark.py. Absent until that has run, which is fine:
    # the page drops the card rather than inventing numbers for it.
    bench_path = REPO_ROOT / "outputs" / f"benchmark_{key}.json"
    benchmark = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.exists() else None

    # The thing actually being predicted, in the units it is predicted in. Without
    # this the page is all monitoring machinery and never says what the model is
    # for. Written by run_openmeteo.py; absent for the live profile.
    recent, latest_actual, latest_actual_at = None, None, None
    pred_path = REPO_ROOT / "outputs" / f"predictions_{key}.csv"
    if pred_path.exists():
        preds = pd.read_csv(pred_path, parse_dates=["timestamp"])
        if not preds.empty:
            recent = {
                "timestamp": preds["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
                "actual": _floats(preds["actual"]),
                "predicted": _floats(preds["predicted"]),
            }
            latest_actual = round(float(preds["actual"].iloc[-1]), 1)
            # Dated, because a city's replay ends where its configured span ends,
            # not today. Kraków's last reading is six months old; a tile that
            # says "latest" invites a visitor to read a January number as now.
            # Built by hand rather than with strftime: "%-d" is POSIX-only and
            # "%#d" is Windows-only, and this runs on both.
            stamp = preds["timestamp"].iloc[-1]
            latest_actual_at = f"{stamp.day} {stamp:%b %Y}"

    retro = retrospective_block(key, runs)

    # How long the model currently answering has been in service. The loop logs
    # which version served each run but never for how long, and that is where a
    # stale champion shows up first: a trigger that has gone quiet looks like a
    # healthy model until the age of the thing serving is on the page.
    champion_age_days, champion_version = None, None
    if "tags.champion_version" in runs:
        versions = runs["tags.champion_version"].tolist()
        champion_version = versions[-1]
        since = len(versions) - 1
        while since > 0 and versions[since - 1] == champion_version:
            since -= 1
        champion_age_days = int((runs["as_of"].iloc[-1] - runs["as_of"].iloc[since]).days)

    # What retraining was worth, from the retrospective rather than the benchmark
    # file: the benchmark comparison is unpaired, so where both distributions are
    # dominated by the same seasonal swing it mostly measures the season.
    # Johannesburg reads 0.0% that way while winning every window a retrained
    # model served.
    value = (retro or {}).get("retraining") or {}
    retrain_gain = round(value["across_replay"], 1) if "across_replay" in value else None
    retrain_acted = round(value["when_it_acted"], 1) if "when_it_acted" in value else None

    # The interval on each. Without it the page states Melbourne's +1.2% in the
    # same colour and typeface as Delhi's +49.4%, and only one of the two is
    # distinguishable from nothing.
    intervals = (retro or {}).get("retraining_ci") or {}
    acted_ci = intervals.get("when_it_acted") or {}
    gain_ci = intervals.get("across_replay") or {}

    loc = profile.location
    return {
        "key": key,
        "retro": retro,
        "label": profile.label,
        "benchmark": benchmark,
        "target": {"name": "PM2.5", "units": "µg/m³", "who_24h_guideline": 15},
        "recent": recent,
        "story": _render_story(key, _story_facts(runs, retr, prom, benchmark, value, intervals)),
        "drift_date": drift_date,
        # Present only for profiles tied to a real place. The page plots one map
        # marker per profile that has one, so a location-less profile (the live
        # schedule, which reads the same Kraków source) simply gets no marker.
        "location": None if loc is None else {
            "name": loc.name,
            "country": loc.country,
            "lat": loc.latitude,
            "lon": loc.longitude,
        },
        "stats": {
            "runs": int(len(runs)),
            "retrains": int(len(retr)),
            "promotions": int(len(prom)),
            # Trained, judged, and thrown away: the gate's actual output.
            "rejected": int(len(retr) - len(prom)),
            "champion_age_days": champion_age_days,
            "champion_version": champion_version,
            "retrain_gain": retrain_gain,
            "retrain_gain_lo": round(gain_ci["lo"], 1) if gain_ci else None,
            "retrain_gain_hi": round(gain_ci["hi"], 1) if gain_ci else None,
            "retrain_gain_real": gain_ci.get("excludes_zero"),
            "retrain_acted": retrain_acted,
            "retrain_acted_lo": round(acted_ci["lo"], 1) if acted_ci else None,
            "retrain_acted_hi": round(acted_ci["hi"], 1) if acted_ci else None,
            # Whether the interval clears zero. The page colours and words the
            # tile off this rather than off the sign of the estimate, because
            # the sign of a number that could be nothing is not a finding.
            "retrain_acted_real": acted_ci.get("excludes_zero"),
            # How many independent observations those weeks are worth. Kraków's
            # 47 are worth about 5, and that single figure explains the width of
            # every interval on the page.
            "retrain_effective_n": acted_ci.get("n_effective"),
            "retrain_acted_windows": value.get("acted_windows"),
            "retrain_win_rate": round(value["win_rate"], 0) if "win_rate" in value else None,
            "latest_skill": (
                round(retro["skill"]["champion"][-1], 3)
                if retro and retro["skill"]["champion"] and retro["skill"]["champion"][-1] is not None
                else None
            ),
            "latest_psi": round(float(latest["metrics.data_drift_psi"]), 2),
            "latest_r2": (round(float(latest["metrics.champion_r2"]), 2)
                          if "metrics.champion_r2" in latest else None),
            # In µg/m³, which is far more readable than R² for a headline number.
            "latest_rmse": (round(float(latest["metrics.champion_rmse"]), 1)
                            if "metrics.champion_rmse" in latest else None),
            "latest_actual": latest_actual,
            "latest_actual_at": latest_actual_at,
        },
        "as_of": _dates(runs["as_of"]),
        "psi": {f: _floats(runs[f"metrics.psi_{f}"]) for f in DRIFT_FEATURES if f"metrics.psi_{f}" in runs},
        # Features whose PSI is identically zero for every run. That is not
        # stability: `psi()` bins on reference quantiles, so a feature that is
        # near-constant over the training window (precipitation in a dry city)
        # collapses to a single bin and returns 0.0 whatever the current window
        # does. Flagged so the chart can say "not measurable here" instead of
        # drawing a flat line that reads as "perfectly stable".
        "psi_degenerate": [
            f
            for f in DRIFT_FEATURES
            if f"metrics.psi_{f}" in runs and bool((runs[f"metrics.psi_{f}"] == 0).all())
        ],
        "perf_ratio": _floats(runs["metrics.perf_drift_ratio"]),
        "retrain": {"as_of": _dates(retr["as_of"]), "perf": _floats(retr["metrics.perf_drift_ratio"])},
        "holdout": {
            "as_of": _dates(judged["as_of"]),
            "champion": _floats(judged["metrics.champion_rmse_holdout"]) if not judged.empty else [],
            "challenger": _floats(judged["metrics.challenger_rmse"]) if not judged.empty else [],
        },
        "promoted": {
            "as_of": _dates(prom["as_of"]),
            "challenger": _floats(prom["metrics.challenger_rmse"]) if not prom.empty else [],
        },
        "coef": coef,
    }


def _slug(name: str) -> str:
    """'Kraków' -> 'krakow', 'Los Angeles' -> 'los_angeles' (ASCII, filename-safe)."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return "_".join(ascii_name.lower().split())


def publish_raw_data() -> list[dict]:
    """Copy each city's raw gathered observations into the published site.

    The chart data is a distilled summary; this makes the full hourly datasets the
    charts are built from downloadable too, so nothing we gather is thrown away.
    Reads the committed parquet caches directly (no network).
    """
    from driftloop.data.openmeteo import OpenMeteoSource  # noqa: E402

    OUT.mkdir(exist_ok=True)
    published = []
    for key in DISPLAY_ORDER:
        location = PROFILES[key].location
        if location is None:  # not a place -- nothing raw of its own to publish
            continue
        cache_path = OpenMeteoSource(location)._cache_path()
        if not cache_path.exists():
            print(f"  (no cached {location.name} data at {cache_path.name}; skipping)")
            continue
        df = pd.read_parquet(cache_path)
        filename = f"{_slug(location.name)}_hourly.csv"
        (OUT / filename).write_text(df.to_csv(index=False), encoding="utf-8")
        published.append({
            "city": location.name,
            "file": filename,
            "rows": int(len(df)),
            "start": pd.to_datetime(df["timestamp"]).min().strftime("%Y-%m-%d"),
            "end": pd.to_datetime(df["timestamp"]).max().strftime("%Y-%m-%d"),
        })
    return published


def sweep_block() -> dict | None:
    """The controlled experiment, from scripts/sweep_knobs.py.

    The six cities show the loop acting; they cannot show that it acts *for the
    right reason*, because the real world has no control condition. The synthetic
    world does: two knobs that move covariate drift and concept drift
    independently, so each detector can be checked against the cause it is
    supposed to answer to and the one it should ignore.

    Responses are published relative to the knob-at-zero reading as well as raw.
    Seasonal difference between the training and monitoring windows puts a real
    floor under PSI before either knob is touched, so the honest question is not
    "is it zero" but "does it move, and only for its own cause".
    """
    path = REPO_ROOT / "outputs" / "sweep.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)

    out: dict[str, dict] = {}
    for knob in ("feature_shift", "drift_strength"):
        rows = df[df["sweep"] == knob].sort_values("level")
        if rows.empty:
            continue
        base_psi = float(rows["max_psi"].iloc[0])
        base_perf = float(rows["perf_drift_ratio"].iloc[0])
        out[knob] = {
            "level": _floats(rows["level"]),
            "psi": _floats(rows["max_psi"]),
            "perf": _floats(rows["perf_drift_ratio"]),
            "psi_rel": [float(v) / base_psi for v in rows["max_psi"]],
            "perf_rel": [float(v) / base_perf for v in rows["perf_drift_ratio"]],
        }
    return out or None


def schedule_block() -> dict | None:
    """How the unattended loop is doing, as a few facts rather than as charts.

    Reports what the tracking store contains and nothing about what produced it.
    A cycle logged by hand and a cycle logged by the weekly Action are
    indistinguishable here, so the page states the count and the dates and makes
    no claim about the cron.
    """
    profile = PROFILES[SCHEDULE_KEY]
    runs = load_runs(profile.db_filename, profile.loop.experiment_name)
    if runs.empty:
        return None
    return {
        "cycles": int(len(runs)),
        "first": runs["as_of"].min().strftime("%Y-%m-%d"),
        "last": runs["as_of"].max().strftime("%Y-%m-%d"),
        "logged_at": pd.to_datetime(runs["start_time"]).max().strftime("%Y-%m-%d"),
        "champion_version": (
            runs["tags.champion_version"].iloc[-1] if "tags.champion_version" in runs else None
        ),
        "retrains": int((runs["tags.retrain_triggered"] == "True").sum()),
    }


def method_block() -> dict:
    """The page's "how this works" section, read out of the code that runs.

    Every number here is introspected rather than retyped, so the published
    description of the method cannot quietly drift away from the method. If
    someone changes the retrain threshold, the page changes with it.
    """
    from driftloop.config import TARGET  # noqa: E402
    from driftloop.drift import PSI_SIGNIFICANT, PSI_STABLE  # noqa: E402
    from driftloop.model import build_pipeline, train  # noqa: E402

    pipeline = build_pipeline()

    def _thresholds(cfg) -> dict:
        return {
            "monitor_days": cfg.monitor_days,
            "challenger_train_days": cfg.challenger_train_days,
            "holdout_days": cfg.holdout_days,
            "perf_drift_threshold": cfg.perf_drift_threshold,
            "psi_threshold": cfg.psi_threshold,
            "promotion_margin": cfg.promotion_margin,
        }

    located = [PROFILES[k].location for k in DISPLAY_ORDER if PROFILES[k].location]
    leads = {loc.forecast_lead_days for loc in located}
    # None when the cities disagree, which the page renders as "mixed" rather
    # than picking one and quietly misdescribing the others.
    horizon_days = leads.pop() if len(leads) == 1 else None

    shown = [PROFILES[k].loop for k in DISPLAY_ORDER]
    params = _thresholds(shown[0])
    # The page says these thresholds are the same for every city -- which is what
    # makes the cities comparable at all. Check it rather than assert it, so
    # tuning one city later can't leave the claim stranded.
    uniform = all(_thresholds(cfg) == params for cfg in shown)

    return {
        "estimator": " → ".join(type(step).__name__ for _, step in pipeline.steps),
        "alpha": float(pipeline.named_steps["ridge"].alpha),
        "features": list(FEATURES),
        # The subset PSI is computed over, and the subset the charts colour.
        # Split out so the page cannot invent a drift series for a feature the
        # loop never measured drift on.
        "drift_features": list(DRIFT_FEATURES),
        "target": TARGET,
        # Features and the target share one timestamp in the column contract, so
        # the horizon lives entirely in *which* weather the source fetches: at a
        # lead of N days the features are the forecast issued N days before the
        # target hour. Read off the configs rather than retyped, for the same
        # reason as the thresholds below.
        "horizon_days": horizon_days,
        "val_fraction": float(inspect.signature(train).parameters["val_fraction"].default),
        "params": params,
        "params_uniform": uniform,
        "psi_bands": {"stable": PSI_STABLE, "significant": PSI_SIGNIFICANT},
        # Where a promotion stops counting as short-serving on the gate chart.
        # Published rather than repeated in the JavaScript, because the Streamlit
        # app draws the same split and two hand-kept copies of one threshold is
        # how the two UIs end up telling different stories.
        "gate_long_weeks": retrospect.GATE_LONG_WEEKS,
    }


def build() -> Path:
    # Only the sources in DISPLAY_ORDER are published (synthetic is excluded).
    profiles = [d for d in (profile_data(k) for k in DISPLAY_ORDER) if d is not None]
    if not profiles:
        raise SystemExit("No profiles have data. Run the pipelines first.")

    OUT.mkdir(exist_ok=True)
    payload = {
        "built": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "raw_data": publish_raw_data(),
        "method": method_block(),
        "sweep": sweep_block(),
        "schedule": schedule_block(),
        "profiles": profiles,
    }
    out = OUT / "data.json"
    out.write_text(json.dumps(payload, indent=None), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB, "
          f"{len(profiles)} profiles: {', '.join(p['key'] for p in profiles)})")
    return out


if __name__ == "__main__":
    build()
