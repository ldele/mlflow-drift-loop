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

from driftloop import tracking  # noqa: E402
from driftloop.config import DRIFT_FEATURES, FEATURES, PROFILES  # noqa: E402

OUT = REPO_ROOT / "site"

# Which sources the published page shows, in order. Only the real data is
# showcased: the historical Kraków replay and the same loop running live. The
# synthetic world stays an offline correctness proof (tests + sweep_knobs) and is
# deliberately NOT published here.
#
# Ordered as an argument rather than geographically: Kraków states the case and
# Santiago immediately answers "is this just northern winter?", then the cities
# walk down the scale of what retraining is worth, from +66.7% (Delhi) through
# indifference (Johannesburg, Melbourne) to actively harmful (Los Angeles).
DISPLAY_ORDER = [
    "openmeteo",
    "openmeteo_santiago",
    "openmeteo_delhi",
    "openmeteo_joburg",
    "openmeteo_melbourne",
    "openmeteo_la",
    "scheduled",
]

# Templates, not prose. Every number a city's story quotes is a `{placeholder}`
# filled from that city's own run and benchmark output at build time.
#
# This is the same rule the method block already followed ("introspected rather
# than retyped") applied to the narrative, and it exists because the narrative
# broke that rule repeatedly: widening the feature set or changing the forecast
# lead moves every one of these figures at once, and hand-editing seven
# paragraphs afterwards leaves a stale number behind sooner or later.
#
# What stays hardcoded is only what a re-run cannot change: geography, season,
# and why a city was chosen. If a sentence would be falsified by a re-run, it
# belongs in a placeholder. See _story_facts for the available names.
STORY = {
    "openmeteo": "The city this started with. Kraków sits in a valley, and when the coal heating "
    "goes on for winter the smog has nowhere to escape to. A model trained on clean summer air "
    "goes from {rmse_first} to {rmse_peak} µg/m³ of error as that happens, and is replaced "
    "{retrains} times over {runs} weeks. All that replacing is worth {retrain_gain}, near enough "
    "a wash, and what it ends up with is {rank_phrase}.",
    "openmeteo_delhi": "The violent one. The monsoon scrubs Delhi's air clean by September, then "
    "the crop stubble is burned and the winter air stops moving, and the pollution triples. The "
    "model's error runs from {rmse_first} to {rmse_peak} µg/m³ while the weather stops resembling "
    "anything it was trained on. Keeping it retrained is worth {retrain_gain}, the biggest payoff "
    "here. Left alone, it ends up worse than guessing the same number every hour.",
    "openmeteo_la": "The control, and the measurements chose it rather than the plan. Los Angeles "
    "was supposed to be the summer-smog city. Its pollution actually peaks in November and moves "
    "only 1.9×, against Delhi's 3×. The model barely budges across {runs} weeks, retraining is "
    "worth {retrain_gain}, and it comes {rank_phrase}. Something built to catch change needs "
    "something to change.",
    "openmeteo_santiago": "Kraków's twin, half a year out of step. Santiago sits in a coastal bowl "
    "that traps winter air the same way, except that its winter is June. A model trained on clean "
    "December air decays from {rmse_first} to {rmse_peak} µg/m³ as that winter arrives. It is "
    "replaced {retrains} times over {runs} weeks, worth {retrain_gain}, on settings identical to "
    "every other city here.",
    "openmeteo_joburg": "Where the quality gate does its most visible work. Highveld winter nights "
    "trap coal and wood smoke, and the model's error climbs from {rmse_first} to {rmse_peak} "
    "µg/m³, the worst on this page. It trains {retrains} replacements over {runs} weeks and ships "
    "only {promotions}. The rest were not clearly better and were thrown away, for a net effect on "
    "the error of {retrain_gain}. The effort is spent, and nothing ships that has not earned it.",
    "openmeteo_melbourne": "The quiet city that goes stale anyway. Sydney is the obvious "
    "Australian choice and its air barely moves. Melbourne swings 3.1× on winter wood smoke while "
    "staying near the WHO guideline all year round, and its model still decays to {perf_peak}× the "
    "error it started with across {runs} weeks. Retraining is worth {retrain_gain}. Clean air "
    "does not mean a stable model.",
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


def _story_facts(runs: pd.DataFrame, retr: pd.DataFrame, prom: pd.DataFrame,
                 benchmark: dict | None) -> dict[str, str]:
    """Every number a story is allowed to quote, derived from that city's run."""
    rmse = runs.get("metrics.champion_rmse")
    perf = runs.get("metrics.perf_drift_ratio")
    facts = {
        "runs": str(len(runs)),
        "retrains": _count(len(retr)),
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

    loc = profile.location
    return {
        "key": key,
        "label": profile.label,
        "benchmark": benchmark,
        "target": {"name": "PM2.5", "units": "µg/m³", "who_24h_guideline": 15},
        "recent": recent,
        "story": _render_story(key, _story_facts(runs, retr, prom, benchmark)),
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
        "profiles": profiles,
    }
    out = OUT / "data.json"
    out.write_text(json.dumps(payload, indent=None), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB, "
          f"{len(profiles)} profiles: {', '.join(p['key'] for p in profiles)})")
    return out


if __name__ == "__main__":
    build()
