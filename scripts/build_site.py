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
from driftloop.config import FEATURES, PROFILES  # noqa: E402

OUT = REPO_ROOT / "site"

# Which sources the published page shows, in order. Only the real data is
# showcased: the historical Kraków replay and the same loop running live. The
# synthetic world stays an offline correctness proof (tests + sweep_knobs) and is
# deliberately NOT published here.
#
# Ordered as an argument rather than geographically: Kraków states the case and
# Santiago immediately answers "is this just northern winter?", then the cities
# walk down the scale of what retraining is worth, from +47.3% (Delhi) through
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

STORY = {
    "openmeteo": "The basin this started with. A champion trained on clean summer air decays from "
    "5.2 to 53.3 µg/m³ of error once the heating season fills the valley with winter smog, and the "
    "loop retrains 8 times across 23 runs. Retraining is worth only +4.4% here, because a week-old "
    "weather forecast is noisy enough to blunt what a fresh model can learn from it.",
    "openmeteo_delhi": "The violent case. Monsoon rain scrubs Delhi's air to a September minimum, "
    "then crop-residue burning and winter inversions triple PM2.5. Feature drift peaks at PSI 9.37 "
    "and the champion's error runs from 20.1 to 71.8 µg/m³. Retraining is worth +47.3%, the "
    "largest payoff on this page, and the never-retrained champion ends up worse than a constant.",
    "openmeteo_la": "The control, chosen by the measurements after the plan said otherwise. Los "
    "Angeles was meant to be the summer-smog city; its PM2.5 actually peaks in November and swings "
    "only 1.9× against Delhi's 3×. The loop retrains twice across 37 runs and promotes once, and "
    "retraining costs it 11.6%. A drift loop needs drift.",
    "openmeteo_santiago": "Kraków's twin, half a year out of phase. Santiago sits in a coastal "
    "basin that traps winter inversions the same way, so a champion trained on clean December air "
    "decays from 8.0 to 60.8 µg/m³ as June arrives. Identical thresholds fire the same 8 retrains "
    "here as in Kraków, in the opposite season, and this time they are worth +30.6%.",
    "openmeteo_joburg": "Drift the loop can see and cannot fix. Highveld winter traps coal and "
    "wood smoke under a nightly inversion, and the champion's error climbs from 14.2 to 89.6 "
    "µg/m³, the worst here. Eight retrains across 20 runs buy −1.9%. Climatology, which is the "
    "training mean for that hour of day and nothing more, beats the served model outright.",
    "openmeteo_melbourne": "The quiet city that goes stale anyway. Sydney is the obvious "
    "Australian choice and its PM2.5 barely moves; Melbourne swings 3.1× on winter wood smoke "
    "while staying near the WHO guideline all year. Its champion still decays to 2.16× its "
    "training error across 31 runs, and retraining buys −1.8%.",
    "synthetic": "A synthetic world with a controllable drift knob, so detection can be shown "
    "to fire exactly when the data is made to shift, and only then.",
    "scheduled": "The same loop running unattended, and not a city at all: this is the Kraków "
    "source again, with one monitoring cycle appended automatically by a weekly GitHub Action. "
    "What it shows is that the machinery still runs with nobody starting it, accruing its own "
    "history over calendar time.",
}


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
    recent, latest_actual = None, None
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

    loc = profile.location
    return {
        "key": key,
        "label": profile.label,
        "benchmark": benchmark,
        "target": {"name": "PM2.5", "units": "µg/m³", "who_24h_guideline": 15},
        "recent": recent,
        "story": STORY[key],
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
        },
        "as_of": _dates(runs["as_of"]),
        "psi": {f: _floats(runs[f"metrics.psi_{f}"]) for f in FEATURES if f"metrics.psi_{f}" in runs},
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
        "profiles": profiles,
    }
    out = OUT / "data.json"
    out.write_text(json.dumps(payload, indent=None), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB, "
          f"{len(profiles)} profiles: {', '.join(p['key'] for p in profiles)})")
    return out


if __name__ == "__main__":
    build()
