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

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driftloop import tracking  # noqa: E402
from driftloop.config import FEATURES, PROFILES  # noqa: E402

OUT = REPO_ROOT / "site"

STORY = {
    "synthetic": "Phase 1 · synthetic data with a controllable drift knob.",
    "openmeteo": "Phase 2 · real Kraków weather + air quality; a summer-trained model walking into winter smog.",
    "scheduled": "Phase 3 · the live loop, one scheduled run appended each week.",
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

    return {
        "key": key,
        "label": profile.label,
        "story": STORY[key],
        "drift_date": drift_date,
        "stats": {
            "runs": int(len(runs)),
            "retrains": int(len(retr)),
            "promotions": int(len(prom)),
            "latest_psi": round(float(latest["metrics.data_drift_psi"]), 2),
            "latest_r2": (round(float(latest["metrics.champion_r2"]), 2)
                          if "metrics.champion_r2" in latest else None),
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


def build() -> Path:
    profiles = [d for d in (profile_data(k) for k in PROFILES) if d is not None]
    if not profiles:
        raise SystemExit("No profiles have data — run the pipelines first.")

    payload = {"built": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "profiles": profiles}
    OUT.mkdir(exist_ok=True)
    out = OUT / "data.json"
    out.write_text(json.dumps(payload, indent=None), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB, "
          f"{len(profiles)} profiles: {', '.join(p['key'] for p in profiles)})")
    return out


if __name__ == "__main__":
    build()
