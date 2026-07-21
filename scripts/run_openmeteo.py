"""Phase 2 demo: the same drift loop, on real Kraków weather + air-quality data.

Trains a champion on summer 2025, then replays weekly scheduled runs into the
winter heating season, when basin inversions send PM2.5 up and the summer-learned
relationship decays for real.

    python scripts/run_openmeteo.py [--fresh]

First run fetches the span from Open-Meteo and caches it to data_cache/; later
runs reuse the cache. Logs to a separate MLflow backend (mlflow_openmeteo.db) so
it never collides with the synthetic Phase 1 runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from driftloop import tracking
from driftloop.config import PROFILES, OpenMeteoConfig
from driftloop.data import OpenMeteoSource
from driftloop.loop import bootstrap_champion, run_simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"
PROFILE = PROFILES["openmeteo"]

# Champion learns summer; the replay walks into the winter smog season.
CHAMPION_TRAIN_START = pd.Timestamp("2025-06-01")
CHAMPION_TRAIN_END = pd.Timestamp("2025-08-01")
FIRST_RUN = pd.Timestamp("2025-08-15")
LAST_RUN = pd.Timestamp("2026-01-20")
STEP_DAYS = 7


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help="wipe the Open-Meteo backend first")
    args = parser.parse_args()

    loop_cfg = PROFILE.loop
    if args.fresh:
        tracking.reset(PROFILE.db_filename)
    tracking.setup(loop_cfg.experiment_name, PROFILE.db_filename)

    om_cfg = OpenMeteoConfig()
    source = OpenMeteoSource(om_cfg)

    print(f"Fetching {om_cfg.name} span {om_cfg.origin.date()} -> {om_cfg.horizon.date()} ...")
    timeline = source.timeline()
    print(f"  {len(timeline)} clean hourly rows "
          f"({timeline['timestamp'].min()} .. {timeline['timestamp'].max()})")
    print(f"  PM2.5 µg/m³  summer mean vs winter mean: "
          f"{_season_mean(timeline, [6, 7, 8]):.1f} vs {_season_mean(timeline, [12, 1, 2]):.1f}\n")

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / PROFILE.meta_filename).write_text(
        json.dumps(
            {
                "drift_date": None,  # real data has no single engineered regime shift
                "location": om_cfg.name,
                "latitude": om_cfg.latitude,
                "longitude": om_cfg.longitude,
                "champion_train_start": CHAMPION_TRAIN_START.isoformat(),
                "champion_train_end": CHAMPION_TRAIN_END.isoformat(),
            }
        ),
        encoding="utf-8",
    )

    print(f"Bootstrapping champion on {CHAMPION_TRAIN_START.date()} -> {CHAMPION_TRAIN_END.date()}")
    version = bootstrap_champion(source, CHAMPION_TRAIN_START, CHAMPION_TRAIN_END, loop_cfg)
    print(f"  registered {loop_cfg.registered_model_name} v{version} as @champion\n")

    print(f"Replaying weekly runs {FIRST_RUN.date()} -> {LAST_RUN.date()}")
    df = run_simulation(source, loop_cfg, FIRST_RUN, LAST_RUN, STEP_DAYS)

    out = OUTPUTS / "simulation_openmeteo.csv"
    df.to_csv(out, index=False)

    cols = ["as_of", "data_drift_psi", "perf_drift_ratio", "champion_rmse", "promotion_decision"]
    with pd.option_context("display.width", 120, "display.max_rows", None):
        print("\n" + df[cols].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    promotions = df[df.promotion_decision == "promoted"]
    print(f"\nruns={len(df)}  retrains={int(df.retrain_triggered.sum())}  promotions={len(promotions)}")
    for _, row in promotions.iterrows():
        print(f"  promoted at {row.as_of:%Y-%m-%d} (gap {row.performance_gap:.2f} RMSE)")
    print(f"\nwrote {out}")
    print("Dashboard:  streamlit run dashboard/app.py   (pick the Open-Meteo profile)")


def _season_mean(df: pd.DataFrame, months: list[int]) -> float:
    return float(df.loc[df["timestamp"].dt.month.isin(months), "pm25"].mean())


if __name__ == "__main__":
    main()
