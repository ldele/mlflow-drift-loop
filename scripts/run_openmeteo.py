"""Phase 2 demo: the same drift loop, on real weather + air-quality data.

Runs one city per invocation, or all of them by default. Each city bootstraps a
champion on a clean training season, then replays weekly scheduled runs into the
season that spoils it -- the windows live on the Profile, because the seasons
don't line up between cities.

    python scripts/run_openmeteo.py [--fresh] [--city krakow|delhi|la|all]

First run fetches each span from Open-Meteo and caches it to data_cache/; later
runs reuse the cache. Each city logs to its own MLflow backend so they reset and
browse independently, and none of them collide with the synthetic Phase 1 runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from driftloop import tracking
from driftloop.config import CITY_CLI_NAMES as CITIES
from driftloop.config import PROFILES, Profile
from driftloop.data import OpenMeteoSource
from driftloop.loop import bootstrap_champion, run_simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"


def run_city(profile: Profile, fresh: bool) -> pd.DataFrame:
    loop_cfg = profile.loop
    om_cfg = profile.location
    plan = profile.replay
    if om_cfg is None or plan is None:
        raise SystemExit(f"profile {profile.key!r} has no location/replay windows")

    if fresh:
        tracking.reset(profile.db_filename)
    tracking.setup(loop_cfg.experiment_name, profile.db_filename)

    source = OpenMeteoSource(om_cfg)

    print(f"\n=== {om_cfg.name} ({om_cfg.country}) ===")
    print(f"Fetching span {om_cfg.origin.date()} -> {om_cfg.horizon.date()} ...")
    timeline = source.timeline()
    print(f"  {len(timeline)} clean hourly rows "
          f"({timeline['timestamp'].min()} .. {timeline['timestamp'].max()})")
    print(f"  PM2.5 µg/m³  summer mean vs winter mean: "
          f"{_season_mean(timeline, [6, 7, 8]):.1f} vs {_season_mean(timeline, [12, 1, 2]):.1f}")

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / profile.meta_filename).write_text(
        json.dumps(
            {
                "drift_date": None,  # real data has no single engineered regime shift
                "location": om_cfg.name,
                "country": om_cfg.country,
                "latitude": om_cfg.latitude,
                "longitude": om_cfg.longitude,
                "champion_train_start": plan.champion_train_start.isoformat(),
                "champion_train_end": plan.champion_train_end.isoformat(),
            }
        ),
        encoding="utf-8",
    )

    print(f"Bootstrapping champion on "
          f"{plan.champion_train_start.date()} -> {plan.champion_train_end.date()}")
    version = bootstrap_champion(
        source, plan.champion_train_start, plan.champion_train_end, loop_cfg
    )
    print(f"  registered {loop_cfg.registered_model_name} v{version} as @champion")

    print(f"Replaying weekly runs {plan.first_run.date()} -> {plan.last_run.date()}")
    df = run_simulation(source, loop_cfg, plan.first_run, plan.last_run, plan.step_days)

    out = OUTPUTS / f"simulation_{profile.key}.csv"
    df.to_csv(out, index=False)

    cols = ["as_of", "data_drift_psi", "perf_drift_ratio", "champion_rmse", "promotion_decision"]
    with pd.option_context("display.width", 120, "display.max_rows", None):
        print("\n" + df[cols].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    promotions = df[df.promotion_decision == "promoted"]
    print(f"\n{om_cfg.name}: runs={len(df)}  retrains={int(df.retrain_triggered.sum())}  "
          f"promotions={len(promotions)}")
    for _, row in promotions.iterrows():
        print(f"  promoted at {row.as_of:%Y-%m-%d} (gap {row.performance_gap:.2f} RMSE)")
    print(f"wrote {out}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help="wipe the city's backend first")
    parser.add_argument(
        "--city",
        choices=[*CITIES, "all"],
        default="all",
        help="which city to run (default: all)",
    )
    args = parser.parse_args()

    keys = list(CITIES) if args.city == "all" else [args.city]
    summary = []
    for name in keys:
        df = run_city(PROFILES[CITIES[name]], args.fresh)
        summary.append((name, len(df), int(df.retrain_triggered.sum()),
                        int((df.promotion_decision == "promoted").sum())))

    if len(summary) > 1:
        print("\n=== summary ===")
        for name, runs, retrains, proms in summary:
            print(f"  {name:<8} runs={runs:<4} retrains={retrains:<4} promotions={proms}")
    print("\nDashboard:  streamlit run dashboard/app.py   (the selector picks the city)")


def _season_mean(df: pd.DataFrame, months: list[int]) -> float:
    return float(df.loc[df["timestamp"].dt.month.isin(months), "pm25"].mean())


if __name__ == "__main__":
    main()
