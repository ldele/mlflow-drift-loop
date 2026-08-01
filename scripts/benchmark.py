"""Score the loop's champion against no-training baselines, and tune its alpha.

    python scripts/benchmark.py [--city <name>|all]

The city names come from config.CITY_CLI_NAMES; --help lists the current set.

Scoring happens per monitoring window -- the same 14-day slices the loop reports
on -- so the served champion, the never-retrained champion and the baselines all
land in one comparable column. The served champion's numbers are read from the
run the loop already logged (outputs/simulation_<profile>.csv), so run
scripts/run_openmeteo.py first.

Writes outputs/benchmark_<profile>.json. No network: reads the committed cache.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from driftloop.benchmark import predictor_columns, score_windows, tune_alpha
from driftloop.config import CITY_CLI_NAMES, PROFILES, Profile
from driftloop.data import OpenMeteoSource

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"


def benchmark_city(profile: Profile) -> dict | None:
    location, plan, cfg = profile.location, profile.replay, profile.loop
    if location is None or plan is None:
        raise SystemExit(f"profile {profile.key!r} has no location/replay windows")

    sim_path = OUTPUTS / f"simulation_{profile.key}.csv"
    if not sim_path.exists():
        print(f"  (no {sim_path.name}; run scripts/run_openmeteo.py first) — skipping")
        return None
    sim = pd.read_csv(sim_path, parse_dates=["as_of"])

    source = OpenMeteoSource(location)
    timeline = source.timeline()
    train = source.get_data(plan.champion_train_start, plan.champion_train_end)

    # The loop's own windows: each run scores the champion on the monitor_days
    # ending at as_of. Reusing them is what makes the columns comparable.
    offset = pd.Timedelta(days=cfg.monitor_days)
    windows = [(stamp - offset, stamp) for stamp in sim["as_of"]]

    lead = location.forecast_lead_days
    columns = predictor_columns(timeline, train, lead_days=lead)
    scored = score_windows(
        columns, windows, served_rmse=sim["champion_rmse"].tolist(), lead_days=lead
    )
    sweep = tune_alpha(train)

    print(f"\n=== {location.name} ===")
    print(f"  train {plan.champion_train_start.date()} -> {plan.champion_train_end.date()}"
          f" ({len(train)} rows) · {len(windows)} monitoring windows of {cfg.monitor_days}d")
    print(f"  alpha: shipped {sweep.shipped:g}, best {sweep.best:g} on {sweep.n_splits}-fold "
          f"forward CV (shipped costs {sweep.penalty_pct:+.1f}%)")
    print(f"\n  {'predictor':<18} {'median RMSE':>12}   notes")
    for s in scored:
        flag = "  [sees past PM2.5]" if s.uses_past_target else ""
        print(f"  {s.name:<18} {s.median_rmse:>12.2f}   {s.detail}{flag}")

    served = next((s for s in scored if s.name == "champion_served"), None)
    frozen = next((s for s in scored if s.name == "champion_frozen"), None)
    if served and frozen:
        delta = (1 - served.median_rmse / frozen.median_rmse) * 100
        print(f"\n  retraining buys {delta:+.1f}% vs never retraining")

    return {
        "city": location.name,
        "windows": len(windows),
        "monitor_days": cfg.monitor_days,
        "lead_days": lead,
        "scored": [asdict(s) for s in scored],
        "alpha": {
            "shipped": sweep.shipped,
            "best": sweep.best,
            "penalty_pct": None if pd.isna(sweep.penalty_pct) else round(sweep.penalty_pct, 2),
            "n_splits": sweep.n_splits,
            "curve": [[a, round(v, 4)] for a, v in sweep.curve],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=[*CITY_CLI_NAMES, "all"], default="all")
    args = parser.parse_args()

    OUTPUTS.mkdir(exist_ok=True)
    for name in (list(CITY_CLI_NAMES) if args.city == "all" else [args.city]):
        key = CITY_CLI_NAMES[name]
        payload = benchmark_city(PROFILES[key])
        if payload is None:
            continue
        out = OUTPUTS / f"benchmark_{key}.json"
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
