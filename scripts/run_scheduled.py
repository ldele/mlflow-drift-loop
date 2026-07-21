"""Phase 3: one incremental scheduled cycle against a persistent backend.

This is what the GitHub Action calls on a cron. Unlike the Phase 1/2 scripts
(which replay a whole timeline in a single process), each invocation does exactly
one thing and then exits:

  * no champion registered yet  -> bootstrap one on the trailing window (first deploy),
  * champion exists             -> run a single monitoring cycle at ``as_of`` and append it.

State lives in a persistent MLflow backend (``mlflow_scheduled.db``) that survives
between runs -- locally that's just the file; in CI the workflow commits it back
so the next scheduled run continues where this one left off. Nothing is ever reset.

    python scripts/run_scheduled.py [--as-of YYYY-MM-DD] [--lag-days N]

``--as-of`` pins the run date (useful for backfills and for local testing across
several "weeks"); without it, the run targets ``today - lag-days`` -- the lag
covers the reanalysis delay in the weather / air-quality feeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from driftloop import tracking
from driftloop.config import PROFILES, OpenMeteoConfig
from driftloop.data import OpenMeteoSource
from driftloop.loop import bootstrap_champion, run_cycle
from driftloop.tracking import load_champion

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"
PROFILE = PROFILES["scheduled"]

# The first champion is trained on the trailing ~2.5 months before deployment.
BOOTSTRAP_TRAIN_DAYS = 75
# Fetch a generous trailing span so any recent champion's training window and the
# monitor/challenger windows are all covered by one cached pull.
TRAILING_FETCH_DAYS = 400


def _resolve_as_of(args: argparse.Namespace) -> pd.Timestamp:
    if args.as_of:
        return pd.Timestamp(args.as_of).normalize()
    # ERA5 / air-quality reanalysis lags real time; step back to a safe date.
    return pd.Timestamp.now().normalize() - pd.Timedelta(days=args.lag_days)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=str, default=None, help="YYYY-MM-DD; default = today - lag")
    parser.add_argument("--lag-days", type=int, default=7)
    args = parser.parse_args()

    as_of = _resolve_as_of(args)
    cfg = PROFILE.loop
    # NB: setup only -- never reset(). The whole point of Phase 3 is persistence.
    tracking.setup(cfg.experiment_name, PROFILE.db_filename)

    om_cfg = OpenMeteoConfig(
        origin=(as_of - pd.Timedelta(days=TRAILING_FETCH_DAYS)).normalize(),
        horizon=(as_of + pd.Timedelta(days=1)).normalize(),
    )
    source = OpenMeteoSource(om_cfg)

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / PROFILE.meta_filename).write_text(
        json.dumps(
            {
                "drift_date": None,
                "location": om_cfg.name,
                "latitude": om_cfg.latitude,
                "longitude": om_cfg.longitude,
            }
        ),
        encoding="utf-8",
    )

    champion = load_champion(cfg.registered_model_name)
    if champion is None:
        train_start = as_of - pd.Timedelta(days=BOOTSTRAP_TRAIN_DAYS)
        print(f"[{as_of.date()}] no champion yet -> bootstrapping on "
              f"{train_start.date()} .. {as_of.date()} (first deploy)")
        version = bootstrap_champion(source, train_start, as_of, cfg)
        print(f"  registered {cfg.registered_model_name} v{version} as @champion")
        return

    result = run_cycle(source, as_of, cfg)
    line = (
        f"[{as_of.date()}] champion v{result.champion_version}  "
        f"psi={result.data_drift_psi:5.2f}  perf_ratio={result.perf_drift_ratio:4.2f}  "
        f"rmse={result.champion_rmse:6.2f}  -> {result.promotion_decision}"
    )
    if result.challenger_rmse is not None:
        line += (f"  (challenger {result.challenger_rmse:.2f} vs "
                 f"champion {result.champion_rmse_holdout:.2f} on holdout)")
    print(line)


if __name__ == "__main__":
    main()
