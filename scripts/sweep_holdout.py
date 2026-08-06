"""How long should the promotion exam be?

`docs/evaluation.md` shows the seven-day exam is well calibrated over the horizon
it tests and reverses sign beyond twenty weeks: promotions that went on to serve
a long time delivered a negative margin despite passing convincingly. And when
the retrain trigger was made more sensitive, the outcome got *worse* in the two
cities where it shipped more models, which pointed at the exam rather than the
trigger as the binding constraint.

This sweeps `holdout_days` to test that. A longer exam is a larger sample of
unseen weather, so it should reject challengers that win a lucky week, and the
prediction is that the delivered margin tracks the promised one further out.

The replay cadence is held at 7 days for every arm, so exam length is the only
thing varying. That is possible only since the cadence guard was corrected:
`step_days >= holdout_days` used to reject a 14-day exam at a weekly cadence
while admitting a 3-day one that overlapped the next monitor window. The real
condition is `step_days + holdout_days >= monitor_days`, which every arm here
satisfies and which shorter exams than 7 would violate.

    python scripts/sweep_holdout.py [--city <name>|all] [--holdouts 7,10,14,21]

Writes outputs/holdout_sweep.csv and outputs/holdout_sweep_gate.csv.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import replace
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from driftloop import retrospect, tracking
from driftloop.config import CITY_CLI_NAMES as CITIES
from driftloop.config import PROFILES, Profile
from driftloop.data import OpenMeteoSource
from driftloop.loop import bootstrap_champion, run_simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"
STEP_DAYS = 7


def run_arm(profile: Profile, source: OpenMeteoSource, holdout: int) -> tuple[dict, list[dict]]:
    """One full replay at one exam length, scored the way the published numbers are."""
    cfg = replace(
        profile.loop,
        holdout_days=holdout,
        experiment_name=f"holdout-{holdout}",
        registered_model_name=f"{profile.loop.registered_model_name}-h{holdout}",
    )
    db = f"mlflow_holdout_{profile.key}_{holdout}.db"

    tracking.reset(db)
    tracking.setup(cfg.experiment_name, db)
    plan = profile.replay

    # A long exam reaches back past the bootstrap champion's training data on the
    # first run, which the in-cycle leak guard rejects. Each city's gap between
    # training end and first run differs (9 days in Santiago, 14 in Kraków), so a
    # fixed start would let some cities run an arm and not others, and the pooled
    # comparison would silently be over a different set of cities per arm.
    #
    # Advance the start in whole replay steps instead, so every city runs every
    # arm on the same weekly grid, a few runs shorter. `runs` is reported per arm
    # so the cost is visible rather than assumed away.
    first_run = plan.first_run
    while first_run - plan.champion_train_end < pd.Timedelta(holdout, unit="D"):
        first_run += pd.Timedelta(STEP_DAYS, unit="D")

    bootstrap_champion(source, plan.champion_train_start, plan.champion_train_end, cfg)
    df = run_simulation(source, cfg, first_run, plan.last_run, STEP_DAYS)

    runs = mlflow.search_runs(
        experiment_names=[cfg.experiment_name], order_by=["attributes.start_time ASC"]
    )
    runs = runs[runs["tags.cycle_type"] == "monitor"].copy()
    runs["as_of"] = pd.to_datetime(runs["params.as_of"])
    runs = runs.sort_values("as_of").reset_index(drop=True)

    models = retrospect.registered_models(MlflowClient(), cfg.registered_model_name)
    frame = runs.copy()
    frame["champion_version"] = runs["tags.champion_version"].astype(int)
    retro = retrospect.build(
        source, frame, models, cfg.monitor_days,
        source.forecast_lead_days, cfg.promotion_margin,
    )
    value = retrospect.retraining_value(retro)

    gate = [
        {
            "city": profile.label, "holdout": holdout,
            "version": g["version"], "weeks_served": g["weeks_served"],
            "exam_margin": g["exam_margin"], "delivered_margin": g["delivered_margin"],
        }
        for g in retro.gate
        if not pd.isna(g["exam_margin"]) and not pd.isna(g["delivered_margin"])
    ]

    # Keyed by date so medians can be recomputed over the window every arm
    # shares. Without that, a longer exam looks worse in any city whose replay
    # opens in its clean season: the later start drops low-error runs and the
    # median rises for a reason that has nothing to do with the exam.
    per_run = {
        d.strftime("%Y-%m-%d"): v
        for d, v in zip(retro.as_of, retro.champion_rmse)
        if not np.isnan(v)
    }

    summary = {
        "city": profile.label,
        "holdout": holdout,
        "runs": len(df),
        "first_run": first_run.strftime("%Y-%m-%d"),
        "retrains": int(df["retrain_triggered"].sum()),
        "promotions": int((df["promotion_decision"] == "promoted").sum()),
        # Of the challengers that were trained, how many cleared the bar. A longer
        # exam should be harder to fluke, so this is expected to fall.
        "promotion_rate": round(
            100 * (df["promotion_decision"] == "promoted").sum() / max(1, int(df["retrain_triggered"].sum())), 1
        ),
        "median_rmse": round(float(np.nanmedian(retro.champion_rmse)), 2),
        "median_skill": round(float(np.nanmedian(retro.champion_skill)), 3),
        "across": round(value.get("across_replay", float("nan")), 2),
        "paired": round(value.get("when_it_acted", float("nan")), 2),
        "win_rate": round(value.get("win_rate", float("nan")), 1),
    }
    return summary, gate, per_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=[*CITIES, "all"], default="all")
    parser.add_argument("--holdouts", default="7,10,14,21")
    args = parser.parse_args()

    holdouts = [int(h) for h in args.holdouts.split(",")]
    keys = list(CITIES) if args.city == "all" else [args.city]

    rows, gate_rows = [], []
    original_root = tracking.REPO_ROOT
    with tempfile.TemporaryDirectory(prefix="driftloop-holdout-", ignore_cleanup_errors=True) as tmp:
        tracking.REPO_ROOT = Path(tmp)
        try:
            for name in keys:
                profile = PROFILES[CITIES[name]]
                if profile.replay is None or profile.location is None:
                    continue
                source = OpenMeteoSource(profile.location)
                source.timeline()
                print(f"\n=== {profile.label} ===", flush=True)
                city_rows, city_series = [], []
                for holdout in holdouts:
                    summary, gate, per_run = run_arm(profile, source, holdout)
                    city_rows.append(summary)
                    city_series.append(per_run)
                    gate_rows.extend(gate)
                    print(
                        f"  holdout {holdout:>2}d: {summary['runs']:>2} runs from "
                        f"{summary['first_run']}, {summary['retrains']:>3} retrains, "
                        f"{summary['promotions']:>2} promoted ({summary['promotion_rate']:>5.1f}% of "
                        f"challengers), median RMSE {summary['median_rmse']:>6.2f}",
                        flush=True,
                    )
                # Every arm scored over the run dates all of them share.
                common = set(city_series[0]).intersection(*city_series[1:]) if city_series else set()
                for summary, series in zip(city_rows, city_series):
                    summary["common_runs"] = len(common)
                    summary["median_rmse_common"] = (
                        round(float(np.median([series[d] for d in sorted(common)])), 2) if common else float("nan")
                    )
                print(f"  on the {len(common)} runs common to every arm: " + ", ".join(
                    f"{s['holdout']}d {s['median_rmse_common']:.2f}" for s in city_rows), flush=True)
                rows.extend(city_rows)
            OUTPUTS.mkdir(exist_ok=True)
            pd.DataFrame(rows).to_csv(OUTPUTS / "holdout_sweep.csv", index=False)
            pd.DataFrame(gate_rows).to_csv(OUTPUTS / "holdout_sweep_gate.csv", index=False)
            print(f"\nwrote {OUTPUTS / 'holdout_sweep.csv'} and {OUTPUTS / 'holdout_sweep_gate.csv'}")
        finally:
            tracking.REPO_ROOT = original_root

    # The question the sweep exists for: does a longer exam predict what the
    # winner goes on to deliver? Pooled across cities, split at the horizon where
    # the seven-day exam was already known to reverse.
    gate = pd.DataFrame(gate_rows)
    if gate.empty:
        return
    print(f"\n=== gate calibration by exam length (pooled, split at {retrospect.GATE_LONG_WEEKS} weeks) ===\n")
    print(f"{'exam':>6} {'group':>12} {'n':>4} {'promised':>9} {'delivered':>10} {'harmful':>8}")
    print("-" * 54)
    for holdout in holdouts:
        g = gate[gate.holdout == holdout]
        for label, sub in (
            (f"< {retrospect.GATE_LONG_WEEKS}w", g[g.weeks_served < retrospect.GATE_LONG_WEEKS]),
            (f">= {retrospect.GATE_LONG_WEEKS}w", g[g.weeks_served >= retrospect.GATE_LONG_WEEKS]),
        ):
            if sub.empty:
                print(f"{holdout:>5}d {label:>12} {0:>4}")
                continue
            print(f"{holdout:>5}d {label:>12} {len(sub):>4} "
                  f"{sub.exam_margin.mean() * 100:>+8.1f}% {sub.delivered_margin.mean() * 100:>+9.1f}% "
                  f"{int((sub.delivered_margin < 0).sum()):>8}")


if __name__ == "__main__":
    main()
