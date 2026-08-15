"""Does making a promotion reversible pay, where making it harder did not?

Four mechanisms have been measured and none pays. A second retrain trigger, a
longer exam, a re-certification schedule, a confidence-aware gate. Two make the
loop notice sooner, one makes it look more often, one makes it judge harder, and
the fourth is actively harmful (docs/DECISIONS.md D3).

They fail for one reason. The loop is a retry loop: it keeps training
challengers and sitting exams until one passes, so the promoted model always
carries the luck of whichever attempt cleared the bar. Raising the bar buys more
attempts and a luckier winner, which is why the confidence gate made the
winner's curse worse rather than better. No rule about *when* an attempt happens
or *how hard* one is can price the *number* of attempts.

`LoopConfig.probation_days` does not try. It leaves the gate alone and makes the
outcome reversible: this many days after a promotion, the new champion is scored
against the model it displaced on a window that postdates them both, and loses
its place if it is worse.

The reason to expect something different is that this is a check rather than a
selection. Nothing is maximised, one model is compared with one alternative,
once, at a time fixed before the result is known. The winner's curse comes from
taking the best of many attempts and there are no attempts here, so the luck
that got a challenger through the exam does not repeat.

    python scripts/sweep_probation.py [--city <name>|all] [--windows 14,21,28]

14 days at a weekly cadence is the first window that postdates the promotion
entirely; at 7 the monitor window still holds the holdout the challenger was
selected on. Longer settings judge a later fortnight, which is a slightly
different question: the further out, the more the verdict mixes "this promotion
was noise" with "the world moved again since".

Writes outputs/probation_sweep.csv and outputs/probation_series.json. The `off`
row reproduces the shipped numbers to the decimal.

Arms are compared on the per-window error series rather than on summary
statistics, so a changed number of promotions cannot move the comparison by
changing what is being averaged.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from driftloop import retrospect, stats, tracking
from driftloop.config import CITY_CLI_NAMES as CITIES
from driftloop.config import PROFILES, Profile
from driftloop.data import OpenMeteoSource
from driftloop.loop import bootstrap_champion, run_simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"

COLUMNS = [
    "city", "probation", "runs", "retrains", "promotions",
    # The mechanism, counted. `judged` is promotions that reached their
    # probation window before the replay ended; `rolled_back` is those undone.
    "judged", "rolled_back", "kept",
    # The verdict the check itself returned, averaged. A promotion's advantage
    # over the model it displaced, on a window neither influenced. Negative
    # means the exam had shipped something worse.
    "mean_probation_margin",
    "median_rmse", "median_skill", "paired", "across", "win_rate",
    "vs_off", "vs_off_lo", "vs_off_hi", "vs_off_real",
    "differing_weeks", "vs_off_acted", "vs_off_acted_lo", "vs_off_acted_hi",
    "vs_off_acted_real",
]


def _slug(days: int | None) -> str:
    return "off" if days is None else f"{days}d"


def run_arm(profile: Profile, source: OpenMeteoSource, days: int | None) -> dict:
    """One full replay at one probation window."""
    slug = _slug(days)
    cfg = replace(
        profile.loop,
        probation_days=days,
        experiment_name=f"probation-{slug}",
        registered_model_name=f"{profile.loop.registered_model_name}-prob-{slug}",
    )
    db = f"mlflow_prob_{profile.key}_{slug}.db"

    tracking.reset(db)
    tracking.setup(cfg.experiment_name, db)
    plan = profile.replay
    bootstrap_champion(source, plan.champion_train_start, plan.champion_train_end, cfg)
    df = run_simulation(source, cfg, plan.first_run, plan.last_run, plan.step_days)

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
    skill = np.asarray(retro.champion_skill, dtype=float)

    decisions = df["probation_decision"] if "probation_decision" in df else pd.Series(dtype=str)
    rolled = int((decisions == "rolled_back").sum())
    kept = int((decisions == "kept").sum())
    margins = pd.to_numeric(df.get("probation_margin"), errors="coerce").dropna()

    return {
        "_rmse": [float(v) for v in retro.champion_rmse],
        "_as_of": [str(d) for d in retro.as_of],
        "city": profile.label,
        "probation": slug,
        "runs": len(df),
        "retrains": int(df["retrain_triggered"].sum()),
        "promotions": int((df["promotion_decision"] == "promoted").sum()),
        "judged": kept + rolled,
        "rolled_back": rolled,
        "kept": kept,
        "mean_probation_margin": round(float(margins.mean()) * 100, 2) if len(margins) else float("nan"),
        "median_rmse": round(float(np.nanmedian(retro.champion_rmse)), 2),
        "median_skill": round(float(np.nanmedian(skill)), 3),
        "paired": round(value.get("when_it_acted", float("nan")), 2),
        "across": round(value.get("across_replay", float("nan")), 2),
        "win_rate": round(value.get("win_rate", float("nan")), 1),
    }


def compare_against_off(city_rows: list[dict]) -> None:
    """Fill each arm's ``vs_off`` columns, in place. See sweep_recertify.py."""
    baseline = next((r for r in city_rows if r["probation"] == "off"), None)
    if baseline is None:
        return
    off_by_date = dict(zip(baseline["_as_of"], baseline["_rmse"]))

    for row in city_rows:
        if row is baseline:
            row["vs_off"] = row["vs_off_lo"] = row["vs_off_hi"] = 0.0
            row["vs_off_real"] = False
            row["differing_weeks"] = 0
            continue
        pairs = [(a, o) for a, o in zip(row["_rmse"], (off_by_date.get(d) for d in row["_as_of"]))
                 if o is not None]
        if not pairs:
            continue
        arm = np.array([p[0] for p in pairs], dtype=float)
        off = np.array([p[1] for p in pairs], dtype=float)
        interval = stats.block_bootstrap((arm, off), stats.pct_improvement_paired)
        row["vs_off"] = round(interval.point, 2)
        row["vs_off_lo"] = round(interval.lo, 2)
        row["vs_off_hi"] = round(interval.hi, 2)
        row["vs_off_real"] = interval.excludes_zero

        acted = stats.differing(arm, off)
        row["differing_weeks"] = int(acted.sum())
        if acted.any():
            acted_interval = stats.block_bootstrap(
                (arm[acted], off[acted]), stats.pct_improvement_paired
            )
            row["vs_off_acted"] = round(acted_interval.point, 2)
            row["vs_off_acted_lo"] = round(acted_interval.lo, 2)
            row["vs_off_acted_hi"] = round(acted_interval.hi, 2)
            row["vs_off_acted_real"] = acted_interval.excludes_zero


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=[*CITIES, "all"], default="all")
    parser.add_argument(
        "--windows", default="14,21,28",
        help="comma-separated probation windows in days, alongside 'off'",
    )
    args = parser.parse_args()

    windows: list[int | None] = [None, *(int(w) for w in args.windows.split(","))]
    keys = list(CITIES) if args.city == "all" else [args.city]

    rows = []
    original_root = tracking.REPO_ROOT
    with tempfile.TemporaryDirectory(prefix="driftloop-prob-", ignore_cleanup_errors=True) as tmp:
        tracking.REPO_ROOT = Path(tmp)
        try:
            for name in keys:
                profile = PROFILES[CITIES[name]]
                if profile.replay is None or profile.location is None:
                    continue
                source = OpenMeteoSource(profile.location)
                source.timeline()
                print(f"\n=== {profile.label} ===", flush=True)
                city_rows = []
                for days in windows:
                    row = run_arm(profile, source, days)
                    city_rows.append(row)
                    print(
                        f"  {row['probation']:>4}: {row['promotions']:>2} promoted, "
                        f"{row['judged']:>2} judged, {row['rolled_back']:>2} rolled back, "
                        f"mean verdict {row['mean_probation_margin']:>7.2f}%, "
                        f"median RMSE {row['median_rmse']:>6.2f}, "
                        f"retraining {row['paired']:+.1f}% paired",
                        flush=True,
                    )
                compare_against_off(city_rows)
                for row in city_rows:
                    if row["probation"] == "off":
                        continue
                    verdict = "clears zero" if row.get("vs_off_real") else "not distinguishable"
                    print(
                        f"    {row['probation']:>4} against off: "
                        f"{row.get('vs_off', float('nan')):+6.2f}% "
                        f"[{row.get('vs_off_lo', float('nan')):+6.2f}, "
                        f"{row.get('vs_off_hi', float('nan')):+6.2f}]  {verdict}"
                        f"   over {row.get('differing_weeks', 0)} changed weeks: "
                        f"{row.get('vs_off_acted', float('nan')):+6.2f}%",
                        flush=True,
                    )
                rows.extend(city_rows)
            OUTPUTS.mkdir(exist_ok=True)
            out = OUTPUTS / "probation_sweep.csv"
            pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(out, index=False)
            series = OUTPUTS / "probation_series.json"
            series.write_text(
                json.dumps(
                    [
                        {"city": r["city"], "probation": r["probation"],
                         "as_of": r["_as_of"], "rmse": r["_rmse"]}
                        for r in rows
                    ],
                    indent=1,
                ),
                encoding="utf-8",
            )
            print(f"\nwrote {out} and {series}")
        finally:
            tracking.REPO_ROOT = original_root


if __name__ == "__main__":
    main()
