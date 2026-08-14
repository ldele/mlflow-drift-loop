"""Does re-examining a serving champion on a schedule pay?

Two sensitivity fixes have been measured and neither pays. A second, scale-free
retrain trigger leaves five of six cities bit-identical at the cautious setting
(`sweep_skill_floor.py`). A longer promotion exam does not repair the
long-serving reversal at any length tested, and costs the cities where
retraining works (`sweep_holdout.py`). Both were attempts to make the loop
notice sooner that something was wrong, and firing more often only feeds more
challengers to a gate that certifies for about five weeks.

What neither addresses is that nothing re-examines an incumbent at all. Both
existing triggers ask whether the champion is failing, and both can answer no
indefinitely: the ratio ratchets shut because its denominator is the champion's
own training error, and PSI saturates. A model can serve for seven months on
training data from another season with every signal quiet.

`LoopConfig.recertify_days` is the third rule. It asks nothing about error, only
how long since the champion last passed a holdout exam, and fires when that
certificate expires. Passing renews it, so the cadence is a schedule rather than
a surrender: a champion that keeps beating challengers keeps its place and sits
the next exam `recertify_days` later.

    python scripts/sweep_recertify.py [--city <name>|all] [--cadences 14,28,35,56]

35 days is the default cadence to beat: `gate_summary` puts the seven-day exam's
shelf life at roughly five weeks, so it re-examines a model as its certificate
lapses rather than at a round number.

Each cadence gets a full replay of its own into a throwaway backend outside the
repository, because a changed trigger changes which models exist and therefore
the whole trajectory. Counting when a rule *would* have fired against the
existing champions cannot answer whether firing helps.

Writes outputs/recertify_sweep.csv, plus outputs/recertify_series.json with the
per-window errors behind it, so a statistic can be re-derived without replaying
anything. The `off` row reproduces the shipped numbers to the decimal, which is
the check that the harness is faithful.

Read the cost columns beside the outcome ones. Re-certification trains a
challenger on a schedule whether or not anything drifted, and an arm that buys
an improvement with three times the training runs has not been shown to be
worth switching on.
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
    "city", "cadence", "runs",
    # Cost. A schedule that fires regardless of drift trains models nobody
    # asked for, and the count is the price of whatever the outcome columns
    # show. `challengers` is every model trained; `promotions` is how many were
    # worth keeping; the difference is the waste the gate caught.
    "retrains", "promotions", "fired_by_recert", "longest_silence",
    # The quantity this trigger exists to bound, and the one no drift signal
    # reports: how stale the serving champion was allowed to get, in days since
    # it last passed an exam. The `off` arm is the number to look at first.
    "max_certified_age", "median_certified_age",
    # Outcome.
    "median_rmse", "median_skill", "worst_skill", "paired", "across", "win_rate",
    # Whether shipping more models shipped more bad ones. The gate is the only
    # thing standing between a schedule and churn, so a cadence that improves
    # the error while raising `gate_harmful` bought it with luck.
    "gate_n", "gate_harmful", "gate_delivered",
    # What the cadence is worth against leaving the trigger alone, week by week,
    # with the interval.
    "vs_off", "vs_off_lo", "vs_off_hi", "vs_off_real",
    # The same comparison over only the weeks the arm changed the serving model.
    # Most weeks it does not, and those ties dominate the median.
    "differing_weeks", "vs_off_acted", "vs_off_acted_lo", "vs_off_acted_hi",
    "vs_off_acted_real",
]


def _slug(cadence: int | None) -> str:
    return "off" if cadence is None else f"{cadence}d"


def _longest_silence(fired: np.ndarray) -> int:
    """The longest run of consecutive weeks that trained nothing.

    The measure the ratchet shows up in: a trigger that has gone permanently
    deaf has one enormous run at the end, which no average over the replay would
    make visible. Re-certification puts a ceiling on this by construction, so
    the column is a check that the mechanism did what it claims rather than a
    finding in itself.
    """
    longest = current = 0
    for f in fired:
        current = 0 if f else current + 1
        longest = max(longest, current)
    return longest


def run_arm(profile: Profile, source: OpenMeteoSource, cadence: int | None) -> dict:
    """One full replay at one cadence, scored the way the published numbers are."""
    slug = _slug(cadence)
    cfg = replace(
        profile.loop,
        recertify_days=cadence,
        experiment_name=f"recertify-{slug}",
        registered_model_name=f"{profile.loop.registered_model_name}-recert-{slug}",
    )
    db = f"mlflow_recert_{profile.key}_{slug}.db"

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

    # Membership, not equality: a run names every rule that fired, so a week the
    # schedule and the ratio both caught reads "ratio+recert".
    reasons = runs["tags.retrain_reason"] if "tags.retrain_reason" in runs else pd.Series(dtype=str)
    fired_by_recert = int(reasons.astype(str).str.contains("recert").sum())

    age = (
        pd.to_numeric(runs["metrics.certified_age_days"], errors="coerce")
        if "metrics.certified_age_days" in runs
        else pd.Series(dtype=float)
    )

    # Pooled over this arm's promotions. Split short from long the way the
    # published calibration does, and report the long group, because that is
    # where the exam reverses sign and where a schedule that ships more models
    # would do its damage.
    gate = retrospect.gate_summary(retro.gate)
    long = gate.get("long", {})

    return {
        # The per-window error series, kept so this arm can be compared against
        # the "off" arm week by week rather than median against median. Both
        # arms replay the same as_of dates, so the two series are paired by
        # construction and the pairing is what makes an interval meaningful.
        "_rmse": [float(v) for v in retro.champion_rmse],
        "_as_of": [str(d) for d in retro.as_of],
        "city": profile.label,
        "cadence": slug,
        "runs": len(df),
        "retrains": int(df["retrain_triggered"].sum()),
        "promotions": int((df["promotion_decision"] == "promoted").sum()),
        "fired_by_recert": fired_by_recert,
        "longest_silence": _longest_silence(df["retrain_triggered"].to_numpy(bool)),
        "max_certified_age": round(float(age.max()), 1) if len(age) else float("nan"),
        "median_certified_age": round(float(age.median()), 1) if len(age) else float("nan"),
        "median_rmse": round(float(np.nanmedian(retro.champion_rmse)), 2),
        "median_skill": round(float(np.nanmedian(skill)), 3),
        "worst_skill": round(float(np.nanmin(skill)), 3),
        "paired": round(value.get("when_it_acted", float("nan")), 2),
        "across": round(value.get("across_replay", float("nan")), 2),
        "win_rate": round(value.get("win_rate", float("nan")), 1),
        "gate_n": int(long.get("n", 0)),
        "gate_harmful": int(long.get("harmful", 0)),
        "gate_delivered": round(float(long["delivered"]) * 100, 2) if long.get("n") else float("nan"),
    }


def compare_against_off(city_rows: list[dict]) -> None:
    """Fill each arm's ``vs_off`` columns, in place.

    The arms of one city replay the same weeks, so an arm and the ``off`` arm
    hold two error values for every window and the comparison is paired. Same
    shape as champion-against-frozen elsewhere in the project, so it uses the
    same statistic and the same block resampling: consecutive monitor windows
    overlap by half, and redrawing single weeks would report a range far
    narrower than the evidence supports.

    Aligned on ``as_of`` rather than on position, so a pairing can never drift
    by a week without the mismatch being dropped instead of scored.
    """
    baseline = next((r for r in city_rows if r["cadence"] == "off"), None)
    if baseline is None:
        return
    off_by_date = dict(zip(baseline["_as_of"], baseline["_rmse"]))

    for row in city_rows:
        if row is baseline:
            row["vs_off"] = 0.0
            row["vs_off_lo"] = row["vs_off_hi"] = 0.0
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

        # And again over only the weeks the arm changed something. Most weeks a
        # changed trigger leaves the serving model alone, so those weeks are
        # exact ties and they drag the median ratio above to zero whatever
        # happens in the rest. See stats.differing.
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
        "--cadences", default="14,28,35,56",
        help="comma-separated re-certification cadences in days, alongside 'off'",
    )
    args = parser.parse_args()

    cadences: list[int | None] = [None, *(int(c) for c in args.cadences.split(","))]
    keys = list(CITIES) if args.city == "all" else [args.city]

    rows = []
    # Every backend this creates is a throwaway. Redirecting REPO_ROOT is the
    # same hook the serving tests use to relocate a registry, and it keeps a
    # sweep from leaving a dozen mlflow_*.db files in the working tree.
    #
    # ignore_cleanup_errors because MLflow keeps its SQLAlchemy engine, and so
    # its sqlite handles, open for the life of the process, and Windows will not
    # unlink an open file. Without it the sweep completes, computes everything,
    # and then dies in the teardown having written nothing.
    original_root = tracking.REPO_ROOT
    with tempfile.TemporaryDirectory(prefix="driftloop-recert-", ignore_cleanup_errors=True) as tmp:
        tracking.REPO_ROOT = Path(tmp)
        try:
            for name in keys:
                profile = PROFILES[CITIES[name]]
                if profile.replay is None or profile.location is None:
                    continue
                source = OpenMeteoSource(profile.location)
                source.timeline()  # warm the parquet cache once, not once per arm
                print(f"\n=== {profile.label} ===", flush=True)
                city_rows = []
                for cadence in cadences:
                    row = run_arm(profile, source, cadence)
                    city_rows.append(row)
                    print(
                        f"  {row['cadence']:>4}: {row['retrains']:>3} trained, "
                        f"{row['promotions']:>2} promoted, oldest certificate "
                        f"{row['max_certified_age']:>6.0f}d, longest silence "
                        f"{row['longest_silence']:>2}, median RMSE {row['median_rmse']:>6.2f}, "
                        f"{row['gate_harmful']}/{row['gate_n']} long promotions harmful",
                        flush=True,
                    )
                compare_against_off(city_rows)
                for row in city_rows:
                    if row["cadence"] == "off":
                        continue
                    verdict = "clears zero" if row.get("vs_off_real") else "not distinguishable"
                    print(
                        f"    {row['cadence']:>4} against off: "
                        f"{row.get('vs_off', float('nan')):+6.2f}% "
                        f"[{row.get('vs_off_lo', float('nan')):+6.2f}, "
                        f"{row.get('vs_off_hi', float('nan')):+6.2f}]  {verdict}"
                        f"   over {row.get('differing_weeks', 0)} changed weeks: "
                        f"{row.get('vs_off_acted', float('nan')):+6.2f}%",
                        flush=True,
                    )
                rows.extend(city_rows)
            # Written before the temp directory is torn down, so a cleanup that
            # cannot remove a locked file still leaves the results behind.
            OUTPUTS.mkdir(exist_ok=True)
            out = OUTPUTS / "recertify_sweep.csv"
            pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(out, index=False)
            # The per-window error series behind every figure above. Each arm is
            # a full replay costing minutes, so re-deriving a statistic from
            # them should not mean running the whole sweep a second time.
            series = OUTPUTS / "recertify_series.json"
            series.write_text(
                json.dumps(
                    [
                        {"city": r["city"], "cadence": r["cadence"],
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
