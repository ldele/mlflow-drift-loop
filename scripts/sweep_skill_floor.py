"""Does waking the retrain trigger back up actually help?

`perf_drift_ratio` ratchets: its denominator is the champion's own training
error, every promotion resets it, and promotions happen at the seasonal peak, so
each new champion inherits a higher bar than the model it replaced and the bar
never comes back down. Los Angeles spends 78% of its replay unable to fire at
any error whatsoever; Kraków 62%.

`docs/evaluation.md` proposed two fixes for that. One of them, an absolute RMSE
floor, cannot be built: to wake Los Angeles, whose deaf stretch tops out at 18
µg/m³, the floor has to sit below 18 -- where Delhi fires on every single run.
There is no value in between, so an "absolute" floor is a per-city tuning knob
in disguise and the cities stop being comparable.

The other fix is a yardstick that does not move when a model is promoted, and
this script measures it. `LoopConfig.skill_floor` fires a retrain when the
champion's skill against a 30-day hour-of-day profile drops below a floor. Being
a ratio against something outside the model, it is scale-free, so one number
works for every city.

Each floor gets a full replay of its own, into a throwaway backend outside the
repository, because a changed trigger changes which models exist and therefore
the whole trajectory. Counting when a rule *would* have fired against the
existing champions cannot answer whether firing helps.

    python scripts/sweep_skill_floor.py [--city <name>|all] [--floors 0,-0.25,-0.5]

Writes outputs/skill_floor_sweep.csv, plus outputs/skill_floor_series.json with
the per-window errors behind it, so a statistic can be re-derived without
replaying anything. The `off` row reproduces the shipped numbers to the
decimal, which is the check that the harness is faithful.
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
    "city", "floor", "runs", "retrains", "promotions", "longest_silence",
    "median_rmse", "median_skill", "worst_skill", "paired", "across", "win_rate",
    "fired_by_skill",
    # What the floor is worth against leaving the trigger alone, week by week,
    # with the interval. Without these the sweep compares one median against
    # another and calls a 7% gap a result, which is the error the rest of the
    # project was corrected for on 2026-08-11. The negative result this script
    # exists to establish is worth more when it is "no detectable difference"
    # than when it is "a difference we did not test".
    "vs_off", "vs_off_lo", "vs_off_hi", "vs_off_real",
    # The same comparison over only the weeks the arm changed the serving
    # model. Most weeks it does not, and those ties dominate the median.
    "differing_weeks", "vs_off_acted", "vs_off_acted_lo", "vs_off_acted_hi",
    "vs_off_acted_real",
]


def _slug(floor: float | None) -> str:
    if floor is None:
        return "off"
    return f"{floor:+.2f}".replace(".", "p").replace("+", "plus").replace("-", "minus")


def _longest_silence(fired: np.ndarray) -> int:
    """The longest run of consecutive weeks that trained nothing.

    The measure the ratchet shows up in: a trigger that has gone permanently
    deaf has one enormous run at the end, which no average over the replay would
    make visible.
    """
    longest = current = 0
    for f in fired:
        current = 0 if f else current + 1
        longest = max(longest, current)
    return longest


def run_arm(profile: Profile, source: OpenMeteoSource, floor: float | None) -> dict:
    """One full replay at one floor, scored the way the published numbers are."""
    slug = _slug(floor)
    cfg = replace(
        profile.loop,
        skill_floor=floor,
        experiment_name=f"skill-floor-{slug}",
        registered_model_name=f"{profile.loop.registered_model_name}-floor-{slug}",
    )
    db = f"mlflow_floor_{profile.key}_{slug}.db"

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
    reasons = runs["tags.retrain_reason"].value_counts() if "tags.retrain_reason" in runs else {}

    return {
        # The per-window error series, kept so this arm can be compared against
        # the "off" arm week by week rather than median against median. Both
        # arms replay the same as_of dates, so the two series are paired by
        # construction and the pairing is what makes an interval meaningful.
        "_rmse": [float(v) for v in retro.champion_rmse],
        "_as_of": [str(d) for d in retro.as_of],
        "city": profile.label,
        "floor": "off" if floor is None else f"{floor:+.2f}",
        "runs": len(df),
        "retrains": int(df["retrain_triggered"].sum()),
        "promotions": int((df["promotion_decision"] == "promoted").sum()),
        "longest_silence": _longest_silence(df["retrain_triggered"].to_numpy(bool)),
        "median_rmse": round(float(np.nanmedian(retro.champion_rmse)), 2),
        "median_skill": round(float(np.nanmedian(skill)), 3),
        "worst_skill": round(float(np.nanmin(skill)), 3),
        "paired": round(value.get("when_it_acted", float("nan")), 2),
        "across": round(value.get("across_replay", float("nan")), 2),
        "win_rate": round(value.get("win_rate", float("nan")), 1),
        "fired_by_skill": int(reasons.get("skill", 0)) + int(reasons.get("both", 0)),
    }


def compare_against_off(city_rows: list[dict]) -> None:
    """Fill each arm's ``vs_off`` columns, in place.

    The arms of one city replay the same weeks, so an arm and the ``off`` arm
    hold two error values for every window and the comparison is paired. That is
    the same shape as champion-against-frozen elsewhere in the project, so it
    uses the same statistic and the same block resampling: consecutive monitor
    windows overlap by half, and redrawing single weeks would report a range far
    narrower than the evidence supports.

    Aligned on ``as_of`` rather than on position. A changed trigger cannot
    currently shorten a replay, but a comparison that silently pairs week 3 of
    one arm with week 4 of another would be invisible if it ever could.
    """
    baseline = next((r for r in city_rows if r["floor"] == "off"), None)
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
        "--floors", default="0,-0.25,-0.5",
        help="comma-separated skill floors to try alongside 'off' (default: 0,-0.25,-0.5)",
    )
    args = parser.parse_args()

    floors: list[float | None] = [None, *(float(f) for f in args.floors.split(","))]
    keys = list(CITIES) if args.city == "all" else [args.city]

    rows = []
    # Every backend this creates is a throwaway. Redirecting REPO_ROOT is the
    # same hook the serving tests use to relocate a registry, and it keeps a
    # sweep from leaving two dozen mlflow_*.db files in the working tree.
    #
    # ignore_cleanup_errors because MLflow keeps its SQLAlchemy engine, and so
    # its sqlite handles, open for the life of the process -- and Windows will
    # not unlink an open file. Without it the sweep completes, computes
    # everything, and then dies in the teardown having written nothing.
    original_root = tracking.REPO_ROOT
    with tempfile.TemporaryDirectory(prefix="driftloop-sweep-", ignore_cleanup_errors=True) as tmp:
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
                for floor in floors:
                    row = run_arm(profile, source, floor)
                    city_rows.append(row)
                    print(
                        f"  floor {row['floor']:>6}: {row['retrains']:>3} retrains, "
                        f"{row['promotions']:>2} promoted, longest silence "
                        f"{row['longest_silence']:>2}, median RMSE {row['median_rmse']:>6.2f}, "
                        f"retraining worth {row['across']:+.1f}% across / {row['paired']:+.1f}% paired",
                        flush=True,
                    )
                compare_against_off(city_rows)
                for row in city_rows:
                    if row["floor"] == "off":
                        continue
                    verdict = "clears zero" if row.get("vs_off_real") else "not distinguishable"
                    print(
                        f"    floor {row['floor']:>6} against off: "
                        f"{row.get('vs_off', float('nan')):+6.2f}% "
                        f"[{row.get('vs_off_lo', float('nan')):+6.2f}, "
                        f"{row.get('vs_off_hi', float('nan')):+6.2f}]  {verdict}",
                        flush=True,
                    )
                rows.extend(city_rows)
            # Written before the temp directory is torn down, so a cleanup that
            # cannot remove a locked file still leaves the results behind. An
            # hour of replays should not be lost to an unlink.
            OUTPUTS.mkdir(exist_ok=True)
            out = OUTPUTS / "skill_floor_sweep.csv"
            pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(out, index=False)
            # The per-window error series behind every figure above. Each arm is
            # a full replay costing minutes, so re-deriving a statistic from
            # them should not mean running the whole sweep a second time. It
            # already did once, when the median of paired ratios turned out to
            # be dominated by ties.
            series = OUTPUTS / "skill_floor_series.json"
            series.write_text(
                json.dumps(
                    [
                        {"city": r["city"], "floor": r["floor"],
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
