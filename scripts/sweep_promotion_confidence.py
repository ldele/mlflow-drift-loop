"""Does making the promotion gate uncertainty-aware pay?

The three mechanisms measured before this one all changed when the loop trains a
challenger. None of them touched how it decides to ship one, and that decision is
the project's own standard applied everywhere except to itself: the gate compares
two RMSEs on a single seven-day window of autocorrelated hourly data and promotes
on the bare difference.

Measured over the 28 shipped promotions, blocking the holdout window in 24-hour
blocks and resampling the exam margin: 11 have an interval reaching below the 5%
margin they were supposed to clear, and 6 reach below zero. Los Angeles promoted
on +21.9% [-6.9, +32.3] and delivered -6.7%, and that single promotion is the
whole "retraining costs 13.4% here" result.

The width of that interval also tracks which cities retraining works in. Median
width by city: Johannesburg 4.9 points, Delhi 5.7, Santiago 6.7, against Kraków
8.8, Melbourne 15.1, Los Angeles 39.2. Where the seasonal swing is large the
exam has signal to select on; where the air is flat the difference between two
models over a week is mostly noise, and selecting on it is a coin flip with a
winner's curse attached.

`LoopConfig.promotion_confidence` makes the challenger clear `promotion_margin`
at the lower bound as well as at the point estimate. Strictly an additional
hurdle, so it can only ever remove a promotion.

    python scripts/sweep_promotion_confidence.py [--city <name>|all]
        [--confidences 0.8,0.9,0.95]

Each confidence gets a full replay of its own into a throwaway backend, because
a blocked promotion changes which models exist and therefore the whole
trajectory. The `off` row reproduces the shipped numbers to the decimal.

Writes outputs/promotion_confidence_sweep.csv and
outputs/promotion_confidence_series.json.

The question is not "does this block bad promotions", which it does by
construction. It is whether the promotions it blocks are worse than the ones it
keeps. A gate that blocks indiscriminately costs the cities where retraining
pays, and the `blocked` and per-city columns are there to price that.
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
    "city", "confidence", "runs", "retrains", "promotions",
    # How many challengers cleared the point estimate and were stopped by the
    # interval. The whole intervention, counted.
    "blocked",
    # The gate's own honesty, on the promotions that survived. If the mechanism
    # works, the survivors should deliver closer to what they promised.
    "gate_n", "gate_harmful", "gate_exam", "gate_delivered", "gate_shrinkage",
    "median_rmse", "median_skill", "paired", "across", "win_rate",
    "vs_off", "vs_off_lo", "vs_off_hi", "vs_off_real",
    "differing_weeks", "vs_off_acted", "vs_off_acted_lo", "vs_off_acted_hi",
    "vs_off_acted_real",
]


def _slug(confidence: float | None) -> str:
    return "off" if confidence is None else f"{confidence:.3f}".rstrip("0").replace(".", "p")


def run_arm(profile: Profile, source: OpenMeteoSource, confidence: float | None) -> dict:
    """One full replay at one confidence, scored the way the published numbers are."""
    slug = _slug(confidence)
    cfg = replace(
        profile.loop,
        promotion_confidence=confidence,
        experiment_name=f"gate-{slug}",
        registered_model_name=f"{profile.loop.registered_model_name}-gate-{slug}",
    )
    db = f"mlflow_gate_{profile.key}_{slug}.db"

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

    # Challengers that beat the margin on the point estimate and were stopped by
    # the interval. Read off the run frame rather than counted in the loop, so
    # the number reflects what was logged rather than what was intended.
    blocked = 0
    if confidence is not None and "metrics.exam_margin_lo" in runs:
        judged = runs.dropna(subset=["metrics.challenger_rmse"])
        point = 1 - judged["metrics.challenger_rmse"] / judged["metrics.champion_rmse_holdout"]
        blocked = int(
            (
                (point > cfg.promotion_margin)
                & (judged["tags.promotion_decision"] == "rejected")
            ).sum()
        )

    models = retrospect.registered_models(MlflowClient(), cfg.registered_model_name)
    frame = runs.copy()
    frame["champion_version"] = runs["tags.champion_version"].astype(int)
    retro = retrospect.build(
        source, frame, models, cfg.monitor_days,
        source.forecast_lead_days, cfg.promotion_margin,
    )
    value = retrospect.retraining_value(retro)
    skill = np.asarray(retro.champion_skill, dtype=float)

    exam = np.array([g["exam_margin"] for g in retro.gate], dtype=float) * 100
    delivered = np.array([g["delivered_margin"] for g in retro.gate], dtype=float) * 100
    harmful = int(sum(1 for g in retro.gate if g["delivered_margin"] < 0))

    return {
        "_rmse": [float(v) for v in retro.champion_rmse],
        "_as_of": [str(d) for d in retro.as_of],
        "city": profile.label,
        "confidence": "off" if confidence is None else f"{confidence:.2f}",
        "runs": len(df),
        "retrains": int(df["retrain_triggered"].sum()),
        "promotions": int((df["promotion_decision"] == "promoted").sum()),
        "blocked": blocked,
        "gate_n": len(retro.gate),
        "gate_harmful": harmful,
        "gate_exam": round(float(np.mean(exam)), 2) if exam.size else float("nan"),
        "gate_delivered": round(float(np.mean(delivered)), 2) if delivered.size else float("nan"),
        # Promise minus delivery, which is the winner's curse in one number. If
        # the gate is selecting on noise this should shrink as the bar tightens.
        "gate_shrinkage": (
            round(float(np.mean(exam) - np.mean(delivered)), 2) if exam.size else float("nan")
        ),
        "median_rmse": round(float(np.nanmedian(retro.champion_rmse)), 2),
        "median_skill": round(float(np.nanmedian(skill)), 3),
        "paired": round(value.get("when_it_acted", float("nan")), 2),
        "across": round(value.get("across_replay", float("nan")), 2),
        "win_rate": round(value.get("win_rate", float("nan")), 1),
    }


def compare_against_off(city_rows: list[dict]) -> None:
    """Fill each arm's ``vs_off`` columns, in place. See sweep_recertify.py."""
    baseline = next((r for r in city_rows if r["confidence"] == "off"), None)
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
        "--confidences", default="0.8,0.9,0.95",
        help="comma-separated one-sided confidences to try alongside 'off'",
    )
    args = parser.parse_args()

    confidences: list[float | None] = [None, *(float(c) for c in args.confidences.split(","))]
    keys = list(CITIES) if args.city == "all" else [args.city]

    rows = []
    original_root = tracking.REPO_ROOT
    with tempfile.TemporaryDirectory(prefix="driftloop-gate-", ignore_cleanup_errors=True) as tmp:
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
                for confidence in confidences:
                    row = run_arm(profile, source, confidence)
                    city_rows.append(row)
                    print(
                        f"  {row['confidence']:>4}: {row['promotions']:>2} promoted "
                        f"({row['blocked']:>2} blocked), median RMSE {row['median_rmse']:>6.2f}, "
                        f"gate promised {row['gate_exam']:>6.1f}% delivered "
                        f"{row['gate_delivered']:>6.1f}% (shrinkage {row['gate_shrinkage']:>5.1f}), "
                        f"{row['gate_harmful']}/{row['gate_n']} harmful",
                        flush=True,
                    )
                compare_against_off(city_rows)
                for row in city_rows:
                    if row["confidence"] == "off":
                        continue
                    verdict = "clears zero" if row.get("vs_off_real") else "not distinguishable"
                    print(
                        f"    {row['confidence']:>4} against off: "
                        f"{row.get('vs_off', float('nan')):+6.2f}% "
                        f"[{row.get('vs_off_lo', float('nan')):+6.2f}, "
                        f"{row.get('vs_off_hi', float('nan')):+6.2f}]  {verdict}"
                        f"   over {row.get('differing_weeks', 0)} changed weeks: "
                        f"{row.get('vs_off_acted', float('nan')):+6.2f}%",
                        flush=True,
                    )
                rows.extend(city_rows)
            OUTPUTS.mkdir(exist_ok=True)
            out = OUTPUTS / "promotion_confidence_sweep.csv"
            pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(out, index=False)
            series = OUTPUTS / "promotion_confidence_series.json"
            series.write_text(
                json.dumps(
                    [
                        {"city": r["city"], "confidence": r["confidence"],
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
