"""Is "retraining pays" a fact about the world, or about a linear model?

Every published result came from a Ridge whose regularisation is close to inert
at the shipped setting, so two readings survive: the world moved and a fresh
model tracks it, or a linear model misspecifies seasonal structure and refitting
papers over that. This script separates them by replaying a city twice under
identical settings, once with the shipped Ridge and once with a gradient-boosted
challenger flexible enough to absorb the nonlinearity.

    python scripts/ablate_model.py                    # Delhi and Los Angeles
    python scripts/ablate_model.py --city delhi

Delhi and Los Angeles by default: the city where retraining pays most and the
control where it measurably costs. If the premium is a linear artefact, Delhi's
should shrink toward Los Angeles's.

Two validity checks are printed with the result. The gradient-boosted model has
to come out the better model, or it absorbed nothing and the comparison is
between two misspecified models. And the Ridge arm has to reproduce the shipped
numbers, or the harness is not faithful.

Writes outputs/model_ablation.csv and outputs/model_ablation_series.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from driftloop import retrospect, stats, tracking
from driftloop.config import CITY_CLI_NAMES as CITIES
from driftloop.config import PROFILES, Profile
from driftloop.data import OpenMeteoSource
from driftloop.loop import bootstrap_champion, run_simulation
from driftloop.model import GBM, RIDGE

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"
DEFAULT_CITIES = ["delhi", "la"]


def run_arm(profile: Profile, source: OpenMeteoSource, kind: str) -> dict:
    """One full replay with one model class, scored the way the published numbers are."""
    cfg = replace(
        profile.loop,
        model_kind=kind,
        experiment_name=f"ablate-{kind}",
        registered_model_name=f"{profile.loop.registered_model_name}-{kind}",
    )
    db = f"mlflow_ablate_{profile.key}_{kind}.db"

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

    # A tree has no coefficients to rebuild from, so this arm is scored by
    # unpickling each registered version. Slower, and the reason the linear path
    # is the default rather than a special case.
    models = retrospect.registered_models(
        MlflowClient(), cfg.registered_model_name, load_artifacts=(kind != RIDGE)
    )
    frame = runs.copy()
    frame["champion_version"] = runs["tags.champion_version"].astype(int)
    retro = retrospect.build(
        source, frame, models, cfg.monitor_days,
        source.forecast_lead_days, cfg.promotion_margin,
    )

    value = retrospect.retraining_value(retro)
    series = retrospect.retraining_series(retro)
    premium = stats.block_bootstrap(
        (series["served_acted"], series["frozen_acted"]), stats.pct_improvement_paired
    ) if series.get("served_acted", np.array([])).size else None

    return {
        "city": profile.label,
        "kind": kind,
        "runs": len(df),
        "retrains": int(df["retrain_triggered"].sum()),
        "promotions": int((df["promotion_decision"] == "promoted").sum()),
        # Does the flexible model fit better at all? If not, it has absorbed
        # nothing and the whole comparison is void.
        "median_rmse": round(float(np.nanmedian(retro.champion_rmse)), 2),
        "acted_windows": value.get("acted_windows", 0),
        "premium": None if premium is None else round(premium.point, 2),
        "premium_lo": None if premium is None else round(premium.lo, 2),
        "premium_hi": None if premium is None else round(premium.hi, 2),
        "premium_real": None if premium is None else premium.excludes_zero,
        "_series": {k: v.tolist() for k, v in series.items()},
        "_champion_rmse": [float(v) for v in retro.champion_rmse],
        "_as_of": [str(d) for d in retro.as_of],
    }


def compare_kinds(arms: list[dict]) -> dict:
    """Ridge against gradient boosting, on the weeks both arms retrained.

    Each arm has its own retraining premium over its own acted weeks, and those
    sets are different because the two model classes promote at different times.
    Reporting only the two premiums leaves the reader comparing numbers computed
    on different windows.

    So the paired difference is taken over the weeks both arms acted on, matched
    by run date. It answers "in a week where both would have retrained, how much
    less is retraining worth to the flexible model", which is the question the
    confound poses. The count is reported because the intersection can
    be much smaller than either arm alone.
    """
    ridge = next((a for a in arms if a["kind"] == RIDGE), None)
    gbm = next((a for a in arms if a["kind"] == GBM), None)
    if ridge is None or gbm is None:
        return {}

    def ratios(arm: dict) -> dict[str, float]:
        s = arm["_series"]
        return {
            d: served / frozen
            for d, served, frozen in zip(s["acted_as_of"], s["served_acted"], s["frozen_acted"])
        }

    r_ratio, g_ratio = ratios(ridge), ratios(gbm)
    shared = sorted(set(r_ratio) & set(g_ratio))
    if len(shared) < 4:
        return {"shared_weeks": len(shared)}

    r = np.array([r_ratio[d] for d in shared])
    g = np.array([g_ratio[d] for d in shared])
    # ridge premium minus gradient-boosted premium, in percentage points.
    # Positive means retraining is worth less to the flexible model, which is
    # what the confound predicts. Negative means it is worth more.
    #
    # Signed this way round rather than as a "drop" because the premium is
    # negative in the control city, where a shrinking magnitude and a falling
    # value point in opposite directions and a column called "drop" would be
    # read backwards.
    delta = stats.block_bootstrap(
        (r, g), lambda a, b: float(np.median((1 - a) - (1 - b)) * 100)
    )
    return {
        "shared_weeks": len(shared),
        "ridge_minus_gbm": round(delta.point, 2),
        "ridge_minus_gbm_lo": round(delta.lo, 2),
        "ridge_minus_gbm_hi": round(delta.hi, 2),
        "ridge_minus_gbm_real": delta.excludes_zero,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=[*CITIES, "all"], default=None)
    args = parser.parse_args()

    keys = list(CITIES) if args.city == "all" else [args.city] if args.city else DEFAULT_CITIES

    rows, comparisons = [], []
    original_root = tracking.REPO_ROOT
    with tempfile.TemporaryDirectory(prefix="driftloop-ablate-", ignore_cleanup_errors=True) as tmp:
        tracking.REPO_ROOT = Path(tmp)
        try:
            for name in keys:
                profile = PROFILES[CITIES[name]]
                if profile.replay is None or profile.location is None:
                    continue
                source = OpenMeteoSource(profile.location)
                source.timeline()
                print(f"\n=== {profile.label} ===", flush=True)

                arms = []
                for kind in (RIDGE, GBM):
                    arm = run_arm(profile, source, kind)
                    arms.append(arm)
                    premium = (
                        "none" if arm["premium"] is None
                        else f"{arm['premium']:+6.2f}% [{arm['premium_lo']:+6.2f},"
                             f"{arm['premium_hi']:+6.2f}]"
                             f"{'  clears zero' if arm['premium_real'] else ''}"
                    )
                    print(
                        f"  {kind:>5}: median RMSE {arm['median_rmse']:>7.2f}  "
                        f"{arm['retrains']:>3} retrains, {arm['promotions']:>2} promoted, "
                        f"acted {arm['acted_windows']:>2}   premium {premium}",
                        flush=True,
                    )

                ridge, gbm = arms[0], arms[1]
                better = ridge["median_rmse"] - gbm["median_rmse"]
                print(
                    f"  gradient boosting is {abs(better):.2f} µg/m³ "
                    f"{'better' if better > 0 else 'WORSE'} than the Ridge"
                    f"{'' if better > 0 else '  <-- the ablation is void here'}",
                    flush=True,
                )

                cmp = compare_kinds(arms)
                cmp["city"] = profile.label
                comparisons.append(cmp)
                if "ridge_minus_gbm" in cmp:
                    print(
                        f"  ridge premium minus gbm premium, over the "
                        f"{cmp['shared_weeks']} weeks both retrained: "
                        f"{cmp['ridge_minus_gbm']:+6.2f} points "
                        f"[{cmp['ridge_minus_gbm_lo']:+6.2f}, "
                        f"{cmp['ridge_minus_gbm_hi']:+6.2f}]"
                        f"{'  clears zero' if cmp['ridge_minus_gbm_real'] else ''}",
                        flush=True,
                    )
                else:
                    print(f"  too few shared weeks to compare: {cmp.get('shared_weeks', 0)}",
                          flush=True)
                rows.extend(arms)

            OUTPUTS.mkdir(exist_ok=True)
            frame = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                                  for r in rows])
            frame.to_csv(OUTPUTS / "model_ablation.csv", index=False)
            (OUTPUTS / "model_ablation_series.json").write_text(
                json.dumps(
                    {"arms": [{"city": r["city"], "kind": r["kind"], "as_of": r["_as_of"],
                               "champion_rmse": r["_champion_rmse"]} for r in rows],
                     "comparisons": comparisons},
                    indent=1,
                ),
                encoding="utf-8",
            )
            print(f"\nwrote {OUTPUTS / 'model_ablation.csv'}")
        finally:
            tracking.REPO_ROOT = original_root


if __name__ == "__main__":
    main()
