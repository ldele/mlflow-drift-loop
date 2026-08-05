"""Drift-loop dashboard.

Reads straight from the MLflow backend, so it shows whatever the last
`python scripts/run_simulation.py` produced. Per-run detail (feature
distributions, champion predictions) comes from artifacts each run logs, so the
detail panels don't touch the data generator -- they keep working in Phase 2.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import theme  # noqa: E402
from driftloop import tracking  # noqa: E402
from driftloop.config import (  # noqa: E402
    CITY_CLI_NAMES,
    DRIFT_FEATURES,
    FEATURES,
    PROFILES,
    TARGET,
)
from driftloop.drift import PSI_SIGNIFICANT, PSI_STABLE  # noqa: E402
from driftloop.model import build_pipeline, train as train_model  # noqa: E402
from driftloop import retrospect  # noqa: E402

st.set_page_config(page_title="Drift loop", page_icon="~", layout="wide")

# What each city's replay is a story *about*. Keyed by profile so a new city
# without an entry degrades to the generic story rather than borrowing another
# city's — or, worse, the live schedule's. All six are filled: three of them were
# missing, and a Melbourne page silently reading "here it's a summer-trained
# model walking into the winter heating season" is worse than no story at all.
CITY_STORY = {
    "openmeteo": " Here it's a summer-trained model walking into the winter heating season, "
    "when basin inversions over a coal-heated valley drive PM2.5 up several-fold, then "
    "back out the other side. That second half of the year is what exposed the retrain rule.",
    "openmeteo_delhi": " Here it's a monsoon-trained model walking into the post-monsoon "
    "burning season, which triples Delhi's PM2.5, the most violent of the six.",
    "openmeteo_la": " Los Angeles is the quiet one: barely a season at all, so the champion "
    "mostly holds and the loop mostly declines to retrain. It is the control, and it earns "
    "its place by failing: there has to be drift for a drift loop to be worth anything.",
    "openmeteo_santiago": " Santiago is Kraków's twin, half a year out of phase: the same "
    "basin trapping the same winter inversions, walked into from a southern-summer-trained "
    "model. Same thresholds, opposite season, which is how you can tell the loop has no "
    "calendar in it.",
    "openmeteo_joburg": " Johannesburg is where the promotion gate does its most visible "
    "work. Highveld coal smoke takes the error from 12 to 82 µg/m³, eleven retrains fire, "
    "and only three challengers clear the 5% margin. The other eight are trained and "
    "thrown away.",
    "openmeteo_melbourne": " Melbourne stays near the WHO guideline all year and its model "
    "still decays to more than four times its training error, which is the case against "
    "reading clean air as a stable model.",
}


# --------------------------------------------------------------------------- #
# Data access (keyed by profile so the cache is per-backend)                  #
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=10)
def load_runs(db_filename: str, experiment: str) -> pd.DataFrame:
    tracking.setup(experiment, db_filename)
    df = mlflow.search_runs(experiment_names=[experiment], order_by=["attributes.start_time ASC"])
    if df.empty or "tags.cycle_type" not in df:
        return pd.DataFrame()
    df = df[df["tags.cycle_type"] == "monitor"].copy()
    if df.empty:
        return df
    df["as_of"] = pd.to_datetime(df["params.as_of"])
    return df.sort_values("as_of").reset_index(drop=True)


@st.cache_data(ttl=10)
def load_versions(db_filename: str, experiment: str, model: str) -> pd.DataFrame:
    tracking.setup(experiment, db_filename)
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model}'")
        # search_model_versions doesn't reliably populate aliases; read the
        # authoritative alias -> version map off the registered model itself.
        alias_map = client.get_registered_model(model).aliases or {}
    except Exception:
        return pd.DataFrame()

    version_alias: dict[str, list[str]] = {}
    for alias, ver in alias_map.items():
        version_alias.setdefault(str(ver), []).append(alias)

    rows = []
    for mv in versions:
        row = {
            "version": int(mv.version),
            "alias": ", ".join(version_alias.get(str(mv.version), [])),
            "train_start": pd.to_datetime(mv.tags.get("train_start")),
            "train_end": pd.to_datetime(mv.tags.get("train_end")),
            "baseline_rmse": float(mv.tags.get("baseline_rmse", "nan")),
            "n_rows": int(mv.tags.get("n_rows", 0)),
        }
        for name in [*FEATURES, "intercept"]:
            row[f"coef_{name}"] = float(mv.tags.get(f"coef_{name}", "nan"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("version").reset_index(drop=True)


@st.cache_data(ttl=60)
def load_run_meta(meta_filename: str) -> dict:
    path = REPO_ROOT / "outputs" / meta_filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifact(db_filename: str, run_id: str, rel: str) -> Path:
    """Locate a run artifact. Prefer MLflow's resolver; fall back to the local
    artifact tree when the stored URI is absolute for another machine — e.g. a
    backend generated on the CI runner and pulled down here (the Scheduled
    profile). The run_id is the on-disk artifact directory name."""
    try:
        return Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=rel))
    except Exception:
        local = tracking._artifact_dir(db_filename) / run_id / "artifacts" / rel
        if local.exists():
            return local
        raise


@st.cache_data(ttl=30)
def load_monitoring(db_filename: str, experiment: str, run_id: str) -> tuple[pd.DataFrame, dict]:
    """Per-run predictions CSV + feature-distribution JSON, resilient to
    cross-machine absolute artifact paths."""
    tracking.setup(experiment, db_filename)
    preds = pd.read_csv(
        _resolve_artifact(db_filename, run_id, "monitoring/monitor_predictions.csv"),
        parse_dates=["timestamp"],
    )
    report = json.loads(
        _resolve_artifact(db_filename, run_id, "monitoring/feature_distributions.json").read_text(
            encoding="utf-8"
        )
    )
    return preds, report


@st.cache_data(ttl=30)
def load_retrospective(db_filename: str, experiment: str, profile_key: str):
    """Every registered version scored on every monitoring window.

    Needs to re-read old windows, so it exists only for the profiles whose data
    is a fixed cached span (see ``replayable_source``). Returns None elsewhere
    and the panels that depend on it say so rather than half-rendering.
    """
    from driftloop.data import replayable_source

    profile = PROFILES[profile_key]
    source = replayable_source(profile)
    if source is None:
        return None
    cfg = profile.loop
    tracking.setup(experiment, db_filename)
    runs = load_runs(db_filename, experiment)
    if runs.empty or "tags.champion_version" not in runs:
        return None
    models = retrospect.registered_models(MlflowClient(), cfg.registered_model_name)
    if not models:
        return None
    frame = runs.copy()
    frame["champion_version"] = runs["tags.champion_version"].astype(int)
    return retrospect.build(
        source, frame, models, cfg.monitor_days,
        profile.location.forecast_lead_days, cfg.promotion_margin,
    )


@st.cache_data(ttl=30)
def load_training_band(db_filename: str, experiment: str, profile_key: str):
    """The range each feature held while the *first* champion was trained.

    Drawn behind the feature series, this states the covariate-drift claim
    physically rather than statistically: the band is what the model was shown,
    the line is what the world did afterwards, and a line leaving its band is
    the case for retraining before any statistic is computed.
    """
    from driftloop.data import replayable_source

    profile = PROFILES[profile_key]
    source = replayable_source(profile)
    if source is None:
        return None
    tracking.setup(experiment, db_filename)
    models = retrospect.registered_models(MlflowClient(), profile.loop.registered_model_name)
    if not models:
        return None
    bootstrap = models[min(models)]  # the version everything else drifted away from
    return (
        retrospect.training_window_stats(source, bootstrap),
        bootstrap.train_start,
        bootstrap.train_end,
    )


@st.cache_data(ttl=30)
def load_importance(db_filename: str, experiment: str, profile_key: str, version: int, as_of: str):
    """How far the prediction moves per 1-sd move in each feature, in µg/m³.

    The comparable version of the coefficient panels. A slope per hPa and a
    slope per W/m² answer different questions, so the largest raw coefficient is
    usually just the feature with the smallest units; scaling each by the
    feature's own spread over a real window puts them all in µg/m³.
    """
    from driftloop.data import replayable_source

    profile = PROFILES[profile_key]
    source = replayable_source(profile)
    if source is None:
        return None
    tracking.setup(experiment, db_filename)
    models = retrospect.registered_models(MlflowClient(), profile.loop.registered_model_name)
    model = models.get(int(version))
    if model is None:
        return None
    # Taken as a string so the cache key is a plain scalar rather than a
    # Timestamp Streamlit has to reach for pickle to hash.
    end = pd.Timestamp(as_of)
    window = source.get_data(end - pd.Timedelta(profile.loop.monitor_days, unit="D"), end)
    if window.empty:
        return None
    ranked = model.importance(window)
    # The two clock terms are folded into one entry: separately they are half a
    # daily cycle each and neither number means anything on its own.
    rows = [(f, v) for f, v in ranked.items() if f in DRIFT_FEATURES]
    rows.append(("hour of day", sum(v for f, v in ranked.items() if f not in DRIFT_FEATURES)))
    rows.sort(key=lambda kv: kv[1], reverse=True)
    return [f for f, _ in rows], [round(v, 3) for _, v in rows]


def champion_age_days(runs: pd.DataFrame) -> tuple[int | None, str | None]:
    """How long the version currently serving has been the one serving."""
    if "tags.champion_version" not in runs or runs.empty:
        return None, None
    versions = runs["tags.champion_version"].tolist()
    current = versions[-1]
    since = len(versions) - 1
    while since > 0 and versions[since - 1] == current:
        since -= 1
    return int((runs["as_of"].iloc[-1] - runs["as_of"].iloc[since]).days), current


def psi_status(value: float) -> tuple[str, str]:
    if value > PSI_SIGNIFICANT:
        return "significant", theme.CRITICAL
    if value > PSI_STABLE:
        return "moderate", theme.WARNING
    return "stable", theme.GOOD


# --------------------------------------------------------------------------- #
# Profile selector: Phase 1 (synthetic) vs Phase 2 (Open-Meteo)               #
# --------------------------------------------------------------------------- #
profile_key = st.sidebar.radio(
    "Data source",
    options=list(PROFILES),
    format_func=lambda k: PROFILES[k].label,
    index=0,
)
# What this app is *for*, said once. The published site is the argument, made
# across all six cities at once and fixed at build time; this is the operator's
# view of one backend, reading whatever the last run left in it, and it carries
# the raw material the site distils away — every run as logged, every registered
# version, the per-run distributions and residuals.
st.sidebar.caption(
    "The operator's view: one city's MLflow backend, live, as the last run left it. "
    "[The published site](https://ldele.github.io/mlflow-drift-loop/) is the report: "
    "all six cities on shared axes, fixed at build time."
)
PROFILE = PROFILES[profile_key]
CFG = PROFILE.loop
DB = PROFILE.db_filename
IS_SYNTHETIC = profile_key == "synthetic"

runs = load_runs(DB, CFG.experiment_name)
meta = load_run_meta(PROFILE.meta_filename)
drift_date = meta.get("drift_date")

st.title("Air quality drift watch")
if IS_SYNTHETIC:
    st.caption("track → detect drift → retrain challenger → promote &nbsp;·&nbsp; Phase 1, synthetic data")
else:
    st.caption(
        "track → detect drift → retrain challenger → promote &nbsp;·&nbsp; "
        f"Phase 2, real weather + air quality · {meta.get('location', 'Open-Meteo')}"
    )

if runs.empty:
    if profile_key == "scheduled":
        st.warning(
            "No scheduled runs yet. The weekly GitHub Action populates this profile. "
            "Locally: `python scripts/run_scheduled.py --as-of YYYY-MM-DD`."
        )
        st.stop()
    # Phase 1/2 demo data isn't committed, so on a fresh host (e.g. Streamlit
    # Cloud) it's absent. Offer to generate it in place — a no-op wherever the
    # backend already has runs.
    script = "run_simulation.py" if IS_SYNTHETIC else "run_openmeteo.py"
    est = "~30s" if IS_SYNTHETIC else "~1 min · fetches Open-Meteo"
    # Scope the run to the city being asked for. Without this the button would
    # rebuild every city, so asking for Delhi would also re-run Kraków and LA.
    city_args = [
        f"--city={name}"
        for name, key in CITY_CLI_NAMES.items()
        if key == profile_key
    ]
    st.warning(f"No data for **{PROFILE.label}** yet.")
    if st.button(f"Generate it now ({est})", type="primary"):
        with st.spinner("Running the pipeline, which populates the MLflow backend…"):
            # The subprocess is a fresh interpreter; ensure it can import the
            # package from src/ even where the project isn't pip-installed (Cloud).
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / script), "--fresh", *city_args],
                check=True,
                cwd=str(REPO_ROOT),
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
            )
        st.cache_data.clear()
        st.rerun()
    st.stop()

promoted = runs[runs["tags.promotion_decision"] == "promoted"]
retrained = runs[runs["tags.retrain_triggered"] == "True"]
latest = runs.iloc[-1]
run_by_as_of = dict(zip(runs["as_of"], runs["run_id"]))
st.caption(
    f"{len(runs)} weekly runs, {runs['as_of'].min():%b %Y} to {runs['as_of'].max():%b %Y}."
)

retro = load_retrospective(DB, CFG.experiment_name, profile_key)
age_days, serving_version = champion_age_days(runs)
# What retraining was worth, both ways. The paired reading is the headline: it
# is the one the project's own evaluation says to trust, and it is what the loop
# is *for*, so it belongs in the tile row rather than four charts down.
value = retrospect.retraining_value(retro) if retro else {}

# Three of these used to be counts of what the machine did, which says nothing
# about whether any of it worked, and two of them had to be subtracted from each
# other to reach the number that matters. Lead with health and freshness instead.
c1, c2, c3, c4, c5 = st.columns(5)
if retro and retro.champion_skill and not np.isnan(retro.champion_skill[-1]):
    skill = retro.champion_skill[-1]
    c1.metric(
        "Skill, latest window",
        f"{skill:+.0%}",
        f"vs. a {retrospect.CLIMATOLOGY_DAYS}-day daily profile",
        delta_color="off",
    )
else:
    c1.metric(
        "Latest champion R²",
        f"{latest['metrics.champion_r2']:.2f}" if "metrics.champion_r2" in latest else "—",
        f"RMSE {latest['metrics.champion_rmse']:.2f}",
        delta_color="off",
    )
c2.metric(
    "Model in service",
    f"{age_days} days" if age_days is not None else "—",
    f"v{serving_version}" if serving_version else None,
    delta_color="off",
)
c3.metric(
    "Challengers shipped",
    f"{len(promoted)}/{len(retrained)}",
    f"{len(retrained) - len(promoted)} rejected by the gate",
    delta_color="off",
)
c4.metric(
    "Latest max PSI",
    f"{latest['metrics.data_drift_psi']:.2f}",
    # The label is "significant" on essentially every run of every city, so as a
    # delta it is a constant dressed as a status. The worst feature does
    # varies and says which ingredient moved.
    f"worst: {latest['tags.worst_feature']}" if "tags.worst_feature" in latest else None,
    delta_color="off",
)
# "Weeks watched" was here, and it is a count of what the machine did rather
# than a verdict on whether it worked — the same objection that retired three of
# the other tiles. The count still appears in the caption above and in the Runs
# tab. This is the number the whole loop exists to move.
if "when_it_acted" in value:
    c5.metric(
        "Retraining was worth",
        f"{value['when_it_acted']:+.1f}%",
        f"over {value['acted_windows']} weeks it was serving · won {value['win_rate']:.0f}%",
        delta_color="off",
    )
else:
    c5.metric(
        "Retraining was worth",
        "—",
        "nothing promoted yet" if retro else "needs a replayable city",
        delta_color="off",
    )

tab_loop, tab_dist, tab_model, tab_sweep, tab_registry, tab_table = st.tabs(
    ["Drift loop", "Feature drift", "Model", "Knob sweep", "Registry", "Runs"]
)

# --------------------------------------------------------------------------- #
# Tab: the drift loop story                                                   #
# --------------------------------------------------------------------------- #
with tab_loop:
    story = (
        "**The story, top to bottom:** the world drifts away from what the champion "
        "was trained on, the champion's error climbs past its threshold, a challenger "
        "is trained and judged on a window neither model has seen, and it takes over."
    )
    if profile_key == "synthetic":
        story += " The shaded band is everything after the engineered regime shift."
    elif profile_key == "scheduled":
        story += (
            " Each point is one **scheduled run** appended over calendar time: the live "
            "loop accruing its own history, one cron fire at a time (Phase 3)."
        )
    else:  # one of the cities
        story += CITY_STORY.get(profile_key, "")
    st.markdown(story)

    fig = theme.base_figure("Data drift: PSI per feature against the champion's training window", "PSI")
    theme.drift_region(fig, drift_date, runs["as_of"].max())
    # DRIFT_FEATURES, not FEATURES: the two cyclical hour terms are fitted but
    # carry no PSI, because every monitoring window contains all 24 hours and
    # their distribution is fixed by construction.
    for feature in DRIFT_FEATURES:
        col = f"metrics.psi_{feature}"
        if col in runs:
            theme.line(fig, runs["as_of"], runs[col], feature, theme.FEATURE_COLOR[feature])
    theme.threshold(fig, runs["as_of"], PSI_SIGNIFICANT, f"significant ({PSI_SIGNIFICANT})")
    st.plotly_chart(fig, width="stretch")

    # The trigger in µg/m³ rather than as a ratio. `error ÷ baseline` hides that
    # the denominator is reset on every promotion; since retrains fire in the
    # dirty season, each champion inherits a higher bar than the one it replaced
    # and the bar never comes back down. Same units on one axis makes the
    # staircase the obvious feature rather than a footnote.
    fig = theme.base_figure(
        "The retrain trigger: champion error against the bar it must cross", "µg/m³"
    )
    theme.drift_region(fig, drift_date, runs["as_of"].max())
    fig.add_scatter(
        x=runs["as_of"], y=runs["metrics.champion_baseline_rmse"] * CFG.perf_drift_threshold,
        name=f"bar ({CFG.perf_drift_threshold}× baseline)", mode="lines",
        line=dict(color=theme.CRITICAL, width=2, dash="dot", shape="hv"),
    )
    theme.line(fig, runs["as_of"], runs["metrics.champion_rmse"], "champion error", theme.SERIES[0])
    if not retrained.empty:
        theme.events(
            fig, retrained["as_of"], retrained["metrics.champion_rmse"],
            "retrain triggered", theme.WARNING, symbol="circle",
        )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Where the staircase ends up far above the error, the trigger has gone quiet and cannot "
        "fire again whatever the model does. The bar was set at the seasonal peak and never comes "
        "back down."
    )

    if retro and retro.champion_skill:
        fig = theme.base_figure(
            f"What the model is worth: skill against a {retrospect.CLIMATOLOGY_DAYS}-day daily profile",
            "skill",
        )
        theme.drift_region(fig, drift_date, runs["as_of"].max())
        theme.line(fig, runs["as_of"], retro.champion_skill, "skill", theme.SERIES[2])
        theme.threshold(fig, runs["as_of"], 0.0, "0 · no better than the baseline")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Scale-free, and its yardstick holds still when a model is promoted, which is what "
            "the retrain ratio above cannot say. The baseline sees recent pollution readings and "
            "the model never does, so it is a hard bar rather than a fair fight."
        )

        # One curve per promoted version, lined up on the day it took over. The
        # logged champion error is a single line across every champion, so no
        # individual model's decay is visible in it.
        if retro.promoted_at:
            fig = theme.base_figure(
                "How each model ages: every promoted version, from the day it took over",
                "skill",
            )
            # Read off the runs rather than assumed: the replay cadence is a
            # config knob, so hard-coding seven would mislabel a different one.
            step_days = (retro.as_of[1] - retro.as_of[0]).days if len(retro.as_of) > 1 else 7
            for version, promoted_on in sorted(retro.promoted_at.items()):
                start = retro.as_of.index(promoted_on) + 1
                skills = retro.version_skill[version][start:]
                if not skills:
                    continue
                serving = str(version) == str(serving_version)
                fig.add_scatter(
                    x=[i * step_days / 7 for i in range(len(skills))], y=skills,
                    name=f"v{version}", mode="lines",
                    line=dict(
                        color=theme.SERIES[1] if serving else theme.SERIES[0],
                        width=2.8 if serving else 1.6,
                    ),
                    opacity=1.0 if serving else 0.45,
                )
            fig.update_xaxes(title=dict(text="weeks in service", font=dict(color=theme.MUTED, size=11)))
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Followed past each model's own retirement, so a line that keeps falling shows what "
                "keeping it would have cost. The version currently serving is drawn solid."
            )

    fig = theme.base_figure("Champion vs. challenger on the held-out window (RMSE, lower is better)", "RMSE")
    # The column only exists once at least one challenger has been trained; on a
    # profile with only "none" cycles (e.g. a fresh Scheduled backend) it's absent.
    if "metrics.challenger_rmse" in runs.columns:
        judged = runs.dropna(subset=["metrics.challenger_rmse"])
    else:
        judged = runs.iloc[:0]
    if judged.empty:
        st.info("No challenger has been trained yet: performance drift never crossed the threshold.")
    else:
        theme.drift_region(fig, drift_date, runs["as_of"].max())
        theme.line(fig, judged["as_of"], judged["metrics.champion_rmse_holdout"], "champion", theme.SERIES[0])
        theme.line(fig, judged["as_of"], judged["metrics.challenger_rmse"], "challenger", theme.SERIES[1])
        if not promoted.empty:
            theme.events(
                fig, promoted["as_of"], promoted["metrics.challenger_rmse"], "promoted", theme.GOOD
            )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Being newer is not a qualification. Both models sit the same exam, a week of air "
            "neither has seen, and the challenger takes the job only by more than the "
            f"{CFG.promotion_margin:.0%} margin. Whether passing it predicts anything is the "
            "next question."
        )

    # Does passing the exam predict anything? The margin the gate decided on
    # cannot also be the evidence it decided well, so this is the out-of-sample
    # check: what each winner delivered against the model it displaced, scored
    # on the windows it went on to serve.
    if retro and retro.gate:
        st.markdown("#### Did the exam predict anything?")
        st.plotly_chart(
            theme.gate_scatter(retro.gate, retrospect.GATE_LONG_WEEKS), width="stretch"
        )
        summary = retrospect.gate_summary(retro.gate)
        short, long = summary["short"], summary["long"]
        parts = []
        if short["n"]:
            parts.append(
                f"Over {short['n']} promotion{'s' if short['n'] != 1 else ''} that served under "
                f"{retrospect.GATE_LONG_WEEKS} weeks the exam promised {short['exam']:+.1%} and "
                f"delivered {short['delivered']:+.1%}, with {short['harmful']} of {short['n']} "
                "leaving the city worse off."
            )
        if long["n"]:
            parts.append(
                f"Over {long['n']} that served {retrospect.GATE_LONG_WEEKS} weeks or more it "
                f"promised {long['exam']:+.1%} and delivered {long['delivered']:+.1%}, with "
                f"{long['harmful']} of {long['n']} harmful."
            )
        st.caption(
            " ".join(parts)
            + " A seven-day exam certifies a model for about a month. The models that end up "
            "serving half a year are the ones the ratcheted trigger above can no longer replace, "
            "so the two faults compound."
        )
        if len(retro.gate) < 4:
            st.caption(
                f"Only {len(retro.gate)} promotions here, which is too few to read as "
                "calibration on its own. The pooled version across all six cities is on the "
                "published site."
            )

# --------------------------------------------------------------------------- #
# Tab: feature distributions (the "why" behind PSI)                           #
# --------------------------------------------------------------------------- #
with tab_dist:
    # The physical story first, because PSI is a summary of it and not the other
    # way round. "0.25, significant" is a number nobody can picture; "fourteen
    # degrees colder than anything the model was shown" is the actual argument.
    band = load_training_band(DB, CFG.experiment_name, profile_key)
    if retro and retro.feature_means and band:
        stats, train_start, train_end = band
        st.markdown("#### What changed in the world")
        st.markdown(
            "Each weather ingredient averaged over every monitoring window, in the units it "
            "is measured in. The shaded band is the middle 80% of the *hourly* values that "
            f"ingredient held while the first model was trained ({train_start:%Y-%m-%d} to "
            f"{train_end:%Y-%m-%d}), a percentile range rather than the full one, so a "
            "single freak hour cannot widen it to cover everything. A line leaving its band "
            "means the model is being asked about conditions it was never shown, which is the "
            "case for retraining before any statistic is computed."
        )
        st.plotly_chart(
            theme.factor_small_multiples(
                retro.as_of, retro.feature_means, stats, DRIFT_FEATURES
            ),
            width="stretch",
        )
        st.caption(
            "Read the bands as a rough guide rather than a test. They are hourly and the "
            "lines are two-week means, so anything with a large day-to-night swing, radiation "
            "most of all, gets a band far wider than a mean could ever leave. "
            "Temperature is where the comparison bites: Kraków's spends the whole winter "
            "below everything the first model was trained on."
        )
        st.divider()

    st.markdown("#### The distributions PSI is summarising")
    st.markdown(
        "How each feature's distribution in one window (filled) has moved away from "
        "the champion's training window (outline). This is what a data-drift "
        "monitor like Evidently shows, logged as an artifact every run."
    )
    options = list(runs["as_of"])
    picked = st.selectbox(
        "Run (as-of date)", options, index=len(options) - 1,
        format_func=lambda d: d.strftime("%Y-%m-%d"), key="dist_run",
    )
    _, report = load_monitoring(DB, CFG.experiment_name, run_by_as_of[picked])

    # Only what this run's artifact actually carries. A report is written once,
    # at the time of the run, and the feature list has grown since — the Live
    # schedule's backend was populated when the model had three features, so
    # indexing DRIFT_FEATURES into it raised a KeyError and took the whole app
    # down on that profile. An old run should render what it has and say what it
    # does not, which is also what will happen to today's runs later.
    shown = [f for f in DRIFT_FEATURES if f in report]
    missing = [f for f in DRIFT_FEATURES if f not in report]
    if missing:
        st.info(
            f"This run predates {', '.join(missing)}. Its drift report was written when the "
            f"model had {len(shown)} features, and reports are not rewritten after the fact."
        )

    # Three panels a row rather than one row of six: at six across, each
    # histogram is too narrow to read the shift the PSI number is summarising.
    per_row = 3
    for start in range(0, len(shown), per_row):
        chunk = shown[start : start + per_row]
        for col, feature in zip(st.columns(per_row), chunk):
            entry = report[feature]
            label, color_ = psi_status(entry["psi"])
            with col:
                st.markdown(f"**{feature}**")
                st.markdown(
                    f"<span style='color:{color_};font-weight:600'>PSI {entry['psi']:.2f} · {label}</span>"
                    f"<br><span style='color:{theme.MUTED};font-size:0.85em'>"
                    f"mean {entry['reference_mean']:.1f} → {entry['current_mean']:.1f}</span>",
                    unsafe_allow_html=True,
                )
                fig = theme.hist_overlay(
                    entry["edges"], entry["reference_counts"], entry["current_counts"],
                    theme.FEATURE_COLOR[feature],
                )
                st.plotly_chart(fig, width="stretch", key=f"hist_{feature}")

# --------------------------------------------------------------------------- #
# Tab: the model itself                                                        #
# --------------------------------------------------------------------------- #
with tab_model:
    versions = load_versions(DB, CFG.experiment_name, CFG.registered_model_name)

    # --- what the model actually is, read out of the code rather than retyped ---
    _pipeline = build_pipeline()
    _val_fraction = inspect.signature(train_model).parameters["val_fraction"].default
    _estimator = " → ".join(type(step).__name__ for _, step in _pipeline.steps)

    st.markdown("#### The model")
    spec_col, param_col = st.columns(2)
    with spec_col:
        st.markdown(
            f"""
| | |
|---|---|
| estimator | `{_estimator}(alpha={_pipeline.named_steps["ridge"].alpha:g})` |
| features | `{"`, `".join(FEATURES)}` |
| target | `{TARGET}` (µg/m³) |
| training | chronological {(1 - _val_fraction):.0%}/{_val_fraction:.0%} tail split, refit on the full window |
| baseline | RMSE on the held-out tail, never a random split |
"""
        )
        st.caption(
            "Ridge on six weather features plus the hour of day, kept simple on purpose: "
            "a small model decays visibly when the relationship shifts, where a larger one "
            "would absorb some of the drift and hide it."
        )
    with param_col:
        st.markdown(
            f"""
| | |
|---|---|
| monitor window | `{CFG.monitor_days} days` |
| challenger training | `{CFG.challenger_train_days} days` |
| holdout | `{CFG.holdout_days} days` |
| retrain trigger | `error ÷ baseline > {CFG.perf_drift_threshold}` |
| PSI significant | `> {CFG.psi_threshold}` (stable < {PSI_STABLE}) |
| promotion margin | `> {CFG.promotion_margin:.0%}` |
"""
        )
        st.caption(
            "Identical for every city, so a city's behaviour reflects its weather and not "
            "its tuning."
        )

    # --- baselines + alpha sweep, from scripts/benchmark.py ---
    bench_path = REPO_ROOT / "outputs" / f"benchmark_{profile_key}.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        st.markdown("#### Does it beat anything?")
        st.markdown(
            f"Median error across the {bench['windows']} monitoring windows of "
            f"{bench['monitor_days']} days each, so the served champion, the "
            "never-retrained champion and four predictors that need no training at all "
            "are directly comparable. Lower is better."
        )
        table = pd.DataFrame(
            [
                {
                    "predictor": s["name"],
                    "median RMSE": round(s["median_rmse"], 2),
                    "sees past PM2.5": "yes" if s["uses_past_target"] else "",
                    "what it does": s["detail"],
                }
                for s in bench["scored"]
            ]
        )
        st.dataframe(table, hide_index=True, width="stretch")

        # What retraining was worth, both ways, because one of them lies.
        #
        # This was a single number taken off the two rows above — median served
        # against median frozen — presented as the verdict. That comparison is
        # unpaired: in a city that promotes nothing until week 14 of 20, most
        # windows compare the first model against itself, both medians land on
        # the same value, and it reads 0.0% while every window a retrained model
        # actually served improved on the original. Publishing one without the
        # other is the mistake the evaluation doc is written against.
        served = next((s for s in bench["scored"] if s["name"] == "champion_served"), None)
        frozen = next((s for s in bench["scored"] if s["name"] == "champion_frozen"), None)
        across = value.get("across_replay")
        if across is None and served and frozen:
            across = (1 - served["median_rmse"] / frozen["median_rmse"]) * 100
        if across is not None:
            st.markdown("##### So was retraining worth it?")
            paired, unpaired = st.columns(2)
            if "when_it_acted" in value:
                paired.metric(
                    "Week by week",
                    f"{value['when_it_acted']:+.1f}%",
                    f"over the {value['acted_windows']} weeks a retrained model was serving · "
                    f"won {value['win_rate']:.0f}% of them",
                    delta_color="off",
                )
            else:
                paired.metric("Week by week", "—", "nothing retrained has served yet",
                              delta_color="off")
            unpaired.metric(
                "Across the whole replay",
                f"{across:+.1f}%",
                f"median of all {value.get('windows', len(runs))} windows against a frozen model",
                delta_color="off",
            )
            st.caption(
                "**Trust the first when they disagree.** The second compares one median against "
                "another, so where both are dominated by the same seasonal swing it largely "
                "measures the season, and every week before the first promotion is a model "
                "compared against itself. The first holds the window fixed, compares the two "
                "models in it, and counts only the weeks a retrained model was in service."
            )
            if across < -0.05:
                st.warning(
                    "Negative across the replay: retraining cost more than it returned here. "
                    "Retraining a city whose world barely moves fits noise, and that is a "
                    "finding rather than a fault."
                )

        alpha = bench.get("alpha")
        if alpha and alpha.get("curve"):
            st.markdown("#### Choosing alpha")
            st.markdown(
                f"Forward-chaining {alpha['n_splits']}-fold CV over the training window. "
                f"`TimeSeriesSplit` never lets a fold train on rows that follow the ones it "
                f"scores; a random split would leak badly on autocorrelated hourly data and "
                f"make every alpha look fine."
            )
            curve = pd.DataFrame(alpha["curve"], columns=["alpha", "cv_rmse"])
            fig = theme.base_figure(None, "CV RMSE", height=280)
            fig.add_scatter(
                x=curve["alpha"], y=curve["cv_rmse"], mode="lines+markers", name="CV RMSE"
            )
            fig.update_xaxes(type="log", title_text="alpha (log scale)")
            fig.add_vline(x=alpha["shipped"], line_dash="dot",
                          annotation_text=f"shipped {alpha['shipped']:g}")
            fig.add_vline(x=alpha["best"], line_dash="dash",
                          annotation_text=f"best {alpha['best']:g}")
            st.plotly_chart(fig, width="stretch")
            penalty = alpha.get("penalty_pct")
            caption = f"Shipped alpha {alpha['shipped']:g}; {alpha['best']:g} scored best"
            if penalty is not None:
                caption += (
                    f", costing {penalty:.1f}% error. The curve is nearly flat, which says the "
                    "model is limited by what a week-old weather forecast can express, not by "
                    "regularisation."
                )
            st.caption(caption)
    else:
        st.info(
            f"No benchmark for **{PROFILE.label}** yet. Run "
            f"`python scripts/benchmark.py --city all` to score it against the baselines."
        )

    # Answers the question the coefficient panels below cannot, and comes first
    # for that reason: which of these ingredients actually drives the answer.
    importance = (
        load_importance(
            DB, CFG.experiment_name, profile_key,
            int(serving_version), runs["as_of"].iloc[-1].isoformat(),
        )
        if serving_version is not None
        else None
    )
    if importance:
        st.markdown("#### What moves the prediction")
        st.markdown(
            "How far the model's answer shifts when each ingredient moves by one of its own "
            f"standard deviations, for v{serving_version} over the most recent window. This is "
            "the comparable version of the coefficients below: those are per °C, per hPa, per "
            "W/m², so the biggest of them is usually just the feature with the smallest units."
        )
        st.plotly_chart(theme.importance_bars(*importance), width="stretch")
        st.caption(
            "Boundary layer height, the depth of air pollution is diluted into, would very "
            "likely top this list and is missing. Open-Meteo does not archive it at a seven-day "
            "lead, so shortwave radiation stands in for it."
        )

    st.markdown("#### Coefficient evolution, a direct picture of concept drift")
    st.markdown(
        "The Ridge is one slope per weather feature and an intercept, in real "
        "units: PM2.5 per °C, per m/s wind, per %RH, per mm, per hPa, per W/m². "
        "Concept drift *is* these slopes changing, so watch them move each time "
        "the champion is retrained. The temperature slope in particular crosses "
        "zero as the summer relationship gives way to autumn."
    )
    st.caption(
        "One panel per feature, because these are per-unit slopes in units that "
        "are not comparable: on a shared axis the large-unit features flatten "
        "the rest against zero. The two cyclical hour terms are fitted but not "
        "drawn, because they encode the daily cycle, which does not invert."
    )
    if versions.empty or versions["coef_temperature"].isna().all():
        st.info("No coefficient tags found. Re-run `scripts/run_simulation.py --fresh`.")
    else:
        st.plotly_chart(
            theme.coef_small_multiples(versions, DRIFT_FEATURES), width="stretch"
        )
        show = versions[["version", "alias", "train_end", *[f"coef_{f}" for f in FEATURES], "coef_intercept"]]
        st.dataframe(
            show.rename(columns={"train_end": "trained through"}),
            width="stretch", hide_index=True,
        )

    st.divider()
    st.markdown("#### Champion fit on the monitored window")
    options = list(runs["as_of"])
    picked = st.selectbox(
        "Run (as-of date)", options, index=len(options) - 1,
        format_func=lambda d: d.strftime("%Y-%m-%d"), key="fit_run",
    )
    preds, _ = load_monitoring(DB, CFG.experiment_name, run_by_as_of[picked])
    resid = preds["actual"] - preds["predicted"]
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((preds["actual"] - preds["actual"].mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE", f"{rmse:.2f}")
    m2.metric("MAE", f"{mae:.2f}")
    m3.metric("R²", f"{r2:.2f}")
    m4.metric("Hours scored", len(preds))

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            theme.scatter_fit(preds["predicted"], preds["actual"], theme.SERIES[0]),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            theme.residual_series(preds["timestamp"], resid, theme.SERIES[0]),
            width="stretch",
        )
    st.caption(
        "Pick an early-summer run and a deep-autumn run to compare: pre-drift the "
        "cloud hugs the diagonal and residuals sit around zero; post-drift (before a "
        "retrain) the fit leans off the line and the residuals bias away from zero."
    )

# --------------------------------------------------------------------------- #
# Tab: knob sweep                                                              #
# --------------------------------------------------------------------------- #
with tab_sweep:
    sweep_path = REPO_ROOT / "outputs" / "sweep.csv"
    if not IS_SYNTHETIC:
        st.info(
            "The knob sweep is a **synthetic-only** diagnostic. It dials concept "
            "drift and covariate drift independently to prove the two detectors are "
            "separable. Real data has no such knobs. Switch to the Synthetic profile "
            "in the sidebar to see it."
        )
    elif not sweep_path.exists():
        st.info("Run `python scripts/sweep_knobs.py` to generate this.")
    else:
        sweep = pd.read_csv(sweep_path)
        st.markdown(
            "Each knob moves **its own** detector and leaves the other flat, which is "
            "why two signals are worth having. `feature_shift` changes the world's "
            "feature distributions (PSI sees it, with no model and no labels); "
            "`drift_strength` changes the relationship being learned (only the "
            "champion's error sees it)."
        )
        left, right = st.columns(2)
        for column, name, driven in (
            (left, "feature_shift", "max_psi"),
            (right, "drift_strength", "perf_drift_ratio"),
        ):
            group = sweep[sweep["sweep"] == name]
            fig = theme.base_figure(f"Sweeping {name}", "signal")
            fig.update_layout(hovermode="x")
            theme.line(fig, group["level"], group["max_psi"], "max PSI", theme.SERIES[0])
            theme.line(fig, group["level"], group["perf_drift_ratio"], "perf drift ratio", theme.SERIES[1])
            fig.update_xaxes(title=dict(text=f"{name} level", font=dict(color=theme.MUTED, size=11)))
            column.plotly_chart(fig, width="stretch")
            column.caption(f"`{driven}` monotonically increasing: **{group[driven].is_monotonic_increasing}**")

# --------------------------------------------------------------------------- #
# Tab: registry                                                                #
# --------------------------------------------------------------------------- #
with tab_registry:
    st.markdown(
        f"Registered model **`{CFG.registered_model_name}`**. MLflow 3 replaced "
        "Staging/Production stages with aliases, so a promotion moves the `champion` "
        "alias onto a new version."
    )
    versions = load_versions(DB, CFG.experiment_name, CFG.registered_model_name)
    if versions.empty:
        st.info("No registered versions yet.")
    else:
        st.dataframe(
            versions[["version", "alias", "train_start", "train_end", "baseline_rmse", "n_rows"]],
            width="stretch", hide_index=True,
        )

    st.markdown("**Promotion history**")
    if promoted.empty:
        st.info("No promotions yet.")
    else:
        st.dataframe(
            promoted[
                ["as_of", "tags.champion_version", "metrics.champion_rmse_holdout",
                 "metrics.challenger_rmse", "metrics.performance_gap"]
            ].rename(
                columns={
                    "tags.champion_version": "new champion v",
                    "metrics.champion_rmse_holdout": "old champion RMSE",
                    "metrics.challenger_rmse": "challenger RMSE",
                    "metrics.performance_gap": "gap",
                }
            ),
            width="stretch", hide_index=True,
        )

# --------------------------------------------------------------------------- #
# Tab: raw run table                                                           #
# --------------------------------------------------------------------------- #
with tab_table:
    st.caption("Every scheduled run, as logged to MLflow.")
    cols = {
        "as_of": "as_of",
        "metrics.data_drift_psi": "max PSI",
        "tags.worst_feature": "worst feature",
        "metrics.champion_rmse": "champion RMSE",
        "metrics.champion_mae": "champion MAE",
        "metrics.champion_r2": "champion R²",
        "metrics.perf_drift_ratio": "perf drift",
        "tags.retrain_triggered": "retrained",
        "metrics.challenger_rmse": "challenger RMSE (holdout)",
        "tags.promotion_decision": "decision",
        "tags.champion_version": "champion v",
    }
    present = {k: v for k, v in cols.items() if k in runs.columns}
    st.dataframe(runs[list(present)].rename(columns=present), width="stretch", hide_index=True)

st.divider()
st.caption("Same data in the MLflow UI:  `mlflow ui --backend-store-uri sqlite:///mlflow.db`")
