"""Drift-loop dashboard.

Reads straight from the MLflow backend, so it shows whatever the last
`python scripts/run_simulation.py` produced. Per-run detail (feature
distributions, champion predictions) comes from artifacts each run logs, so the
detail panels don't touch the data generator -- they keep working in Phase 2.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
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
from driftloop.config import FEATURES, PROFILES  # noqa: E402
from driftloop.drift import PSI_SIGNIFICANT, PSI_STABLE  # noqa: E402

st.set_page_config(page_title="Drift loop", page_icon="~", layout="wide")


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


@st.cache_data(ttl=30)
def load_monitoring(db_filename: str, experiment: str, run_id: str) -> tuple[pd.DataFrame, dict]:
    """Download the per-run predictions CSV and feature-distribution JSON."""
    tracking.setup(experiment, db_filename)
    preds_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="monitoring/monitor_predictions.csv"
    )
    dist_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="monitoring/feature_distributions.json"
    )
    preds = pd.read_csv(preds_path, parse_dates=["timestamp"])
    report = json.loads(Path(dist_path).read_text(encoding="utf-8"))
    return preds, report


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
PROFILE = PROFILES[profile_key]
CFG = PROFILE.loop
DB = PROFILE.db_filename
IS_SYNTHETIC = profile_key == "synthetic"

runs = load_runs(DB, CFG.experiment_name)
meta = load_run_meta(PROFILE.meta_filename)
drift_date = meta.get("drift_date")

st.title("MLflow drift loop")
if IS_SYNTHETIC:
    st.caption("track → detect drift → retrain challenger → promote &nbsp;·&nbsp; Phase 1, synthetic data")
else:
    st.caption(
        "track → detect drift → retrain challenger → promote &nbsp;·&nbsp; "
        f"Phase 2, real weather + air quality · {meta.get('location', 'Open-Meteo')}"
    )

if runs.empty:
    script = "run_simulation.py" if IS_SYNTHETIC else "run_openmeteo.py"
    st.warning(
        f"No runs for this profile yet. Generate them first:\n\n```\npython scripts/{script} --fresh\n```"
    )
    st.stop()

promoted = runs[runs["tags.promotion_decision"] == "promoted"]
retrained = runs[runs["tags.retrain_triggered"] == "True"]
latest = runs.iloc[-1]
run_by_as_of = dict(zip(runs["as_of"], runs["run_id"]))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Scheduled runs", len(runs))
c2.metric("Retrains triggered", len(retrained))
c3.metric("Promotions", len(promoted))
c4.metric(
    "Latest max PSI",
    f"{latest['metrics.data_drift_psi']:.2f}",
    latest["tags.data_drift_label"],
    delta_color="off",
)
c5.metric(
    "Latest champion R²",
    f"{latest['metrics.champion_r2']:.2f}" if "metrics.champion_r2" in latest else "—",
    f"RMSE {latest['metrics.champion_rmse']:.2f}",
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
    elif profile_key == "openmeteo":
        story += (
            " Here it's a summer-trained model walking into the winter heating season, "
            "when basin inversions drive PM2.5 up several-fold."
        )
    else:  # scheduled
        story += (
            " Each point is one **scheduled run** appended over calendar time — the live "
            "loop accruing its own history, one cron fire at a time (Phase 3)."
        )
    st.markdown(story)

    fig = theme.base_figure("Data drift — PSI per feature vs. the champion's training window", "PSI")
    theme.drift_region(fig, drift_date, runs["as_of"].max())
    for i, feature in enumerate(FEATURES):
        col = f"metrics.psi_{feature}"
        if col in runs:
            theme.line(fig, runs["as_of"], runs[col], feature, theme.SERIES[i])
    theme.threshold(fig, runs["as_of"], PSI_SIGNIFICANT, f"significant ({PSI_SIGNIFICANT})")
    st.plotly_chart(fig, width="stretch")

    fig = theme.base_figure(
        "Performance drift — champion RMSE now ÷ champion RMSE at training time", "ratio"
    )
    theme.drift_region(fig, drift_date, runs["as_of"].max())
    theme.line(fig, runs["as_of"], runs["metrics.perf_drift_ratio"], "perf drift ratio", theme.SERIES[0])
    theme.threshold(
        fig, runs["as_of"], CFG.perf_drift_threshold, f"retrain trigger ({CFG.perf_drift_threshold})"
    )
    if not retrained.empty:
        theme.events(
            fig, retrained["as_of"], retrained["metrics.perf_drift_ratio"],
            "retrain triggered", theme.WARNING, symbol="circle",
        )
    st.plotly_chart(fig, width="stretch")

    fig = theme.base_figure("Champion vs. challenger on the held-out window (RMSE, lower is better)", "RMSE")
    judged = runs.dropna(subset=["metrics.challenger_rmse"])
    if judged.empty:
        st.info("No challenger has been trained yet — performance drift never crossed the threshold.")
    else:
        theme.drift_region(fig, drift_date, runs["as_of"].max())
        theme.line(fig, judged["as_of"], judged["metrics.champion_rmse_holdout"], "champion", theme.SERIES[0])
        theme.line(fig, judged["as_of"], judged["metrics.challenger_rmse"], "challenger", theme.SERIES[1])
        if not promoted.empty:
            theme.events(
                fig, promoted["as_of"], promoted["metrics.challenger_rmse"], "promoted", theme.GOOD
            )
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# Tab: feature distributions (the "why" behind PSI)                           #
# --------------------------------------------------------------------------- #
with tab_dist:
    st.markdown(
        "The PSI number on the previous tab is a summary of *this*: how each "
        "feature's distribution in the latest window (filled) has moved away from "
        "the champion's training window (outline). This is what a data-drift "
        "monitor like Evidently shows — logged as an artifact every run."
    )
    options = list(runs["as_of"])
    picked = st.selectbox(
        "Run (as-of date)", options, index=len(options) - 1,
        format_func=lambda d: d.strftime("%Y-%m-%d"), key="dist_run",
    )
    _, report = load_monitoring(DB, CFG.experiment_name, run_by_as_of[picked])

    cols = st.columns(len(FEATURES))
    for col, feature in zip(cols, FEATURES):
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

    st.markdown("#### Coefficient evolution — a direct picture of concept drift")
    st.markdown(
        "The Ridge is three slopes and an intercept (in real units: PM2.5 per °C, "
        "per m/s wind, per %RH). Concept drift *is* these slopes changing, so watch "
        "them move each time the champion is retrained — the temperature slope in "
        "particular crosses zero as the summer relationship gives way to autumn."
    )
    if versions.empty or versions["coef_temperature"].isna().all():
        st.info("No coefficient tags found — re-run `scripts/run_simulation.py --fresh`.")
    else:
        fig = theme.base_figure(None, "coefficient (per unit)", height=320)
        fig.update_layout(hovermode="closest")
        theme.coef_lines(fig, versions, FEATURES)
        st.plotly_chart(fig, width="stretch")
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
            "The knob sweep is a **synthetic-only** diagnostic — it dials concept "
            "drift and covariate drift independently to prove the two detectors are "
            "separable. Real data has no such knobs. Switch to the Synthetic profile "
            "in the sidebar to see it."
        )
    elif not sweep_path.exists():
        st.info("Run `python scripts/sweep_knobs.py` to generate this.")
    else:
        sweep = pd.read_csv(sweep_path)
        st.markdown(
            "Each knob moves **its own** detector and leaves the other flat — which is "
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
        "Staging/Production stages with aliases — a promotion moves the `champion` "
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
    st.caption("Every scheduled run, exactly as logged to MLflow.")
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
