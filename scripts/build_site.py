"""Render a self-contained interactive HTML dashboard for GitHub Pages.

GitHub Pages can't run Streamlit (no server), but it serves JavaScript fine -- so
this bakes each profile's data into interactive Plotly charts at build time. The
visitor gets hover/zoom/legend-toggle and a profile switcher, with the data
frozen at the last build (the weekly Action rebuilds it).

Reads only the MLflow *metrics/tags* (no artifact files), so it's immune to the
absolute-artifact-path issue and needs nothing but the sqlite backends.

    python scripts/build_site.py            # -> site/index.html
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "dashboard"))

import theme  # noqa: E402  (reuse the same palette + figure helpers)
from driftloop import tracking  # noqa: E402
from driftloop.config import FEATURES, PROFILES  # noqa: E402
from driftloop.drift import PSI_SIGNIFICANT  # noqa: E402

OUT = REPO_ROOT / "site"
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


# --------------------------------------------------------------------------- #
# Data access (plain, no Streamlit)                                           #
# --------------------------------------------------------------------------- #
def load_runs(db: str, experiment: str) -> pd.DataFrame:
    tracking.setup(experiment, db)
    df = mlflow.search_runs(experiment_names=[experiment], order_by=["attributes.start_time ASC"])
    if df.empty or "tags.cycle_type" not in df:
        return pd.DataFrame()
    df = df[df["tags.cycle_type"] == "monitor"].copy()
    if df.empty:
        return df
    df["as_of"] = pd.to_datetime(df["params.as_of"])
    return df.sort_values("as_of").reset_index(drop=True)


def load_versions(db: str, experiment: str, model: str) -> pd.DataFrame:
    tracking.setup(experiment, db)
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model}'")
        alias_map = client.get_registered_model(model).aliases or {}
    except Exception:
        return pd.DataFrame()
    va: dict[str, list[str]] = {}
    for alias, ver in alias_map.items():
        va.setdefault(str(ver), []).append(alias)
    rows = []
    for mv in versions:
        row = {
            "version": int(mv.version),
            "alias": ", ".join(va.get(str(mv.version), [])),
            "train_end": pd.to_datetime(mv.tags.get("train_end")),
        }
        for name in [*FEATURES, "intercept"]:
            row[f"coef_{name}"] = float(mv.tags.get(f"coef_{name}", "nan"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("version").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figures (reuse theme.py so it matches the Streamlit app)                     #
# --------------------------------------------------------------------------- #
def _fig_html(fig, div_id: str) -> str:
    fig.update_layout(autosize=True, width=None)
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )


def profile_figures(runs: pd.DataFrame, versions: pd.DataFrame, drift_date, key: str) -> list[str]:
    retrained = runs[runs["tags.retrain_triggered"] == "True"]
    promoted = runs[runs["tags.promotion_decision"] == "promoted"]
    x_end = runs["as_of"].max()
    html: list[str] = []

    fig = theme.base_figure("Data drift — PSI per feature vs. the champion's training window", "PSI")
    theme.drift_region(fig, drift_date, x_end)
    for i, feature in enumerate(FEATURES):
        col = f"metrics.psi_{feature}"
        if col in runs:
            theme.line(fig, runs["as_of"], runs[col], feature, theme.SERIES[i])
    theme.threshold(fig, runs["as_of"], PSI_SIGNIFICANT, f"significant ({PSI_SIGNIFICANT})")
    html.append(_fig_html(fig, f"{key}_psi"))

    fig = theme.base_figure("Performance drift — champion RMSE now ÷ RMSE at training time", "ratio")
    theme.drift_region(fig, drift_date, x_end)
    theme.line(fig, runs["as_of"], runs["metrics.perf_drift_ratio"], "perf drift ratio", theme.SERIES[0])
    theme.threshold(fig, runs["as_of"], 1.25, "retrain trigger (1.25)")
    if not retrained.empty:
        theme.events(fig, retrained["as_of"], retrained["metrics.perf_drift_ratio"],
                     "retrain triggered", theme.WARNING, symbol="circle")
    html.append(_fig_html(fig, f"{key}_perf"))

    fig = theme.base_figure("Champion vs. challenger on the held-out window (RMSE)", "RMSE")
    judged = runs.dropna(subset=["metrics.challenger_rmse"]) if "metrics.challenger_rmse" in runs else runs.iloc[:0]
    if not judged.empty:
        theme.drift_region(fig, drift_date, x_end)
        theme.line(fig, judged["as_of"], judged["metrics.champion_rmse_holdout"], "champion", theme.SERIES[0])
        theme.line(fig, judged["as_of"], judged["metrics.challenger_rmse"], "challenger", theme.SERIES[1])
        if not promoted.empty:
            theme.events(fig, promoted["as_of"], promoted["metrics.challenger_rmse"], "promoted", theme.GOOD)
    else:
        fig.add_annotation(text="No challenger trained yet — no retrain has fired.",
                           showarrow=False, font=dict(color=theme.MUTED))
    html.append(_fig_html(fig, f"{key}_holdout"))

    fig = theme.base_figure("Coefficient evolution — the model's slopes across versions", "coefficient")
    fig.update_layout(hovermode="closest")
    if not versions.empty and not versions["coef_temperature"].isna().all():
        theme.coef_lines(fig, versions, FEATURES)
    else:
        fig.add_annotation(text="Only one model version so far.", showarrow=False, font=dict(color=theme.MUTED))
    html.append(_fig_html(fig, f"{key}_coef"))

    return html


def stat_row(runs: pd.DataFrame) -> str:
    retrains = int((runs["tags.retrain_triggered"] == "True").sum())
    promotions = int((runs["tags.promotion_decision"] == "promoted").sum())
    latest = runs.iloc[-1]
    r2 = f"{latest['metrics.champion_r2']:.2f}" if "metrics.champion_r2" in latest else "—"
    tiles = [
        ("Scheduled runs", len(runs)),
        ("Retrains", retrains),
        ("Promotions", promotions),
        ("Latest PSI", f"{latest['metrics.data_drift_psi']:.2f}"),
        ("Latest champion R²", r2),
    ]
    cells = "".join(f'<div class="tile"><div class="tile-v">{v}</div><div class="tile-k">{k}</div></div>'
                    for k, v in tiles)
    return f'<div class="tiles">{cells}</div>'


# --------------------------------------------------------------------------- #
# Page assembly                                                                #
# --------------------------------------------------------------------------- #
CSS = """
:root { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --line:#c3c2b7; }
* { box-sizing: border-box; }
body { margin:0; background:var(--surface); color:var(--ink);
  font-family: system-ui,-apple-system,"Segoe UI",sans-serif; }
header { padding: 28px 24px 8px; max-width: 1100px; margin: 0 auto; }
h1 { margin:0; font-size: 26px; }
.sub { color: var(--ink2); margin: 6px 0 0; }
.built { color: var(--muted); font-size: 12px; margin-top: 4px; }
nav { max-width:1100px; margin: 12px auto 0; padding: 0 24px; }
select { font: inherit; padding: 8px 12px; border:1px solid var(--line); border-radius:8px;
  background:#fff; color:var(--ink); }
main { max-width:1100px; margin: 8px auto 48px; padding: 0 24px; }
.tiles { display:flex; flex-wrap:wrap; gap:12px; margin: 18px 0 8px; }
.tile { flex:1; min-width:130px; border:1px solid var(--grid); border-radius:10px; padding:12px 14px; }
.tile-v { font-size: 26px; font-weight:600; }
.tile-k { color: var(--muted); font-size: 12px; margin-top:2px; }
.chart { border:1px solid var(--grid); border-radius:10px; margin:14px 0; padding:6px; }
section { display:none; }
section.active { display:block; }
a { color:#2a78d6; }
footer { max-width:1100px; margin:0 auto; padding: 0 24px 40px; color:var(--muted); font-size:13px; }
"""

STORY = {
    "synthetic": "Phase 1 · synthetic data with a controllable drift knob.",
    "openmeteo": "Phase 2 · real Kraków weather + air quality; a summer-trained model walking into winter smog.",
    "scheduled": "Phase 3 · the live loop, one scheduled run appended each week.",
}


def build() -> Path:
    sections, options = [], []
    for key, profile in PROFILES.items():
        cfg = profile.loop
        runs = load_runs(profile.db_filename, cfg.experiment_name)
        if runs.empty:
            continue
        versions = load_versions(profile.db_filename, cfg.experiment_name, cfg.registered_model_name)
        meta_path = REPO_ROOT / "outputs" / profile.meta_filename
        drift_date = None
        if meta_path.exists():
            import json
            drift_date = json.loads(meta_path.read_text(encoding="utf-8")).get("drift_date")

        charts = "".join(f'<div class="chart">{h}</div>'
                         for h in profile_figures(runs, versions, drift_date, key))
        sections.append(
            f'<section data-profile="{key}"><p class="sub">{STORY[key]}</p>'
            f"{stat_row(runs)}{charts}</section>"
        )
        options.append((key, profile.label))

    if not sections:
        raise SystemExit("No profiles have data — run the pipelines first.")

    built = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    opts_html = "".join(f'<option value="{k}">{label}</option>' for k, label in options)
    body = f"""<header>
  <h1>MLflow drift loop</h1>
  <p class="sub">track → detect drift → retrain challenger → promote</p>
  <p class="built">Static snapshot · built {built} · rebuilt weekly by GitHub Actions ·
    <a href="https://github.com/ldele/mlflow-drift-loop">source</a></p>
</header>
<nav><label>Data source&nbsp; <select id="profile">{opts_html}</select></label></nav>
<main>{''.join(sections)}</main>
<footer>Charts are interactive (hover, zoom, toggle series). Data is frozen at build time;
the full live app is the Streamlit dashboard in the repo.</footer>
<script>
  const sel = document.getElementById('profile');
  function show(p) {{
    document.querySelectorAll('main section').forEach(s => {{
      const on = s.dataset.profile === p;
      s.classList.toggle('active', on);
      if (on) s.querySelectorAll('.plotly-graph-div').forEach(d => window.Plotly && Plotly.Plots.resize(d));
    }});
  }}
  sel.addEventListener('change', e => show(e.target.value));
  show(sel.value);
</script>"""

    html = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>MLflow drift loop</title>"
        f'<script src="{PLOTLY_CDN}"></script><style>{CSS}</style></head><body>{body}</body></html>'
    )

    OUT.mkdir(exist_ok=True)
    (OUT / ".nojekyll").write_text("", encoding="utf-8")  # serve files as-is
    out = OUT / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB, {len(options)} profiles: "
          f"{', '.join(k for k, _ in options)})")
    return out


if __name__ == "__main__":
    build()
