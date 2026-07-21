# MLflow drift loop

**track → detect drift → retrain challenger → promote**, logged to MLflow. The
PoC from [`../mlflow_drift_project_plan.md`](../mlflow_drift_project_plan.md).

- **Phase 1 — synthetic.** The whole machinery against generated data with a
  controllable drift knob, so detection provably fires when we want it to.
- **Phase 2 — real (Open-Meteo).** The *same loop* on real Kraków weather +
  air-quality data, where a summer-trained PM2.5 model decays for real as the
  winter heating season and basin inversions arrive.
- **Phase 3 — scheduled.** The same loop again, but run **one incremental cycle
  at a time** against a **persistent** backend, driven by a GitHub Action on a
  weekly cron — the live loop accruing its own history over calendar time.

The payoff is one screen: drift climbs → the champion's error crosses a
threshold → a challenger is trained and judged on a window neither model has
seen → it takes over. Watch it in the dashboard (a sidebar toggle switches
between the three phases), or in the MLflow UI.

## Quickstart

```powershell
# from this folder
uv venv -p "C:\Users\LDELEZ\AppData\Local\Python\pythoncore-3.12-64\python.exe"
uv pip install --system-certs -e ".[dev]"

# Phase 1 — synthetic
.venv\Scripts\python.exe scripts\run_simulation.py --fresh   # bootstrap + replay ~5 months of weekly runs
.venv\Scripts\python.exe scripts\sweep_knobs.py              # prove the two drift signals are independent

# Phase 2 — real data (fetches + caches Open-Meteo, then runs the same loop)
.venv\Scripts\python.exe scripts\run_openmeteo.py --fresh

# Phase 3 — one incremental cycle (bootstrap on first run, monitor thereafter).
# --as-of lets you backfill / replay "weekly" runs locally:
.venv\Scripts\python.exe scripts\run_scheduled.py --as-of 2025-09-15   # first fire: bootstrap
.venv\Scripts\python.exe scripts\run_scheduled.py --as-of 2025-10-15   # next fire: monitor + maybe retrain

.venv\Scripts\python.exe -m streamlit run dashboard\app.py   # the dashboard (sidebar: pick the phase)
```

> **Windows / interpreter note.** The `ssl` module in some
> `python-build-standalone` builds (what `uv python install` fetches) aborts
> with `OPENSSL_Uplink: no OPENSSL_Applink` the moment anything loads a CA
> certificate — which MLflow does on import. Build the venv from a python.org
> interpreter instead (the `-p …pythoncore-3.12…` above). Check any interpreter
> with: `python -c "import ssl; ssl.create_default_context()"`.

> **Corporate-proxy / TLS note.** This machine sits behind a TLS-intercepting
> proxy, so both `uv` (`--system-certs`) and the Open-Meteo HTTP calls need the
> OS trust store. The Open-Meteo source calls `truststore.inject_into_ssl()`
> for exactly this reason; without it you'd get `CERTIFICATE_VERIFY_FAILED`.

## What each signal is (the heart of the project)

Two **independent** drift signals, so the champion/challenger logic isn't
circular:

| Signal | Measures | Needs a model? | Drives |
|---|---|---|---|
| **Data drift** (PSI, KS cross-check) | the *world* changed | no | early warning |
| **Performance drift** (champion RMSE now ÷ at training time) | the *model* is failing | champion only | the retrain trigger |

`sweep_knobs.py` proves they're independent: the synthetic world has two knobs,
and each moves exactly one signal.

- `feature_shift` (covariate drift) → **PSI** climbs, perf-drift flat.
- `drift_strength` (concept drift) → **perf-drift** climbs, PSI flat.

## No evaluation leak

When performance drift fires, the challenger trains on recent history **ending
before** a most-recent `holdout` window, and both champion and challenger are
scored on that holdout — which neither model has seen. Promotion happens only if
the challenger wins by a margin. `run_simulation` refuses a run cadence shorter
than the holdout, and `tests/test_loop.py` asserts the windows can't overlap.
(This is the leak the original prototype had — it trained the challenger on part
of the window it was then scored on.)

## Phase 2 — real data (Open-Meteo)

The whole point of the swappable data layer: Phase 2 is a **second `DataSource`
behind the same `get_data(start, end)` contract**. The loop, drift detection, and
dashboard don't change — only the data does.

- **Two endpoints, joined on time.** Weather (temperature, wind, humidity) from
  the ERA5 archive API; PM2.5 from the air-quality API. Fetched separately,
  inner-joined on the hourly timestamp, NaN rows dropped.
- **Fetch-once, slice-many.** The whole span is fetched once and cached to
  `data_cache/*.parquet`; every window request slices the cache — no repeated
  API hits, and the data is stable across runs.
- **Kraków, summer → winter.** A champion trained on summer 2025 is replayed
  weekly into the winter heating season. Real numbers from the run: summer PM2.5
  mean **~10 µg/m³** vs winter **~54**; the champion's RMSE climbs from ~4.5 to
  ~49, triggering **9 retrains and 7 promotions** (two early retrains were
  *rejected* — the challenger didn't clear the margin on the held-out window).

Each phase logs to its **own** MLflow backend file (`mlflow.db` /
`mlflow_openmeteo.db`) so they reset and browse independently; the dashboard's
sidebar toggle switches between them.

## Phase 3 — scheduled automation

Phases 1–2 *replay* a timeline in one process. Phase 3 is the loop as it would
actually run in production: [`run_scheduled.py`](scripts/run_scheduled.py) does
**exactly one thing per invocation** — bootstrap a champion if none exists,
otherwise run a single monitoring cycle at `as_of` and append it — against a
**persistent** backend (`mlflow_scheduled.db`) that is never reset.

[`.github/workflows/drift-loop.yml`](.github/workflows/drift-loop.yml) fires it on
a **weekly cron** (plus a manual button). Because each Action runs on a fresh VM,
the MLflow state is committed back to the repo so the next run continues from it —
the simplest zero-infra persistence, and it keeps the whole drift history
versioned and inspectable. (For production you'd point `MLFLOW_TRACKING_URI` at a
hosted tracking server with an object-store artifact root and drop the commit-back
step; a note in the workflow says as much.)

Locally you can play the cron forward with `--as-of`. Six separate invocations
(one bootstrap + five biweekly cycles Sep→Dec) accrue a clean version history —
champion **v1 → v4** as autumn/winter drift triggers retrains — in the persistent
backend, viewable under the dashboard's **Scheduled** profile.

> **One-time setup / caveat.** The Action needs the repo on GitHub with Actions
> write permission enabled. Also: MLflow stores artifact paths as absolute URIs,
> so a backend generated on the CI runner won't resolve *artifact files* if opened
> on a different machine — the metrics/params/tags (the whole drift story) travel
> fine; only the per-run prediction/distribution artifacts are path-bound. Runs
> generated locally are fully self-consistent locally.

## Viewing it online

Two ways to put the dashboard on the web, both free:

- **GitHub Pages (live): <https://ldele.github.io/mlflow-drift-loop/>** — an
  interactive static dashboard. The data lives in a plain, inspectable
  [`data.json`](https://ldele.github.io/mlflow-drift-loop/data.json) that
  [`scripts/build_site.py`](scripts/build_site.py) distils from the MLflow
  backends; the committed shell (`site/index.html` + `site/app.js`) fetches it and
  renders the Plotly charts client-side (hover / zoom / profile switch — no server).
  [`.github/workflows/pages.yml`](.github/workflows/pages.yml) rebuilds and
  republishes on every push and weekly, so it tracks the drift over time. Pages
  can't run Streamlit itself (no server-side execution) but happily serves the JS.
- **Streamlit Community Cloud** — the full interactive app. Deploy it in a few
  clicks at <https://share.streamlit.io> → *New app* → this repo, branch `master`,
  main file `dashboard/app.py` (pick Python 3.12 under *Advanced*). `requirements.txt`
  is committed for it. The Scheduled profile shows the committed live data; the
  Phase 1/2 profiles show a **Generate it now** button that populates them in place.

## Layout

```
src/driftloop/
  config.py          column contract, knobs/thresholds, Open-Meteo + profile configs
  data/
    base.py          the get_data(start, end) interface + a contract validator
    synthetic.py     deterministic synthetic world, two drift knobs
    openmeteo.py     real source: weather + air quality, joined + disk-cached
  drift.py           PSI, KS, distribution report, and performance-drift computations
  model.py           the (deliberately simple) Ridge pipeline + metrics + coefficients
  tracking.py        MLflow setup, registry, champion alias, per-profile backend/reset
  loop.py            one scheduled run: detect → maybe retrain → maybe promote
scripts/
  run_simulation.py  Phase 1: bootstrap + replay weekly runs across the synthetic shift
  sweep_knobs.py     the two-knob independence demo (offline, no MLflow)
  run_openmeteo.py   Phase 2: fetch/cache real data, then run the same loop
  run_scheduled.py   Phase 3: one incremental cycle against the persistent backend
  build_site.py      distil the MLflow backends into site/data.json for Pages
.github/workflows/
  drift-loop.yml     Phase 3: weekly cron that runs run_scheduled.py + persists state
  pages.yml          build data.json + publish the static dashboard to GitHub Pages
site/
  index.html, app.js committed static shell; fetches data.json, renders the charts
dashboard/
  app.py             Streamlit dashboard, profile selector (Phase 1 / 2 / 3)
  theme.py           chart palette + shared Plotly layout
tests/               data contract, drift math, no-leak guards, Open-Meteo (mocked)
```

## The dashboard

Six tabs, all reading from the MLflow backend:

- **Drift loop** — the three-chart story (data drift, performance drift,
  champion-vs-challenger), with the post-regime-shift era shaded and retrain /
  promotion events marked inline.
- **Feature drift** — per-feature reference-vs-latest distribution overlays
  annotated with PSI (the "why" behind the drift number), read from a
  distribution-report artifact each run logs. Evidently-style.
- **Model** — **coefficient evolution** across model versions (the Ridge slopes
  in real units; the temperature slope crossing zero *is* the concept drift),
  plus the champion's **predicted-vs-actual + residuals** on a chosen run with
  RMSE / MAE / R².
- **Knob sweep**, **Registry**, **Runs** — the two-knob independence proof, the
  version/alias/promotion history, and the raw run table.

## What MLflow tracks

Each scheduled run logs, as time-series metrics: `data_drift_psi`,
`perf_drift_ratio`, `champion_rmse`, `champion_mae`, `champion_r2`,
`champion_baseline_rmse`, per-feature `psi_*`/`ks_*`, and — when a challenger is
trained — `challenger_rmse`, `champion_rmse_holdout`, `performance_gap`. Tags
record `drift_detected`, `retrain_triggered`, `promotion_decision`. Each run also
logs two **artifacts** under `monitoring/` (the champion's predictions on the
window, and the feature-distribution report) so the detail panels stay decoupled
from the data source. Registered versions carry the model's learned
**coefficients as tags**, and promotions move the `champion` alias in the Model
Registry, giving an auditable history.

> **Backend & registry note.** The plan said "local file store", but MLflow's
> Model Registry needs a database backend, so this uses a local **SQLite** file
> (`mlflow.db`) — still zero-setup, single-file. MLflow 3 also replaced
> `Staging`/`Production` stage transitions with **aliases**; a promotion moves
> the `champion` alias onto a new version.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

## Next

All three planned phases are built. Natural hardening steps from here: a hosted
MLflow tracking server + object-store artifacts (so CI state isn't committed to
git), an alert when a promotion fires, and serving the champion behind an API —
serving was explicitly out of scope for v1.
