# Air quality drift watch

A model goes stale when the world stops matching what it was trained on. This
watches for that happening, retrains when it does, and ships the new model only
when it wins on data neither has seen.

It runs on real air quality from six cities on six continents. The loop is the
point: drift detection, a model registry, champion/challenger, promotion gates,
and a weekly cron that keeps going on its own.

**[See it running →](https://ldele.github.io/mlflow-drift-loop/)**

![The dashboard](docs/images/dashboard.png)

## What the model predicts

A city's hourly PM2.5, seven days ahead.

The features are the weather forecast for the target hour as it stood a week
earlier, pulled from Open-Meteo's archive of previous model runs. At that lead
the weather forecast is already wrong by about 4.5 °C on temperature, and the
PM2.5 model inherits all of it before adding any error of its own. That is the
honest version of the chain, and it is the reason the benchmark below looks the
way it does.

The lead is one number, `FORECAST_LEAD_DAYS` in `config.py`. Set it to 0 and the
features come from the ERA5 analysis instead, which turns the same code into a
same-hour estimator with no forecasting in it. Seven is the ceiling, because
Open-Meteo archives previous runs out to day seven and no further.

## The loop

One scheduled run a week, four steps:

| | | |
|---|---|---|
| **1. Monitor** | score the live champion on the last 14 days | |
| **2. Detect** | PSI on the features, and error against training time | two signals, independently |
| **3. Retrain** | error crossed 1.25×, so train a challenger on 45 days | only if step 2 says so |
| **4. Promote** | both scored on a held-out week neither has seen | challenger wins by 5% or it is rejected |

**Why two signals.** Data drift needs no labels and no model, which makes it the
early warning. Performance drift is what triggers the retrain. Deciding "has it
drifted?" by comparing champion to challenger would be circular: you would need a
challenger before you were allowed to decide you needed one.

**No evaluation leak.** The challenger trains on a window that stops before the
holdout, and the champion was trained long before it. Both are scored on data
neither has seen, and the challenger has to clear a margin rather than edge
ahead. The baselines are held to the same rule: a forecaster issuing seven days
out may use readings up to that moment and no later, so persistence repeats a
week-old observation instead of yesterday's.

## Six cities that disagree

Weather forecasts from the [Open-Meteo](https://open-meteo.com/) historical
forecast archive, joined on the hour with observed PM2.5 from its air-quality
API. Each city trains a champion on a clean season and replays weekly into the
season that ruins it.

| | span | PM2.5 swing | champion RMSE | retrains / runs | retraining worth |
|---|---|---|---|---|---|
| **Kraków** | May 25 → Feb 26 | 10 → 54, winter smog in a basin | 5.1 → peak 51.8 | 9 / 23 | +10.1% |
| **Santiago** | Oct 25 → Jul 26 | 18 → 94, winter inversion in a basin | 6.4 → peak 66.2 | 11 / 22 | **+26.1%** |
| **Delhi** | May 25 → Feb 26 | 42 → 127, post-monsoon burning | 20.0 → peak 66.0 | 6 / 16 | **+66.7%** |
| **Johannesburg** | Nov 25 → Jul 26 | 23 → 84, Highveld coal smoke | 11.6 → peak 81.6 | 9 / 20 | 0.0% |
| **Melbourne** | Sep 25 → Jul 26 | 5 → 15, winter wood heaters | 3.3 → peak 13.2 | 7 / 31 | −7.1% |
| **Los Angeles** | Sep 25 → Jul 26 | 15 → 29, a mild winter bump | 17.3 → peak 19.2 | 9 / 37 | −8.2% |

The cities were picked on measurements rather than reputation. Eighteen
candidates were fetched and ranked by PM2.5 swing before any of them was wired
in, which cost two obvious choices. Sydney turned out flat, at 1.5×, with no
story in it. No Brazilian city worked either: the Amazon burning-arc cities peak
at only about 14 µg/m³ in this window, and São Paulo swings 1.7×, less than Los
Angeles does.

### The loop has no calendar in it

Kraków, Delhi and Los Angeles all peak in northern winter, which leaves a fair
objection open. Are the thresholds quietly encoding a season? Santiago,
Johannesburg and Melbourne peak in June, July and August, when Kraków and Delhi
are at their cleanest, and they run on settings identical to every other city's.

Santiago answers it most directly, because it is Kraków's twin. A coastal basin
that traps winter inversions the same way, half a year out of phase. Both
champions decay by an order of magnitude (5.1 to 51.8 in Kraków, 6.4 to 66.2 in
Santiago), both draw retrains off the same thresholds in opposite seasons, and
retraining pays in both (+10.1% and +26.1%).

### Where it stops paying

Johannesburg is where the promotion gate does the most visible work. The
champion's error climbs from 11.6 to 81.6 µg/m³, the worst on the page, and nine
retrains produce only two promotions: the other seven challengers failed to clear
the 5% margin and were thrown away. Retraining nets exactly 0.0%. The loop spends
the effort, the gate declines to pay, and nothing ships that did not earn it.

Los Angeles is the control, and the measurements chose it for that. It was
picked as the summer-smog city; hourly PM2.5 over 2025–26 peaks in November and
bottoms out in June. Its champion barely moves across 37 runs, retraining costs
8.2%, and climatology beats it outright. A drift loop needs drift.

Melbourne is the counter-intuitive case. Its air stays near the WHO guideline all
year, yet the champion still decays to 2.01× its training error, and retraining
buys −7.1%. Clean air does not imply a stable model.

A seventh source, **Live schedule**, is not a city. It is the same Kraków data
run one incremental cycle at a time by a weekly GitHub Action, accruing its own
history over calendar time. It has run only a handful of cycles so far, and the
page says so on its own tab rather than letting two points pass for a trend. An
eighth, synthetic, has two independent drift knobs and backs the offline
correctness proof in [`sweep_knobs.py`](scripts/sweep_knobs.py). It is not
published.

## Does the model beat anything?

![Benchmarks](docs/images/benchmark.png)

Every predictor is scored on the same 14-day monitoring windows the loop reports
on, so the served champion, a champion that never retrained, and four
no-training baselines all land in one comparable column.

They are grouped by what each predictor is allowed to see rather than ranked in a
single list. Persistence and seasonal naive get the readings available when the
forecast was issued. The champion gets the weather forecast and nothing else.

- **The champion beats persistence in all six cities.** At a seven-day lead the
  last available reading is a week old, which is most of the way to useless, so
  the weather-based model earns its place. This is the finding that flips with
  the horizon: at a one-hour lead persistence wins everywhere by 3× or more.
- **Retraining earns its keep where drift is real:** +66.7% in Delhi, +26.1% in
  Santiago, +10.1% in Kraków. Delhi's frozen champion ends up worse than a
  constant, which is what concept drift looks like when nobody intervenes.
- **It costs 8.2% in Los Angeles** and 7.1% in Melbourne, and breaks exactly even
  in Johannesburg. Retraining a city whose world barely moves fits noise.
- **The champion is the best predictor of the six in Kraków, Delhi and
  Johannesburg**, second in Santiago, and still loses to climatology in Los
  Angeles. Where PM2.5 barely moves, a week-old forecast carries too little
  signal to beat an hour-of-day average.

`alpha=1.0` ships. Forward-chaining CV wants far heavier regularisation than that
in most cities, up to 1000, which is itself a signal: the fitted relationship is
weak enough that shrinking it toward zero costs almost nothing.

## Does the detection actually work?

![The controlled experiment](docs/images/control.png)

No real city can answer that. Its drift has no known cause and no control
condition, so a detector firing at random would look much the same. The synthetic
world has both: two knobs that move covariate drift and concept drift
independently, and each detector can be checked against the cause it should
answer to and the one it should ignore.

| knob turned 0 → 2 | data drift (PSI) | performance drift |
|---|---|---|
| **covariate** (feature distributions shift) | **1.77×** | 1.01× |
| **concept** (the relationship changes) | 1.00× | **5.92×** |

Each detector responds to its own cause and ignores the other. PSI across the
entire concept sweep is not merely stable, it is identical to fifteen decimal
places, because changing the relationship between features and target cannot move
a statistic computed on the features alone.

That is the property the two-signal design rests on, and it is the one thing the
six cities cannot demonstrate. It is published on the page for the same reason it
is here: it is the evidence, and the cities are the application.

## The model

![Method](docs/images/method.png)

`StandardScaler → Ridge(alpha=1.0)` on eight features, predicting hourly PM2.5
seven days out. Trained on a chronological 80/20 tail split, never a random one,
then refit on the full window.

Six are observed weather, chosen for how pollution accumulates and clears rather
than for what was easy to fetch: temperature, wind speed, humidity,
**precipitation** (which scavenges particulates out of the air), **surface
pressure** (subsidence inversions that trap them) and **shortwave radiation**
(daytime convective mixing that dilutes them). Two more encode hour-of-day as a
point on a circle, so the model can express a daily cycle rather than losing to
an hour-of-day average.

Only the six weather features carry a PSI. The clock cannot drift: every
monitoring window contains all 24 hours, so its distribution is fixed by
construction, and a drift chart that plotted it would be showing a flat line
forever.

The features and the target share one timestamp in the column contract, so the
horizon lives entirely in which weather the data source fetches. Nothing
downstream of the fetch knows the difference, which is why the same loop, the
same drift maths and the same registry served both framings without modification.

It is kept small so that it decays visibly when the relationship shifts, where a
larger model would absorb some of the drift and hide it. **The modelling is not
the interesting part of this project.** The loop around it is, and it would be
unchanged if you dropped in something bigger. Every threshold is identical across
cities, so a city's behaviour reflects its weather rather than its tuning.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"

python scripts/run_openmeteo.py --fresh     # all six cities (--city krakow|santiago|delhi|joburg|melbourne|la)
python scripts/benchmark.py                 # baselines + alpha sweep
python scripts/build_site.py                # -> site/data.json

streamlit run dashboard/app.py              # the full app
mlflow ui --backend-store-uri sqlite:///mlflow_openmeteo.db
```

The synthetic proof is `scripts/run_simulation.py --fresh` and
`scripts/sweep_knobs.py`. The live cycle is `scripts/run_scheduled.py --as-of
YYYY-MM-DD`.

## What MLflow tracks

Per monitoring run, as time-series metrics: `data_drift_psi`,
`perf_drift_ratio`, `champion_rmse`/`mae`/`r2`, `champion_baseline_rmse`,
per-feature `psi_*` and `ks_*`, plus `challenger_rmse`, `champion_rmse_holdout`
and `performance_gap` when a challenger exists. Tags record `drift_detected`,
`retrain_triggered` and `promotion_decision`. Registered versions carry their
learned coefficients as tags, and a promotion moves the `champion` **alias**,
which is an auditable version history.

> The Model Registry needs a database backend, so this uses a local SQLite file.
> MLflow 3 replaced `Staging`/`Production` transitions with aliases.

## Nothing gathered is thrown away

Raw hourly observations are committed as `data_cache/*.parquet` instead of being
re-fetched, so the charts always match a fixed, inspectable dataset. The forecast
lead is part of each cache filename, because the same place over the same span
holds different features at lead 0 and lead 7. The weekly Action commits
`mlflow_scheduled.db` back to the repo so each cycle continues from the last. The
published site carries its own data: a distilled `data.json` plus one raw CSV per
city, both downloadable from the live page.

## Layout

```
src/driftloop/    config, data sources, drift math, model, loop, benchmark
scripts/          run_openmeteo · benchmark · build_site · run_scheduled · sweep_knobs
site/             committed shell (index.html, app.js) + generated data.json
dashboard/        Streamlit app and shared chart theme
docs/wireframes/  the drawings each UI was built from
tests/            data contract, drift math, no-leak guards, baseline fairness, Open-Meteo (mocked)
```

Also on **Streamlit Community Cloud**: deploy this repo with `dashboard/app.py`
as the main file, Python 3.12.

## Limitations

- **State lives in git.** Committing the SQLite backend back to the repo is the
  simplest zero-infra persistence and keeps history versioned, but it grows the
  repo. Production would point `MLFLOW_TRACKING_URI` at a hosted tracking server.
- **Artifact paths are absolute.** A backend generated on the CI runner resolves
  metrics, params and tags anywhere, but not the per-run prediction files.
- **No serving.** The champion is registered and promoted, never served behind an
  API. That and an alert on promotion are the obvious next steps.
- **R² is still weak, and that is the honest headline.** Widening from three
  features to eight improved every city (Los Angeles went from −2.13 to −0.98 on
  the latest window, Melbourne from −0.28 to +0.12) but R² on the most-drifted
  window is still near zero or negative in four of six. Predicting an hour's
  PM2.5 from a week-old weather forecast is genuinely hard, and the page should be
  read with that in mind: the loop is the demonstration, not the model.
- **Boundary layer height is missing and it is the feature I most want.** It sets
  the volume pollution is diluted into and would likely matter more than anything
  in the list. Open-Meteo does not archive previous model runs for it, so at a
  seven-day lead it returns null. Shortwave radiation is the stand-in.
- **No autoregressive features.** Giving the champion a lag term would help most
  in the low-drift cities where a constant currently beats it. It would also
  blunt the drift story, since an autoregressive model absorbs regime change
  instead of decaying visibly through it. That tension is a design decision.
- **One lead time.** Everything here is measured at seven days. The interesting
  sweep, error against lead from 1 to 7 days, is not run.
