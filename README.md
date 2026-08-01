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
| **Kraków** | May 25 → Feb 26 | 10 → 54, winter smog in a basin | 5.2 → peak 53.3 | 8 / 23 | +4.4% |
| **Santiago** | Oct 25 → Jul 26 | 18 → 94, winter inversion in a basin | 8.0 → peak 60.8 | 8 / 22 | **+30.6%** |
| **Delhi** | May 25 → Feb 26 | 42 → 127, post-monsoon burning | 20.1 → peak 71.8 | 5 / 16 | **+47.3%** |
| **Johannesburg** | Nov 25 → Jul 26 | 23 → 84, Highveld coal smoke | 14.2 → peak 89.6 | 8 / 20 | −1.9% |
| **Melbourne** | Sep 25 → Jul 26 | 5 → 15, winter wood heaters | 3.3 → peak 15.9 | 7 / 31 | −1.8% |
| **Los Angeles** | Sep 25 → Jul 26 | 15 → 29, a mild winter bump | 16.7 → peak 18.2 | 2 / 37 | −11.6% |

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
champions decay by roughly an order of magnitude (5.2 to 53.3 in Kraków, 8.0 to
60.8 in Santiago) and the loop fires exactly 8 retrains in each, off the same
thresholds, in opposite seasons.

What the two cities do *not* share is the payoff. Retraining is worth +30.6% in
Santiago and +4.4% in Kraków. Detection is symmetric; the value of acting on it
is not, because Kraków's original champion happens to stay closer to usable as
the winter comes in. That asymmetry is a result, not a defect in the pairing.

### Where it stops paying

Johannesburg drifts more than anywhere else here and gains the least from
retraining. The champion's error climbs from 14.2 to 89.6 µg/m³, the worst on the
page, and eight retrains buy −1.9%. Climatology, which is the training mean for
that hour of day and nothing more, beats the served model outright. Noticing that
a model has gone stale is a different problem from being able to fix it, and only
the largest drift signal in the project makes that visible.

Los Angeles is the control, and the measurements chose it for that. It was
picked as the summer-smog city; hourly PM2.5 over 2025–26 peaks in November and
bottoms out in June. Its champion barely moves across 37 runs, the loop retrains
twice and promotes once, and retraining costs 11.6%. A drift loop needs drift.

Melbourne is the counter-intuitive case. Its air stays near the WHO guideline all
year, yet the champion still decays to 2.16× its training error, and retraining
buys −1.8%. Clean air does not imply a stable model.

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
- **Retraining earns its keep where drift is real:** +47.3% in Delhi, +30.6% in
  Santiago. Delhi's frozen champion ends up worse than a constant, which is what
  concept drift looks like when nobody intervenes.
- **It costs 11.6% in Los Angeles** and buys nothing in Johannesburg (−1.9%) or
  Melbourne (−1.8%). Retraining a city whose world barely moves fits noise.
- **The champion still loses to a constant in three cities.** Climatology beats
  it in Johannesburg, and the training mean beats it in Melbourne. Where PM2.5
  barely moves, a week-old weather forecast carries too little signal to improve
  on an average.

`alpha=1.0` ships. Forward-chaining CV picks anywhere from 0.001 in Los Angeles
to 1000 in Delhi and Johannesburg, worth between 0.0% and 4.7% error depending on
the city. The curve is nearly flat everywhere, which says the model is limited by
what three forecast weather features can express rather than by its
regularisation.

## The model

![Method](docs/images/method.png)

`StandardScaler → Ridge(alpha=1.0)` on forecast temperature, wind speed and
humidity, predicting hourly PM2.5 seven days out. Trained on a chronological
80/20 tail split, never a random one, then refit on the full window.

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
- **Three features is not enough.** The alpha curve is flat in every city, which
  says the ceiling is the feature set rather than the regularisation. Boundary
  layer height, precipitation and a wind-direction term would all plausibly help.
- **No autoregressive features.** Giving the champion a lag term would help most
  in the low-drift cities where a constant currently beats it. It would also
  blunt the drift story, since an autoregressive model absorbs regime change
  instead of decaying visibly through it. That tension is a design decision.
- **One lead time.** Everything here is measured at seven days. The interesting
  sweep, error against lead from 1 to 7 days, is not run.
