# Methodology

How the thing works. For whether it works, see [evaluation.md](evaluation.md).

## What the model predicts

A city's hourly PM2.5, seven days ahead.

The features are the weather forecast for the target hour as it stood a week
earlier, pulled from Open-Meteo's archive of previous model runs. At that lead
the weather forecast is already wrong by about 4.5 °C on temperature, and the
PM2.5 model inherits all of it before adding any error of its own. That is the
honest version of the chain, and it is the reason the benchmarks look the way
they do.

The lead is one number, `FORECAST_LEAD_DAYS` in `config.py`. Set it to 0 and the
features come from the ERA5 analysis instead, which turns the same code into a
same-hour estimator with no forecasting in it. Seven is the ceiling, because
Open-Meteo archives previous runs out to day seven and no further.

## The model

![Method](images/method.png)

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

`alpha=1.0` ships. Forward-chaining CV wants far heavier regularisation than that
in most cities, up to 1000, which is itself a signal: the fitted relationship is
weak enough that shrinking it toward zero costs almost nothing.

## Why two signals

Data drift needs no labels and no model, which makes it the early warning.
Performance drift is what triggers the retrain. Deciding "has it drifted?" by
comparing champion to challenger would be circular: you would need a challenger
before you were allowed to decide you needed one.

PSI above 0.25 counts as a significant feature-distribution shift (industry
convention: <0.10 stable, 0.10–0.25 moderate, >0.25 significant). The retrain
trigger is the champion's RMSE on the monitor window against its RMSE at
training time, at 1.25× — "the model got 25% worse".

## No evaluation leak

The challenger trains on a window that stops before the holdout, and the
champion was trained long before it. Both are scored on data neither has seen,
and the challenger has to clear a 5% margin rather than edge ahead.

```
...........[==== challenger train ====][= holdout =] as_of
                    [====== monitor window ========]
[== champion train ==]  (much earlier, never overlaps holdout)
```

The loop raises rather than warns if the holdout would overlap the champion's
training data, and `run_simulation` refuses a cadence shorter than the holdout,
which would let a freshly promoted champion be judged on its own training data.
Both are asserted in `tests/test_loop.py`.

The baselines are held to the same rule: a forecaster issuing seven days out may
use readings up to that moment and no later, so persistence repeats a week-old
observation instead of yesterday's.

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

## Serving

Three properties, because they are the ones a reviewer asks about:

- **The alias is the contract, not a version number.** Nothing in the service
  pins a version. Promote in the registry and `POST /reload` picks the new one
  up without a redeploy — that is the seam between the weekly loop and serving.
- **Serving never writes to the tracking store.** It sets the tracking URI and
  reads. It does not call `setup()`, which would create an experiment as a side
  effect; a read-only consumer should leave no trace.
- **The hour encoding is not reimplemented.** `add_cyclical_features` is the
  same function the training path uses, so the served features cannot drift away
  from the trained ones.

Timestamps are normalised to naive UTC before the hour is read off them. The
training data is GMT, so an aware timestamp from another zone has to be
converted first, or the diurnal encoding would be hours out of phase with what
the model learned. `tests/test_serving.py` asserts that `12:00Z` and
`14:00+02:00` predict identically.

The Docker image replays the loop from the committed parquet cache at build time
instead of copying the SQLite backend in. That is forced rather than chosen:
MLflow stores artifact locations as absolute URIs, so a backend built on a
developer's machine resolves to paths the container does not have. Rebuilding
inside the image is also what proves the replay is deterministic — it reproduces
the same champion, with the same baseline RMSE, from the same committed data.
The install is editable for a related reason: `tracking.REPO_ROOT` is derived
from the package file's location, so a non-editable install would put the
backend inside site-packages.

## Nothing gathered is thrown away

Raw hourly observations are committed as `data_cache/*.parquet` instead of being
re-fetched, so the charts always match a fixed, inspectable dataset. The forecast
lead is part of each cache filename, because the same place over the same span
holds different features at lead 0 and lead 7. The weekly Action commits
`mlflow_scheduled.db` back to the repo so each cycle continues from the last. The
published site carries its own data: a distilled `data.json` plus one raw CSV per
city, both downloadable from the live page.
