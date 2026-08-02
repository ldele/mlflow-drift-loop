# Air quality drift watch

[![ci](https://github.com/ldele/mlflow-drift-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/ldele/mlflow-drift-loop/actions/workflows/ci.yml)

A model goes stale when the world stops matching what it was trained on. This
watches for that happening, retrains when it does, and ships the new model only
when it wins on data neither has seen.

It predicts a city's hourly PM2.5 seven days out, from the weather forecast for
that hour. It runs on real air quality from six cities on six continents. The
loop is the point: drift detection, a model registry, champion/challenger,
promotion gates, and a weekly cron that keeps going on its own.

**[See it running →](https://ldele.github.io/mlflow-drift-loop/)**

![The dashboard](docs/images/dashboard.png)

## The loop

One scheduled run a week, four steps:

| | | |
|---|---|---|
| **1. Monitor** | score the live champion on the last 14 days | |
| **2. Detect** | PSI on the features, and error against training time | two signals, independently |
| **3. Retrain** | error crossed 1.25×, so train a challenger on 45 days | only if step 2 says so |
| **4. Promote** | both scored on a held-out week neither has seen | challenger wins by 5% or it is rejected |

Two signals because data drift needs no labels, which makes it the early
warning, while performance drift is what triggers the retrain. Nothing is judged
on data it was trained on, and the challenger must clear a margin rather than
edge ahead — [methodology.md](docs/methodology.md) has the window layout and the
leak guards.

## Six cities that disagree

Each trains a champion on a clean season and replays weekly into the season that
ruins it, on thresholds identical across every city.

| | PM2.5 swing | retrains / runs | retraining worth |
|---|---|---|---|
| **Kraków** | 10 → 54, winter smog in a basin | 9 / 23 | +10.1% |
| **Santiago** | 18 → 94, winter inversion in a basin | 11 / 22 | **+26.1%** |
| **Delhi** | 42 → 127, post-monsoon burning | 6 / 16 | **+66.7%** |
| **Johannesburg** | 23 → 84, Highveld coal smoke | 9 / 20 | 0.0% |
| **Melbourne** | 5 → 15, winter wood heaters | 7 / 31 | −7.1% |
| **Los Angeles** | 15 → 29, a mild winter bump | 9 / 37 | −8.2% |

Retraining pays where drift is real and costs where it is not — the result worth
having. Half of these peak in June–August and half in northern winter, which is
how you can tell the thresholds are not quietly encoding a season.

[evaluation.md](docs/evaluation.md) has the per-city findings, the benchmarks
against four no-training baselines, and the controlled experiment behind the
two-signal design.

## Serving the champion

The loop's entire output is one alias. `champion` points at whichever version
last cleared the promotion gate, and the API reads that alias rather than a
pinned version — so promoting in the registry is what changes what gets served.

![The serving API](docs/images/serving.png)

```bash
python scripts/serve.py --city krakow                  # http://localhost:8000/docs
docker build -t drift-serve --build-arg CITY=delhi .   # one image per city
docker run -p 8000:8000 drift-serve
```

Kraków serves **version 10**, the champion left standing after nine retrains,
and `/model` reports the window it was trained on — so you can see how old the
thing answering you is.

It does not poll: a promotion is picked up when `/reload` is called, because
swapping a model under live traffic without anyone asking is worse than serving
a slightly stale one. `/model` reports which version is answering and how stale
it is. Predictions come back twice — `pm25` floored at zero for consumers, and
`pm25_raw` as the unconstrained Ridge actually said, because a clamp that hides
the model's behaviour is how you stop noticing it.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"

python scripts/run_openmeteo.py --fresh     # all six cities (--city krakow|santiago|delhi|joburg|melbourne|la)
python scripts/benchmark.py                 # baselines + alpha sweep
python scripts/build_site.py                # -> site/data.json

streamlit run dashboard/app.py              # the full app
python scripts/serve.py --city krakow       # serve the champion on :8000
mlflow ui --backend-store-uri sqlite:///mlflow_openmeteo.db

pytest -q && ruff check .                   # what CI gates on
```

The synthetic proof is `scripts/run_simulation.py --fresh` and
`scripts/sweep_knobs.py`. The live cycle is `scripts/run_scheduled.py --as-of
YYYY-MM-DD`.

Also on **Streamlit Community Cloud**: deploy this repo with `dashboard/app.py`
as the main file, Python 3.12.

## Layout

```
src/driftloop/    config, data sources, drift math, model, loop, benchmark, serving
scripts/          run_openmeteo · benchmark · build_site · run_scheduled · sweep_knobs · serve
site/             committed shell (index.html, app.js) + generated data.json
dashboard/        Streamlit app and shared chart theme
docs/             methodology · evaluation · wireframes each UI was built from
tests/            data contract, drift math, no-leak guards, baseline fairness, serving, Open-Meteo (mocked)
```

- **[methodology.md](docs/methodology.md)** — the model, the features and why
  each one, the window layout, the leak guards, what MLflow records.
- **[evaluation.md](docs/evaluation.md)** — per-city results, benchmarks against
  four baselines, the controlled drift experiment, and the limitations.

**R² is weak, and that is the honest headline.** Predicting an hour's PM2.5 from
a week-old weather forecast is genuinely hard. The loop is the demonstration,
not the model — see the limitations for the full list of what this does not do.
