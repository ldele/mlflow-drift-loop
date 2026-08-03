# Air quality drift watch

[![ci](https://github.com/ldele/mlflow-drift-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/ldele/mlflow-drift-loop/actions/workflows/ci.yml)

Models go stale. The world they were trained on moves on, and usually nobody
notices until something downstream breaks.

This system notices on its own. Every week it marks its own homework. When the
model it is running starts slipping, it trains a replacement, and it refuses to
ship that replacement unless it beats the one already in service. Nobody is
watching it. It has been running since 2025.

The job it does is forecasting how dirty a city's air will be a week from now,
using nothing but the weather forecast. Six cities on six continents.

**[See it running →](https://ldele.github.io/mlflow-drift-loop/)**

![The dashboard](docs/images/dashboard.png)

## The loop

One run a week, four steps:

| | | |
|---|---|---|
| **1. Mark its homework** | check the last 14 days of forecasts against what the air actually did | |
| **2. Look for trouble, two ways** | has the weather stopped looking like what the model learned from, and separately, is the model getting things wrong? | two signals, on purpose |
| **3. Train a rival** | error is 1.25× what it used to be, so train a fresh model on the last 180 days | only if step 2 says so |
| **4. Make it earn the job** | both sit the same exam, a week of air neither has seen | the newcomer wins by 5% or it is thrown away |

**Why two signals and not one.** The first looks only at the weather coming in,
so it can raise a hand immediately, without waiting to find out whether the model
was wrong. But "the world looks different" is not the same as "the model is
failing". Kraków shows the gap: through the summer its weather drifts further
from training than anywhere else here, while the model quietly gets better. So
the cheap alarm watches, and only the expensive one, the one that asks whether we
actually got this wrong, can authorise spending money on a retrain.

**Nothing is graded on work it has already seen.** The replacement trains on a
window that stops before the exam, the incumbent was trained long before it, and
both are marked on the same unseen week. Details in
[methodology.md](docs/methodology.md).

## Six cities that disagree

Retraining pays where the world really moved, and costs where it did not. That
is the finding worth having, and it takes contrasting cities to show it.

Each city trains a model on a clean season, then runs week by week into the
season that ruins it. Every setting is identical across all six, so where two
cities behave differently, it is their air that differs and not their tuning.

| | how bad it gets (µg/m³) | weeks watched | retrains | was retraining worth it? |
|---|---|---|---|---|
| **Delhi** | 42 → 127, crop burning after the monsoon | 40 | 9 | **+43.8%** |
| **Santiago** | 18 → 94, winter smog trapped in a bowl | 22 | 13 | **+12.9%** |
| **Kraków** | 8 → 57, coal heating in a valley | 48 | 14 | +0.2% |
| **Johannesburg** | 23 → 84, winter coal smoke | 20 | 11 | 0.0% |
| **Melbourne** | 5 → 15, winter wood heaters | 31 | 8 | 0.0% |
| **Los Angeles** | 15 → 29, a mild winter bump | 37 | 3 | −8.9% |

In Delhi, where the air transforms, keeping the model fresh cuts its error by
nearly half. In Los Angeles, where the air barely moves, retraining makes things
8.9% worse by chasing noise. Los Angeles is the control, and it earns its place
by failing.

Johannesburg is where the promotion gate does its most visible work. Eleven
retrains, three shipped. The other eight replacements were not good enough and
were thrown away, and the net effect on error is zero. The system spent the
effort, declined to pay, and nothing shipped that had not earned it.

Half these cities are dirtiest in June to August and the other half in December
to January, which is how you can tell the thresholds are not secretly encoding a
season.

[evaluation.md](docs/evaluation.md) has the city-by-city detail, how the model
scores against four "do nothing clever" baselines, and a controlled experiment
showing that each of the two alarms answers only to the thing it is meant to
watch.

### What a full year exposed

The cities originally stopped at their winter peak. On that half of the story,
retraining looked like a clear win everywhere. Running them through the return
trip, as the air gets clean again, reversed the sign: retraining came out 29.6%
worse in Kraków and 4.3% worse in Delhi.

The fault was in the retraining rule, not the machinery. A replacement trained on
the last 45 days only ever sees one season. It is excellent in the season it was
born in and wrong as soon as the year turns. Widening that window to 180 days
took Delhi from −4.3% to +43.8%.

This is the most useful thing in the repository. The loop was behaving correctly
the whole time, faithfully shipping replacements that won their exam and then
aged badly, and only a full year of data made it visible.

## Serving the champion

The whole output of the loop is one label. `champion` points at whichever model
last passed its exam, and the API serves whatever wears that label, so promoting
a model is deploying it. No redeploy, no config change.

![The serving API](docs/images/serving.png)

```bash
python scripts/serve.py --city krakow      # http://localhost:8000/docs
docker build -t drift-serve --build-arg CITY=delhi .
docker run -p 8000:8000 drift-serve
```

Kraków serves the model left standing after 14 retrains, and `/model` reports the
window it was trained on, so you can see how old the thing answering you is.

It does not poll. A new model is picked up when `/reload` is called, because
swapping the model under live traffic without anyone asking is worse than serving
a slightly stale one. Predictions come back twice: `pm25` floored at zero for
whoever is consuming it, and `pm25_raw` as the model actually said. A clamp that
hides what your model is doing is how you stop noticing it.

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

Also on **Streamlit Community Cloud**: deploy this repo with `dashboard/app.py`
as the main file, Python 3.12.

## Layout

```
src/driftloop/    config, data sources, drift math, model, loop, benchmark, serving
scripts/          run_openmeteo · benchmark · build_site · run_scheduled · sweep_knobs · serve
site/             committed shell (index.html, app.js) + generated data.json
dashboard/        Streamlit app and shared chart theme
docs/             methodology · evaluation · wireframes each UI was built from
tests/            data contract, drift math, no-leak guards, baseline fairness, serving, charts
```

- **[methodology.md](docs/methodology.md)** covers how it works: the model, the
  features and why each one, the window layout, the guards against cheating.
- **[evaluation.md](docs/evaluation.md)** covers whether it works: per-city
  results, the baselines, the controlled experiment, and the limitations.

The model itself is weak, and that belongs up front rather than in a footnote.
Guessing an hour's pollution from a week-old weather forecast is hard, and the
numbers show it. What is worth looking at is the machinery around the model,
which would be unchanged if you dropped in something far better.
