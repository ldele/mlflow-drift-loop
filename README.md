# Air quality drift watch

[![ci](https://github.com/ldele/mlflow-drift-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/ldele/mlflow-drift-loop/actions/workflows/ci.yml)

A model in service goes stale as the world it was trained on moves, and usually
nobody notices until something downstream breaks. This repository is a loop that
notices: every week it checks the model in service, trains a replacement when the
model is slipping, and ships the replacement only if it beats the model already
doing the job.

It is also a measured account of how well such a loop works, including where it
does not. The test bed is a hard forecasting job: how dirty a city's air will be
a week from now, from nothing but the weather forecast. Six cities are replayed
week by week through a year, and one runs live every Monday.

**[See it running →](https://ldele.github.io/mlflow-drift-loop/)**

![The dashboard](docs/images/dashboard.png)

## The loop

One run a week, four steps:

| | | |
|---|---|---|
| **1. Mark its homework** | score the last 14 days of forecasts against the PM2.5 recorded for those days | a model's estimate of the air, not a station ([why](docs/methodology.md#the-prediction-task)) |
| **2. Look for trouble** | has the weather stopped looking like the training weather, and is the model's error rising? | two signals, on purpose |
| **3. Train a rival** | when error is 1.25× what it was at training, or the model has fallen 50% behind a plain daily profile | only if step 2 says so |
| **4. Make it earn the job** | both models sit the same exam, a week of air neither has seen | the rival wins by 5% or is thrown away |

The weather check needs no recorded outcomes, so it can speak first. But weather
that looks different is not a model that is failing, so only a rise in the
model's own error can order a retrain. Nothing is graded on data it has seen: the
rival trains on a window that stops before the exam. Details are in
[methodology.md](docs/methodology.md).

## What was measured

These are findings about the loop. The cities are where it was tested, picked
because they disagree with one another. Each percentage compares the loop with
the same first model never retrained, week by week, with a 95% interval
([how the intervals are built](docs/evaluation-first-loop.md#what-the-intervals-cost-this-page)).

1. **Retraining pays where the air changes, and nowhere else.** Four cities gain,
   led by Delhi at **+22.8% [+11.8, +36.4]**<!--fig:delhi.acted-->. Los Angeles, the
   control, and Kraków show nothing measurable. Under a tree model tuned the same
   way Delhi still reads +13.4% [+8.0, +27.0]<!--fig:ablation.delhi.gbm_tuned-->, so the
   gain is not a straight line being refitted to a season. On nine cities outside the
   six it paid in 3<!--fig:heldout.p1.pays-->, harmed none, and the seasonal swing measured
   the year before did not say which, against a
   [prediction written first](docs/registrations/2026-11-11-premium-on-held-out-cities.md).

2. **A default setting doubled the first headline.** This page once led with
   Delhi at +49.4% [+33.7, +61.8]<!--fig:first.delhi.acted--> and with retraining costing
   Los Angeles −13.4% [−36.9, −3.8]<!--fig:first.la.acted-->. The model's regularisation was
   a library default, and 21.6<!--fig:first.ablation.delhi.alpha_points--> of Delhi's points
   were that default. The loop now chooses the setting at every fit.
   [HISTORY.md](docs/HISTORY.md) keeps every earlier figure and what corrected it.

3. **The retrain alarm goes deaf, and a second one bounds the silence.** The alarm
   compares the model with its own error at training time, so every promotion
   raises the bar. Left alone it was silent for
   30<!--fig:rebaseline.krakow.shipped.silence--> weeks in Kraków and
   35<!--fig:rebaseline.la.shipped.silence--> of 36<!--fig:la.weeks--> in Los Angeles. A second
   alarm that no promotion can reset cuts those to
   22<!--fig:rebaseline.krakow.fixed.silence--> and 16<!--fig:rebaseline.la.fixed.silence-->. On nine
   cities the study had never used it shortened the silence in all five where it
   fired. It is not shown to make the forecasts better.

4. **No rule about when to retrain, or how hard to judge, made the forecasts
   better.** Five fixes were replayed on two differently set loops. None shows a
   gain in two cities on the loop that ships, and a harm seen on the first loop
   did not repeat.

5. **A seven-day exam vouches for a model for about a month.** Across
   28<!--fig:gate.short.n--> promotions that served under twenty weeks the exam promised
   +12.7%<!--fig:gate.short.exam--> and the winners delivered
   **+8.0% [+5.1, +11.0]**<!--fig:gate.short.delivered-->. The 2<!--fig:gate.long.n--> that served
   longer both ended up worse than the model they replaced.

The model itself is weak, and that belongs up front: an hour's pollution from a
week-old weather forecast is hard to guess. The subject is the loop around it.

## Six cities

Each city trains a first model on a clean season, then runs week by week into the
season that spoils it. Every setting is the same in all six.

| | PM2.5 swing (µg/m³) | weeks | retrains | promoted | retraining, week by week | as first published |
|---|---|---|---|---|---|---|
| **Delhi** | 42 → 127 | 39<!--fig:delhi.weeks--> | 8<!--fig:delhi.retrains--> | 6<!--fig:delhi.promotions--> | **+22.8% [+11.8, +36.4]**<!--fig:delhi.acted--> | +49.4% [+33.7, +61.8]<!--fig:first.delhi.acted--> |
| **Santiago** | 18 → 94 | 21<!--fig:santiago.weeks--> | 13<!--fig:santiago.retrains--> | 7<!--fig:santiago.promotions--> | **+17.5% [+8.6, +37.2]**<!--fig:santiago.acted--> | +17.3% [+9.0, +37.0]<!--fig:first.santiago.acted--> |
| **Johannesburg** | 23 → 87 | 19<!--fig:joburg.weeks--> | 12<!--fig:joburg.retrains--> | 3<!--fig:joburg.promotions--> | **+9.7% [+4.6, +14.0]**<!--fig:joburg.acted--> | +14.9% [+8.7, +21.4]<!--fig:first.joburg.acted--> |
| **Melbourne** | 5 → 15 | 30<!--fig:melbourne.weeks--> | 9<!--fig:melbourne.retrains--> | 4<!--fig:melbourne.promotions--> | **+5.6% [+2.0, +6.6]**<!--fig:melbourne.acted--> | +1.2% [−0.6, +3.0]<!--fig:first.melbourne.acted--> |
| **Kraków** | 8 → 57 | 48<!--fig:krakow.weeks--> | 21<!--fig:krakow.retrains--> | 7<!--fig:krakow.promotions--> | +5.7% [−14.8, +26.8]<!--fig:krakow.acted--> | +6.5% [−15.3, +27.5]<!--fig:first.krakow.acted--> |
| **Los Angeles** | 15 → 29 | 36<!--fig:la.weeks--> | 7<!--fig:la.retrains--> | 5<!--fig:la.promotions--> | −3.5% [−25.6, +3.0]<!--fig:la.acted--> | −13.4% [−36.9, −3.8]<!--fig:first.la.acted--> |

Bold where the interval excludes zero. The last column is what this table said
until 2026-11-10, under a default regularisation setting and one retrain alarm.
Melbourne's rise is the one to read with care: each figure is measured against
that configuration's own first model, and Melbourne's error over the whole replay
barely moved.

The intervals are wide because a year of weekly windows in one city is far less
information than it looks. Kraków's 47<!--fig:krakow.acted.n--> comparisons carry about
5.4<!--fig:krakow.acted.n_eff--> independent observations, and Los Angeles's
35<!--fig:la.acted.n--> about 3.3<!--fig:la.acted.n_eff-->.

## Five fixes, measured on two loops

Five fixes were built and replayed across all six cities twice: on the loop as it
first shipped, and on the loop as it ships, with the expected results
[written down first](docs/registrations/2026-11-10-fixes-on-the-loop-as-shipped.md).
A setting counts only where it clears zero, in the same direction, in two cities.

| the fix | what it changes | on the first loop | on the loop as it ships |
|---|---|---|---|
| a second retrain alarm | when the loop notices | a gain or a harm, depending on the setting | no accuracy effect in two cities; shorter silences: **kept** |
| a longer exam | how much evidence one exam has | nothing | nothing |
| a re-certification schedule | how often the exam is sat | nothing | nothing |
| a confidence-aware gate | how hard one exam is to pass | **harmful** in two cities | nothing; a harm in one city |
| rollback | whether the result can be undone | nothing | nothing |

The second alarm is kept as a bound on how long the loop can go deaf, with no
claim that it improves forecasts ([D1](docs/DECISIONS.md)). On the tuned loop it
improves Los Angeles by +6.6% [−1.1, +9.3]<!--fig:floor.la.-0.50.acted-->, where the first
loop showed +11.8% [+2.1, +17.2]<!--fig:first.floor.la.-0.50.acted-->. One notch more
aggressive it costs Kraków **−20.1% [−47.5, −5.5]**<!--fig:floor.krakow.-0.25.acted-->.

It was then tested on nine cities outside the six, with the cities, their replay
windows and four predictions
[committed before their data was fetched](docs/registrations/2026-11-10-second-alarm-on-held-out-cities.md).
It shortened the longest silence wherever it fired. Its effect on accuracy had no
consistent sign: +8.8% [−1.4, +11.1]<!--fig:heldout.istanbul.acted--> in Istanbul,
−6.0% [−19.0, +9.2]<!--fig:heldout.dhaka.acted--> in Dhaka.

"Nothing" in the table means not detectable: a fix worth a few percent would not
show in either run. The workings are in
[evaluation.md](docs/evaluation.md#the-five-fixes-and-the-model-ablation-on-the-loop-as-it-ships).

## How the numbers are kept honest

- **Every figure carries an interval.** A moving-block bootstrap, because weekly
  windows overlap and are not independent ([stats.py](src/driftloop/stats.py)).
- **Every figure on these pages is checked.** Each carries a key into
  `outputs/figures.json`, in a comment that does not render, and CI fails when a
  page disagrees with the outputs.
- **Predictions come before results.** The re-run of the fixes and the held-out
  test each have a registration under [docs/registrations](docs/registrations).
- **A corrected figure keeps its predecessor.** [HISTORY.md](docs/HISTORY.md) is
  the trail, and [DECISIONS.md](docs/DECISIONS.md) records what was decided on
  the evidence and when.

## Serving the champion

The loop's whole output is one label. `champion` points at whichever model last
passed its exam, and the API serves whatever wears that label, so promoting a
model is deploying it.

![The serving API](docs/images/serving.png)

```bash
python scripts/serve.py --city krakow      # http://localhost:8000/docs
docker build -t drift-serve --build-arg CITY=delhi .
docker run -p 8000:8000 drift-serve
```

`/model` reports the window the serving model was trained on. Nothing polls and
nothing calls `/reload` for you: a new model is picked up when someone asks.
Predictions come back twice, `pm25` floored at zero and `pm25_raw` as the model
said it.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"

python scripts/run_openmeteo.py --fresh     # replay all six cities (--city krakow|santiago|delhi|joburg|melbourne|la)
python scripts/benchmark.py                 # baselines, and the penalty sweep
python scripts/uncertainty.py               # every headline with its interval
python scripts/figures.py                   # -> outputs/figures.json, and any page it contradicts
python scripts/build_site.py                # -> site/data.json

streamlit run dashboard/app.py              # the dashboard
python scripts/serve.py --city krakow       # serve the champion on :8000
pytest -q && ruff check .                   # what CI gates on
```

The experiments, each of which replays cities from the committed data:

```bash
python scripts/sweep_skill_floor.py         # one of five fix sweeps (--loop first for the loop as first shipped)
python scripts/score_fixes.py               # read all five sweeps on both loops under one rule
python scripts/ablate_model.py              # the same loop under a model that can bend
python scripts/rebaseline.py                # the first loop against the loop as it ships
python scripts/held_out_cities.py           # the second alarm on nine cities outside the six
python scripts/held_out_premium.py          # what retraining is worth on the same nine
```

## Where things are

```
src/driftloop/    the loop, drift measures, model, scoring, statistics, serving
scripts/          replays, experiments, figures, the site build, the weekly run
site/             the published page: index.html, compare.html and their scripts
dashboard/        the Streamlit app
docs/             methodology, evaluation, decisions, history, registrations
tests/            no-leak guards, drift maths, the loop's rules, published figures
```

- **[methodology.md](docs/methodology.md)** — how the loop works: the model, the
  inputs, what the drift index computes, the window layout, the guards against
  grading a model on data it has seen.
- **[evaluation.md](docs/evaluation.md)** — whether it works: the current results,
  the re-run of the fixes, the held-out test, and the limitations.
- **[evaluation-first-loop.md](docs/evaluation-first-loop.md)** — the full analysis
  of the loop as it first shipped, kept as the reasoning that led to the change.
- **[DECISIONS.md](docs/DECISIONS.md)** and **[HISTORY.md](docs/HISTORY.md)** — what
  was decided, and how every number changed.
