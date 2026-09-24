# Air quality drift watch

[![ci](https://github.com/ldele/mlflow-drift-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/ldele/mlflow-drift-loop/actions/workflows/ci.yml)

Models go stale. The world they were trained on moves on, and usually nobody
notices until something downstream breaks.

This one notices on its own. Every week it marks its own homework, and when the
model in service starts slipping it trains a replacement — then refuses to ship
that replacement unless it beats the model already doing the job.

The job: forecast how dirty a city's air will be a week from now, from nothing
but the weather forecast. Six cities on six continents, each replayed week by
week through a full year, from 2025 into 2026.

**[See it running →](https://ldele.github.io/mlflow-drift-loop/)**

![The dashboard](docs/images/dashboard.png)

## What it found

Five findings, each with the interval that decides whether to believe it. Every
percentage is a 95% moving-block bootstrap over autocorrelated weekly windows —
[why that and not an ordinary one](docs/evaluation.md#what-the-intervals-cost-this-page).

1. **Retraining pays where the air really changed**, and by much less than it
   first appears. Delhi runs **+49.4% [+34, +62]**<!--fig:delhi.acted--> better week by week. But four
   fifths of that was the shipped model being under-regularised and linear. Tune
   it properly, put a tuned tree against it, and the premium falls to
   **+9.7% [+8.0, +26.0]**<!--fig:ablation.delhi.gbm_tuned--> — a fifth of the size, still clear of zero.

2. **And it costs where the air did not — for the model that ships.** Los Angeles
   is the control, and week by week the loop leaves it
   **−13.4% [−37, −4]**<!--fig:la.acted--> behind never retraining. That rests on a single
   promotion, won on an exam margin of +21.9% [−6.9, +32.3]<!--nofig: printed by scripts/sweep_promotion_confidence.py, not written to outputs/-->,
   and it does not survive a better model: under the tuned tree it reads
   −6.9% [−13.2, +2.8]<!--fig:ablation.la.gbm_tuned-->, which is not distinguishable from zero. Where the
   seasonal swing is small, the gate has almost no signal to select on.

3. **The retrain alarm goes deaf.** It grades the model against its own past, and
   every promotion resets that bar upward. Kraków ratchets so high that its last
   30<!--fig:floor.krakow.off.silence--> weeks cannot fire at any error; Los Angeles is silent for
   35<!--fig:floor.la.off.silence--> runs out of 36<!--fig:la.weeks-->. A second, model-independent trigger helps
   one city: at its cautious setting it leaves five cities bit-identical and, in
   the 25<!--fig:floor.la.-0.50.differing--> weeks it acts, improves Los Angeles by
   **+11.8% [+2.1, +17.2]**<!--fig:floor.la.-0.50.acted--> — because what harmed that city was one
   bad model left serving far too long.

4. **Five fixes were built and measured. None pays, and one is harmful.** They
   fail for one reason, and it took all five to see it: the loop retries until a
   challenger passes, so raising any bar buys more attempts and a luckier winner
   rather than fewer bad promotions.

5. **A seven-day exam certifies a model for about five weeks.** Across
   25<!--fig:gate.short.n--> short-serving promotions it delivers on its promise:
   +12.4%<!--fig:gate.short.exam--> promised, **+9.8% [+6.4, +14.0]**<!--fig:gate.short.delivered--> delivered.
   Beyond twenty weeks it reverses — though that group is 3<!--fig:gate.long.n--> promotions, and
   is reported as three.

**Two of the six cities show no measurable effect at all** once intervals are
attached: Kraków and Melbourne. Both were previously reported here as small
positive results, and neither is distinguishable from nothing.

The model is weak, and that belongs up front rather than in a footnote. Guessing
an hour's pollution from a week-old weather forecast is hard. What is worth
looking at is the machinery around the model, which
[the ablation](docs/evaluation.md#is-the-finding-about-the-world-or-about-a-linear-model)
confirms would be unchanged if you dropped in something far better.

## The loop

One run a week, four steps:

| | | |
|---|---|---|
| **1. Mark its homework** | check the last 14 days of forecasts against what the air did | |
| **2. Look for trouble, two ways** | has the weather stopped looking like what the model learned from, and separately, is the model's error rising? | two signals, on purpose |
| **3. Train a rival** | error is 1.25× what it used to be, so train a fresh model on the last 180 days | only if step 2 says so |
| **4. Make it earn the job** | both sit the same exam, a week of air neither has seen | the newcomer wins by 5% or it is thrown away |

Why two signals and not one? The first watches only the weather coming in, so it
can raise a hand immediately, without waiting to find out whether the model was
wrong. But "the world looks different" is not the same as "the model is failing".
Kraków shows the gap: through the summer its weather drifts further from training
than anywhere else here, while the model quietly gets better. So the cheap alarm
watches, and only the expensive one — the one that asks whether we got this wrong
— can authorise spending money on a retrain.

Nothing is graded on work it has already seen. The replacement trains on a window
that stops before the exam, the incumbent was trained long before it, and both
are marked on the same unseen week. Details in
[methodology.md](docs/methodology.md).

## Five fixes, and why four of them fail

Each failure named the next thing to try, so five fixes were built. Every one was
replayed across all six cities against the shipped loop, week by week, with
intervals.

| the fix | what it changes | what it did |
|---|---|---|
| a second retrain trigger | when the loop notices | helps the one deaf city, nothing elsewhere |
| a longer exam | how much evidence one exam has | nothing measurable anywhere |
| a re-certification schedule | how often the exam is sat | bounds staleness, buys no accuracy |
| a confidence-aware gate | how hard one exam is to pass | **actively harmful** |
| rollback | whether the result can be undone | no harm, and no proof of benefit |

The trigger that helps fires when skill against a plain 30-day daily profile
drops below a floor, which nothing about promoting a model can move. Set high
enough, it wakes Kraków too: at `skill < 0` the longest silence there falls from
30<!--fig:floor.krakow.off.silence--> weeks to 5<!--fig:floor.krakow.+0.00.silence-->. It then costs Kraków accuracy in the weeks it
acts, clearest at `skill < −0.25`: **−24.0% [−39.4, −5.5]**<!--fig:floor.krakow.-0.25.acted-->. Only the
cautious `skill < −0.5` harms no city here. It still ships switched off, and
whether it should is open ([D1](docs/DECISIONS.md)).

The other four fail for the reason finding 4 gives. Underneath all five is one
measurement limit: Los Angeles's single promotion was rolled back when re-judged
at 14 days and kept when re-judged at 21 or 28 — the same decision, three
windows, opposite answers. A fortnight of hourly air cannot resolve the
difference the loop is asking about. The workings are in
[evaluation.md](docs/evaluation.md), and what was decided on top of them in
[DECISIONS.md](docs/DECISIONS.md).

## Six cities that disagree

Each city trains a model on a clean season, then runs week by week into the
season that ruins it. Every setting is identical across all six, so where two
cities behave differently, it is their air that differs and not their tuning.

| | how bad it gets (µg/m³) | weeks | retrains | shipped | across the replay | week by week |
|---|---|---|---|---|---|---|
| **Delhi** | 42 → 127, crop burning after the monsoon | 39<!--fig:delhi.weeks--> | 9<!--fig:delhi.retrains--> | 8<!--fig:delhi.promotions--> | +43.7% [+29, +64]<!--fig:delhi.replay--> | **+49.4% [+34, +62]**<!--fig:delhi.acted--> |
| **Santiago** | 18 → 94, winter smog trapped in a bowl | 21<!--fig:santiago.weeks--> | 13<!--fig:santiago.retrains--> | 7<!--fig:santiago.promotions--> | +16.8% [−0, +42]<!--fig:santiago.replay--> | **+17.3% [+9, +37]**<!--fig:santiago.acted--> |
| **Kraków** | 8 → 57, coal heating in a valley | 48<!--fig:krakow.weeks--> | 14<!--fig:krakow.retrains--> | 7<!--fig:krakow.promotions--> | +0.2% [−57, +36]<!--fig:krakow.replay--> | +6.5% [−15, +28]<!--fig:krakow.acted--> |
| **Johannesburg** | 23 → 87, winter coal smoke | 19<!--fig:joburg.weeks--> | 11<!--fig:joburg.retrains--> | 3<!--fig:joburg.promotions--> | −0.0% [−0, +21]<!--fig:joburg.replay--> | **+14.9% [+9, +21]**<!--fig:joburg.acted--> |
| **Melbourne** | 5 → 15, winter wood heaters | 30<!--fig:melbourne.weeks--> | 8<!--fig:melbourne.retrains--> | 4<!--fig:melbourne.promotions--> | +0.1% [−3, +7]<!--fig:melbourne.replay--> | +1.2% [−1, +3]<!--fig:melbourne.acted--> |
| **Los Angeles** | 15 → 29, a mild winter bump | 36<!--fig:la.weeks--> | 1<!--fig:la.retrains--> | 1<!--fig:la.promotions--> | −7.9% [−29, +0]<!--fig:la.replay--> | **−13.4% [−37, −4]**<!--fig:la.acted--> |

Bold where the interval excludes zero. The two columns disagree, and the second
is the one to trust: "across the replay" compares median error against never
retraining without holding the week fixed, so where both are dominated by the
same seasonal swing it mostly measures the season. "Week by week" compares the
two models on the same window, over the weeks a retrained model was serving.

The intervals are wide because a long replay of a persistent process is far less
informative than its length suggests. Kraków's 47<!--fig:krakow.acted.n--> weekly comparisons
carry about 5.5<!--fig:krakow.acted.n_eff--> independent observations, and Los Angeles's
35<!--fig:la.acted.n--> about 2.9<!--fig:la.acted.n_eff-->.

[evaluation.md](docs/evaluation.md) has the city-by-city detail, the model against
four "do nothing clever" baselines, the gate calibration in full, and a
controlled experiment showing that each alarm responds to its own cause and
ignores the other. How the headline got here, including the full year of data
that reversed its sign and the ablation that cut it to a fifth, is in
[HISTORY.md](docs/HISTORY.md).

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
window it was trained on, so you can see how stale the model answering you is.

Nothing polls, and nothing calls `/reload` for you: a new model is picked up when
someone asks for it, because swapping the model under live traffic without anyone
asking is worse than serving a slightly stale one. Predictions come back twice:
`pm25` floored at zero for whoever is consuming it, and `pm25_raw` as the model
said. A clamp that hides what your model is doing is how you stop noticing it.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"

python scripts/run_openmeteo.py --fresh     # all six cities (--city krakow|santiago|delhi|joburg|melbourne|la)
python scripts/benchmark.py                 # baselines + alpha sweep
python scripts/uncertainty.py               # every headline, with its confidence interval
python scripts/ablate_model.py              # is the finding about the world or the model class?
python scripts/uncertainty.py --sensitivity # ... and how much the block length moves it
python scripts/sweep_skill_floor.py         # does waking the retrain trigger help? (in one city)
python scripts/sweep_holdout.py             # does a longer promotion exam help? (no)
python scripts/figures.py                   # -> outputs/figures.json, and which pages it contradicts
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
src/driftloop/    config, data sources, drift math, model, loop, retrospect, stats,
                  benchmark, serving, figures
scripts/          run_openmeteo · benchmark · uncertainty · ablate_model · rebaseline ·
                  figures · build_site · run_scheduled · sweep_knobs · sweep_skill_floor ·
                  sweep_holdout · serve
site/             committed shell (index.html + app.js, compare.html + compare.js,
                  shared.css) + generated data.json
dashboard/        Streamlit app and shared chart theme
docs/             methodology · evaluation · decisions · history · wireframes each UI
                  was built from
tests/            data contract, drift math, no-leak guards, baseline fairness,
                  retrospective scoring, serving, charts, site assets, published figures
```

- **[methodology.md](docs/methodology.md)** — how it works: what a Ridge does and
  why it is barely doing it here, the features and the physics behind each one,
  what PSI computes and where it stops meaning anything, the window layout, the
  guards against cheating, and a reading list.
- **[evaluation.md](docs/evaluation.md)** — whether it works: per-city results,
  the baselines, the controlled experiment, and the limitations.
- **[DECISIONS.md](docs/DECISIONS.md)** — the calls made on top of the findings,
  with the evidence and the date. Two are open: whether the skill floor should
  still ship switched off now that the evidence has reversed (D1), and whether
  the shipped `alpha` should stay a library default (D5).
- **[HISTORY.md](docs/HISTORY.md)** — how every number above changed, and what
  the correction was each time. The first headline was measured four flattering
  ways at once, and the claim ended at about a fifth of where it started.
- **[stats.py](src/driftloop/stats.py)** — how much to believe it: why the weekly
  windows are not independent observations, why that needs a block bootstrap
  rather than an ordinary one, and the two places the bootstrap has to admit it
  cannot help.

Every figure on these pages that an output file holds carries a key into
`outputs/figures.json`, in a comment that does not render, and
`tests/test_figures.py` fails when a page disagrees with it. After re-running any
script above, `python scripts/figures.py` rewrites that file and lists every
sentence the new numbers contradict.
