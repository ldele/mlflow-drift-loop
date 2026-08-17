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
   first appears. Delhi runs **+49.4% [+34, +62]** better week by week. But four
   fifths of that was the shipped model being under-regularised and linear. Tune
   it properly, put a tuned tree against it, and the premium falls to **+9.7%
   [+8.0, +26.0]** — a fifth of the size, still clear of zero.

2. **And it costs where the air did not.** Los Angeles is the control, and the
   loop leaves it **13.4% worse [−37, −4]**. The interval excludes zero, so that
   is a result rather than an anecdote — and it is the fragile half of the
   headline. It rests on a single promotion, won on an exam margin of +21.9%
   [−6.9, +32.3]: an exam that could not establish the challenger was better at
   all. Where the seasonal swing is small, the gate has almost no signal to
   select on.

3. **The retrain alarm goes deaf.** It grades the model against its own past, and
   every promotion resets that bar upward. Kraków's ratchets so high that its last
   30 weeks cannot fire at any error; Los Angeles is silent for 35 runs out of 36.
   A second, model-independent trigger fixes that: at its cautious setting it
   leaves five cities bit-identical and improves Los Angeles by **+11.8% [+2.1,
   +17.2]** — the same city finding 2 says retraining harms, because what harmed
   it was one bad model left serving far too long.

4. **Five fixes were built and measured. None pays, and one is harmful.** They
   fail for one reason, and it took all five to see it: the loop retries until a
   challenger passes, so raising any bar buys more attempts and a luckier winner
   rather than fewer bad promotions.

5. **A seven-day exam certifies a model for about five weeks.** Across 25
   short-serving promotions it delivers on its promise: +12.4% promised,
   **+9.8% [+6.4, +14.0]** delivered. Beyond twenty weeks it reverses — though
   that group is three promotions, and is reported as three.

**Two of the six cities show no measurable effect at all** once intervals are
attached: Kraków and Melbourne. Both were previously reported here as small
positive results, and neither is distinguishable from nothing.

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

## Where the loop is wrong

**The expensive signal is measurably the wrong shape.** It compares the model
against *its own* error at training time, and every promotion resets that
comparison. Retrains fire in the dirty season, so each new model inherits a
higher bar than the one it replaced and the bar ratchets upward. Kraków's rises
from 3.7 to 45.8 µg/m³, after which nothing can cross it: the last 30 of its 48
weeks are a 210-day-old model reported as healthier than it has ever been, while
its skill against a plain 30-day daily profile is the worst it has ever been.
Finding that is what the site is built to do.

Then five fixes were built, because each failure named the next thing to try.
Every one was replayed across all six cities against the shipped loop, week by
week, with intervals.

| the fix | what it changes | what it did |
|---|---|---|
| a second retrain trigger | when the loop notices | helps the one deaf city, nothing elsewhere |
| a longer exam | how much evidence one exam has | nothing measurable anywhere |
| a re-certification schedule | how often the exam is sat | bounds staleness, buys no accuracy |
| a confidence-aware gate | how hard one exam is to pass | **actively harmful** |
| rollback | whether the result can be undone | no harm, and no proof of benefit |

The one fix that moved anything is worth a note, because the obvious version of
it cannot be built. An absolute error floor needs to be low enough to wake the
quietest city and high enough not to retrain the dirtiest one every week, and the
gap between them is empty. A *model-independent* yardstick works: the trigger
also fires when skill against the daily profile drops below a floor, which
nothing about promoting a model can move. Kraków's longest silence falls from 30
weeks to 5. It still ships switched off — flipping a default is a decision rather
than a finding — but the case for switching it on is now the stronger one.

**The other four fail for one reason.** The loop is a retry procedure: it keeps
training challengers and sitting exams until one passes, so the model it promotes
always carries the luck of whichever attempt cleared the bar. Raising the bar does
not buy fewer bad promotions, it buys more attempts and a luckier winner — which
is why the strictest gate is the harmful one. Nothing about *when* the loop acts,
*how hard* it judges, or *whether it can undo the result* prices the number of
attempts.

Underneath all five is one measurement limit. Los Angeles's single promotion was
won by +21.9% [−6.9, +32.3], rolled back when re-judged at 14 days and kept when
re-judged at 21 or 28: the same decision, three windows, opposite answers. A
fortnight of hourly air cannot resolve the difference the loop is asking about.
That is a limit of the problem rather than a bug in the code, and a better answer
than a mechanism that happened to work — but only because the fixes were run
instead of argued. The workings are in [evaluation.md](docs/evaluation.md), and
what was decided on top of them in [DECISIONS.md](docs/DECISIONS.md).

## Six cities that disagree

Each city trains a model on a clean season, then runs week by week into the
season that ruins it. Every setting is identical across all six, so where two
cities behave differently, it is their air that differs and not their tuning.

Two columns for retraining, because one of them lies. The first compares median
error across the whole replay against never retraining. It is unpaired, so in a
city that promotes nothing until week 14 of 20, most windows compare the first
model against itself and the answer collapses toward zero. The second holds the
window fixed and compares the two models week by week, over the weeks a retrained
model was actually serving.

| | how bad it gets (µg/m³) | weeks | retrains | shipped | across the replay | week by week |
|---|---|---|---|---|---|---|
| **Delhi** | 42 → 127, crop burning after the monsoon | 39 | 9 | 8 | +43.7% [+29, +64] | **+49.4% [+34, +62]** |
| **Santiago** | 18 → 94, winter smog trapped in a bowl | 21 | 13 | 7 | +16.8% [−0, +43] | **+17.3% [+9, +37]** |
| **Kraków** | 8 → 57, coal heating in a valley | 48 | 14 | 7 | +0.2% [−57, +36] | +6.5% [−15, +28] |
| **Johannesburg** | 23 → 87, winter coal smoke | 19 | 11 | 3 | −0.0% [−0, +21] | **+14.9% [+9, +21]** |
| **Melbourne** | 5 → 15, winter wood heaters | 30 | 8 | 4 | +0.1% [−3, +7] | +1.2% [−1, +3] |
| **Los Angeles** | 15 → 29, a mild winter bump | 36 | 1 | 1 | −7.9% [−29, +0] | **−13.4% [−37, −4]** |

Bold where the interval excludes zero. In Delhi, where the air transforms,
keeping the model fresh roughly halves its error. In Los Angeles the loop fires
once in thirty-six weeks, and the median week runs 13.4% behind leaving the model
alone — so the control does not merely fail to benefit, it is measurably harmed.
Los Angeles earns its place by failing.

Kraków is the sharper lesson in the other direction, and it is a lesson about
what a week of weather is worth. Its interval excludes zero under an ordinary
bootstrap and includes it once the autocorrelation between overlapping weeks is
respected — which is why the
[block-length sweep](docs/evaluation.md#the-block-length-is-a-knob-so-here-is-the-sweep)
is published rather than summarised. Its forty-seven comparisons carry about
**five** independent observations; Los Angeles's thirty-five carry **three**. A
long replay of a persistent process is far less informative than its length
suggests, and that one fact explains every wide interval above.

Johannesburg is where the promotion gate does its most visible work, and where
the unpaired number misleads hardest. Eleven retrains, three shipped, the other
eight thrown away for failing the margin. Across the replay that reads as 0.0%.
Week by week, in the six weeks a retrained model was serving, it beat the
original in all six by a median of 14.9% [+9, +21]. Six weeks is the thinnest
evidence among the four surviving cities, and 6-of-6 is worth [61%, 100%] rather
than certainty — a Wilson interval, because a bootstrap cannot express doubt at a
boundary.

Half these cities are dirtiest in June to August and the other half in December
to January, which is how you can tell the thresholds are not secretly encoding a
season.

**How the exam can be checked at all.** Every promotion left a prediction behind
— the margin the challenger won by — so the gate can be marked against what each
winner went on to deliver. That is where finding 5 comes from, and the three
long-serving promotions that reverse it are worth naming individually: −8.0%,
−6.7% and −3.6%. All negative, none near zero, no estimable magnitude. Tripling
the exam does not help, which is how you can tell this is drift rather than a
small sample. And the models that serve half a year are the ones the ratcheted
trigger can no longer replace, so the two faults compound.

[evaluation.md](docs/evaluation.md) has the city-by-city detail, how the model
scores against four "do nothing clever" baselines, the gate calibration in full,
and a controlled experiment showing that each alarm responds to its own cause and
ignores the other.

### What a full year exposed

The cities originally stopped at their winter peak. On that half of the story,
retraining looked like a clear win everywhere. Running them through the return
trip, as the air gets clean again, reversed the sign: retraining came out 29.6%
worse in Kraków and 7.2% worse in Delhi.

The fault was in the retraining rule, not the machinery. A replacement trained on
the last 45 days only ever sees one season. It is excellent in the season it was
born in and wrong as soon as the year turns. Widening that window to 180 days
took Delhi from −7.2% to +43.7%.

This is the most useful finding in the repository. The loop was behaving
correctly the whole time, faithfully shipping replacements that won their exam
and then aged badly, and only a full year of data made it visible.

## Is the finding about the world, or about a linear model?

The shipped model is a Ridge whose regularisation is nearly inert, which is close
to a plain linear projection. So every result above was compatible with a duller
explanation: linear models misspecify seasonal structure, and refitting is how
you paper over it.

Tuning a Ridge and a gradient-boosted challenger on the same protocol separates
the three effects. Delhi's +49.4% decomposes into **21.6 points lost to the
shipped `alpha=1.0`, 18.0 points lost to linearity, and +9.7% left over**. The
largest single component of this project's headline number was a library default
nobody chose. Los Angeles's harm stops being measurable under the better model:
−6.9% [−13.2, +2.8].

**The confound was real and larger than this page used to claim.** An earlier
version ran the same check with a tree at library defaults, watched it lose to
the Ridge, and read that as evidence the problem was close to linear. The tree
was simply undertrained — tuning it is worth 26.5%. A check that could not run
had been reported as evidence for the thing it failed to test.
[The ablation in full](docs/evaluation.md#is-the-finding-about-the-world-or-about-a-linear-model).

The model is weak, and that belongs up front rather than in a footnote. Guessing
an hour's pollution from a week-old weather forecast is hard, and the numbers
show it. What is worth looking at is the machinery around the model — which the
ablation confirms would be unchanged if you dropped in something far better.

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
                  benchmark, serving
scripts/          run_openmeteo · benchmark · uncertainty · ablate_model · build_site ·
                  run_scheduled · sweep_knobs · sweep_skill_floor · sweep_holdout · serve
site/             committed shell (index.html + app.js, compare.html + compare.js,
                  shared.css) + generated data.json
dashboard/        Streamlit app and shared chart theme
docs/             methodology · evaluation · wireframes each UI was built from
tests/            data contract, drift math, no-leak guards, baseline fairness,
                  retrospective scoring, serving, charts, site assets
```

- **[methodology.md](docs/methodology.md)** — how it works: what a Ridge does and
  why it is barely doing it here, the features and the physics behind each one,
  what PSI computes and where it stops meaning anything, the window layout, the
  guards against cheating, and a reading list.
- **[evaluation.md](docs/evaluation.md)** — whether it works: per-city results,
  the baselines, the controlled experiment, and the limitations.
- **[DECISIONS.md](docs/DECISIONS.md)** — the calls made on top of the findings,
  with the evidence and the date. One is open: whether the skill floor should
  still ship switched off now that the evidence has reversed.
- **[stats.py](src/driftloop/stats.py)** — how much to believe it: why the weekly
  windows are not independent observations, why that needs a block bootstrap
  rather than an ordinary one, and the two places the bootstrap has to admit it
  cannot help.
