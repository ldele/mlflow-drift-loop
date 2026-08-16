# Air quality drift watch

[![ci](https://github.com/ldele/mlflow-drift-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/ldele/mlflow-drift-loop/actions/workflows/ci.yml)

Models go stale. The world they were trained on moves on, and usually nobody
notices until something downstream breaks.

This system notices on its own. Every week it marks its own homework. When the
model it is running starts slipping, it trains a replacement, and it refuses to
ship that replacement unless it beats the one already in service. Six cities are
replayed week by week across a full year of weather, from 2025 into 2026.

The job it does is forecasting how dirty a city's air will be a week from now,
using nothing but the weather forecast. Six cities on six continents.

**[See it running →](https://ldele.github.io/mlflow-drift-loop/)**

![The dashboard](docs/images/dashboard.png)

## What it found, in sixty seconds

Five findings, each with the interval that decides whether to believe it. Every
percentage below is a 95% moving-block bootstrap over autocorrelated weekly
windows; details in [evaluation.md](docs/evaluation.md#what-the-intervals-cost-this-page).

1. **Retraining pays where the world really moved, and by much less than it
   first appears.** Delhi +49.4% [+33.7, +61.8] week by week for the model that
   ships. Tune that model properly and swap it for a better one and the premium
   falls to **+9.7% [+8.0, +26.0]**, still clear of zero. The effect is real;
   four fifths of it was the shipped model being under-regularised and linear
   (see the caveat below).
2. **And it costs where it did not.** Los Angeles, the control, −13.4%
   [−36.9, −3.8]. The interval excludes zero, so the control is a result rather
   than an anecdote. This is the fragile half of the headline, in two ways. Swap
   the Ridge for a tree and the harm stops being measurable (see the caveat
   below): with a properly tuned tree the harm is −6.9% [−13.2, +2.8] and no
   longer clear of zero. And the whole −13.4% rests on a **single promotion**,
   made on an exam margin of +21.9% [−6.9, +32.3]: an exam that could not
   establish the challenger was better at all. Where the seasonal swing is small
   the exam has little signal to select on, so part of "retraining costs where
   the world did not move" is a statement about the gate's precision rather than
   about the world.
3. **The retrain trigger goes deaf**, because it measures a model against its
   own past and every promotion resets that bar upward. The fix was built and
   measured. At its cautious setting it leaves five cities bit-identical and
   improves the deafest by +11.8% [+2.1, +17.2]. Attaching intervals turned
   that from a wash into a result, and into an argument for switching it on.
4. **Five fixes were built and measured. None pays, and one is harmful.** A
   longer exam does nothing anywhere: eighteen paired comparisons, not one clear
   of zero. A re-certification schedule bounds staleness and buys no accuracy. A
   confidence-aware gate is worse than the gate it replaces. Making promotion
   reversible does no harm and proves nothing. They fail for one reason, and it
   took all five to see it: the loop retries until a challenger passes, so
   raising any bar buys more attempts and a luckier winner.
5. **The promotion gate has a shelf life.** It is honest for about five weeks
   (+9.8% [+6.4, +14.0] delivered across 25 promotions) and reverses beyond
   twenty, though that second group is three promotions, and is reported as
   three promotions.

One caveat that belongs up here rather than in a footnote. **Two of the six
cities show no measurable effect at all** once intervals are attached: Kraków
and Melbourne.

A second caveat has now been tested properly, and it takes most of finding 1
away. The shipped model is a Ridge at `alpha=1.0`, a library default its own
sweep beats by 11.9% in Delhi, so "retraining pays" was entangled with both
"this model is under-regularised" and "linear models need refitting to track
seasonality". Tuning the Ridge and a gradient-boosted challenger on the same
protocol separates the three. Delhi's +49.4% decomposes into **21.6 points lost
to the shipped `alpha`, 18.0 points lost to linearity, and +9.7% left over**.
The effect is real and it is a fifth of what this page used to claim. Los
Angeles's harm stops being measurable under the better model.

An earlier version of this README reported the opposite, on the strength of a
test that could not run: a tree at library defaults lost to the Ridge, and that
was read as evidence the confound was small when it was really evidence the
challenger was undertrained. Tuning it is worth 26.5%.
[The ablation in full](docs/evaluation.md#is-the-finding-about-the-world-or-about-a-linear-model).

## The loop

One run a week, four steps:

| | | |
|---|---|---|
| **1. Mark its homework** | check the last 14 days of forecasts against what the air did | |
| **2. Look for trouble, two ways** | has the weather stopped looking like what the model learned from, and separately, is the model's error rising? | two signals, on purpose |
| **3. Train a rival** | error is 1.25× what it used to be, so train a fresh model on the last 180 days | only if step 2 says so |
| **4. Make it earn the job** | both sit the same exam, a week of air neither has seen | the newcomer wins by 5% or it is thrown away |

Why two signals and not one? The first looks only at the weather coming in, so it
can raise a hand immediately, without waiting to find out whether the model was
wrong. But "the world looks different" is not the same as "the model is failing".
Kraków shows the gap: through the summer its weather drifts further from training
than anywhere else here, while the model quietly gets better. So the cheap alarm
watches, and only the expensive one, the one that asks whether we got this wrong,
can authorise spending money on a retrain.

Nothing is graded on work it has already seen. The replacement trains on a window
that stops before the exam, the incumbent was trained long before it, and both
are marked on the same unseen week. Details in
[methodology.md](docs/methodology.md).

### The alarm that goes deaf

**The expensive signal is measurably the wrong shape.** It compares the model
against *its own* error at training time, and every promotion resets that
comparison. Retrains fire in the dirty season, so each new model inherits a
higher bar than the one it replaced and the bar ratchets upward. It can step
down, and does once in Kraków and three times in Delhi, but only slightly and
only early. Kraków's rises from 3.7 to 45.8 µg/m³, after which nothing can
cross it: the last 30 of
its 48 weeks are a 210-day-old model reported as healthier than it has ever
been, while its skill against a plain 30-day daily profile is the worst it has
ever been. Finding that is what the site is built to do.

Then the fix was built. Two were proposed. An absolute error floor cannot be
built at all: waking the quietest city needs a floor low enough to retrain the
dirtiest one every week, and the gap between them is empty. A model-independent
yardstick can be, and now is. The trigger also fires when skill against the daily
profile drops below a floor, which nothing about promoting a model can move, and
it wakes the alarm as intended: Kraków's longest silence falls from 30 weeks to 5.

Whether that pays was a wash until the arms were compared week by week instead of
median against median. Most weeks a changed trigger leaves the serving model
alone, and those ties dragged every comparison to zero. Measured properly, the
floor has large effects in both directions: pushed hard it makes Kraków 24% worse
and Delhi 9% worse in the weeks it acts, and at its cautious setting it leaves
five cities identical week for week while improving Los Angeles by 11.8%
[+2.1, +17.2]. It still ships switched off, because flipping a default is a
decision rather than a finding, but the case for switching it on is now the
stronger one. Argued in [evaluation.md](docs/evaluation.md).

That pointed at the seven-day exam instead, so the exam was lengthened to 10, 14
and 21 days and replayed the same way. It does not pay either. Over the horizon
it tests the promise gets more honest, and beyond that horizon nothing moves:
every long-serving promotion still delivers a negative margin at every length,
and a longer exam blocks challengers that the cities needing them cannot afford
to lose.

Neither fix addresses the reversal. Waking the trigger helps the city that never
retrains and harms the two where retraining already pays. Lengthening the exam
does nothing measurable anywhere. What both pointed at was a third mechanism:
nothing re-examines a champion on fresh unseen data while it is serving.

### Five fixes, and what they add up to

So that third mechanism was built, and then two more, because each failure named
the next thing to try. Every one of them was replayed across all six cities
against the shipped loop, week by week, with intervals.

| the fix | what it changes | what it did |
|---|---|---|
| a second retrain trigger | when the loop notices | helps the one deaf city, nothing elsewhere |
| a longer exam | how much evidence one exam has | nothing measurable anywhere |
| a re-certification schedule | how often the exam is sat | bounds staleness, buys no accuracy |
| a confidence-aware gate | how hard one exam is to pass | **actively harmful** |
| rollback | whether the result can be undone | no harm, and no proof of benefit |

**They fail for one reason, and it took all five to find it.** The loop is a
retry procedure: it keeps training challengers and sitting exams until one
passes, so the model it promotes always carries the luck of whichever attempt
cleared the bar. Raising the bar does not buy fewer bad promotions, it buys more
attempts and a luckier winner, which is why the strictest gate is the harmful
one. Nothing about *when* the loop acts, *how hard* it judges, or *whether it
can undo the result* prices the number of attempts.

Underneath all five is one measurement limit. The gate compares two models on a
week of hourly air, and Los Angeles's promotion shows what that is worth: an
exam margin of +21.9% [−6.9, +32.3], rolled back when re-judged at 14 days and
kept when re-judged at 21 or 28. The same decision, three windows, opposite
answers. A fortnight cannot resolve the difference the loop is asking about.

That is a limit of the problem rather than a bug in the code, and it is a
better answer than a mechanism that happened to work. None of it would exist if
the fixes had been argued instead of run. The full workings, including the two
decisions this left open, are in
[evaluation.md](docs/evaluation.md) and [DECISIONS.md](docs/DECISIONS.md).

## Six cities that disagree

Retraining pays where the world really moved, and costs where it did not. That
is the finding worth having, and it takes contrasting cities to show it.

Each city trains a model on a clean season, then runs week by week into the
season that ruins it. Every setting is identical across all six, so where two
cities behave differently, it is their air that differs and not their tuning.

Two columns for retraining, because one of them lies. The first compares the
median error across the whole replay against never retraining. It is unpaired,
so in a city that promotes nothing until week 14 of 20 most windows compare the
first model against itself and the answer collapses toward zero. The second holds
the window fixed and compares the two models week by week, over the weeks a
retrained model was serving.

| | how bad it gets (µg/m³) | weeks | retrains | shipped | across the replay | week by week |
|---|---|---|---|---|---|---|
| **Delhi** | 42 → 127, crop burning after the monsoon | 39 | 9 | 8 | +43.7% [+29, +64] | **+49.4% [+34, +62]** |
| **Santiago** | 18 → 94, winter smog trapped in a bowl | 21 | 13 | 7 | +16.8% [−0, +43] | **+17.3% [+9, +37]** |
| **Kraków** | 8 → 57, coal heating in a valley | 48 | 14 | 7 | +0.2% [−57, +36] | +6.5% [−15, +28] |
| **Johannesburg** | 23 → 87, winter coal smoke | 19 | 11 | 3 | −0.0% [−0, +21] | **+14.9% [+9, +21]** |
| **Melbourne** | 5 → 15, winter wood heaters | 30 | 8 | 4 | +0.1% [−3, +7] | +1.2% [−1, +3] |
| **Los Angeles** | 15 → 29, a mild winter bump | 36 | 1 | 1 | −7.9% [−29, +0] | **−13.4% [−37, −4]** |

Bold where the interval excludes zero. **Kraków and Melbourne do not clear that
bar**, and saying so costs less than the alternative: both were previously
reported here as small positive results, and neither is distinguishable from
nothing. Kraków is the sharper lesson. Its interval excludes zero under an
ordinary bootstrap and includes it once the autocorrelation between overlapping
weeks is respected, which is why the
[block-length sweep](docs/evaluation.md#the-block-length-is-a-knob-so-here-is-the-sweep)
is published rather than summarised.

In Delhi, where the air transforms, keeping the model fresh roughly halves its
error. In Los Angeles the loop fires once in thirty-six weeks, and the median
week runs 13.4% behind leaving the model alone, an interval of [−37, −4], so
the control does not merely fail to benefit, it is measurably harmed. Los
Angeles earns its place by failing, and now it does so with evidence.

A note on what these weeks are worth. Kraków's forty-seven comparisons carry
about **five** independent observations once autocorrelation is accounted for,
and Los Angeles's thirty-five carry **three**. Long replays of a persistent
process are much less informative than their length suggests, and that single
fact explains every wide interval above.

Johannesburg is where the promotion gate does its most visible work, and where
the unpaired number misleads hardest. Eleven retrains, three shipped, the other
eight thrown away for failing to clear the margin. Across the replay that reads
as 0.0%. Week by week, in the six weeks a retrained model was serving, it beat
the original in all six by a median of 14.9% [+9, +21]. Six weeks is the
thinnest evidence among the four surviving cities, and a 6-of-6 win rate is
worth [61%, 100%] rather than certainty, because the bootstrap cannot express doubt at
a boundary, so that one is a Wilson interval.

Half these cities are dirtiest in June to August and the other half in December
to January, which is how you can tell the thresholds are not secretly encoding a
season.

**A seven-day exam certifies a model for a month, not for half a year.** Every
promotion left a prediction behind, the margin the challenger won by, so the gate
can be checked against what each winner went on to deliver. Across 25
short-serving promotions it is well calibrated: +12.4% [+9, +17] promised,
+9.8% [+6, +14] delivered, none of them harmful. Beyond twenty weeks it reverses
sign, but that group is three promotions, so it is reported as three:
delivered margins of −8.0%, −6.7% and −3.6%. All negative, none near zero, and
no estimable magnitude. Tripling the exam does not help, which is how you can
tell this is drift rather than a small sample. And the models that serve half a
year are the ones the ratcheted trigger can no longer replace, so the two faults
compound.

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

This is the most useful finding in the repository. The loop was behaving correctly
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
window it was trained on, so you can see how stale the model answering you is.

It does not poll. A new model is picked up when `/reload` is called, because
swapping the model under live traffic without anyone asking is worse than serving
a slightly stale one. Predictions come back twice: `pm25` floored at zero for
whoever is consuming it, and `pm25_raw` as the model said. A clamp that
hides what your model is doing is how you stop noticing it.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"

python scripts/run_openmeteo.py --fresh     # all six cities (--city krakow|santiago|delhi|joburg|melbourne|la)
python scripts/benchmark.py                 # baselines + alpha sweep
python scripts/uncertainty.py               # every headline, with its confidence interval
python scripts/ablate_model.py               # is the finding about the world or the model class?
python scripts/uncertainty.py --sensitivity # ... and how much the block length moves it
python scripts/sweep_skill_floor.py         # does waking the retrain trigger help? (no)
python scripts/sweep_holdout.py             # does a longer promotion exam help? (also no)
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

- **[methodology.md](docs/methodology.md)** covers how it works: what a Ridge
  does and why it is barely doing it here, the features and the physics behind
  each one, what PSI computes and where it stops meaning anything, the window
  layout, the guards against cheating, and a reading list.
- **[evaluation.md](docs/evaluation.md)** covers whether it works: per-city
  results, the baselines, the controlled experiment, and the limitations.

- **[decisions.md](docs/DECISIONS.md)** covers the calls made on top of the
  findings, with the evidence and the date. One is open: whether the skill floor
  should still ship switched off now that the evidence has reversed.
- **[stats.py](src/driftloop/stats.py)** covers how much to believe it: why the
  weekly windows are not independent observations, why that needs a block
  bootstrap rather than an ordinary one, and the two places the bootstrap has to
  admit it cannot help.

The model itself is weak, and that belongs up front rather than in a footnote.
Guessing an hour's pollution from a week-old weather forecast is hard, and the
numbers show it. What is worth looking at is the machinery around the model,
which would be unchanged if you dropped in something far better.

That last sentence was a claim, and it has now been tested. A Ridge whose
regularisation is nearly inert is close to a linear projection, so every result
on this page was compatible with a duller explanation: linear models misspecify
seasonal structure, and refitting is how you paper over it.

Swapping in a gradient-boosted challenger at library defaults produced a model
that fits *worse* in both cities, and that was read here as evidence the problem
is close to linear. It was not. The tree was undertrained: tuning it is worth
26.5% in Delhi, and once both classes are tuned on the same protocol the premium
falls from +49.4% to +9.7%. **The confound was real and larger than this page
claimed**, and the mistake was treating a check that could not run as evidence
for the thing it failed to test.
[Argued in full](docs/evaluation.md#is-the-finding-about-the-world-or-about-a-linear-model).

Getting there meant working around a decision made for elegance. `retrospect.py`
rebuilds each version from its coefficient tags, nine numbers, and only because
the model is linear, so the tree had to be scored by unpickling its logged
artifact instead. Worth knowing before making that trade: it is cheap, it is
correct, and it quietly narrows what can be asked later.
