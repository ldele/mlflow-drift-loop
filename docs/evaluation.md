# Evaluation

Whether it works, and where it does not. For how it works, see
[methodology.md](methodology.md).

## Six cities that disagree

![All six cities side by side](images/compare.png)

Weather forecasts from the [Open-Meteo](https://open-meteo.com/) historical
forecast archive, joined on the hour with observed PM2.5 from its air-quality
API. Each city trains a model on a clean season and replays week by week into the
season that ruins it, and then out the other side.

| | span | PM2.5 swing | model error, start → worst | retrains / weeks | across replay | week by week |
|---|---|---|---|---|---|---|
| **Delhi** | May 25 → Jul 26 | 42 → 127, post-monsoon burning | 33.4 → 99.0 | 9 / 39 | **+43.7%** | **+49.4%**, won 92% of 38 |
| **Santiago** | Oct 25 → Jul 26 | 18 → 94, winter inversion in a basin | 6.2 → 68.3 | 13 / 21 | **+16.8%** | **+17.3%**, won 100% of 16 |
| **Kraków** | May 25 → Jul 26 | 8 → 57, winter smog in a basin | 5.1 → 54.5 | 14 / 48 | +0.2% | +6.5%, won 72% of 47 |
| **Johannesburg** | Nov 25 → Jul 26 | 23 → 87, Highveld coal smoke | 12.0 → 82.3 | 11 / 19 | 0.0% | **+14.9%**, won 100% of 6 |
| **Melbourne** | Sep 25 → Jul 26 | 5 → 15, winter wood heaters | 3.8 → 15.0 | 8 / 30 | +0.1% | +1.2%, won 70% of 27 |
| **Los Angeles** | Sep 25 → Jul 26 | 15 → 29, a mild winter bump | 16.4 → 20.8 | 1 / 36 | −7.9% | **−13.4%**, won 29% of 35 |

The two retraining columns disagree, and the second is the one to trust when they
do. "Across replay" compares the median error of what was served against the
median of the first model held frozen. That comparison is unpaired: where both
distributions are dominated by the same seasonal swing, it largely measures the
season. Johannesburg promotes nothing until week 14 of 20, so 70% of its windows
have the two models identical, both medians land on the same value, and the
column reads 0.0%. "Week by week" holds the window fixed, compares the two models
in it, and reports the median of those per-window ratios over the weeks a
retrained model was serving, alongside how often it won.

Which windows those are is decided by the version in service, not by whether the
two error figures differ. The served figure is the loop's own logged error and
the frozen one is rebuilt from coefficient tags, so for the identical model they
agree only to six decimals; testing them for equality marks every window as
retrained and averages in the ones from before anything was promoted.

The same question has a boundary case at the very first run. The bootstrap
champion is registered before any monitoring cycle exists, so a first run that
promotes has no earlier row to read the outgoing version off, and its own tag has
already been overwritten with the winner. The replaced version is in the registry
rather than in the run log: it is the lowest registered version, the same model
the frozen comparison uses. Without that rule, a window served by the bootstrap
champion counts as retrained and a real promotion goes unjudged by the gate
calibration below.

### Every replay starts clear of its own training data

Each city's `first_run` sits at least `monitor_days` after the bootstrap
champion's training ends, so the first monitor window contains no hour the
champion fitted. That was not true until 2026-08-06. Five of six cities started 9
to 11 days after training against a 14-day window, and run 0 scored the champion
on 3 to 5 days of its own training data, understating its error by 1.0% in
Johannesburg to 15.2% in Melbourne. No retrain decision changed, but a champion
graded on its own homework is the exact failure the rest of this page is built to
avoid, so the five replays were moved one week later and every number here was
regenerated.

Los Angeles moved most. It now fires one retrain in 36 weeks rather than three in
37, because two of the three fired inside the window that has been dropped, and
its paired result falls from −2.8% to −13.4%. That is the control city getting
more clearly negative, which is the direction the correction should push a city
whose retraining never paid. `tests/test_loop.py` asserts the gap for every
shipped profile, so a new city cannot reintroduce it.

The cities were picked on measurements rather than reputation. Eighteen
candidates were fetched and ranked by PM2.5 swing before any of them was wired
in, which cost two obvious choices. Sydney turned out flat, at 1.5×, with no
story in it. No Brazilian city worked either: the Amazon burning-arc cities peak
at only about 14 µg/m³ in this window, and São Paulo swings 1.7×, less than Los
Angeles does.

The same exercise settled the European slot. Over a current twelve-month window
Kraków swings 6.9×, ahead of Milan (6.2×), Tuzla (6.2×) and Katowice (6.1×),
and comfortably ahead of the cities that make the "worst air in Europe"
headlines: Sarajevo 4.7×, Skopje 4.4×, Sofia 3.7×. Reputation was a poor guide
again.

### What a full year exposed

Every city originally stopped at its dirty-season peak. On that half of the
story, retraining looked like an unambiguous win in both Kraków and Delhi.

Running them through the return trip, as the air gets clean again, reversed the
sign. Retraining came out worse than never retraining at all in both cities.

The cause was the retraining rule rather than the loop. A challenger trained on
the previous 45 days only ever sees one season, so it is excellent in the season
it was born in and wrong as soon as the year turns. The loop was behaving
correctly throughout. Each challenger won its holdout exam at the moment it was
promoted, and then aged badly.

Widening `challenger_train_days` from 45 to 180 fixed it:

| | 45-day window | 180-day window |
|---|---|---|
| Delhi, retraining worth | −7.2% | **+43.7%** |
| Delhi, median served error | 77.06 | **40.44** |
| Kraków, retraining worth | −29.6% | +0.2% |
| Kraków, median served error | 22.70 | **17.49** |

The paired reading tells the same story more quietly, +4.5% against +49.4% in
Delhi: even on the weeks a 45-day challenger was serving it barely helped, and
across the replay it did real damage.

Only a full annual cycle made this visible, which is the argument for running the
replay past the point that flatters the system.

### Where it stops paying

Los Angeles is the control, and the measurements chose it for that. It was picked
as the summer-smog city; hourly PM2.5 over 2025–26 peaks in November and bottoms
out in June. Its model barely moves across 36 weeks and the trigger fires once.
That single promotion wins 29% of the 35 weeks it then serves, and the median
week runs 13.4% behind having left the first model alone. There has to be drift
for a drift loop to earn anything.

Johannesburg is where the promotion gate does the most visible work. Error
climbs from 12.0 to 82.3 µg/m³, the worst on the page, and eleven retrains
produce only three promotions: the other eight challengers failed to clear the 5%
margin and were thrown away. The loop spends the effort, the gate declines to
pay, and nothing ships that did not earn it. When something did ship it worked:
in the six weeks a retrained model was serving it beat the original in all six,
by a median of 14.9%.

Melbourne is the counter-intuitive case. Its air stays near the WHO guideline all
year, yet the model still decays to more than four times its training error, so
clean air does not imply a stable model. Retraining still helps there, by 1.2%
week on week, which is the smallest real effect on the page.

### The loop has no calendar in it

Kraków, Delhi and Los Angeles all peak in northern winter, which leaves a fair
objection open. Are the thresholds quietly encoding a season? Santiago,
Johannesburg and Melbourne peak in June, July and August, when Kraków and Delhi
are at their cleanest, and they run on settings identical to every other city's.

Santiago answers it most directly, because it is Kraków's twin: a basin that
traps winter inversions the same way, half a year out of phase. Both decay by an
order of magnitude, both draw retrains off the same thresholds in opposite
seasons.

Kraków also supplies the cleanest evidence that the two signals are independent.
Across the second half of its replay, 21 weeks from February to July 2026, its
weather drifts *further* from the training window than anywhere else on the page,
while the model runs at well under its training error. Data drift screams, the
retrain trigger stays quiet, and no retrain fires. A system that retrained on
distribution shift alone would have burned twenty retrains there for no reason.

The trigger reaches the right answer by a route that had stopped working, which
is worth stating plainly rather than counting as a win. Measured against a
yardstick that holds still when a model is promoted, that same champion is at its
worst in that stretch, not its best. See
[The retrain trigger stops measuring staleness](#the-retrain-trigger-stops-measuring-staleness).

**Live schedule** is not a city and no longer shares the city selector. It is the
same Kraków data run one incremental cycle at a time, keeping its history in a
tracking store committed back to the repository. Two cycles have been recorded,
covering 2026-07-14 to 2026-07-20 and last logged on 2026-07-27. Drawing a city's
worth of charts from two points invited a comparison with a 48-week replay that
could only mislead, so it is now a short status block in the method section.

One caveat there is worth being exact about: the tracking store records that a
cycle happened, not what started it, so a cycle run by hand and one run by the
weekly Action are indistinguishable in it. The repository contains no commits
from the workflow's bot identity, and the two cycles are timestamped at 09:14 and
09:50 UTC against a 06:00 cron. That is consistent with both being run by hand.
Whether the Action itself has ever fired cannot be established from the
repository, and the page claims only the count and the dates.

A further source, synthetic, has two independent drift knobs and backs the
offline correctness proof in [`sweep_knobs.py`](../scripts/sweep_knobs.py). It is
not published.

## Does it beat the baselines?

![Benchmarks](images/benchmark.png)

Every predictor is scored on the same 14-day monitoring windows the loop reports
on, so the served model, a model that never retrained, and four no-training
baselines all land in one comparable column. They are grouped by what each is
allowed to see rather than ranked in a single list: persistence and seasonal
naive get the readings available when the forecast was issued, and the model gets
the weather forecast and nothing else.

Median RMSE, µg/m³, lower is better:

| | Delhi | Santiago | Kraków | Johannesburg | Melbourne | Los Angeles |
|---|---|---|---|---|---|---|
| **served model** | **40.44** | 24.54 | 17.49 | **21.13** | **3.97** | 10.91 |
| never retrained | 71.85 | 29.51 | 17.52 | 21.13 | 3.98 | **10.12** |
| one model, all six cities | 46.14 | 27.61 | 19.79 | 22.50 | 6.79 | 10.13 |
| climatology | 43.81 | 28.23 | 17.86 | 22.03 | 4.25 | 10.68 |
| training mean | 46.50 | 28.16 | 18.40 | 23.76 | 4.13 | 10.44 |
| persistence | 52.23 | 20.85 | **16.80** | 29.11 | 5.43 | 13.80 |
| seasonal naive | 49.72 | **20.19** | 18.56 | 30.56 | 5.12 | 14.34 |

- The model is the best predictor in Delhi, Johannesburg and Melbourne, and
  second in Kraków. It beats persistence in four of six.
- Retraining pays where drift is real. Delhi's never-retrained model ends up
  worse than a constant, which is what happens when nobody intervenes.
- Santiago is where the baselines win. Its pollution is persistent enough
  from week to week that repeating a stale reading beats forecasting from
  weather, even though retraining still clearly helps the model itself.

### Six models, and whether one would do

![One model over all six cities against six separate ones](images/pooled.png)

One Ridge over all six cities at once, trained once on the six training windows
and never retrained, is the obvious cheaper arrangement. It carries a separate
intercept per city, and that intercept does most of the work: mean PM2.5 runs
from 7 µg/m³ in Melbourne to 84 in Delhi, and the same model without it scores
43% worse. So the weather slopes are shared across cities and only the level is
learned per place.

The answer is that six models win, and the margin is smaller than the layout
implies. Pooling loses to the city's own retrained model in five of six cities,
but beats the city's own *frozen* model in two, and in Delhi it is not close:
46.14 against 71.85 µg/m³. Training on five other cities is worth more than a
year of staleness and less than keeping one city's model current. Where pooling
hurts most is Melbourne, the cleanest city by a distance, whose weather-to-
pollution relationship the other five outvote.

`alpha=1.0` ships. Forward-chaining CV wants far heavier regularisation in most
cities, up to 1000, which is itself a signal: the fitted relationship is weak
enough that shrinking it toward zero costs almost nothing (between 0.1% and 11.9%
depending on the city).

## What the model is worth

RMSE alone cannot answer that, and neither can R². RMSE has no scale, so a
filthy city and a clean one are not comparable and neither are two seasons of
the same city. R² normalises by the window's own variance, which in a calm
fortnight is tiny, so it reports catastrophe (−5.93 in Kraków's last window) for
a modest absolute error.

The page now leads with a skill score against a baseline you could deploy
instead: the hour-of-day profile of the previous 30 days, its reference
period ending a full forecast lead before the window starts so nothing it
averages was unavailable when the forecast went out. It does see recent PM2.5,
which the model never does, so it is a hard bar rather than a like-for-like
comparison, the same information-set distinction the benchmark table draws.

These models earn their keep in the dirty season and
lose to a rule of thumb in the clean one. Kraków's champion runs at +43% skill in
January and −167% in July.

This matters beyond presentation, because **the skill score and the retrain
trigger disagree about the same model at the same moment**. In Kraków's final
window the trigger reads 0.28, the healthiest it has ever been and nowhere near
firing, while skill reads −1.67, the worst it has ever been. Only one of them
can be right.

## The retrain trigger stops measuring staleness

Not after the first few promotions. `perf_drift_ratio` divides the champion's
current error by *its own* error at training time, and every promotion resets
the denominator. Retrains fire in the dirty season, so each new champion inherits
a higher bar than the one it replaced, and the bar never comes back down:

| | baseline, first → last | last retrain | runs of silence after it | highest ratio in those runs |
|---|---|---|---|---|
| **Los Angeles** | 9.7 → 18.3 | run 1 of 36 | **35** | 1.14 |
| **Kraków** | 3.7 → 45.8 | run 18 of 48 | **30** | 0.99 |
| **Delhi** | 18.8 → 57.1 | run 33 of 39 | 6 | 1.07 |
| **Santiago** | 6.7 → 52.7 | run 18 of 21 | 3 | 0.84 |

Once the bar has ratcheted to the seasonal peak the trigger cannot fire again,
whatever the model does. Kraków spends 62% of its timeline in that state, serving
a 210-day-old model. In that city `corr(champion_rmse, ratio)` is 0.29 while
`corr(baseline_rmse, ratio)` is −0.72: the signal tracks *which champion happens
to be in service* more strongly than how well that champion is doing.

Two changes were proposed for it. Measure the trigger against something outside
the model, as the skill score above does, since it holds still when a model is
promoted. And put an absolute error floor alongside the ratio, so being bad in
absolute terms is sufficient on its own.

Both have now been built and measured, and the answer is not the one the proposal
expected.

## Fixing the trigger: one fix is impossible, the other does not pay

**The absolute floor is not implementable.** A floor has to be one number for
every city, or the thresholds stop being identical and the cities stop being
comparable, which is the property the whole six-city comparison rests on. There
is no such number. Los Angeles's deaf stretch tops out at 20.8 µg/m³, so waking
it needs a floor near 20 or below; at 20 µg/m³ Delhi fires on 100% of its runs
and Johannesburg on 68%, and at 25 Los Angeles never fires at all. The gap
between "wakes the quietest city" and "does not retrain the dirtiest city every
week" is empty. An absolute floor is a per-city tuning knob wearing a disguise.

**The model-independent yardstick can be built, and it changes almost nothing.**
`LoopConfig.skill_floor` fires a retrain when the champion's skill against the
30-day daily profile drops below a floor. Being a ratio against something outside
the model it is scale-free, so one number does work for every city, and it cannot
be moved by promoting anything. [`sweep_skill_floor.py`](../scripts/sweep_skill_floor.py)
replays all six cities at several floors, a full replay per arm, because a
changed trigger changes which models exist and therefore the whole trajectory.

It does wake the trigger up. Kraków's longest silence falls from 30 weeks to 5,
Los Angeles's from 29 to 12. Median champion error, µg/m³, lower is better:

| | trigger off | skill < 0 | skill < −0.25 | skill < −0.5 |
|---|---|---|---|---|
| **Kraków** | **17.49** | 18.79 | 18.55 | 17.49 |
| **Delhi** | **40.44** | 44.11 | 44.91 | 40.44 |
| **Los Angeles** | 10.91 | **10.15** | 10.45 | **10.15** |
| **Santiago** | 24.54 | 24.54 | 24.54 | 24.54 |
| **Johannesburg** | 21.13 | 21.13 | 21.13 | 21.13 |
| **Melbourne** | 3.97 | 3.97 | 3.97 | 3.97 |

At the conservative floor it is a no-op in **five of the six**, to the last
decimal. Kraków is the clearest case of why: it fires six extra retrains there
and the gate rejects every one, so the loop does more work and ships nothing.

Los Angeles is the exception, and it is a real one: the floor takes it from 10.91
to 10.15 µg/m³, and its retraining result from −7.9% to −0.3% across the replay.
That is the city whose trigger is deafest, silent for 35 of 36 runs, and the
floor is what wakes it. At the aggressive floor the picture is mixed instead:
Delhi and Kraków get worse by 9% and 7%, Los Angeles improves by the same 7%, and
three cities do not move.

So the ratchet is real as a mechanism and close to inert as a harm in five
cities out of six. In the sixth, the cautious floor is free: no city is worse and
the control city stops losing money. That is a weaker case for shipping it off
than the first version of this experiment produced, and it is worth being explicit
that the change came from correcting the first-run windows above rather than from
new evidence about the floor itself.

**The most consistent explanation is that the trigger was never the bottleneck.**
Firing more often only pushes more challengers at a gate that certifies for about
five weeks, which is the finding below. Delhi's promotions go from 8 to 13 and
its error rises with them. Los Angeles is the exception that fits: its air barely
moves, so successive models are nearly identical and a short certificate costs it
nothing.

That pointed the next experiment at the length of the exam, which was also
measured, and it does not pay either. See
[Lengthening the exam does not fix it](#lengthening-the-exam-does-not-fix-it).

### Verified against other measures and fixed baselines

A conclusion that rests on one summary statistic deserves a second look, so the
three cities that move were re-run and scored two further ways. The first swaps
the statistic: mean RMSE and median MAE instead of median RMSE. The second drops
the arm-against-arm comparison and scores every arm against the same fixed
predictors, which cannot be moved by anything the trigger does.

| | floor | median RMSE | mean RMSE | median MAE | vs. climatology | vs. persistence | vs. training mean |
|---|---|---|---|---|---|---|---|
| **Kraków** | skill < 0 | ×1.074 | ×1.032 | ×1.074 | 0.965 → 1.037 | 1.041 → 1.118 | 0.950 → 1.021 |
| | skill < −0.5 | ×1.000 | ×1.000 | ×1.000 | unchanged | unchanged | unchanged |
| **Delhi** | skill < 0 | ×1.091 | ×1.021 | ×1.044 | 0.973 → 1.061 | 0.774 → 0.844 | 0.870 → 0.949 |
| | skill < −0.5 | ×1.000 | ×1.000 | ×1.000 | unchanged | unchanged | unchanged |
| **Los Angeles** | skill < 0 | ×0.930 | ×0.898 | ×0.861 | 1.010 → 0.939 | 0.791 → 0.735 | 1.045 → 0.972 |
| | skill < −0.5 | ×0.930 | ×0.938 | ×0.874 | 1.010 → 0.939 | 0.791 → 0.735 | 1.045 → 0.972 |

The first three columns are ratios against that city's own trigger-off arm, so
above 1 means the floor made it worse. The baseline columns are the served
model's error over that baseline's, so below 1 means the model wins.

All three statistics agree in direction in all three cities, which rules out an
artifact of the median. The baseline columns then say something the
arm-against-arm comparison could not: they show the floor moving cities across
the line that matters. An aggressive floor flips Kraków from beating a constant
to losing to one, 0.950 to 1.021, and flips both Kraków and Delhi from beating
the daily profile to losing to it. Los Angeles crosses the other way on both, and
does so at the cautious floor too, going from 1.045 to 0.972 against a constant
and 1.010 to 0.939 against the daily profile.

Two smaller readings. Mean RMSE moves less than median in the harmed cities,
×1.032 against ×1.074 in Kraków, so the damage sits in the typical week rather
than in the tail. And the cautious floor is a true no-op in Kraków and Delhi, to
three decimals on every measure, while in Los Angeles it is worth 7% on RMSE and
13% on MAE.

The floor ships **off by default**, and that decision is now closer than it was.
The case for off is that five of six cities do not move and the sixth is the
control, where "improving" means losing less. The case for on, at the cautious
setting, is that nothing is harmed and Los Angeles stops being beaten by a
constant. What settles it is not in this data: one replay of six cities cannot
tell you whether a knob that helps one city out of six is finding a real effect
or fitting that city's noise, and the response to that is to leave the default
alone and say so.

## A seven-day exam certifies a model for a month, not half a year

![What the exam promised against what it delivered](images/gate.png)

Every promotion left a prediction behind: the margin the challenger won its
seven-day exam by. That margin is the number the decision was made *on*, so it
cannot also be evidence the decision was right. The out-of-sample check is what
each winner went on to deliver over the weeks it served, measured against the
model it displaced and scored on those same windows. That is a counterfactual
the loop has no reason to compute at the time, and which
[`retrospect.py`](../src/driftloop/retrospect.py) computes afterwards by
rebuilding every registered version from its logged coefficients.

28 promotions across six cities:

| promotions that served | n | exam promised | delivered | harmful |
|---|---|---|---|---|
| under 20 weeks | 25 | +12.4% | **+9.8%** | 0 |
| 20 weeks or more | 3 | +14.4% | **−6.1%** | **3 of 3** |

The gate is honest and well calibrated over the horizon it tests, and not one
short-serving promotion left its city worse off. But every model that ended up serving
twenty weeks or more delivered a *negative* margin despite passing the same exam
just as convincingly. A seven-day exam can certify a model for about a month and
cannot certify it for half a year.

The two faults compound. Those long-serving models serve long precisely because
they were promoted at the seasonal peak, which is what sets the retrain bar out
of reach. So the trigger goes quiet, the model stays, and the exam that cleared
it is asked to stand for far longer than it can.

Nothing here is a leak. The challenger never trains on its holdout, and the
delivered margin is measured from the run *after* the promotion, because the
monitor window at the promotion itself overlaps the challenger's training data.
Dropping that window is what makes the comparison clean.

### Lengthening the exam does not fix it

The obvious reading of the table above is that seven days is too small a sample:
a challenger can win a lucky week, and a longer exam would catch it. That was the
top open experiment after the trigger work, and it is wrong.

[`sweep_holdout.py`](../scripts/sweep_holdout.py) replays all six cities at exam
lengths of 7, 10, 14 and 21 days, holding the replay cadence at 7 days so exam
length is the only variable. Pooled gate calibration, split at the same twenty
weeks:

| exam | served < 20 weeks | promised | delivered | | served ≥ 20 weeks | promised | delivered | harmful |
|---|---|---|---|---|---|---|---|---|
| **7 days** | 25 | +12.4% | +9.8% | | 3 | +14.4% | −6.1% | 3 of 3 |
| **10 days** | 23 | +12.9% | +10.8% | | 3 | +10.8% | −2.7% | 3 of 3 |
| **14 days** | 25 | +10.7% | +8.9% | | 3 | +13.1% | −7.2% | 3 of 3 |
| **21 days** | 20 | +12.1% | +11.8% | | 3 | +13.9% | −9.7% | 3 of 3 |

Tripling the exam does what more data should do over the horizon it tests: the
promise gets more honest, from a 2.6-point gap at seven days to 0.3 points at
twenty-one. **Beyond that horizon it changes nothing.** Every long-serving
promotion still delivers a negative margin at every exam length, and the reversal
is deeper at 21 days than at 7.

That is the useful part. The reversal is not a sample-size problem, because it
survives tripling the sample. A model that serves twenty weeks fails because the
world moved after it was certified, and **no exam taken at promotion time can
certify against drift that has not happened yet.**

The cost side is unambiguous. A longer exam is a stricter filter, and the
promotion rate falls with it: in Delhi from 89% of challengers to 39%, in Kraków
from 50% to 28%, in Melbourne from 50% to 18%. In cities where retraining pays,
blocking challengers costs error. Median champion error over the runs every arm
shares:

| | 7 days | 10 days | 14 days | 21 days |
|---|---|---|---|---|
| **Kraków** | 17.67 | **16.76** | 17.62 | 17.51 |
| **Delhi** | 40.91 | 40.76 | 40.91 | 42.64 |
| **Santiago** | **25.59** | 25.96 | 27.61 | 28.39 |
| **Los Angeles** | 10.96 | 10.98 | **10.95** | 11.12 |
| **Johannesburg** | 21.32 | 21.32 | 21.32 | 21.32 |
| **Melbourne** | **4.08** | 4.10 | 4.09 | 4.09 |

Compared on the runs every arm shares, not on each arm's own span. A longer exam
has to start later, because it would otherwise reach back into the bootstrap
champion's training data, and the runs it drops are early clean-season ones with
low error. Uncontrolled, that alone moved Santiago's apparent penalty from 11% to
21%.

Ten days is worth 5% in Kraków and nothing anywhere else. Twenty-one days costs
11% in Santiago and 4% in Delhi, the two cities where retraining pays most, for
no gain elsewhere. Seven days stays.

**What both experiments together say.** Making the trigger more sensitive does
not pay. Making the exam longer does not pay. So the long-serving reversal is
neither a detection-sensitivity problem nor a sample-size one, and what is
missing is a third mechanism: nothing ever re-examines a serving champion on
fresh unseen data. The exam happens once, at promotion. The trigger
compares the champion against its own history and ratchets itself deaf. A
periodic re-certification, running the holdout exam against a fresh challenger on
a fixed schedule regardless of what the trigger says, is the change these two
results point at. It has not been built.

## Whether the detection works at all

![The controlled experiment](images/control.png)

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

The two-signal design rests on that property, and it is the one property the six
cities cannot demonstrate. It is published on the page for the same reason it
is here: it is the evidence, and the cities are the application.

## Limitations

- R² is still weak, and it belongs up front. Widening from three
  features to eight improved every city, but R² on the most-drifted window is
  still near zero or negative in several. Predicting an hour's PM2.5 from a
  week-old weather forecast is hard, and the page should be read with
  that in mind: the loop is the demonstration, not the model.
- The retrain trigger ratchets, and it is shipped that way on purpose. The bar
  rises at every promotion and never falls, so after the seasonal peak the
  trigger cannot fire at all. The fix for it now exists in the code
  (`LoopConfig.skill_floor`) and is switched off, because replaying all six
  cities with it on changes nothing at a cautious setting and makes two of them
  worse at an aggressive one. See
  [Fixing it](#fixing-the-trigger-one-fix-is-impossible-the-other-does-not-pay).
  Lengthening the exam does not pay either, so what is left open is periodic
  re-certification of the serving champion, which has not been built.
- **On run 0, five of six cities score the bootstrap champion partly on its own
  training data.** `first_run` is set 9 to 11 days after `champion_train_end` in
  every city except Kraków, and the monitor window is 14 days, so 3 to 5 days of
  the first window is data the champion fitted. Its error there is understated by
  1.0% (Johannesburg) to 15.2% (Melbourne). No city's retrain decision changes:
  the trigger reads the same side of 1.25 either way in all six. It affects one
  run of the twenty to forty-eight in each replay, and it is a genuine leak of
  the kind this page otherwise guards against. The fix is to push `first_run` out
  to at least `champion_train_end + monitor_days`, at a cost of one run per city
  and a re-baseline of every number on this page, so it is reported rather than
  silently applied.
- **The cadence guard was the wrong condition until 2026-08-05.**
  `run_simulation` required `step_days >= holdout_days`, which is the same rule
  as the correct one only at the shipped values. It admitted a 3-day exam at a
  weekly cadence, where four days of every monitor window is the training data of
  the champion promoted the run before, and it rejected a 14-day exam, which is
  clean. The real condition is `step_days + holdout_days >= monitor_days`, and it
  is now enforced with a test on each side. No shipped configuration was
  affected; the sweep above is what needed the correction.
- The skill baseline sees recent PM2.5 and the model does not. That is stated
  wherever the number appears, and it is deliberate: it is the alternative you
  could deploy. A like-for-like baseline would have to be built from the
  champion's own training window, which inherits the champion's staleness and so
  cannot measure it.
- 180 days is argued rather than tuned. It was chosen to span more than one
  season after 45 was shown to fail, not swept. A proper sweep of the retrain
  window against retraining value is the obvious next experiment.
- Boundary layer height is missing, and it is the feature I most want. It sets
  the volume pollution is diluted into, and would likely dominate the feature
  importance ranking. Open-Meteo does not archive previous model runs for it, so at a
  seven-day lead it returns null. Shortwave radiation is the stand-in.
- No autoregressive features. Giving the model a lag term would help most in
  the low-drift cities where a constant currently beats it. It would also blunt
  the drift story, since an autoregressive model absorbs regime change instead of
  decaying visibly through it. That tension is a design decision.
- One lead time. Everything here is measured at seven days. The interesting
  sweep, error against lead from 1 to 7 days, is not run.
- State lives in git. Committing the SQLite backend back to the repo is the
  simplest zero-infra persistence and keeps history versioned, but it grows the
  repo. Production would point `MLFLOW_TRACKING_URI` at a hosted tracking server.
- Artifact paths are absolute. A backend generated on the CI runner resolves
  metrics, params and tags anywhere, but not the per-run prediction files.
- Serving is single-node. One container per city, holding its own SQLite
  backend, is the shape of a zero-infra demo rather than of production. A
  real deployment would point every replica at one hosted tracking server so they
  promote in step, and `POST /reload` would be called by the loop rather than by
  hand.
