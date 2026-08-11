# Methodology

How the system works. For whether it works, see [evaluation.md](evaluation.md).

A linear model forecasts a city's hourly PM2.5 a week ahead from the weather
forecast. Once a week a loop scores that model on the fortnight just gone, checks
two independent drift signals, trains a replacement when either fires, and
promotes the replacement only if it wins a blind exam. This document follows that
weekly run from raw weather to a promotion decision, then covers the machinery
that keeps the result honest.

## Vocabulary

Used precisely throughout, so the prose does not stop to re-explain them.

| term | meaning |
|---|---|
| **champion** | the model currently in service, pointed at by the `champion` alias in the MLflow registry |
| **challenger** | a freshly trained model competing to replace the champion |
| **bootstrap champion** | the first champion, trained before any monitoring run exists |
| **promotion** | moving the `champion` alias onto a challenger. This is the deployment step |
| **promotion gate** | the rule that decides it: the challenger must beat the champion by more than 5% on the holdout |
| **monitor window** | the 14 days ending at the run date, used to score the champion and measure drift |
| **holdout** | the last 7 days before the run date, excluded from challenger training, used as the exam |
| **forecast lead** | how far ahead the features look. At lead 7 the features are the forecast issued a week before the target hour |
| **covariate drift** | the incoming feature distributions have moved away from training. Needs no labels and no model |
| **concept drift** | the relationship between features and target has changed. Only visible in error |
| **PSI** | Population Stability Index, the statistic used to quantify covariate drift |
| **skill score** | error relative to a reference predictor: `1 − RMSE_model / RMSE_reference` |
| **climatology** | the reference predictor for skill: the hour-of-day mean of the previous 30 days |
| **the ratchet** | the retrain trigger's failure mode, where its threshold rises at every promotion and never falls |
| **replay** | running the weekly loop over a fixed historical span, one simulated week at a time |

---

## The prediction task

Hourly PM2.5 concentration in µg/m³, seven days ahead.

The features are the weather forecast *for the target hour, as it stood a week
earlier*, taken from Open-Meteo's archive of previous model runs. That archive
stores what the forecast said at the time rather than what the weather turned out
to be, so the model answers a question a city could ask on a Monday: given the
weather we currently expect next Monday, how dirty will the air be?

The framing has a cost worth stating up front. At a seven-day lead the weather
forecast is already wrong by about 4.5 °C on temperature, and the PM2.5 model
inherits that error before adding any of its own. Two error sources stacked is
why the benchmarks in [evaluation.md](evaluation.md) land where they do.

The forecast lead is one constant, `FORECAST_LEAD_DAYS`. At 0 the features come
from the ERA5 reanalysis instead, which turns the same code into a same-hour
estimator with no forecasting in it. Seven is the ceiling rather than a choice:
Open-Meteo archives previous model runs out to day seven and no further.

Nothing downstream of the data fetch knows which of the two it is doing. Features
and target share one timestamp in the column contract, so the horizon lives
entirely in which weather the source retrieves. The same loop, drift maths and
registry served both framings unmodified, which is the strongest evidence that
the machinery is not coupled to the problem.

---

## The model

![Method](images/method.png)

`StandardScaler → Ridge(alpha=1.0)` on eight features, fitted on a chronological
80/20 tail split, then refit on the full window. The tail split is what produces
the champion's **baseline RMSE**, the number performance drift is later measured
against.

### What ridge regression does

Ordinary least squares picks coefficients minimising squared error, with the
closed form `β = (XᵀX)⁻¹Xᵀy`. It has two weaknesses. When feature columns are
close to collinear, `XᵀX` approaches singular and its inverse produces enormous,
mutually cancelling coefficients that fit the training window and mean nothing
individually. And every coefficient is fitted at full strength, including noise.

Ridge adds a penalty on coefficient size:

```
minimise   Σᵢ (yᵢ − β₀ − Σⱼ βⱼxᵢⱼ)²   +   α · Σⱼ βⱼ²
```

which has the closed form `β = (XᵀX + αI)⁻¹Xᵀy`. Adding α down the diagonal of
`XᵀX` is the ridge the method is named after, and it makes the matrix invertible
even under collinearity. Hoerl and Kennard proposed it in 1970 for that purpose.
The trade is bias: ridge returns coefficients that are too small, in exchange for
coefficients that do not lurch when the training window shifts by a week. β₀ is
excluded from the penalty, since shrinking the intercept would pull the model
toward predicting zero pollution rather than toward the mean.

Here that insurance is barely exercised. The weather features are correlated over
Kraków's training window:

```
                     temp   wind   humid  precip  press  radiation
temperature          1.00  -0.13  -0.66   -0.16  -0.16   0.61
humidity            -0.66   0.02   1.00    0.33  -0.34  -0.62
shortwave_radiation  0.61   0.10  -0.62   -0.08   0.11   1.00
```

but correlated is a long way from collinear. The condition number of the
standardised feature matrix is **6.9**, where trouble starts north of 30 and an
ill-posed problem runs into the thousands. At the shipped α = 1.0 the fitted
coefficients match plain least squares to three decimal places, so **the
regularisation is close to inert at the setting that ships**. It stays because
every challenger is fitted on a different 180-day window that nobody inspects
first.

### Why the features are standardised

`Σβⱼ²` sums coefficients as though they were comparable. They are not: surface
pressure runs around 1000 hPa and precipitation around 0.1 mm, so the slope per
hPa is tiny and the slope per mm large before any physics is considered.
Penalising them together in raw units would decide that pressure matters less
than rain because of the recording unit. `StandardScaler` puts every feature on
mean 0, standard deviation 1 first, so the penalty applies to how much the
prediction moves per typical move in that feature. Ridge on unstandardised
features is a defect, not a style choice.

### Reading the coefficients

Because scaling happens inside the pipeline, `ridge.coef_` is in z-score units
tied to one training window's spread, so two versions are not comparable.
`model.effective_coefficients` folds the scaler back in:

```
slope_j    = coef_j / scale_j
intercept  = intercept_ − Σⱼ coef_j · mean_j / scale_j
```

giving µg/m³ per °C, per m/s, per hPa. Those are comparable across versions and
are what the coefficient charts plot.

They also carry more weight than presentation. A fitted model is now eight slopes
and an intercept, so `log_and_register` writes it into registry tags, and
[`retrospect.py`](../src/driftloop/retrospect.py) rebuilds any version from the
registry alone and scores it on any window as a weighted sum of columns: no
unpickling, no refitting, no artifact paths. Much of the analysis below exists
because a linear model serialises into nine numbers.

### Why not lasso, or gradient boosting

Lasso would zero some coefficients and shorten the feature list. That is usually
a selling point and is wrong here: a coefficient chart with three features pinned
flat at zero says less about how the world is changing, not more.

Gradient boosting would score better and would absorb some of the drift rather
than decaying through it, which is what people mean when they call a flexible
model resilient. The demonstration depends on the decay staying legible.

So the model is small on purpose, and **the modelling is not the interesting part
of this project**. Every threshold below is identical across all six cities, so
where two cities behave differently it is their air that differs, not their
tuning.

### Choosing alpha

`alpha=1.0` ships, which is scikit-learn's default and so needs justifying. It is
swept per city on a logarithmic grid, scored by five-fold forward-chaining
cross-validation (`TimeSeriesSplit`), which never lets a fold train on rows that
follow the rows it scores. On autocorrelated hourly data a random split leaks
badly enough to make every α look fine (Bergmeir & Benítez, 2012).

Every city wants heavier regularisation than 1.0:

| | best α | cost of shipping 1.0 |
|---|---|---|
| Santiago | 10 | 0.1% |
| Los Angeles | 100 | 2.1% |
| Melbourne | 1000 | 1.5% |
| Kraków | 1000 | 3.6% |
| Johannesburg | 1000 | 4.4% |
| Delhi | 1000 | 11.9% |

The curve is nearly flat, which is why 1.0 survives a sweep that disagrees with
it everywhere. On Kraków's window, α = 1000 shrinks each coefficient by 15%
(temperature) to 49% (surface pressure), and costs 3.6%. **If shrinking every
slope most of the way to zero costs between 0.1% and 12% of error, the
relationship this model can express is weak.** That is the fairest one-line
summary of the modelling here, and the reason the project's interest lies in the
loop.

---

## The features

Six observed weather variables, chosen for how pollution accumulates and clears,
plus two terms encoding the clock.

**temperature.** Cold air under warmer air is an inversion: a lid, with the city
underneath. It is also the cleanest available proxy for the heating season in
Kraków and the wood-burner season in Melbourne.

**wind_speed.** Ventilation. Still air lets pollution accumulate; moving air
carries it away. Its slope is the one most likely to change sign between
versions, because wind that clears a valley can equally import smoke into it.

**humidity.** Water condenses onto existing aerosol, making particles heavier and
optically larger, and high humidity feeds the chemistry that forms new ones.

**precipitation.** Wet deposition: rain scavenges particulates out of the air. The
most direct removal mechanism here and the most awkward feature statistically,
being zero for between 70% of hours (Johannesburg) and 96% (Santiago), so its
distribution is a spike at zero with a thin tail.

**surface_pressure.** High pressure means subsiding air, which warms as it
descends and caps the city with a subsidence inversion. This is the mechanism
behind Kraków's and Santiago's winter smog.

**shortwave_radiation.** Sunlight heats the ground, the ground heats the air, and
the mixing layer deepens: a larger volume to dilute the same emissions into. It
also drives the photochemistry forming secondary particles, so its slope is
ambiguous in sign.

**Missing: boundary layer height**, the depth of that mixing layer. It would
summarise most of the above in one number and would likely dominate the
feature-importance chart. Open-Meteo does not archive previous model runs for it,
so at a seven-day lead it returns null, and using it would mean abandoning the
forecast framing. Shortwave radiation stands in. Seinfeld and Pandis is the
reference for every mechanism above.

### The clock

```
hour_sin = sin(2π · hour / 24)      hour_cos = cos(2π · hour / 24)
```

A raw integer hour would tell a linear model that 23:00 sits twenty-three units
from midnight rather than one. The sine-cosine pair places the day on a circle,
so hour 23 neighbours hour 0 and the model can express a daily cycle.

One limitation: a single harmonic expresses one peak and one trough per day, and
urban PM2.5 is usually bimodal, with morning and evening traffic peaks. Fitting
both needs a second harmonic at `4π·hour/24`. The model currently fits one hump
to a two-hump shape, and closing that gap is a cheap experiment not yet run.

Twenty-four one-hot dummies were rejected twice over: they would add twenty-four
columns to a model with six real features, and a table of per-hour means *is* the
climatology baseline the model is later measured against, so building it in would
fold the yardstick into what it measures.

Only the six weather features carry a PSI. The clock cannot drift, because every
monitor window contains all 24 hours and its distribution is fixed by
construction. That is why `DRIFT_FEATURES` and `FEATURES` are separate lists.

---

## One weekly run

Four steps, of which the last two are conditional.

```
...........[==== challenger train ====][= holdout =] as_of
                    [====== monitor window ========]
[== champion train ==]  (much earlier, never overlaps holdout)
```

### Step 1: score the champion

The champion predicts the monitor window and its RMSE, MAE and R² are logged.
That RMSE feeds both drift signals below.

### Step 2: two drift signals

| signal | what it measures | needs labels? | needs a model? |
|---|---|---|---|
| covariate drift (PSI) | the world changed | no | no |
| performance drift | the model is failing | yes | the champion |

Covariate drift needs neither labels nor a model, which makes it the early
warning: it can raise a hand the moment incoming weather stops looking familiar,
before any error is observable. Performance drift needs the truth to have
arrived, so it is always late, and it is the only one allowed to authorise a
retrain. Detecting drift by comparing champion against challenger would be
circular, since you would need a challenger before being allowed to decide you
needed one.

**Covariate drift, and what PSI computes.** Sort the training window's values for
one feature into ten equal buckets, drop the monitor window's values into the
same buckets, and measure how differently they fall:

```
PSI = Σ over buckets  (current_share − reference_share) · ln(current_share / reference_share)
```

Bucket edges come from training deciles, so training is even by construction and
any imbalance belongs to the current window. Formally this is the symmetrised
Kullback–Leibler divergence between the binned distributions, known as Jeffreys
divergence. The conventional reading, inherited from credit scoring, is below
0.10 stable, 0.10 to 0.25 moderate, above 0.25 significant.

**PSI saturates here, and the charts say only what survives that.** Both shares
are clamped at ε = 1e-6 to keep the logarithm finite, so a bucket holding zero
current rows contributes a fixed `(1e-6 − 0.1)·ln(1e-6/0.1) ≈ 1.15` however far
away the current data has moved. With ten deciles the ceiling is about 11.5. In
Kraków's worst temperature window scores 11.53, near that ceiling, and **9.22 of
it, 80%, comes from eight empty buckets**, because a January fortnight and the
previous June do not overlap. Every city sits above 0.25 on almost every run,
with median readings twelve to thirty-eight times the threshold. So PSI is dependable as a yes/no and undependable as a magnitude, and
both UIs show a green/amber/red band rather than a number. A Kolmogorov–Smirnov
statistic is logged as a cross-check, but on 336 hourly rows its p-values are
vanishingly small for effects of no practical size.

**Performance drift, and the ratchet.** The trigger is the champion's monitor
RMSE against its own baseline RMSE, at 1.25×: a quarter worse than at training
time sends it back to school.

Kraków demonstrates both why the two signals must stay separate and why this one
is built wrong. Through the second half of its replay its features drift further
from training than any other city while the champion runs under its baseline, so
covariate drift alone would have ordered twenty pointless retrains. The loop
declines, correctly.

But that baseline was set in deep winter. Because every promotion resets the
denominator and promotions happen at the seasonal peak, the threshold ratchets
upward and never falls: the champion serving that stretch has a baseline of
45.8 µg/m³, and almost nothing can cross 1.25× of it. Measured against a
reference that holds still, the same champion is at its worst there rather than
its best, with skill of −1.67 against climatology having been +0.43 in January.
See [evaluation.md](evaluation.md#the-retrain-trigger-stops-measuring-staleness).

**A second trigger exists for that, and ships disabled.** `LoopConfig.skill_floor`
also fires when skill against climatology drops below a floor:

```
retrain if   rmse_now / baseline_rmse > 1.25        (ratchets)
        or   1 − rmse_now / rmse_climatology < floor  (holds still on promotion)
```

Being a ratio against something outside the model, it is scale-free, so one
number works for every city and no promotion can move it. It only ever adds a way
to fire. Every run is tagged `retrain_reason` as `ratio`, `skill`, `both` or
`none`.

It ships at `None` because it was measured rather than assumed:
[`sweep_skill_floor.py`](../scripts/sweep_skill_floor.py) replays all six cities
at several floors, and a conservative floor leaves the outcome identical in five
of six while an aggressive one makes two cities worse for one improvement. It
wakes the trigger as designed, and the waking is not worth having. Numbers in
[evaluation.md](evaluation.md#fixing-the-trigger-one-fix-is-impossible-the-other-pays-in-one-city).
The knob stays in the code so the next person to propose this fix finds it
already built and already answered.

### Step 3: train a challenger

`challenger_train_days = 180`, and the number has a history.

It was 45, which looks natural: recent, responsive, plenty of hourly rows. It was
wrong for a reason only a full year of replay exposed. PM2.5 is seasonal, so a
challenger trained on six weeks sees one season, wins its exam honestly, and is
mismatched the moment the year turns. Replays that stopped at the winter peak hid
this entirely. Extended through the recovery, retraining came out 29.6% worse in
Kraków and 7.2% worse in Delhi; widening to 180 days took Delhi to +43.7%. See
[evaluation.md](evaluation.md#what-a-full-year-exposed).

180 is argued rather than tuned, on the grounds that it spans more than one
season. Sweeping it is the cheapest experiment left.

### Step 4: the promotion gate

Champion and challenger both predict the holdout, a week neither has trained on,
and the challenger is promoted only if its RMSE is more than 5% lower.

Seven days is measured rather than assumed.
[`sweep_holdout.py`](../scripts/sweep_holdout.py) replays every city at 7, 10, 14
and 21 days. A longer exam makes the promise more honest over the horizon it
tests and does nothing beyond it: long-serving promotions still reverse at every
length. It also filters harder, rejecting challengers that the cities where
retraining pays cannot afford to lose. See
[evaluation.md](evaluation.md#lengthening-the-exam-does-not-fix-it).

**No evaluation leak**, and two guards make that structural rather than
aspirational. Within a run, the loop raises if the holdout would overlap the
champion's training data. Across runs, `run_simulation` requires

```
step_days + holdout_days >= monitor_days
```

because a challenger promoted at `as_of` trained up to `as_of − holdout_days`,
and the next run's monitor window opens at `as_of + step_days − monitor_days`.
Below that bound the champion is scored on hours it fitted, which flatters it and
suppresses the retrain trigger. Both guards are asserted in `tests/test_loop.py`,
because a leak that surfaces only as suspiciously good results is a defect you
talk yourself into believing.

That second guard used to read `step_days >= holdout_days`, which is the same
rule at the shipped values (7 + 7 = 14) and wrong on both sides of them: it
admitted a 3-day exam at a weekly cadence, where four days of every monitor
window is the new champion's own training data, and rejected a 14-day exam, which
is clean. A guard that is right only at its default is one nobody has tested away
from it.

The baselines are held to the same rule. A forecaster issuing seven days out may
use readings up to that moment and no later, so persistence repeats a week-old
observation rather than yesterday's. That makes it much weaker than the textbook
version, and fair.

---

## Judging the loop afterwards

The loop logs what it decided at the time, which is enough to run it and not
enough to judge it. Three questions need a model scored on windows it never
served:

- **Does an individual model decay?** The logged champion error is one line
  across eight different champions, so no single model's decay is visible in it.
- **Is the model worth having?** An error is a verdict only against an
  alternative you could have deployed instead.
- **Did the gate work?** That needs the *replaced* champion scored on the windows
  its replacement went on to serve, a counterfactual the loop never computes.

[`retrospect.py`](../src/driftloop/retrospect.py) answers all three from the
coefficient tags, reconstructed to the six decimals they carry, which
`tests/test_retrospect.py` asserts against the loop's independently logged RMSE.

### The skill score

```
skill = 1 − RMSE_model / RMSE_reference
```

1 is perfect, 0 matches the reference, negative means the reference won. This is
a skill score in the sense forecast verification has used for decades (Murphy,
1988); the family is usually written over MSE, and RMSE gives the same ordering
on a gentler scale. The choice of reference is the whole content of the metric.

The reference here is climatology: the hour-of-day mean of the previous 30 days.
A month is long enough to average out weather and short enough to remain the
current season. Two properties earn it the job. It is scale-free, so a filthy
city and a clean one become comparable, and so do two seasons of one city. And
its yardstick does not move when a model is promoted, which is the property the
retrain ratio lacks and the reason the two disagree about the same model at the
same moment.

Why not R²? R² is the same construction with the reference swapped for the
window's own mean. In a calm fortnight that mean is an unusually strong
predictor, so R² reports catastrophe, −5.93 in Kraków's final window, for a
modest absolute error. It measures the window's variance more than the model.

One caveat, repeated wherever the number appears: climatology sees recent PM2.5
and the model never does, so this is a hard bar rather than a like-for-like
comparison. That is the intent, since climatology is a predictor you could deploy
instead. A like-for-like reference would have to come from the champion's own
training window, which inherits its staleness and so cannot measure it.

### Three windowing rules

**The climatology reference ends a full forecast lead before the window it
scores**, so every hour it averages was observable when that forecast was issued.

**The delivered-margin comparison starts at the run after a promotion.** The
monitor window at the promotion itself begins 14 days before the run date while
the challenger trained up to 7 days before it, so half that window lies inside
the challenger's own training data.

**The version that served a window is not always the version tagged on it.** A
promoting run monitors with the outgoing champion, then overwrites the tag with
the winner, so the two differ by one on promotion runs. `retrospect` keeps
`champion_version` (the tag, correct for when a model started, so decay curves
use it) and `serving_version` (correct for what was in service, so the retraining
comparison uses it). The first run has no predecessor at all, since the bootstrap
champion predates every monitoring cycle, so a first run that promotes reads its
outgoing version from the registry as the lowest registered version. Kraków and
Los Angeles both promote on their first run; without that rule each lost a real
promotion from the gate calibration and gained a phantom week of credit.

---

## Serving the champion

**The alias is the contract, not a version number.** Nothing in the service pins
a version. Promote in the registry, call `POST /reload`, and the new champion is
live without a redeploy. That alias is the seam joining the weekly loop to
production.

**Serving never writes to the tracking store.** It sets the tracking URI and
reads. It does not call `setup()`, which would create an experiment as a side
effect; a read-only consumer should leave no trace.

**The hour encoding is not reimplemented.** `add_cyclical_features` is the same
function the training path uses, so served features cannot diverge from trained
ones. That closes training/serving skew by construction rather than by
discipline.

Timestamps are normalised to naive UTC before the hour is read off them, because
the training data is GMT and an aware timestamp from another zone would put the
diurnal encoding out of phase. `tests/test_serving.py` asserts that `12:00Z` and
`14:00+02:00` predict identically.

Predictions return twice: `pm25` floored at zero for the consumer, and `pm25_raw`
as the model said it. A clamp that hides what a model is doing is how you stop
noticing that it is doing it.

The Docker image replays the loop from the committed parquet cache at build time
instead of copying the SQLite backend in. That is forced rather than chosen:
MLflow stores artifact locations as absolute URIs, so a backend built on a
developer's machine resolves to paths the container lacks. Rebuilding inside the
image also proves the replay is deterministic, reproducing the same champion with
the same baseline RMSE from the same committed data. The install is editable for
a related reason: `tracking.REPO_ROOT` derives from the package file's location,
so a non-editable install would put the backend inside `site-packages`.

---

## What is recorded

**Per run, to MLflow.** Time-series metrics `data_drift_psi`,
`perf_drift_ratio`, `champion_rmse`/`mae`/`r2`, `champion_baseline_rmse`,
per-feature `psi_*` and `ks_*`, plus `challenger_rmse`, `champion_rmse_holdout`
and `performance_gap` when a challenger exists. Tags record `drift_detected`,
`retrain_triggered`, `retrain_reason` and `promotion_decision`. Two artifacts per
run, the champion's predictions and a per-feature distribution report, let the
dashboard show per-run detail without touching the data source.

**Per version, to the registry.** Learned coefficients as tags, and the
`champion` alias moved on promotion, which gives an auditable version history.

> The Model Registry needs a database backend, so this uses a local SQLite file.
> MLflow 3 replaced `Staging`/`Production` stage transitions with aliases.

**Per city, to the repository.** Raw hourly observations are committed as
`data_cache/*.parquet` rather than re-fetched, so every chart matches a fixed,
inspectable dataset. The forecast lead is part of each filename, because one
place over one span holds different features at lead 0 and lead 7. The weekly
Action commits `mlflow_scheduled.db` back so each cycle continues from the last,
and the published site carries a distilled `data.json` plus one raw CSV per city.

---

## Putting an interval on it

Every headline this project reports is a statistic over a few dozen weekly
windows. Until 2026-08-11 all of them were published as bare point estimates,
which is the one place the project was not applying its own standard to itself.
[`stats.py`](../src/driftloop/stats.py) is the correction, and
[`scripts/uncertainty.py`](../scripts/uncertainty.py) regenerates every figure
in [evaluation.md](evaluation.md) that carries a bracket.

**The observations are not independent, and an ordinary bootstrap is wrong here.**
Two effects compound. A monitor window is 14 days wide and the replay steps 7
days, so consecutive windows share half their hours by construction. And
pollution episodes persist: a bad week is followed by a bad week far more often
than chance. Measured, the lag-1 autocorrelation of the paired error series runs
from 0.08 (Johannesburg) to 0.85 (Los Angeles).

An IID bootstrap resamples single weeks and destroys that structure, so the
resampled series looks like it carries more independent information than it
does, and the interval comes out too narrow, which is the direction that turns
noise into a finding. The moving-block bootstrap (Künsch, 1989) resamples contiguous
*blocks* of weeks instead, so the dependence inside a block survives.

**Paired statistics are resampled with shared block indices.** The week-by-week
premium compares two models on the same window, so the two error series must be
resampled together; separating them would compare one model's week 4 against
another's week 31 and call the difference uncertainty.
`tests/test_stats.py::test_pairs_are_resampled_together` asserts it by feeding
in two identical series, whose paired improvement must be zero in every
resample.

**Block length is the one free parameter, so the sweep is published.** The
default is `L = round(n ** (1/3))`, the usual rate, floored at 2 and capped at
`n // 2`. The cap matters because a block as long as the series would
resample it unchanged and report zero width, which is the worst thing this
module could do. Because the choice is a judgement,
`sensitivity_to_block_length` re-runs each interval across a range of `L`, and
[evaluation.md](evaluation.md#the-block-length-is-a-knob-so-here-is-the-sweep)
prints the result. It immediately earned itself: Kraków's premium excludes zero
at `L=1` and includes it from `L=4`, so that conclusion is a function of a
modelling choice and is now labelled as one.

**Where the bootstrap cannot help, it says so.** Two failure modes are handled
explicitly rather than papered over. At a boundary, in a city that won every
window it acted on, every resample also wins every window and the interval
collapses to a false [100, 100]; win rates therefore use a Wilson interval,
which handles proportions at 0 and 1 properly. And below four observations the
bootstrap returns an explicitly infinite interval rather than a narrow invented
one, which is what forces the gate's three long-serving promotions to be
reported as three listed numbers instead of a mean with a range.

**No p-values.** The question here is never "is this different from zero" in
isolation but "how big is it, and could it plausibly be nothing", which an
interval answers directly. Where a reader wants the null-hypothesis version, an
interval excluding zero carries it.

The seed and resample count are fixed constants: an interval that moves when you
rerun it is not something a reader can check.

## References

**Uncertainty**

- Künsch, H. R. (1989). *The Jackknife and the Bootstrap for General Stationary
  Observations.* Annals of Statistics, 17(3), 1217–1241. The moving-block
  bootstrap.
- Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap.* JASA,
  89(428), 1303–1313.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.*
  Chapman & Hall. The percentile interval is §13.3.
- Wilson, E. B. (1927). *Probable Inference, the Law of Succession, and
  Statistical Inference.* JASA, 22(158), 209–212.

**Ridge regression**

- Hoerl, A. E., & Kennard, R. W. (1970). *Ridge Regression: Biased Estimation for
  Nonorthogonal Problems.* Technometrics, 12(1), 55–67.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
  Learning*, 2nd ed., §3.4. [Free PDF](https://hastie.su.domains/ElemStatLearn/).
- scikit-learn user guide,
  [Ridge regression and classification](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression).

**Validating models on time series**

- Bergmeir, C., & Benítez, J. M. (2012). *On the use of cross-validation for time
  series predictor evaluation.* Information Sciences, 191, 192–213.
- scikit-learn,
  [`TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

**Drift, and systems that live with it**

- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014).
  *A Survey on Concept Drift Adaptation.* ACM Computing Surveys, 46(4). The
  source of the covariate/concept drift distinction used above.
- Widmer, G., & Kubat, M. (1996). *Learning in the Presence of Concept Drift and
  Hidden Contexts.* Machine Learning, 23(1), 69–101. Where "hidden context" comes
  from, which is what a season is.
- Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning
  Systems.* NeurIPS.
- Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). *The ML Test
  Score: A Rubric for ML Production Readiness and Technical Debt Reduction.* IEEE
  Big Data.

**PSI and distribution distance**

- Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley. The origin of PSI and its
  0.10 / 0.25 conventions.
- Jeffreys, H. (1946). *An invariant form for the prior probability in estimation
  problems.* Proc. R. Soc. A, 186, 453–461.
- scipy,
  [`ks_2samp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).

**Forecast verification**

- Murphy, A. H. (1988). *Skill Scores Based on the Mean Square Error and Their
  Relationships to the Correlation Coefficient.* Monthly Weather Review, 116(12),
  2417–2424.
- Jolliffe, I. T., & Stephenson, D. B. (2012). *Forecast Verification: A
  Practitioner's Guide in Atmospheric Science*, 2nd ed. Wiley.

**Air quality and the weather that drives it**

- Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and Physics:
  From Air Pollution to Climate Change*, 3rd ed. Wiley. Inversions, mixing
  height, wet deposition and hygroscopic growth.
- WHO (2021). [*Global Air Quality
  Guidelines*](https://www.who.int/publications/i/item/9789240034228). The
  5 µg/m³ annual and 15 µg/m³ 24-hour PM2.5 guidelines Melbourne is measured
  against.
- Hersbach, H., et al. (2020). *The ERA5 global reanalysis.* Quarterly Journal of
  the Royal Meteorological Society, 146(730), 1999–2049.
- [Open-Meteo](https://open-meteo.com/) historical forecast and air-quality APIs.
  Free for non-commercial use, and it archives previous forecast runs, which is
  the only reason the seven-day framing is possible.

**Tooling**

- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html),
  including the aliases that replaced stage transitions in MLflow 3.
