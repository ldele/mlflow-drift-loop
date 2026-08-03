# Evaluation

Whether it works, and where it does not. For how it works, see
[methodology.md](methodology.md).

## Six cities that disagree

Weather forecasts from the [Open-Meteo](https://open-meteo.com/) historical
forecast archive, joined on the hour with observed PM2.5 from its air-quality
API. Each city trains a model on a clean season and replays week by week into the
season that ruins it, and then out the other side.

| | span | PM2.5 swing | model error, start → worst | retrains / weeks | retraining worth |
|---|---|---|---|---|---|
| **Delhi** | May 25 → Jul 26 | 42 → 127, post-monsoon burning | 20.0 → 99.0 | 9 / 40 | **+43.8%** |
| **Santiago** | Oct 25 → Jul 26 | 18 → 94, winter inversion in a basin | 6.4 → 68.3 | 13 / 22 | **+12.9%** |
| **Kraków** | May 25 → Jul 26 | 8 → 57, winter smog in a basin | 5.1 → 54.5 | 14 / 48 | +0.2% |
| **Johannesburg** | Nov 25 → Jul 26 | 23 → 84, Highveld coal smoke | 11.6 → 82.3 | 11 / 20 | 0.0% |
| **Melbourne** | Sep 25 → Jul 26 | 5 → 15, winter wood heaters | 3.3 → 15.0 | 8 / 31 | 0.0% |
| **Los Angeles** | Sep 25 → Jul 26 | 15 → 29, a mild winter bump | 17.3 → 19.5 | 3 / 37 | −8.9% |

The cities were picked on measurements rather than reputation. Eighteen
candidates were fetched and ranked by PM2.5 swing before any of them was wired
in, which cost two obvious choices. Sydney turned out flat, at 1.5×, with no
story in it. No Brazilian city worked either: the Amazon burning-arc cities peak
at only about 14 µg/m³ in this window, and São Paulo swings 1.7×, less than Los
Angeles does.

The same exercise settled the European slot. Over a current twelve-month window
Kraków swings 6.9×, ahead of Milan (6.2×), Tuzla (6.2×) and Katowice (6.1×),
and comfortably ahead of the cities that actually make the "worst air in Europe"
headlines: Sarajevo 4.7×, Skopje 4.4×, Sofia 3.7×. Reputation was a poor guide
again.

### What a full year exposed

Every city originally stopped at its dirty-season peak. On that half of the
story, retraining looked like an unambiguous win: +10.1% in Kraków, +66.7% in
Delhi.

Running them through the return trip, as the air gets clean again, reversed the
sign. Retraining came out **29.6% worse** in Kraków and **4.3% worse** in
Delhi, and the served model finished *last* of six predictors in both.

The cause was the retraining rule rather than the loop. A challenger trained on
the previous 45 days only ever sees one season, so it is excellent in the season
it was born in and wrong as soon as the year turns. The loop was behaving
correctly throughout. Each challenger won its holdout exam at the moment it was
promoted, and then aged badly.

Widening `challenger_train_days` from 45 to 180 fixed it:

| | 45-day window | 180-day window |
|---|---|---|
| Delhi, retraining worth | −4.3% | **+43.8%** |
| Delhi, served model rank | 6th of 6 | **1st of 6** |
| Kraków, retraining worth | −29.6% | +0.2% |
| Kraków, served model rank | 6th of 6 | 2nd of 6 |

Only a full annual cycle made this visible, which is the argument for running the
replay past the point that flatters the system.

### Where it stops paying

Los Angeles is the control, and the measurements chose it for that. It was picked
as the summer-smog city; hourly PM2.5 over 2025–26 peaks in November and bottoms
out in June. Its model barely moves across 37 weeks, only three retrains ever
fire, and retraining costs 8.9%. There has to be drift for a drift loop to earn
anything.

Johannesburg is where the promotion gate does the most visible work. Error
climbs from 11.6 to 82.3 µg/m³, the worst on the page, and eleven retrains
produce only three promotions: the other eight challengers failed to clear the 5%
margin and were thrown away. Retraining nets 0.0%. The loop spends the
effort, the gate declines to pay, and nothing ships that did not earn it.

Melbourne is the counter-intuitive case. Its air stays near the WHO guideline all
year, yet the model still decays to more than four times its training error, so
clean air does not imply a stable model.

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
weather drifts *further* from the training window than anywhere else on the
page, while the model runs at well under its training error. Data
drift screams, performance is fine, and the loop correctly does nothing. A system
that retrained on distribution shift alone would have burned twenty retrains
there for no reason.

A seventh source, **Live schedule**, is not a city. It is the same Kraków data
run one incremental cycle at a time by a weekly GitHub Action, accruing its own
history over calendar time. It has run only a handful of cycles so far, and the
page says so on its own tab rather than letting two points pass for a trend. An
eighth, synthetic, has two independent drift knobs and backs the offline
correctness proof in [`sweep_knobs.py`](../scripts/sweep_knobs.py). It is not
published.

## Does the model beat anything?

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
| **served model** | **40.33** | 23.48 | 17.49 | **21.01** | **3.87** | 11.05 |
| never retrained | 71.71 | 26.96 | 17.52 | 21.01 | 3.87 | **10.14** |
| climatology | 42.81 | 26.33 | 17.86 | 21.89 | 4.19 | 10.72 |
| training mean | 45.29 | 26.32 | 18.40 | 23.62 | 3.98 | 10.46 |
| persistence | 49.64 | 20.71 | **16.80** | 27.71 | 5.34 | 13.79 |
| seasonal naive | 49.60 | **19.72** | 18.56 | 30.41 | 5.11 | 14.31 |

- The model is the best predictor in Delhi, Johannesburg and Melbourne, and
  second in Kraków. It beats persistence in four of six.
- Retraining pays where drift is real: +43.8% in Delhi, +12.9% in Santiago. Delhi's never-retrained model ends up worse than a constant, which is
  what happens when nobody intervenes.
- It costs 8.9% in Los Angeles, and breaks even in Johannesburg and Melbourne.
  Retraining a city whose world barely moves fits noise.
- Santiago is where the baselines win. Its pollution is persistent enough
  from week to week that repeating a stale reading beats forecasting from
  weather, even though retraining still clearly helps the model itself.

`alpha=1.0` ships. Forward-chaining CV wants far heavier regularisation in most
cities, up to 1000, which is itself a signal: the fitted relationship is weak
enough that shrinking it toward zero costs almost nothing (between 0.1% and 11.9%
depending on the city).

## Does the detection actually work?

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

The two-signal design rests on that property, and it is the one thing the six
cities cannot demonstrate. It is published on the page for the same reason it
is here: it is the evidence, and the cities are the application.

## Limitations

- R² is still weak, and it belongs up front. Widening from three
  features to eight improved every city, but R² on the most-drifted window is
  still near zero or negative in several. Predicting an hour's PM2.5 from a
  week-old weather forecast is hard, and the page should be read with
  that in mind: the loop is the demonstration, not the model.
- The retrain trigger is relative to each model's own baseline. A model that
  was trained on a hard season has a high baseline error, so "1.25× worse than
  training" is a soft bar for it. Delhi's late replay shows this: error above 100
  µg/m³ with the ratio still under the threshold, so no retrain fires. An
  absolute floor alongside the ratio would catch it.
- 180 days is argued rather than tuned. It was chosen to span more than one
  season after 45 was shown to fail, not swept. A proper sweep of the retrain
  window against retraining value is the obvious next experiment.
- Boundary layer height is missing, and it is the feature I most want. It sets
  the volume pollution is diluted into and would likely matter more than anything
  in the list. Open-Meteo does not archive previous model runs for it, so at a
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
