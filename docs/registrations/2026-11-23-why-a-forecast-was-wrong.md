# Registration — why was a forecast wrong?

**Registered:** 2026-11-23, written on `b3a5b30` with a clean working tree. To be committed by the
owner before any code that splits a week's error exists. The commit that adds this file is the
evidence of that order.
**Arm code exists at registration:** no. Neither the fetch of day-0 weather nor the script that
splits the error exists. The loop and its replays are unchanged.

## Seen before writing

- **Every published result**, including the calendar test (`a989c6a`, `0a9b149`) with its
  all-weeks reading, and each city's served error week by week (`champion_rmse`), which the site
  and the dashboard draw.
- **One probe of the forecast archive:** Kraków, 10 January 2026, temperature and humidity at six
  hours. The archive's plain series, the forecast for the same day, sat 6 to 12 °C from the
  week-ahead forecast. The ERA5 archive sat 0.5 to 4.1 °C from the same-day forecast, colder in
  five of the six hours. This is the reason day-0 weather is taken from the forecast archive and
  not from ERA5 (see "The four scorings").
- **No week's error has been split**, in any city, by any method.

## Question, in plain words

Each week the model in service misses by some amount. Three things can cause a miss, and
retraining can fix only one of them:

1. **The weather forecast was wrong.** The model is fed the forecast issued a week ahead. When
   that forecast missed, the model was fed the wrong weather.
2. **The weather-to-pollution relation moved.** The same weather now brings different air than in
   the months the model learned from: a season turning, heating switched on, fires.
3. **Neither.** What the model cannot see at all: emissions, the recent air, how the target
   itself is made.

**How is each city's error split between the three, and does the second, the only part
retraining can fix, track where retraining paid?**

Why it matters: the calendar test found that the alarms and the exam add nothing measurable over
retraining every week, and retraining itself pays in some cities only. This says why. If the
relation part is small, there is little for any retrain rule to win, and the next step is saying
how far to trust a forecast (plan item 4), not another rule. It also gives each week a reason
for its miss, which the loop cannot give today.

## Units

**The city.** The fifteen of the calendar test: the six, on the replay windows in
`config.PROFILES`, and the nine, on the windows fixed by
`docs/registrations/2026-11-10-second-alarm-on-held-out-cities.md`, with their data files and
hashes in `outputs/held_out/candidates.json`. The loop as it ships (`config.SHIPPED`).

**The week**, inside a city: the seven days between two runs, served by the model in service after
the earlier run. Weeks do not overlap, unlike the loop's 14-day monitor window, so no hour is
counted twice. A city has 19 to 48 of them.

**Gate, before any week is split.** A city enters if both hold:

- the day-0 weather covers at least 95% of the hours in its replay;
- its shipped replay reproduces its published premium in `outputs/held_out/premium.csv` to 0.1.

The day-0 files are fetched after this file is committed. Their hashes go in an addendum before
any split is computed.

## The four scorings

Each week is scored four times, two models by two kinds of weather:

| | weather forecast a week ahead | weather forecast on the day |
|---|---|---|
| **the model in service** | A: the served error | B: its twin, fed day-0 weather |
| **a model refitted through the week** | C: week-ahead weather | D: day-0 weather |

- **A** is the champion in service, rebuilt from its registry tags and fed the week-ahead
  features. It is what the loop measured.
- **B**, the twin, is the same pipeline trained on the same rows with the penalty tuned the same
  way, but trained and scored on day-0 weather.
- **C** and **D** are the same pipeline fitted on the 28 days with the week in the middle, and
  cross-fitted by day: each day of the week is predicted by a model fitted on the other 27 days.
  No hour is scored by a model that saw it, and the refit learns the relation as it stands that
  month. Where the replay ends, the 28 days shift to stay inside the data.
- **Day-0 weather** is the forecast archive's plain series (`historical-forecast-api`, no
  `_previous_day` suffix): the most recent run for each hour, from the same forecasting system as
  the week-ahead features. Not the ERA5 archive, which the loop's lead-0 path reads today. A model
  trained on one system and scored on another would count the gap between the two systems as
  forecast error.

## Outcome in each city

Errors are squared and summed over all the city's hours, so the parts add up:

- **weather part** = ½ [(A − B) + (C − D)]
- **relation part** = ½ [(A − C) + (B − D)]
- **unexplained** = D

The three sum to A. Averaging the two orders, weather first and relation first, removes the
dependence on which is taken first: it is the Shapley split of two factors (Shapley, 1953). Each
part is published as a share of A. Shares come from sums over hours, so a smoggy week weighs more
than a clean one. Each city's shares carry a 95% interval from the project's moving-block
bootstrap over weeks (`stats.block_bootstrap`, default block length, seed and resamples).

A week's own split uses the same formula on that week's sums.

## Predictions

The confidences are the author's, given before anything was run. "Two thirds" is of the cities
that pass the gate, rounded up: 10 of 15, or 8 of 12.

| # | Prediction | Comparison | Fails if | Confidence |
|---|---|---|---|---|
| P1 | What the model cannot see is the largest part. | each city's three shares | unexplained is the largest in fewer than two thirds of the cities | 75% |
| P2 | The relation, the part retraining can fix, is the smallest. | each city's three shares | the relation part is the smallest in fewer than two thirds of the cities | 60% |
| P3 | Where the relation part is larger, retraining paid more. | Spearman correlation across the cities, relation share against the premium in `premium.csv` | the one-sided permutation p-value is 0.05 or more | 55% |
| P4 | Most weeks the loop retrained were not weeks the relation moved. | the weeks the shipped loop retrained, pooled over cities: the share whose relation part is the largest of the three | that share is one half or more | 65% |

In P4 the week is the seven days ending at the run that retrained: the hours the alarm acted on.

**Why these directions.** P1: the model sees six weather inputs and the hour of day, and never the
recent air or emissions. P2: the calendar test found retraining's gains small, and a small
relation part is the simplest reason. Against it, Delhi's premium is +22.8%. P3 is the validity
check: if the relation part measures what retraining fixes, it should rank the cities as their
premiums do. It also fails if the split measures something else, which is why it is registered.
P4: the error alarm reads a 14-day error ratio, which a bad weather forecast inflates as surely as
a moved relation.

**Reported, not scored:** the four cells and three parts per city with intervals; the interaction,
(A − B) − (C − D); each order on its own; the refit window at 14 and 56 days; what retraining
actually reaches, as the share of the relation part that a refit on the 180 days before each week
recovers on week-ahead weather (the calendar test's weekly refit); and the mean share of each part
across cities with a t interval.

## How it is decided

P1, P2 and P4 are decided by counts. Their Wilson intervals (`stats.wilson_interval`, and a
bootstrap over cities for P4's pooled weeks) are reported and decide nothing. P3 is decided by a
one-sided permutation test, 10,000 permutations with a fixed seed. It is the only decision taken
on a test, so no correction for multiple comparisons applies.

The design was simulated before writing, with the experiment-design skill's `interval_check.py`
(15 units, 1 sample each, two thirds of units moved):

```
interval level 0.9833 (0.95 corrected for 3 comparisons)
                                    t interval  percentile  should be
contains the true value                  0.953       0.957      0.983
excludes zero, true effect zero          0.021       0.026      0.017
```

At fifteen units a t interval across cities ran narrow: it held the true value in 0.953 of runs
where 0.983 was asked. So the cross-city t intervals are description and decide nothing. The
simulation draws a yes or no per unit while the shares here are continuous, so it bounds how these
intervals behave rather than measuring it.

## Controls, run before any split is read

- **Reproduction.** Cell A, computed by the new code, equals the loop's logged `champion_rmse` at
  every run of the six, to the precision of the coefficient tags. The nine pass the gate's
  reproduction check.
- **Inert weather.** With the day-0 features replaced by the week-ahead ones, B equals A and D
  equals C in every week, so the weather part is exactly zero.
- **Positive control for the relation part.** The simulated city with the concept dial
  (`drift_strength`) at 2.0 against 0.0, the weather dial (`feature_shift`) at 0, the settings
  `scripts/sweep_knobs.py` uses. The relation share at 2.0 has to exceed the one at 0.0 by a
  block-bootstrap interval clear of zero. The simulated features carry no forecast error, so the
  weather part is zero at both settings, by the inert check.

## Void condition

- Fewer than 12 of the 15 cities pass the gate.
- Any control fails. The replays are deterministic, so a reproduction mismatch means the code is
  not measuring the loop it claims to.

## What each outcome would decide

| Outcome | What it means | What follows |
|---|---|---|
| P1 and P2 hold | most of the error is out of the model's reach, and of what is in reach the weather forecast outweighs the relation: retraining works on the smallest part | led with, beside the calendar result. Plan item 4, saying how far to trust a forecast, before any change to retraining |
| P2 fails | the relation part is not the smallest in a third or more of the cities | retraining has material the loop leaves unused, and the alarms are worth another look, scored against the weekly refit |
| P3 holds | the relation part ranks cities as retraining's premium does | the split can be published week by week as the loop's reason for a miss, on the site and the dashboard, and used in item 4 |
| P3 fails | it does not | the split is not shown to measure what retraining fixes: reported as a description, and not published as a reason for any week |
| P4 holds | most weeks the loop retrained were weather misses or unexplained | evidence for a "do not trust this forecast" path ahead of retraining, in item 4 |
| P4 fails | most weeks the loop retrained were the relation moving | the alarms aim at the right thing, and their lateness, one to three weeks, is the problem |
| void | the code or the data did not do what this file says | reported as such, with what failed |

Whatever the outcome, this analysis does not change the loop's configuration. Changing it is the
owner's call, recorded in `docs/DECISIONS.md`.

## Limits known in advance

1. **The target is CAMS model output,** made from CAMS's own analysed weather. Day-0 weather is
   close to what made the target, so the weather part may read larger here than against a station
   (plan item 5). The ranking of cities is less exposed than the shares.
2. **Day-0 weather is still a forecast,** hours ahead, not the weather that happened. The weather
   part is a lower bound on what a perfect forecast would buy.
3. **The refit window draws the line between relation and unexplained.** A shorter window puts
   more in the relation part, including shifts too brief for any retrain to catch. 28 days is a
   choice. It is reported at 14 and 56 days.
4. **The relation part is an upper bound on what retraining can fix,** because the refit sees the
   week itself. The retraining reach, reported beside it, shows how much a real refit gets.
5. **One model class.** A model that can bend might move error between the relation part and the
   unexplained one. The ablation showed Delhi's gain survives a tree, not that this split would.
6. **Small units.** Fifteen cities, each holding about 2 to 10 independent observations. Counts and
   one rank correlation are what fifteen units can carry. Each city's shares are a description.
7. **The six are not out of sample.** Every setting of the loop was chosen on them. The nine carry
   the question out of sample.

## Addenda

**2026-11-23, the gate's coverage, before any week is split.** Nothing above was edited. The
same-day weather was fetched with `python scripts/split_error.py --gate`, through a new switch on
the source, `day0_from_forecast_archive`. ERA5 stays the default at lead 0, so nothing the loop
reads has changed. The result is in `outputs/split/gate.json`.

Every city's same-day weather covers 100% of the hours of its week-ahead data, so all fifteen pass
this half of the gate. The other half, reproducing the published premium, needs the replays and
is checked by the split before any week is split.

A check that the files hold what the design says, made on the inputs alone: across the fifteen
cities the same-day temperature differs from the week-ahead one by 1.0 to 3.0 °C on average, with
a correlation of 0.83 to 0.96, and the target is identical in every hour. No model has been scored
on these files and no week has been split.

The files, in `data_cache/`, with the first 16 hex digits of each one's SHA-256 as fetched:

| City | Hours | Covered | File | SHA-256 |
|---|---|---|---|---|
| Delhi | 10,584 | 100% | `openmeteo_28p6139_77p209_2025-05-01_2026-07-15_lead0d_forecast.parquet` | `12a4e80121601112` |
| Santiago | 6,576 | 100% | `openmeteo_-33p4489_-70p6693_2025-10-15_2026-07-15_lead0d_forecast.parquet` | `04b8461777bc4c93` |
| Kraków | 10,584 | 100% | `openmeteo_50p0647_19p945_2025-05-01_2026-07-15_lead0d_forecast.parquet` | `44938120ababb1e0` |
| Johannesburg | 6,168 | 100% | `openmeteo_-26p2041_28p0473_2025-11-01_2026-07-15_lead0d_forecast.parquet` | `550bab12eb4dcbe9` |
| Melbourne | 7,632 | 100% | `openmeteo_-37p8136_144p9631_2025-09-01_2026-07-15_lead0d_forecast.parquet` | `fb4741f88e50e77b` |
| Los Angeles | 7,752 | 100% | `openmeteo_34p0522_-118p2437_2025-09-01_2026-07-20_lead0d_forecast.parquet` | `73518e4a0182ec56` |
| Tokyo | 10,584 | 100% | `openmeteo_35p6762_139p6503_2025-05-01_2026-07-15_lead0d_forecast.parquet` | `ee9428d339f879d9` |
| Shanghai | 10,584 | 100% | `openmeteo_31p2304_121p4737_2025-05-01_2026-07-15_lead0d_forecast.parquet` | `96a42c3d64a57759` |
| São Paulo | 10,584 | 100% | `openmeteo_-23p5505_-46p6333_2025-05-01_2026-07-15_lead0d_forecast.parquet` | `f76e0b8508f498d5` |
| Mexico City | 10,584 | 100% | `openmeteo_19p4326_-99p1332_2025-05-01_2026-07-15_lead0d_forecast.parquet` | `02c1bb4fd6f83de8` |
| Cairo | 10,584 | 100% | `openmeteo_30p0444_31p2357_2025-05-01_2026-07-15_lead0d_forecast.parquet` | `2005d6ef60526fad` |
| Dhaka | 10,584 | 100% | `openmeteo_23p8103_90p4125_2025-05-01_2026-07-15_lead0d_forecast.parquet` | `b901afdef5288f3c` |
| Karachi | 10,584 | 100% | `openmeteo_24p8607_67p0011_2025-05-01_2026-07-15_lead0d_forecast.parquet` | `784faf9ebff0ed07` |
| Buenos Aires | 10,584 | 100% | `openmeteo_-34p6037_-58p3816_2025-05-01_2026-07-15_lead0d_forecast.parquet` | `9c9a6bbfe26a65e5` |
| Istanbul | 10,584 | 100% | `openmeteo_41p0082_28p9784_2025-05-01_2026-07-15_lead0d_forecast.parquet` | `f128240937eea832` |
