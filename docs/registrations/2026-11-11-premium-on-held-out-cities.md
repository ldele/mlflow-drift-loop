# Registration — what retraining is worth, on cities the study was not built on

**Registered:** 2026-11-11, written on `53ae184` with the owner's uncommitted trim in the working
tree, which touches no code this test uses. To be committed by the owner before any interval in
this file's scope is computed and before any seasonal swing below is measured. The commit that
adds this file is the evidence of that order.
**Arm code exists at registration:** in part. The replays exist: `scripts/held_out_cities.py`
built the nine cities' replays and ran them on 2026-11-11. The code that takes the interval from
them, and the code that measures the swing, are not written.

## Seen before writing: the point estimates

The previous test, of the second alarm, already stored this test's point estimates. The
`paired` column of `outputs/held_out/alarm.csv`, committed in `adfc357`, is the week-by-week
premium against the first model, never retrained, for each of the nine cities. On the
alarm-on rows it is the loop as it ships. The author of this file read that column while
checking what was stored, before writing anything below. The values range from about −4% to
about +26%, and three cities are above +20%.

What has **not** been computed or seen: any interval on those values, and any seasonal swing of
any of the nine cities. The rule below is built so the threshold that sorts the nine comes from
the six cities alone. Even so, a predictor chosen by someone who has seen the outcomes is
weaker evidence than one chosen blind. The strength this test can claim is set by that, under
Limits.

For the same reason this file gives no confidence figure for each prediction, unlike the two
registrations before it. A confidence stated after seeing the point estimates would say nothing
about the design.

## Question, in plain words

The project's headline is that retraining pays where a city's air has a strong season and shows
nothing measurable where it does not. That was measured on six cities picked by hand for
contrast. **Can a city's seasonal swing in PM2.5, measured on a year before the replay, tell in
advance which of nine new cities retraining will pay in?**

"Retraining pays" means what it means in the six-city table: over the weeks the loop served a
retrained model, its forecast error is lower than the first model's would have been on the same
hours, and the 95% interval on that difference is above zero.

## What earlier runs left open

1. Whether the premium is positive at all outside the six.
2. Whether the season explains where it is positive, or whether the six were chosen in a way
   that made it look so.
3. Whether retraining ever does measurable harm against leaving the first model in place.

This test answers 2, which is the headline, and scores 3. It reports 1 but cannot claim it,
since the point estimates were seen.

## Units

**The city** is the unit for the headline's prediction. There are nine: Tokyo, Shanghai, São
Paulo, Mexico City, Cairo, Dhaka, Karachi, Buenos Aires and Istanbul, the nine walked in the
order fixed by `docs/registrations/2026-11-10-second-alarm-on-held-out-cities.md`. Their data
files and hashes are in `outputs/held_out/candidates.json`. Lagos, Manila and Kinshasa were not
reached in that walk, have never been fetched, and are not part of this test.

**The weekly monitoring window** is the unit inside a city. Each window is scored twice on the
same hours: by the model the loop served, and by the city's first model, never retrained. Only
the windows in which the loop served a retrained model count, as in the six-city table. A city
holds 30 to 47 windows, and by the six-city measure about 2 to 10 independent observations.

**The replays are the ones already run.** Each city replays the loop as it ships
(`config.SHIPPED`, second alarm at −0.5), on the replay windows set by rule in the earlier
registration: the first model trains on the cleanest 75 days ending between 2025-07-15 and
2025-11-27, the first weekly run is 15 days later, the last is 2026-07-10. Nothing about the
loop or the windows changes for this test.

## The predictor: seasonal swing, measured on the year before

**Swing** is the highest calendar-month mean of hourly PM2.5 divided by the lowest, over the
twelve months 2024-05-01 to 2025-04-30. A ratio, because the premium is a ratio of errors and
the six cities' swings are already written as ratios.

That year ends before any city's first model trains (the earliest training window opens
2025-05-19) and before any window is scored. The swing therefore uses no hour the loop trained
on or was scored on. It is the kind of number an operator could look up before deciding where
to run the loop.

The data: hourly `pm2_5` from the Open-Meteo air-quality endpoint the project already uses
(`driftloop.data.openmeteo.AIR_QUALITY_URL`), at each city's coordinates as listed in
`scripts/held_out_cities.py` and, for the six, in `config.PROFILES`, with the same timezone
setting the project's source uses. A month counts only if at least half its hours return a
value.

**The threshold** comes from the six cities and nothing else. Measure the swing the same way for
the four whose week-by-week premium on the loop as it ships clears zero (Delhi, Santiago,
Johannesburg, Melbourne, per `outputs/uncertainty.json`). The threshold is the smallest of those
four swings. A held-out city at or above it is predicted to **pay**; below it, predicted **not to
pay**.

The same threshold is also applied to Kraków and Los Angeles and reported. That is in-sample and
is not scored. By the swing column of the README (monthly lows and highs over each replay span)
Kraków's is the widest of the six, about 7×, and its premium covers zero, so the rule is expected to miss Kraków in-sample.

## Outcome in each city

The week-by-week premium against the first model, never retrained, with its 95% moving-block
bootstrap interval: `retrospect.retraining_uncertainty(...)["when_it_acted"]`, the statistic and
interval of the six-city table's "retraining, week by week" column. Each city is read as one of:

- **pays**: the interval is entirely above zero;
- **harms**: the interval is entirely below zero;
- **not shown**: the interval covers zero, or the city has fewer than 5 windows served by a
  retrained model, in which case no interval is read (the threshold the previous test used).

## Predictions

| # | Prediction | Comparison that tests it | Fails if |
|---|---|---|---|
| P1 | The swing, with the threshold set on the six, sorts the nine cities into pays and the rest. | each city's predicted label (pays / not) against its outcome (pays / not pays, where "harms" and "not shown" both count as not pays) | it sorts 6 or fewer of the 9 correctly |
| P2 | Retraining harms no held-out city. | each city's outcome | any city reads **harms** |

P2 is the weaker test. The most negative point estimate already seen is about −4%, so it is
likely to hold, and it is scored for completeness rather than as evidence.

Reported, not scored: the number of cities that pay; the rank correlation between swing and the
point premium across the nine; and the chance that a random assignment of the same number of
"pays" labels to the nine cities would sort at least as many correctly, computed exactly by
counting.

## Controls, run before any held-out interval is read

- **Positive control.** The same scoring path, run on Delhi's committed replay, must reproduce
  its published premium and interval, +22.8% [+11.8, +36.4], to one decimal. Delhi is where the
  effect must appear.
- **Negative control.** The same for Los Angeles, −3.5% [−25.6, +3.0], where it must not.
- **Reproduction.** Each held-out city's re-run must reproduce the `paired` value its alarm-on
  row has in `outputs/held_out/alarm.csv` to 0.01. The replays are deterministic, so a mismatch
  means the run is not the one that was already seen.

There is no inert arm. The comparison is the loop against its own first model on the same hours;
a city where no retrained model is ever served has no windows to compare and reads "not shown".

## Scoring

The statistic and interval are fixed by the project: the median over windows of the ratio of
served to first-model error, as a percentage improvement, with a 95% moving-block bootstrap
interval (`driftloop.stats.block_bootstrap`, default block length, seed and resamples).

Nine intervals are read at an uncorrected 95%, as every table in the project is. If retraining
did nothing anywhere, about one city in twenty would still read "pays" or "harms" by chance, so
roughly a one-in-three chance that at least one of the nine does.

The interval was checked by simulation for the registration of 2026-11-10: on about five
independent observations a percentile interval is too narrow and still misses a real effect most
of the time. The six-city intervals are 5 to 42 points wide. So a city with a real premium of a
few percent will usually read "not shown". This biases P1 one way: it makes "predicted to pay,
did not" the likely error, and it can make P1 fail when the swing is right and the measurement
too weak.

## Void condition

- P1 and P2 are void if the positive or negative control fails, or if any city fails
  reproduction.
- P1 is void if the swing cannot be measured for any of the four threshold cities or any of the
  nine: fewer than 10 of the 12 months counting under the rule above.

## What would count as the headline failing

The headline is that retraining pays where the season is strong, not where it is weak. It fails
on these cities if **P1 fails**: the season, measured in advance, does not tell where retraining
pays. It is contradicted outright if **P2 fails**: retraining does measurable harm somewhere.

If P1 holds only because almost every city is on one side of the threshold and almost every city
has the same outcome, the exact chance figure will say so, and the result is reported as weak.

## What each outcome would decide

| Outcome | What it means | What follows |
|---|---|---|
| P1 holds, chance figure small | the season measured in advance picks where retraining pays, on nine new cities | the headline is described as scored once out of sample, with the seen point estimates stated beside it |
| P1 holds, chance figure large | the sort is right but no better than labelling at random would be | reported as consistent with the headline and not as support for it |
| P1 fails, mostly "predicted to pay, not shown" | the season may be right and the measurement too small, or the season is not what matters | the headline is narrowed to the six cities it was found on; the pages say it did not carry over |
| P1 fails, mostly "predicted not to pay, pays" | retraining pays more widely than the season predicts | the headline's "nothing measurable where the season is weak" is withdrawn |
| P2 fails | retraining against the first model did harm in a new city | reported first, ahead of P1, and to the owner as a question about the promotion gate |
| void | the controls or the swing data cannot carry the question | reported as such, with what failed |

The loop's configuration is not changed by any outcome of this analysis.

## Limits known in advance

1. **The point estimates were seen before the predictor was set.** The threshold is mechanical
   and comes from the six, but the choice of swing as the predictor, and of a ratio over a
   difference, was made by someone who had seen them. The brief for this test proposed the swing
   before they were read, which limits but does not remove the problem. At best the result is
   "scored once", and a reader may reasonably discount it further.
2. **Nine cities, each with 2 to 10 independent observations.** A modest real premium will often
   read "not shown", so P1 can fail for lack of measurement. A pass needs both a right predictor
   and a measurable effect.
3. **One year of swing predicts another year's season.** If 2024-25 was unusual in a city, the
   swing misdescribes the replay year. That adds noise to the predictor, which makes P1 harder to
   pass, not easier.
4. **The replay windows differ from the six.** The six were picked by hand to put a clean season
   before the one that spoils it; the nine follow a rule that suits the northern hemisphere
   better. A city whose replay does not reach its polluted season may show no premium whatever
   its swing.
5. **The candidates are not a sample of cities.** Twelve large cities listed from memory, in one
   order; nine reached. The result holds for these nine.
6. **PM2.5 is CAMS model output,** not a station measurement, in every city, as for the six.
   Both the swing and the premium are measured on it.

## Addenda

<!-- Dated. Each states what changed, why, and whether any result had been seen. -->
