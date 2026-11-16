# Registration — does the loop beat retraining on a calendar?

**Registered:** 2026-11-15, written on `7a7447b` with a clean working tree. To be committed by
the owner before `scripts/calendar_baseline.py` is run on any city. The commit that adds this file
is the evidence of that order.
**Arm code exists at registration:** yes. Both new arms are settings the loop already has
(`recertify_days`, `promotion_margin`), so nothing in `src/` changes. The script that replays and
scores them, and its tests, are in the same commit. The tests run on the synthetic source only, and
the script has not been run on city data.

## Seen before writing

- **Every published result**, including the re-certification sweep on the loop as it ships
  (`outputs/recertify_sweep.csv`). Its 14-day cadence is the arm closest to the weekly exam
  below. Against the loop with re-certification off, it read −3.7% [−10.9, +2.2] in Kraków and
  −4.6% [−12.3, +7.4] in Delhi over the weeks it changed something. It covered zero in every city
  where it changed anything. The weekly cadence has never been run.
- **No replay of either weekly arm**, in any city.
- **Outside work.** A 2026 preprint on streaming models found that simple periodic retraining
  beat retraining triggered by drift, under both sudden and gradual drift
  ([arXiv 2608.19488](https://arxiv.org/pdf/2608.19488)). It informs P2. It uses a different
  model, a different task and immediate labels, so it is not evidence about this loop.

## Question, in plain words

Every premium the project publishes compares the loop with its first model never retrained.
That shows retraining is worth something. It does not show the loop's machinery is: the drift
alarms that decide *when* to retrain, and the exam that decides *whether* to ship. The obvious
rival is a calendar: **would retraining every week, whatever the alarms say, have done as well?**

There are two calendars, because the loop does two things:

- **Weekly exam.** Train a challenger every week and let it sit the usual exam. Same exam, no
  alarms.
- **Weekly refit.** Train a model every week and serve it. No alarms, no exam.

## Units

**The city.** Fifteen of them:

- **The six**, on the replay windows in `config.PROFILES`: Delhi, Santiago, Kraków, Johannesburg,
  Melbourne, Los Angeles. Every setting of the loop was chosen on these.
- **The nine** walked by `docs/registrations/2026-11-10-second-alarm-on-held-out-cities.md`, on the
  windows that registration fixed by rule: Tokyo, Shanghai, São Paulo, Mexico City, Cairo, Dhaka,
  Karachi, Buenos Aires and Istanbul. Their data files and hashes are in
  `outputs/held_out/candidates.json`. They have been used twice before (the second alarm, the
  premium), and never for this question.

**The weekly monitoring window** is the unit inside a city. All three arms replay the same weeks,
so every window has an error from each arm on the same hours, and the comparison is paired.

## The arms

All three are the loop as it ships (`config.SHIPPED`: penalty tuned at every fit, second alarm at
−0.5) on the same windows. Each starts from the same first model and differs only as stated:

| arm | setting changed | what it does |
|---|---|---|
| **shipped** | none | the reference: retrain when an alarm fires, ship if the exam is passed by 5% |
| **weekly exam** | `recertify_days=1` | a challenger every run, which sits the same exam |
| **weekly refit** | `recertify_days=1`, `promotion_margin=-inf` | a challenger every run, always shipped |

`recertify_days=1` fires every run, because at a seven-day step a certificate is always at least
seven days old. With `promotion_margin=-inf` every challenger clears the exam. The alarms stay on in
both weekly arms and change nothing there, since every run already retrains.

The weekly refit trains on the same window as every challenger: the 180 days that end where the
seven-day exam week starts. A refit written from scratch would also use that last week. Keeping the
window the same means the arms differ in when they retrain and whether they examine, and in
nothing else. Limits, item 2.

## Outcome in each city

For arm X against arm Y: the week-by-week improvement of X over Y, `(1 − median(X/Y)) × 100`,
over the windows where the two arms' errors differ (`stats.differing`). Its 95% interval is the
project's moving-block bootstrap (`stats.block_bootstrap`, default block length, seed and
resamples). Positive means X forecast better. This is the statistic the five-fix table reads as
"over the weeks it changed the serving model". Each comparison in each city reads as one of:

- **X better:** the interval is entirely above zero;
- **X worse:** the interval is entirely below zero;
- **not shown:** the interval covers zero, or fewer than 5 windows differ.

## The reading rule

Over the fifteen cities, with *b* cities reading X better and *w* reading X worse:

| verdict | when |
|---|---|
| **better** | *b* ≥ 3 and *w* = 0 |
| **worse** | *w* ≥ 3 and *b* = 0 |
| **mixed** | *b* ≥ 1, *w* ≥ 1 and *b* + *w* ≥ 3 |
| **not detectable** | anything else |

Three rather than the two-city rule of 2026-11-10, because there are fifteen cities and not six.
If nothing differed anywhere, and each interval had a 5% chance of clearing zero in a given
direction, which is twice its label because the percentile interval is known to run narrow on
this few independent observations, then three or more cities clearing in one direction would
happen about 4% of the time. The two-city rule on the six alone is also reported, for comparison
with the five-fix table, and is not scored.

## Predictions

The confidences are the author's, given before anything was run.

| # | Prediction | Comparison | Fails if | Confidence |
|---|---|---|---|---|
| P1 | Sitting the exam every week does not beat the shipped loop. | weekly exam against shipped | the verdict is **better** | 75% |
| P2 | The shipped loop does not beat a blind weekly refit. | weekly refit against shipped | the verdict is **worse** | 55% |
| P3 | Under a weekly calendar the exam does no measurable harm. | weekly exam against weekly refit | the verdict is **worse** | 70% |
| P4 | The exam turns most weekly challengers away. | weekly exam's promotions over its runs | the share is half or more in 4 or more of the 15 cities | 70% |

**Why these directions.** P1: the 14-day cadence already covered zero everywhere, and more exams
are more attempts at a noisy one-week test, which the project has found buys luckier winners
rather than better ones. P2 is the prediction this test exists for, and the least sure. A refit on
the last 180 days is always current, and the preprint found the calendar wins. Against that, the
exam stops some bad refits in quiet cities. If P2 fails, it is the first evidence that the loop's
machinery beats a calendar on accuracy. P3: the exam's one-week margin promised +12.7% and
delivered +8.0% [+5.1, +11.0] over its short-serving promotions, so it is noisy but points the
right way.

**Reported, not scored:** each arm's premium against the first model never retrained, the README's
column, in every city; models trained and promotions per arm; the longest silence; the
all-weeks version of each comparison; and the six-city two-city reading.

## Controls, run before any comparison is read

- **Reproduction, the six.** The shipped arm in each of the six must reproduce its published
  week-by-week premium and interval in `outputs/uncertainty.json` to one decimal.
- **Reproduction, the nine.** The shipped arm in each of the nine must reproduce the `paired` value
  of its alarm-on row in `outputs/held_out/alarm.csv` to 0.01.
- **Mechanism.** Both weekly arms must train a challenger on every run, and the weekly refit must
  promote on every run.

## Void condition

- Everything is void if any city fails reproduction or any weekly arm fails the mechanism check.
  The replays are deterministic, so a mismatch means the harness is not measuring the loop it
  claims to.
- A city where a comparison has fewer than 5 differing windows reads "not shown" for that
  comparison. It is not dropped.

## What each outcome would decide

| Outcome | What it means | What follows |
|---|---|---|
| P2 holds, weekly refit **not detectable** or **mixed** | the loop's alarms and exam are not shown to beat a blind calendar on accuracy | the README's findings add it beside the premium: retraining pays, and the alarms and exam are not shown to be why. Their case becomes cost (models trained) and the bound on silence |
| P2 holds, weekly refit **better** | the alarms and the exam cost accuracy against a calendar | the same, plus a question to the owner: a decision record on whether to ship a calendar |
| P2 fails, weekly refit **worse** | the machinery beats a blind calendar | reported as the loop's first measured case for its own design, and led with |
| P1 fails, weekly exam **better** | the alarms miss retrains that would have paid | D2, re-certification, reopened for the owner |
| P3 fails, weekly exam **worse** than refit | the exam picks lucky challengers often enough to do harm | reported next to the gate's calibration, as the strongest form of the winner's-curse finding |
| P4 fails | most weekly challengers pass | the cost reading changes: the exam is not where the calendar's waste is caught |
| void | the harness or the arms did not do what they claim | reported as such, with what failed |

Whatever the outcome, this analysis does not change the loop's configuration. Changing it is the
owner's call, recorded in `docs/DECISIONS.md`.

## Limits known in advance

1. **Small units.** A city's replay holds about 2 to 10 independent observations. A difference
   of a few percent between arms would read "not shown" in most cities. "Not detectable" means
   not detectable at this size, not equal.
2. **The weekly refit is not quite the naive one.** It leaves out the most recent week, as every
   challenger does, so it is a calendar refit one week stale. A refit that also used that week
   might do slightly better, which would lean against P2 failing.
3. **One cadence.** Weekly is the shortest the loop can run and the obvious rival. A monthly
   calendar is not tested.
4. **The six are not out of sample.** Every setting was chosen on them. The nine carry this
   question out of sample, and they were picked by a rule fixed for a different question.
5. **Fifteen intervals, uncorrected,** as everywhere in the project. The reading rule's
   three-city threshold is the only guard against multiplicity.
6. **PM2.5 is CAMS model output,** not a station measurement, in every city.
7. **Cost is counted, not priced.** A weekly arm trains about 30 to 47 models per city against
   the shipped loop's 6 to 21. The table reports the count. Whether that compute is worth a given
   accuracy gain is not this test's question.

## Addenda

**2026-11-16, after every result had been seen.** Nothing above was edited. This file was
committed as `a989c6a` before the script was run on any city, and run on 2026-11-15 with
`python scripts/calendar_baseline.py`. Outputs are in `outputs/calendar/`: `arms.csv`,
`comparisons.csv`, `series.json` and `scores.json`.

Every control passed, so nothing is void. All fifteen shipped arms reproduced what was
published, both weekly arms trained a challenger on every run, and the weekly refit promoted on
every run.

| | comparison | verdict | better | worse |
|---|---|---|---|---|
| P1 | weekly exam against shipped | not detectable: **held** | 1 | 0 |
| P2 | weekly refit against shipped | **better**: held | 3 | 0 |
| P3 | weekly exam against weekly refit | **worse**: **failed** | 0 | 4 |
| P4 | weekly exam promoted in half its runs or more | 0 cities: held | | |

P2 held in the form the outcome table lists second: the weekly refit is better than the shipped
loop, in Dhaka, Buenos Aires and Istanbul, by 2 to 5 points, with no city the other way. All three
are among the nine. On the six alone no comparison reaches a verdict under the two-city rule.

One thing the registration did not anticipate. Its explanation for a P3 failure, that "the exam
picks lucky challengers often enough to do harm", is one reading. Under a weekly calendar a rejected
challenger leaves an older model serving, so the cost the exam shows here may come from the models
it keeps rather than the ones it lets through. The four harms are 0.6 to 2.1 points, and this run
cannot tell the two readings apart. The failure is scored as registered.

A correction to limit 7: the shipped loop trained 2 to 21 models per city, not 6 to 21. Mexico
City trained 2.

Results are in `docs/evaluation.md`, "The loop against retraining on a calendar".
