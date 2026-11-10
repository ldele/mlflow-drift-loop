# Registration — the five fixes and the model ablation, on the loop as it ships

**Registered:** 2026-11-10, written on a clean working tree at `0e5e9a5`, before any replay
below was run. It was not committed before the run, so git cannot show that order; the record
of it is this file's own date and the session that wrote it.
**Arm code exists at registration:** yes. The five sweep scripts and `ablate_model.py` exist
and have been run before, on the loop as it first shipped. What does not exist yet is any
result of running them on the loop as it ships now.

## Question

Do the verdicts on the five fixes, and the reason given for four of them failing, hold on a
loop whose model is tuned at every fit and which already has a second retrain alarm?

The reason matters more than the verdicts. The claim under test is about the method: a loop
that retries until a challenger passes an exam promotes the luckiest attempt, so making any
single attempt harder or more frequent cannot repair it. That claim was reached on a loop
with a badly set model, and a badly set model leaves more for any retrain to recover.

## What the first run left open

1. Whether each verdict was a property of the loop or of the default penalty it ran at.
2. Whether the second alarm helps by itself on a tuned loop. The side-by-side replay of
   2026-09-22 switched it on together with the tuning, so it cannot say.
3. Whether a model that can bend still takes most of the retraining premium once the
   straight-line model is tuned at every fit.
4. Whether any of it holds in a city the settings were not looked at on.

This experiment answers 1, 2 and 3. It does not answer 4: it runs on the same six cities, and
a result that only repeats on the units it was found on has not been tested out of sample.

## Units

One unit is one weekly monitoring window in one city, on which two arms' serving models are
scored against the same hours. The windows are fixed by `config.PROFILES` and number 19 to 48
per city, 193 in all. They are not independent: consecutive weeks share a season. The six-city
table puts the independent information at about 2 to 10 effective observations per city.

The six cities are the six the project already has, chosen for contrast and not sampled. The
input data is the committed cache, one file per city, by the first 16 hex digits of its sha256:

| city | file in `data_cache/` | sha256 |
|---|---|---|
| Kraków | `openmeteo_50p0647_19p945_2025-05-01_2026-07-15_lead7d.parquet` | `757db1655d0a24ea` |
| Delhi | `openmeteo_28p6139_77p209_2025-05-01_2026-07-15_lead7d.parquet` | `ee2960b895348c21` |
| Los Angeles | `openmeteo_34p0522_-118p2437_2025-09-01_2026-07-20_lead7d.parquet` | `6e06a4f4ff8ddfcd` |
| Santiago | `openmeteo_-33p4489_-70p6693_2025-10-15_2026-07-15_lead7d.parquet` | `e719f6529a0f6f6f` |
| Johannesburg | `openmeteo_-26p2041_28p0473_2025-11-01_2026-07-15_lead7d.parquet` | `f84b84ae14e6558c` |
| Melbourne | `openmeteo_-37p8136_144p9631_2025-09-01_2026-07-15_lead7d.parquet` | `871c84225b80f483` |

## Arms

Every arm is a full replay of one city under one setting. A changed setting changes which
models exist, so nothing is estimated from the existing champions.

| experiment | reference arm | arms compared with it |
|---|---|---|
| second retrain alarm | alarm off, model tuned | fires at skill below −0.5 (what ships), −0.25, 0 |
| longer exam | 7 days (what ships) | 10, 14, 21 days |
| re-certification | off (what ships) | every 14, 28, 35, 56 days |
| confidence-aware gate | off (what ships) | 0.80, 0.90, 0.95 |
| rollback | off (what ships) | judged at 14, 21, 28 days |
| model class, Delhi and Los Angeles | Ridge as shipped | Ridge tuned once, tree tuned once, both with the second alarm on |

The settings are the ones the first run used, so the two runs differ only in the loop
underneath. That is 16 settings across six cities, 96 arm-and-city pairs, plus the ablation.

The reference arm is the instrument check. In four of the five sweeps it is the loop as it
ships, so it has to reproduce the six published premiums to one decimal. Replays are
deterministic, so an arm that changes nothing would reproduce the reference arm exactly, and
no separate inert arm is run. The positive control is the retraining premium itself: the same
statistic, on the same windows, separates Delhi's loop from never retraining at
+22.8% and cannot separate Kraków's, which shows the size of effect each city can carry.

## Predictions

"Clears zero" means the 95% interval on the week-by-week difference, over the weeks the arm
served a different model from the reference arm, excludes zero.

| # | Prediction | Comparison that tests it | Fails if |
|---|---|---|---|
| P1 | The second alarm at −0.5 improves Los Angeles by itself on a tuned loop. | −0.5 against off, Los Angeles | the interval covers zero or is negative |
| P2 | At −0.5 the alarm harms no city. | −0.5 against off, each city | any city's interval clears zero on the negative side |
| P3 | The aggressive settings still cost a city accuracy. | 0 and −0.25 against off | no city clears zero on the negative side at either setting |
| P4 | A longer exam still changes nothing measurable. | 10, 14, 21 days against 7 | one exam length clears zero, in the same direction, in two cities |
| P5 | Re-certification still trains and promotes more and buys no accuracy. | each cadence against off | one cadence clears zero on the positive side in two cities |
| P6 | The confidence-aware gate is still harmful. | each confidence against off | fewer than two arm-and-city pairs clear zero on the negative side, or any clears it on the positive side |
| P7 | The stricter gate still widens the gap between what the exam promised and what the model delivered. | promised minus delivered, 0.95 against off, per city | the gap narrows in four or more of the six cities |
| P8 | Rollback still does no harm and shows no gain. | each window against off | two or more pairs clear zero in either direction |
| P9 | Under a tree tuned on the same protocol, Delhi's premium stays clear of zero and below the tuned Ridge's. | premiums of the two tuned-once arms, Delhi | the tree's interval covers zero, or its premium is the higher one |
| P10 | Los Angeles shows no measurable effect under either tuned-once model. | the two premiums, Los Angeles | either interval excludes zero |

How sure these are. P4, P5 and P8 are the safest: they predict no detectable change from a
design that can rarely detect one. P6 and P7 are the ones the method claim rests on, and I
put them at about 60%: the mechanism does not depend on the penalty, but a tuned model makes
challengers more alike, which shrinks every exam margin and could leave the gate blocking
almost everything. P1 is the least certain. The first run's gain in Los Angeles came from
replacing one bad model that served for 35 weeks, and tuning alone may already have changed
which model that is.

## Scoring

The statistic is the one every table on the page uses: the paired percentage improvement in
RMSE between two arms on the same weekly windows, over the weeks the two served different
models, with a 95% moving-block bootstrap interval (`driftloop.stats`, 2,000 resamples, fixed
seed). The figure over every week is reported beside it.

A verdict on a fix needs the same setting to clear zero in the same direction in **two
cities**. One arm-and-city pair clearing zero is not a result. The first run produced 55
intervals, 15 of them clear of zero; the re-run will produce up to 96, none corrected for
the others, and at a true rate of zero about five would clear it by chance if the intervals
held their nominal 95%.

They probably hold less. With about five independent observations per city the design is
small, and a simulation of a five-unit design, corrected for 16 comparisons, reads:

```
design: 5 units, 1 samples each, 50% of units moved; true mean difference -0.475; 2000 simulated runs
interval level 0.9969 (0.95 corrected for 16 comparisons)
                                    t interval  percentile  should be
contains the true value                  0.938       0.938      0.997
excludes zero, true effect zero          0.001       0.018      0.003
excludes zero, planned effect            0.024       0.158
mean half-width of the t interval: 1.370
note: the planned effect is detected in fewer than 0.8 of runs. Add units, not samples: repeated samples of one unit add little.
```

That simulation is of independent units and a plain percentile interval, and the page uses a
block bootstrap on autocorrelated weeks, so it is an approximation of this design and not a
calibration of it. What it supports is the direction of the caution: a percentile interval on
five units is too narrow and misses a real effect most of the time. So "nothing measurable"
below means not detectable at this size, and is never read as "no effect".

The first run's intervals give the size each experiment could resolve. Their median
half-width was 1.3 points for the exam length, 2.7 for the confidence gate, 5.5 for rollback,
7.7 for the second alarm and 8.6 for re-certification. An effect smaller than that is below
what the sweep can see.

## Void condition

A sweep is void for a city where its reference arm fails to reproduce that city's published
premium to one decimal. The ablation is void for a city where the tuned tree is not the better
model over the replay, since then neither arm fits and the comparison says nothing about
model class. In the first run the tree's margin in Los Angeles was 0.02 µg/m³, so that city
may well be void.

## What each outcome would decide

| Outcome | What it means | What follows |
|---|---|---|
| P1 or P2 fails | the second alarm was adopted on evidence that does not carry over | D1 is reported to the owner as reopened; the shipped setting is not changed by this analysis |
| P6 and P7 hold | the retry bias is a property of the procedure, on two differently set loops | the README states it as a finding scored twice on the same six cities |
| P6 or P7 fails | the harm was specific to the first loop | the method claim is withdrawn to "on the loop as first shipped", and the README's finding 5 is rewritten |
| any fix meets the two-city rule on the positive side | a mechanism pays on the current loop | reported as scored once; switching it on is the owner's decision |
| P9 fails with the tree's premium at zero | on a tuned loop the premium is about the model class | the six-city table is reported as a property of a linear model |
| every prediction holds | the first verdicts repeat | the pages drop "not re-run" and say what was re-run and where it still was not tested |

## Limits known in advance

The same six cities, picked for contrast, carry both runs. Agreement between the runs shows
the verdicts survive a change of loop. It does not show they would survive a seventh city.

The design is small. Two to ten effective observations per city means most real effects of a
few percent would go unseen, and the intervals are narrower than their label.

The settings were chosen before the first run and are reused, but the reading rule of two
cities is new. It is stricter than the first run's reading, which discounted single results
case by case. The first run's verdicts are re-read under the new rule in the write-up, so the
two runs are compared on one standard.

The tree is tuned once, on the first training window, while the shipped Ridge is tuned at
every fit. The ablation therefore compares the two tuned-once arms with each other and keeps
the shipped arm only as the instrument check.

## Addenda

**2026-11-10, after every result had been seen.** Nothing above was edited. Two things about
the predictions, found in scoring them:

P6's failing condition counted arm-and-city pairs. Santiago at 0.90 and Santiago at 0.95 both
clear zero as a harm, so P6 holds as worded, but the two settings blocked the same promotions
and produced the same replay. That is one result counted twice. The write-up scores the gate
under the two-city rule in "Scoring", where it is not shown harmful, and takes the more
cautious branch of the outcome table: the method claim is narrowed to the loop as first
shipped.

P7's failing condition was a narrowing in four or more cities. The gap narrowed in three and
widened in three, so P7 did not fail, and it has no support either. The write-up says both.

Results are in `docs/evaluation.md`, "The five fixes and the model ablation, on the loop as
it ships". `python scripts/score_fixes.py` recomputes the verdict table from the sweep
outputs.
