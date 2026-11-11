# Evaluation

Whether it works, and where it does not. For how it works, see
[methodology.md](methodology.md).

## The loop as it ships now

Two settings changed on 2026-11-10, and every city was replayed again under them.

- **The model's regularisation strength is chosen at every fit**, by
  cross-validation that keeps time in order. It had been a library default
  nobody chose ([D5](DECISIONS.md)).
- **A second retrain alarm runs beside the first.** It fires when the model falls
  50% behind a plain daily profile of recent air, a comparison no promotion can
  reset ([D1](DECISIONS.md)).

What retraining is worth under that loop, week by week against the same first
model never retrained, with what this page reported before the change beside it:

| | retrains | shipped | across the replay | week by week | week by week, as first shipped |
|---|---|---|---|---|---|
| **Delhi** | 8<!--fig:delhi.retrains--> | 6<!--fig:delhi.promotions--> | +17.9% [+9.7, +42.5]<!--fig:delhi.replay--> | **+22.8% [+11.8, +36.4]**<!--fig:delhi.acted--> | +49.4% [+33.7, +61.8]<!--fig:first.delhi.acted--> |
| **Santiago** | 13<!--fig:santiago.retrains--> | 7<!--fig:santiago.promotions--> | +16.9% [+0.0, +42.5]<!--fig:santiago.replay--> | **+17.5% [+8.6, +37.2]**<!--fig:santiago.acted--> | +17.3% [+9.0, +37.0]<!--fig:first.santiago.acted--> |
| **Johannesburg** | 12<!--fig:joburg.retrains--> | 3<!--fig:joburg.promotions--> | +0.0% [−0.0, +14.2]<!--fig:joburg.replay--> | **+9.7% [+4.6, +14.0]**<!--fig:joburg.acted--> | +14.9% [+8.7, +21.4]<!--fig:first.joburg.acted--> |
| **Melbourne** | 9<!--fig:melbourne.retrains--> | 4<!--fig:melbourne.promotions--> | +0.9% [−2.8, +9.2]<!--fig:melbourne.replay--> | **+5.6% [+2.0, +6.6]**<!--fig:melbourne.acted--> | +1.2% [−0.6, +3.0]<!--fig:first.melbourne.acted--> |
| **Kraków** | 21<!--fig:krakow.retrains--> | 7<!--fig:krakow.promotions--> | −0.2% [−57.0, +35.0]<!--fig:krakow.replay--> | +5.7% [−14.8, +26.8]<!--fig:krakow.acted--> | +6.5% [−15.3, +27.5]<!--fig:first.krakow.acted--> |
| **Los Angeles** | 7<!--fig:la.retrains--> | 5<!--fig:la.promotions--> | −3.0% [−27.0, +6.2]<!--fig:la.replay--> | −3.5% [−25.6, +3.0]<!--fig:la.acted--> | −13.4% [−36.9, −3.8]<!--fig:first.la.acted--> |

Bold where the interval excludes zero. Four cities show a gain, Kraków and Los
Angeles show nothing measurable, and no city shows a harm. The replay reproduces
the `fixed` arm of the [side-by-side run](#the-loop-as-it-ships-now-against-the-loop-as-it-first-shipped)
to the decimal, which is also where the two configurations are compared in full
and where Melbourne's move from nothing to a small gain is read with caution.

The promotion exam, re-scored under the same loop: across
28<!--fig:gate.short.n--> promotions that served under twenty weeks it promised
+12.7% [+10.2, +15.5]<!--fig:gate.short.exam--> and delivered
+8.0% [+5.1, +11.0]<!--fig:gate.short.delivered-->, and 2<!--fig:gate.short.harmful--> of them left
their city worse off. Under the first loop it was 25<!--fig:first.gate.short.n-->
promotions and none harmful, so "no short-serving promotion ever hurt" is a
sentence this page can no longer write. The 2<!--fig:gate.long.n--> models that served
twenty weeks or more both ended up worse than the model they replaced.

## The five fixes and the model ablation, on the loop as it ships

Run on 2026-11-10: each sweep script at its default `--loop shipped`,
`scripts/ablate_model.py`, and `scripts/score_fixes.py` to read them all under one rule. Ten
predictions, that rule, and what each outcome would decide were written before any replay,
in [the registration](registrations/2026-11-10-fixes-on-the-loop-as-shipped.md). The file was
not committed before the runs, so its own date is the only evidence of that order.

The question is about the method more than about the cities. The first run concluded that no
rule about when the loop retrains, or how hard it judges, can repair it, and that one such
rule does harm. It reached that on a loop whose model sat at a default penalty. A badly set
model leaves more for any retrain to recover, so every fix that changes which retrains happen
was being measured against an inflated stake.

The instrument held. In all five sweeps the arm that is the shipped loop reproduces the six
published premiums to one decimal, 30 checks of 30, and the ablation's shipped arm matches
them to two.

### Which verdicts repeated

The rule was fixed before the run: a setting counts for or against a fix only where its
interval clears zero, in the same direction, in two cities. Both runs are read under it
here, so the first run is held to the stricter standard as well.

| the fix | on the loop as first shipped | on the loop as it ships |
|---|---|---|
| second retrain alarm | harmful one notch above the cautious setting (Kraków, Delhi), a gain at the most aggressive (Los Angeles, Santiago), one city at the cautious one | no setting reaches two cities |
| longer exam | nothing: 0 of 18 intervals clear zero | nothing: 0 of 18 |
| re-certification | nothing: 3 of 11 clear zero, each in a single city | nothing: 1 of 12 |
| confidence-aware gate | harmful at 0.90 and 0.95 (Kraków, Delhi) | nothing: Santiago alone |
| rollback | nothing: 1 of 7 | nothing: 0 of 5 |

Across the five sweeps the first loop produced 55 intervals, 15 of them clear of zero. The
loop as it ships produced 59, and 6 are.

**No fix improves accuracy on the loop that ships.** Four mechanisms that change when an exam
is sat or how it is judged, and a fifth that adds an alarm, were replayed on two differently
set loops. On this one no setting of any of them meets the rule on the positive side. On the
first loop one did, the second alarm at its most aggressive setting, and that did not repeat:
Santiago's share of it now reads +3.0% [−0.0, +6.5]<!--fig:floor.santiago.+0.00.acted-->.

**The harm did not repeat.** On the first loop the confidence-aware gate at 0.95 cost Kraków
−2.8% [−3.8, −1.8]<!--fig:first.confidence.krakow.0.95.acted--> and Delhi
−3.9% [−6.0, −0.7]<!--fig:first.confidence.delhi.0.95.acted-->. On the loop as it ships those
read −2.1% [−4.0, +3.9]<!--fig:confidence.krakow.0.95.acted--> and
+0.4% [−0.9, +1.5]<!--fig:confidence.delhi.0.95.acted-->, and the only city clear of zero is
Santiago, at −2.5% [−3.4, −1.0]<!--fig:confidence.santiago.0.95.acted-->.

The first run explained that harm as a property of any procedure that retries: a harder exam
buys more attempts, and whichever challenger finally passes did so on a rarer piece of luck.
Its evidence was the gap between what the exam promised and what the winner delivered, which
widened under the strict gate in four of six cities. On the loop as it ships it widens in
three and narrows in three. So the explanation remains a fair account of the first loop and
is not demonstrated on this one. What both loops do show is the over-promise itself: winners
that served under twenty weeks delivered +9.8%<!--fig:first.gate.short.delivered--> against a
promised +12.4%<!--fig:first.gate.short.exam--> on the first loop, and
+8.0%<!--fig:gate.short.delivered--> against +12.7%<!--fig:gate.short.exam--> on this one.

**The accuracy gain the second alarm was adopted on did not repeat.** This sweep switches
the alarm on by itself, on a loop that already tunes its model, which the side-by-side run
could not do. At the shipped setting it improves Los Angeles by
+6.6% [−1.1, +9.3]<!--fig:floor.la.-0.50.acted--> over the 22<!--fig:floor.la.-0.50.differing--> weeks
it changed the serving model, against +11.8% [+2.1, +17.2]<!--fig:first.floor.la.-0.50.acted--> on
the first loop. The interval now reaches below zero.

What the alarm does without any statistics is bound how long the loop can go deaf. With it
off, the tuned loop cannot fire for 35<!--fig:floor.la.off.silence--> of Los Angeles's
36<!--fig:la.weeks--> weeks and for 30<!--fig:floor.krakow.off.silence--> of Kraków's
48<!--fig:krakow.weeks-->. With it on those silences are 16<!--fig:floor.la.-0.50.silence--> and
22<!--fig:floor.krakow.-0.50.silence-->, and nothing changes in the other four cities. So the
alarm is a bound on silence, shown on both loops, and an accuracy gain shown on one. The
decision to ship it was taken on both, reopened on this result, and confirmed on the first
ground alone: it stays on as a bound on silence, with no claim about accuracy
([D1](DECISIONS.md)).

One notch more aggressive it still costs: Kraków loses
−20.1% [−47.5, −5.5]<!--fig:floor.krakow.-0.25.acted--> in the weeks the alarm acts, against
−24.0% [−39.4, −5.5]<!--fig:first.floor.krakow.-0.25.acted--> before.

### The ten predictions

| # | predicted | what happened |
|---|---|---|
| P1 | the alarm at −0.5 improves Los Angeles by itself | **failed**: the interval covers zero |
| P2 | at −0.5 it harms no city | held: one week differs in Kraków, none elsewhere but Los Angeles |
| P3 | the aggressive settings still cost a city | held: Kraków at −0.25 |
| P4 | a longer exam changes nothing measurable | held: 0 of 18 |
| P5 | re-certification trains more and buys no accuracy | held: one city clears zero, Los Angeles at 35 days, +17.0% [+8.3, +21.9]<!--fig:recertify.la.35d.acted--> over 9 weeks |
| P6 | the confidence-aware gate is still harmful | held as worded, **not under the two-city rule** |
| P7 | the stricter gate still widens the promised-to-delivered gap | not failed, and not supported: wider in three cities, narrower in three |
| P8 | rollback does no harm and shows no gain | held: 0 of 5 |
| P9 | Delhi's premium survives a tuned tree, below the Ridge's | held |
| P10 | Los Angeles shows no measurable effect under either tuned model | held |

P6 was written as "at least two arm-and-city pairs clear zero as a harm, and none as a gain".
Santiago at 0.90 and Santiago at 0.95 satisfy it. Those two settings blocked the same
promotions and produced the same replay, so they are one result counted twice, and the
prediction's wording was too weak to mean what it was meant to. The table above reads the
gate under the two-city rule that the same registration fixed for verdicts.

Seven predictions held outright. The three that did not are the three the registration
marked as least sure, and two of them are the two the method claim rested on.

### The model ablation

Delhi and Los Angeles, replayed under the Ridge as it ships and under two models tuned once
on the first training window with the same splitter and folds. The second alarm is on in
every arm.

| | Ridge, as shipped | Ridge, tuned once | tree, tuned once | median error, Ridge tuned once → tree |
|---|---|---|---|---|
| **Delhi** | +22.8% [+11.8, +36.4]<!--fig:ablation.delhi.ridge--> | +21.1% [+14.7, +36.7]<!--fig:ablation.delhi.ridge_tuned--> | **+13.4% [+8.0, +27.0]**<!--fig:ablation.delhi.gbm_tuned--> | 39.72<!--fig:ablation.delhi.ridge_tuned.rmse--> → 38.68<!--fig:ablation.delhi.gbm_tuned.rmse--> |
| **Los Angeles** | −3.5% [−25.6, +3.0]<!--fig:ablation.la.ridge--> | −4.6% [−26.2, +1.7]<!--fig:ablation.la.ridge_tuned--> | −2.7% [−13.2, +5.6]<!--fig:ablation.la.gbm_tuned--> | 10.57<!--fig:ablation.la.ridge_tuned.rmse--> → 10.50<!--fig:ablation.la.gbm_tuned.rmse--> |

Both validity checks pass. The shipped arm reproduces the published premiums, and the tree
is the better model in both cities, by 1.04 µg/m³ in Delhi and by 0.07 in Los Angeles. The
second margin is thin enough that the Los Angeles comparison should carry little weight.

Delhi's premium stays clear of zero under a model that can bend, and is smaller. Week for
week over the 37<!--fig:ablation.delhi.shared_weeks--> weeks both models retrained, the tree
gives up 6.8 points [5.7, 10.4]<!--fig:ablation.delhi.ridge_minus_gbm--> of premium, where on
the first loop it gave up 13.9 points [8.0, 16.7]<!--fig:first.ablation.delhi.ridge_minus_gbm-->.
So on this loop roughly a third of what retraining earns in Delhi is a straight line being
refitted to a season, and the rest survives the change of model. Los Angeles shows no
measurable effect under any arm.

### What the re-run says about measuring a loop

A default setting in the model did more than double the headline. It also enlarged the
effects measured on top of that model. On the tuned loop the gate's harm falls from two
cities to one, the alarm's gain in Los Angeles roughly halves and loses its interval, the
share of Delhi's premium that a flexible model absorbs halves, and the count of intervals
clear of zero falls from 15 of 55 to 6 of 59. Each is one comparison on six cities, so this
is a pattern noticed after the results and not a tested claim. As a working rule it is cheap:
an experiment on a control loop inherits whatever is wrong with the model inside it, so the
model is tuned before the loop is measured.

### Limits of the re-run

The same six cities carry both runs. Two runs agreeing shows a verdict survives a change of
loop, and says nothing about a seventh city. Only the second alarm has since been taken to
new cities, in the next section.

The design is small. Each city carries about 2 to 10 independent observations, the intervals
are not corrected for one another, and a percentile interval on that few units is narrower
than its label. "Nothing" in the table means not detectable at this size. The first run's
intervals were about 1 to 9 points wide on each side, and an effect smaller than that would
not have been seen in either run.

The two-city rule is new. It was fixed before this run and after the first, and it is
stricter than the case-by-case reading the first run was given.

The tree is tuned once and the shipped Ridge at every fit, which is why the ablation compares
the two tuned-once arms with each other.

## The second alarm, on cities the study had not used

Run on 2026-11-11 with `python scripts/held_out_cities.py`. The candidate cities and their
order, the rule for their replay windows, which of them are tested, and four predictions were
fixed in [the registration](registrations/2026-11-10-second-alarm-on-held-out-cities.md), and
committed as `6d8f993` before any of their data was fetched. It is the first result in this
project measured on cities it was not found on.

Twelve large cities outside the six countries were listed in a fixed order. They were taken
one at a time with the alarm off, until two turned up whose first alarm goes deaf: at least
ten quiet weeks in a row, and at least three quiet weeks with the model 50% behind the daily
profile. Nine were walked before the second such city appeared, none was dropped for missing
data, and all nine then ran with the alarm on at −0.5. Each city's first model trains on its
cleanest 75 days between July and November 2025, by rule.

| | weekly runs | longest silence, alarm off → on | weeks the alarm changed the serving model | week by week, on against off |
|---|---|---|---|---|
| Tokyo | 47<!--fig:heldout.tokyo.runs--> | 30<!--fig:heldout.tokyo.off.silence--> → 25<!--fig:heldout.tokyo.on.silence--> | 4<!--fig:heldout.tokyo.changed--> | +7.3% [+6.5, +10.9]<!--fig:heldout.tokyo.acted--> |
| Shanghai | 47<!--fig:heldout.shanghai.runs--> | 25<!--fig:heldout.shanghai.off.silence--> → 23<!--fig:heldout.shanghai.on.silence--> | 1<!--fig:heldout.shanghai.changed--> | one week, no interval |
| São Paulo | 31<!--fig:heldout.sao_paulo.runs--> | 14<!--fig:heldout.sao_paulo.off.silence--> → 14<!--fig:heldout.sao_paulo.on.silence--> | 0<!--fig:heldout.sao_paulo.changed--> | identical |
| Mexico City | 31<!--fig:heldout.mexico_city.runs--> | 25<!--fig:heldout.mexico_city.off.silence--> → 25<!--fig:heldout.mexico_city.on.silence--> | 0<!--fig:heldout.mexico_city.changed--> | identical |
| **Cairo** | 34<!--fig:heldout.cairo.runs--> | 19<!--fig:heldout.cairo.off.silence--> → 13<!--fig:heldout.cairo.on.silence--> | 4<!--fig:heldout.cairo.changed--> | −4.5% [−7.2, +1.0]<!--fig:heldout.cairo.acted--> |
| Dhaka | 43<!--fig:heldout.dhaka.runs--> | 32<!--fig:heldout.dhaka.off.silence--> → 18<!--fig:heldout.dhaka.on.silence--> | 28<!--fig:heldout.dhaka.changed--> | −6.0% [−19.0, +9.2]<!--fig:heldout.dhaka.acted--> |
| Karachi | 38<!--fig:heldout.karachi.runs--> | 34<!--fig:heldout.karachi.off.silence--> → 34<!--fig:heldout.karachi.on.silence--> | 0<!--fig:heldout.karachi.changed--> | identical |
| Buenos Aires | 35<!--fig:heldout.buenos_aires.runs--> | 21<!--fig:heldout.buenos_aires.off.silence--> → 21<!--fig:heldout.buenos_aires.on.silence--> | 0<!--fig:heldout.buenos_aires.changed--> | identical |
| **Istanbul** | 44<!--fig:heldout.istanbul.runs--> | 22<!--fig:heldout.istanbul.off.silence--> → 15<!--fig:heldout.istanbul.on.silence--> | 6<!--fig:heldout.istanbul.changed--> | +8.8% [−1.4, +11.1]<!--fig:heldout.istanbul.acted--> |

Bold marks the two cities with a deaf first alarm, which carry the predictions about a gain.
The instrument check passed first: Los Angeles comes out deaf and Santiago does not, and Los
Angeles's alarm-on arm reproduces its published premium.

| # | predicted | what happened |
|---|---|---|
| H1 | the alarm harms no city walked | held: no interval clears zero on the negative side |
| H2 | it shortens the longest silence in each city with a deaf first alarm | held: 19 → 13 weeks in Cairo, 22 → 15 in Istanbul |
| H3 | the gain does not reach the standard for a claim | held: neither of the two clears zero |
| H4 | the direction is positive in both | **failed**: Cairo's is negative |

**The bound on silence carries over.** In the five cities where the alarm fired at all, the
longest silence is shorter in every one, and in none of the nine is it longer. That is the one
thing the pages claim for the alarm, and it now holds on nine cities it was never tuned on.

**The accuracy gain does not.** The alarm's effect on accuracy has no consistent sign. Three
point estimates are positive and two are negative, and the only city where it changed many
weeks is one of the negative ones: Dhaka, over 28<!--fig:heldout.dhaka.changed--> weeks, at
−6.0% [−19.0, +9.2]<!--fig:heldout.dhaka.acted-->, with its median error over the replay rising from
24.52<!--fig:heldout.dhaka.off.rmse--> to 26.45<!--fig:heldout.dhaka.on.rmse--> µg/m³. That interval covers zero, so it is not a
harm shown. It is also not nothing: the registration set a city harmed at −0.5 as the
evidence that would switch the alarm off, and the largest sample here leans that way. Under
the outcome table written beforehand, H4 failing means the gain once seen in Los Angeles is
described from now on as an observation on the first loop, and not as something the alarm
does.

Tokyo clears zero on the positive side. It is not one of the two cities the gain was
registered on, it rests on 4<!--fig:heldout.tokyo.changed--> changed weeks where the registration asked for five
before reading an interval, and it is one of five intervals read without correction. It is
reported and not counted.

**A long silence is not always a fault.** Karachi's first alarm is quiet for
34<!--fig:heldout.karachi.off.silence--> of 38<!--fig:heldout.karachi.runs--> weeks and Mexico City's for 25<!--fig:heldout.mexico_city.off.silence--> of
31<!--fig:heldout.mexico_city.runs-->, and in neither did the model fall 50% behind the daily profile in a quiet
week, so the second alarm never fired. The six-city write-up below reads every long silence
as an alarm gone deaf. On these two the model was simply never bad enough, by the second
alarm's own yardstick, to need replacing.

**The rule that picked the two test cities guessed badly.** It counted, with the alarm off,
the quiet weeks in which the alarm would have fired. Dhaka had one such week and the alarm
went on to change 28. Cairo had six and it changed four. One early promotion sends a replay
down a different path, so the number of weeks an alarm is eligible to fire does not predict
how far the two arms end up apart. The rule could not leak the outcome, since it reads only
the alarm-off arm. It did put the gain test on two cities with 4 and 6 changed weeks and left
out the one city that could have carried it.

### Limits of the held-out test

Twelve candidates listed from memory are not a sample of the world's cities. They are cities
nobody had looked at when the rule was written.

Each city holds about 2 to 10 independent observations and five intervals are read without
correction, so a real effect of a few percent in either direction would usually be missed. The
accuracy result is "not shown, in either direction".

The replay windows were set by a rule and the six cities' were picked by hand, so the new
cities are not replayed exactly as the six were. A window chosen as the cleanest stretch
between July and November also suits the northern cities better than São Paulo or Buenos
Aires.

## What retraining is worth on the same nine cities

Run on 2026-11-11 with `python scripts/held_out_premium.py`. The question, how each city is
read, the predictor and two predictions were fixed in
[the registration](registrations/2026-11-11-premium-on-held-out-cities.md), committed as
`b69c755` before any interval or swing was computed. The point estimates had already been
seen, because the alarm test above stored them, and the registration says so.

Each city replays the loop as it ships on the windows the alarm test used, and is compared week
by week with its first model, never retrained, as in the six-city table. Retraining **pays** in a
city when that interval is above zero. The predictor is the city's **seasonal swing**: its
highest monthly mean PM2.5 divided by its lowest, over May 2024 to April 2025, a year that ends
before any city trains or is scored. A city was predicted to pay if its swing was at least that
of the least seasonal of the four cities among the six where retraining pays. Before any held-out
interval was read, Delhi and Los Angeles reproduced their published intervals through the same
code, and each of the nine reproduced its stored point estimate.

| | swing, 2024-25 | weeks served by a retrained model | week by week, against the first model |
|---|---|---|---|
| Dhaka | 6.9<!--fig:heldout.dhaka.swing-->× | 41<!--fig:heldout.dhaka.premium.n--> | **+25.6% [+4.6, +47.9]**<!--fig:heldout.dhaka.premium--> |
| Istanbul | 4.0<!--fig:heldout.istanbul.swing-->× | 39<!--fig:heldout.istanbul.premium.n--> | +21.4% [−7.9, +26.6]<!--fig:heldout.istanbul.premium--> |
| Shanghai | 3.2<!--fig:heldout.shanghai.swing-->× | 46<!--fig:heldout.shanghai.premium.n--> | −2.3% [−11.7, +5.6]<!--fig:heldout.shanghai.premium--> |
| São Paulo | 2.4<!--fig:heldout.sao_paulo.swing-->× | 8<!--fig:heldout.sao_paulo.premium.n--> | −0.1% [−3.5, +2.2]<!--fig:heldout.sao_paulo.premium--> |
| Karachi | 2.4<!--fig:heldout.karachi.swing-->× | 37<!--fig:heldout.karachi.premium.n--> | **+22.1% [+8.1, +34.2]**<!--fig:heldout.karachi.premium--> |
| Tokyo | 2.1<!--fig:heldout.tokyo.swing-->× | 46<!--fig:heldout.tokyo.premium.n--> | **+3.2% [+1.4, +10.8]**<!--fig:heldout.tokyo.premium--> |
| Mexico City | 2.1<!--fig:heldout.mexico_city.swing-->× | 30<!--fig:heldout.mexico_city.premium.n--> | −3.9% [−12.2, +1.9]<!--fig:heldout.mexico_city.premium--> |
| Buenos Aires | 2.0<!--fig:heldout.buenos_aires.swing-->× | 34<!--fig:heldout.buenos_aires.premium.n--> | −1.6% [−10.8, +1.5]<!--fig:heldout.buenos_aires.premium--> |
| Cairo | 2.0<!--fig:heldout.cairo.swing-->× | 31<!--fig:heldout.cairo.premium.n--> | +6.4% [−49.6, +16.2]<!--fig:heldout.cairo.premium--> |

Bold where the interval excludes zero.

| # | predicted | what happened |
|---|---|---|
| P1 | the swing sorts the nine into the cities where retraining pays and the rest | **failed**: 3<!--fig:heldout.p1.correct--> of 9 sorted right |
| P2 | retraining harms no held-out city | held: no interval is below zero |

**Retraining paid in 3<!--fig:heldout.p1.pays--> of the nine and harmed none, and the swing did not
say which in advance.** Under the outcome table written beforehand, "retraining pays where the
season is strong" is now a finding about the six cities it was found on, and it did not carry over
to new ones. Seeing the point estimates first could only have helped the prediction pass, so the
failure stands as scored.

**The threshold failed before it reached the nine.** Melbourne pays, and its swing over 2024-25
was 1.4<!--fig:swing.melbourne-->×, so every one of the nine sat above the line and the prediction was
"all nine pay". A rule that predicts the same thing everywhere cannot sort: labels handed out at
random would have done as well every time (chance 1.0<!--fig:heldout.p1.chance-->). Across the nine, the
swing and the point premium rank together only loosely, at a correlation of
+0.42<!--fig:heldout.spearman-->, which is reported and not scored.

**What this says about measuring the loop.** Two parts of the design did not hold. A threshold set
by the weakest of four cities is set by one city, here the one whose gain this page already reads
with caution. And a city's swing in one year does not fix it in the next: Melbourne's air swings
about 3× over its replay and 1.4× the year before. The season may still be what matters; this test
cannot show it, and a repeat should measure the predictor on a year it can trust and score it
without a cut-off.

The limits are those of the alarm test above, plus one: the predictor was chosen by someone who
had seen the point estimates. Each city carries about 2 to 11 independent observations, so a city
where retraining is worth a few percent will usually read as nothing measurable.

## The analysis of the first loop

The full analysis of the loop as it ran until 2026-11-10 is in
[evaluation-first-loop.md](evaluation-first-loop.md): each city in detail, the baselines,
why the retrain alarm goes deaf, how well the promotion exam predicts, the five fixes as
first measured, and the first model ablation. It is kept, with every figure still checked
against that loop's outputs, because it is the reasoning that led to the change. Where it
and this page differ, this page describes the loop as it ships.

## The loop as it ships now, against the loop as it first shipped

Run 2026-09-22 with `python scripts/rebaseline.py`, and it is the next step the
section above keeps asking for, taken through the loop rather than beside it. It
was written to sit next to the published figures and replace none of them, which
is the habit [HISTORY.md](HISTORY.md) exists to keep. On 2026-11-10 its `fixed`
arm became the configuration every profile ships, and the text below is left as
it was written before that decision: "shipped" in it means the loop as first
shipped.

Two arms per city, identical but for the two decisions this project has measured
and not taken: the Ridge penalty chosen at every fit by forward-chaining CV over
the window being fitted (D5), and the model-independent skill floor at −0.5 (D1).

**The shipped arm reproduces all six published premiums to the decimal**, so the
harness is measuring what this page measured.

| | premium, first shipped | premium, as it ships now | median served RMSE | fires | longest silence |
|---|---|---|---|---|---|
| **Delhi** | +49.4% [+33.7, +61.8]<!--fig:rebaseline.delhi.shipped--> | **+22.8% [+11.8, +36.4]**<!--fig:rebaseline.delhi.fixed--> | 40.44<!--fig:rebaseline.delhi.shipped.rmse--> → 40.11<!--fig:rebaseline.delhi.fixed.rmse--> | 9<!--fig:rebaseline.delhi.shipped.fires--> → 8<!--fig:rebaseline.delhi.fixed.fires--> of 39 | 21w<!--fig:rebaseline.delhi.shipped.silence--> → 21w<!--fig:rebaseline.delhi.fixed.silence--> |
| **Santiago** | +17.3% [+9.0, +37.0]<!--fig:rebaseline.santiago.shipped--> | +17.5% [+8.6, +37.1]<!--fig:rebaseline.santiago.fixed--> | 24.54<!--fig:rebaseline.santiago.shipped.rmse--> → 24.55<!--fig:rebaseline.santiago.fixed.rmse--> | 13<!--fig:rebaseline.santiago.shipped.fires--> → 13<!--fig:rebaseline.santiago.fixed.fires--> of 21 | 3w<!--fig:rebaseline.santiago.shipped.silence--> → 3w<!--fig:rebaseline.santiago.fixed.silence--> |
| **Johannesburg** | +14.9% [+8.7, +21.4]<!--fig:rebaseline.joburg.shipped--> | **+9.7% [+4.6, +14.0]**<!--fig:rebaseline.joburg.fixed--> | 21.13<!--fig:rebaseline.joburg.shipped.rmse--> → 21.72<!--fig:rebaseline.joburg.fixed.rmse--> | 11<!--fig:rebaseline.joburg.shipped.fires--> → 12<!--fig:rebaseline.joburg.fixed.fires--> of 19 | 3w<!--fig:rebaseline.joburg.shipped.silence--> → 3w<!--fig:rebaseline.joburg.fixed.silence--> |
| **Kraków** | +6.5% [−15.3, +27.5]<!--fig:rebaseline.krakow.shipped--> | +5.7% [−14.8, +26.8]<!--fig:rebaseline.krakow.fixed--> | 17.49<!--fig:rebaseline.krakow.shipped.rmse--> → 17.49<!--fig:rebaseline.krakow.fixed.rmse--> | 14<!--fig:rebaseline.krakow.shipped.fires--> → 21<!--fig:rebaseline.krakow.fixed.fires--> of 48 | **30w<!--fig:rebaseline.krakow.shipped.silence--> → 22w<!--fig:rebaseline.krakow.fixed.silence-->** |
| **Melbourne** | +1.2% [−0.6, +3.0]<!--fig:rebaseline.melbourne.shipped--> | **+5.6% [+2.0, +6.6]**<!--fig:rebaseline.melbourne.fixed--> | 3.97<!--fig:rebaseline.melbourne.shipped.rmse--> → 4.01<!--fig:rebaseline.melbourne.fixed.rmse--> | 8<!--fig:rebaseline.melbourne.shipped.fires--> → 9<!--fig:rebaseline.melbourne.fixed.fires--> of 30 | 15w<!--fig:rebaseline.melbourne.shipped.silence--> → 15w<!--fig:rebaseline.melbourne.fixed.silence--> |
| **Los Angeles** | −13.4% [−36.9, −3.8]<!--fig:rebaseline.la.shipped--> | **−3.5% [−25.6, +3.0]**<!--fig:rebaseline.la.fixed--> | 10.91<!--fig:rebaseline.la.shipped.rmse--> → 10.21<!--fig:rebaseline.la.fixed.rmse--> | **1<!--fig:rebaseline.la.shipped.fires--> → 7<!--fig:rebaseline.la.fixed.fires--> of 36** | **35w<!--fig:rebaseline.la.shipped.silence--> → 16w<!--fig:rebaseline.la.fixed.silence-->** |

Bold where an interval moved across zero, or where the loop's behaviour changed.

**The premium falls where the model was doing the work.** Delhi's more than
halves and stays clear of zero, which is the ablation's finding arriving through
the loop: a champion that is no longer under-regularised leaves less for a
retrain to recover. Johannesburg's falls by a third the same way.

**Los Angeles stops being a harm.** −3.5% [−25.6, +3.0]<!--fig:rebaseline.la.fixed--> covers zero. The cost
half of the headline does not survive here either, which is three separate
measurements now saying so.

**The loop acts where it had stopped acting.** Los Angeles fires 7 times in 36
weeks instead of once, and its longest silence falls from 35 weeks to 16;
Kraków's from 30 to 22. That is the skill floor doing what its own sweep said,
and it is the difference between a loop that adjusts and one that has gone deaf.

**Melbourne went the other way, and the prediction written before the run said it
would not.** It moves from +1.2% [−0.6, +3.0]<!--fig:rebaseline.melbourne.shipped--> to +5.6% [+2.0, +6.6]<!--fig:rebaseline.melbourne.fixed-->: a null
result becoming a positive one. Read it as a caution rather than a win. The
premium compares an arm against *its own* frozen champion, so a tuned control
that generalises worse would raise the premium without the loop improving at all,
and Melbourne's median served error does get slightly worse (3.97 → 4.01).
Johannesburg's worsens too (21.13 → 21.72). Tuning per fit is not free, and a
premium is not an accuracy.

Scored against the predictions written before the first replay: Delhi falls
(yes), Los Angeles covers zero (yes), Kraków and Melbourne stay null (**no** —
Melbourne moved), the quiet cities fire more and go silent for less (yes, in Los
Angeles and Kraków), median error improves in most cities (**no** — two improve,
two are flat, two are slightly worse).

### Can the ratcheting trigger just be deleted?

The `fixed` arm leaves the ratio trigger in place and adds a second trigger
beside it, which compensates for the fault rather than removing it. The cheaper
fix would be to delete the ratchet: one trigger, model-independent, nothing to
reset upward. A third arm asks that directly — tuning and the floor as before,
`perf_drift_threshold` set to infinity so the ratio can never fire. Added after
the first run, so it is exploratory rather than predicted.

**It does not work, and it is not close.** Without the ratio trigger the loop
mostly stops acting:

| | fires, both triggers | fires, floor only | longest silence | median served RMSE |
|---|---|---|---|---|
| Johannesburg | 12 of 19 | **0 of 19** | 3w → 19w | 21.72 → 21.72 |
| Melbourne | 9 of 30 | **0 of 30** | 15w → 30w | 4.01 → 4.04 |
| Santiago | 13 of 21 | 2 of 21 | 3w → 11w | 24.55 → 24.55 |
| Delhi | 8 of 39 | 3 of 39 | 21w → 21w | 40.11 → **44.67** |
| Los Angeles | 7 of 36 | 2 of 36 | 16w → 34w | 10.21 → 9.91 |
| Kraków | 21 of 48 | 11 of 48 | 22w → 15w | 17.49 → 18.25 |

Two cities never retrain at all, and Delhi — where retraining genuinely pays —
loses five of its eight retrains and 4.6 µg/m³ of accuracy. The floor asks
whether the champion is 50% worse than a daily profile, which is a backstop for a
model that has become bad in absolute terms, not a detector of the ordinary
decay the ratio catches. The two triggers answer different questions, and the
ratchet is the price of the cheap one.

So the loop keeps both, and the ratchet stays a known fault with a backstop
rather than a fault that was removed. Read the floor-only premiums with care:
with two acted windows or none, the bootstrap returns intervals like
[−102.3, +28.6]<!--fig:rebaseline.krakow.floor_only--> and, in Los Angeles, [−inf, +inf]. That is the resampling saying
it has nothing to work with, and it is the reason those rows are read as
behaviour rather than as premiums.

**What this does not settle.** D1 and D5 are decisions, and this is evidence for
them rather than a substitute for taking them. Both were taken on 2026-11-10. One replay per city, six cities
chosen for contrast, no multiple-comparisons correction, and the two arms do not
share a control. The tuning grid had to be widened before any of it could be
believed: a 180-day challenger window optimises at alpha 3000, past
`ALPHA_GRID`'s top of 1000, so the loop tunes over `LOOP_ALPHA_GRID` instead and
the published grid is left exactly as it was.

## Limitations

### Of the evidence

- **Two of the six cities show nothing measurable.** Kraków reads
  +5.7% [−14.8, +26.8]<!--fig:krakow.acted--> and Los Angeles −3.5% [−25.6, +3.0]<!--fig:la.acted-->.
- **Six cities, chosen for contrast and not sampled.** Only the second alarm has been tested
  outside them. The premiums, the exam's calibration and the other four fixes have not.
- **The design is small.** A city's replay holds about 2 to 10 independent observations, and
  no interval is corrected for the others read beside it. "Nothing measurable" means not
  detectable at this size.
- **The exam's long-serving group is 2<!--fig:gate.long.n--> promotions.** That gives a direction
  and no size, and the twenty-week split was chosen after seeing where the evidence turned.
- **Each premium is measured against its own configuration's first model.** A first model
  that generalises worse raises the premium without the loop serving better forecasts, which
  is why Melbourne's rise is read with care.
- **The model-class check rests on two cities**, and in Los Angeles the tree is the better
  model by only 0.07 µg/m³.
- **The pooled baseline is fitted at the default penalty** while each city's own model is
  tuned at every fit, which tilts "one model or six" a little against the pooled model.
- **The retrain-alarm and exam analyses in
  [evaluation-first-loop.md](evaluation-first-loop.md) were not redone** on the loop as it
  ships. The sweeps and the ablation were.
- **The target is a model's estimate, not a measurement.** PM2.5 here is CAMS output, so a
  change in how CAMS is produced would read as drift.

### Of the loop

- **The first retrain alarm still ratchets.** The second is a backstop and not a repair: the
  longest silences fall from 30<!--fig:rebaseline.krakow.shipped.silence--> weeks to
  22<!--fig:rebaseline.krakow.fixed.silence--> in Kraków and from 35<!--fig:rebaseline.la.shipped.silence--> to
  16<!--fig:rebaseline.la.fixed.silence--> in Los Angeles, and deleting the first alarm was tried and is worse.
- **The second alarm is kept for the silences it shortens, not for accuracy.** On nine cities
  outside the six its effect on accuracy had no consistent sign, and the city with the most
  changed weeks, Dhaka, leans negative at −6.0% [−19.0, +9.2]<!--fig:heldout.dhaka.acted-->.
- **The model is weak.** R² on the most-drifted window is near zero or negative in several
  cities. An hour's PM2.5 from a week-old weather forecast is hard, and the loop is the
  subject here, not the model.
- **Inputs it lacks.** Boundary layer height, which sets the volume pollution is diluted
  into, is not archived at a seven-day lead. There are no lagged PM2.5 terms, by choice: they
  would help in the quiet cities and would also absorb the drift the loop exists to catch.
- **Settings argued and not swept.** The 180-day training window, and the single seven-day
  lead.
- **It is a zero-infrastructure demo.** State lives in git, artifact paths are absolute, and
  serving is one container per city with `POST /reload` called by hand. A production loop
  would use a hosted tracking server and have the loop call the reload.
