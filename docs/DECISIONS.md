# Decisions

Choices that changed what ships, with the evidence that drove them and the date.
Findings live in [evaluation.md](evaluation.md); this is for the calls made on
top of them, which are a different kind of thing and were previously mixed in.

---

## D1. The skill floor ships switched on, at −0.5

**Date:** 2026-08-11, decided 2026-11-10
**Status:** decided: on, at `skill < −0.5`, as a bound on how long the loop can go
deaf. No claim is made that it improves accuracy. It was reopened and confirmed on
the day it was decided, and the reasoning below is kept as it was written.
**Affects:** `LoopConfig.skill_floor`, set to −0.5 by `config.SHIPPED` in all six
city profiles and the live schedule. A bare `LoopConfig` still has it off.

### Decided 2026-11-10

**Switched on, by the project's owner, together with D5.** The entry below ends
by saying that flipping a shipped default belongs to whoever owns the project.
This is that call.

What it was taken on. The sweep: at −0.5 five cities are unchanged to the last
digit and Los Angeles improves by +11.8% [+2.1, +17.2]<!--fig:first.floor.la.-0.50.acted--> over the weeks
the floor acted. The side-by-side replay: with the floor on and the penalty
tuned, the longest stretch in which the loop could not fire falls from
35<!--fig:rebaseline.la.shipped.silence--> weeks to 16<!--fig:rebaseline.la.fixed.silence--> in Los Angeles and from
30<!--fig:rebaseline.krakow.shipped.silence--> to 22<!--fig:rebaseline.krakow.fixed.silence--> in Kraków.

What it was not taken on. "What would settle it", below, asks for a seventh and
an eighth city chosen before the floor is swept on them. They were never run. The
floor ships on six cities picked for contrast, one helped and five untouched, and
that gap is listed in [evaluation.md](evaluation.md#limitations). A city harmed at
−0.5 is the evidence that would reverse this.

### Reopened 2026-11-10, by the re-run on the loop as it ships

The decision above was taken on two things, and one of them did not hold. The
re-run switched the alarm on by itself, on a loop that already tunes its model,
with the prediction written first that it would still improve Los Angeles. It
reads +6.6% [−1.1, +9.3]<!--fig:floor.la.-0.50.acted--> over the 22<!--fig:floor.la.-0.50.differing--> weeks it changed the
serving model. The interval covers zero, and the prediction failed.

The other thing held. With the alarm off, the tuned loop cannot fire for
35<!--fig:floor.la.off.silence--> of Los Angeles's 36<!--fig:la.weeks--> weeks or for 30<!--fig:floor.krakow.off.silence--> of Kraków's 48<!--fig:krakow.weeks-->;
with it on, the longest silences are 16<!--fig:floor.la.-0.50.silence--> and 22<!--fig:floor.krakow.-0.50.silence-->. It changes the serving model
in no other city, bar one week in Kraków.

So the alarm is a bound on how long the loop can go deaf, and it is not shown to
improve accuracy. Keeping it on is defensible as the first and not as the second.
The setting was not changed by the analysis that found this, because whether a
staleness bound alone is worth a shipped default is the owner's call. The seventh
and eighth city would still settle the accuracy question either way.

### Confirmed 2026-11-10, on the narrower ground

**It stays on, by the project's owner, as a bound on silence and nothing more.**
The pages say what it is kept for and make no claim that it improves the
forecasts. It costs nothing measurable in five cities, and the silences it
prevents are the same on both loops.

What would change this. A city it harms at −0.5 would switch it off. A seventh and
an eighth city, chosen before the alarm is swept on them, are what could turn the
Los Angeles gain into a claim about accuracy or retire it.

### The choice

The retrain trigger ratchets: it compares a model against its own error at
training time, every promotion resets that comparison, and promotions happen at
the seasonal peak, so the bar rises and never falls. `LoopConfig.skill_floor` is
the fix. It fires a retrain when skill against a 30-day daily profile drops below
a floor, which nothing about promoting a model can move.

It has shipped at `None` since it was built, on the grounds that replaying six
cities with it on changed nothing worth having.

### What changed

Adding intervals, and then fixing the statistic that produced them. Details in
[evaluation.md](evaluation.md#what-the-medians-were-hiding-in-both-directions);
the short version is that comparing arms by their median error hid effects in
both directions, because most weeks a changed trigger leaves the serving model
alone and those ties dominate any median of paired ratios.

At the cautious floor, `skill < −0.5`:

- Five cities are **bit-identical** to the trigger left alone. Not close, the
  same error every week.
- Los Angeles acts in 25<!--fig:first.floor.la.-0.50.differing--> of 35<!--fig:first.la.acted.n--> weeks and improves by **+11.78% [+2.09, +17.17]**<!--fig:first.floor.la.-0.50.acted-->.

Los Angeles is the city the ratchet leaves deafest, silent for 35 of 36 runs, and
the one where retraining otherwise measurably hurts.

### The argument for switching it on

A floor that cannot harm any city in the set and measurably helps the one the
ratchet disables is, on this evidence, free. The earlier reading called the same
observation "a closer call than it sounds"; with an interval attached it is no
longer a close call in the direction it was assumed to fall.

### The argument for leaving it off

Three things, none decisive alone.

The whole-replay figure for Los Angeles is **+2.49% [+0.00, +15.01]**<!--fig:first.floor.la.-0.50.all-->, which
touches zero. The acted-week effect is established; the effect on the replay as a
whole is not, because ten of the 35 weeks are ties and they pull the median in.

"Cannot harm any city" is a statement about six cities picked for contrast, not
about cities in general. A seventh could behave like Kraków does at
`skill < −0.25`, where the floor costs **−23.98% [−39.40, −5.48]**<!--fig:first.floor.krakow.-0.25.acted--> in the weeks
it acts. The gap between a floor that is inert and a floor that is destructive is
one setting wide.

Every published number in this repository was produced with the floor off.
Turning it on moves the six-city table, the gate calibration and the holdout
sweep at once, and the before-and-after comparison the project is built on would
have to be restated rather than extended.

**Added 2026-08-14.** The Los Angeles result is one arm of eighteen in that
sweep, and the page it sits on now reports more than two hundred intervals with
no multiple-comparisons correction. That does not make it wrong, and it is a
wide interval well clear of zero rather than a marginal one, but it is the kind
of claim multiplicity reaches. See
[evaluation.md](evaluation.md#what-is-still-not-corrected). It shifts this
decision's weight further toward the resolution below: two cities chosen before
the floor is swept on them settle it in a way that re-reading the same eighteen
arms cannot.

### Decision, as it stood until 2026-11-10

**No change.** The default stays `None`, the knob stays in the code, and the
evidence for turning it on is written down where the next person will find it.

Flipping a shipped default is a decision rather than a finding, and this one
belongs to whoever owns the project rather than to whoever ran the sweep. What
the analysis is entitled to say is that the case has reversed, and it says so.

### What would settle it

A seventh and eighth city, chosen before the floor is swept on them, at
`skill < −0.5` only. If the floor is inert in both and the Los Angeles result
reproduces in any city with a comparably deaf trigger, switch it on. If either
new city is harmed, the ceiling on how much a floor can be trusted is lower than
six cities suggested, and it stays off for a documented reason rather than an
inherited one.

---

## D2. Re-certification ships switched off, and this time the evidence agrees

**Date:** 2026-08-13
**Status:** settled
**Affects:** `LoopConfig.recertify_days`, currently `None`

### Re-run on the loop as it ships, 2026-11-10

No change. Replayed on the tuned loop with the second alarm on, no cadence clears
zero in two cities. One arm does in one city, Los Angeles at 35 days, at
+17.0% [+8.3, +21.9]<!--fig:recertify.la.35d.acted--> over 9 weeks. Every cadence trains more models than the
loop does without it.

### The choice

Two measured negative results pointed at the same missing mechanism. Making the
retrain trigger more sensitive does not pay (D1, and
[evaluation.md](evaluation.md#fixing-the-trigger-one-fix-is-impossible-the-other-pays-in-one-city)).
Making the promotion exam longer does not pay
([evaluation.md](evaluation.md#lengthening-the-exam-does-not-fix-it)). Both
change *when the loop notices something is wrong*, and neither addresses the fact
that nothing re-examines an incumbent at all.

`LoopConfig.recertify_days` is that third mechanism. It re-sits the holdout exam
when the serving champion's certificate expires, regardless of every drift
signal, and passing renews it.

### What the measurement says

Full detail in
[evaluation.md](evaluation.md#re-certification-bounds-staleness-and-staleness-was-not-the-thing-that-hurt).
Three findings, in the order they matter.

**The mechanism works.** It bounds how long a model may serve unexamined, in
every city, at every cadence. On the shipped settings Los Angeles served a model
for 252 days after its last exam with every drift signal quiet. No existing
trigger reports that number, let alone bounds it.

**The bound does not convert into accuracy.** Of 24 city-by-cadence comparisons,
three clear zero: Los Angeles at 14 days (+13.8%) and 28 days (+6.4%), and Delhi
at 35 days (**−10.8%**, harmful). Johannesburg and Melbourne are bit-identical at
every cadence tested, having trained roughly twice as many challengers to get
there.

**The effect is the gate's, not the schedule's.** Kraków at 28 days fires seven
extra times, the gate rejects all seven, and the replay is bit-identical. Kraków
at 35 days fires six extra times, the gate lets one through, and the city ends
6.7% worse. Re-certification does not improve the gate's calibration; it puts
more decisions through a gate already known to reverse sign past five weeks.

### The argument for switching it on

"No model serves more than N days without being re-examined" is a real property,
and it is the kind of thing an auditor asks for. It is worth separating from
accuracy: an operator may want the guarantee whatever it does to error, and this
is the only mechanism in the project that can offer it.

At 28 days it improves the one city the ratchet disables, leaves three cities
bit-identical, and moves the other two by amounts no interval distinguishes from
zero. That is the same shape as the skill floor at `−0.5`.

### The argument for leaving it off

The benefit is one city, and it is the weakest city in the set. Los Angeles has
the fewest effective observations of any city here, it is the control rather than
a result, and it is the city where retraining measurably *hurts*. An intervention
whose only measurable gain is "retrains a city that should not be retraining, and
the outcome is less bad than the one unlucky promotion it had before" is not a
mechanism that has been shown to work. It is regression to the mean with a
schedule attached.

The harm is real and larger than the benefit in absolute terms. Delhi at 35 days
loses 10.8%, and Delhi is the city where retraining pays most.

The cost is uniform where the benefit is not. Every arm trains more challengers
in every city: Melbourne 8 to 19, Johannesburg 11 to 16, Kraków 14 to 34. In four
of six cities that buys nothing at all.

And no cadence is safe across the set. 28 days is the best of the four and it is
best by luck: what separates it from 35 days is which extra challengers a
seven-day exam happened to pass.

### Decision

**No change.** The default stays `None`.

Unlike D1, the evidence and the default agree here, so this is a settled decision
rather than an open one. The knob stays in the code because the staleness bound
is worth having on its own terms, and because `certified_age_days` is now logged
on every run whether the trigger is on or not: the measurement was the valuable
part, and it survives the decision.

### What would change it

Fixing the gate first. Every argument against switching this on reduces to the
same thing: re-certification feeds challengers to an exam that cannot certify
past five weeks, so more exams means more mis-certifications. A gate that stayed
calibrated over a model's real service life would make this mechanism worth
re-measuring, and in that order. That is now the top open item, and it is the
first one in this project pointed at the gate rather than at the trigger.

---

## D3. The promotion gate keeps deciding on a point estimate

**Date:** 2026-08-14
**Status:** settled
**Affects:** `LoopConfig.promotion_confidence`, currently `None`

### Re-run on the loop as it ships, 2026-11-10

Still off, and for a weaker reason than the one below. On the first loop the gate
cost Kraków and Delhi accuracy. On the loop as it ships the only city clear of
zero is Santiago, at −2.5% [−3.4, −1.0]<!--fig:confidence.santiago.0.95.acted-->, and the gap between what the exam
promised and what the winner delivered widens in three cities and narrows in
three. The gate is not shown to harm the current loop and not shown to help it.
"What this rules out", below, was argued from the first loop and stands as an
account of that loop.

### The choice

The gate compares two RMSEs on one seven-day window of autocorrelated hourly
data and promotes on the bare difference. Every number this project publishes
carries an interval; the decision that produces those numbers did not.

That is measurably a real defect, not a stylistic one. Over the 28 shipped
promotions, 11 have an exam margin whose interval reaches below the 5% they were
required to clear and 6 reach below zero, and the width of that interval sorts
the cities by whether retraining works in them at all. Full detail in
[evaluation.md](evaluation.md#the-gate-decides-on-a-point-estimate-and-making-it-decide-on-an-interval-is-worse).

`LoopConfig.promotion_confidence` requires the challenger to clear
`promotion_margin` at the lower bound as well as at the point estimate.

### What the measurement says

**It does not work, and the failure is informative.** Of 18 city-by-confidence
arms, five have an interval clear of zero against the shipped gate and **all five
are harmful**. Kraków loses 3.1%, Delhi 3.9%. Not one arm shows a gain the data
can establish.

The winner's curse it was built to remove gets **worse** in four of six cities.
Kraków's promise-minus-delivery goes 4.4 to 5.7, Delhi's 3.1 to 5.0.

The reason is that blocking a promotion defers it rather than preventing it. The
loop retries, so a higher bar per attempt buys more attempts, and the challenger
that eventually clears a higher bar cleared it on a more extreme fluctuation.
Los Angeles ends with exactly one promotion at every confidence tested, having
gone from 1 retrain to 6 while the blocked champion sat there going stale.

### Decision

**No change.** The default stays `None`.

The knob stays in the code and `exam_margin_lo` is logged whenever it is on,
because the diagnosis it produced is sound and worth keeping even though the
treatment failed. The gate really is deciding on noise. Tightening the threshold
is simply not how that gets fixed.

### What this rules out, which is the useful part

Four mechanisms have now been built and measured: a second retrain trigger, a
longer exam, a re-certification schedule, and a confidence-aware gate. Two make
the loop notice sooner, one makes it look more often, one makes it judge harder.
**None of them pays, and this one is actively harmful.**

They fail for one reason, and it took the fourth to see it. The loop is a retry
loop. It keeps training challengers and sitting exams until one passes, so the
promoted model is always selected from a sequence of attempts and always carries
the luck of whichever attempt cleared the bar. Every knob tried so far adjusts
*when* an attempt happens or *how hard* a single attempt is to pass. Neither can
address a procedure whose bias comes from the number of attempts.

### What would change it

A rule that prices the attempts rather than the attempt. Two candidates, neither
built: carry the count of exams a champion has faced into the decision, so a bar
that has been assaulted twelve times is harder to clear than one facing its
first challenger; or judge a promoted model on a window *after* promotion and
roll it back when it fails, which stops the loop treating promotion as
irreversible and makes the selection self-correcting.

The second is the more interesting because the project has no rollback at all,
and because a decision that can be undone does not need to be right first time.

---

## D4. Probation ships off, on "unproven" rather than "harmful"

**Date:** 2026-08-14
**Status:** settled
**Affects:** `LoopConfig.probation_days`, currently `None`

### Re-run on the loop as it ships, 2026-11-10

No change. None of the 5 intervals the sweep produced clears zero in either
direction, and 7 of the 55 promotions it judged were rolled back.

### The choice

D3 established that the loop's promotions are selected by a retry procedure and
that no per-attempt rule can price the number of attempts. `probation_days` is
the response: leave the gate alone and make its output reversible. A promotion
is provisional, and after the configured window the new champion is scored
against the model it displaced on a window that postdates them both.

It is a check, not a selection, which is why it was worth trying after four
sharper gates failed. There is no best-of-many here to inherit a bias from.

### What the measurement says

Full detail in
[evaluation.md](evaluation.md#making-the-promotion-reversible-and-why-that-does-not-rescue-it-either).

**It does no harm.** Of 18 city-by-window arms, one has an interval clear of
zero and it is positive (Kraków +1.5% [+0.8, +2.0]<!--fig:first.probation.krakow.21d.acted--> at 21 days). The confidence
gate produced five harmful arms and no positive ones. On that comparison alone
this is the best-behaved mechanism of the five.

**It is not a result.** One arm in eighteen clearing a 95% interval is the rate
eighteen tests produce by chance, and this project does not correct for multiple
comparisons. Kraków's gain rests on a single rollback affecting seven weeks.

**The diagnostic is the valuable part.** The mean probation verdict is positive
in 17 of 18 arms and only 8 of 49 judged promotions were rolled back.
Promotions are, at two to four weeks out, mostly correct. That agrees with the
gate calibration: the exam is good for about five weeks, and a probation window
inside that horizon finds a decision that was right at the time.

### Decision

**No change.** The default stays `None`.

Off on "unproven", which is a different verdict from D3's "harmful" and worth
distinguishing. The mechanism is sound, it costs nothing when it finds nothing,
and it is the only intervention here that could be switched on without an
argument about damage. What it lacks is evidence that it helps.

### What this rules out, and what it leaves

Probation is caught between two requirements that cannot both be met. A window
short enough to isolate the promotion decision, before the world moves again, is
too short to find anything wrong: at 14 to 28 days the promotions are fine. A
window long enough to catch the reversal the gate calibration actually shows, at
twenty weeks and beyond, cannot separate "this promotion was noise" from "the
world moved after it".

**So the failure mode is not attributable to a single decision.** Five
mechanisms have now been built against it and the fifth is the one that shows
why: the loop's long-serving models fail through the accumulation of time, not
through a mistake at any point a verdict could be attached to.

Los Angeles is the demonstration. Its one promotion, which delivered −6.7% over
36 weeks, is rolled back when judged at 14 days and kept when judged at 21 or
28. Same promotion, three windows, opposite verdicts, because a fortnight of
hourly data cannot resolve a difference that small, which is the same limit that
made the seven-day exam unreliable in the first place.

### What would change it

Not another decision rule. The five tried here exhaust the ways to adjust when
the loop acts, how hard it judges, and whether it can undo the result, and the
measurement limit underneath them is the same one throughout. What is left is to
attack the measurement: a target with more signal per unit time, a model whose
errors are less autocorrelated, or a horizon long enough that a fortnight is not
the unit of evidence. Those are properties of the problem rather than of the
loop, which is a fair conclusion for a project about a loop to reach.

---

## D5. The loop chooses its own alpha at every fit

**Date:** 2026-08-14, decided 2026-11-10
**Status:** decided: tuned per fit. Open for three months before that, and the
reasoning below is kept as it was written while it was.
**Affects:** `LoopConfig.tune_alpha`, set by `config.SHIPPED` in all six city
profiles and the live schedule. `alpha=1.0` remains what a bare `LoopConfig` fits
at, and the ablation and the five mechanism sweeps pin it (`config.FIRST_SHIPPED`)
so that each still reproduces its own published table.

### Decided 2026-11-10

**Tuned at every fit, by the project's owner, together with D1.** "What would
settle it", below, asks what the six-city table is for. The answer taken is the
second one: it is read as a measurement of what retraining is worth, so it is
measured on a model whose setting somebody chose.

What changed. Every city was replayed under the new configuration, and the
headline moved as the side-by-side run of 2026-09-22 said it would: Delhi from
+49.4% [+33.7, +61.8]<!--fig:first.delhi.acted--> to +22.8% [+11.8, +36.4]<!--fig:delhi.acted-->, Los Angeles from
−13.4% [−36.9, −3.8]<!--fig:first.la.acted--> to −3.5% [−25.6, +3.0]<!--fig:la.acted-->. The earlier figures stay on
the pages beside the new ones.

What was re-measured afterwards. The ablation and the five mechanism sweeps were re-run on
the tuned loop the same day, with predictions written first; the results are in
[evaluation.md](evaluation.md#the-five-fixes-and-the-model-ablation-on-the-loop-as-it-ships).
Under a tuned tree Delhi's premium is +13.4% [+8.0, +27.0]<!--fig:ablation.delhi.gbm_tuned-->, still clear of zero.

### The choice

Every model this project has ever trained ran at `alpha=1.0`, which is
scikit-learn's default and was never chosen. `benchmark.tune_alpha` has reported
for some time that it is not the best setting in any of the six cities, and the
page described the curve as "nearly flat" and left it there.

The model-class ablation makes that harder to leave alone, because it puts a
number on what the default costs the *conclusions* rather than just the error.

### What the measurement says

Forward-chaining CV on each city's bootstrap window, from
[evaluation.md](evaluation.md#is-the-finding-about-the-world-or-about-a-linear-model):

- **The default costs real accuracy.** 11.9% in Delhi, 4.4% in Johannesburg,
  3.6% in Kraków, and under 2.2% in the other three.
- **And it inflates the retraining premium.** Delhi's +49.35% falls to +27.73%
  from tuning `alpha` alone, with no change of model class. That is 21.6 of the
  49.4 points, the single largest component of the headline, and larger than the
  18.0 points attributable to linearity.

The mechanism is not mysterious. A model penalised properly generalises across
seasons better, so it decays less, so a retrain has less to recover. Retraining
was partly compensating for a hyper-parameter nobody set.

### The argument for changing it

The project's own standard, applied everywhere else, is that a setting should be
measured rather than inherited. `alpha=1.0` is inherited. The sweep that says so
has been in the repository, and on the published page, for weeks.

And the headline is currently reported for a model that is known to be worse
than one line of configuration away. A reader is entitled to ask why.

### The argument for leaving it

Changing it restates the entire project. Every city figure, the gate
calibration, all five mechanism sweeps and both open decisions were produced at
`alpha=1.0`. This is not an extension, it is a re-baseline, and the before and
after would not be comparable.

The tuned values are also extreme in a way worth pausing on: `alpha=1000` in
Delhi shrinks the slopes most of the way to zero. That is a legitimate CV answer
and it is close to saying the features barely help, which is consistent with
every model here sitting a few percent above a constant predictor. Shipping it
would make the coefficient-evolution charts, which are a large part of what this
project shows, much less legible for a gain the reader cannot see.

And the honest framing does not require the change. What the ablation
establishes is *reported*: the premium is decomposed, the default is named as
the largest term, and the surviving effect is stated at +9.7%.

### Decision, as it stood until 2026-11-10

**No change, and this one is genuinely open.** The default stays at 1.0 and the
cost of that is now written down in three places rather than implied by a flat
curve.

Flipping it is a re-baseline of everything, which is the owner's call and not
the analysis's. What the analysis is entitled to say is that the setting is
unchosen, that it is the largest single component of the project's headline
number, and that both facts are now on the page.

### What would settle it

Deciding what this project is for. If it is a demonstration of a drift loop,
`alpha=1.0` is a defensible fixed point as long as its cost is published, which
it now is. If the six-city table is meant to be read as a measurement of what
retraining is worth, it should be measured on a model somebody chose, and the
re-baseline is unavoidable.
