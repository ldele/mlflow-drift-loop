# Decisions

Choices that changed what ships, with the evidence that drove them and the date.
Findings live in [evaluation.md](evaluation.md); this is for the calls made on
top of them, which are a different kind of thing and were previously mixed in.

---

## D1. The skill floor stays switched off, and the evidence now argues the other way

**Date:** 2026-08-11
**Status:** open, and left that way by whoever wrote this
**Affects:** `LoopConfig.skill_floor`, currently `None`

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
- Los Angeles acts in 25 of 35 weeks and improves by **+11.78% [+2.09, +17.17]**.

Los Angeles is the city the ratchet leaves deafest, silent for 35 of 36 runs, and
the one where retraining otherwise measurably hurts.

### The argument for switching it on

A floor that cannot harm any city in the set and measurably helps the one the
ratchet disables is, on this evidence, free. The earlier reading called the same
observation "a closer call than it sounds"; with an interval attached it is no
longer a close call in the direction it was assumed to fall.

### The argument for leaving it off

Three things, none decisive alone.

The whole-replay figure for Los Angeles is **+2.49% [+0.00, +15.01]**, which
touches zero. The acted-week effect is established; the effect on the replay as a
whole is not, because ten of the 35 weeks are ties and they pull the median in.

"Cannot harm any city" is a statement about six cities picked for contrast, not
about cities in general. A seventh could behave like Kraków does at
`skill < −0.25`, where the floor costs **−23.98% [−39.40, −5.48]** in the weeks
it acts. The gap between a floor that is inert and a floor that is destructive is
one setting wide.

Every published number in this repository was produced with the floor off.
Turning it on moves the six-city table, the gate calibration and the holdout
sweep at once, and the before-and-after comparison the project is built on would
have to be restated rather than extended.

### Decision

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
zero and it is positive (Kraków +1.5% [+0.8, +2.0] at 21 days). The confidence
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
