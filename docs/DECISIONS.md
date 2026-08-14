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
