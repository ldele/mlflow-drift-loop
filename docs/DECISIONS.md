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
