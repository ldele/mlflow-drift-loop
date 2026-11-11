# Decisions

Choices that changed what ships, with the evidence and the date. Findings are in
[evaluation.md](evaluation.md); the first loop's are in
[evaluation-first-loop.md](evaluation-first-loop.md).

---

## D1. The skill floor ships switched on, at −0.5

**Status:** on since 2026-11-10, as a bound on how long the loop can go without being able
to retrain. No accuracy claim.
**Affects:** `LoopConfig.skill_floor`, set to −0.5 by `config.SHIPPED` in the six city
profiles and the live schedule. A bare `LoopConfig` has it off.

**What it is.** The first retrain alarm compares a model with its own error at training
time, and every promotion resets that bar, so it rises until the alarm cannot fire. The
skill floor is a second alarm: retrain when skill against a 30-day daily profile falls below
a floor that no promotion can move.

**How the decision went.**

- **2026-08-11, left off.** On the first loop at −0.5, five cities were identical week for
  week and Los Angeles improved by +11.78% [+2.09, +17.17]<!--fig:first.floor.la.-0.50.acted-->
  over the 25<!--fig:first.floor.la.-0.50.differing--> of 35<!--fig:first.la.acted.n--> weeks
  it acted. Over all weeks Los Angeles read +2.49% [+0.00, +15.01]<!--fig:first.floor.la.-0.50.all-->,
  and at −0.25 Kraków lost −23.98% [−39.40, −5.48]<!--fig:first.floor.krakow.-0.25.acted-->,
  so one setting separates inert from harmful. The gain was one arm of eighteen
  ([the sweep](evaluation-first-loop.md#what-the-medians-were-hiding-in-both-directions),
  [multiplicity](evaluation-first-loop.md#what-is-still-not-corrected)). Changing a shipped
  default was left to the owner.
- **2026-11-10, switched on by the owner, with D5.** Taken on the Los Angeles gain and on
  the silences: the longest stretch the loop could not fire falls from
  35<!--fig:rebaseline.la.shipped.silence--> weeks to 16<!--fig:rebaseline.la.fixed.silence-->
  in Los Angeles and from 30<!--fig:rebaseline.krakow.shipped.silence--> to
  22<!--fig:rebaseline.krakow.fixed.silence--> in Kraków.
- **2026-11-10, reopened by the re-run.** On a loop that already tunes its model, the alarm
  alone reads +6.6% [−1.1, +9.3]<!--fig:floor.la.-0.50.acted--> in Los Angeles over the
  22<!--fig:floor.la.-0.50.differing--> weeks it changed the serving model. The prediction
  that the gain would repeat failed. The silences held: with the alarm off the loop cannot
  fire for 35<!--fig:floor.la.off.silence--> of Los Angeles's 36<!--fig:la.weeks--> weeks or
  30<!--fig:floor.krakow.off.silence--> of Kraków's 48<!--fig:krakow.weeks-->; with it on the
  longest silences are 16<!--fig:floor.la.-0.50.silence--> and
  22<!--fig:floor.krakow.-0.50.silence-->.
- **2026-11-10, confirmed by the owner,** as a bound on silence only.
- **2026-11-11, tested on nine cities outside the six,** on rules committed before the data
  was fetched ([evaluation.md](evaluation.md#the-second-alarm-on-cities-the-study-had-not-used)).
  The alarm shortened the longest silence in all five cities where it fired. The accuracy
  gain is retired: +8.8% [−1.4, +11.1]<!--fig:heldout.istanbul.acted--> in Istanbul and
  −4.5% [−7.2, +1.0]<!--fig:heldout.cairo.acted--> in Cairo. Dhaka is the one to watch: the
  alarm changed its serving model for 28<!--fig:heldout.dhaka.changed--> weeks and reads
  −6.0% [−19.0, +9.2]<!--fig:heldout.dhaka.acted-->.

**What would change it.** A city harmed at −0.5, meaning an interval clear of zero. Whether
Dhaka's lean is enough is the owner's call.

---

## D2. Re-certification ships switched off, and this time the evidence agrees

**Status:** off, settled 2026-08-13, unchanged by the re-run of 2026-11-10.
**Affects:** `LoopConfig.recertify_days`, `None`.

**What it is.** Re-sit the promotion exam when the serving model's certificate expires,
whatever the drift signals say
([detail](evaluation-first-loop.md#re-certification-bounds-staleness-and-staleness-was-not-the-thing-that-hurt)).
It was built after a more sensitive alarm
([here](evaluation-first-loop.md#fixing-the-trigger-one-fix-is-impossible-the-other-pays-in-one-city))
and a longer exam ([here](evaluation-first-loop.md#lengthening-the-exam-does-not-fix-it))
both failed to pay.

**Evidence, first loop.** It bounds how long a model serves unexamined in every city;
without it Los Angeles served one for 252 days. The bound does not become accuracy: of 24
comparisons three clear zero, Los Angeles at 14 days (+13.8%) and 28 days (+6.4%), and Delhi
at 35 days (−10.8%). Every arm trains more models (Kraków 14 to 34). The effect is the extra
promotions it sends to an exam that cannot certify past about five weeks.

**Evidence, loop as it ships.** No cadence clears zero in two cities. One arm does in one:
Los Angeles at 35 days, +17.0% [+8.3, +21.9]<!--fig:recertify.la.35d.acted--> over 9 weeks.

**Decision.** Off. The knob stays, and `certified_age_days` is logged on every run.

**What would change it.** An exam that stays calibrated over a model's real service life.

---

## D3. The promotion gate keeps deciding on a point estimate

**Status:** off, settled 2026-08-14. Still off after the re-run, for a weaker reason.
**Affects:** `LoopConfig.promotion_confidence`, `None`.

**What it is.** The exam promotes on the bare difference of two RMSEs over one week. Over
the first loop's 28 promotions, 11 had an exam margin whose interval reached below the 5%
they had to clear, and 6 reached below zero
([detail](evaluation-first-loop.md#the-gate-decides-on-a-point-estimate-and-making-it-decide-on-an-interval-is-worse)).
`promotion_confidence` requires the challenger to clear the margin at the interval's lower
bound too.

**Evidence, first loop.** Of 18 arms five clear zero and all five are harmful (Kraków −3.1%,
Delhi −3.9%). The gap between what the exam promised and what the winner delivered widens in
four of six cities. Blocking a promotion defers it: the loop retries, and the challenger
that clears a higher bar does so on a rarer fluctuation.

**Evidence, loop as it ships.** The only city clear of zero is Santiago, at
−2.5% [−3.4, −1.0]<!--fig:confidence.santiago.0.95.acted-->, and the promise-delivery gap
widens in three cities and narrows in three. Not shown to harm, not shown to help.

**Decision.** Off. `exam_margin_lo` is logged whenever it is on.

**What would change it.** A rule that counts attempts: carry the number of exams a model has
faced into the decision, or judge a promotion afterwards and undo it (D4).

---

## D4. Probation ships off, on "unproven" rather than "harmful"

**Status:** off, settled 2026-08-14, unchanged by the re-run.
**Affects:** `LoopConfig.probation_days`, `None`.

**What it is.** A promotion is provisional: after the window the new model is scored against
the one it replaced on data after both, and rolled back if worse
([detail](evaluation-first-loop.md#making-the-promotion-reversible-and-why-that-does-not-rescue-it-either)).

**Evidence, first loop.** Of 18 arms one clears zero, and it is positive: Kraków
+1.5% [+0.8, +2.0]<!--fig:first.probation.krakow.21d.acted--> at 21 days, from one rollback
over seven weeks. One in eighteen is what chance produces. 8 of 49 judged promotions were
rolled back, so promotions are mostly right two to four weeks out. Los Angeles's one
promotion is rolled back when judged at 14 days and kept at 21 or 28.

**Evidence, loop as it ships.** None of the 5 intervals clears zero, and 7 of 55 judged
promotions were rolled back.

**Decision.** Off. It does no measured harm and is not shown to help.

**What would change it.** Not another decision rule. A window short enough to isolate one
promotion finds little wrong, and one long enough to reach the failure cannot tell a bad
promotion from a world that moved. What is left is the measurement: a target with more
signal per week, or a longer record.

---

## D5. The loop chooses its own alpha at every fit

**Status:** tuned at every fit since 2026-11-10. Open for three months before that.
**Affects:** `LoopConfig.tune_alpha`, set by `config.SHIPPED`. A bare `LoopConfig` fits at
`alpha=1.0`, and `config.FIRST_SHIPPED` pins it so the first loop's tables still reproduce.

**What it is.** Every model ran at `alpha=1.0`, scikit-learn's default, which nobody chose.

**Evidence** ([the ablation](evaluation-first-loop.md#is-the-finding-about-the-world-or-about-a-linear-model)).
The default cost 11.9% in error in Delhi, 4.4% in Johannesburg, 3.6% in Kraków and under
2.2% elsewhere. It also inflated the result: tuning `alpha` alone took Delhi's +49.35% to
+27.73%, which is 21.6 of the 49.4 points and more than the 18.0 from leaving linearity. A
properly penalised model decays less, so a retrain has less to recover.

**How the decision went.**

- **2026-08-14, left at 1.0.** Changing it restates every figure, and `alpha=1000` in Delhi
  shrinks the slopes most of the way to zero, which makes the coefficient charts harder to
  read. The cost was published instead.
- **2026-11-10, tuned, by the owner, with D1.** The six-city table is read as a measurement
  of what retraining is worth, so it is measured on a model whose setting was chosen. Delhi
  moved from +49.4% [+33.7, +61.8]<!--fig:first.delhi.acted--> to
  +22.8% [+11.8, +36.4]<!--fig:delhi.acted--> and Los Angeles from
  −13.4% [−36.9, −3.8]<!--fig:first.la.acted--> to −3.5% [−25.6, +3.0]<!--fig:la.acted-->.
  The earlier figures stay on the pages beside the new ones. The ablation and the five
  sweeps were re-run the same day with predictions written first
  ([evaluation.md](evaluation.md#the-five-fixes-and-the-model-ablation-on-the-loop-as-it-ships));
  under a tuned tree Delhi's premium is +13.4% [+8.0, +27.0]<!--fig:ablation.delhi.gbm_tuned-->.
