# How the findings changed

Every number on this project's pages has been rewritten at least once, and the pages are
rewritten in place. This file is the trail: what was published, what replaced it, and what the
correction was. Reconstructed from the commit history on 2026-09-22, because most of it existed
nowhere else.

The short version: the first headline was measured in a way that flattered the loop in four
separate ways at once. Each was found and fixed. The claim ended at roughly a fifth of where it
started, and one half of it should have been withdrawn and was not.

## The headline, version by version

"Retraining worth" is whatever figure the README led with that day. It changes meaning at
2026-08-04, where an unpaired comparison across the whole replay gives way to a paired one over
the weeks a retrained model was actually serving. That switch is itself one of the corrections.

| date | commit | Delhi | Los Angeles | Kraków | what changed |
|---|---|---|---|---|---|
| 07-31 | `cb04f0f` | +53.8% | −12.7% | +26.9% | first published headline |
| 08-01 | `95b6ae1` | +47.3% | −11.6% | +4.4% | the model given a real 7-day forecast |
| 08-01 | `b1a15b8` | +66.7% | −8.2% | +10.1% | features widened from 3 to 8 |
| 08-03 | `5e10624` | +43.8% | −8.9% | +0.2% | full annual replay; challenger window 45 → 180 days |
| 08-04 | `3aff123` | +49.4% | 0.0% | +6.4% | paired week-by-week metric added and led with |
| 08-05 | `679ac9a` | +49.4% | −2.8% | +6.5% | first-run promotion credited to the right model |
| 08-06 | `e1f91a3` | +49.4% | −13.4% | +6.5% | first monitor window no longer overlaps training data |
| 08-11 | `bc0bf4c` | +49.4% [+33.7, +61.8] | −13.4% [−36.9, −3.8] | +6.5% [−15.3, +27.5] | moving-block bootstrap intervals |
| 08-12 | `541db47` | +36.3% [+17.4, +47.6] | −3.5% [−9.7, +1.2] | — | ablation arm: a tree at library defaults |
| 08-16 | `d22977b` | +9.7% [+8.0, +26.0] | −6.9% [−13.2, +2.8] | — | ablation arm: both model classes tuned |

The last two rows are arms of an ablation rather than replacements for the shipped figure. The
README still leads with the shipped model's numbers, which is a choice and not an oversight — but
see "The half that should have been withdrawn" below.

## What each correction was

**The model was not forecasting** (`95b6ae1`, 2026-08-01). Until this commit the features were
the analysed weather for the hour being predicted, so the model was a same-hour estimator with no
forecasting in it. `FORECAST_LEAD_DAYS = 7` makes the features the forecast as it stood a week
earlier, which is the job the README describes, and the model inherits the weather forecast's own
error. Kraków's premium fell from +26.9% to +4.4% on the harder problem.

**The replay stopped where the story was flattering** (`5e10624`, 2026-08-03). Kraków and Delhi
ran to 2026-01-20 and 2026-01-25, which is just past their dirty-season peak. The loop was only
ever watched through the world getting worse, which is the half where any refit helps. Both now
run to 2026-07-10.

**A challenger could only learn one season** (same commit). `challenger_train_days` was 45. A
six-week window sees one season, wins its exam, and is wrong as soon as the year turns. The
half-year replay had been hiding it. Held at 45 days over a full year, retraining comes out
**−7.2% in Delhi and −29.6% in Kraków**; at 180 days, +43.7% and +0.2%. This is the most useful
finding in the repository and it was only visible once the replay ran past the flattering point.

**The metric compared the wrong things** (`3aff123`, 2026-08-04). The unpaired figure takes median
error across the whole replay against never retraining, so in a city that promotes nothing until
late, most windows compare the first model against itself and the answer collapses toward zero.
Johannesburg read 0.0% unpaired and +14.9% paired, over the six weeks a retrained model was
actually serving.

**Two first-run accounting bugs** (`679ac9a` and `e1f91a3`, 2026-08-05 and 08-06).
- A first run that promotes had no earlier row to read the outgoing version from, so the promotion
  was credited to the wrong model.
- Five of six replays started 9 to 11 days after the champion's training ended, against a 14-day
  monitor window, so run 0 scored the champion on 3 to 5 days of its own training data and
  understated its error by up to 15%.

Fixing the second dropped one run from five cities. Two of Los Angeles's three retrains fell
inside the dropped window, so it went from 3 retrains to 1 and from −2.8% to −13.4%. From that
point the entire Los Angeles result rests on a single promotion. `test_loop.py` now asserts the
gap for every shipped profile.

**Nothing carried an interval until 2026-08-11** (`bc0bf4c`). With a 95% moving-block bootstrap,
Kraków and Melbourne stopped being results: both had been published as small positive findings.
Kraków's 47 weekly comparisons carry about five independent observations, which is what makes
every interval on the page as wide as it is.

**A confound check that could not fail was read as evidence** (`541db47` → `d22977b`). The first
ablation swapped in a gradient-boosted challenger at library defaults. It lost to the Ridge, and
the README concluded that "finding 1 does not depend on the model class". The tree was undertrained:
tuning is worth 26.5% to it, and the shipped Ridge was undertuned by 11.9%, so tuning only the tree
would have inverted the unfairness. With both classes tuned on the same protocol, Delhi's +49.35%
decomposes into **21.62 points lost to the shipped `alpha`, 17.99 to linearity, and +9.74% left
over**. The earlier conclusion is withdrawn in `docs/evaluation.md`.

**The `alpha` default was seen on day one and misjudged.** The 2026-07-31 README already said
`alpha=1.0` ships while forward-chaining CV picks 30 to 100, "worth 0.1-2.4% error", and called the
curve nearly flat. That judged the setting by its effect on the model's accuracy rather than by its
effect on the headline, where it is worth 21.6 points. `docs/DECISIONS.md` D5 records it: the page
described the curve as nearly flat and left it there.

## The half that should have been withdrawn

Finding 2, "retraining costs where the air did not move", was written on 2026-07-31 as "it costs
12.7% in Los Angeles. A drift loop needs drift." Los Angeles has since read −11.6, −8.2, −8.9,
0.0, −2.8 and −13.4, and −3.5 and −6.9 under better models. The sentence survived all of it, with
a new caption each time: a cost, then a coin toss, then a coin toss that costs money, then
measurably harmed.

`docs/evaluation.md` states the conclusion plainly — the cost half "holds for the model that ships
and cannot be established for the best model available" — while the README and the published
report still lead with it. Those two should agree. What the evidence supports is narrower and
more interesting: where the air barely moves, the promotion gate has almost no signal to select
on. Los Angeles's exam intervals are 39.2 points wide at the median, its one promotion was won on
+21.9% [−6.9, +32.3], and the ratcheted trigger then left that model serving for 35 of 36 weeks.

## The scheduled loop, which is a separate story

The replay is the measurement. The weekly Action is the thing actually running, and its record is
thinner than the replay's:

- Champion v1 was bootstrapped 2026-07-21 on three features at `alpha=1.0`.
- The Action then failed every Monday from 2026-08-03 to 2026-09-07, because `b1a15b8` had widened
  `FEATURES` under a model already fitted on three columns. Fixed in `a69dfb6` on 2026-09-11.
- The six missed weeks were never run. The loop resumed at the current week, so those weekly
  decisions do not exist.
- v2 was promoted on 2026-09-21, winning its seven-day exam by 0.51 µg/m³ (6.35 against 6.86).
- Every mechanism this project built and measured is switched off in that config: the skill floor,
  re-certification, probation and the confidence gate. The retrain alarm is the ratcheting one, now
  set at 1.25 × v2's own baseline of 5.45.

In two months of running, the loop made one retraining decision, and it is configured to reproduce
the ratchet this winter.

## What this history is for

Three habits are worth keeping from it, and they are cheaper than the corrections were.

1. **Run the replay past the point that flatters the system.** Stopping at the peak cost six
   weeks of wrong conclusions, and nothing else would have exposed the 45-day window.
2. **A check that cannot fail is not evidence.** The untuned tree lost, and losing was the only
   result it could produce.
3. **Judge a confound by its effect on the claim, not on the model.** `alpha` looked worth 2% of
   accuracy and was worth 21.6 points of the headline.

And one this file exists because of: when a published number changes, keep the old one beside it
with the reason. The README does that for Kraków and Melbourne. It did not for anything above.
