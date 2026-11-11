# How the findings changed

Every number on this project's pages has been rewritten at least once, in place. This file
is the trail: what was published, what replaced it and why. It was reconstructed from the
commit history on 2026-09-22 and has been kept up since.

In short: the first headline was measured in four ways that each flattered the loop. Each
was found and fixed, and the claim ended at less than half of where it started.

## The headline, version by version

"Retraining worth" is the figure the README led with that day: percent lower error than
never retraining. From 2026-08-04 it is the paired week-by-week figure.

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
| 11-10 | this change | +22.8% [+11.8, +36.4] | −3.5% [−25.6, +3.0] | +5.7% [−14.8, +26.8] | the loop itself: penalty tuned at every fit, second retrain alarm on |

The rows dated 08-12 and 08-16 are arms of an ablation, not replacements for the shipped
figure. The last row is the loop itself being changed.

## What each correction was

- **The model was not forecasting** (`95b6ae1`, 2026-08-01). The features were the analysed
  weather for the hour being predicted. `FORECAST_LEAD_DAYS = 7` makes them the forecast
  issued a week earlier. Kraków fell from +26.9% to +4.4%.
- **The replay stopped at the flattering point** (`5e10624`, 2026-08-03). Kraków and Delhi
  ended just past their dirty-season peak. Both now run to 2026-07-10.
- **A challenger could only learn one season** (same commit). At a 45-day training window
  over a full year, retraining comes out −7.2% in Delhi and −29.6% in Kraków; at 180 days,
  +43.7% and +0.2%.
- **The metric compared the wrong things** (`3aff123`, 2026-08-04). Unpaired medians across
  the replay mostly compare the first model with itself in a city that promotes late.
  Johannesburg read 0.0% unpaired and +14.9% paired.
- **Two first-run accounting bugs** (`679ac9a`, `e1f91a3`, 2026-08-05 and 08-06). A first
  run that promotes was credited to the wrong model, and run 0 scored the first model on 3
  to 5 days of its own training data in five cities. Fixing the second took Los Angeles from
  three retrains to one and from −2.8% to −13.4%.
- **Nothing carried an interval until 2026-08-11** (`bc0bf4c`). With a 95% moving-block
  bootstrap, Kraków and Melbourne stopped being results. Kraków's 47 weekly comparisons
  carry about five independent observations.
- **A check that could not fail was read as evidence** (`541db47` to `d22977b`). The first
  ablation used a tree at library defaults, which lost, and the README concluded the finding
  did not depend on the model class. With both classes tuned, Delhi's +49.35% splits into
  21.62 points from the default `alpha`, 17.99 from linearity and +9.74% left over.
- **The `alpha` default was seen on day one and misjudged.** The 2026-07-31 README called
  the tuning curve nearly flat. That judged the setting by the model's accuracy, not by its
  effect on the headline, where it was worth 21.6 points.
- **A restated figure had drifted from its output** (2026-09-23). Santiago's interval was
  published as [−0, +43] where the output holds 42.49. Every page now fails CI when a keyed
  figure disagrees with `outputs/figures.json`.
- **The configuration was changed, three months after the evidence said to** (2026-11-10).
  The six profiles and the live schedule took the penalty tuned at every fit and the second
  retrain alarm. Delhi reads +22.8% [+11.8, +36.4], Los Angeles −3.5% [−25.6, +3.0] and no
  longer a harm, Melbourne +5.6% [+2.0, +6.6]. The same pass found the benchmark's
  never-retrained model still fitted at the default penalty (+44.2% for Delhi where the
  replay reads +17.9%), and a site sentence saying no short-serving promotion left its city
  worse off, when two of 28 had.
- **The five fixes were measured again, and two results shrank** (2026-11-10). Re-run on
  the loop as it ships, with ten predictions and a reading rule written first: an effect
  counts only where it clears zero, in the same direction, in two cities. Seven predictions
  held. The confidence gate, published as "actively harmful", cost two cities on the first
  loop and one on this one. The second alarm's Los Angeles gain went from
  +11.8% [+2.1, +17.2] to +6.6% [−1.1, +9.3]. Intervals clear of zero fell from 15 of 55 to
  6 of 59. The alarm stays on, by the owner's decision, as a bound on silence.
- **One mechanism met cities it was not found on** (2026-11-11). The second alarm was run on
  nine other cities, with candidates, windows and four predictions committed (`6d8f993`)
  before any data was fetched. It shortened the longest silence in all five cities where it
  fired. The accuracy gain did not appear: Istanbul +8.8% [−1.4, +11.1], Cairo
  −4.5% [−7.2, +1.0], Dhaka −6.0% [−19.0, +9.2].

## The half that should have been withdrawn

"Retraining costs where the air did not move" was written on 2026-07-31 as "it costs 12.7%
in Los Angeles". Los Angeles has since read −11.6, −8.2, −8.9, 0.0, −2.8 and −13.4, and the
sentence survived each with a new caption. On the loop as it ships it reads
−3.5% [−25.6, +3.0], and the README no longer makes the claim. The published report has not
been changed.

What the evidence supports is narrower: where the air barely moves, the promotion exam has
almost no signal to select on. Los Angeles's exam intervals are 39.2 points wide at the
median, and its one promotion on the first loop was won on +21.9% [−6.9, +32.3].

## The scheduled loop, which is a separate story

- The first model was bootstrapped on 2026-07-21, on three features at `alpha=1.0`.
- The weekly job failed every Monday from 2026-08-03 to 2026-09-07, because `b1a15b8`
  widened the features under a model fitted on three columns. Fixed in `a69dfb6` on
  2026-09-11. The six missed weeks were never run.
- v2 was promoted on 2026-09-21, winning its exam by 0.51 µg/m³ (6.35 against 6.86).
- Until 2026-11-10 the live loop ran with one retrain alarm and the default penalty. Since
  then it runs the same configuration as the six replays. The model serving at the change
  was v7, fitted at the default penalty; it serves until a rival passes its exam.
- One cycle failed on 2026-11-02 on a busy-server reply from Open-Meteo, and the next Monday
  caught the week up. The fetch now retries.

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
