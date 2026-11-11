# Registration — the second retrain alarm, on cities the study has not used

**Registered:** 2026-11-10, on a clean working tree at `4c6bbe6`. To be committed, together
with the script that implements it, before any candidate city's data is fetched. The commit
that adds this file is the evidence of that order.
**Arm code exists at registration:** yes. `scripts/sweep_skill_floor.py` already replays a
city with the alarm off and on, and `scripts/held_out_cities.py` is written alongside this
file. Neither has been run on any city listed below, and none of their data is in
`data_cache/`.

## Question

Does the second retrain alarm, at the shipped setting of −0.5, harm a city it was not tuned
on, and does the accuracy gain once seen in Los Angeles appear in a new city whose first
alarm goes deaf?

The alarm is kept today as a bound on how long the loop can go without firing, with no claim
that it improves forecasts (`docs/DECISIONS.md`, D1). This is the test D1 has asked for since
August, and it is the first time any verdict in this project meets a city it was not found on.

## What earlier runs left open

1. Whether the alarm harms a city outside the six.
2. Whether its accuracy gain is real. It cleared zero in Los Angeles on the first loop and
   covers zero there on the loop as it ships.
3. Whether the shorter silences it produces are a property of the mechanism or of those six
   replays.

This experiment answers 1 and 3 and attempts 2. It cannot settle 2 unless the effect is large,
for the reason given under Scoring.

## Units

One unit is one weekly monitoring window in one city, scored under two arms on the same
hours. Each city will hold 30 to 48 of them, and by the six-city table's measure about 2 to
10 independent observations.

**Candidate cities, in a fixed order.** Twelve large urban areas, one per country, none in a
country the study already uses, listed from memory in rough order of population. The order is
fixed by this file and is not claimed to be an exact ranking. Coordinates are the city centre.

| # | city | latitude | longitude |
|---|---|---|---|
| 1 | Tokyo, JP | 35.6762 | 139.6503 |
| 2 | Shanghai, CN | 31.2304 | 121.4737 |
| 3 | São Paulo, BR | −23.5505 | −46.6333 |
| 4 | Mexico City, MX | 19.4326 | −99.1332 |
| 5 | Cairo, EG | 30.0444 | 31.2357 |
| 6 | Dhaka, BD | 23.8103 | 90.4125 |
| 7 | Karachi, PK | 24.8607 | 67.0011 |
| 8 | Buenos Aires, AR | −34.6037 | −58.3816 |
| 9 | Istanbul, TR | 41.0082 | 28.9784 |
| 10 | Lagos, NG | 6.5244 | 3.3792 |
| 11 | Manila, PH | 14.5995 | 120.9842 |
| 12 | Kinshasa, CD | −4.4419 | 15.2663 |

**Data.** Each candidate is fetched over 2025-05-01 to 2026-07-15 at a seven-day lead, the
span and lead Kraków and Delhi use. A candidate is dropped, and the reason recorded, if fewer
than 9,500 clean hourly rows come back, which is 90% of the span.

**Replay windows, by rule.** The six existing cities had their windows picked by hand to put a
clean training season before the season that spoils it. Here the same idea is a rule, so
nobody chooses a window after seeing a result. The first model trains on the 75 consecutive
days with the lowest mean PM2.5 among all 75-day windows ending between 2025-07-15 and
2025-11-27; the earliest wins a tie. The first weekly run is 15 days after that window ends,
the last is 2026-07-10, and runs are 7 days apart, which gives at least 30.

**Which candidates are walked.** Candidates are taken in order. For each, only the alarm-off
arm is run first, and from it two numbers are read: the longest run of weeks in which the
first alarm did not fire, and the number of those quiet weeks in which the model's skill was
below −0.5. A city counts as having **a deaf first alarm** when the first number is at least
10 and the second at least 3. That is the condition under which the second alarm has anything
to do, and it is read from the arm without the second alarm, so it says nothing about what the
second alarm achieves. The walk stops at the second city with a deaf first alarm, or at the
end of the list. Every city walked then gets the alarm-on arm.

## Arms

| arm | what the loop is given |
|---|---|
| off | the loop as it ships, with the second alarm disabled |
| −0.5 | the loop as it ships |

No other setting is run. D1 asks for −0.5 only, and the aggressive settings have already been
shown to cost a city accuracy on both loops.

The instrument is checked on two cities whose answers are known, before any candidate is
fetched: Los Angeles must come out with a deaf first alarm and Santiago without one, and Los
Angeles's −0.5 arm must reproduce its published premium to one decimal. Replays are
deterministic, so there is no separate inert arm: a city where the alarm never fires
reproduces its off arm exactly, and that is recorded as the alarm doing nothing there.

## Predictions

"Clears zero" means the 95% interval on the week-by-week difference between the two arms,
over the weeks they served different models, excludes zero.

| # | Prediction | Comparison that tests it | Fails if |
|---|---|---|---|
| H1 | The alarm harms no city walked. | −0.5 against off, each city | any city clears zero on the negative side |
| H2 | In each city with a deaf first alarm, the second alarm shortens the longest silence. | longest silence, off against −0.5 | it is the same or longer in either such city |
| H3 | The gain does not reach the standard for a claim. | −0.5 against off, the cities with a deaf first alarm | both such cities clear zero on the positive side |
| H4 | The direction is positive where the alarm acts. | sign of the week-by-week difference in each such city | it is negative in either |

How sure these are. H1 about 85%: at this setting the alarm changed the serving model in two
of six cities and harmed neither. H2 about 90%, since a quiet week with skill below −0.5 is
where the alarm fires by construction. H3 about 70%: I expect at most one city to clear zero,
because a gain the size of Los Angeles's is smaller than this design usually detects. H4
about 60%.

## Scoring

The statistic and interval are the ones every table on the page uses: the paired percentage
improvement in RMSE between two arms on the same weekly windows, over the weeks they served
different models, with a 95% moving-block bootstrap interval (`driftloop.stats`).

The standard for a claim is the one registered for the five fixes on 2026-11-10: the same
direction, clear of zero, in two cities. Here that means both cities with a deaf first alarm.

The design is small, and the earlier registration's simulation applies unchanged: on about
five independent observations a percentile interval is too narrow and still misses a real
effect most of the time. Los Angeles's own interval on the loop as it ships is 5 points wide
on each side around a gain of 6.6%. So this experiment can find a harm, confirm the change in
behaviour, and confirm a large gain. It cannot rule out a modest one, and a null here will be
reported as "not shown", never as "no effect".

## Void condition

The harm and silence predictions are void if the instrument check fails on Los Angeles or
Santiago. The gain predictions, H3 and H4, are void if no walked city has a deaf first alarm,
or if the alarm changes the serving model in fewer than 5 weeks in every such city, since
there is then no interval to read.

## What each outcome would decide

| Outcome | What it means | What follows |
|---|---|---|
| H1 fails | the alarm harms a city at the shipped setting | reported to the owner as the evidence D1 names for switching it off; the setting is not changed by this analysis |
| H2 fails | the bound on silence is not a property of the mechanism | the one claim the pages make for the alarm is withdrawn, and D1 is reported as reopened |
| H3 fails | the gain clears zero in two new cities | reported as scored once on new units; restoring an accuracy claim is the owner's decision |
| H3 holds, H4 holds | a consistent direction that does not reach the standard | the pages keep "a bound on silence, no accuracy claim" and add the new cities as evidence for it |
| H4 fails | the alarm's effect on accuracy has no consistent sign | the Los Angeles gain is described as a first-loop observation only |
| gain predictions void | the cities cannot carry the question | reported as such, with how many candidates were walked |

## Limits known in advance

Twelve candidates listed from memory are not a sample of the world's cities. They are simply
cities nobody looked at before writing the rule.

The window rule is new and differs from how the six were chosen. A replay that starts in the
cleanest stretch between July and November suits the northern hemisphere better than the
southern, where the clean season may fall outside that range.

The condition for a deaf first alarm selects cities on the reference arm. It cannot leak the
alarm's effect, but it does mean the gain is tested only where the alarm has room to act, so
a gain found here says nothing about a typical city.

Up to twelve cities are read for harm at an uncorrected 95%. One clearing zero by chance is
plausible, so a single harmed city will be reported with that caveat and with its interval.

The target is CAMS model output in every city, as for the six.

## Addenda

**2026-11-11, after every result had been seen.** Nothing above was edited. This file and the
script were committed as `6d8f993` at 10:14 on 2026-11-11, before the first candidate was
fetched. The fetched files and their hashes are in `outputs/held_out/candidates.json`.

Nine candidates were walked, Tokyo to Istanbul, and none was dropped. Cairo and Istanbul were
the two with a deaf first alarm. H1, H2 and H3 held. H4 failed: Cairo's direction is negative.

Three things the registration did not anticipate.

The void condition for the gain predictions required fewer than 5 changed weeks in every city
with a deaf first alarm. Cairo had 4 and Istanbul 6, so the predictions are not void, but
Cairo's sign rests on fewer weeks than the script's own threshold for reading an interval. H4
is scored as failed and that is said beside it.

Tokyo clears zero on the positive side over 4 changed weeks. It is not one of the two cities
the gain predictions were registered on, so it is reported and not counted.

The condition for a deaf first alarm did not predict where the alarm would change the serving
model. Dhaka failed it and had 28 changed weeks; Cairo passed it and had 4. A later test of
this kind should pick its cities on something other than a count of eligible weeks, and the
write-up says so.

Results are in `docs/evaluation.md`, "The second alarm, on cities the study had not used".
`python scripts/held_out_cities.py` repeats the run from the committed data.
