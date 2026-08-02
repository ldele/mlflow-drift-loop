# Evaluation

Whether it works, and where it does not. For how it works, see
[methodology.md](methodology.md).

## Six cities that disagree

Weather forecasts from the [Open-Meteo](https://open-meteo.com/) historical
forecast archive, joined on the hour with observed PM2.5 from its air-quality
API. Each city trains a champion on a clean season and replays weekly into the
season that ruins it.

| | span | PM2.5 swing | champion RMSE | retrains / runs | retraining worth |
|---|---|---|---|---|---|
| **Kraków** | May 25 → Feb 26 | 10 → 54, winter smog in a basin | 5.1 → peak 51.8 | 9 / 23 | +10.1% |
| **Santiago** | Oct 25 → Jul 26 | 18 → 94, winter inversion in a basin | 6.4 → peak 66.2 | 11 / 22 | **+26.1%** |
| **Delhi** | May 25 → Feb 26 | 42 → 127, post-monsoon burning | 20.0 → peak 66.0 | 6 / 16 | **+66.7%** |
| **Johannesburg** | Nov 25 → Jul 26 | 23 → 84, Highveld coal smoke | 11.6 → peak 81.6 | 9 / 20 | 0.0% |
| **Melbourne** | Sep 25 → Jul 26 | 5 → 15, winter wood heaters | 3.3 → peak 13.2 | 7 / 31 | −7.1% |
| **Los Angeles** | Sep 25 → Jul 26 | 15 → 29, a mild winter bump | 17.3 → peak 19.2 | 9 / 37 | −8.2% |

The cities were picked on measurements rather than reputation. Eighteen
candidates were fetched and ranked by PM2.5 swing before any of them was wired
in, which cost two obvious choices. Sydney turned out flat, at 1.5×, with no
story in it. No Brazilian city worked either: the Amazon burning-arc cities peak
at only about 14 µg/m³ in this window, and São Paulo swings 1.7×, less than Los
Angeles does.

### The loop has no calendar in it

Kraków, Delhi and Los Angeles all peak in northern winter, which leaves a fair
objection open. Are the thresholds quietly encoding a season? Santiago,
Johannesburg and Melbourne peak in June, July and August, when Kraków and Delhi
are at their cleanest, and they run on settings identical to every other city's.

Santiago answers it most directly, because it is Kraków's twin. A coastal basin
that traps winter inversions the same way, half a year out of phase. Both
champions decay by an order of magnitude (5.1 to 51.8 in Kraków, 6.4 to 66.2 in
Santiago), both draw retrains off the same thresholds in opposite seasons, and
retraining pays in both (+10.1% and +26.1%).

### Where it stops paying

Johannesburg is where the promotion gate does the most visible work. The
champion's error climbs from 11.6 to 81.6 µg/m³, the worst on the page, and nine
retrains produce only two promotions: the other seven challengers failed to clear
the 5% margin and were thrown away. Retraining nets exactly 0.0%. The loop spends
the effort, the gate declines to pay, and nothing ships that did not earn it.

Los Angeles is the control, and the measurements chose it for that. It was
picked as the summer-smog city; hourly PM2.5 over 2025–26 peaks in November and
bottoms out in June. Its champion barely moves across 37 runs, retraining costs
8.2%, and climatology beats it outright. A drift loop needs drift.

Melbourne is the counter-intuitive case. Its air stays near the WHO guideline all
year, yet the champion still decays to 2.01× its training error, and retraining
buys −7.1%. Clean air does not imply a stable model.

A seventh source, **Live schedule**, is not a city. It is the same Kraków data
run one incremental cycle at a time by a weekly GitHub Action, accruing its own
history over calendar time. It has run only a handful of cycles so far, and the
page says so on its own tab rather than letting two points pass for a trend. An
eighth, synthetic, has two independent drift knobs and backs the offline
correctness proof in [`sweep_knobs.py`](../scripts/sweep_knobs.py). It is not
published.

## Does the model beat anything?

![Benchmarks](images/benchmark.png)

Every predictor is scored on the same 14-day monitoring windows the loop reports
on, so the served champion, a champion that never retrained, and four
no-training baselines all land in one comparable column.

They are grouped by what each predictor is allowed to see rather than ranked in a
single list. Persistence and seasonal naive get the readings available when the
forecast was issued. The champion gets the weather forecast and nothing else.

- **The champion beats persistence in all six cities.** At a seven-day lead the
  last available reading is a week old, which is most of the way to useless, so
  the weather-based model earns its place. This is the finding that flips with
  the horizon: at a one-hour lead persistence wins everywhere by 3× or more.
- **Retraining earns its keep where drift is real:** +66.7% in Delhi, +26.1% in
  Santiago, +10.1% in Kraków. Delhi's frozen champion ends up worse than a
  constant, which is what concept drift looks like when nobody intervenes.
- **It costs 8.2% in Los Angeles** and 7.1% in Melbourne, and breaks exactly even
  in Johannesburg. Retraining a city whose world barely moves fits noise.
- **The champion is the best predictor of the six in Kraków, Delhi and
  Johannesburg**, second in Santiago, and still loses to climatology in Los
  Angeles. Where PM2.5 barely moves, a week-old forecast carries too little
  signal to beat an hour-of-day average.

## Does the detection actually work?

![The controlled experiment](images/control.png)

No real city can answer that. Its drift has no known cause and no control
condition, so a detector firing at random would look much the same. The synthetic
world has both: two knobs that move covariate drift and concept drift
independently, and each detector can be checked against the cause it should
answer to and the one it should ignore.

| knob turned 0 → 2 | data drift (PSI) | performance drift |
|---|---|---|
| **covariate** (feature distributions shift) | **1.77×** | 1.01× |
| **concept** (the relationship changes) | 1.00× | **5.92×** |

Each detector responds to its own cause and ignores the other. PSI across the
entire concept sweep is not merely stable, it is identical to fifteen decimal
places, because changing the relationship between features and target cannot move
a statistic computed on the features alone.

That is the property the two-signal design rests on, and it is the one thing the
six cities cannot demonstrate. It is published on the page for the same reason it
is here: it is the evidence, and the cities are the application.

## Limitations

- **R² is still weak, and that is the honest headline.** Widening from three
  features to eight improved every city (Los Angeles went from −2.13 to −0.98 on
  the latest window, Melbourne from −0.28 to +0.12) but R² on the most-drifted
  window is still near zero or negative in four of six. Predicting an hour's
  PM2.5 from a week-old weather forecast is genuinely hard, and the page should be
  read with that in mind: the loop is the demonstration, not the model.
- **Boundary layer height is missing and it is the feature I most want.** It sets
  the volume pollution is diluted into and would likely matter more than anything
  in the list. Open-Meteo does not archive previous model runs for it, so at a
  seven-day lead it returns null. Shortwave radiation is the stand-in.
- **No autoregressive features.** Giving the champion a lag term would help most
  in the low-drift cities where a constant currently beats it. It would also
  blunt the drift story, since an autoregressive model absorbs regime change
  instead of decaying visibly through it. That tension is a design decision.
- **One lead time.** Everything here is measured at seven days. The interesting
  sweep, error against lead from 1 to 7 days, is not run.
- **State lives in git.** Committing the SQLite backend back to the repo is the
  simplest zero-infra persistence and keeps history versioned, but it grows the
  repo. Production would point `MLFLOW_TRACKING_URI` at a hosted tracking server.
- **Artifact paths are absolute.** A backend generated on the CI runner resolves
  metrics, params and tags anywhere, but not the per-run prediction files.
- **Serving is single-node.** One container per city, holding its own SQLite
  backend, is the honest shape of a zero-infra demo rather than of production. A
  real deployment would point every replica at one hosted tracking server so they
  promote in step, and `POST /reload` would be called by the loop rather than by
  hand.
