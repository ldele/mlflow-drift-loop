"""How to read each chart, in one place, for both user interfaces.

The published site and the Streamlit app draw the same quantities and are
written in different languages, so a reading kept in each would be two readings
within a month. This module is the single copy. `scripts/build_site.py`
serialises it into `site/data.json` for the browser; `dashboard/app.py` imports
it directly.

## Why a report needs this at all

A published report is normally a chart plus someone's reading of it. Neither of
these has a someone: they rebuild from whatever the last run produced, and
nobody writes a caption for the version that comes out. So the reading has to be
written once, as a rule rather than as an observation -- what the marks mean,
what the arithmetic behind them is, and what it would mean for the line to go
each way. A reader arriving at a shape nobody has seen yet can still interpret
it.

## The four parts

``read``   what the marks are, ending in something to look at on this chart
``math``   the arithmetic in a line or two, so the axis label is not the only
           statement of what is being computed
``moves``  the directional reading: what it means when this goes up, down, or
           nowhere. This is the part a static caption cannot carry, because it
           has to hold for a shape the author has not seen
``next``   what would improve the *monitoring*, not the air. Each chart knows
           something about its own blind spot, and saying so is how the pages
           point at their own next piece of work

## Two conventions that keep it honest

**Thresholds are placeholders, never literals.** ``{perf}`` and ``{monitor}``
are filled by :func:`context` from the config the loop is actually running. A
guide that retyped "1.25x" would be one edit away from describing a loop that no
longer exists, and nothing would fail.

**The prose is Markdown, not HTML.** Streamlit renders it directly; the site
converts the two constructs used -- ``backticks`` and ``*emphasis*`` -- and
escapes everything else. Punctuation is Unicode rather than entities for the
same reason: one string has to satisfy both renderers.
"""

from __future__ import annotations

from dataclasses import dataclass

# Charts that exist in only one of the two interfaces still live here. The set
# is the union, not the intersection, so moving a chart between the site and the
# dashboard does not mean rewriting its reading.
SITE_ONLY = frozenset({
    "forecast", "map", "pooled", "ablation", "compare_city", "compare_value",
})
DASHBOARD_ONLY = frozenset({
    "distributions", "alpha_sweep", "fit_scatter", "residuals",
})


@dataclass(frozen=True)
class Guide:
    """One chart's reading. ``moves`` is ordered: most common case first."""

    read: str
    math: str
    moves: tuple[tuple[str, str], ...]
    next: str

    def filled(self, values: dict[str, object]) -> dict:
        """Substitute the threshold placeholders, once, at build time."""
        fill = lambda s: s.format(**values)  # noqa: E731
        return {
            "read": fill(self.read),
            "math": fill(self.math),
            "moves": [[fill(a), fill(b)] for a, b in self.moves],
            "next": fill(self.next),
        }


GUIDES: dict[str, Guide] = {
    # ---------------------------------------------------------------- per city
    "forecast": Guide(
        read=(
            "Two lines over the most recent window: what the monitors actually measured, and what "
            "the model said about that hour {lead} earlier. The vertical gap between them is the "
            "error every other chart summarises. Look at the peaks rather than the flat stretches, "
            "because that is where a pollution forecast either earns its keep or does not."
        ),
        math=(
            "No arithmetic yet. This is the raw prediction, before it is scored against anything, "
            "which is why it comes first."
        ),
        moves=(
            ("they track each other",
             "the model has the shape of the week roughly right, which is the most this one is asked for"),
            ("prediction flat through a measured spike",
             "the weather forecast gave no warning of the episode. A limit of the input rather than of the model"),
            ("prediction consistently under the peaks",
             "expected. Fitting squared error on a skewed target pulls every answer toward the middle, "
             "and pollution episodes are the tail"),
        ),
        next=(
            "The model answers with a single number and no statement of confidence, so a reader "
            "cannot tell a confident wrong answer from an uncertain one. A prediction interval here "
            "would make that difference visible."
        ),
    ),
    "skill": Guide(
        read=(
            "One line, against zero. Zero is not *no error*: it is the score of a rule of thumb that "
            "needs no model at all, namely that this hour usually looks like it did over the last "
            "{climatology} days. Above the line the model is worth running. Below it, the rule of "
            "thumb would have done better and the model is not paying for itself."
        ),
        math=(
            "`skill = 1 − RMSE(model) / RMSE(baseline)`. +0.30 means 30% less error than the rule of "
            "thumb. There is no floor on the negative side: −1.0 means twice the error."
        ),
        moves=(
            ("rises",
             "usually the dirty season arriving. There is more signal to find when the air is doing something"),
            ("drops below zero",
             "a month-old daily profile is beating a weather model that week"),
            ("falls steadily with no retrain",
             "staleness the trigger has not noticed. Cross-check it against the retrain chart"),
        ),
        next=(
            "Nothing about promoting a model can move this number, which is exactly the property the "
            "retrain trigger lacks. A trigger watching this instead was built, measured, and ships "
            "switched off."
        ),
    ),
    "decay": Guide(
        read=(
            "One line per model version, and the x-axis is weeks in service rather than a calendar "
            "date. That alignment is the whole point: it lays every model's first month on top of "
            "every other model's first month, so ageing can be compared between models that never "
            "served at the same time."
        ),
        math=(
            "Every version is scored on every window, including the ones it never served. That is a "
            "counterfactual — what this model *would* have done had it stayed — which the loop has "
            "no reason to compute while it is running."
        ),
        moves=(
            ("one line falls, the others do not",
             "that model aged badly. Something about when it was trained did not generalise"),
            ("all the lines fall together",
             "the world moved. No model would have held up, so this is not a retraining failure"),
            ("a line stays flat past twenty weeks",
             "the gate certified something durable. The calibration chart says that is rare"),
        ),
        next=(
            "Nothing re-examines a champion on fresh data while it is serving; it is only ever "
            "judged on the day it was promoted. A re-certification schedule was built, and bounded "
            "staleness without buying accuracy."
        ),
    ),
    "trigger": Guide(
        read=(
            "The solid line is the error of whichever model is in service. The dotted staircase is "
            "the bar that error has to cross to buy a retrain, and the markers are retrains firing. "
            "The staircase shape is the failure this whole project is built around, so it is worth "
            "a moment."
        ),
        math=(
            "The bar is `{perf}×` the model's own RMSE at the moment it was trained. A ratio rather "
            "than an absolute number, so one threshold covers Delhi at 127 µg/m³ and Melbourne at "
            "15 — and, fatally, it resets every time a model is promoted."
        ),
        moves=(
            ("the bar steps up",
             "a promotion just reset it, at whatever the dirty season was costing that week. It now "
             "takes more error than before to fire"),
            ("error climbs but never reaches the bar",
             "the alarm has ratcheted deaf. The model can get arbitrarily worse and nothing happens"),
            ("the bar steps down",
             "a challenger happened to train on a calmer stretch. It does happen, and never by "
             "enough to matter"),
        ),
        next=(
            "The denominator is the model's own past, which is what makes it ratchet. Any real fix "
            "has to be a yardstick that promoting a model cannot move — which is what the skill "
            "chart already is."
        ),
    ),
    "psi": Guide(
        read=(
            "One row or line per weather ingredient, one reading per week, showing how far that "
            "ingredient has drifted from the training window. Read the block rather than any single "
            "point: these cities go red early and stay red, which is all a statistic that saturates "
            "can honestly tell you."
        ),
        math=(
            "Population Stability Index. Bin both periods, then sum `(p − q) · ln(p / q)` across the "
            "bins. The bands are the industry convention: under {psi_stable} stable, up to "
            "{psi_significant} shifted, above that properly different."
        ),
        moves=(
            ("one ingredient crosses the threshold",
             "it no longer looks like anything the model was trained on"),
            ("they all cross at once",
             "the season turned. Expected, and not by itself a reason to retrain"),
            ("one stays flat all year",
             "that ingredient is not seasonal in this city. Surface pressure usually behaves this way"),
        ),
        next=(
            "PSI saturates: once a feature is over the line it stays there however much further the "
            "world moves, so this is a dependable yes-or-no and an undependable how-much. Computing "
            "it over the model's component space rather than per raw feature would restore the "
            "gradient."
        ),
    ),
    "factors": Guide(
        read=(
            "One panel per ingredient, in the units it is measured in, because *PSI 0.25* is not a "
            "picturable quantity and *fourteen degrees colder than anything the model saw* is. The "
            "line is that ingredient averaged over each {monitor}-day window; the band is where the "
            "same average sat while the first model was trained. A line outside its band is the "
            "model being asked about weather it has never seen."
        ),
        math=(
            "The band is the 10th–90th percentile of {monitor}-day rolling means over the training "
            "window — deliberately the same window length as the line, so both sides describe the "
            "same quantity and leaving the band means something."
        ),
        moves=(
            ("the line leaves the band and stays out",
             "sustained covariate drift. This is the case for retraining, before any statistic is computed"),
            ("it leaves and comes back",
             "a season passing. The model will be wrong for a while and then right again"),
            ("it stays inside the band while error rises",
             "the weather did not change, so the error is concept drift: the same weather now "
             "produces different air"),
        ),
        next=(
            "This is the readable version of the PSI chart — it carries direction and magnitude in "
            "physical units, which PSI throws away. The two disagreeing would be informative, and "
            "neither chart currently flags it."
        ),
    ),
    "distributions": Guide(
        read=(
            "The histogram PSI is a summary of. Training window against the most recent monitoring "
            "window, one ingredient at a time. PSI compresses each of these pairs into a single "
            "number, and this is what that number was computed from."
        ),
        math=(
            "The same bins PSI uses. Where the two histograms overlap, the term contributes nothing; "
            "where one has mass the other does not, it contributes a lot. A long tail present in one "
            "period and absent in the other is what drives the statistic."
        ),
        moves=(
            ("the two shapes sit on top of each other",
             "no drift in this ingredient, whatever the season is doing elsewhere"),
            ("one is shifted sideways",
             "the ordinary case: colder, calmer, wetter than training. PSI reports this well"),
            ("one is the same centre but a different width",
             "the case PSI reports poorly. The mean has not moved and the model is still seeing "
             "conditions it was never shown"),
        ),
        next=(
            "Only the latest window is drawn. An animation or a small multiple across the replay "
            "would show *when* a distribution moved, which is the question the retrain trigger "
            "actually needs answered."
        ),
    ),
    "holdout": Guide(
        read=(
            "Two series: the model in service and the challenger just trained, both scored on the "
            "same {holdout} days of air that neither has ever seen. Being newer is not a "
            "qualification — the challenger takes the job only by winning this."
        ),
        math=(
            "RMSE on the held-out week. Promotion needs "
            "`1 − rmse_challenger / rmse_champion > {margin}`, so winning by a nose is not enough."
        ),
        moves=(
            ("challenger well below champion",
             "a clear win. Whether it lasts is the calibration chart's question, not this one"),
            ("the two nearly equal",
             "the exam is deciding on noise. A week of hourly air cannot separate two models this close"),
            ("champion below challenger",
             "the gate working. The challenger was trained and thrown away, which is most of what "
             "the gate does"),
        ),
        next=(
            "The margin is a point estimate with no interval attached. Requiring the interval to "
            "clear the bar instead was built, and is *worse* — because the loop simply retries until "
            "something passes."
        ),
    ),
    "importance": Guide(
        read=(
            "How far the model's answer moves when each ingredient moves by one standard deviation, "
            "in µg/m³. This is the comparable version of the coefficients: a coefficient per °C and "
            "a coefficient per hPa cannot be ranked against each other, and these can."
        ),
        math=(
            "`|coefficient| × the feature's standard deviation` over the window. Magnitude only — "
            "the coefficient chart carries the sign."
        ),
        moves=(
            ("one bar dominates",
             "the model is close to a single-variable rule, whatever its eight inputs suggest"),
            ("the bars are roughly even",
             "it is genuinely combining ingredients"),
            ("the order changes after a retrain",
             "the new model believes something different about what drives the air here"),
        ),
        next=(
            "This is one version on one window. Tracking it across versions would separate retrains "
            "that change *what* the model believes from retrains that only change *how much*."
        ),
    ),
    "coefficients": Guide(
        read=(
            "One panel per ingredient, one point per model version, in the ingredient's own units. "
            "Each panel has its own scale on purpose: °C, hPa and W/m² on one axis leaves five of "
            "the six features pinned flat against the sixth."
        ),
        math=(
            "Coefficients recovered in original units by undoing the scaler, so they read as µg/m³ "
            "per °C rather than per standard deviation. This is also how old versions are rebuilt "
            "for scoring, without keeping a pickled model for each."
        ),
        moves=(
            ("a line crosses zero",
             "the model has changed its mind about direction: wind used to clear the air and now "
             "dirties it. The single strongest sign something structural moved"),
            ("it drifts steadily",
             "the relationship is moving with the season, which is what retraining is for"),
            ("it jumps at one version",
             "that challenger learned something quite different from its predecessor. Worth checking "
             "what it was trained through"),
        ),
        next=(
            "A sign flip is a loud signal and nothing in the loop watches for one. It is the most "
            "obvious candidate for a third retrain trigger, and it has not been built or measured."
        ),
    ),
    "alpha_sweep": Guide(
        read=(
            "Cross-validated error against the Ridge penalty, with the shipped setting marked. The "
            "curve is usually flat-bottomed over a wide range, which is the useful part: it says how "
            "much the exact value matters, not just which value wins."
        ),
        math=(
            "Forward-chaining cross-validation on the champion's own training window, never a random "
            "split. Hourly weather is autocorrelated, and a random split leaks the answer across the "
            "fold boundary badly enough to make any setting look fine."
        ),
        moves=(
            ("a flat bottom",
             "the penalty is not critical. Pick anything in the flat region and move on"),
            ("the shipped marker sits off the minimum",
             "the default is costing accuracy. It does here, and the ablation prices it: most of the "
             "headline retraining premium was this"),
            ("the optimum at an end of the swept range",
             "the grid is truncating the answer rather than containing it. Extend it and re-run"),
        ),
        next=(
            "The sweep runs on the bootstrap window only. Challengers train on a longer window and "
            "inherit a penalty chosen for a shorter one, which nothing currently checks."
        ),
    ),
    "benchmark": Guide(
        read=(
            "A table rather than a chart, because the question is a ranking and not a shape: does "
            "the model beat the things you could have done without one. Grouped by what each "
            "predictor is allowed to see, which is the only fair way to line them up. Lower is "
            "better. Read the gap to the nearest baseline rather than the model's own number, which "
            "has no scale."
        ),
        math=(
            "Median RMSE over the same {monitor}-day windows the charts use. The baselines get to "
            "see recent pollution readings and the model never does, so this is a hard bar rather "
            "than a fair fight — and it is the right bar, because those are the alternatives you "
            "could actually deploy."
        ),
        moves=(
            ("the model is below every baseline",
             "it is paying for itself. Check by how much before celebrating"),
            ("a baseline beats it",
             "in the clean season this is normal and still worth knowing: it puts a floor under what "
             "the loop can be worth"),
            ("the frozen model beats the served one",
             "retraining cost this city accuracy. That is a result, not a bug, and Los Angeles is "
             "here to produce it"),
        ),
        next=(
            "Every baseline here is a persistence or profile rule. A seasonal-climatology baseline "
            "would be a harder bar and would say more about whether the model has learned weather or "
            "merely learned the calendar."
        ),
    ),
    "fit_scatter": Guide(
        read=(
            "Predicted against measured for one run, with the diagonal as a perfect answer. A "
            "scatter rather than a line because the question is about the *spread* of the error, "
            "not its shape over time: how wrong the model is at each level of pollution, which the "
            "single RMSE above averages away."
        ),
        math=(
            "Each point is one scored hour. RMSE is the root of the mean squared vertical distance "
            "to the diagonal, so a handful of badly missed peaks move it more than a great many "
            "small misses."
        ),
        moves=(
            ("the cloud hugs the diagonal",
             "the model is tracking. Compare an early-summer run against a deep-autumn one to see this break"),
            ("it leans flatter than the diagonal",
             "the model under-calls high hours and over-calls low ones — regression toward the mean, "
             "and the usual failure of a squared-error fit on a skewed target"),
            ("it fans out at the right",
             "the error grows with the level of pollution, so the model is least reliable exactly "
             "when the answer matters most"),
        ),
        next=(
            "RMSE weights a missed episode and a quiet hour by the same rule. An error measure "
            "weighted toward the hours anyone would act on would say more about whether this is "
            "useful, and nothing here computes one."
        ),
    ),
    "residuals": Guide(
        read=(
            "What is left over after the prediction, hour by hour. The useful property of a "
            "residual plot is that a *good* one is boring: noise around zero with no pattern. Any "
            "structure visible here is signal the model failed to use."
        ),
        math=(
            "`residual = measured − predicted`. Positive means the model under-predicted that hour, "
            "which for pollution is the direction that matters."
        ),
        moves=(
            ("scattered around zero",
             "nothing left to extract. This is the good case"),
            ("drifting away from zero over the run",
             "bias, not noise: the model is systematically wrong in one direction and a retrain "
             "would recover it"),
            ("a repeating daily shape",
             "the hour-of-day encoding is not capturing the cycle, which is a modelling fault rather "
             "than a drift one"),
        ),
        next=(
            "Residuals are shown per run and never aggregated. Stacking them by hour of day across "
            "the replay would separate the daily-cycle failure from the seasonal one, which is a "
            "question this project asks repeatedly and answers by eye."
        ),
    ),
    # ------------------------------------------------------------- cross-city
    "map": Guide(
        read=(
            "One marker per city under watch. Colour is how far that city's weather has wandered "
            "from what its model was trained on; size is how long it has been watched. The point of "
            "the globe is that these are six continents: half these cities are dirtiest in June to "
            "August and half in December to January, which is how you can tell the thresholds are "
            "not secretly encoding a season."
        ),
        math=(
            "Colour is the worst per-feature PSI on the latest window, banded on the same convention "
            "as the drift chart. Size is the number of weeks replayed, not a measure of anything "
            "about the air."
        ),
        moves=(
            ("a marker turns red",
             "that city's incoming weather no longer resembles its training window"),
            ("red markers in both hemispheres at once",
             "opposite seasons drifting together, which is a property of the calendar rather than of "
             "the loop"),
            ("a large marker still green",
             "a long replay with no covariate drift. That is the control condition this study needs "
             "and rarely gets"),
        ),
        next=(
            "Six markers. The claims on this page about what does and does not harm a city are "
            "claims about these six, and the cheapest way to strengthen every one of them is a "
            "seventh."
        ),
    ),
    "pooled": Guide(
        read=(
            "Median error over the same monitoring windows, lower is better, three predictors per "
            "city: the loop's retrained champion, that same first model left frozen, and one Ridge "
            "trained across all six cities at once and never retrained."
        ),
        math=(
            "The pooled model gets a separate intercept per city, which is doing most of its work: "
            "mean pollution runs from 7 µg/m³ in Melbourne to 84 in Delhi, and the same model "
            "without that adjustment scores 43% worse."
        ),
        moves=(
            ("pooled below frozen",
             "training on five other cities is worth more than a year of staleness"),
            ("pooled above the served champion",
             "and worth less than keeping one city's model current. Both hold here"),
            ("pooled wins outright somewhere",
             "that city's weather-to-pollution relationship is close enough to the others to borrow. "
             "It does not happen here"),
        ),
        next=(
            "Six models is the arrangement that wins, but the margin is not large. A seventh city, "
            "especially one unlike these six, is what would test whether it keeps winning."
        ),
    ),
    "gate": Guide(
        read=(
            "One point per promotion. Across: the margin the challenger won its exam by. Up: what it "
            "actually delivered over the weeks it then served, measured against the model it "
            "displaced. A promise kept sits on the diagonal."
        ),
        math=(
            "The exam margin is the number the promotion decision was made *on*, so it cannot also "
            "be the evidence that the decision was right — that would be marking your own homework. "
            "The delivered margin is `1 − median(rmse_winner) / median(rmse_displaced)` over the "
            "winner's service windows, which is out of sample for the decision."
        ),
        moves=(
            ("points near the diagonal",
             "the exam is calibrated: it promises what it delivers"),
            ("points below the diagonal",
             "over-promising. The exam saw something that did not last"),
            ("short-serving above, long-serving below",
             "a certificate with a shelf life. That is what this chart shows, and it is about five weeks"),
        ),
        next=(
            "The shelf life is measurable and nothing in the loop uses it. Re-certifying at that "
            "cadence was built and measured: it bounds staleness and buys no accuracy, because it "
            "only exercises a gate whose calibration is the actual problem."
        ),
    ),
    "control_covariate": Guide(
        read=(
            "A made-up world with two dials, because a real city has no control condition — nobody "
            "knows what the right answer was supposed to be, so an alarm firing at random would look "
            "much like one that worked. This dial moves the weather and leaves the "
            "weather-to-pollution relationship alone. The weather alarm should climb; the error "
            "alarm should not."
        ),
        math=(
            "Both responses are indexed to their reading with the dial at zero rather than plotted "
            "on two axes. PSI and an error ratio share no unit, and a second axis would let the two "
            "curves be scaled into any story at all. One axis means a flat line is honestly flat."
        ),
        moves=(
            ("weather alarm climbs, error alarm flat at 1.0",
             "the pass. Each detector answers only to its own cause"),
            ("both climb",
             "the two alarms are not separable, and having two of them buys nothing"),
            ("neither moves",
             "the dial is not doing what it claims, so the experiment says nothing"),
        ),
        next=(
            "This is the one claim no real city can settle. Its cost is that the synthetic world is "
            "much simpler than Kraków, so it can show the detectors are separable without showing "
            "they are *sensitive enough*."
        ),
    ),
    "control_concept": Guide(
        read=(
            "The same made-up world, other dial. This one changes how weather turns into pollution "
            "and leaves the weather distributions alone: concept drift with no covariate drift. The "
            "error alarm should climb, and the weather alarm should not — it is right not to, "
            "because the weather genuinely has not changed."
        ),
        math=(
            "Indexed to the dial-at-zero reading, as in the panel beside it. Seasonal difference "
            "between the training and monitoring windows puts a real floor under PSI before either "
            "dial is touched, so the honest question is not *is it zero* but *does it move, and only "
            "for its own cause*."
        ),
        moves=(
            ("error alarm climbs, weather alarm flat",
             "the pass, and the more important half: this is the drift no label-free detector can see"),
            ("the weather alarm climbs too",
             "PSI is picking up something it should not, and the case for two independent signals "
             "weakens"),
        ),
        next=(
            "Concept drift is the expensive one to detect, because it needs labels and therefore "
            "needs waiting. Nothing here shortens that wait, and a leading indicator for it would be "
            "the single largest improvement available to this loop."
        ),
    ),
    "ablation": Guide(
        read=(
            "What retraining was worth in each city, under the model that ships and under two tuned "
            "properly on the same protocol. The error bars are the point of the chart rather than "
            "decoration on it: a bar crossing the zero line means retraining cannot be shown to have "
            "paid at all."
        ),
        math=(
            "Each premium is a paired week-by-week comparison against never retraining, with a "
            "moving-block bootstrap for the interval — blocks, because consecutive weekly windows "
            "overlap and are not independent observations."
        ),
        moves=(
            ("the premium shrinks as the model improves",
             "part of what looked like drift was really a straight line failing to fit a season"),
            ("it survives unchanged",
             "the effect is about the air, not about the model class"),
            ("it crosses zero under the better model",
             "the result does not survive, and that city's finding is withdrawn. It happens here to "
             "Los Angeles"),
        ),
        next=(
            "Two cities. Running the same three arms on the other four is the obvious next step, and "
            "the decomposition is a strong claim currently resting on one of them."
        ),
    ),
    "compare_city": Guide(
        read=(
            "One panel per city on shared axes, so the six can be compared by eye rather than by "
            "remembering the previous page. Shared axes are the whole reason this page exists: a "
            "chart auto-scaled per city makes Melbourne's 5-to-15 µg/m³ look like Delhi's 42-to-127."
        ),
        math="The same underlying series as the per-city page, redrawn on one scale. Nothing is recomputed.",
        moves=(
            ("one city is flat where the others swing",
             "its air has no season worth tracking, which is what makes it useful as a control"),
            ("two cities move together",
             "shared seasonality, not a shared cause. These are on six different continents"),
        ),
        next=(
            "Six cities chosen for contrast is a good design and a small sample. Every *cannot harm "
            "any city here* claim on this site is a claim about these six."
        ),
    ),
    "compare_value": Guide(
        read=(
            "Two bars per city, because one of them lies. The first compares median error across the "
            "whole replay against never retraining. The second holds the window fixed and compares "
            "the two models week by week, over the weeks a retrained model was actually serving. The "
            "error bars are on the second, because that is the one to read."
        ),
        math=(
            "The first figure is unpaired: in a city that promotes nothing until week 14 of 20, most "
            "windows compare the first model against itself and the answer collapses toward zero. "
            "Johannesburg reads 0.0% that way and +14.9% paired."
        ),
        moves=(
            ("the two bars disagree sharply",
             "the city promoted late or rarely. Trust the paired one"),
            ("the interval crosses zero",
             "not a finding, whatever the height of the bar. Two of these six are in that position"),
            ("a bar goes negative",
             "retraining measurably harmed that city. Los Angeles is here on purpose"),
        ),
        next=(
            "The paired figure conditions on the weeks the loop acted, which is conditioning on "
            "something the treatment caused. Both figures are published side by side for that "
            "reason, and the conditioning is an open question rather than a settled one."
        ),
    ),
}


def context(cfg, *, horizon_days: int | None, climatology_days: int, psi_bands: dict) -> dict:
    """The threshold values the guides quote, taken from the running config.

    Every one of these appears in prose somewhere above. Passing them in rather
    than importing the defaults means a profile with a non-default threshold
    gets a description of *its* loop.
    """
    return {
        "monitor": cfg.monitor_days,
        "holdout": cfg.holdout_days,
        "perf": cfg.perf_drift_threshold,
        "margin": cfg.promotion_margin,
        "climatology": climatology_days,
        "psi_stable": psi_bands["stable"],
        "psi_significant": psi_bands["significant"],
        "lead": f"{horizon_days} days" if horizon_days is not None else "a week",
    }


def payload(values: dict[str, object]) -> dict[str, dict]:
    """Every guide with its placeholders filled, for `site/data.json`."""
    return {key: guide.filled(values) for key, guide in GUIDES.items()}


def markdown(key: str, values: dict[str, object]) -> str | None:
    """One guide as a Markdown block, for Streamlit.

    ``None`` for an unknown key so a caller can skip the expander entirely
    rather than rendering an empty one.
    """
    guide = GUIDES.get(key)
    if guide is None:
        return None
    g = guide.filled(values)
    parts = [g["read"], f"**The maths.** {g['math']}"]
    if g["moves"]:
        parts.append("**What it means when it moves**")
        # A definition list is not Markdown, so this is the closest honest
        # rendering: the condition bolded, the reading after an em dash.
        parts.extend(f"- **{when}** — {then}" for when, then in g["moves"])
    parts.append(f"**What would improve this.** {g['next']}")
    return "\n\n".join(parts)
