"""How to read each chart, in one place, for both user interfaces.

The published site and the Streamlit app draw the same quantities and are
written in different languages, so a reading kept in each would be two readings
within a month. This module is the single copy. `scripts/build_site.py`
serialises it into `site/data.json` for the browser; `dashboard/app.py` imports
it directly.

## The four parts of a reading

``shows``  what the marks are, in a sentence or two
``data``   where the numbers come from and the exact period they cover
``units``  what each axis is measured in, in plain words
``read``   two or three "if you see this, it means that" pairs

A reading describes the chart. It does not report a result: these pages rebuild
from whatever the last run produced, so a sentence saying what a chart "shows
here" would be wrong the first time the data moved.

## Two kinds of placeholder

**Thresholds** such as ``{perf}`` and ``{monitor}`` are filled by
:func:`context` from the config the loop is running, so a reading cannot quote a
setting the loop no longer uses.

**City facts** such as ``{period}`` and ``{training}`` differ from one city to
the next. :func:`city_context` builds them. The dashboard fills them at once.
The site keeps them as tokens in `data.json` and fills them in the browser when
a city is picked, because one page serves all six.

The prose is Markdown, not HTML: Streamlit renders it directly, and the site
converts ``backticks`` and ``*emphasis*`` and escapes everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

# Charts that exist in only one of the two interfaces still live here. The set
# is the union, not the intersection, so moving a chart between the site and the
# dashboard does not mean rewriting its reading.
SITE_ONLY = frozenset({
    "forecast", "map", "pooled", "ablation", "compare_city", "compare_value",
})
DASHBOARD_ONLY = frozenset({
    "distributions", "alpha_sweep", "fit_scatter", "residuals",
})

#: The placeholders that change with the city, left for the site to fill.
CITY_TOKENS = ("city", "period", "runs", "latest", "training")

PM25 = "µg/m³ of PM2.5: micrograms of fine particles in a cubic metre of air"
RMSE = "RMSE in µg/m³: the typical size of an hourly miss, with large misses counting for more"
SOURCE = (
    "PM2.5 is the CAMS air-quality model's estimate, served by Open-Meteo. It is not a station "
    "reading."
)


class _Keep(dict):
    """A mapping that leaves an unknown placeholder as it was written."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class Guide:
    """One chart's reading. ``read`` is ordered: the most common case first."""

    shows: str
    data: str
    units: str
    read: tuple[tuple[str, str], ...]

    def filled(self, values: dict[str, object]) -> dict:
        """Substitute every placeholder ``values`` holds, and leave the rest."""
        keep = _Keep(values)
        fill = lambda s: s.format_map(keep)  # noqa: E731
        return {
            "shows": fill(self.shows),
            "data": fill(self.data),
            "units": fill(self.units),
            "read": [[fill(a), fill(b)] for a, b in self.read],
        }


GUIDES: dict[str, Guide] = {
    # ---------------------------------------------------------------- per city
    "forecast": Guide(
        shows=(
            "Hourly PM2.5 over the latest {monitor} days: the recorded value, and what the model "
            "in service predicted for that hour {lead} earlier. The gap between the two lines is "
            "the error every other chart summarises."
        ),
        data=(
            "{city}, {latest}. " + SOURCE + " Each prediction uses only the weather forecast "
            "issued {lead} before the hour it is for."
        ),
        units=PM25 + ". Times are UTC. The dotted line is the WHO 24-hour guideline, 15 µg/m³.",
        read=(
            ("the two lines move together", "the model has the shape of that fortnight right"),
            ("the prediction stays flat through a spike",
             "the weather forecast gave no warning of the episode"),
            ("the prediction sits under the peaks",
             "expected: a least-squares fit pulls its answers toward the average"),
        ),
    ),
    "skill": Guide(
        shows=(
            "How the model in service compares, week by week, with a rule of thumb that needs no "
            "model: this hour of the day usually looks as it did over the previous {climatology} "
            "days.{floor_note}"
        ),
        data=(
            "{city}, one point per weekly run, {period} ({runs} runs). Each point scores the "
            "{monitor} days before that run. The rule of thumb sees recent PM2.5 and the model "
            "never does."
        ),
        units=(
            "Skill has no unit: `1 − model error ÷ rule-of-thumb error`. 0 is a tie, +0.3 is 30% "
            "less error than the rule of thumb, −1 is twice its error."
        ),
        read=(
            ("above zero", "the model beat the rule of thumb that week"),
            ("below zero", "the rule of thumb would have done better"),
            ("falling for weeks with no retrain",
             "the model is going stale and the first alarm has not caught it"),
        ),
    ),
    "decay": Guide(
        shows=(
            "The skill of every model this city promoted, lined up by weeks since it took over "
            "and not by date, so models that never served together can be compared. Each line "
            "runs on past the model's retirement, which shows what keeping it would have cost."
        ),
        data=(
            "{city}, {period}. Every model version is scored on every weekly window, including "
            "the windows it never served."
        ),
        units=(
            "Across: weeks since the model took over. Up: skill, where 0 is a tie with the rule "
            "of thumb and higher is better."
        ),
        read=(
            ("one line falls and the others hold", "that model aged badly"),
            ("every line falls together", "the air changed, and no model would have held up"),
            ("a line stays level past twenty weeks", "a model that lasted, which is rare"),
        ),
    ),
    "trigger": Guide(
        shows=(
            "The error of the model in service (solid line), the bar that error must cross to "
            "set off a retrain (dotted), and the weeks a retrain fired (markers). The bar is "
            "{perf}× the serving model's own error when it was trained, so every promotion "
            "resets it."
        ),
        data=(
            "{city}, one point per weekly run, {period} ({runs} runs). Error is measured on the "
            "{monitor} days before each run."
        ),
        units=RMSE + ".",
        read=(
            ("the bar steps up",
             "a model was promoted during a dirty stretch, so it now takes more error to fire"),
            ("the error rises but stays under the bar",
             "this alarm cannot fire, however stale the model"),
            ("a marker below the bar",
             "the second alarm fired: the model had fallen too far behind the rule of thumb"),
        ),
    ),
    "psi": Guide(
        shows=(
            "How far each weather input has moved from the weather the serving model was trained "
            "on, one reading per input per week."
        ),
        data=(
            "{city}, {period} ({runs} runs). Each reading compares the {monitor} days before a "
            "run with the serving model's training window. The inputs are weather forecasts "
            "issued {lead} ahead."
        ),
        units=(
            "Population Stability Index, which has no unit: both periods are sorted into bins and "
            "the differences added up. Under {psi_stable} is stable, up to {psi_significant} "
            "shifted, above that different."
        ),
        read=(
            ("one input crosses a band", "that input no longer looks like the training weather"),
            ("every input crosses together",
             "the season turned, which is not by itself a reason to retrain"),
            ("an input stays in the top band for months",
             "the index tops out, so it says the weather differs and not by how much"),
        ),
    ),
    "factors": Guide(
        shows=(
            "Each weather input in its own units, averaged over every {monitor}-day window "
            "(line), against the range that same average covered while the first model was "
            "being trained (band)."
        ),
        data=(
            "{city}, {period}. The band is the middle 80% of {monitor}-day averages inside the "
            "first training window, {training}. Inputs are forecasts issued {lead} ahead."
        ),
        units=(
            "Temperature in °C, wind in m/s, humidity in %, precipitation in mm per hour, "
            "pressure in hPa, sunlight reaching the ground in W/m²."
        ),
        read=(
            ("the line leaves the band and stays out", "weather the first model never saw"),
            ("it leaves and comes back", "a season passing"),
            ("the line stays inside while error rises",
             "the weather is the same and the air is not, so the link between them changed"),
        ),
    ),
    "distributions": Guide(
        shows=(
            "For each weather input, how often each value occurred in the training window and in "
            "the latest monitoring window. These are the two shapes the drift index compares."
        ),
        data="{city}. The serving model's training window against the latest {monitor} days, {latest}.",
        units=(
            "Across: the input's own unit (°C, m/s, %, mm per hour, hPa, W/m²). Up: the share of "
            "hours at that value."
        ),
        read=(
            ("the two shapes overlap", "no drift in that input"),
            ("one is shifted sideways",
             "colder, calmer or wetter than training, which the index picks up well"),
            ("same centre, different width",
             "conditions the model was not shown, which the index picks up poorly"),
        ),
    ),
    "holdout": Guide(
        shows=(
            "Each time a retrain fired: the error of the model in service and of the rival just "
            "trained, on the same {holdout} days that neither had seen. The rival is promoted "
            "only if it wins by more than {margin_pct}."
        ),
        data="{city}, one pair of points per retrain, {period}.",
        units=RMSE + ", on the exam days. Lower is better. A star marks a promotion.",
        read=(
            ("the rival is well below the serving model",
             "a clear win on the exam; whether it lasts is a separate question"),
            ("the two are nearly level",
             "{holdout} days of hourly air cannot separate models this close"),
            ("the rival is above", "it was rejected, which is the usual outcome"),
        ),
    ),
    "importance": Guide(
        shows=(
            "How much the serving model's prediction moves when each input moves by a typical "
            "amount, one standard deviation."
        ),
        data="{city}, the model serving at the end of the replay, over {latest}.",
        units=(
            "µg/m³ of predicted PM2.5 per one-standard-deviation change in the input. Size only; "
            "the coefficient chart gives the direction."
        ),
        read=(
            ("one bar dominates", "the model is close to a one-input rule"),
            ("the bars are roughly even", "the model is combining its inputs"),
        ),
    ),
    "coefficients": Guide(
        shows=(
            "What each model version learned about each weather input: one panel per input, one "
            "point per version."
        ),
        data=(
            "{city}, every version trained during {period}, placed at the end of its training "
            "window."
        ),
        units=(
            "µg/m³ of PM2.5 per unit of the input: per °C, per m/s, per %, per mm, per hPa, per "
            "W/m². Each panel has its own scale."
        ),
        read=(
            ("a line crosses zero",
             "the model reversed its view of which way that input pushes the air"),
            ("a steady drift", "the link is moving with the season"),
            ("a jump at one version", "that model learned something its predecessor had not"),
        ),
    ),
    "alpha_sweep": Guide(
        shows=(
            "Cross-validated error at each strength of the Ridge penalty, with the best value and "
            "the library default marked. The loop makes this choice again at every fit."
        ),
        data=(
            "{city}, the first training window, {training}. Five folds, each trained on earlier "
            "hours and scored on later ones."
        ),
        units="Across: the penalty `alpha`, on a log scale, no unit. Up: " + RMSE + ".",
        read=(
            ("a flat bottom", "the exact value matters little"),
            ("the default sits above the lowest point", "the default would cost accuracy"),
            ("the best value is at the edge of the range", "the range tried is too narrow"),
        ),
    ),
    "benchmark": Guide(
        shows=(
            "The served model beside the alternatives that need no retraining, grouped by what "
            "each one is allowed to see."
        ),
        data=(
            "{city}, the median over {runs} weekly windows of {monitor} days, {period}. The "
            "second group sees past PM2.5; the model never does."
        ),
        units=RMSE + ". Lower is better.",
        read=(
            ("the served model is lowest in its group",
             "it beats what a weather forecast alone manages without retraining"),
            ("a predictor that sees past PM2.5 is lower",
             "knowing recent air beat a weather-only model, which is common in a clean season"),
            ("the never-retrained model is lower than the served one",
             "retraining cost accuracy over this replay"),
        ),
    ),
    "fit_scatter": Guide(
        shows=(
            "Predicted against recorded PM2.5 for one run, one point per hour. A perfect "
            "prediction sits on the diagonal."
        ),
        data="{city}, the {monitor} days before the run picked above.",
        units="Both axes are " + PM25 + ".",
        read=(
            ("the points hug the diagonal", "the model tracked that fortnight"),
            ("the cloud lies flatter than the diagonal",
             "dirty hours were under-called and clean hours over-called"),
            ("the cloud fans out to the right", "the error grows as the air gets dirtier"),
        ),
    ),
    "residuals": Guide(
        shows="Recorded minus predicted PM2.5, hour by hour, for one run.",
        data="{city}, the {monitor} days before the run picked above.",
        units="µg/m³. Above zero the model predicted too little.",
        read=(
            ("scattered around zero", "nothing systematic is left to fix"),
            ("drifting to one side", "a bias that a retrain would remove"),
            ("a pattern that repeats daily", "the model is missing part of the day's cycle"),
        ),
    ),
    # ------------------------------------------------------------- cross-city
    "map": Guide(
        shows=(
            "One marker per city. Colour is how far its latest weather has moved from the weather "
            "its model was trained on. Size is how many weeks it was replayed."
        ),
        data="All six cities, each at the last run of its own replay ({span}).",
        units=(
            "Colour: the largest drift index among the six weather inputs, banded at "
            "{psi_stable} and {psi_significant}. Size: weeks."
        ),
        read=(
            ("a red marker", "that city's latest weather no longer resembles its training weather"),
            ("red in both hemispheres",
             "opposite seasons at once, so the bands are not tied to one calendar"),
        ),
    ),
    "pooled": Guide(
        shows=(
            "Three ways to serve each city: its own model retrained by the loop, its own first "
            "model never retrained, and one model trained once on all six cities."
        ),
        data=(
            "The median over each city's weekly windows ({span}). The shared model is fitted on "
            "the six first training windows, with its own baseline level for each city."
        ),
        units=(
            RMSE + ". Lower is better. The cities differ several-fold in level, so compare bars "
            "within a city and not between cities."
        ),
        read=(
            ("the shared model is below the never-retrained one",
             "other cities' data is worth more than leaving one model to go stale"),
            ("the shared model is above the retrained one",
             "keeping a city's own model current is worth more still"),
        ),
    ),
    "gate": Guide(
        shows=(
            "One point per promotion, from all six cities. Across: the margin the rival won its "
            "exam by. Up: how much better it then was than the model it replaced, over the weeks "
            "it served."
        ),
        data=(
            "Every promotion in the six replays ({span}). The delivered margin compares the two "
            "models on the same weekly windows."
        ),
        units=(
            "Both axes are percent less RMSE. On the diagonal the exam's promise was kept. Below "
            "zero the promotion left the city worse off."
        ),
        read=(
            ("points near the diagonal", "the exam delivered what it promised"),
            ("points under the diagonal", "the exam promised more than the model delivered"),
            ("the long-serving points are lowest",
             "an exam passed once does not vouch for a model months later"),
        ),
    ),
    "control_covariate": Guide(
        shows=(
            "A simulated city, where the cause of the drift is known because it was set. This "
            "dial changes the weather and leaves alone how weather turns into pollution. The "
            "weather alarm should rise and the error alarm should not."
        ),
        data="Simulated hourly data, one full replay per dial setting. No real city is involved.",
        units=(
            "Across: the dial, where 0 is no drift. Up: each alarm's reading divided by its "
            "reading at 0, so 1.0 means unchanged."
        ),
        read=(
            ("the weather alarm rises and the error alarm stays at 1.0",
             "each alarm answers only to its own cause"),
            ("both rise", "the two alarms cannot be told apart"),
            ("neither moves", "the dial did nothing, and the test says nothing"),
        ),
    ),
    "control_concept": Guide(
        shows=(
            "The same simulated city, the other dial: the weather stays as it was and the way it "
            "turns into pollution changes. The error alarm should rise and the weather alarm "
            "should not."
        ),
        data="Simulated hourly data, one full replay per dial setting. No real city is involved.",
        units=(
            "Across: the dial, where 0 is no drift. Up: each alarm's reading divided by its "
            "reading at 0, so 1.0 means unchanged."
        ),
        read=(
            ("the error alarm rises and the weather alarm stays flat",
             "correct: this kind of drift cannot be seen until outcomes are recorded"),
            ("the weather alarm rises too", "it is reacting to something it should not"),
        ),
    ),
    "ablation": Guide(
        shows=(
            "What retraining was worth in two cities under three models: the Ridge as it ships, a "
            "Ridge tuned once, and a gradient-boosted tree tuned once."
        ),
        data=(
            "Delhi and Los Angeles, their full replays ({span}), with both retrain alarms on in "
            "every arm. Each value compares the loop with the same first model never retrained, "
            "over the weeks a retrained model was serving."
        ),
        units=(
            "Percent less RMSE than never retraining. The bars are 95% intervals, and one that "
            "crosses zero means no measurable effect."
        ),
        read=(
            ("the value shrinks under the tree",
             "part of the gain was a straight line being refitted to a season"),
            ("it stays clear of zero under the tree", "the gain survives a model that can bend"),
            ("the bar crosses zero", "no measurable effect in that city"),
        ),
    ),
    "compare_city": Guide(
        shows=(
            "The skill of each city's serving model, one panel per city, all on one vertical "
            "scale so the six can be compared by eye."
        ),
        data="Each city's own replay ({span}), one point per weekly run.",
        units=(
            "Skill has no unit: 0 is a tie with a rule of thumb built from the previous "
            "{climatology} days, and higher is better."
        ),
        read=(
            ("one city is flat where the others swing", "its air has little season to track"),
            ("two cities move together", "a shared season, not a shared cause"),
        ),
    ),
    "compare_value": Guide(
        shows=(
            "What retraining was worth in each city, measured two ways: across the whole replay, "
            "and week by week over the weeks a retrained model was serving."
        ),
        data="Each city's full replay ({span}), against its own first model never retrained.",
        units=(
            "Percent less RMSE than never retraining. The interval is drawn on the week-by-week "
            "bar, which is the one to read."
        ),
        read=(
            ("the two bars disagree",
             "the city promoted late or rarely, so trust the week-by-week bar"),
            ("the interval crosses zero", "no measurable effect, whatever the height of the bar"),
            ("a bar below zero", "retraining cost that city accuracy"),
        ),
    ),
}


def _day(stamp) -> str:
    """ "15 Aug 2025": unambiguous in any locale, and short enough for a sentence."""
    return f"{stamp.day} {stamp.strftime('%b %Y')}"


def replay_span(profiles) -> str:
    """The months the replays cover between them, as "Aug 2025 to Jul 2026"."""
    plans = [p.replay for p in profiles if getattr(p, "replay", None) is not None]
    if not plans:
        return "the replay"
    first = min(plan.first_run for plan in plans)
    last = max(plan.last_run for plan in plans)
    return f"{first.strftime('%b %Y')} to {last.strftime('%b %Y')}"


def context(cfg, *, horizon_days: int | None, climatology_days: int, psi_bands: dict,
            span: str = "the replay") -> dict:
    """The values every reading quotes, taken from the running config.

    Passing them in, not importing the defaults, means a profile with a
    non-default threshold gets a description of *its* loop.
    """
    floor = getattr(cfg, "skill_floor", None)
    return {
        "monitor": cfg.monitor_days,
        "holdout": cfg.holdout_days,
        "perf": cfg.perf_drift_threshold,
        "margin_pct": f"{cfg.promotion_margin:.0%}",
        "climatology": climatology_days,
        "psi_stable": psi_bands["stable"],
        "psi_significant": psi_bands["significant"],
        "lead": f"{horizon_days} days" if horizon_days is not None else "a week",
        "floor_note": (
            "" if floor is None
            else f" A second retrain alarm fires when skill drops below {floor:g}.".replace("-", "−")
        ),
        "span": span,
    }


def city_context(label: str, as_of, monitor_days: int, training=None) -> dict[str, str]:
    """The facts that change with the city: its name, its dates, its run count.

    ``as_of`` is the run dates in order. ``training`` is the first model's
    training window as ``(start, end)``, where it is known.
    """
    dates = list(as_of)
    if not dates:
        return {"city": label, "period": "no runs yet", "runs": "0",
                "latest": "no runs yet", "training": "not trained yet"}
    first, last = dates[0], dates[-1]
    window_start = last - timedelta(days=monitor_days)
    return {
        "city": label,
        "period": f"{_day(first)} to {_day(last)}",
        "runs": str(len(dates)),
        "latest": f"{_day(window_start)} to {_day(last)}",
        "training": (
            f"{_day(training[0])} to {_day(training[1])}" if training
            else "the first model's training window"
        ),
    }


def all_cities_context(span: str, monitor_days: int, runs: tuple[int, int]) -> dict[str, str]:
    """City facts for a page that draws every city at once. ``runs`` is the
    fewest and the most weekly runs any one city has."""
    return {
        "city": "Each city",
        "period": f"its own replay, {span}",
        "runs": f"{runs[0]} to {runs[1]}",
        "latest": f"its last {monitor_days} days",
        "training": "its own first training window",
    }


def payload(values: dict[str, object]) -> dict[str, dict]:
    """Every reading for `site/data.json`, with whatever ``values`` holds filled in."""
    return {key: guide.filled(values) for key, guide in GUIDES.items()}


def markdown(key: str, values: dict[str, object]) -> str | None:
    """One reading as a Markdown block, for Streamlit.

    ``None`` for an unknown key so a caller can skip the expander entirely.
    """
    guide = GUIDES.get(key)
    if guide is None:
        return None
    g = guide.filled(values)
    parts = [g["shows"], f"**Data.** {g['data']}", f"**Units.** {g['units']}", "**Reading it**"]
    # A definition list is not Markdown, so this is the closest honest
    # rendering: the condition bolded, the meaning after an em dash.
    parts.append("\n".join(f"- **{when}** — {then}" for when, then in g["read"]))
    return "\n\n".join(parts)
