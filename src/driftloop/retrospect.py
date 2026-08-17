"""Scoring every model version on every window, after the fact.

The loop logs what *it* decided at the time: the champion in service, its RMSE,
and the ratio that drove the retrain. That is enough to run the loop and not
enough to judge it, because three questions need models scored on windows they
never served:

1. **Does an individual model decay?** The logged ``champion_rmse`` is one line
   across eight different champions, so no single model's decay is visible in
   it. Scoring one version across every window separates the models.
2. **Is the model any good?** RMSE has no scale. A number is only a verdict
   against an alternative you could have deployed instead.
3. **Did the promotion gate work, or did it overfit a seven-day exam?** The gate
   saw one holdout margin. Whether that margin was *delivered* over the
   challenger's service life is a different number, and it needs the replaced
   champion scored on windows it never served.

None of this needs refitting. ``log_and_register`` writes every version's
coefficients as tags in original feature units, so a version is reconstructable
from the registry alone as ``intercept + sum(coef_i * x_i)``. That is exact to
the six decimals the tags are written with, which ``tests/test_retrospect.py``
asserts against the loop's own logged RMSE.

## The skill baseline

``skill = 1 - rmse_model / rmse_climatology``: positive means the model beat the
baseline, zero means it matched it, negative means it lost.

The baseline is an hour-of-day mean over the ``CLIMATOLOGY_DAYS`` before the
window, and is not derived from the champion's training window. One built from
the champion's own training data would inherit its staleness: a winter-trained
climatology also predicts high in July, so it could not measure staleness at
all. The baseline has to be independent of the model it judges.

The same causality rule applies as everywhere else here. The reference period
ends ``forecast_lead_days`` before the window starts, so every hour it averages
was observable when the forecast for the window's first hour was issued. It does
see recent PM2.5, which the model never does; that asymmetry is stated wherever
the number is published. The justification is that it is the alternative that
could actually be deployed: if a month-old daily profile beats the model, the
model is not paying for itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from driftloop import stats
from driftloop.config import DRIFT_FEATURES, FEATURES, TARGET, TIMESTAMP
from driftloop.data.base import DataSource

# How much history the climatology baseline averages over. A month is long
# enough to average out weather and short enough to still be the current season,
# which is the property that keeps this baseline honest across a full year.
CLIMATOLOGY_DAYS = 30

# Where a promotion stops counting as short-serving. The seven-day exam holds
# its calibration for roughly five weeks and reverses sign past this, so the
# split follows the evidence rather than a round number. Defined here and
# published through data.json, because both UIs draw it and a threshold kept by
# hand in two languages ends up telling two stories.
GATE_LONG_WEEKS = 20


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


@dataclass(frozen=True)
class RegisteredModel:
    """A registered version, rebuilt from its coefficient tags.

    The tags hold slopes in *original* units (see ``model.effective_coefficients``),
    so prediction is a plain dot product and needs no scaler.
    """

    version: int
    coefficients: dict[str, float]
    intercept: float
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    baseline_rmse: float

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), self.intercept, dtype=float)
        for feature, coef in self.coefficients.items():
            out += coef * df[feature].to_numpy(dtype=float)
        return out

    def importance(self, reference: pd.DataFrame) -> dict[str, float]:
        """|slope| x the feature's spread: µg/m³ of prediction per 1-sd move.

        Raw slopes are per original unit and not comparable to each other, since
        a slope per hPa and one per W/m² answer different questions. Scaling each
        by the feature's own standard deviation over a reference window puts them
        in µg/m³ and makes them comparable.
        """
        return {
            feature: abs(coef) * float(np.std(reference[feature].to_numpy(dtype=float)))
            for feature, coef in self.coefficients.items()
        }


@dataclass(frozen=True)
class ArtifactModel:
    """A registered version loaded from its logged artifact rather than its tags.

    The fallback for anything that is not a straight line. A tree has no slopes
    to write into the registry, so scoring one means unpickling it: slower, and
    it needs the artifact store to still be reachable.

    Same surface as ``RegisteredModel`` so ``build`` cannot tell them apart.
    """

    version: int
    pipeline: object
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    baseline_rmse: float

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.predict(df[FEATURES]), dtype=float)

    def importance(self, reference: pd.DataFrame) -> dict[str, float]:
        """Not available. A tree has no per-feature slope to scale."""
        return {}


def registered_models(client, model_name: str, load_artifacts: bool = False):
    """Rebuild every registered version so it can be scored on any window.

    The fast path reads coefficient tags: exact, and no artifact store needed.
    Versions whose tags predate coefficient logging are skipped rather than
    guessed at.

    ``load_artifacts`` turns on the slow path for versions with no coefficients,
    unpickling each instead. Off by default: it is slower, it needs artifact
    paths a backend built elsewhere will not have, and for the shipped Ridge it
    only reproduces what the tags already say. The ablation is why it exists.
    """
    out: dict[int, RegisteredModel | ArtifactModel] = {}
    unscoreable: list[int] = []
    for mv in client.search_model_versions(f"name='{model_name}'"):
        tags = mv.tags or {}
        version = int(mv.version)
        common = {
            "train_start": pd.to_datetime(tags.get("train_start")),
            "train_end": pd.to_datetime(tags.get("train_end")),
            "baseline_rmse": float(tags.get("baseline_rmse", "nan")),
        }
        if "coef_intercept" in tags and all(f"coef_{f}" in tags for f in FEATURES):
            out[version] = RegisteredModel(
                version=version,
                coefficients={f: float(tags[f"coef_{f}"]) for f in FEATURES},
                intercept=float(tags["coef_intercept"]),
                **common,
            )
        elif load_artifacts:
            import mlflow

            out[version] = ArtifactModel(
                version=version,
                pipeline=mlflow.sklearn.load_model(f"models:/{model_name}/{version}"),
                **common,
            )
        else:
            unscoreable.append(version)
    if unscoreable and not out:
        raise ValueError(
            f"no version of {model_name!r} carries coefficient tags "
            f"({len(unscoreable)} skipped). Pass load_artifacts=True to score a "
            "non-linear model from its logged artifact."
        )
    return dict(sorted(out.items()))


def climatology_prediction(
    source: DataSource,
    window: pd.DataFrame,
    window_start: pd.Timestamp,
    lead_days: int,
    days: int = CLIMATOLOGY_DAYS,
) -> np.ndarray:
    """Hour-of-day means from the period ending ``lead_days`` before the window.

    Ending the reference early is what makes this usable as a baseline at all: a
    forecast for the window's first hour was issued ``lead_days`` earlier, so
    anything later than that was not available to whoever issued it.
    """
    reference_end = window_start - pd.Timedelta(lead_days, unit="D")
    reference = source.get_data(reference_end - pd.Timedelta(days, unit="D"), reference_end)
    if reference.empty:
        return np.full(len(window), np.nan)
    by_hour = reference.groupby(pd.to_datetime(reference[TIMESTAMP]).dt.hour)[TARGET].mean()
    hours = pd.to_datetime(window[TIMESTAMP]).dt.hour
    # A reference month missing an hour entirely leaves NaN, which _rmse drops
    # rather than silently scoring that hour against zero.
    return hours.map(by_hour).to_numpy(dtype=float)


def climatology_skill(
    source: DataSource,
    window: pd.DataFrame,
    window_start: pd.Timestamp,
    lead_days: int,
    model_rmse: float,
    days: int = CLIMATOLOGY_DAYS,
) -> float:
    """``1 - model_rmse / climatology_rmse`` on one window.

    Lives here rather than in ``loop`` so the live retrain trigger and the
    after-the-fact analysis answer to the same baseline under the same causality
    rule. Two implementations would drift apart and leave the trigger and the
    chart disagreeing about one model.

    NaN where the baseline is unavailable, which callers must treat as "no
    opinion" rather than as a failing model.
    """
    actual = window[TARGET].to_numpy(dtype=float)
    reference = _rmse(actual, climatology_prediction(source, window, window_start, lead_days, days))
    if not reference or np.isnan(reference) or np.isnan(model_rmse):
        return float("nan")
    return 1.0 - model_rmse / reference


def training_window_stats(
    source: DataSource, model: RegisteredModel, window_days: int
) -> dict[str, dict[str, float]]:
    """Each feature's central range over the window a model was trained on,
    measured the same way as the series drawn in front of it.

    Drawn behind the feature series as a band: the band is what the model was
    shown, the line is what the world did afterwards, and a line leaving its band
    is the physical form of the covariate-drift claim.

    ``window_days`` is the monitor window the line averages over, and passing the
    right one is what makes the chart readable. The band is the 10th-90th
    percentile of ``window_days``-long rolling means, so both sides describe the
    same quantity. Percentiles of the raw hourly values answer a different
    question -- how far one *hour* strays -- and radiation shows what that costs:
    it runs from zero to full sun every day, so its hourly band was wide enough
    that no fortnightly mean could ever leave it, and the chart shipped with a
    caption telling the reader not to trust it.

    The rolling windows overlap, so this describes the training window rather
    than sampling from it. That is the object wanted here: the question is what
    range a fortnightly mean held while the model was learning, not how uncertain
    that range is.
    """
    window = source.get_data(model.train_start, model.train_end + pd.Timedelta(1, unit="h"))
    stats: dict[str, dict[str, float]] = {}
    if window.empty:
        return stats
    indexed = window.set_index(TIMESTAMP).sort_index()
    # Only fully-formed windows. A partial one averages fewer days than the line
    # does and lands wherever the training window happened to start.
    full_from = indexed.index[0] + pd.Timedelta(window_days, unit="D")
    for feature in DRIFT_FEATURES:
        if feature not in indexed:
            continue
        hourly = indexed[feature].astype(float)
        rolling = hourly.rolling(f"{window_days}D").mean().loc[full_from:]
        # A training window shorter than one monitor window holds no comparable
        # mean at all. Collapse the band onto the one value that is honest there
        # rather than widening it with partial windows.
        values = rolling.to_numpy() if len(rolling) else hourly.to_numpy()
        stats[feature] = {
            "mean": float(np.mean(hourly.to_numpy())),
            "lo": float(np.percentile(values, 10)),
            "hi": float(np.percentile(values, 90)),
        }
    return stats


@dataclass
class Retrospective:
    """Every version scored on every window, plus what that says about the loop."""

    as_of: list[pd.Timestamp] = field(default_factory=list)
    climatology_rmse: list[float] = field(default_factory=list)
    champion_version: list[int] = field(default_factory=list)
    # Which version served each window, which is not always the one tagged on
    # the run: a run that promotes monitors with the outgoing champion and then
    # writes the winner onto the tag. `champion_version` keeps the tag, which is
    # correct for "when did this model start"; this answers "what was serving".
    serving_version: list[int] = field(default_factory=list)
    champion_rmse: list[float] = field(default_factory=list)
    champion_skill: list[float] = field(default_factory=list)
    # Per-window mean of each weather feature, and of the target: the covariate
    # story in the units the features are measured in (°C, m/s, hPa) rather than
    # as a PSI, whose scale is not readable.
    feature_means: dict[str, list[float]] = field(default_factory=dict)
    target_mean: list[float] = field(default_factory=list)
    # version -> its RMSE on every window, including ones it never served, which
    # is what turns the single champion line into one decay curve per model.
    version_rmse: dict[int, list[float]] = field(default_factory=dict)
    version_skill: dict[int, list[float]] = field(default_factory=dict)
    promoted_at: dict[int, pd.Timestamp] = field(default_factory=dict)
    gate: list[dict] = field(default_factory=list)

    def skill_of(self, version: int) -> list[float]:
        return self.version_skill.get(version, [])


def retraining_value(result: Retrospective) -> dict[str, float | int]:
    """What retraining was worth, measured two ways because they answer differently.

    ``across_replay`` compares the median error of the champion the loop served
    against the median of the first champion held frozen, over every window. It
    is the product measure: what the whole arrangement delivered, including the
    weeks before the loop had retrained anything.

    ``when_it_acted`` compares the two models *window by window* and takes the
    median of the per-window ratio, over the windows where a retrained model was
    actually serving. It answers the different question of whether retraining
    helped when it happened.

    Both are published because the first misleads on its own. Comparing one
    median against another is unpaired, so where both distributions are dominated
    by the same seasonal swing the comparison mostly measures the season.
    Johannesburg promotes nothing until run 14 of 20, so 70% of its windows have
    the two models identical, both medians land on the same value, and the
    headline reads 0.0% while every window a retrained model served improved.
    """
    arrays = retraining_series(result)
    if not arrays:
        return {}
    served, frozen = arrays["served"], arrays["frozen"]

    out: dict[str, float | int] = {
        "across_replay": float((1 - np.median(served) / np.median(frozen)) * 100),
        "windows": int(served.size),
    }

    served_acted, frozen_acted = arrays["served_acted"], arrays["frozen_acted"]
    out["acted_windows"] = int(served_acted.size)
    if served_acted.size:
        ratio = served_acted / frozen_acted
        out["when_it_acted"] = float((1 - np.median(ratio)) * 100)
        out["win_rate"] = float((served_acted < frozen_acted).mean() * 100)
    return out


def retraining_series(result: Retrospective) -> dict[str, np.ndarray]:
    """The paired per-window error series the retraining headlines are computed from.

    Split out from ``retraining_value`` so the uncertainty on those headlines is
    resampled from the same numbers they were computed on. A bootstrap that
    reconstructs its own input resamples a different statistic, and the
    discrepancy surfaces as an interval that excludes its own point estimate.

    Returns ``served``/``frozen`` over every usable window, and
    ``served_acted``/``frozen_acted`` over the windows a retrained model was in
    service for. Empty dict when there is nothing to compare.
    """
    frozen_version = min(result.version_rmse) if result.version_rmse else None
    if frozen_version is None or not result.champion_rmse:
        return {}

    served_all = np.asarray(result.champion_rmse, dtype=float)
    frozen_all = np.asarray(result.version_rmse[frozen_version], dtype=float)
    usable = ~(np.isnan(served_all) | np.isnan(frozen_all))
    if not usable.any():
        return {}

    # Windows where a retrained model was serving, so the comparison is about
    # retraining rather than a model against itself.
    #
    # Keyed on the version in service, not on whether the two error values
    # differ. The served figure is the loop's logged error and the frozen one is
    # reconstructed from coefficient tags, so for the same model they agree only
    # to the six decimals the tags carry. An equality test therefore counted
    # every window as retrained, pulling Johannesburg's paired result from +20%
    # to zero on thirteen windows where nothing had been retrained yet.
    serving = np.asarray(result.serving_version or result.champion_version, dtype=int)
    acted = usable & (serving != frozen_version)

    as_of = np.asarray([str(d) for d in result.as_of])
    return {
        "served": served_all[usable],
        "frozen": frozen_all[usable],
        "as_of": as_of[usable],
        "served_acted": served_all[acted],
        "frozen_acted": frozen_all[acted],
        # The run dates behind the acted arrays, so two replays of one city under
        # different settings are compared on the weeks they share rather than by
        # position. Different settings promote at different times, so the two
        # acted sets differ in length and index alignment would pair up
        # unrelated weeks.
        "acted_as_of": as_of[acted],
    }


def retraining_uncertainty(
    result: Retrospective,
    block: int | None = None,
    resamples: int = stats.DEFAULT_RESAMPLES,
    seed: int = stats.DEFAULT_SEED,
) -> dict[str, stats.Interval]:
    """Block-bootstrap intervals for the three retraining headlines.

    Consecutive windows overlap by half and the weather underneath them is
    autocorrelated, so the resampling is over contiguous blocks of weeks rather
    than over individual weeks. See ``driftloop.stats`` for why, and for what
    the interval is and is not entitled to claim.

    The paired figures resample ``served`` and ``frozen`` with the *same* block
    indices, so a week's two error values are never separated.
    """
    arrays = retraining_series(result)
    if not arrays:
        return {}

    kwargs = {"block": block, "resamples": resamples, "seed": seed}
    out = {
        "across_replay": stats.block_bootstrap(
            (arrays["served"], arrays["frozen"]), stats.pct_improvement_unpaired, **kwargs
        )
    }
    if arrays["served_acted"].size:
        paired = (arrays["served_acted"], arrays["frozen_acted"])
        out["when_it_acted"] = stats.block_bootstrap(paired, stats.pct_improvement_paired, **kwargs)
        out["win_rate"] = stats.block_bootstrap(paired, stats.win_rate, **kwargs)
    return out


def gate_uncertainty(
    gate: list[dict],
    long_weeks: int = GATE_LONG_WEEKS,
    resamples: int = stats.DEFAULT_RESAMPLES,
    seed: int = stats.DEFAULT_SEED,
) -> dict[str, dict[str, stats.Interval]]:
    """Intervals on the gate-calibration means, split short-serving vs long.

    Resampled IID (block length 1) rather than in blocks, unlike the retraining
    figures. These rows are one per promotion, not one per week: irregularly
    spaced, pooled across six cities, and consecutive promotions do not overlap
    the way consecutive monitor windows do. Blocking would impose a serial
    structure the rows do not have.

    The ``long`` group carries the project's central claim on three promotions
    across six cities, and the interval width says so.
    """
    out: dict[str, dict[str, stats.Interval]] = {}
    for label, rows in (
        ("short", [g for g in gate if g["weeks_served"] < long_weeks]),
        ("long", [g for g in gate if g["weeks_served"] >= long_weeks]),
    ):
        if not rows:
            out[label] = {}
            continue
        exam = np.array([g["exam_margin"] for g in rows], dtype=float) * 100
        delivered = np.array([g["delivered_margin"] for g in rows], dtype=float) * 100
        kwargs = {"block": 1, "resamples": resamples, "seed": seed}
        out[label] = {
            "exam": stats.block_bootstrap((exam,), stats.mean_stat, **kwargs),
            "delivered": stats.block_bootstrap((delivered,), stats.mean_stat, **kwargs),
        }
    return out


def gate_summary(gate: list[dict], long_weeks: int = GATE_LONG_WEEKS) -> dict[str, dict[str, float]]:
    """The gate's calibration, split by how long the winner went on to serve.

    Two groups rather than a correlation. The claim is not that the exam is
    noisy but that it has a shelf life: honest over the horizon it tests, and
    sign-reversed beyond it. A single correlation would average the two regimes
    into one unremarkable number.

    ``harmful`` counts promotions that delivered a negative margin, where the
    winner was worse over its service life than the model it displaced. That is
    the failure mode the gate exists to prevent, so it is counted rather than
    read off a scatter.
    """
    out: dict[str, dict[str, float]] = {}
    for label, rows in (
        ("short", [g for g in gate if g["weeks_served"] < long_weeks]),
        ("long", [g for g in gate if g["weeks_served"] >= long_weeks]),
    ):
        if not rows:
            out[label] = {"n": 0}
            continue
        out[label] = {
            "n": len(rows),
            "exam": float(np.mean([g["exam_margin"] for g in rows])),
            "delivered": float(np.mean([g["delivered_margin"] for g in rows])),
            "harmful": int(sum(1 for g in rows if g["delivered_margin"] < 0)),
        }
    return out


def _service_spans(versions: list[int]) -> list[tuple[int, int, int]]:
    """Every contiguous stretch a version was the serving champion, in order.

    ``(version, first_index, last_index)`` per stretch, so a version that serves
    twice appears twice. That is not hypothetical: a rollback restores an earlier
    version, and keying spans by version instead would record it as having served
    the interval in between, which belongs to the model that replaced it.

    The consequences of getting this wrong are quiet. ``weeks_served`` would
    count someone else's weeks, the gate calibration would score a promotion over
    windows it did not serve, and the split between short- and long-serving
    promotions would move models across it.
    """
    stretches: list[tuple[int, int, int]] = []
    for i, version in enumerate(versions):
        if stretches and stretches[-1][0] == version and stretches[-1][2] == i - 1:
            first = stretches[-1][1]
            stretches[-1] = (version, first, i)
        else:
            stretches.append((version, i, i))
    return stretches


def build(
    source: DataSource,
    runs: pd.DataFrame,
    models: dict[int, RegisteredModel],
    monitor_days: int,
    lead_days: int,
    promotion_margin: float,
) -> Retrospective:
    """Score every reconstructable version on every monitoring window.

    ``runs`` is the loop's own monitor-cycle log, one row per scheduled run, with
    ``as_of`` and ``champion_version`` columns plus the holdout metrics the gate
    calibration reads. Everything else is derived from it, so this cannot
    disagree with what the loop did; it only adds what the loop had no reason to
    compute at the time.
    """
    result = Retrospective()
    if runs.empty or not models:
        return result

    windows: list[pd.DataFrame] = []
    result.feature_means = {f: [] for f in DRIFT_FEATURES}
    for as_of in runs["as_of"]:
        start = as_of - pd.Timedelta(monitor_days, unit="D")
        window = source.get_data(start, as_of)
        windows.append(window)
        result.as_of.append(as_of)
        actual = window[TARGET].to_numpy(dtype=float)
        climatology = climatology_prediction(source, window, start, lead_days)
        result.climatology_rmse.append(_rmse(actual, climatology))
        result.target_mean.append(float(np.mean(actual)) if len(actual) else float("nan"))
        for feature in DRIFT_FEATURES:
            values = window[feature].to_numpy(dtype=float)
            result.feature_means[feature].append(float(np.mean(values)) if len(values) else float("nan"))

    for version, model in models.items():
        rmses, skills = [], []
        for window, climatology_rmse in zip(windows, result.climatology_rmse):
            actual = window[TARGET].to_numpy(dtype=float)
            value = _rmse(actual, model.predict(window))
            rmses.append(value)
            skills.append(
                float("nan")
                if not climatology_rmse or np.isnan(climatology_rmse) or np.isnan(value)
                else 1.0 - value / climatology_rmse
            )
        result.version_rmse[version] = rmses
        result.version_skill[version] = skills

    champion_versions = [int(v) for v in runs["champion_version"]]
    result.champion_version = champion_versions

    # Whether each run promoted. The loop tags every monitor run with its
    # decision; where the tag is missing, fall back to the champion tag changing
    # between two runs, which only a promotion does. That fallback cannot see a
    # first run that promoted, since there is no earlier tag to differ from, so
    # it degrades to the pre-tag behaviour rather than to something wrong.
    if "tags.promotion_decision" in runs:
        promoted = [str(v) == "promoted" for v in runs["tags.promotion_decision"]]
    else:
        promoted = [i > 0 and v != champion_versions[i - 1] for i, v in enumerate(champion_versions)]
    # What was serving before the first run is not in the run log: the bootstrap
    # champion is registered before any monitor cycle exists. So on a first run
    # that promotes, which Kraków and Los Angeles both do, there is no earlier
    # row to read the outgoing version off, and the tag credits the incoming
    # model with a window the bootstrap served. The bootstrap is the lowest
    # registered version, which is the same model `retraining_value` freezes.
    bootstrap_version = min(models)
    result.serving_version = [
        (champion_versions[i - 1] if i > 0 else bootstrap_version) if promoted[i] else v
        for i, v in enumerate(champion_versions)
    ]

    # The champion series comes from the loop's own logged error rather than
    # this module's reconstruction. The two disagree on promotion runs, and the
    # logged one is right: the run monitored with the old champion before
    # overwriting the tag, so reading the tag credits the new model with a window
    # it never served, on data it trained on. Delhi scored +44.8% for retraining
    # that way against the benchmark's +43.8%.
    #
    # The per-version curves keep using the tag, which is correct for them: a
    # decay curve should start where a model was promoted.
    logged = runs["metrics.champion_rmse"] if "metrics.champion_rmse" in runs else None
    for i, version in enumerate(champion_versions):
        fallback = result.version_rmse.get(version, [float("nan")] * len(windows))[i]
        value = float(logged.iloc[i]) if logged is not None and not pd.isna(logged.iloc[i]) else fallback
        climatology = result.climatology_rmse[i]
        result.champion_rmse.append(value)
        result.champion_skill.append(
            float("nan")
            if not climatology or np.isnan(climatology) or np.isnan(value)
            else 1.0 - value / climatology
        )

    spans = _service_spans(champion_versions)
    # Where each version's decay curve begins: the first window it served, even
    # if it later lost its place and came back.
    for version, first, _ in spans:
        result.promoted_at.setdefault(version, result.as_of[first])

    result.gate = _gate_calibration(
        result, runs, spans, champion_versions, promoted, bootstrap_version, promotion_margin
    )
    return result


def _gate_calibration(
    result: Retrospective,
    runs: pd.DataFrame,
    spans: list[tuple[int, int, int]],
    champion_versions: list[int],
    promoted: list[bool],
    bootstrap_version: int,
    promotion_margin: float,
) -> list[dict]:
    """What the seven-day exam promised against what the winner went on to do.

    The gate promotes on a single holdout window, so that margin is in-sample
    for the decision. The out-of-sample check is what the new champion delivered
    over its service life against the champion it replaced, scored on those same
    windows: the counterfactual the loop had no reason to compute.

    A working gate shows the two rising together. A gate overfitting a short
    exam shows exam margins scattered against delivered ones.

    One row per *stretch of service*, not per version. A promotion that is later
    rolled back is judged over the weeks it actually served, and the restored
    version's own stretch is not a promotion and is skipped.
    """
    out: list[dict] = []
    for version, first, last in spans:
        # A stretch begins where the champion tag changed, which happens on a
        # promotion, on a rollback, and at index 0 where the tag simply starts.
        # Asking the run whether it promoted separates the three, and is what
        # lets a first run that did promote be judged, against the bootstrap
        # rather than against a row that does not exist. Without it Kraków and
        # Los Angeles each lost a promotion, and `champion_versions[first - 1]`
        # at first == 0 would have indexed the end of the list.
        if not promoted[first]:
            continue
        replaced = champion_versions[first - 1] if first > 0 else bootstrap_version
        if replaced == version or version not in result.version_rmse or replaced not in result.version_rmse:
            continue

        row = runs.iloc[first]
        champion_holdout = row.get("metrics.champion_rmse_holdout")
        challenger_holdout = row.get("metrics.challenger_rmse")
        if champion_holdout is None or pd.isna(champion_holdout) or pd.isna(challenger_holdout):
            continue

        # From the run after the promotion. The monitor window at the promotion
        # itself is [as_of-14d, as_of), whose first half is inside the window the
        # challenger trained on. One step later the window is [as_of-7d,
        # as_of+7d), which the challenger never trained on.
        #
        # At least one window even for a champion replaced the following week:
        # scoring it one step past its retirement is a counterfactual rather than
        # a leak, and dropping those promotions would bias the calibration toward
        # the long-serving models that are the interesting failures.
        served = slice(first + 1, max(last + 1, first + 2))
        new_rmse = [v for v in result.version_rmse[version][served] if not np.isnan(v)]
        old_rmse = [v for v in result.version_rmse[replaced][served] if not np.isnan(v)]
        if not new_rmse or not old_rmse:
            continue

        new_median, old_median = float(np.median(new_rmse)), float(np.median(old_rmse))
        out.append(
            {
                "version": version,
                "replaced": replaced,
                "as_of": result.as_of[first],
                # What the gate saw: the challenger's edge on the holdout exam.
                "exam_margin": float(1.0 - challenger_holdout / champion_holdout)
                if champion_holdout
                else float("nan"),
                # What it delivered: the same edge over the windows it served,
                # against the displaced model scored on those windows too.
                "delivered_margin": float(1.0 - new_median / old_median) if old_median else float("nan"),
                "weeks_served": last - first + 1,
                "required_margin": promotion_margin,
            }
        )
    return out
