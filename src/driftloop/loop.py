"""The scheduled run: detect -> maybe retrain -> maybe promote, all logged.

Window layout for one run at time ``as_of`` (half-open windows)::

    ...........[==== challenger train ====][= holdout =] as_of
                        [====== monitor window ========]
    [== champion train ==]  (much earlier, never overlaps holdout)

- **monitor** drives both drift signals (data drift vs. the champion's training
  distribution, and the champion's current RMSE vs. its baseline).
- **holdout** is the judge. It is excluded from the challenger's training data
  and post-dates the champion's, so neither model has seen it. ``run_cycle``
  raises rather than promote on a holdout that overlaps either.

Three rules can call for a challenger, and any of them is enough. Two ask
whether the champion is failing: its error against its own training baseline
(``perf_drift_threshold``) and its skill against a daily profile
(``skill_floor``). The third asks nothing about the model's error at all, only
how long since it last passed an exam (``recertify_days``), because the first
two can both stay quiet indefinitely while a model goes stale.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from driftloop.config import DRIFT_FEATURES, LoopConfig
from driftloop.data.base import DataSource
from driftloop.drift import DataDriftResult, compute_data_drift, compute_perf_drift, distribution_report
from driftloop import stats
from driftloop.model import error_metrics, predictions_frame, rmse, squared_errors, train
from driftloop.retrospect import climatology_skill
from driftloop.tracking import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    ChampionRef,
    load_champion,
    load_version,
    log_and_register,
    mark_certified,
    mark_probation_cleared,
    mark_promoted,
    promote,
)

HOUR = pd.Timedelta(1, unit="h")

# Block length and resample count for the promotion gate's interval. See
# `_exam_margin_lower_bound` for both.
_GATE_BLOCK_HOURS = 24
_GATE_RESAMPLES = 2_000


@dataclass
class CycleResult:
    """Everything one scheduled run decided, in one flat record."""

    as_of: pd.Timestamp
    champion_version: str
    data_drift_psi: float
    data_drift_label: str
    worst_feature: str
    champion_rmse: float
    champion_mae: float
    champion_r2: float
    champion_baseline_rmse: float
    perf_drift_ratio: float
    data_drift_detected: bool
    perf_drift_detected: bool
    retrain_triggered: bool
    promotion_decision: str  # "none" | "promoted" | "rejected"
    # Skill against a 30-day daily profile, and whether it crossed the floor.
    # NaN and False when the floor is disabled, so a backend predating it reads
    # the same as one with it switched off.
    champion_skill: float = float("nan")
    skill_drift_detected: bool = False
    # How long since the serving champion last passed an exam, and whether that
    # is past `recertify_days`. The age is recorded whether or not the trigger
    # is switched on, because it is the quantity the ratchet hides: a champion
    # can be 200 days past its last exam with every drift signal quiet.
    certified_age_days: float = float("nan")
    recertify_due: bool = False
    # The exam margin's lower bound, when the gate is asked for one. NaN where
    # no challenger was trained or the confidence gate is off. Recorded even on
    # runs it does not change, because the gap between this and the point
    # estimate is how much of the exam was ever real.
    exam_margin_lo: float = float("nan")
    # Probation. "none" where nothing was due, otherwise whether the promotion
    # being judged survived. `probation_margin` is the champion's advantage over
    # the model it displaced on a window that postdates them both, so a negative
    # value is a promotion that should not have happened.
    probation_decision: str = "none"  # "none" | "kept" | "rolled_back"
    probation_margin: float = float("nan")
    challenger_rmse: float | None = None
    champion_rmse_holdout: float | None = None
    performance_gap: float | None = None
    per_feature_psi: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row = {k: v for k, v in self.__dict__.items() if k != "per_feature_psi"}
        row.update({f"psi_{k}": v for k, v in self.per_feature_psi.items()})
        return row


def bootstrap_champion(
    source: DataSource,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    cfg: LoopConfig,
) -> str:
    """Train the first champion and register it under the ``champion`` alias."""
    df = source.get_data(train_start, train_end)
    trained = train(df, kind=cfg.model_kind, params=cfg.model_params)

    with mlflow.start_run(run_name=f"bootstrap-{train_end.date()}"):
        mlflow.set_tags({"cycle_type": "bootstrap", "promotion_decision": "promoted"})
        mlflow.log_params(
            {
                "as_of": train_end.isoformat(),
                "train_start": trained.train_start.isoformat(),
                "train_end": trained.train_end.isoformat(),
                "n_train_rows": trained.n_rows,
            }
        )
        mlflow.log_metric("champion_baseline_rmse", trained.baseline_rmse)
        version = log_and_register(trained, cfg.registered_model_name, alias=CHAMPION_ALIAS)
        mlflow.set_tag("champion_version", version)
    return version


def run_cycle(source: DataSource, as_of: pd.Timestamp, cfg: LoopConfig) -> CycleResult:
    """One scheduled run. Assumes a champion already exists."""
    champion = load_champion(cfg.registered_model_name)
    if champion is None:
        raise RuntimeError("no champion registered -- run bootstrap_champion first")

    monitor_start = as_of - pd.Timedelta(cfg.monitor_days, unit="D")
    monitor = source.get_data(monitor_start, as_of)

    # Probation runs before anything else, so the rest of the cycle monitors
    # whichever model is actually going to serve rather than one already known
    # to have lost its place.
    probation_decision, probation_margin = _judge_probation(champion, monitor, as_of, cfg)
    if probation_decision == "rolled_back":
        champion = load_champion(cfg.registered_model_name)

    reference = source.get_data(champion.train_start, champion.train_end + HOUR)

    # --- Signal 1: data drift (no model involved) ---
    data_drift: DataDriftResult = compute_data_drift(reference, monitor, DRIFT_FEATURES)

    # --- Signal 2: performance drift (champion only) ---
    champion_metrics = error_metrics(champion.pipeline, monitor)
    champion_rmse = champion_metrics["rmse"]
    perf = compute_perf_drift(champion.baseline_rmse, champion_rmse, cfg.perf_drift_threshold)

    # --- Signal 2b: the same question against a yardstick that holds still ---
    #
    # `perf` divides by the champion's own training error, so every promotion
    # resets the denominator and the bar ratchets upward. This asks instead
    # whether the champion still beats a 30-day hour-of-day profile of recent
    # pollution, which nothing about promoting a model can move. Strictly an
    # additional way to fire; the gate still decides what ships.
    champion_skill = float("nan")
    skill_detected = False
    if cfg.skill_floor is not None:
        champion_skill = climatology_skill(
            source, monitor, monitor_start, source.forecast_lead_days, champion_rmse
        )
        # NaN means the baseline had no data to average: no opinion, not a
        # failing model.
        skill_detected = not pd.isna(champion_skill) and champion_skill < cfg.skill_floor

    # --- Signal 3: the certificate expired ---
    #
    # Not a drift signal at all, which is the point. Signals 1 and 2 both wait
    # to be told the model is failing, and both can be wrong about that
    # indefinitely: the ratio ratchets shut and PSI saturates. This asks only
    # how long it has been since the champion was last examined on data it had
    # never seen, which is a fact about the calendar that nothing the loop does
    # can suppress.
    certified_age_days = float((as_of - champion.certified_at()) / pd.Timedelta(1, unit="D"))
    recertify_due = cfg.recertify_days is not None and certified_age_days >= cfg.recertify_days

    result = CycleResult(
        as_of=as_of,
        champion_version=champion.version,
        data_drift_psi=data_drift.max_psi,
        data_drift_label=data_drift.label(),
        worst_feature=data_drift.worst_feature,
        champion_rmse=champion_rmse,
        champion_mae=champion_metrics["mae"],
        champion_r2=champion_metrics["r2"],
        champion_baseline_rmse=champion.baseline_rmse,
        perf_drift_ratio=perf.ratio,
        data_drift_detected=data_drift.detected(cfg.psi_threshold),
        perf_drift_detected=perf.detected,
        retrain_triggered=perf.detected or skill_detected or recertify_due,
        promotion_decision="none",
        per_feature_psi=dict(data_drift.per_feature_psi),
        champion_skill=champion_skill,
        skill_drift_detected=skill_detected,
        certified_age_days=certified_age_days,
        recertify_due=recertify_due,
        probation_decision=probation_decision,
        probation_margin=probation_margin,
    )

    challenger = None
    if result.retrain_triggered:
        holdout_start = as_of - pd.Timedelta(cfg.holdout_days, unit="D")
        challenger_start = as_of - pd.Timedelta(cfg.challenger_train_days, unit="D")

        # Leak guard: the judging window must post-date the champion's training.
        if champion.train_end >= holdout_start:
            raise ValueError(
                f"holdout window [{holdout_start}, {as_of}) overlaps the champion's "
                f"training data (ends {champion.train_end}). Increase the run interval "
                f"or reduce holdout_days."
            )

        challenger_df = source.get_data(challenger_start, holdout_start)
        holdout = source.get_data(holdout_start, as_of)
        challenger = train(challenger_df, kind=cfg.model_kind, params=cfg.model_params)

        champ_holdout = rmse(champion.pipeline, holdout)
        chal_holdout = rmse(challenger.pipeline, holdout)

        result.champion_rmse_holdout = champ_holdout
        result.challenger_rmse = chal_holdout
        result.performance_gap = champ_holdout - chal_holdout
        clears_point = chal_holdout < champ_holdout * (1 - cfg.promotion_margin)

        if cfg.promotion_confidence is not None:
            result.exam_margin_lo = _exam_margin_lower_bound(
                challenger.pipeline, champion.pipeline, holdout, cfg.promotion_confidence
            )
            # Both hurdles, so the confidence gate can only ever subtract a
            # promotion. A lower bound that cannot be computed (a window too
            # short to resample) is no opinion, and the point estimate decides
            # alone rather than a missing interval blocking by default.
            clears_interval = pd.isna(result.exam_margin_lo) or (
                result.exam_margin_lo > cfg.promotion_margin
            )
            clears_point = clears_point and clears_interval

        result.promotion_decision = "promoted" if clears_point else "rejected"

    _log_cycle(result, challenger, data_drift, cfg, champion.pipeline, monitor, reference)
    return result


def _log_monitoring_artifacts(champion_pipeline, monitor: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Log the per-run drift report and champion predictions as artifacts.

    The dashboard reads these files rather than the data source, so a panel does
    not have to know where the rows came from.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        preds = predictions_frame(champion_pipeline, monitor)
        preds.to_csv(tmp_path / "monitor_predictions.csv", index=False)

        report = distribution_report(reference, monitor, DRIFT_FEATURES)
        (tmp_path / "feature_distributions.json").write_text(json.dumps(report), encoding="utf-8")

        mlflow.log_artifacts(str(tmp_path), artifact_path="monitoring")


def _judge_probation(
    champion: ChampionRef, monitor: pd.DataFrame, as_of: pd.Timestamp, cfg: LoopConfig
) -> tuple[str, float]:
    """Re-run a promotion decision on a window that postdates it, once.

    Returns ``(decision, margin)`` where the margin is the serving champion's
    fractional RMSE advantage over the model it displaced. Negative means the
    promotion made things worse, and the ``champion`` alias is moved back.

    Deliberately not a second exam. It compares the two models that the original
    decision was between, on a window fixed by the calendar rather than chosen,
    and it happens exactly once per promotion. There is nothing to select from,
    so there is no winner's curse to inherit, which is the whole reason this is
    worth trying after three sharper gates failed.

    Silent for a bootstrap champion, which displaced nothing, and for a version
    already judged. A version whose predecessor cannot be loaded is left alone
    rather than rolled back into a model that is not there.
    """
    if cfg.probation_days is None or champion.probation_cleared:
        return "none", float("nan")
    if champion.promoted_at is None or champion.replaced_version is None:
        return "none", float("nan")
    if (as_of - champion.promoted_at).days < cfg.probation_days:
        return "none", float("nan")

    try:
        replaced = load_version(cfg.registered_model_name, champion.replaced_version)
    except Exception:
        mark_probation_cleared(cfg.registered_model_name, champion.version)
        return "none", float("nan")

    serving_rmse = rmse(champion.pipeline, monitor)
    replaced_rmse = rmse(replaced, monitor)
    if pd.isna(serving_rmse) or pd.isna(replaced_rmse) or not replaced_rmse:
        return "none", float("nan")

    margin = float(1.0 - serving_rmse / replaced_rmse)
    # Judged once either way, so a promotion is provisional for one window and
    # then settled. Leaving it open would re-judge the same decision every run
    # against a window that keeps moving, which is the retrying this mechanism
    # exists to avoid repeating.
    mark_probation_cleared(cfg.registered_model_name, champion.version)
    if margin >= 0:
        return "kept", margin

    promote(cfg.registered_model_name, champion.replaced_version)
    return "rolled_back", margin


def _exam_margin_lower_bound(challenger, champion, holdout: pd.DataFrame, confidence: float) -> float:
    """The lower bound of the challenger's RMSE advantage on the holdout window.

    Resampled in 24-hour blocks. Hourly pollution carries a strong diurnal cycle
    and episodes that run for days, so redrawing single hours would treat 168
    observations as 168 independent ones and report a range far narrower than
    the week supports. A day is the shortest block that keeps a whole cycle
    intact; at a seven-day holdout that leaves seven blocks, which is few, and
    the resulting bound is conservative rather than sharp.

    ``confidence`` is one-sided, because the gate asks a one-sided question. A
    two-sided interval at alpha leaves alpha/2 in each tail, so the alpha that
    puts ``1 - confidence`` below the lower bound is twice that.

    Fewer resamples than a published interval: this crosses a threshold rather
    than being printed, and a replay pays for it once per retrain.
    """
    champion_err = squared_errors(champion, holdout)
    challenger_err = squared_errors(challenger, holdout)
    if champion_err.size < _GATE_BLOCK_HOURS * 2:
        return float("nan")
    interval = stats.block_bootstrap(
        (challenger_err, champion_err),
        stats.rmse_margin,
        block=_GATE_BLOCK_HOURS,
        resamples=_GATE_RESAMPLES,
        alpha=2 * (1 - confidence),
    )
    return interval.lo


def _retrain_reason(result: CycleResult) -> str:
    """Every trigger that fired this run, joined with ``+``, or ``none``."""
    reasons = [
        name
        for name, fired in (
            ("ratio", result.perf_drift_detected),
            ("skill", result.skill_drift_detected),
            ("recert", result.recertify_due),
        )
        if fired
    ]
    return "+".join(reasons) if reasons else "none"


def _log_cycle(
    result: CycleResult,
    challenger,
    data_drift: DataDriftResult,
    cfg: LoopConfig,
    champion_pipeline,
    monitor: pd.DataFrame,
    reference: pd.DataFrame,
) -> None:
    """Write the run to MLflow, and move the champion alias if we promoted."""
    with mlflow.start_run(run_name=f"cycle-{result.as_of.date()}"):
        mlflow.set_tags(
            {
                "cycle_type": "monitor",
                "data_drift_detected": str(result.data_drift_detected),
                "perf_drift_detected": str(result.perf_drift_detected),
                "skill_drift_detected": str(result.skill_drift_detected),
                "recertify_due": str(result.recertify_due),
                # Every rule that fired, joined, so they can be counted
                # separately: each trigger was added to catch weeks where the
                # ones before it went quiet, and that is only checkable if a
                # week names all of its reasons rather than the first one.
                # "ratio" is the champion against its own training error,
                # "skill" against the daily profile, "recert" an expired
                # certificate. Count with `in`, not equality: a week can read
                # "ratio+recert".
                "retrain_reason": _retrain_reason(result),
                "retrain_triggered": str(result.retrain_triggered),
                "promotion_decision": result.promotion_decision,
                "probation_decision": result.probation_decision,
                "data_drift_label": result.data_drift_label,
                "worst_feature": result.worst_feature,
                "champion_version": result.champion_version,
            }
        )
        mlflow.log_params({"as_of": result.as_of.isoformat(), "monitor_days": cfg.monitor_days})

        metrics = {
            "data_drift_psi": result.data_drift_psi,
            "champion_rmse": result.champion_rmse,
            "champion_mae": result.champion_mae,
            "champion_r2": result.champion_r2,
            "champion_baseline_rmse": result.champion_baseline_rmse,
            "perf_drift_ratio": result.perf_drift_ratio,
            # 1.0 / 0.0 so the events show up as a step function in the UI.
            "retrain_triggered": float(result.retrain_triggered),
            "promotion_event": float(result.promotion_decision == "promoted"),
            # Logged whether or not the trigger is on, because it is the measure
            # the ratchet hides: with every drift signal quiet, this is the only
            # number that still moves.
            "certified_age_days": result.certified_age_days,
        }
        # Only when the floor is on. A logged NaN reads as a measurement rather
        # than as "not computed", and both UIs would have to tell them apart.
        if not pd.isna(result.champion_skill):
            metrics["champion_skill"] = result.champion_skill
        for feature, value in data_drift.per_feature_psi.items():
            metrics[f"psi_{feature}"] = value
        for feature, value in data_drift.per_feature_ks.items():
            metrics[f"ks_{feature}"] = value
        if result.challenger_rmse is not None:
            metrics["challenger_rmse"] = result.challenger_rmse
            metrics["champion_rmse_holdout"] = result.champion_rmse_holdout
            metrics["performance_gap"] = result.performance_gap
        if not pd.isna(result.exam_margin_lo):
            metrics["exam_margin_lo"] = result.exam_margin_lo
        if not pd.isna(result.probation_margin):
            metrics["probation_margin"] = result.probation_margin
            metrics["rollback_event"] = float(result.probation_decision == "rolled_back")
        mlflow.log_metrics(metrics)
        _log_monitoring_artifacts(champion_pipeline, monitor, reference)

        if challenger is not None:
            alias = CHAMPION_ALIAS if result.promotion_decision == "promoted" else CHALLENGER_ALIAS
            version = log_and_register(challenger, cfg.registered_model_name, alias=alias)
            mlflow.set_tag("challenger_version", version)
            if result.promotion_decision == "promoted":
                promote(cfg.registered_model_name, version)
                # Which version this displaced, recorded now because the tag
                # below is about to overwrite the only other record of it.
                mark_promoted(
                    cfg.registered_model_name, version, result.as_of, result.champion_version
                )
                mlflow.set_tag("champion_version", version)
            elif cfg.recertify_days is not None:
                # The incumbent sat an exam on unseen data and kept its place,
                # so its certificate is renewed from today. Only recorded when
                # the trigger is on: a backend that never uses it should carry
                # no tag suggesting otherwise.
                #
                # A promoted challenger is not marked. It has no `last_certified`
                # tag, so its certificate runs from the end of its own training
                # data, which is `holdout_days` before today. That is the more
                # conservative reading and the more honest one: what expires is
                # the currency of what the model knows.
                mark_certified(cfg.registered_model_name, result.champion_version, result.as_of)


def run_simulation(
    source: DataSource,
    cfg: LoopConfig,
    first_run: pd.Timestamp,
    last_run: pd.Timestamp,
    step_days: int = 7,
) -> pd.DataFrame:
    """Replay the scheduled loop over a timeline and return one row per run."""
    # A challenger promoted at `as_of` trained up to `as_of - holdout_days`. The
    # next run monitors [as_of + step_days - monitor_days, as_of + step_days), so
    # the two windows touch at step_days + holdout_days == monitor_days and
    # overlap below it, which scores a champion on hours it fitted and suppresses
    # the retrain trigger.
    #
    # Not `step_days >= holdout_days`, which agrees only at the shipped values
    # (7 + 7 == 14): that form admitted holdout_days=3 at a weekly cadence, where
    # four days of every monitor window is the new champion's training data, and
    # rejected a clean holdout_days=14. Both directions are pinned in
    # tests/test_loop.py.
    if step_days + cfg.holdout_days < cfg.monitor_days:
        raise ValueError(
            f"step_days ({step_days}) + holdout_days ({cfg.holdout_days}) must be at least "
            f"monitor_days ({cfg.monitor_days}); otherwise each monitor window reaches back "
            f"into the training data of the champion promoted the run before, and the "
            f"champion is scored on hours it fitted."
        )
    rows = []
    as_of = first_run
    while as_of <= last_run:
        rows.append(run_cycle(source, as_of, cfg).as_row())
        as_of = as_of + pd.Timedelta(step_days, unit="D")
    return pd.DataFrame(rows)
