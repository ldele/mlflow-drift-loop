"""The scheduled run: detect -> maybe retrain -> maybe promote, all logged.

Window layout for one run at time ``as_of`` (half-open windows)::

    ...........[==== challenger train ====][= holdout =] as_of
                        [====== monitor window ========]
    [== champion train ==]  (much earlier, never overlaps holdout)

- **monitor** drives both drift signals (data drift vs. the champion's training
  distribution, and the champion's current RMSE vs. its baseline).
- **holdout** is the judge. It is excluded from the challenger's training data
  and post-dates the champion's, so *neither model has seen it*. That is the
  leak the original prototype had: it trained the challenger on part of the
  window it was then scored on.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from driftloop.config import FEATURES, LoopConfig
from driftloop.data.base import DataSource
from driftloop.drift import DataDriftResult, compute_data_drift, compute_perf_drift, distribution_report
from driftloop.model import error_metrics, predictions_frame, rmse, train
from driftloop.tracking import CHALLENGER_ALIAS, CHAMPION_ALIAS, load_champion, log_and_register, promote

HOUR = pd.Timedelta(hours=1)


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
    """Train the very first champion and register it. Step 2 of the plan."""
    df = source.get_data(train_start, train_end)
    trained = train(df)

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

    monitor = source.get_data(as_of - pd.Timedelta(days=cfg.monitor_days), as_of)
    reference = source.get_data(champion.train_start, champion.train_end + HOUR)

    # --- Signal 1: data drift (no model involved) ---
    data_drift: DataDriftResult = compute_data_drift(reference, monitor, FEATURES)

    # --- Signal 2: performance drift (champion only) ---
    champion_metrics = error_metrics(champion.pipeline, monitor)
    champion_rmse = champion_metrics["rmse"]
    perf = compute_perf_drift(champion.baseline_rmse, champion_rmse, cfg.perf_drift_threshold)

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
        retrain_triggered=perf.detected,
        promotion_decision="none",
        per_feature_psi=dict(data_drift.per_feature_psi),
    )

    challenger = None
    if perf.detected:
        holdout_start = as_of - pd.Timedelta(days=cfg.holdout_days)
        challenger_start = as_of - pd.Timedelta(days=cfg.challenger_train_days)

        # Leak guard: the judging window must post-date the champion's training.
        if champion.train_end >= holdout_start:
            raise ValueError(
                f"holdout window [{holdout_start}, {as_of}) overlaps the champion's "
                f"training data (ends {champion.train_end}). Increase the run interval "
                f"or reduce holdout_days."
            )

        challenger_df = source.get_data(challenger_start, holdout_start)
        holdout = source.get_data(holdout_start, as_of)
        challenger = train(challenger_df)

        champ_holdout = rmse(champion.pipeline, holdout)
        chal_holdout = rmse(challenger.pipeline, holdout)

        result.champion_rmse_holdout = champ_holdout
        result.challenger_rmse = chal_holdout
        result.performance_gap = champ_holdout - chal_holdout
        result.promotion_decision = (
            "promoted" if chal_holdout < champ_holdout * (1 - cfg.promotion_margin) else "rejected"
        )

    _log_cycle(result, challenger, data_drift, cfg, champion.pipeline, monitor, reference)
    return result


def _log_monitoring_artifacts(champion_pipeline, monitor: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Log the per-run drift report + champion predictions as artifacts.

    This is the standard "each run leaves a report behind" pattern. It also keeps
    the dashboard decoupled from the data source: it reads these files, not the
    generator, so nothing about the panels changes when Phase 2 swaps in real data.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        preds = predictions_frame(champion_pipeline, monitor)
        preds.to_csv(tmp_path / "monitor_predictions.csv", index=False)

        report = distribution_report(reference, monitor, FEATURES)
        (tmp_path / "feature_distributions.json").write_text(json.dumps(report), encoding="utf-8")

        mlflow.log_artifacts(str(tmp_path), artifact_path="monitoring")


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
                "retrain_triggered": str(result.retrain_triggered),
                "promotion_decision": result.promotion_decision,
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
        }
        for feature, value in data_drift.per_feature_psi.items():
            metrics[f"psi_{feature}"] = value
        for feature, value in data_drift.per_feature_ks.items():
            metrics[f"ks_{feature}"] = value
        if result.challenger_rmse is not None:
            metrics["challenger_rmse"] = result.challenger_rmse
            metrics["champion_rmse_holdout"] = result.champion_rmse_holdout
            metrics["performance_gap"] = result.performance_gap
        mlflow.log_metrics(metrics)
        _log_monitoring_artifacts(champion_pipeline, monitor, reference)

        if challenger is not None:
            alias = CHAMPION_ALIAS if result.promotion_decision == "promoted" else CHALLENGER_ALIAS
            version = log_and_register(challenger, cfg.registered_model_name, alias=alias)
            mlflow.set_tag("challenger_version", version)
            if result.promotion_decision == "promoted":
                promote(cfg.registered_model_name, version)
                mlflow.set_tag("champion_version", version)


def run_simulation(
    source: DataSource,
    cfg: LoopConfig,
    first_run: pd.Timestamp,
    last_run: pd.Timestamp,
    step_days: int = 7,
) -> pd.DataFrame:
    """Replay the scheduled loop over a timeline and return one row per run."""
    if step_days < cfg.holdout_days:
        raise ValueError(
            f"step_days ({step_days}) must be >= holdout_days ({cfg.holdout_days}); "
            "otherwise a freshly promoted champion would be judged on its own training data."
        )
    rows = []
    as_of = first_run
    while as_of <= last_run:
        rows.append(run_cycle(source, as_of, cfg).as_row())
        as_of = as_of + pd.Timedelta(days=step_days)
    return pd.DataFrame(rows)
