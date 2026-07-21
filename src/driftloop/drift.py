"""Dual drift signals.

| Signal           | What it measures  | Needs a model? |
|------------------|-------------------|----------------|
| Data drift (PSI) | the world changed | no             |
| Performance drift| the model failing | champion only  |

Keeping them independent is the point: data drift is the early warning, and it
can be computed with no labels and no model at all. Performance drift is the
action signal that triggers a retrain. Detecting drift by "champion vs
challenger RMSE" would be circular -- you would need a challenger before you
were allowed to decide you needed one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from driftloop.config import FEATURES

# Conventional PSI reading. Kept here so the thresholds have one home.
PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between two samples of one feature.

    Bin edges come from the *reference* quantiles, so the reference is uniform
    across bins by construction and any imbalance is attributable to `current`.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        return float("nan")

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if edges.size < 2:  # degenerate (constant) reference feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_frac = np.histogram(reference, bins=edges)[0] / reference.size
    cur_frac = np.histogram(current, bins=edges)[0] / current.size
    ref_frac = np.clip(ref_frac, eps, None)
    cur_frac = np.clip(cur_frac, eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and p-value."""
    result = stats.ks_2samp(np.asarray(reference, float), np.asarray(current, float))
    return float(result.statistic), float(result.pvalue)


@dataclass
class DataDriftResult:
    """PSI (headline) plus KS (cross-check) for every feature."""

    per_feature_psi: dict[str, float] = field(default_factory=dict)
    per_feature_ks: dict[str, float] = field(default_factory=dict)
    per_feature_ks_pvalue: dict[str, float] = field(default_factory=dict)

    @property
    def max_psi(self) -> float:
        return max(self.per_feature_psi.values())

    @property
    def mean_psi(self) -> float:
        return float(np.mean(list(self.per_feature_psi.values())))

    @property
    def worst_feature(self) -> str:
        return max(self.per_feature_psi, key=self.per_feature_psi.__getitem__)

    def detected(self, threshold: float = PSI_SIGNIFICANT) -> bool:
        return self.max_psi > threshold

    def label(self) -> str:
        if self.max_psi > PSI_SIGNIFICANT:
            return "significant"
        if self.max_psi > PSI_STABLE:
            return "moderate"
        return "stable"


def compute_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
) -> DataDriftResult:
    """Compare the current feature distributions against the training ones."""
    features = features or FEATURES
    result = DataDriftResult()
    for feature in features:
        result.per_feature_psi[feature] = psi(reference[feature].to_numpy(), current[feature].to_numpy())
        stat, pvalue = ks(reference[feature].to_numpy(), current[feature].to_numpy())
        result.per_feature_ks[feature] = stat
        result.per_feature_ks_pvalue[feature] = pvalue
    return result


def distribution_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
    bins: int = 20,
) -> dict:
    """Per-feature histograms (shared bins) of reference vs current, plus PSI/KS.

    JSON-serialisable, so it can be logged as an MLflow artifact each run and read
    back by the dashboard -- the standard "log a drift report per run" pattern,
    and it keeps the dashboard decoupled from the data source (works in Phase 2).
    """
    features = features or FEATURES
    report: dict[str, dict] = {}
    for feature in features:
        ref = reference[feature].to_numpy(dtype=float)
        cur = current[feature].to_numpy(dtype=float)
        combined = np.concatenate([ref, cur])
        edges = np.linspace(float(np.min(combined)), float(np.max(combined)), bins + 1)
        stat, pvalue = ks(ref, cur)
        report[feature] = {
            "edges": edges.tolist(),
            "reference_counts": np.histogram(ref, bins=edges)[0].tolist(),
            "current_counts": np.histogram(cur, bins=edges)[0].tolist(),
            "reference_mean": float(np.mean(ref)),
            "current_mean": float(np.mean(cur)),
            "psi": psi(ref, cur),
            "ks_stat": stat,
            "ks_pvalue": pvalue,
        }
    return report


@dataclass
class PerfDriftResult:
    """How much worse the champion is now than it was at training time."""

    baseline_rmse: float
    current_rmse: float
    threshold: float

    @property
    def ratio(self) -> float:
        if self.baseline_rmse <= 0:
            return float("inf")
        return self.current_rmse / self.baseline_rmse

    @property
    def detected(self) -> bool:
        return self.ratio > self.threshold


def compute_perf_drift(baseline_rmse: float, current_rmse: float, threshold: float) -> PerfDriftResult:
    return PerfDriftResult(baseline_rmse=baseline_rmse, current_rmse=current_rmse, threshold=threshold)
