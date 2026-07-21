"""Prove the two drift signals are independent and each responds to its own knob.

Sweeps ``feature_shift`` (covariate drift) and ``drift_strength`` (concept
drift) one at a time, and measures both detectors each time. Expected result:

    feature_shift  -> PSI climbs monotonically, perf_drift_ratio stays flat
    drift_strength -> perf_drift_ratio climbs monotonically, PSI stays flat

That is plan step 3, sharpened: the plan asked to check the data-drift signal
against "the knob", but a knob that only changes ``f``'s coefficients cannot
move a feature-distribution statistic. Two knobs, one per signal.

Runs entirely offline -- no MLflow, a few seconds.

    python scripts/sweep_knobs.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from driftloop.config import FEATURES, LoopConfig, SyntheticConfig
from driftloop.data import SyntheticSource
from driftloop.drift import compute_data_drift, compute_perf_drift
from driftloop.model import rmse, train

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"

TRAIN_START = pd.Timestamp("2025-04-01")
TRAIN_END = pd.Timestamp("2025-07-01")
# A monitor window well past the drift date so the ramp is fully applied.
MONITOR_START = pd.Timestamp("2025-10-15")
MONITOR_END = pd.Timestamp("2025-10-29")

LEVELS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]


def measure(drift_strength: float, feature_shift: float, cfg: LoopConfig) -> dict:
    source = SyntheticSource(
        SyntheticConfig(drift_strength=drift_strength, feature_shift=feature_shift)
    )
    train_df = source.get_data(TRAIN_START, TRAIN_END)
    monitor_df = source.get_data(MONITOR_START, MONITOR_END)

    champion = train(train_df)
    data_drift = compute_data_drift(train_df, monitor_df, FEATURES)
    perf = compute_perf_drift(
        champion.baseline_rmse, rmse(champion.pipeline, monitor_df), cfg.perf_drift_threshold
    )
    return {
        "drift_strength": drift_strength,
        "feature_shift": feature_shift,
        "max_psi": data_drift.max_psi,
        "mean_psi": data_drift.mean_psi,
        "perf_drift_ratio": perf.ratio,
        "champion_rmse": perf.current_rmse,
        "baseline_rmse": perf.baseline_rmse,
    }


def main() -> None:
    cfg = LoopConfig()
    rows = []
    for level in LEVELS:  # covariate drift only
        rows.append({"sweep": "feature_shift", **measure(0.0, level, cfg), "level": level})
    for level in LEVELS:  # concept drift only
        rows.append({"sweep": "drift_strength", **measure(level, 0.0, cfg), "level": level})

    df = pd.DataFrame(rows)
    OUTPUTS.mkdir(exist_ok=True)
    out = OUTPUTS / "sweep.csv"
    df.to_csv(out, index=False)

    for sweep, group in df.groupby("sweep", sort=False):
        print(f"\n--- sweeping {sweep} (the other knob held at 0) ---")
        print(
            group[["level", "max_psi", "perf_drift_ratio"]].to_string(
                index=False, float_format=lambda v: f"{v:8.3f}"
            )
        )
        driven = "max_psi" if sweep == "feature_shift" else "perf_drift_ratio"
        monotone = group[driven].is_monotonic_increasing
        print(f"    {driven} monotonically increasing: {monotone}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
