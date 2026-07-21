"""End-to-end Phase 1 demo: bootstrap a champion, then replay ~5 months of
weekly scheduled runs across a synthetic autumn regime shift.

    python scripts/run_simulation.py [--fresh]

Writes every run to MLflow (sqlite backend) and a tidy CSV to outputs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from driftloop import tracking
from driftloop.config import LoopConfig, SyntheticConfig
from driftloop.data import SyntheticSource
from driftloop.loop import bootstrap_champion, run_simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"

# The champion learns spring/summer, then the world turns to autumn on 2025-09-15.
CHAMPION_TRAIN_START = pd.Timestamp("2025-04-01")
CHAMPION_TRAIN_END = pd.Timestamp("2025-07-01")
FIRST_RUN = pd.Timestamp("2025-07-08")
LAST_RUN = pd.Timestamp("2025-12-15")
STEP_DAYS = 7


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help="wipe the experiment and registry first")
    parser.add_argument("--drift-strength", type=float, default=1.0)
    parser.add_argument("--feature-shift", type=float, default=1.0)
    args = parser.parse_args()

    loop_cfg = LoopConfig()
    if args.fresh:
        tracking.reset()
    tracking.setup(loop_cfg.experiment_name)

    syn_cfg = SyntheticConfig(drift_strength=args.drift_strength, feature_shift=args.feature_shift)
    source = SyntheticSource(syn_cfg)

    # A little context the dashboard reads to annotate the regime shift, so it
    # doesn't hardcode a date. (The loop itself stays data-source-agnostic.)
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "run_meta.json").write_text(
        json.dumps(
            {
                "drift_date": syn_cfg.drift_date.isoformat(),
                "drift_strength": syn_cfg.drift_strength,
                "feature_shift": syn_cfg.feature_shift,
                "champion_train_start": CHAMPION_TRAIN_START.isoformat(),
                "champion_train_end": CHAMPION_TRAIN_END.isoformat(),
            }
        ),
        encoding="utf-8",
    )

    print(f"Bootstrapping champion on {CHAMPION_TRAIN_START.date()} -> {CHAMPION_TRAIN_END.date()}")
    version = bootstrap_champion(source, CHAMPION_TRAIN_START, CHAMPION_TRAIN_END, loop_cfg)
    print(f"  registered {loop_cfg.registered_model_name} v{version} as @champion\n")

    print(f"Replaying weekly runs {FIRST_RUN.date()} -> {LAST_RUN.date()}")
    df = run_simulation(source, loop_cfg, FIRST_RUN, LAST_RUN, STEP_DAYS)

    OUTPUTS.mkdir(exist_ok=True)
    out = OUTPUTS / "simulation.csv"
    df.to_csv(out, index=False)

    cols = ["as_of", "data_drift_psi", "perf_drift_ratio", "champion_rmse", "promotion_decision"]
    with pd.option_context("display.width", 120, "display.max_rows", None):
        print("\n" + df[cols].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    promotions = df[df.promotion_decision == "promoted"]
    print(f"\nruns={len(df)}  retrains={int(df.retrain_triggered.sum())}  promotions={len(promotions)}")
    for _, row in promotions.iterrows():
        print(f"  promoted at {row.as_of:%Y-%m-%d} (gap {row.performance_gap:.2f} RMSE)")
    print(f"\nwrote {out}")
    print("MLflow UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("Dashboard:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
