"""Replay all six cities on the loop as it should have been configured.

    python scripts/rebaseline.py            # all six cities, both arms
    python scripts/rebaseline.py --city la

Two arms per city, identical but for the configuration under test:

- ``shipped``  what every published number was produced under: the Ridge at the
  library default ``alpha=1.0``, and the ratcheting ratio as the only retrain
  trigger. This arm is the faithfulness check -- it has to reproduce the figures
  in ``docs/evaluation.md``, or the harness is not measuring what it claims to.
- ``fixed``    the same loop with the two decisions this project has already
  measured and not taken: the penalty chosen per fit by forward-chaining CV
  (D5), and the model-independent skill floor switched on at -0.5 (D1).

Nothing here replaces a published number. It is run to sit beside them, because
the one habit this project's own history says to keep is that a changed figure
keeps the old one and the reason next to it (``docs/HISTORY.md``).

**What was predicted, written 2026-09-22 before the first replay ran.** The
ablation already tuned two cities at the champion's bootstrap window only, so
these are extrapolations from it and from the sweeps, not restatements:

1. Delhi's premium falls a long way in the fixed arm -- toward the tuned Ridge's
   +27.7%, and by more than its interval's width. A model that is no longer
   under-regularised has less for a retrain to recover.
2. Los Angeles's harm shrinks and its interval covers zero. Under the tuned
   Ridge alone the ablation had it at -12.8% [-33.6, -3.3], still clear of zero,
   so if the floor is doing anything this is where it shows.
3. Kraków and Melbourne stay indistinguishable from zero in both arms. Nothing
   here addresses the reason, which is that 47 weeks carry about five
   independent observations.
4. The fixed arm fires more often in the quiet cities and its longest silence
   falls, most in Kraków and Los Angeles. That is the floor doing what the
   sweep said it does.
5. Median served error improves in the fixed arm in most cities, most in Delhi.
   If it does not, tuning per fit is buying nothing and the case for D5 is
   weaker than the ablation suggested.

**The third arm was added after the first run and is exploratory**, which is the
difference between it and the five predictions above. It was added because the
`fixed` arm compensates for the ratchet instead of removing it, and "delete the
broken trigger" is the cheaper fix if it works. It does not: see the section in
docs/evaluation.md.

Writes outputs/rebaseline.csv and outputs/rebaseline.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path[:0] = [str(Path(__file__).resolve().parent)]

from ablate_model import CITIES, run_arm  # noqa: E402

from driftloop import tracking  # noqa: E402
from driftloop.benchmark import LOOP_ALPHA_GRID  # noqa: E402
from driftloop.config import PROFILES  # noqa: E402
from driftloop.data import OpenMeteoSource  # noqa: E402
from driftloop.model import RIDGE  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"

#: The configurations, as (label, LoopConfig overrides).
#:
#: The third arm asks the question the second one dodges. The ratio trigger is
#: the ratchet -- it divides by the champion's own training error, so every
#: promotion resets the bar upward and the alarm eventually cannot fire. `fixed`
#: leaves it in place and adds a second trigger beside it, which compensates for
#: the fault rather than removing it. `floor_only` takes it out: if the two arms
#: behave the same, the ratcheting trigger is carrying nothing and the honest fix
#: is deleting it.
ARMS: tuple[tuple[str, dict], ...] = (
    ("shipped", {}),
    ("fixed", {"tune_alpha": True, "skill_floor": -0.5}),
    ("floor_only", {"tune_alpha": True, "skill_floor": -0.5, "perf_drift_threshold": float("inf")}),
)


def silence(label: str) -> dict:
    """How long this arm went without firing, and what it chose when it did.

    The premium says whether retraining paid. This says whether the loop was
    still able to act at all, which is the failure the ratchet produces and the
    one a premium cannot show: a loop that never fires has no premium to lose.

    Read straight after the arm ran, while the tracking URI still points at that
    arm's own throwaway backend.
    """
    runs = mlflow.search_runs(
        experiment_names=[f"ablate-{label}"], order_by=["attributes.start_time ASC"]
    )
    runs = runs[runs["tags.cycle_type"] == "monitor"]
    fired = [t == "True" for t in runs["tags.retrain_triggered"]]

    longest = run = 0
    for f in fired:
        run = 0 if f else run + 1
        longest = max(longest, run)

    return {"runs": len(fired), "fires": sum(fired), "longest_silence": longest}


def alphas(model_name: str) -> list[float]:
    """Every penalty this arm's registered versions were fitted at, in order."""
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        return []
    out = []
    for mv in sorted(versions, key=lambda v: int(v.version)):
        if "alpha" in mv.tags:
            out.append(float(mv.tags["alpha"]))
    return out


def pct(value: float | None) -> str:
    return "     -" if value is None else f"{value:+6.1f}%"


def interval(row: dict) -> str:
    if row["premium"] is None:
        return "no acted windows"
    verdict = "clears zero" if row["premium_real"] else "covers zero"
    return f"[{row['premium_lo']:+.1f}, {row['premium_hi']:+.1f}] {verdict}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=[*CITIES, "all"], default="all")
    args = parser.parse_args()

    keys = list(CITIES) if args.city == "all" else [args.city]
    rows: list[dict] = []

    original_root = tracking.REPO_ROOT
    with tempfile.TemporaryDirectory(prefix="driftloop-rebaseline-", ignore_cleanup_errors=True) as tmp:
        tracking.REPO_ROOT = Path(tmp)
        try:
            for name in keys:
                profile = PROFILES[CITIES[name]]
                if profile.replay is None or profile.location is None:
                    continue
                source = OpenMeteoSource(profile.location)
                source.timeline()
                print(f"\n=== {profile.label} ===", flush=True)

                for label, overrides in ARMS:
                    arm_label = f"rebaseline_{label}"
                    arm = run_arm(profile, source, RIDGE, None, arm_label, **overrides)
                    arm.update(silence(arm_label))
                    arm["arm"] = label
                    chosen = alphas(f"{profile.loop.registered_model_name}-{arm_label}")
                    arm["alphas"] = chosen
                    rows.append(arm)

                    print(
                        f"  {label:<8} premium {pct(arm['premium'])} {interval(arm):<34}"
                        f" median RMSE {arm['median_rmse']:6.2f}"
                        f"  fires {arm['fires']:>2}/{arm['runs']:<2}"
                        f"  promotions {arm['promotions']:>2}"
                        f"  longest silence {arm['longest_silence']:>2}w",
                        flush=True,
                    )
                    if chosen:
                        print(f"           alphas chosen: {chosen}", flush=True)
                        # A sweep that lands on its own last value is reporting
                        # the edge of the grid, not the optimum. Printed rather
                        # than raised: the arm is still worth reading, but its
                        # penalty is a lower bound on what the data wanted.
                        if max(chosen) >= max(LOOP_ALPHA_GRID):
                            print(
                                f"           WARNING: a chosen alpha sits at the top of the grid"
                                f" ({max(LOOP_ALPHA_GRID):g}); widen LOOP_ALPHA_GRID and re-run",
                                flush=True,
                            )
        finally:
            tracking.REPO_ROOT = original_root

    OUTPUTS.mkdir(exist_ok=True)
    frame = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    )
    frame.to_csv(OUTPUTS / "rebaseline.csv", index=False)
    (OUTPUTS / "rebaseline.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {OUTPUTS / 'rebaseline.csv'} and {OUTPUTS / 'rebaseline.json'}")


if __name__ == "__main__":
    main()
