"""Put a confidence interval on every number the project reports.

The six-city table, the win rates and the gate calibration were all published as
bare point estimates. On 19 to 48 autocorrelated weekly windows, several of them
were never distinguishable from zero, and the table gave a reader no way to tell
which. This script is the answer to that.

    python scripts/uncertainty.py                # every city, plus the pooled gate
    python scripts/uncertainty.py --city delhi   # one city
    python scripts/uncertainty.py --sensitivity  # also sweep the block length

Writes ``outputs/uncertainty.json`` and prints the markdown tables that
``docs/evaluation.md`` and the README quote. Regenerate after any change that
moves the numbers, and paste the tables rather than editing them by hand: a
figure typed into prose is one nobody re-derives when the code moves under it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

# City labels carry accents and the tables carry "≥". The Windows console
# defaults to cp1252, which cannot encode either, so the script would die after
# printing four correct tables.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from driftloop import retrospect, stats, tracking  # noqa: E402
from driftloop.config import PROFILES  # noqa: E402
from driftloop.data import replayable_source  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "uncertainty.json"

# The city profiles, in the order the README's table uses.
CITY_KEYS = [
    "openmeteo_delhi",
    "openmeteo_santiago",
    "openmeteo",
    "openmeteo_joburg",
    "openmeteo_melbourne",
    "openmeteo_la",
]


def build_retrospective(key: str):
    """Rebuild one city's retrospective from its committed MLflow backend.

    Returns ``(retrospective, runs)``; the run frame is needed alongside because
    retrain counts live on the run tags rather than in the retrospective.
    """
    import build_site  # local import: it configures MLflow on import

    profile = PROFILES[key]
    cfg = profile.loop
    runs = build_site.load_runs(profile.db_filename, cfg.experiment_name)
    if runs.empty or "tags.champion_version" not in runs:
        return None, None
    source = replayable_source(profile)
    if source is None or profile.location is None:
        return None, None
    tracking.setup(cfg.experiment_name, profile.db_filename)
    models = retrospect.registered_models(MlflowClient(), cfg.registered_model_name)
    if not models:
        return None, None
    frame = runs.copy()
    frame["champion_version"] = runs["tags.champion_version"].astype(int)
    retro = retrospect.build(
        source, frame, models, cfg.monitor_days,
        profile.location.forecast_lead_days, cfg.promotion_margin,
    )
    return retro, runs


def _count_tag(runs, column: str, value: str) -> int:
    return int((runs[column] == value).sum()) if column in runs else 0


def city_row(key: str) -> dict | None:
    retro, runs = build_retrospective(key)
    if retro is None or not retro.as_of:
        return None
    value = retrospect.retraining_value(retro)
    intervals = retrospect.retraining_uncertainty(retro)
    return {
        "key": key,
        "label": PROFILES[key].label,
        "weeks": len(retro.as_of),
        # Three different counts that the original table collapsed into one
        # column called "retrains". A challenger is *trained* whenever the
        # trigger fires; it is *promoted* only if it clears the gate; and it can
        # only be *calibrated* if there is a post-promotion window to score it
        # on. Johannesburg trains eleven and ships three, which is the whole
        # point of having a gate, so the three numbers belong in three columns.
        "retrains": _count_tag(runs, "tags.retrain_triggered", "True"),
        "promotions": _count_tag(runs, "tags.promotion_decision", "promoted"),
        "calibrated": len(retro.gate),
        "acted_windows": value.get("acted_windows", 0),
        "point": {k: value.get(k) for k in ("across_replay", "when_it_acted", "win_rate")},
        "interval": {k: v.to_dict() for k, v in intervals.items()},
        "gate": retro.gate,
        "_intervals": intervals,
        "_series": retrospect.retraining_series(retro),
    }


def markdown_city_table(rows: list[dict]) -> str:
    """The six-city table, with every percentage carrying its interval."""
    out = [
        "| | weeks | retrains | shipped | across the replay | week by week | won |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        intervals = row["_intervals"]
        across = intervals.get("across_replay")
        paired = intervals.get("when_it_acted")
        out.append(
            f"| **{row['label']}** | {row['weeks']} | {row['retrains']} | {row['promotions']} | "
            f"{across.format() if across else '—'} | "
            f"{paired.format() if paired else '—'} | "
            f"{_win_cell(row)} |"
        )
    return "\n".join(out)


def _win_cell(row: dict) -> str:
    """Win rate, reported with the Wilson interval rather than the bootstrap one.

    The block bootstrap degenerates at the boundary: Santiago and Johannesburg
    won *every* window they acted on, so every resample also wins every window
    and the interval collapses to [100, 100]. That is not certainty, it is the
    percentile bootstrap being unable to represent uncertainty at a boundary, a
    known failure and a dangerous one to publish uncorrected.

    Wilson handles the boundary properly (16 of 16 gives roughly [80, 100]).
    Its cost is assuming independent trials, which weekly wins are not, so it is
    if anything still too narrow. Both flaws point the same way, so the table
    reports the win rate as "at least this good".
    """
    won = row["_intervals"].get("win_rate")
    n = row["acted_windows"]
    if won is None or not n:
        return "—"
    successes = int(round(won.point / 100 * n))
    lo, hi = stats.wilson_interval(successes, n)
    return f"{won.point:.0f}% [{lo:.0f}, {hi:.0f}] of {n}"


def markdown_effective_n(rows: list[dict]) -> str:
    """How much independent information each replay carries."""
    out = [
        "| | weeks compared | effective n | lag-1 autocorrelation | block length |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        paired = row["_intervals"].get("when_it_acted")
        if paired is None:
            continue
        rho = (paired.n - paired.n_effective) / (paired.n + paired.n_effective)
        out.append(
            f"| **{row['label']}** | {paired.n} | {paired.n_effective:.0f} | "
            f"{rho:.2f} | {paired.block_length} |"
        )
    return "\n".join(out)


def pooled_gate(rows: list[dict]) -> tuple[dict, dict]:
    """Gate calibration pooled across every city, with intervals."""
    gate = [g for row in rows for g in row["gate"]]
    return retrospect.gate_summary(gate), retrospect.gate_uncertainty(gate)


def markdown_gate(summary: dict, intervals: dict, gate: list[dict]) -> str:
    """Gate calibration, reported so that a group of three cannot masquerade as a trend.

    Below four promotions the bootstrap refuses to produce an interval, and it is
    right to: resampling three numbers tells you about those three numbers. But
    printing ``[-inf, +inf]`` in a published table is worse than useless, so a
    small group is reported as what it is: the individual delivered margins,
    listed, plus a Wilson interval on the proportion that came out harmful,
    which is the one claim three observations can support.
    """
    out = [
        "| promotions | n | exam promised | delivered | harmful |",
        "|---|---|---|---|---|",
    ]
    labels = {"short": f"served < {retrospect.GATE_LONG_WEEKS} weeks",
              "long": f"served >= {retrospect.GATE_LONG_WEEKS} weeks"}
    for key, label in labels.items():
        block = summary.get(key, {})
        if not block.get("n"):
            continue
        rows = [
            g for g in gate
            if (g["weeks_served"] < retrospect.GATE_LONG_WEEKS) == (key == "short")
        ]
        ci = intervals.get(key, {})
        exam, delivered = ci.get("exam"), ci.get("delivered")
        small = block["n"] < 4

        if small:
            values = sorted(g["delivered_margin"] * 100 for g in rows)
            exam_cell = f"{block['exam'] * 100:+.1f}% (n too small to bound)"
            delivered_cell = (
                f"{block['delivered'] * 100:+.1f}%, and the {block['n']} values are "
                + ", ".join(f"{v:+.1f}%" for v in values)
            )
        else:
            exam_cell = exam.format() if exam else f"{block['exam'] * 100:+.1f}%"
            delivered_cell = delivered.format() if delivered else f"{block['delivered'] * 100:+.1f}%"

        lo, hi = stats.wilson_interval(block["harmful"], block["n"])
        harmful_cell = f"{block['harmful']}/{block['n']} [{lo:.0f}%, {hi:.0f}%]"
        out.append(f"| {label} | {block['n']} | {exam_cell} | {delivered_cell} | {harmful_cell} |")
    return "\n".join(out)


def markdown_sensitivity(rows: list[dict]) -> str:
    """The headline at several block lengths, including the IID bootstrap."""
    blocks = (1, 2, 3, 4, 6, 8)
    out = ["| | " + " | ".join(f"L={b}" for b in blocks) + " |",
           "|---" * (len(blocks) + 1) + "|"]
    for row in rows:
        retro_series = row.get("_series")
        if not retro_series:
            continue
        cells = []
        for b in blocks:
            if b > max(1, retro_series["served_acted"].size // 2):
                cells.append("—")
                continue
            interval = stats.block_bootstrap(
                (retro_series["served_acted"], retro_series["frozen_acted"]),
                stats.pct_improvement_paired, block=b,
            )
            cells.append("—" if interval.n == 0 else f"[{interval.lo:+.0f}, {interval.hi:+.0f}]")
        out.append(f"| **{row['label']}** | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", help="one profile key or city name; default all")
    parser.add_argument("--sensitivity", action="store_true", help="sweep the block length")
    args = parser.parse_args()

    keys = CITY_KEYS
    if args.city:
        match = [k for k in CITY_KEYS if args.city.lower() in (k, PROFILES[k].label.lower())]
        keys = match or [args.city]

    rows = []
    for key in keys:
        row = city_row(key)
        if row is None:
            print(f"  skipped {key}: no usable replay", file=sys.stderr)
            continue
        rows.append(row)

    if not rows:
        print("no cities produced a retrospective", file=sys.stderr)
        raise SystemExit(1)

    summary, gate_intervals = pooled_gate(rows)
    all_gate = [g for row in rows for g in row["gate"]]

    print("\n## Retraining value, with intervals\n")
    print(markdown_city_table(rows))
    print("\n## How much independent information each replay carries\n")
    print(markdown_effective_n(rows))
    print("\n## Promotion-gate calibration, pooled\n")
    print(markdown_gate(summary, gate_intervals, all_gate))
    if args.sensitivity:
        print("\n## Week-by-week premium at several block lengths (L=1 is IID)\n")
        print(markdown_sensitivity(rows))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "generated_from": "committed MLflow backends + data_cache parquet",
                "seed": stats.DEFAULT_SEED,
                "resamples": stats.DEFAULT_RESAMPLES,
                "alpha": stats.DEFAULT_ALPHA,
                "cities": [
                    {k: v for k, v in row.items() if not k.startswith("_") and k != "gate"}
                    for row in rows
                ],
                "gate": {
                    "summary": summary,
                    "interval": {
                        group: {k: v.to_dict() for k, v in ci.items()}
                        for group, ci in gate_intervals.items()
                    },
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUTPUT.relative_to(Path.cwd()) if OUTPUT.is_relative_to(Path.cwd()) else OUTPUT}")


if __name__ == "__main__":
    main()
