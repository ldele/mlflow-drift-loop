"""Each week's forecast error in three parts, read from ``outputs/split/`` for display.

``scripts/split_error.py`` computes the split under its registration and writes
the four scorings of every week to ``weeks.csv``. The site and the dashboard
draw it week by week, and this module is the one place both turn those four
scorings into the three parts. Out of scope: computing the cells, scoring the
predictions, or anything ``split_error.py`` decides.

The formula is the registered one. ``tests/test_split_error.py`` checks it
against the script's own, so the pages cannot drift from what was scored.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PARTS = ("weather", "relation", "unexplained")


def parts(a, b, c, d):
    """The weather, relation and unexplained parts of a served error ``a``.

    ``a`` is the model in service on week-ahead weather, ``b`` its twin on
    same-day weather, ``c`` and ``d`` a refit through the week on each. Each
    factor is averaged over both orders (the Shapley split), so the three add
    up to ``a``. Works on numbers and on arrays alike.
    """
    return 0.5 * ((a - b) + (c - d)), 0.5 * ((a - c) + (b - d)), d


def city_weeks(weeks: pd.DataFrame, city: str) -> pd.DataFrame:
    """One city's weeks: when each ended, whether the loop retrained, and its parts.

    Parts are per hour, in squared µg/m³, so a week with a missing hour is not
    drawn smaller than a full one. The largest part is the one the split names.
    """
    rows = weeks[weeks["city"] == city].sort_values("week_end")
    hours = rows["hours"].astype(float)
    weather, relation, unexplained = parts(rows["A"], rows["B"], rows["C"], rows["D"])
    return pd.DataFrame({
        "week_end": pd.to_datetime(rows["week_end"]),
        "retrained": rows["retrained"].astype(str) == "True",
        "weather": weather / hours,
        "relation": relation / hours,
        "unexplained": unexplained / hours,
        "error": rows["A"].astype(float) / hours,
        "largest": rows["largest"],
    }).reset_index(drop=True)


def load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """``weeks.csv`` and ``cities.csv`` from ``root``, or ``None`` where either is missing."""
    weeks_path, cities_path = root / "weeks.csv", root / "cities.csv"
    if not (weeks_path.exists() and cities_path.exists()):
        return None
    return pd.read_csv(weeks_path), pd.read_csv(cities_path)
