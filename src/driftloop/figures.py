"""Every published figure, from the outputs that produced it, and a check that prose agrees.

The same numbers are restated across the README, evaluation, decisions and
methodology pages, and those pages are rewritten in place. A figure that moves
under a re-run has to be found and edited in every one of them by hand, and
missing one leaves a page asserting something the outputs no longer say.

So a figure in prose carries its key, in a comment that does not render::

    Delhi runs **+49.4% [+34, +62]**<!--fig:delhi.acted--> better week by week.

``check_text`` reads the number and interval written before each marker and
checks them against ``outputs/figures.json`` at the precision the prose chose:
``+49.4`` has to be the canonical value rounded to one decimal, ``+34`` to none.
The prose keeps its own rounding, and cannot keep a stale value.

An interval no output file holds is declared rather than keyed, with the reason
where the next reader will see it::

    +21.9% [−6.9, +32.3]<!--nofig: printed by sweep_promotion_confidence.py-->

``figures.json`` is written by ``scripts/figures.py`` from the local outputs, and
is committed because most of those outputs are not: CI checks the prose against
the committed file, and only a machine holding the outputs can check the file.
Out of scope: formatting a figure for a page (the site reads ``data.json``), and
any figure the outputs do not hold.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from driftloop import stats

FIGURES_FILE = "figures.json"

CITY_SLUGS = {
    "Delhi": "delhi",
    "Santiago": "santiago",
    "Kraków": "krakow",
    "Johannesburg": "joburg",
    "Melbourne": "melbourne",
    "Los Angeles": "la",
}

# --------------------------------------------------------------------------- #
# Collecting: outputs/ -> {key: value}                                          #
# --------------------------------------------------------------------------- #


def _put(figures: dict[str, float], key: str, point: float, lo: float | None = None,
         hi: float | None = None) -> None:
    figures[key] = point
    if lo is not None and hi is not None:
        figures[f"{key}.lo"] = lo
        figures[f"{key}.hi"] = hi


def _interval(figures: dict[str, float], key: str, block: dict) -> None:
    _put(figures, key, block["point"], block["lo"], block["hi"])


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _uncertainty(data: dict, figures: dict[str, float]) -> None:
    for city in data["cities"]:
        slug = CITY_SLUGS[city["label"]]
        figures[f"{slug}.weeks"] = city["weeks"]
        figures[f"{slug}.retrains"] = city["retrains"]
        figures[f"{slug}.promotions"] = city["promotions"]
        _interval(figures, f"{slug}.replay", city["interval"]["across_replay"])
        acted = city["interval"]["when_it_acted"]
        _interval(figures, f"{slug}.acted", acted)
        figures[f"{slug}.acted.n"] = acted["n"]
        figures[f"{slug}.acted.n_eff"] = acted["n_effective"]
        # Win rates carry Wilson's interval, not the bootstrap's: see
        # stats.wilson_interval for why the bootstrap fails at 100%.
        won = city["point"]["win_rate"]
        lo, hi = stats.wilson_interval(round(won * acted["n"] / 100), acted["n"])
        _put(figures, f"{slug}.won", won, lo, hi)

    gate = data["gate"]
    for length in ("short", "long"):
        summary = gate["summary"][length]
        figures[f"gate.{length}.n"] = summary["n"]
        figures[f"gate.{length}.harmful"] = summary["harmful"]
        for part in ("exam", "delivered"):
            block = gate["interval"][length][part]
            # An interval over three promotions is infinite, and not a figure.
            bounded = abs(block["lo"]) != float("inf")
            _put(figures, f"gate.{length}.{part}", block["point"],
                 block["lo"] if bounded else None, block["hi"] if bounded else None)


def _premium(figures: dict[str, float], key: str, row: dict[str, str]) -> None:
    """A premium and its interval. None where nothing was promoted; no interval where
    one window cannot bound it."""
    if not row["premium"]:
        return
    lo, hi = float(row["premium_lo"]), float(row["premium_hi"])
    bounded = abs(lo) != float("inf")
    _put(figures, key, float(row["premium"]), lo if bounded else None, hi if bounded else None)


def _ablation(rows: list[dict[str, str]], figures: dict[str, float]) -> None:
    arms: dict[str, dict[str, float]] = {}
    for row in rows:
        slug = CITY_SLUGS[row["city"]]
        _premium(figures, f"ablation.{slug}.{row['kind']}", row)
        figures[f"ablation.{slug}.{row['kind']}.rmse"] = float(row["median_rmse"])
        arms.setdefault(slug, {})[row["kind"]] = float(row["premium"])
    # The decomposition the README quotes: what the shipped alpha costs, then
    # what linearity costs, each in points of premium.
    for slug, premium in arms.items():
        figures[f"ablation.{slug}.alpha_points"] = premium["ridge"] - premium["ridge_tuned"]
        figures[f"ablation.{slug}.linearity_points"] = premium["ridge_tuned"] - premium["gbm_tuned"]


# Each mechanism sweep: its output file, the key prefix, and the column naming the arm.
SWEEPS = (
    ("skill_floor_sweep.csv", "floor", "floor"),
    ("recertify_sweep.csv", "recertify", "cadence"),
    ("promotion_confidence_sweep.csv", "confidence", "confidence"),
    ("probation_sweep.csv", "probation", "probation"),
)


def _sweep(rows: list[dict[str, str]], prefix: str, arm: str, figures: dict[str, float]) -> None:
    """Every arm against the ``off`` arm, over every week and over the weeks it acted."""
    for row in rows:
        key = f"{prefix}.{CITY_SLUGS[row['city']]}.{row[arm]}"
        figures[f"{key}.retrains"] = int(row["retrains"])
        figures[f"{key}.promotions"] = int(row["promotions"])
        figures[f"{key}.differing"] = int(row["differing_weeks"])
        if "longest_silence" in row:
            figures[f"{key}.silence"] = int(row["longest_silence"])
        if row["vs_off_acted"]:
            _put(figures, f"{key}.acted", float(row["vs_off_acted"]),
                 float(row["vs_off_acted_lo"]), float(row["vs_off_acted_hi"]))
            _put(figures, f"{key}.all", float(row["vs_off"]),
                 float(row["vs_off_lo"]), float(row["vs_off_hi"]))


def _rebaseline(rows: list[dict[str, str]], figures: dict[str, float]) -> None:
    for row in rows:
        key = f"rebaseline.{CITY_SLUGS[row['city']]}.{row['arm']}"
        _premium(figures, key, row)
        figures[f"{key}.rmse"] = float(row["median_rmse"])
        figures[f"{key}.retrains"] = int(row["retrains"])
        figures[f"{key}.promotions"] = int(row["promotions"])
        figures[f"{key}.fires"] = int(row["fires"])
        figures[f"{key}.silence"] = int(row["longest_silence"])


def collect(outputs: Path) -> dict[str, float]:
    """Every keyed figure, read from the output files that hold it.

    Raises FileNotFoundError when an output is missing: a partial set would
    write a figures file that silently drops the keys prose still cites.
    """
    figures: dict[str, float] = {}
    _uncertainty(json.loads((outputs / "uncertainty.json").read_text(encoding="utf-8")), figures)
    _ablation(_rows(outputs / "model_ablation.csv"), figures)
    for name, prefix, arm in SWEEPS:
        _sweep(_rows(outputs / name), prefix, arm, figures)
    _rebaseline(_rows(outputs / "rebaseline.csv"), figures)
    return {
        key: value if isinstance(value, int) else round(float(value), 4)
        for key, value in sorted(figures.items())
    }


def documents(root: Path) -> list[Path]:
    """Every tracked page whose figures are checked. HISTORY.md is left out because it
    quotes superseded figures on purpose: they are the record, not a claim."""
    listed = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    return [root / name for name in listed if name != "docs/HISTORY.md"]


def load(outputs: Path) -> dict[str, float]:
    """The committed figures, as ``scripts/figures.py`` last wrote them."""
    return json.loads((outputs / FIGURES_FILE).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Checking: prose -> the figures it cites                                       #
# --------------------------------------------------------------------------- #

_NUM = r"[−+-]?\d+(?:\.\d+)?"
_UNIT = r"(?:\s*(?:µg/m³|points?|weeks?|w))?"
MARKER = re.compile(r"<!--\s*(?:fig:(?P<key>[\w.+\-]+)|nofig:[^>]*?)\s*-->")
# The figure written immediately before a marker: a point, an interval, or both,
# optionally bold. Anchored at the end so it reads the last figure, not the first.
_BEFORE = re.compile(
    rf"(?:(?P<point>{_NUM})%?{_UNIT}\s*)?"
    rf"(?:\[(?P<lo>{_NUM})%?,\s*(?P<hi>{_NUM})%?\])?[*_]*\s*\Z"
)
_INTERVAL = re.compile(rf"(?:{_NUM}%?\s*)?\[{_NUM}%?,\s*{_NUM}%?\][*_]*")


@dataclass(frozen=True)
class Problem:
    line: int
    key: str
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.key}: {self.message}"


def _number(text: str) -> tuple[float, int]:
    """The value written, and how many decimals it was written to."""
    text = text.replace("−", "-")
    decimals = len(text.split(".")[1]) if "." in text else 0
    return float(text), decimals


def _agrees(written: str, canonical: float) -> bool:
    value, decimals = _number(written)
    return abs(value - canonical) <= 0.5 * 10**-decimals + 1e-9


def check_text(text: str, figures: dict[str, float]) -> list[Problem]:
    """Every keyed figure in ``text`` that disagrees with ``figures``, or cannot be read."""
    problems = []
    for match in MARKER.finditer(text):
        key = match.group("key")
        if key is None:
            continue
        line = text.count("\n", 0, match.start()) + 1
        before = text[text.rfind("\n", 0, match.start()) + 1 : match.start()]
        # Stop at the previous marker, so two figures on one line stay apart.
        previous = list(MARKER.finditer(before))
        if previous:
            before = before[previous[-1].end() :]
        written = _BEFORE.search(before)
        if written is None or (written.group("point") is None and written.group("lo") is None):
            problems.append(Problem(line, key, "no figure written before the marker"))
            continue
        for part, suffix in (("point", ""), ("lo", ".lo"), ("hi", ".hi")):
            value = written.group(part)
            if value is None:
                continue
            if key + suffix not in figures:
                problems.append(Problem(line, key + suffix, "no such figure"))
            elif not _agrees(value, figures[key + suffix]):
                problems.append(Problem(
                    line, key + suffix,
                    f"prose says {value}, outputs say {figures[key + suffix]:g}",
                ))
    return problems


def unbound(text: str) -> list[tuple[int, str]]:
    """Every interval in ``text`` that carries neither a key nor a declared reason."""
    found = []
    for match in _INTERVAL.finditer(text):
        if not MARKER.match(text, match.end()):
            found.append((text.count("\n", 0, match.start()) + 1, match.group()))
    return found
