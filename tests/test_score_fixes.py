"""The rule that reads the fix sweeps: two cities, same direction, or it is not a verdict."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]

import score_fixes  # noqa: E402

COLUMNS = ["city", "confidence", "paired", "vs_off_acted", "vs_off_acted_real"]


def _sweep(tmp_path: Path, rows: list[tuple]) -> Path:
    with (tmp_path / "promotion_confidence_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return tmp_path


def test_one_city_clearing_zero_is_not_a_verdict(tmp_path):
    folder = _sweep(tmp_path, [
        ("A", "off", 5.0, "", ""), ("B", "off", 5.0, "", ""),
        ("A", "0.95", 4.0, -2.5, "True"), ("B", "0.95", 5.0, -1.0, "False"),
    ])
    result = score_fixes.read_sweep(folder, "confidence-aware gate")
    assert result["verdict"] == "nothing measurable"
    assert (result["intervals"], result["harmed"], result["helped"]) == (2, 1, 0)


def test_two_cities_in_the_same_direction_at_one_setting_is(tmp_path):
    folder = _sweep(tmp_path, [
        ("A", "0.95", 4.0, -2.5, "True"), ("B", "0.95", 5.0, -1.0, "True"),
    ])
    assert score_fixes.read_sweep(folder, "confidence-aware gate")["verdict"] == "harmful"


def test_the_same_city_at_two_settings_does_not_add_up_to_two(tmp_path):
    # Santiago at 0.90 and at 0.95 was one replay counted twice.
    folder = _sweep(tmp_path, [
        ("A", "0.90", 4.0, -2.5, "True"), ("A", "0.95", 4.0, -2.5, "True"),
    ])
    assert score_fixes.read_sweep(folder, "confidence-aware gate")["verdict"] == "nothing measurable"


def test_an_arm_that_changed_nothing_has_no_interval(tmp_path):
    folder = _sweep(tmp_path, [("A", "0.80", 5.0, "", "")])
    assert score_fixes.read_sweep(folder, "confidence-aware gate")["intervals"] == 0
