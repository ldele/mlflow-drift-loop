"""The rules that pick the held-out cities and their replay windows.

Both are fixed by docs/registrations/2026-11-10-second-alarm-on-held-out-cities.md
before any candidate's data is read, so a change here is a change to a registered
design and should fail loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]

import held_out_cities as held  # noqa: E402

from driftloop.config import PROFILES, TARGET, TIMESTAMP  # noqa: E402


def _timeline(clean_from: str, clean_days: int = 90) -> pd.DataFrame:
    """Hourly PM2.5 at 50, with one clean stretch at 5."""
    hours = pd.date_range(held.SPAN_ORIGIN, held.SPAN_HORIZON, freq="h", inclusive="left")
    pm25 = np.full(len(hours), 50.0)
    start = pd.Timestamp(clean_from)
    pm25[(hours >= start) & (hours < start + pd.Timedelta(clean_days, unit="D"))] = 5.0
    return pd.DataFrame({TIMESTAMP: hours, TARGET: pm25})


def test_the_first_model_trains_on_the_cleanest_stretch():
    plan = held.replay_windows(_timeline("2025-08-01"))
    assert plan.champion_train_start >= pd.Timestamp("2025-08-01")
    assert plan.champion_train_end <= pd.Timestamp("2025-08-01") + pd.Timedelta(90, unit="D")
    assert plan.champion_train_end - plan.champion_train_start == pd.Timedelta(75, unit="D")


def test_the_earliest_of_equally_clean_windows_wins():
    # Fifteen spare days of clean air, so sixteen windows tie at the minimum.
    plan = held.replay_windows(_timeline("2025-08-01"))
    assert plan.champion_train_start == pd.Timestamp("2025-08-01")


def test_a_clean_season_outside_the_allowed_range_is_not_reached_for():
    plan = held.replay_windows(_timeline("2026-01-01"))
    assert held.TRAIN_END_FIRST <= plan.champion_train_end <= held.TRAIN_END_LAST


def test_the_replay_starts_clear_of_its_training_data_and_runs_long_enough():
    plan = held.replay_windows(_timeline("2025-09-10"))
    monitor = PROFILES["openmeteo"].loop.monitor_days
    assert (plan.first_run - plan.champion_train_end).days >= monitor
    assert (plan.last_run - plan.first_run).days // plan.step_days + 1 >= 30


def _off_arm(fired: list[bool], skill: list[float], silence: int) -> dict:
    return {"_fired": fired, "_skill": skill, "longest_silence": silence}


def test_a_deaf_first_alarm_needs_both_a_long_silence_and_quiet_weeks_below_the_floor():
    quiet, bad = [False] * 12, [-0.9] * 12
    assert held.deafness(_off_arm(quiet, bad, 12))["deaf"]
    # A long silence while the model is fine: the second alarm has nothing to do.
    assert not held.deafness(_off_arm(quiet, [0.2] * 12, 12))["deaf"]
    # A bad model the first alarm keeps catching: it is not deaf.
    assert not held.deafness(_off_arm([True] * 12, bad, 0))["deaf"]
    # Below the floor, but for too few weeks.
    assert not held.deafness(_off_arm(quiet, [-0.9, -0.9] + [0.2] * 10, 12))["deaf"]


def test_a_week_the_first_alarm_fired_in_does_not_count_as_quiet():
    fired = [True, True, True] + [False] * 10
    skill = [-0.9, -0.9, -0.9] + [0.1] * 10
    verdict = held.deafness(_off_arm(fired, skill, 10))
    assert verdict["quiet_weeks_below_floor"] == 0 and not verdict["deaf"]


def test_the_candidates_are_twelve_cities_the_study_has_not_used():
    used = {p.location.country for p in PROFILES.values() if p.location is not None}
    countries = [c.country for c in held.CANDIDATES]
    assert len(held.CANDIDATES) == 12
    assert len(set(countries)) == 12, "one city per country"
    assert not used & set(countries)


def test_only_the_shipped_setting_is_tested():
    assert held.FLOOR == PROFILES["openmeteo"].loop.skill_floor
