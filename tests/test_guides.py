"""The chart readings stay attached to the charts they describe.

``driftloop.guides`` is prose, so nothing in it can fail at runtime and nothing
here checks that it is *good*. What these check is that it is still *connected*:
a chart with no reading renders a bare plot, a reading whose chart was deleted
renders nowhere at all, and neither shows up as an error in either interface.

The two interfaces are checked differently because they fail differently. The
site is JavaScript, so the reference is a string this file greps for. The
dashboard is Python drawing charts imperatively, so the check is structural:
every ``plotly_chart`` call must be followed by a panel.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from driftloop import guides
from driftloop.config import LoopConfig

REPO = Path(__file__).resolve().parents[1]
DASHBOARD = REPO / "dashboard" / "app.py"
SITE = REPO / "site"

# The one chart drawn in a loop, so one panel covers the whole grid rather than
# repeating under each of six histograms. Keyed by the figure builder, because
# the line number moves and the builder does not.
SHARED_PANEL = ("theme.hist_overlay",)


SPAN = "Aug 2025 to Jul 2026"


@pytest.fixture(scope="module")
def thresholds() -> dict:
    """What the site builder fills: the loop's settings, and nothing about a city."""
    return guides.context(
        LoopConfig(), horizon_days=7, climatology_days=30,
        psi_bands={"stable": 0.1, "significant": 0.25}, span=SPAN,
    )


@pytest.fixture(scope="module")
def values(thresholds) -> dict:
    """What the dashboard fills: the settings and one city's facts."""
    runs = list(pd.date_range("2025-08-15", "2026-07-10", freq="7D"))
    city = guides.city_context(
        "Kraków", runs, 14, (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-08-01"))
    )
    return {**thresholds, **city}


def _parts(guide: dict) -> list[str]:
    return [guide["shows"], guide["data"], guide["units"],
            *(side for pair in guide["read"] for side in pair)]


def _dashboard_keys() -> set[str]:
    src = DASHBOARD.read_text(encoding="utf-8")
    keys = set(re.findall(r'guide\(\s*"([a-z_]+)"\s*\)', src))
    # The knob sweep passes its key through a loop variable, so the literals sit
    # in the tuple it iterates rather than at the call.
    keys |= set(re.findall(r'"[a-z_]+",\s*"(control_\w+)"', src))
    return keys


def _site_keys() -> set[str]:
    keys: set[str] = set()
    for page in ("app.js", "compare.js"):
        src = (SITE / page).read_text(encoding="utf-8")
        keys |= set(re.findall(r'guide:\s*"([a-z_]+)"', src))
        keys |= set(re.findall(r'guideHTML\(\s*"([a-z_]+)"', src))
    return keys


def test_every_reading_is_reachable_from_an_interface() -> None:
    """No prose about a chart that no longer exists.

    A guide nothing references is invisible: it costs nothing at runtime and is
    quietly wrong forever, which is the failure mode this whole module was
    written to avoid on the other side.
    """
    orphans = set(guides.GUIDES) - _site_keys() - _dashboard_keys()
    assert not orphans, f"readings nothing draws: {sorted(orphans)}"


def test_neither_interface_names_a_reading_that_does_not_exist() -> None:
    """A renamed key renders an empty panel rather than an error."""
    for label, keys in (("site", _site_keys()), ("dashboard", _dashboard_keys())):
        missing = keys - set(guides.GUIDES)
        assert not missing, f"{label} asks for readings that do not exist: {sorted(missing)}"


def test_the_single_interface_lists_are_accurate() -> None:
    """``SITE_ONLY`` and ``DASHBOARD_ONLY`` describe where charts actually are.

    They are documentation that can go stale, so they are asserted rather than
    trusted: moving a chart between the two interfaces has to update them.
    """
    site, dash = _site_keys(), _dashboard_keys()
    assert guides.SITE_ONLY == site - dash, (
        f"SITE_ONLY says {sorted(guides.SITE_ONLY)}, the code says {sorted(site - dash)}"
    )
    assert guides.DASHBOARD_ONLY == dash - site, (
        f"DASHBOARD_ONLY says {sorted(guides.DASHBOARD_ONLY)}, the code says {sorted(dash - site)}"
    )


def test_every_dashboard_chart_has_a_panel_under_it() -> None:
    """Structural, because the dashboard draws charts imperatively.

    Walks each ``plotly_chart`` call to the end of its statement and requires a
    ``guide(...)`` within the next few lines. That is what catches the case this
    file exists for: someone adds a chart and does not add its reading.
    """
    lines = DASHBOARD.read_text(encoding="utf-8").splitlines()
    unguarded = []
    for i, line in enumerate(lines):
        if ".plotly_chart(" not in line:
            continue
        depth, end = 0, i
        for end in range(i, len(lines)):
            depth += lines[end].count("(") - lines[end].count(")")
            if depth <= 0:
                break
        following = " ".join(lines[end + 1:end + 5])
        preceding = " ".join(lines[max(0, i - 6):i])
        if "guide(" in following or any(s in preceding for s in SHARED_PANEL):
            continue
        unguarded.append(f"line {i + 1}: {line.strip()[:60]}")
    assert not unguarded, "dashboard charts with no reading under them:\n" + "\n".join(unguarded)


def test_the_thresholds_in_the_prose_come_from_the_config() -> None:
    """A reading quoting "1.25x" as a literal would survive a change to the loop.

    Filling against a config with nothing at its default is what catches that:
    if the placeholders are live, none of the shipped values can survive.
    """
    odd = LoopConfig(monitor_days=99, holdout_days=97, perf_drift_threshold=9.5,
                     promotion_margin=0.91)
    filled = guides.payload(guides.context(
        odd, horizon_days=3, climatology_days=95,
        psi_bands={"stable": 0.93, "significant": 0.94}, span=SPAN,
    ))
    blob = " ".join(part for guide in filled.values() for part in _parts(guide))
    for shipped in ("1.25", "14-day", "0.25 shifted", "more than 5%"):
        assert shipped not in blob, (
            f"{shipped!r} survives a config change, so it is a literal and not a placeholder"
        )
    assert "9.5" in blob and "99-day" in blob and "91%" in blob, (
        "placeholders did not take the new config"
    )


def test_no_placeholder_is_left_unfilled(values) -> None:
    """A typo'd placeholder would reach the page as text in braces."""
    for key, guide in guides.payload(values).items():
        for part in _parts(guide):
            assert "{" not in part and "}" not in part, f"{key} still holds a placeholder: {part}"


def test_the_site_payload_leaves_only_the_city_facts_to_fill(thresholds) -> None:
    """The builder fills the loop's settings; the browser fills what differs by city.

    Anything else left in braces is a placeholder nobody will ever fill.
    """
    tokens = set()
    for guide in guides.payload(thresholds).values():
        for part in _parts(guide):
            tokens |= set(re.findall(r"\{(\w+)\}", part))
    assert tokens, "no reading names its city or its period"
    assert tokens <= set(guides.CITY_TOKENS), f"nothing fills {sorted(tokens - set(guides.CITY_TOKENS))}"


def test_every_reading_has_all_four_parts(values) -> None:
    """What it shows, the data behind it, the units, and how to read it."""
    for key, guide in guides.payload(values).items():
        assert len(guide["shows"]) > 40, f"{key}: 'shows' is too short to describe a chart"
        assert guide["data"], f"{key}: nothing on the data"
        assert guide["units"], f"{key}: nothing on the units"
        assert len(guide["read"]) >= 2, f"{key}: fewer than two ways to read it"


def test_a_reading_stays_short(values) -> None:
    """These replaced panels three times as long. A reading is a caption, not an essay."""
    for key, guide in guides.payload(values).items():
        words = sum(len(part.split()) for part in _parts(guide))
        assert words <= 150, f"{key}: {words} words"


def test_a_city_reading_states_its_exact_period(values) -> None:
    body = guides.markdown("trigger", values)
    assert "15 Aug 2025 to 10 Jul 2026 (48 runs)" in body
    assert "Kraków" in body and "RMSE in µg/m³" in body
    assert "26 Jun 2026 to 10 Jul 2026" in guides.markdown("forecast", values)
    assert "1 Jun 2025 to 1 Aug 2025" in guides.markdown("factors", values)


def test_a_profile_with_no_runs_still_reads() -> None:
    city = guides.city_context("Live schedule", [], 14)
    assert city["runs"] == "0" and "{" not in " ".join(city.values())


def test_the_prose_is_markdown_and_not_html(values) -> None:
    """One string has to satisfy Streamlit and the browser.

    Streamlit renders Markdown and escapes HTML; the site escapes HTML and then
    converts the two Markdown constructs. An entity or a tag written into the
    prose would show up literally in the dashboard.
    """
    tag = re.compile(r"</?[a-zA-Z]")
    entity = re.compile(r"&[a-zA-Z]+;|&#\d+;")
    for key, guide in guides.payload(values).items():
        for part in _parts(guide):
            assert not tag.search(part), f"{key} contains an HTML tag: {part[:80]}"
            assert not entity.search(part), f"{key} contains an HTML entity: {part[:80]}"
            # A leading ">" is a Markdown blockquote, which is not what any of
            # these mean. Bare comparisons belong inside a code span.
            assert not part.lstrip().startswith(">"), f"{key} opens with a blockquote"
