"""Every page quoting a figure agrees with the outputs that produced it.

The pages are rewritten in place and restate the same figures in several places,
so a re-run that moves a number leaves every restatement to be found by hand.
These tests make a missed one a red build rather than a page that quietly
asserts what the outputs no longer say.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from driftloop import figures

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"

# Pages every interval on which is keyed or declared.
FULLY_KEYED = (
    "README.md", "docs/DECISIONS.md", "docs/evaluation.md", "docs/evaluation-first-loop.md",
    "docs/methodology.md",
)


@pytest.fixture(scope="module")
def committed() -> dict[str, float]:
    return figures.load(OUTPUTS)


def test_every_keyed_figure_agrees_with_the_committed_figures(committed):
    problems = [
        f"{path.relative_to(REPO_ROOT)}: {problem}"
        for path in figures.documents(REPO_ROOT)
        for problem in figures.check_text(path.read_text(encoding="utf-8"), committed)
    ]
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", FULLY_KEYED)
def test_every_interval_is_keyed_or_declared(name):
    found = figures.unbound((REPO_ROOT / name).read_text(encoding="utf-8"))
    assert not found, f"{name}: key these or declare them <!--nofig: why-->: {found}"


def test_history_is_not_checked():
    # It quotes the figures each correction replaced, which the outputs no longer hold.
    assert REPO_ROOT / "docs/HISTORY.md" not in figures.documents(REPO_ROOT)


def test_committed_figures_match_the_outputs(committed):
    if not (OUTPUTS / "uncertainty.json").exists():
        pytest.skip("outputs not built here; run scripts/figures.py where they are")
    assert figures.collect(OUTPUTS) == committed, "stale: run python scripts/figures.py"


# --------------------------------------------------------------------------- #
# The check itself, on text it must reject.                                     #
# --------------------------------------------------------------------------- #

FIGS = {"x": 49.3507, "x.lo": 33.7378, "x.hi": 61.8067, "n": 39}


@pytest.mark.parametrize("written", ["+49.35%", "+49.4%", "+49%", "49.4", "**+49.4%**"])
def test_a_figure_agrees_at_whatever_precision_it_is_written(written):
    assert figures.check_text(f"Delhi runs {written}<!--fig:x--> better.", FIGS) == []


@pytest.mark.parametrize("written", ["+49.3%", "+50%", "+49.36%", "−49.4%"])
def test_a_stale_or_misrounded_figure_is_caught(written):
    (problem,) = figures.check_text(f"Delhi runs {written}<!--fig:x--> better.", FIGS)
    assert problem.key == "x"


def test_each_bound_of_an_interval_is_checked():
    (problem,) = figures.check_text("**+49.4% [+34, +61]**<!--fig:x-->", FIGS)
    assert problem.key == "x.hi" and "+61" in problem.message


def test_unicode_minus_is_a_minus():
    assert figures.check_text("−49.4%<!--fig:y-->", {"y": -49.35}) == []


def test_two_figures_on_one_line_are_read_separately():
    text = "from 30<!--fig:a--> weeks to 5<!--fig:b-->."
    assert figures.check_text(text, {"a": 30, "b": 5}) == []
    (problem,) = figures.check_text(text, {"a": 30, "b": 6})
    assert problem.key == "b"


def test_a_figure_carrying_a_unit_is_read():
    assert figures.check_text("45.8 µg/m³<!--fig:a-->, 35w<!--fig:b-->", {"a": 45.8, "b": 35}) == []


def test_an_unknown_key_is_reported_not_skipped():
    (problem,) = figures.check_text("+49.4%<!--fig:nope-->", FIGS)
    assert problem.message == "no such figure"


def test_a_marker_with_no_figure_before_it_is_reported():
    (problem,) = figures.check_text("Delhi<!--fig:x-->", FIGS)
    assert "no figure" in problem.message


def test_an_interval_is_unbound_until_keyed_or_declared():
    assert figures.unbound("won by +21.9% [−6.9, +32.3] once") == [(1, "+21.9% [−6.9, +32.3]")]
    assert figures.unbound("won by +21.9% [−6.9, +32.3]<!--fig:x--> once") == []
    assert figures.unbound("won by +21.9% [−6.9, +32.3]<!--nofig: printed only--> once") == []


def test_a_declaration_is_not_checked():
    assert figures.check_text("+21.9% [−6.9, +32.3]<!--nofig: printed only-->", {}) == []


def test_the_first_shipped_figures_are_kept_beside_the_current_ones(committed):
    """The pages quote what they said before 2026-11-10 next to what they say now,
    so both sets have to be in the file the pages are checked against."""
    for city in figures.CITY_SLUGS.values():
        assert f"{city}.acted" in committed
        assert f"{figures.FIRST}{city}.acted" in committed
    assert committed["first.delhi.acted"] != committed["delhi.acted"]


def test_the_held_out_cities_are_keyed_by_a_plain_slug(committed):
    assert figures._city_slug("São Paulo") == "sao_paulo"
    # Both arms' silences for a city the alarm fired in, and none it did not reach.
    assert committed["heldout.istanbul.off.silence"] > committed["heldout.istanbul.on.silence"]
    assert not any(key.startswith("heldout.lagos") for key in committed)


def test_the_held_out_premium_is_keyed_with_its_interval_and_its_score(committed):
    walked = ("tokyo", "shanghai", "sao_paulo", "mexico_city", "cairo", "dhaka", "karachi",
              "buenos_aires", "istanbul")
    for city in walked:
        for suffix in ("premium", "premium.lo", "premium.hi", "premium.n", "swing"):
            assert f"heldout.{city}.{suffix}" in committed, f"heldout.{city}.{suffix}"
    # The six carry their swing under their own slugs, and the threshold is the
    # smallest swing among the four where retraining pays.
    paying = [committed[f"swing.{c}"] for c in ("delhi", "santiago", "joburg", "melbourne")]
    assert committed["heldout.threshold"] == pytest.approx(min(paying), abs=1e-4)
    assert 0 <= committed["heldout.p1.correct"] <= 9
    assert 0 <= committed["heldout.p1.chance"] <= 1
