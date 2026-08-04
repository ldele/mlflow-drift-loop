"""The published site's local references resolve.

Cheap, and it guards a failure mode with no other alarm on it: the stylesheet
was inline in ``index.html`` until a second page needed it, and a mistyped
``href`` after that split would serve both pages completely unstyled while every
other test still passed. Same for the scripts, and for the cross-links between
the two pages.

Only local paths are checked. The Plotly CDN tag is deliberately skipped: a test
that reaches the network fails on a train.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parents[1] / "site"
PAGES = ["index.html", "compare.html"]

# src="..." / href="..." that are not absolute URLs, protocol-relative, anchors,
# or mailto: links.
LOCAL_REF = re.compile(r'(?:src|href)="(?!https?:|//|#|mailto:)([^"]+)"')


@pytest.mark.parametrize("page", PAGES)
def test_page_exists(page: str) -> None:
    assert (SITE / page).is_file(), f"{page} is missing from site/"


@pytest.mark.parametrize("page", PAGES)
def test_every_local_reference_resolves(page: str) -> None:
    html = (SITE / page).read_text(encoding="utf-8")
    refs = sorted(set(LOCAL_REF.findall(html)))
    assert refs, f"{page} references nothing local, which means the regex stopped matching"

    # data.json and the per-city CSVs are build outputs rather than committed
    # source, so their absence is a "run build_site.py" state and not a broken
    # link. Everything else has to be on disk.
    generated = {"data.json"}
    missing = [
        ref for ref in refs
        if ref not in generated and not ref.endswith(".csv") and not (SITE / ref).is_file()
    ]
    assert not missing, f"{page} points at files that do not exist: {missing}"


def test_the_two_pages_link_to_each_other() -> None:
    """Either page is a dead end without the other."""
    index = (SITE / "index.html").read_text(encoding="utf-8")
    compare = (SITE / "compare.html").read_text(encoding="utf-8")
    assert 'href="compare.html"' in index, "index.html never offers the all-cities view"
    assert 'href="index.html"' in compare, "compare.html has no way back"


def test_both_pages_share_one_stylesheet() -> None:
    """Two copies of the palette drift apart; the theme variables exist so a
    colour is defined once."""
    for page in PAGES:
        html = (SITE / page).read_text(encoding="utf-8")
        assert 'href="shared.css"' in html, f"{page} does not use the shared stylesheet"
        assert "<style>" not in html, f"{page} has reintroduced inline CSS"

    css = (SITE / "shared.css").read_text(encoding="utf-8")
    # The tokens both pages' JS palettes are written against.
    for token in ("--surface", "--ink", "--muted", "--good", "--warn", "--crit"):
        assert token in css, f"shared.css is missing {token}"
