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


def test_the_published_payload_never_shows_an_estimate_without_its_interval() -> None:
    """A number on the page carries its range, the same rule the docs follow.

    Printing Melbourne's +1.2% in the same green as Delhi's +49.4% claims two
    findings where there is one, which is the error `docs/evaluation.md` is
    written against. This makes it hard to reintroduce on the page while the
    documents say otherwise.

    Skipped when `data.json` has not been built, which is a "run build_site.py"
    state rather than a failure.
    """
    import json

    payload = SITE / "data.json"
    if not payload.is_file():
        pytest.skip("site/data.json not built")

    data = json.loads(payload.read_text(encoding="utf-8"))
    profiles = data["profiles"] if isinstance(data, dict) and "profiles" in data else data
    items = profiles.items() if isinstance(profiles, dict) else [(p["key"], p) for p in profiles]

    bare = []
    for key, profile in items:
        stats = profile.get("stats") or {}
        if stats.get("retrain_acted") is None:
            continue
        for field in ("retrain_acted_lo", "retrain_acted_hi", "retrain_acted_real"):
            if stats.get(field) is None:
                bare.append(f"{key}.{field}")
    assert not bare, f"published estimates with no interval attached: {bare}"


def test_at_least_one_city_is_published_as_not_distinguishable_from_zero() -> None:
    """Kraków and Melbourne do not clear zero, and the page has to say so.

    A guard against the interval fields being present but always true, which
    would pass the test above while telling the reader nothing. If a rerun ever
    makes every city a finding, this fails and the claim gets re-checked by a
    human rather than quietly widened.
    """
    import json

    payload = SITE / "data.json"
    if not payload.is_file():
        pytest.skip("site/data.json not built")

    data = json.loads(payload.read_text(encoding="utf-8"))
    profiles = data["profiles"] if isinstance(data, dict) and "profiles" in data else data
    items = profiles.values() if isinstance(profiles, dict) else profiles
    judged = [p["stats"] for p in items if (p.get("stats") or {}).get("retrain_acted_real") is not None]
    assert judged, "no city carries a verdict at all"
    assert not all(s["retrain_acted_real"] for s in judged), (
        "every city now reads as a finding; re-check before publishing"
    )
