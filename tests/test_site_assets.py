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


def test_the_ablation_publishes_its_validity_check_with_the_margin() -> None:
    """The section that can take the headline away carries its own caveats.

    The ablation compares model *classes* only where the tuned tree is really the
    better model; where it is not, both arms are misspecified and the comparison
    says nothing about linearity. A page that prints the premiums without that
    check invites the reader to conclude something the experiment did not
    establish, which is the exact mistake an earlier version of this analysis
    made.

    The margin, not just the boolean: it is 0.02 µg/m³ in Los Angeles, and a
    check that passes by 0.02 should be visible as one.
    """
    import json

    payload = SITE / "data.json"
    if not payload.is_file():
        pytest.skip("site/data.json not built")

    ablation = json.loads(payload.read_text(encoding="utf-8")).get("ablation")
    if not ablation:
        pytest.skip("outputs/model_ablation.csv not built")

    for city in ablation["cities"]:
        check = city.get("tree_is_better")
        assert check and check.get("margin") is not None, (
            f"{city['city']} publishes premiums with no model-quality check"
        )
        for arm in city["arms"]:
            missing = [f for f in ("lo", "hi", "real") if arm.get(f) is None]
            assert not missing, f"{city['city']}/{arm['kind']} premium is missing {missing}"


def test_the_committed_ablation_still_matches_the_loop_it_describes() -> None:
    """The ablation snapshot has not gone stale against the rest of the page.

    The replay behind this section is expensive and does not change week to week,
    so ``outputs/model_ablation*`` is committed and the weekly rebuild reads it
    rather than re-running it. Change a loop threshold and every other number on
    the page moves while these do not.

    ``build_site`` re-derives the untouched Ridge arm's premium and compares it
    against the premium it measured for the same city on the same run; the page
    refuses to draw the charts where they disagree. This fails the build instead,
    so the disagreement is caught before it is published rather than after.
    """
    import json

    payload = SITE / "data.json"
    if not payload.is_file():
        pytest.skip("site/data.json not built")

    ablation = json.loads(payload.read_text(encoding="utf-8")).get("ablation")
    if not ablation:
        pytest.skip("outputs/model_ablation.csv not built")

    stale = [
        f"{c['city']}: ablation says {c['reproduces_published']['arm']:+.1f}%, "
        f"the page says {c['reproduces_published']['published']:+.1f}%"
        for c in ablation["cities"]
        if c.get("reproduces_published") and not c["reproduces_published"]["passed"]
    ]
    assert not stale, "re-run scripts/ablate_model.py -- " + "; ".join(stale)
def test_every_chart_carries_a_reading() -> None:
    """No chart ships without a "How to read this" panel behind it.

    The page rebuilds itself weekly and nobody writes a caption for the version
    that comes out, so the reading has to be written once as a rule rather than
    as an observation. That only works if it is attached to every chart; one
    chart without a guide is the one a reader stalls on.

    Checked statically rather than in a browser, because there is no JS test
    runner here. Every ``chartCard`` call must close on an options object naming
    a guide, which is a shape this file can see and a linter cannot.
    """
    # Lookbehind so the declaration of chartCard is not counted as a call of it.
    calls = re.compile(r"(?<!function )chartCard\(\s*jobs,")
    # The options object a chartCard call closes on: `..., { guide: "x" });`.
    # KNOBS entries also carry a `guide:` and are deliberately not matched, since
    # they end in a comma rather than in `);`.
    options = re.compile(r"\{[^{}]*guide:[^{}]*\}\s*\)\s*;")

    for page in ("app.js", "compare.js"):
        src = (SITE / page).read_text(encoding="utf-8")
        n_calls = len(calls.findall(src))
        n_guided = len(options.findall(src))
        assert n_calls, f"{page} has no chartCard calls -- has it been renamed?"
        assert n_calls == n_guided, (
            f"{page}: {n_calls} charts, {n_guided} with a guide. "
            "Every chartCard call needs a `guide:` in its options object."
        )
