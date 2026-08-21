/* Render the "How to read this" panel under a chart.
 *
 * The prose is not here. It lives in `src/driftloop/guides.py`, because the
 * Streamlit app draws the same quantities and would otherwise need a second
 * copy: two readings of one chart, drifting apart from the first edit. That
 * module is the single source, `scripts/build_site.py` fills in the loop's
 * thresholds and writes the result into `data.json`, and this file turns it
 * into markup.
 *
 * So a guide arrives here already complete. Nothing in this file knows what any
 * chart means, which is the property that keeps the two interfaces agreeing.
 *
 * Shared by index.html and compare.html, which each own a separate copy of
 * `chartCard`. One renderer, two callers.
 */

/* The prose is authored as Markdown so Streamlit can render it untouched, which
 * leaves this side two constructs to convert and everything else to escape.
 *
 * Escaping first, and unconditionally: these strings arrive through data.json
 * and are written into innerHTML, so the conversion below has to be the only
 * way markup can enter. Doing it the other way round would let a stray angle
 * bracket in someone's prose become a tag. */
function guideMarkdown(text) {
  const escaped = String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return escaped
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

/* Returns "" for an unknown key or a payload built before the guides existed,
 * so a chart without one degrades to the card it was before rather than
 * throwing. `tests/test_site_assets.py` is what stops that being silent.
 *
 * `<details>` rather than a button and a class toggle: it brings its own
 * keyboard handling and its own open/closed state, and it prints expanded. */
function guideHTML(key, data) {
  const g = key && data && data.guides && data.guides[key];
  if (!g) return "";
  const moves = (g.moves || [])
    .map(([when, then]) => `<dt>${guideMarkdown(when)}</dt><dd>${guideMarkdown(then)}</dd>`)
    .join("");
  return (
    `<details class="guide">` +
    `<summary>How to read this</summary>` +
    `<div class="guide-body">` +
    `<p>${guideMarkdown(g.read)}</p>` +
    `<p><strong>The maths.</strong> ${guideMarkdown(g.math)}</p>` +
    (moves
      ? `<p class="guide-lead"><strong>What it means when it moves</strong></p>` +
        `<dl class="guide-moves">${moves}</dl>`
      : "") +
    `<p><strong>What would improve this.</strong> ${guideMarkdown(g.next)}</p>` +
    `</div></details>`
  );
}
