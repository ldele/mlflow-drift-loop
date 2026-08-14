/* The all-cities view.
 *
 * Reads the same data.json the per-city page does, so the two cannot disagree
 * about a number: anything shown here is the same field the main page shows,
 * arranged so the six cities sit side by side instead of one at a time.
 *
 * Deliberately small. Everything city-specific (the story, the drift heatmap,
 * the coefficients) stays on the main page; what belongs here is only what needs
 * more than one city to mean anything.
 */

const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const CONFIG = { displayModeBar: false, responsive: true };

const THEMES = {
  light: {
    surface: "#ffffff", ink: "#14130f", ink2: "#55534d", muted: "#8a867d",
    grid: "#ebe9e3", axis: "#d7d5cc", border: "rgba(15,14,10,0.10)",
    series: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    good: "#0a9a2e", warn: "#c98500", crit: "#d03b3b",
  },
  dark: {
    surface: "#17171a", ink: "#f4f4f2", ink2: "#b8b7b0", muted: "#8b8983",
    grid: "#29292c", axis: "#3a3a3e", border: "rgba(255,255,255,0.10)",
    series: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
    good: "#26c24a", warn: "#f0b03a", crit: "#e05656",
  },
};

let DATA, CITIES = [];

function resolveTheme() {
  const attr = document.documentElement.getAttribute("data-theme");
  if (attr === "dark" || attr === "light") return attr;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
const P = () => THEMES[resolveTheme()];
/* No sign on a value that rounds to zero: "−0.0%" reads as a loss that rounded
 * away rather than as breaking even. */
const pct = (v, d = 1) => {
  const rounded = Number(Math.abs(v).toFixed(d));
  return rounded === 0 ? `0.${"0".repeat(d)}%` : `${v >= 0 ? "+" : "−"}${rounded.toFixed(d)}%`;
};

function plotBase(pal, yTitle, opts = {}) {
  return {
    paper_bgcolor: pal.surface, plot_bgcolor: pal.surface,
    font: { family: FONT, size: 12, color: pal.ink2 },
    margin: { l: 52, r: 16, t: 10, b: 34 },
    height: opts.height || 262,
    hovermode: opts.hovermode || "x unified",
    showlegend: false,
    hoverlabel: { bgcolor: pal.surface, bordercolor: pal.border, font: { family: FONT, color: pal.ink } },
    xaxis: { showgrid: false, linecolor: pal.axis, tickfont: { color: pal.muted, size: 11 }, zeroline: false, ticklen: 0 },
    yaxis: {
      title: { text: yTitle, font: { color: pal.muted, size: 11 } },
      gridcolor: pal.grid, linecolor: pal.axis,
      tickfont: { color: pal.muted, size: 11 }, zeroline: false, ticklen: 0, nticks: 5,
      range: opts.yrange,
    },
    shapes: [], annotations: [],
  };
}

function legendHTML(chips) {
  return chips.map((c) => {
    const sw = c.kind === "dot"
      ? `<span class="swatch dot" style="background:${c.color}"></span>`
      : c.kind === "dash"
      ? `<span class="swatch dash" style="color:${c.color}"></span>`
      : `<span class="swatch" style="background:${c.color}"></span>`;
    return `<span class="lg">${sw}${c.label}</span>`;
  }).join("");
}

function chartCard(jobs, title, desc, chips, traces, layout, container, wide = false) {
  const card = document.createElement("section");
  card.className = wide ? "card card-wide" : "card";
  card.innerHTML =
    `<div class="card-head"><h3>${title}</h3><div class="legend">${legendHTML(chips)}</div></div>` +
    (desc ? `<p class="desc">${desc}</p>` : "") +
    `<div class="plot"></div>`;
  document.getElementById(container).appendChild(card);
  jobs.push({ div: card.querySelector(".plot"), traces, layout });
}

/* Same two-pass plot the main page uses, and for the same reason: Plotly falls
 * back to a hard-coded 700px when it cannot measure the container during
 * newPlot, and an explicit width would opt the chart out of `responsive`. */
function plotAll(jobs) {
  jobs.forEach((j) =>
    Plotly.newPlot(j.div, j.traces, j.layout, CONFIG).then(() => Plotly.Plots.resize(j.div)));
}

/* ---------- headline tiles ---------- */

function renderTiles() {
  const pal = P();
  const withSkill = CITIES.filter((c) => c.stats.latest_skill != null);
  const positive = withSkill.filter((c) => c.stats.latest_skill > 0);
  const stale = CITIES.filter((c) => (c.stats.champion_age_days ?? 0) >= 120);
  const acted = CITIES.map((c) => c.stats.retrain_acted).filter((v) => v != null).sort((a, b) => a - b);
  // Averaged across the two middle values on an even count. Taking the upper one
  // reported Johannesburg's +20.2% as the median of six cities when the median
  // is +14.8%, which overstates the typical city by a third.
  const medianActed = !acted.length
    ? null
    : acted.length % 2
    ? acted[(acted.length - 1) / 2]
    : (acted[acted.length / 2 - 1] + acted[acted.length / 2]) / 2;

  const tiles = [
    { v: CITIES.length, k: "Cities watched" },
    { v: CITIES.reduce((s, c) => s + c.stats.runs, 0), k: "Weeks of monitoring in total" },
    {
      v: `${CITIES.reduce((s, c) => s + c.stats.promotions, 0)}/${CITIES.reduce((s, c) => s + c.stats.retrains, 0)}`,
      k: `Challengers shipped · ${CITIES.reduce((s, c) => s + c.stats.rejected, 0)} rejected by the gate`,
    },
    {
      v: `${positive.length}/${withSkill.length}`,
      color: positive.length > withSkill.length / 2 ? pal.good : pal.crit,
      k: "Cities whose model currently beats a rule of thumb",
    },
    medianActed == null ? null : (() => {
      // Uncoloured on purpose. This is the median of six estimates, and two of
      // the six do not clear zero, so painting it green would assert something
      // those two do not support.
      //
      // The count rides in the label rather than taking a tile of its own: a
      // seventh tile wraps the row onto two lines.
      const judged = CITIES.filter((c) => c.stats.retrain_acted_real != null);
      const real = judged.filter((c) => c.stats.retrain_acted_real).length;
      return {
        v: pct(medianActed),
        k: "Retraining, week by week, median across cities"
          + (judged.length ? ` · measurable in ${real} of ${judged.length}` : ""),
      };
    })(),
    {
      v: stale.length, color: stale.length ? pal.warn : undefined,
      k: "Cities serving a model over 120 days old",
    },
  ].filter(Boolean);

  document.getElementById("tiles").innerHTML = tiles.map((t) =>
    `<div class="tile"><div class="tile-v"${t.color ? ` style="color:${t.color}"` : ""}>${t.v}</div>` +
    `<div class="tile-k">${t.k}</div></div>`).join("");
}

/* ---------- one skill panel per city ---------- */

/* Separate cards rather than a Plotly subplot grid, so each panel keeps its own
 * date axis. The cities do not share a calendar: Johannesburg starts in
 * February and Kraków the previous August, and forcing them onto one x-axis
 * would leave most panels mostly empty.
 *
 * The y-axis *is* shared, and set from the full pooled range, because the whole
 * question here is which cities fall further than others. */
function renderSkillGrid() {
  const pal = P();
  const host = document.getElementById("skill-grid");
  host.innerHTML = "";
  const jobs = [];

  const all = CITIES.flatMap((c) => c.retro?.skill?.champion || []).filter((v) => v != null);
  if (!all.length) return;
  const lo = Math.min(...all, 0), hi = Math.max(...all, 0);
  const pad = (hi - lo) * 0.06;

  CITIES.forEach((city, i) => {
    const skill = city.retro?.skill?.champion;
    if (!skill) return;
    const lay = plotBase(pal, "skill", { yrange: [lo - pad, hi + pad] });
    lay.shapes.push({
      type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 0, y1: 0,
      line: { color: pal.muted, width: 1, dash: "dot" }, layer: "below",
    });
    const worth = city.stats.retrain_acted;
    // The interval belongs next to the number here for the same reason it does
    // on the index page. Kraków's "+6.5% week by week" spans [-15.3, +27.5] and
    // read alone it is a small positive result, which is not what the data says.
    // Named apart from the `lo`/`hi` this function already uses for the y-axis
    // range. Reusing those names threw a temporal-dead-zone ReferenceError that
    // killed the whole render, and a page with no charts at all is what a
    // headless capture then quietly saves.
    const worthLo = city.stats.retrain_acted_lo, worthHi = city.stats.retrain_acted_hi;
    const range = (worthLo == null || worthHi == null) ? ""
      : ` [${pct(worthLo, 1)}, ${pct(worthHi, 1)}]`
        + (city.stats.retrain_acted_real === false ? ", not distinguishable from zero" : "");
    chartCard(jobs, city.label,
      `${city.stats.runs} weeks · model in service ${city.stats.champion_age_days} days` +
      (worth == null ? "" : ` · retraining ${pct(worth)} week by week${range}`),
      [{ label: "skill vs. a 30-day daily profile", color: pal.series[i % 6], kind: "line" }],
      [{
        x: city.retro.as_of, y: skill, mode: "lines", type: "scatter",
        line: { color: pal.series[i % 6], width: 2.2 }, showlegend: false,
        hovertemplate: "%{x}<br>skill %{y:+.2f}<extra></extra>",
      }],
      lay, "skill-grid");
  });

  plotAll(jobs);
}

/* ---------- what retraining was worth ---------- */

function renderValue() {
  const pal = P();
  const host = document.getElementById("value-charts");
  host.innerHTML = "";
  const jobs = [];

  const rows = CITIES.filter((c) => c.stats.retrain_gain != null);
  if (!rows.length) return;

  const lay = plotBase(pal, "% better than never retraining", { hovermode: "closest", height: 330 });
  lay.barmode = "group";
  lay.margin.b = 54;
  lay.shapes.push({
    type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 0, y1: 0,
    line: { color: pal.muted, width: 1 }, layer: "below",
  });

  const traces = [
    {
      type: "bar", x: rows.map((c) => c.label), y: rows.map((c) => c.stats.retrain_gain),
      marker: { color: pal.series[3] }, showlegend: false,
      hovertemplate: "%{x}: %{y:+.1f}%<extra>across the replay</extra>",
    },
    {
      type: "bar", x: rows.map((c) => c.label), y: rows.map((c) => c.stats.retrain_acted ?? 0),
      marker: { color: pal.series[0] }, showlegend: false,
      // Error bars on the paired reading, because this chart is the one place a
      // reader compares the six cities to each other by eye. Without them
      // Melbourne's +1.2% and Delhi's +49.4% are the same kind of object drawn
      // at different heights, and only one of the two is a finding. Plotly wants
      // the bar lengths rather than the endpoints, so the interval is expressed
      // as distances from the estimate.
      error_y: {
        type: "data", symmetric: false,
        array: rows.map((c) => (c.stats.retrain_acted_hi == null ? 0
          : c.stats.retrain_acted_hi - c.stats.retrain_acted)),
        arrayminus: rows.map((c) => (c.stats.retrain_acted_lo == null ? 0
          : c.stats.retrain_acted - c.stats.retrain_acted_lo)),
        color: pal.ink, thickness: 1.4, width: 5,
      },
      customdata: rows.map((c) => [
        c.stats.retrain_acted_windows, c.stats.retrain_win_rate,
        c.stats.retrain_acted_lo, c.stats.retrain_acted_hi,
      ]),
      hovertemplate:
        "%{x}: %{y:+.1f}% [%{customdata[2]:+.1f}, %{customdata[3]:+.1f}]<br>"
        + "over %{customdata[0]} weeks, won %{customdata[1]}%<extra>week by week</extra>",
    },
  ];

  chartCard(jobs, "Retraining, measured two ways",
    "Where the two bars disagree, trust the blue one. An orange bar near zero next to a tall blue one means the city spent most of its replay before anything had been promoted, so the unpaired comparison is measuring the season rather than the retraining. The whiskers are 95% intervals on the blue bar: where one crosses zero, that city shows no measurable effect at all.",
    [
      { label: "across the whole replay", color: pal.series[3], kind: "dot" },
      { label: "week by week, when it acted", color: pal.series[0], kind: "dot" },
    ],
    traces, lay, "value-charts", true);

  // Every claim here is counted off the data rather than asserted, including the
  // lead sentence. "Retraining helped in every city once the comparison is
  // paired" was true when written and stopped being true when an accounting fix
  // moved Los Angeles below zero. A sentence a re-run can falsify is derived
  // from the re-run.
  const gap = rows
    .map((c) => ({ label: c.label, d: (c.stats.retrain_acted ?? 0) - c.stats.retrain_gain }))
    .sort((a, b) => b.d - a.d)[0];
  const acted = (c) => c.stats.retrain_acted ?? 0;
  // Counted on whether the interval clears zero, not on the sign of the
  // estimate. Five of six are positive and only three of those are
  // distinguishable from nothing, so "positive in 5 of 6" was the chart's own
  // caption overselling what the chart now visibly shows.
  const gained = rows.filter((c) => c.stats.retrain_acted_real && acted(c) > 0).length;
  const noEffect = rows.filter((c) => c.stats.retrain_acted_real === false);
  const worst = rows.slice().sort((a, b) => acted(a) - acted(b))[0];
  const names = (cs) => cs.map((c) => c.label).join(" and ");
  document.getElementById("value-note").innerHTML =
    `<strong>Pairing the comparison rescues some of these cities, and not all of them.</strong> ` +
    `Week by week, ${gained} of ${rows.length} show a gain the interval can separate from zero` +
    (noEffect.length
      ? `, and ${names(noEffect)} show no measurable effect at all. `
      : ". ") +
    `The widest gap between the two readings is ${gap.label}, at ` +
    `${gap.d.toFixed(1)} percentage points. ` +
    (acted(worst) > 0
      ? `The weakest of them, ${worst.label}, gains only ${pct(acted(worst))}, and the effort ` +
        `still has to be paid for. `
      : `And retraining is not free: ${worst.label} comes out at ${pct(acted(worst))} even ` +
        `paired, winning ${worst.stats.retrain_win_rate}% of the weeks it acts, which is worse ` +
        `than a coin toss and still costs the compute. `) +
    `What it does mean is that the headline number was answering a different question ` +
    `than the one it appeared to answer.`;

  plotAll(jobs);
}

/* ---------- six models or one ---------- */

const POOLED_LABEL = {
  champion_served: "its own model, retrained",
  champion_frozen: "its own model, frozen",
  pooled_cities: "one model for all six",
};

function renderPooled() {
  const section = document.getElementById("pooled-section");
  const rows = CITIES.filter((c) => c.benchmark?.scored?.some((s) => s.name === "pooled_cities"));
  section.hidden = rows.length < 2;
  if (section.hidden) return;

  const pal = P();
  document.getElementById("pooled-charts").innerHTML = "";
  const jobs = [];

  const pick = (c, name) => c.benchmark.scored.find((s) => s.name === name)?.median_rmse ?? null;
  const names = ["champion_served", "champion_frozen", "pooled_cities"];
  const colors = [pal.series[0], pal.series[3], pal.series[1]];

  const lay = plotBase(pal, "median error (µg/m³)", { hovermode: "closest", height: 330 });
  lay.barmode = "group";
  lay.margin.b = 54;

  chartCard(jobs, "One model for all six cities, against six models",
    "Median error over the same monitoring windows, lower is better. The pooled model gets a separate intercept per city, which is doing most of the work: mean pollution runs from 7 µg/m³ in Melbourne to 84 in Delhi, and the same model without that adjustment scores 43% worse.",
    names.map((n, i) => ({ label: POOLED_LABEL[n], color: colors[i], kind: "dot" })),
    names.map((name, i) => ({
      type: "bar", name: POOLED_LABEL[name],
      x: rows.map((c) => c.label), y: rows.map((c) => pick(c, name)),
      marker: { color: colors[i] }, showlegend: false,
      hovertemplate: `%{x}: %{y:.2f} µg/m³<extra>${POOLED_LABEL[name]}</extra>`,
    })),
    lay, "pooled-charts", true);

  const beatsFrozen = rows.filter((c) => pick(c, "pooled_cities") < pick(c, "champion_frozen"));
  const beatsServed = rows.filter((c) => pick(c, "pooled_cities") < pick(c, "champion_served"));
  const worst = rows
    .map((c) => ({ label: c.label, r: pick(c, "pooled_cities") / pick(c, "champion_served") }))
    .sort((a, b) => b.r - a.r)[0];
  document.getElementById("pooled-note").innerHTML =
    `<strong>Six models win, and by less than the layout implies.</strong> ` +
    `One model over every city loses to the city's own retrained model in ` +
    `${rows.length - beatsServed.length} of ${rows.length}, and beats the city's own <em>frozen</em> ` +
    `model in ${beatsFrozen.length}. Training on five other cities is worth more than a year of ` +
    `staleness and less than keeping one city's model current. ` +
    `It hurts most in ${worst.label}, at ${worst.r.toFixed(2)}× the error of that city's own model. ` +
    `Read the other way, this is a limit on the features rather than on the arrangement: six weather ` +
    `variables and an hour of day do not carry whatever makes a city's air behave the way it does, ` +
    `so the per-city intercept has to stand in for all of it. Better features would likely narrow ` +
    `the gap between the blue and orange bars everywhere.`;

  plotAll(jobs);
}

/* ---------- boot ---------- */

function renderAll() {
  renderTiles();
  renderSkillGrid();
  renderValue();
  renderPooled();
}

function setupTheme() {
  // Same ?theme=light|dark override the per-city page honours, so a link that
  // pins the palette keeps working when a reader follows it between the two.
  const asked = new URLSearchParams(location.search).get("theme");
  const saved = asked === "dark" || asked === "light" ? asked : localStorage.getItem("driftloop-theme");
  if (saved === "dark" || saved === "light") document.documentElement.setAttribute("data-theme", saved);
  document.getElementById("theme").addEventListener("click", () => {
    const next = resolveTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("driftloop-theme", next);
    renderAll();
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!document.documentElement.getAttribute("data-theme")) renderAll();
  });
}

async function main() {
  setupTheme();
  try {
    const resp = await fetch("data.json", { cache: "no-cache" });
    if (!resp.ok) throw new Error(`data.json → HTTP ${resp.status}`);
    DATA = await resp.json();
  } catch (e) {
    document.getElementById("error").textContent =
      "Couldn't load data.json (" + e.message + "). If viewing locally, serve the folder over HTTP.";
    return;
  }
  CITIES = (DATA.profiles || []).filter((p) => p.location);
  document.getElementById("built").textContent = `Snapshot · built ${DATA.built}`;
  const raw = DATA.raw_data || [];
  document.getElementById("data-links").innerHTML =
    (raw.length
      ? `<strong>Raw hourly data:</strong> ` +
        raw.map((r) => `<a href="${r.file}">${r.city}</a>`).join(", ") + " &nbsp;·&nbsp; "
      : "") +
    `<strong>Chart data:</strong> <a href="data.json">data.json</a>`;
  renderAll();
}

document.addEventListener("DOMContentLoaded", main);
