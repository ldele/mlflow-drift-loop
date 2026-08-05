/* Client-side renderer for the drift-loop dashboard.
 * Fetches data.json (from scripts/build_site.py) and builds interactive Plotly
 * charts. Chart titles, legends, and descriptions live in the HTML card — the
 * Plotly canvas holds only the data — and everything is theme-aware (light/dark).
 * Palette follows the validated data-viz reference: categorical slots 1-6 taken
 * in their fixed order (blue, orange, aqua, yellow, magenta, green), never
 * cycled. Both modes clear the CVD and normal-vision floors on the adjacent
 * pairlist. Three light-mode slots sit under 3:1 against the surface, so the
 * relief rule applies: every chart carries a legend, and the footer publishes
 * data.json plus a CSV per city as the table view. */

const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
// Read from data.json rather than retyped, so adding a feature in config.py
// cannot leave the page plotting a set the loop no longer uses. Colour is keyed
// on a feature's index here, which is what keeps a feature the same colour in
// the drift chart and the coefficient chart.
let DRIFT_FEATURES = [];
const PSI_SIGNIFICANT = 0.25;
// The lower band boundary, used only to colour map markers. Industry convention:
// <0.10 stable, 0.10–0.25 moderate, >0.25 significant.
const PSI_MODERATE = 0.1;
const PERF_THRESHOLD = 1.25;
const CONFIG = { displayModeBar: false, responsive: true };

// Used by both the method spec and the benchmark card, which describe the same
// horizon and should not word it differently.
const days = (n) => `${n} day${n === 1 ? "" : "s"}`;

const THEMES = {
  light: {
    surface: "#ffffff", ink: "#14130f", ink2: "#55534d", muted: "#8a867d",
    grid: "#ebe9e3", axis: "#d7d5cc", border: "rgba(15,14,10,0.10)",
    series: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    good: "#0a9a2e", warn: "#c98500", crit: "#d03b3b",
    drift: "rgba(236,131,90,0.09)",
  },
  dark: {
    surface: "#17171a", ink: "#f4f4f2", ink2: "#b8b7b0", muted: "#8b8983",
    grid: "#29292c", axis: "#3a3a3e", border: "rgba(255,255,255,0.10)",
    series: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
    good: "#26c24a", warn: "#f0b03a", crit: "#e05656",
    drift: "rgba(236,131,90,0.15)",
  },
};

let DATA, byKey = {}, current;

function resolveTheme() {
  const attr = document.documentElement.getAttribute("data-theme");
  if (attr === "dark" || attr === "light") return attr;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
const P = () => THEMES[resolveTheme()];

/* ---------- Plotly building blocks ---------- */

function plotBase(pal, yTitle, opts = {}) {
  return {
    paper_bgcolor: pal.surface, plot_bgcolor: pal.surface,
    font: { family: FONT, size: 12, color: pal.ink2 },
    margin: { l: 50, r: 16, t: 10, b: 32 },
    height: opts.height || 262,
    hovermode: opts.hovermode || "x unified",
    showlegend: false,
    hoverlabel: { bgcolor: pal.surface, bordercolor: pal.border, font: { family: FONT, color: pal.ink } },
    xaxis: { showgrid: false, linecolor: pal.axis, tickfont: { color: pal.muted, size: 11 }, zeroline: false, ticklen: 0 },
    yaxis: {
      title: { text: yTitle, font: { color: pal.muted, size: 11 } },
      gridcolor: pal.grid, griddash: "solid", linecolor: pal.axis,
      tickfont: { color: pal.muted, size: 11 }, zeroline: false, ticklen: 0, nticks: 5,
    },
    shapes: [], annotations: [],
  };
}

function driftRegion(lay, pal, driftDate, xEnd) {
  if (!driftDate) return;
  lay.shapes.push({
    type: "rect", xref: "x", yref: "paper", x0: driftDate, x1: xEnd, y0: 0, y1: 1,
    fillcolor: pal.drift, line: { width: 0 }, layer: "below",
  });
  lay.shapes.push({
    type: "line", xref: "x", yref: "paper", x0: driftDate, x1: driftDate, y0: 0, y1: 1,
    line: { color: pal.muted, width: 1, dash: "dash" }, layer: "below",
  });
  lay.annotations.push({
    x: driftDate, y: 1, xref: "x", yref: "paper", text: "regime shift", showarrow: false,
    font: { color: pal.muted, size: 10 }, xanchor: "left", yanchor: "top", xshift: 5, yshift: -2,
    bgcolor: pal.surface, bordercolor: pal.border, borderpad: 3,
  });
}

function thresholdLine(lay, pal, value, label) {
  lay.shapes.push({
    type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: value, y1: value,
    line: { color: pal.muted, width: 1, dash: "dot" }, layer: "below",
  });
  lay.annotations.push({
    xref: "paper", x: 1, y: value, yref: "y", text: label, showarrow: false,
    font: { color: pal.muted, size: 10 }, xanchor: "right", yanchor: "bottom", xshift: -3, yshift: 2,
    bgcolor: pal.surface, borderpad: 1,
  });
}

/* A horizontal band across the full width, for a range rather than a cut-off:
 * the conventional PSI reading bands, or the spread of values a model was
 * trained on. Drawn below the data and unlabelled on the canvas, since the card's
 * legend names it, so the plot area stays free of text. */
function bandShape(lay, y0, y1, color) {
  lay.shapes.push({
    type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y", y0, y1,
    fillcolor: color, line: { width: 0 }, layer: "below",
  });
}

/* Optional keys are attached rather than set to undefined: Plotly's cleanData
 * walks every key that is *present*, so `marker: undefined` throws where an
 * absent marker is fine. */
function lineT(x, y, name, color, dash, opts = {}) {
  const trace = {
    x, y, name, mode: opts.mode || "lines", type: "scatter",
    // `hv` holds each value flat until the next point rather than sloping
    // between them, which is what a threshold that only changes on promotion
    // does. Sloping it would imply the bar drifts continuously.
    line: { color, width: opts.width || 2.4, dash: dash || null, shape: opts.shape || "linear" },
    showlegend: false,
    hovertemplate: (opts.hover || "%{y:.3f}") + "<extra>" + name + "</extra>",
  };
  if (trace.mode.includes("markers")) {
    trace.marker = { size: opts.size || 6, color, line: { width: 0 } };
  }
  if (opts.opacity != null) trace.opacity = opts.opacity;
  return trace;
}

const eventT = (pal, x, y, name, color, symbol, size) => ({
  x, y, name, mode: "markers", type: "scatter",
  marker: { color, size: size || 12, symbol: symbol || "star", line: { color: pal.surface, width: 2 } },
  hovertemplate: "%{x|%Y-%m-%d}<extra>" + name + "</extra>",
});

function annotated(lay, pal, text) {
  lay.annotations.push({
    text, showarrow: false, xref: "paper", yref: "paper", x: 0.5, y: 0.5,
    font: { color: pal.muted, size: 13 },
  });
}

/* A grid of one small panel per series, each on its OWN y-axis.
 *
 * For quantities that share a unit, one chart with a line per series is better:
 * you can compare heights directly. For quantities that don't, it is a trap.
 * The model coefficients are per *original* unit, so in Krakow precipitation
 * moves across ~27 units while shortwave radiation moves across ~0.05 -- 500x --
 * and on one shared axis four of the six features are pinned flat against zero.
 *
 * Plotly.js has no subplot builder, so the axis grid is laid out by hand: panel
 * i gets axis pair (i+1), positioned by domain. Panels share the x range but
 * never the y. */
function smallMultiples(pal, items, opts = {}) {
  // Follow the same breakpoint the .charts grid uses: below it the card is a
  // single narrow column, where three panels across are thinner than their own
  // axis labels. Plotly's `responsive` rescales a plot but cannot re-flow a
  // domain grid, so the column count is decided here, at plot time.
  const ncols = opts.ncols || (window.innerWidth < 760 ? 2 : 3);
  const nrows = Math.ceil(items.length / ncols);
  const gapX = 0.06, gapY = 0.16;
  const cellW = (1 - gapX * (ncols - 1)) / ncols;
  const cellH = (1 - gapY * (nrows - 1)) / nrows;

  const lay = {
    paper_bgcolor: pal.surface, plot_bgcolor: pal.surface,
    font: { family: FONT, size: 12, color: pal.ink2 },
    margin: { l: 44, r: 14, t: 22, b: 30 },
    height: opts.height || 150 * nrows + 60,
    hovermode: "closest",
    showlegend: false,
    hoverlabel: { bgcolor: pal.surface, bordercolor: pal.border, font: { family: FONT, color: pal.ink } },
    shapes: [], annotations: [],
  };

  const traces = items.map((item, i) => {
    const col = i % ncols, row = Math.floor(i / ncols);
    const n = i + 1;
    const ax = n === 1 ? "xaxis" : `xaxis${n}`;
    const ay = n === 1 ? "yaxis" : `yaxis${n}`;
    const xref = n === 1 ? "x" : `x${n}`;
    const yref = n === 1 ? "y" : `y${n}`;

    const x0 = col * (cellW + gapX);
    // Rows are laid out top-down, but Plotly's paper y runs bottom-up.
    const y0 = 1 - (row * (cellH + gapY) + cellH);

    lay[ax] = {
      domain: [x0, x0 + cellW], anchor: yref,
      showgrid: false, linecolor: pal.axis, zeroline: false, ticklen: 0,
      tickfont: { color: pal.muted, size: 10 }, nticks: 3,
    };
    lay[ay] = {
      domain: [y0, y0 + cellH], anchor: xref,
      gridcolor: pal.grid, linecolor: pal.axis, zeroline: false, ticklen: 0,
      tickfont: { color: pal.muted, size: 10 }, nticks: 4,
    };
    // A band, where the panel has one: the range this feature held over the
    // window the first model was trained on. The line leaving the band is the
    // covariate-drift claim, stated in the feature's own units instead of as a
    // PSI whose scale nobody can read.
    if (item.band && item.band.lo != null) {
      lay.shapes.push({
        type: "rect", xref: `${xref} domain`, x0: 0, x1: 1,
        yref, y0: item.band.lo, y1: item.band.hi,
        fillcolor: item.color, opacity: 0.13, line: { width: 0 }, layer: "below",
      });
    }
    // Zero is the only reference worth drawing: crossing it is the event. Only
    // meaningful where zero is a boundary rather than just a small value, so
    // the coefficient panels draw it and the feature-level panels do not.
    if (opts.zeroLine !== false) {
      lay.shapes.push({
        type: "line", xref: `${xref} domain`, x0: 0, x1: 1,
        yref, y0: 0, y1: 0, line: { color: pal.axis, width: 1 }, layer: "below",
      });
    }
    lay.annotations.push({
      text: item.name, showarrow: false,
      xref: `${xref} domain`, yref: `${yref} domain`, x: 0, y: 1,
      xanchor: "left", yanchor: "bottom", yshift: 4,
      font: { color: pal.ink, size: 11.5 },
    });

    return {
      x: item.x, y: item.y, name: item.name, mode: "lines+markers", type: "scatter",
      xaxis: xref, yaxis: yref,
      line: { color: item.color, width: 2 },
      marker: { size: 6, color: item.color, line: { color: pal.surface, width: 1.5 } },
      hovertemplate: "%{y:.3f}<extra>" + item.name + "</extra>",
    };
  });

  return { traces, layout: lay };
}

/* ---------- HTML pieces ---------- */

function legendHTML(chips) {
  return chips.map((c) => {
    let sw;
    if (c.kind === "star") sw = `<span class="swatch" style="color:${c.color};font-size:13px;line-height:1">★</span>`;
    else if (c.kind === "dot") sw = `<span class="swatch dot" style="background:${c.color}"></span>`;
    else if (c.kind === "dash") sw = `<span class="swatch dash" style="color:${c.color}"></span>`;
    else sw = `<span class="swatch" style="background:${c.color}"></span>`;
    return `<span class="lg">${sw}${c.label}</span>`;
  }).join("");
}

/* Build the card and append it, but DON'T plot yet — collect a job. Plotting is
 * deferred until every card is in the DOM so the CSS grid has settled into its
 * final column count; otherwise Plotly measures a detached/half-laid-out width
 * and hard-codes its 700px default, overflowing the card. */
function chartCard(jobs, title, desc, chips, traces, layout, container = "charts", wide = false) {
  const card = document.createElement("section");
  // A small-multiples grid needs the whole row: at half width its panels are
  // narrower than their own axis labels.
  card.className = wide ? "card card-wide" : "card";
  card.innerHTML =
    `<div class="card-head"><h3>${title}</h3><div class="legend">${legendHTML(chips)}</div></div>` +
    `<p class="desc">${desc}</p><div class="plot"></div>`;
  document.getElementById(container).appendChild(card);
  jobs.push({ div: card.querySelector(".plot"), traces, layout });
}

/* The tiles answer three questions in order: what is being predicted, is the
 * model any good at it, and did the loop earn its keep. The counts they used to
 * spend three of five slots on (runs / retrains / promotions) said what the
 * machine *did* and nothing about whether any of it worked, and two of them had
 * to be subtracted from each other to reach the number that matters. */
/* A signed percentage, with no sign at all on zero. "−0.0%" is what the
 * arithmetic produces when retraining breaks exactly even, and it reads as a
 * loss that rounded away rather than as a wash. */
const pct = (v, digits = 0) => {
  const rounded = Number(Math.abs(v).toFixed(digits));
  return rounded === 0 ? `0.${"0".repeat(digits)}%`.replace(".%", "%")
    : `${v >= 0 ? "+" : "−"}${rounded.toFixed(digits)}%`;
};

/* Plot every collected job, then force a re-measure.
 *
 * Deferring the plot until all cards are in the DOM is not on its own enough:
 * Plotly reads the container width during newPlot and, whenever that read comes
 * back unusable, silently falls back to a hard-coded 700px and writes it into
 * the layout. A 700px canvas in a 509px card overflows the card and scrolls the
 * whole page sideways.
 *
 * `Plots.resize` re-reads the container after layout has settled. Writing a
 * measured `layout.width` before newPlot instead is about 20% quicker and is
 * wrong: an explicit width opts the plot out of `responsive`, so every chart
 * then keeps its old size through a viewport change and overflows again. The
 * second pass buys the resize behaviour and is worth its cost. */
function plotAll(jobs) {
  jobs.forEach((j) =>
    Plotly.newPlot(j.div, j.traces, j.layout, CONFIG).then(() => Plotly.Plots.resize(j.div)));
}

function statTiles(stats, target, retro) {
  const pal = P();
  const guide = target?.who_24h_guideline;
  const air = stats.latest_actual;
  // Lead with the physical quantity. Without it the page is all monitoring
  // machinery and never says what is predicted or how bad the air actually is.
  const airColor = air == null || guide == null ? pal.ink
    : air >= guide * 3 ? pal.crit : air >= guide ? pal.warn : pal.good;

  const tiles = [];
  if (air != null) {
    // Dated on purpose. A city's replay ends where its configured span ends, so
    // "latest" can be months old, and an unqualified reading beside a WHO
    // guideline reads as a statement about the air right now.
    const at = stats.latest_actual_at;
    tiles.push({
      v: air, sub: target.units, color: airColor,
      k: `Measured ${target.name}${at ? ` on ${at}` : ""}` +
        `${guide ? ` · WHO 24h guideline ${guide}` : ""}`,
    });
  }
  // The health number. Scale-free, so unlike the raw error it is comparable
  // between a clean city and a filthy one, and unlike the retrain trigger its
  // yardstick does not move every time a model is promoted.
  if (stats.latest_skill != null) {
    const s = stats.latest_skill * 100;
    const days = retro?.skill?.climatology_days ?? 30;
    tiles.push({
      v: pct(s), color: s >= 0 ? pal.good : pal.crit,
      k: `${s >= 0 ? "Better" : "Worse"} than a ${days}-day daily profile · latest window`,
    });
  }
  if (stats.champion_age_days != null) {
    // A trigger that has gone quiet is indistinguishable from a model that is
    // fine, until you notice how long the thing has been sitting there.
    tiles.push({
      v: stats.champion_age_days, sub: "days",
      color: stats.champion_age_days >= 120 ? pal.warn : undefined,
      k: `Current model in service · v${stats.champion_version}`,
    });
  }
  tiles.push({
    v: `${stats.promotions}/${stats.retrains}`,
    k: `Challengers shipped · ${stats.rejected} rejected by the gate`,
  });
  if (stats.retrain_gain != null) {
    // Two numbers, because one of them lies on its own. The replay-wide figure
    // compares the median error of what was served against the median of the
    // original held frozen, which is an unpaired comparison: in a city that
    // promotes nothing until week 14 of 20, most windows have the two models
    // identical and both medians land on the same value. Johannesburg reads
    // 0.0% that way and won every week in which a retrained model was serving.
    const acted = stats.retrain_acted;
    tiles.push({
      v: pct(stats.retrain_gain, 1), color: stats.retrain_gain >= 0 ? pal.good : pal.crit,
      k: "Retraining, across the whole replay" +
        (acted == null ? "" : ` · ${pct(acted, 1)} over the ${stats.retrain_acted_windows}`
          + ` weeks it served a retrained model`),
    });
  }
  tiles.push({ v: stats.runs, k: "Weeks watched" });

  document.getElementById("tiles").innerHTML = tiles.map((t) =>
    `<div class="tile"><div class="tile-v"${t.color ? ` style="color:${t.color}"` : ""}>${t.v}` +
    `${t.sub ? `<span class="tile-u">${t.sub}</span>` : ""}</div>` +
    `<div class="tile-k">${t.k}</div></div>`).join("");
}

/* ---------- map ---------- */

/* Profiles tied to a real place. A profile without a location (the live schedule,
 * which reads the same Kraków source as its historical twin) gets no marker rather
 * than stacking a second one on the same point. */
const cityProfiles = () => DATA.profiles.filter((p) => p.location);

function psiStatus(pal, psi) {
  if (psi == null) return pal.muted;
  if (psi >= PSI_SIGNIFICANT) return pal.crit;
  if (psi >= PSI_MODERATE) return pal.warn;
  return pal.good;
}

/* Move the globe to a new centre in one step.
 *
 * This was an eased animation, and it was measurably the wrong idea. Rotating an
 * orthographic globe means re-projecting the land, ocean, country, coastline and
 * graticule paths, which costs 25-120ms per frame here whether it is driven
 * through `Plotly.relayout` or by rotating the projection and re-rendering the
 * subplot directly. That is a ceiling of roughly eleven frames a second, so no
 * amount of frame-pacing made it smooth: an eased tween sampled that coarsely
 * reads as a stutter, and all the machinery to schedule it (easing, a spin
 * counter to void superseded frames, promise chaining to avoid a backlog,
 * reduced-motion handling) was work spent making the judder regular.
 *
 * Dragging feels smooth by comparison because the pointer is the clock. The
 * frame rate is the same; the user is supplying it, so latency reads as weight
 * rather than as dropped frames. Nothing here can borrow that.
 *
 * So: one relayout, one re-projection, done. The globe stays draggable, which is
 * where the smooth interaction lives. */
function centreGlobeOn(gd, target) {
  const from = gd.layout?.geo?.projection?.rotation || { lon: 0, lat: 0 };
  if (Math.abs(target.lon - from.lon) < 0.01 && Math.abs(target.lat - from.lat) < 0.01) return;
  Plotly.relayout(gd, {
    "geo.projection.rotation.lon": target.lon,
    "geo.projection.rotation.lat": target.lat,
  });
}

function renderCityList(cities, pal) {
  document.getElementById("city-cap").textContent =
    `${cities.length} monitored ${cities.length === 1 ? "city" : "cities"}`;
  const list = document.getElementById("city-list");
  list.innerHTML = "";
  cities.forEach((c) => {
    const color = psiStatus(pal, c.stats.latest_psi);
    const loc = c.location;
    const b = document.createElement("button");
    b.className = "city-row";
    b.setAttribute("aria-selected", String(c.key === current));
    b.innerHTML =
      `<span class="cmain"><span class="cname">${loc.name}</span>` +
      `<span class="csub">${loc.country} · ${loc.lat.toFixed(2)}, ${loc.lon.toFixed(2)}</span></span>` +
      `<span class="pill" style="color:${color};border-color:${color};` +
      `background:color-mix(in srgb, ${color} 14%, transparent)">` +
      `PSI ${c.stats.latest_psi == null ? "—" : c.stats.latest_psi.toFixed(2)}</span>`;
    b.addEventListener("click", () => selectProfile(c.key));
    list.appendChild(b);
  });
}

/* The globe is rebuilt only when it genuinely has to be — a theme flip (colours
 * are baked into the layout) or a change in which cities exist. A mere change of
 * selection restyles and spins the plot that is already there, because tearing it
 * down and re-plotting is what would make the spin impossible. */
let globeState = { plotted: false, theme: null, sig: null };

function renderMap() {
  const card = document.getElementById("map-card");
  const cities = cityProfiles();
  // Nothing placeable (e.g. a synthetic-only build) — drop the card entirely
  // rather than showing an empty globe.
  card.hidden = cities.length === 0;
  if (!cities.length) return;

  const pal = P();
  const gd = document.getElementById("globe");
  const theme = resolveTheme();
  const sig = cities.map((c) => c.key).join("|");
  // -1 when the active profile has no location, which is a real state: the live
  // schedule is selected. Nothing is ringed, and the globe keeps a neutral centre.
  const sel = cities.findIndex((c) => c.key === current);
  const maxRuns = Math.max(...cities.map((c) => c.stats.runs), 1);

  const focus = sel >= 0 ? cities[sel].location : {
    lat: cities.reduce((s, c) => s + c.location.lat, 0) / cities.length,
    lon: cities.reduce((s, c) => s + c.location.lon, 0) / cities.length,
  };
  const ringColor = cities.map((_, i) => (i === sel ? pal.ink : pal.surface));
  const ringWidth = cities.map((_, i) => (i === sel ? 3 : 1.5));

  if (globeState.plotted && globeState.theme === theme && globeState.sig === sig) {
    Plotly.restyle(gd, { "marker.line.color": [ringColor], "marker.line.width": [ringWidth] }, [0]);
    centreGlobeOn(gd, focus);
    renderCityList(cities, pal);
    return;
  }

  const trace = {
    type: "scattergeo", mode: "markers",
    lat: cities.map((c) => c.location.lat),
    lon: cities.map((c) => c.location.lon),
    text: cities.map((c) => c.location.name),
    customdata: cities.map((c) => [c.location.country, c.stats.latest_psi, c.stats.runs]),
    marker: {
      color: cities.map((c) => psiStatus(pal, c.stats.latest_psi)),
      // sqrt so the dot's *area* tracks run count rather than its radius
      size: cities.map((c) => 13 + 11 * Math.sqrt(c.stats.runs / maxRuns)),
      line: { color: ringColor, width: ringWidth },
    },
    hovertemplate:
      "<b>%{text}</b>, %{customdata[0]}<br>PSI %{customdata[1]} · %{customdata[2]} runs<extra></extra>",
  };

  const layout = {
    paper_bgcolor: pal.surface,
    font: { family: FONT, size: 12, color: pal.ink2 },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    // An orthographic globe is a circle, so it fills its box by height, not width.
    // Tall enough that the card doesn't read as mostly empty.
    height: 380,
    showlegend: false,
    hoverlabel: { bgcolor: pal.surface, bordercolor: pal.border, font: { family: FONT, color: pal.ink } },
    geo: {
      // Orthographic = an actual globe, and it is draggable. Vector geography
      // ships with Plotly, so this needs no tile server and no API key.
      projection: { type: "orthographic", rotation: { lon: focus.lon, lat: focus.lat } },
      bgcolor: pal.surface,
      showland: true, landcolor: pal.grid,
      showocean: true, oceancolor: pal.surface,
      showcountries: true, countrycolor: pal.axis,
      showcoastlines: true, coastlinecolor: pal.axis,
      showframe: true, framecolor: pal.axis,
      lataxis: { showgrid: true, gridcolor: pal.grid },
      lonaxis: { showgrid: true, gridcolor: pal.grid },
    },
  };

  // newPlot on a live div keeps previously bound handlers, which would stack a
  // second click listener on every theme flip.
  if (gd.removeAllListeners) gd.removeAllListeners("plotly_click");

  Plotly.newPlot(gd, [trace], layout, CONFIG).then(() => {
    gd.on("plotly_click", (ev) => {
      const i = ev.points[0]?.pointIndex;
      if (i != null) selectProfile(cities[i].key);
    });
  });
  globeState = { plotted: true, theme, sig };

  renderCityList(cities, pal);
}

/* ---------- render ---------- */

/* A source with a handful of cycles can't support the readings its charts invite:
 * two points still draw a line, and a line reads as a trend. The live schedule
 * starts from nothing and accrues one cycle a week, so it says how young it is
 * until it isn't — the caveat clears itself as the history fills in. */
const YOUNG_RUNS = 8;

function renderStory(p) {
  const el = document.getElementById("story");
  el.textContent = p.story;
  const n = p.stats.runs;
  if (n >= YOUNG_RUNS) return;
  const caveat = document.createElement("span");
  caveat.className = "story-caveat";
  caveat.textContent =
    ` Only ${n} cycle${n === 1 ? " has" : "s have"} run so far, so the charts below are` +
    ` sparse and none of this is a trend yet.`;
  el.appendChild(caveat);
}

function render() {
  const p = byKey[current];
  const pal = P();
  renderStory(p);
  renderMap();
  statTiles(p.stats, p.target, p.retro);
  const charts = document.getElementById("charts");
  charts.innerHTML = "";
  const jobs = [];
  const xEnd = p.as_of[p.as_of.length - 1];
  // Features whose PSI is identically zero are dropped rather than drawn: the
  // statistic is undefined for them here (a near-constant reference collapses to
  // one bin), and a flat line at zero reads as "perfectly stable", which is the
  // opposite of "not measurable".
  const degenerate = p.psi_degenerate || [];
  const feat = DRIFT_FEATURES.filter((f) => p.psi[f] && !degenerate.includes(f));
  const R = p.retro;
  let lay, traces;

  // 0. What the model predicts, in the units it predicts it in. This is the
  // only card that shows the quantity itself; every other one shows a statistic
  // about it, which is a lot to ask a reader to infer.
  if (p.recent && p.recent.timestamp.length) {
    const t = p.target || { name: "target", units: "" };
    lay = plotBase(pal, `${t.name} (${t.units})`);
    if (t.who_24h_guideline) {
      thresholdLine(lay, pal, t.who_24h_guideline, `WHO 24h guideline · ${t.who_24h_guideline}`);
    }
    chartCard(jobs,
      `What the model forecasts: ${t.name}`,
      `What the air did, against what the model thought it would do. The call was made ` +
      `<em>a week before it happened</em>, working only from the weather forecast, with no ` +
      `knowledge of how dirty the air had been lately. The gap between the two lines is the ` +
      `mistake every other chart on this page is measuring.`,
      [
        { label: `measured ${t.name}`, color: pal.series[0], kind: "line" },
        { label: "champion prediction", color: pal.series[3], kind: "line" },
      ],
      [
        lineT(p.recent.timestamp, p.recent.actual, `measured ${t.name}`, pal.series[0]),
        lineT(p.recent.timestamp, p.recent.predicted, "champion prediction", pal.series[3], "dot"),
      ],
      lay);
  }

  // 1. Skill against a baseline you could deploy instead.
  //
  // The page used to headline raw RMSE and R². RMSE has no scale, so a filthy
  // city and a clean one cannot be compared and neither can two seasons of the
  // same city. R² normalises by the window's own variance, which in a calm
  // fortnight is tiny, so it reports catastrophe for a modest absolute error.
  // A skill score against a fixed alternative avoids both.
  if (R?.skill) {
    const days = R.skill.climatology_days;
    lay = plotBase(pal, "skill");
    thresholdLine(lay, pal, 0, "0 · no better than the baseline");
    driftRegion(lay, pal, p.drift_date, xEnd);
    chartCard(jobs,
      "What the model is worth",
      `These models earn their keep in the dirty season and lose to a rule of thumb in the clean one. ` +
      `Above the line the model beats the rule of thumb, which is "this hour usually looks like it did over the last ${days} days". Below the line it loses to it. ` +
      `That baseline gets to see recent pollution readings and the model never does, so it is a hard bar rather than a fair fight. It is here because it is the thing you could deploy instead.`,
      [{ label: `skill vs. ${days}-day daily profile`, color: pal.series[2], kind: "line" }],
      [lineT(p.as_of, R.skill.champion, "skill", pal.series[2], null, { hover: "%{y:+.2f}" })],
      lay);
  }

  // 2. One decay curve per model. THE chart the page was missing: the logged
  // champion error is a single line across eight different champions, so no
  // individual model's decay is visible in it. Scored on every window including
  // ones it never served, so a curve that keeps falling past its retirement is
  // what keeping that model would have cost.
  if (R?.decay?.length) {
    const serving = String(p.stats.champion_version);
    const n = R.decay.length;
    // A line per promoted model does not survive contact with a longer replay.
    // Kraków already has seven and gains one per promotion, and past about four
    // the chart stops being readable as anything but a smudge. So the cohort is
    // summarised and only two models are ever named: the one answering now, and
    // the longest unbroken run, which is where staleness shows up if it shows up
    // anywhere. Individual lines are kept only while there are few enough to
    // tell apart, since a quartile band over three models says less than the
    // three lines do.
    const SUMMARISE_ABOVE = 3;
    const longest = R.decay.reduce((a, b) => (b.served_weeks > a.served_weeks ? b : a));
    const quantile = (sorted, q) => {
      const i = (sorted.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
      return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
    };

    traces = [];
    const chipsD = [];
    if (n > SUMMARISE_ABOVE) {
      const weeks = [], p25 = [], p50 = [], p75 = [];
      const span = Math.max(...R.decay.map((d) => d.skill.length));
      for (let i = 0; i < span; i++) {
        const values = R.decay
          .map((d) => d.skill[i])
          .filter((v) => v != null && Number.isFinite(v))
          .sort((a, b) => a - b);
        // Stop where too few models are still running to summarise honestly,
        // rather than letting the band narrow to a single survivor's line.
        if (values.length < 3) break;
        weeks.push(i);
        p25.push(quantile(values, 0.25));
        p50.push(quantile(values, 0.5));
        p75.push(quantile(values, 0.75));
      }
      traces.push(
        { x: weeks, y: p75, mode: "lines", type: "scatter", line: { width: 0 },
          showlegend: false, hoverinfo: "skip" },
        { x: weeks, y: p25, mode: "lines", type: "scatter", line: { width: 0 },
          fill: "tonexty", fillcolor: "rgba(42,120,214,0.15)", showlegend: false,
          hovertemplate: "week %{x}<extra>middle half of models</extra>" },
        lineT(weeks, p50, "median model", pal.series[0], null,
          { width: 2.6, hover: "week %{x}: median skill %{y:+.2f}" }),
      );
      chipsD.push(
        { label: `median of ${n} models`, color: pal.series[0], kind: "line" },
        { label: "middle half", color: pal.series[0], kind: "dot" },
      );
    } else {
      R.decay.forEach((d, i) => traces.push(
        lineT(d.weeks, d.skill, `v${d.version}`, pal.series[0], null,
          { mode: "lines+markers", size: 5, width: 1.8,
            opacity: 0.35 + 0.4 * (i / Math.max(1, n - 1)),
            hover: "week %{x}: skill %{y:+.2f}" })));
      chipsD.push({ label: "each promoted model", color: pal.series[0], kind: "line" });
    }

    // Usually the same model: a champion the trigger can no longer replace is
    // both the longest run and the one still answering. Drawing it twice in two
    // colours invents a second model that does not exist, so they collapse into
    // one line whose label says it is both.
    const current = R.decay.find((d) => String(d.version) === serving);
    const named = current && current.version === longest.version
      ? [[longest, pal.crit,
          `serving now, and the longest run · v${longest.version}, ${longest.served_weeks}w`]]
      : [
          [longest, pal.crit, `longest run · v${longest.version}, ${longest.served_weeks}w`],
          [current, pal.series[1], `serving now · v${serving}`],
        ];
    for (const [entry, color, label] of named) {
      if (!entry) continue;
      traces.push(lineT(entry.weeks, entry.skill, label, color, null,
        { mode: "lines+markers", size: 5, width: 2.8, hover: "week %{x}: skill %{y:+.2f}" }));
      chipsD.push({ label, color, kind: "line" });
    }

    lay = plotBase(pal, "skill", { hovermode: "closest" });
    lay.xaxis.title = { text: "weeks in service", font: { color: pal.muted, size: 11 } };
    lay.margin.b = 44;
    thresholdLine(lay, pal, 0, "0 · no better than the baseline");
    chartCard(jobs,
      "How each model ages once it takes over",
      `Every model this city promoted, lined up on the day it took over rather than on the calendar, and followed past its own retirement so a line that keeps falling shows what keeping it would have cost. ` +
      (n > SUMMARISE_ABOVE
        ? `${n} models are summarised as a median and the middle half, because one line each stops being readable. `
        : `All ${n} are drawn, since there are few enough to tell apart. `) +
      `Two are named: the model answering now, and the longest unbroken run, which is where staleness shows up first.`,
      chipsD, traces, lay, "charts", true);
  }

  // 3. The retrain trigger, in µg/m³ instead of as a ratio.
  //
  // As a ratio the staircase is invisible: `error ÷ baseline` hides that the
  // denominator is reset on every promotion, and because retrains fire in the
  // dirty season each new champion inherits a *higher* bar than the one it
  // replaced. Plotting both in the same unit makes the ratchet the obvious
  // feature of the chart rather than a footnote in the methodology.
  if (R?.trigger) {
    lay = plotBase(pal, `error (${p.target?.units || "µg/m³"})`);
    driftRegion(lay, pal, p.drift_date, xEnd);
    traces = [
      lineT(p.as_of, R.trigger.bar, "retrain bar", pal.crit, "dot", { shape: "hv", width: 2 }),
      lineT(p.as_of, R.trigger.rmse, "champion error", pal.series[0]),
    ];
    const chipsT = [
      { label: "champion error", color: pal.series[0], kind: "line" },
      { label: "the bar it must cross", color: pal.crit, kind: "dash" },
    ];
    if (p.retrain.as_of.length) {
      traces.push(eventT(pal, p.retrain.as_of,
        p.retrain.as_of.map((d) => R.trigger.rmse[p.as_of.indexOf(d)]),
        "retrain triggered", pal.warn, "circle", 10));
      chipsT.push({ label: "retrain fires", color: pal.warn, kind: "dot" });
    }
    chartCard(jobs,
      "When it retrains, and against what bar",
      "A retrain fires when the error crosses the dotted bar, which is 1.25× whatever the model in service scored when it was trained. " +
      "The bar is a staircase because every promotion resets it, and promotions happen in the dirty season, so each new model inherits a higher bar than the one it replaced and the bar never comes back down. " +
      "Where the staircase ends up far above the error, the trigger has gone quiet and cannot fire again whatever the model does.",
      chipsT, traces, lay);
  }

  // 4. Weather drift as a band per ingredient per week, not six lines.
  //
  // Six series over four decades of log scale is unreadable, and the reading it
  // invites is one the statistic cannot support: PSI saturates near 11.5,
  // because at a seasonal gap most of the training range holds no current data
  // and every empty bin adds a fixed amount whatever the true distance. So it
  // is a dependable yes/no and an undependable how-much. Showing only the band
  // says the part that is true, and says it at a glance: mostly red, from the
  // first week, in every city.
  const bandOf = (v) => (v == null ? null : v > PSI_SIGNIFICANT ? 2 : v > PSI_MODERATE ? 1 : 0);
  lay = plotBase(pal, null, { height: 40 + 26 * feat.length, hovermode: "closest" });
  lay.margin.l = 132;
  lay.yaxis.showgrid = false;
  lay.yaxis.linecolor = "rgba(0,0,0,0)";
  // Every row named. plotBase asks for five ticks, which on a six-category axis
  // makes Plotly label every other one, and a heatmap row with no label is a
  // row the reader cannot identify.
  lay.yaxis.tickmode = "linear";
  lay.yaxis.dtick = 1;
  delete lay.yaxis.nticks;
  traces = [{
    type: "heatmap",
    x: p.as_of,
    // Reversed: Plotly puts the first category at the bottom, and reading order
    // should match the legend above it.
    y: [...feat].reverse(),
    z: [...feat].reverse().map((f) => p.psi[f].map(bandOf)),
    customdata: [...feat].reverse().map((f) => p.psi[f]),
    zmin: 0, zmax: 2, showscale: false, xgap: 1, ygap: 3,
    colorscale: [
      [0, pal.good], [0.333, pal.good],
      [0.333, pal.warn], [0.666, pal.warn],
      [0.666, pal.crit], [1, pal.crit],
    ],
    hovertemplate: "%{y}, %{x}<br>PSI %{customdata:.2f}<extra></extra>",
  }];
  chartCard(jobs,
    "How far the weather has moved from training",
    "The early warning. It compares incoming weather against what the model was trained on, one ingredient at a time, and needs no labels and no model, so it can raise a hand before any mistake shows up. " +
    "Green means the ingredient still looks like training, amber that it has shifted, red that it is properly different. " +
    "Read the whole block rather than any one cell. These cities go red early and stay red, which is all a statistic that maxes out long before the season does can honestly tell you. It is called PSI, and the bands are the industry convention." +
    (degenerate.length
      ? ` Not shown: ${degenerate.join(", ")}, which barely varies during training here, so the statistic is undefined rather than zero.`
      : ""),
    [
      { label: "looks like training", color: pal.good, kind: "dot" },
      { label: "shifted", color: pal.warn, kind: "dot" },
      { label: "properly different", color: pal.crit, kind: "dot" },
    ],
    traces, lay);

  // 5. The physical story, in the units the features are measured in. This is
  // what "the world moved" means before it is compressed into a statistic, and
  // it is the chart that justifies retraining at all.
  if (R?.factors) {
    const fm = R.factors.features;
    const shown = DRIFT_FEATURES.filter((f) => fm[f] && !degenerate.includes(f));
    const sm = smallMultiples(pal, shown.map((f, i) => ({
      name: f, x: p.as_of, y: fm[f], color: pal.series[i],
      band: R.factors.trained_on[f] || {},
    })), { zeroLine: false });
    chartCard(jobs,
      "What changed in the world",
      `Each weather ingredient, averaged over every monitoring window, in its own units. The shaded band is the range that ingredient held while the first model was being trained (${R.factors.bootstrap_train[0]} to ${R.factors.bootstrap_train[1]}). ` +
      "A line that leaves its band means the model is being asked about conditions it was never shown. This is the case for retraining, before any statistic is computed.",
      shown.map((f, i) => ({ label: f, color: pal.series[i], kind: "dot" })),
      sm.traces, sm.layout, "charts", true);
  }

  // 6. Champion vs. challenger on the held-out week
  lay = plotBase(pal, "RMSE");
  let chips3;
  if (p.holdout.as_of.length) {
    driftRegion(lay, pal, p.drift_date, xEnd);
    traces = [
      lineT(p.holdout.as_of, p.holdout.champion, "champion", pal.series[0]),
      lineT(p.holdout.as_of, p.holdout.challenger, "challenger", pal.series[1]),
    ];
    chips3 = [
      { label: "champion", color: pal.series[0], kind: "line" },
      { label: "challenger", color: pal.series[1], kind: "line" },
    ];
    if (p.promoted.as_of.length) {
      traces.push(eventT(pal, p.promoted.as_of, p.promoted.challenger, "promoted", pal.good, "star", 14));
      chips3.push({ label: "promoted", color: pal.good, kind: "star" });
    }
  } else {
    traces = [];
    chips3 = [];
    annotated(lay, pal, "No challenger trained yet, because no retrain has fired.");
  }
  chartCard(jobs,
    "The exam: champion vs. challenger",
    "Being newer is not a qualification. The model in service and the one just trained sit the same exam, a week of air neither has ever seen, and the newcomer only takes the job if it wins by more than 5% rather than by a nose. Whether passing this exam predicts anything is tested further down the page.",
    chips3, traces, lay);

  // 7. What the model leans on. Answers the question the coefficient chart
  // cannot: the slopes are per original unit, so a slope per hPa and a slope
  // per W/m² are not comparable, and the biggest number there is usually just
  // the feature with the smallest units.
  if (R?.importance) {
    const imp = R.importance;
    lay = plotBase(pal, null, { height: 230, hovermode: "closest" });
    lay.margin.l = 132;
    lay.yaxis.gridcolor = "rgba(0,0,0,0)";
    lay.xaxis.title = { text: "µg/m³ of prediction per 1-sd move", font: { color: pal.muted, size: 11 } };
    lay.xaxis.showgrid = true;
    lay.xaxis.gridcolor = pal.grid;
    lay.margin.b = 44;
    traces = [{
      type: "bar", orientation: "h",
      // Reversed so the largest sits at the top: Plotly stacks the first
      // category at the bottom of a horizontal bar chart.
      y: [...imp.features].reverse(), x: [...imp.values].reverse(),
      marker: { color: pal.series[0] },
      hovertemplate: "%{x:.1f} µg/m³ per 1-sd<extra>%{y}</extra>",
    }];
    chartCard(jobs,
      "What moves the prediction",
      `How much the model's answer shifts when each ingredient moves by one standard deviation. This is the comparable version of the coefficients below, in µg/m³, for version ${imp.version} over the most recent window. ` +
      "Boundary layer height, the depth of air that pollution is diluted into, would very likely top this list and is missing. Open-Meteo does not archive it at a seven-day lead, so shortwave radiation stands in for it.",
      [], traces, lay, "charts", true);
  }

  // 8. Model coefficients
  let chips4 = [];
  // Two versions is the least that can show a coefficient moving, and plotting
  // one is worse than plotting none: a single point gives Plotly a zero-width
  // time axis, which it fills with millisecond ticks. Say "not yet" instead.
  const coefPts = p.coef?.train_end?.length ?? 0;
  if (coefPts > 1) {
    // Same features and the same colour indices as the drift chart above, so a
    // reader tracking one feature across the two charts is tracking one colour.
    // The cyclical hour terms are fitted but not drawn: they encode the daily
    // cycle, which does not invert, and this chart is about signs flipping.
    //
    // One panel per feature rather than six lines on one axis -- see
    // smallMultiples() for why a shared y-axis cannot work for these units.
    const sm = smallMultiples(pal, DRIFT_FEATURES.map((f, i) => ({
      name: f, x: p.coef.train_end, y: p.coef[f], color: pal.series[i],
    })));
    traces = sm.traces;
    lay = sm.layout;
    chips4 = DRIFT_FEATURES.map((f, i) => ({ label: f, color: pal.series[i], kind: "dot" }));
  } else {
    lay = plotBase(pal, "coefficient", { hovermode: "closest" });
    traces = [];
    annotated(lay, pal, coefPts === 1
      ? "Only one model version so far, so there is no movement to trace yet."
      : "No model versions recorded yet.");
  }
  chartCard(jobs,
    "Model coefficients",
    "What the model believes about each ingredient, and how that belief shifts every time it is retrained. A line crossing zero is the model changing its mind about which way something pushes: wind used to clear the air, now it dirties it. Each panel has its own scale, because the ingredients are measured in different units.",
    chips4, traces, lay, "charts", coefPts > 1);

  renderBenchmark(p);
  // Both profile-independent, but re-rendered here so a theme flip recolours
  // them with everything else rather than leaving a section in the old palette.
  renderPooled();
  renderGate();
  renderControl(DATA.sweep);

  // All cards are in the DOM now and the grid has settled — plot at the real width.
  plotAll(jobs);
}

/* ---------- benchmark card ---------- */

const BENCH_LABEL = {
  champion_served: "Champion (loop)",
  champion_frozen: "Champion (never retrained)",
  pooled_cities: "One model, all six cities",
  persistence: "Persistence",
  seasonal_naive: "Seasonal naive",
  climatology: "Climatology",
  train_mean: "Training mean",
};

/* Per city, so this renders with the charts rather than in the static method
 * section above the footer. Dropped entirely when scripts/benchmark.py hasn't
 * run, rather than showing an empty frame.
 *
 * Grouped by information set, not ranked in one list. Persistence and seasonal
 * naive are handed the recent PM2.5; the champion only ever sees weather. In a
 * single ranking the two autoregressive baselines win in every city, which reads
 * as "the model loses" when what it actually says is that the groups are
 * answering different questions. Within-group order is the comparison that
 * means something. The gap between the groups is still reported, in its own
 * verdict tile, because it is a real result and burying it would be the same
 * dishonesty pointing the other way. */
function renderBenchmark(p) {
  const card = document.getElementById("bench");
  const b = p.benchmark;
  card.hidden = !b || !b.scored?.length;
  if (card.hidden) return;

  const byRmse = (x, y) => x.median_rmse - y.median_rmse;
  const weatherOnly = b.scored.filter((s) => !s.uses_past_target).sort(byRmse);
  const seesPast = b.scored.filter((s) => s.uses_past_target).sort(byRmse);

  const served = b.scored.find((s) => s.name === "champion_served");
  const frozen = b.scored.find((s) => s.name === "champion_frozen");
  const gain = served && frozen ? (1 - served.median_rmse / frozen.median_rmse) * 100 : null;

  const label = (s) => BENCH_LABEL[s.name] || s.name;
  const row = (s) =>
    `<tr${s.name.startsWith("champion") ? ' class="is-model"' : ""}>` +
    `<td>${label(s)}</td>` +
    `<td class="num">${s.median_rmse.toFixed(2)}</td>` +
    `<td class="bench-note">${s.detail}</td>` +
    `</tr>`;
  const group = (title, gloss, list) =>
    (list.length
      ? `<tr><td class="bench-group" colspan="3">${title}: <span>${gloss}</span></td></tr>` +
        list.map(row).join("")
      : "");

  const rows =
    group("Weather forecast only",
      "the six weather variables forecast for the target hour, plus the hour itself. What the champion sees.",
      weatherOnly) +
    group("Sees past PM2.5",
      "the readings available when the forecast was issued, a week before the target hour.",
      seesPast);

  const a = b.alpha;
  const verdict = [];
  if (gain != null) {
    verdict.push(`<div><b style="color:${gain >= 0 ? "var(--good)" : "var(--crit)"}">` +
      `${gain >= 0 ? "+" : ""}${gain.toFixed(1)}%</b>` +
      `${gain >= 0 ? "better for retraining" : "worse for retraining"}</div>`);
  }
  if (weatherOnly.length) {
    verdict.push(`<div><b>${label(weatherOnly[0])}</b>lowest on the forecast alone</div>`);
  }
  if (weatherOnly.length && seesPast.length) {
    // Which group wins flips with the horizon: repeating the last reading is
    // nearly unbeatable an hour out and nearly useless a week out. Report the
    // gap in whichever direction it actually runs.
    const w = weatherOnly[0], s = seesPast[0];
    const forecastWins = w.median_rmse <= s.median_rmse;
    const ratio = forecastWins ? s.median_rmse / w.median_rmse : w.median_rmse / s.median_rmse;
    // A ratio that rounds to "1.0×" is a statement that nothing separates the
    // two groups, dressed up as a finding. Give it the extra digit rather than
    // publishing a headline number that says nothing.
    verdict.push(`<div><b>${ratio.toFixed(ratio < 1.1 ? 2 : 1)}×</b>` +
      (forecastWins
        ? `${label(w)} below ${label(s)}, the best that past readings manage`
        : `${label(s)} below ${label(w)}, and it sees past PM2.5`) +
      `</div>`);
  }
  if (a) {
    verdict.push(`<div><b>alpha ${a.shipped}</b>shipped; ${a.best} scored best on ` +
      `${a.n_splits}-fold forward CV, costing ` +
      `${a.penalty_pct == null ? "—" : `${a.penalty_pct.toFixed(1)}%`} error</div>`);
  }

  card.innerHTML =
    `<div class="card-head"><h3>Does the model beat anything?</h3></div>` +
    `<p class="desc">Median error across the ${b.windows} monitoring windows of ` +
    `${b.monitor_days} days each, the same slices the charts above report on. Grouped by what ` +
    `each predictor is allowed to see: a forecaster issuing ${days(b.lead_days)} out may use ` +
    `readings up to that moment and no later, so the baselines below repeat a week-old ` +
    `observation rather than yesterday's. Lower is better.</p>` +
    `<div class="bench-wrap"><table class="bench">` +
    `<thead><tr><th>Predictor</th><th class="num">Median RMSE</th><th>What it does</th></tr></thead>` +
    `<tbody>${rows}</tbody></table></div>` +
    `<div class="verdict">${verdict.join("")}</div>`;
}

/* ---------- one model for every city? ---------- */

const POOLED_LABEL = { champion_served: "its own model, retrained", champion_frozen: "its own model, frozen",
  pooled_cities: "one model for all six" };

/* Grouped bars per city rather than a table, because the shape of the answer is
 * the answer: pooling helps where a city's own training window was a poor guide
 * to its future, and hurts where the city is unlike the others. */
function renderPooled() {
  const section = document.getElementById("pooled");
  const cities = (DATA.profiles || []).filter((p) => p.benchmark?.scored?.some((s) => s.name === "pooled_cities"));
  section.hidden = cities.length < 2;
  if (section.hidden) return;

  const pal = P();
  const host = document.getElementById("pooled-charts");
  host.innerHTML = "";
  const jobs = [];

  const pick = (p, name) => p.benchmark.scored.find((s) => s.name === name)?.median_rmse ?? null;
  const names = ["champion_served", "champion_frozen", "pooled_cities"];
  const colors = [pal.series[0], pal.series[3], pal.series[1]];

  const lay = plotBase(pal, "median error (µg/m³)", { hovermode: "closest", height: 320 });
  lay.barmode = "group";
  lay.margin.b = 54;
  const traces = names.map((name, i) => ({
    type: "bar", name: POOLED_LABEL[name],
    x: cities.map((p) => p.label), y: cities.map((p) => pick(p, name)),
    marker: { color: colors[i] }, showlegend: false,
    hovertemplate: `%{x}: %{y:.2f} µg/m³<extra>${POOLED_LABEL[name]}</extra>`,
  }));

  chartCard(jobs,
    "One model for all six cities, against six models",
    "Median error over the same monitoring windows, lower is better. The pooled model gets a separate intercept per city, which is doing most of the work: mean pollution runs from 7 µg/m³ in Melbourne to 84 in Delhi, and the same model without that adjustment scores 43% worse. So the weather slopes are shared and only the level is learned per place.",
    names.map((name, i) => ({ label: POOLED_LABEL[name], color: colors[i], kind: "dot" })),
    traces, lay, "pooled-charts", true);

  const beatsFrozen = cities.filter((p) => pick(p, "pooled_cities") < pick(p, "champion_frozen"));
  const beatsServed = cities.filter((p) => pick(p, "pooled_cities") < pick(p, "champion_served"));
  const delhi = cities.find((p) => p.label === "Delhi");
  document.getElementById("pooled-note").innerHTML =
    `<strong>The answer is no, and the near miss is the interesting part.</strong> One model over ` +
    `every city loses to the city's own retrained model in ${cities.length - beatsServed.length} of ` +
    `${cities.length} cities. But it beats the city's own <em>frozen</em> model in ` +
    `${beatsFrozen.length}, and in Delhi it is not close: ` +
    `${delhi ? `${pick(delhi, "pooled_cities").toFixed(1)} against ${pick(delhi, "champion_frozen").toFixed(1)} µg/m³` : ""}. ` +
    `Training on five other cities is worth more than a year of staleness, and less than keeping ` +
    `one city's model current. The cheap arrangement is not one model; it is one model plus the ` +
    `retraining loop, which is what the six cities here already are. ` +
    `Where pooling hurts most is Melbourne, the cleanest city by a distance, whose weather-to-` +
    `pollution relationship the other five outvote.`;

  plotAll(jobs);
}

/* ---------- did the promotion gate work? ---------- */

/* The gate promotes on one seven-day exam. That margin is in-sample *for the
 * decision*, being the number the decision was made on, so it cannot say
 * whether the decision was right. The out-of-sample check is what the winner
 * went on to deliver against the model it displaced, scored on the windows it
 * served, which is a counterfactual the loop has no reason to compute
 * at the time and `retrospect.py` computes afterwards.
 *
 * Pooled across every city rather than drawn per city, because five points per
 * city cannot show a pattern and twenty-seven can, and because the pattern is a
 * property of the *rule* rather than of any one place. */
function renderGate() {
  const section = document.getElementById("gate");
  // Published in data.json rather than declared here: the Streamlit app draws
  // the same split, and one threshold kept by hand in two languages is how the
  // two UIs end up quietly telling different stories. The literal is the
  // fallback for a data.json built before the field existed.
  const GATE_LONG_WEEKS = DATA.method?.gate_long_weeks ?? 20;
  const rows = (DATA.profiles || []).flatMap((p) =>
    (p.retro?.gate || []).map((g) => ({ ...g, city: p.label })));
  section.hidden = rows.length < 4;
  if (section.hidden) return;

  const pal = P();
  const host = document.getElementById("gate-charts");
  host.innerHTML = "";
  const jobs = [];

  const long = rows.filter((r) => r.weeks >= GATE_LONG_WEEKS);
  const short = rows.filter((r) => r.weeks < GATE_LONG_WEEKS);
  const mean = (a, k) => (a.length ? a.reduce((s, r) => s + r[k], 0) / a.length : NaN);

  const pt = (list, name, color) => ({
    x: list.map((r) => r.exam * 100), y: list.map((r) => r.delivered * 100),
    text: list.map((r) => `${r.city} v${r.version} · served ${r.weeks}w`),
    name, mode: "markers", type: "scatter",
    marker: { color, size: 11, line: { color: pal.surface, width: 1.5 } },
    showlegend: false,
    hovertemplate: "%{text}<br>exam %{x:+.1f}% → delivered %{y:+.1f}%<extra></extra>",
  });

  const all = rows.map((r) => r.exam * 100).concat(rows.map((r) => r.delivered * 100));
  const lo = Math.min(...all, 0) - 4, hi = Math.max(...all) + 4;

  const lay = plotBase(pal, "what it went on to deliver (%)", { hovermode: "closest", height: 330 });
  lay.xaxis.title = { text: "margin on the seven-day exam (%)", font: { color: pal.muted, size: 11 } };
  lay.xaxis.showgrid = true;
  lay.xaxis.gridcolor = pal.grid;
  lay.margin.b = 46;
  // y = x is where a perfectly calibrated exam would put every point.
  lay.shapes.push({
    type: "line", xref: "x", yref: "y", x0: lo, y0: lo, x1: hi, y1: hi,
    line: { color: pal.muted, width: 1, dash: "dot" }, layer: "below",
  });
  // Below this line the promotion made things worse than leaving the old model.
  lay.shapes.push({
    type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 0, y1: 0,
    line: { color: pal.crit, width: 1 }, layer: "below",
  });
  lay.annotations.push({
    xref: "paper", x: 0, y: 0, yref: "y", text: "0 · no better than the model it replaced",
    showarrow: false, font: { color: pal.muted, size: 10 },
    xanchor: "left", yanchor: "bottom", xshift: 3, yshift: 2, bgcolor: pal.surface, borderpad: 1,
  });

  chartCard(jobs,
    "What the exam promised against what it delivered",
    `One point per promotion, across every city. Horizontal is the margin the challenger won its exam by, which is the number the gate decided on. Vertical is what it went on to deliver over the weeks it then served, measured against the model it replaced and scored on those same windows. ` +
    `On the dotted diagonal the exam predicted the outcome perfectly. Below the red line the promotion made things worse.`,
    [
      { label: `served under ${GATE_LONG_WEEKS} weeks`, color: pal.series[2], kind: "dot" },
      { label: `served ${GATE_LONG_WEEKS}+ weeks`, color: pal.crit, kind: "dot" },
      { label: "perfect calibration", color: pal.muted, kind: "dash" },
    ],
    [pt(short, "short", pal.series[2]), pt(long, "long", pal.crit)],
    lay, "gate-charts", true);

  document.getElementById("gate-note").innerHTML =
    `<strong>The result:</strong> the exam is honest and well calibrated over the horizon it tests. ` +
    `Across ${short.length} promotions that served under ${GATE_LONG_WEEKS} weeks it promised ` +
    `${pct(mean(short, "exam") * 100, 1)} and delivered ${pct(mean(short, "delivered") * 100, 1)}, and not one of them ` +
    `made things worse. But every one of the ${long.length} models that ended up serving ${GATE_LONG_WEEKS} weeks or more ` +
    `delivered a <em>negative</em> margin, ${pct(mean(long, "exam") * 100, 1)} promised against ` +
    `${pct(mean(long, "delivered") * 100, 1)} delivered, despite passing the same exam just as convincingly. ` +
    `A seven-day exam can certify a model for about a month. It cannot certify one for half a year. ` +
    `And the long-serving models are long-serving for a reason: they were promoted at the seasonal peak, ` +
    `which sets the retrain bar so high that nothing crosses it afterwards. So the two faults compound. ` +
    `The trigger goes quiet, the model stays, and the exam that cleared it is asked to stand for far longer than it can.`;

  plotAll(jobs);
}

/* ---------- controlled experiment ---------- */

/* Both detectors are plotted against both knobs, indexed to their reading at
 * knob zero. Indexing rather than a second y-axis: PSI and an error ratio have
 * no common unit, and a dual axis would let the two curves be scaled into any
 * story at all. Sharing one axis means the flat line is honestly flat.
 *
 * Colour follows the detector, not the chart, so the same series is the same
 * colour in both panels. */
function renderControl(sweep) {
  const section = document.getElementById("control");
  section.hidden = !sweep;
  if (!sweep) return;

  const pal = P();
  const host = document.getElementById("control-charts");
  host.innerHTML = "";
  const jobs = [];

  const KNOBS = [
    {
      key: "feature_shift",
      title: "Turning the covariate knob",
      desc: "The feature distributions are pushed further from the training window, while the " +
        "relationship between weather and pollution is left alone. Data drift should climb and " +
        "performance drift should not move.",
    },
    {
      key: "drift_strength",
      title: "Turning the concept knob",
      desc: "The relationship between weather and pollution is changed, while the feature " +
        "distributions are left alone. Performance drift should climb and data drift should not " +
        "move.",
    },
  ];

  const chips = [
    { label: "data drift (PSI)", color: pal.series[0], kind: "line" },
    { label: "performance drift", color: pal.series[1], kind: "line" },
  ];

  for (const knob of KNOBS) {
    const s = sweep[knob.key];
    if (!s) continue;
    const lay = plotBase(pal, "response, × of knob at zero");
    lay.xaxis.title = { text: "knob level", font: { color: pal.muted, size: 11 } };
    lay.margin.b = 44;
    thresholdLine(lay, pal, 1, "1.0 · unchanged");
    chartCard(jobs, knob.title, knob.desc, chips,
      [
        lineT(s.level, s.psi_rel, "data drift (PSI)", pal.series[0]),
        lineT(s.level, s.perf_rel, "performance drift", pal.series[1]),
      ],
      lay, "control-charts");
  }

  // Stated from the data rather than written down, so the claim cannot outlive
  // the result it describes.
  const fs = sweep.feature_shift, ds = sweep.drift_strength;
  const last = (a) => a[a.length - 1];
  if (fs && ds) {
    document.getElementById("control-note").innerHTML =
      `<strong>The result:</strong> turn the dial that changes how weather becomes pollution, and ` +
      `the "is it getting things wrong" alarm climbs ${last(ds.perf_rel).toFixed(1)}× while the ` +
      `"does the weather look different" alarm does not move at all ` +
      `(${last(ds.psi_rel).toFixed(2)}×). It is right not to: the weather has not changed. Turn ` +
      `the other dial and it reverses, with the weather alarm climbing ` +
      `${last(fs.psi_rel).toFixed(1)}× and the error alarm staying put ` +
      `(${last(fs.perf_rel).toFixed(2)}×). Each answers only to the thing it is supposed to ` +
      `watch, which is the whole reason for having two, and it is the one claim no real city can ` +
      `settle, because in a real city nobody knows what the right answer was.`;
  }

  plotAll(jobs);
}

/* ---------- method section ---------- */

/* Rendered from data.json rather than written into the HTML, because
 * build_site.py introspects these values out of the running code. Retyping them
 * here would let the page describe a model the loop stopped being. */
function specRows(el, rows) {
  el.innerHTML = rows
    .map(([k, v, gloss]) =>
      `<dt>${k}</dt><dd>${v}${gloss ? ` <span class="gloss">${gloss}</span>` : ""}</dd>`)
    .join("");
}

function renderMethod(m) {
  if (!m) return;
  const p = m.params;
  const pct = (f) => `${Math.round(f * 100)}%`;
  const code = (s) => `<code>${s}</code>`;

  document.getElementById("method").hidden = false;
  document.getElementById("method-sub").textContent =
    "Every city on this page comes out of the same four steps, run once a week. No city gets " +
    "special treatment: they all run on identical settings, so where two cities behave " +
    "differently, it is their air that differs and not their tuning.";

  const steps = [
    ["Mark its homework", `Take the model in service and check the last ${days(p.monitor_days)} of forecasts against what the air did.`],
    ["Look for trouble, two ways", "First, whether the weather has stopped resembling what the model learned from. Separately, whether the model is getting things wrong. The first can fire before any damage shows. Only the second is allowed to spend money."],
    ["Train a rival", `If the error is ${p.perf_drift_threshold}× what it used to be, train a fresh model on the last ${days(p.challenger_train_days)} and let it apply for the job.`],
    ["Make it earn the job", `Both sit the same exam, ${days(p.holdout_days)} of air neither has seen. The newcomer has to win by ${pct(p.promotion_margin)} rather than squeak past, or it is thrown away.`],
  ];
  document.getElementById("steps").innerHTML = steps
    .map(([title, desc], i) =>
      `<li class="step"><div class="step-n">Step ${i + 1}</div>` +
      `<div class="step-t">${title}</div><div class="step-d">${desc}</div></li>`)
    .join("");

  specRows(document.getElementById("model-spec"), [
    ["estimator", code(`${m.estimator}(alpha=${m.alpha})`)],
    ["features", m.features.map(code).join(" ")],
    ["target", code(m.target), "µg/m³"],
    ["horizon",
      m.horizon_days == null ? "mixed across cities"
        : m.horizon_days > 0 ? `${days(m.horizon_days)} ahead`
        : "none, a same-hour estimate",
      m.horizon_days > 0
        ? "the features are the weather forecast for the target hour as it was issued that far ahead, so the forecast's own error is carried into the prediction"
        : "this hour's weather maps to this hour's pollution, so it is not a forecast"],
    ["training", `chronological ${pct(1 - m.val_fraction)}/${pct(m.val_fraction)} tail split`,
      "then refit on the full window"],
    ["baseline", "RMSE on the held-out tail", "never a random split, so it is measured the same way production will be"],
  ]);

  document.getElementById("param-desc").textContent = m.params_uniform
    ? "Every threshold the loop decides with, in one place, and identical for every city so a city's " +
      "behaviour reflects its weather and not its tuning."
    : "Every threshold the loop decides with. These are the values for the first source shown; they are " +
      "not currently identical across cities.";

  const b = m.psi_bands;
  specRows(document.getElementById("param-spec"), [
    ["monitor window", code(days(p.monitor_days)), "what the champion is scored on"],
    ["challenger training", code(days(p.challenger_train_days)), "recent history a challenger learns from"],
    ["holdout", code(days(p.holdout_days)), "judged on this, excluded from both"],
    ["retrain trigger", code(`error ÷ baseline > ${p.perf_drift_threshold}`), `${pct(p.perf_drift_threshold - 1)} worse`],
    ["PSI significant", code(`> ${p.psi_threshold}`), `stable below ${b.stable}, moderate ${b.stable} to ${b.significant}`],
    ["promotion margin", code(`> ${pct(p.promotion_margin)}`), "or the challenger is rejected"],
  ]);
}

/* The unattended loop, as a count of cycles and the dates they cover.
 *
 * It used to sit in the city selector, where it drew the same charts as a city
 * on two data points and invited a comparison with a 48-week replay. It is a
 * mode, not a place. Reported here as facts about what the tracking store
 * holds, with no claim about what put them there: a cycle run by hand and a
 * cycle run by the weekly Action leave identical records, so asserting the cron
 * is working would be asserting something this page cannot see. */
function renderSchedule(s) {
  const card = document.getElementById("schedule-card");
  card.hidden = !s;
  if (!s) return;
  const plural = s.cycles === 1 ? "cycle has" : "cycles have";
  document.getElementById("schedule-desc").innerHTML =
    `Alongside the six replays, the same code runs one cycle at a time against live Kraków data, ` +
    `keeping its history in a tracking store committed back to the repository. So far ${s.cycles} ` +
    `${plural} been recorded, covering ${s.first} to ${s.last}, most recently logged on ` +
    `${s.logged_at}. It is still serving its first model and has triggered ${s.retrains} retrains, ` +
    `which is what two cycles can be expected to show. ` +
    `<span class="gloss">The records do not say what ran them. A cycle started by hand and one ` +
    `started by the weekly job are indistinguishable in the store, so this is a count of what is ` +
    `there rather than evidence the schedule is firing.</span>`;
}

function renderDataLinks(data) {
  // One raw CSV per monitored city, so the list grows with the cities.
  const raw = data.raw_data || [];
  const parts = [];
  if (raw.length) {
    const links = raw.map((r) =>
      `<a href="${r.file}" title="${r.start} → ${r.end}">${r.city} (${r.rows.toLocaleString()} h)</a>`);
    parts.push(`<strong>Raw hourly data:</strong> ${links.join(", ")}`);
  }
  parts.push(`<strong>Chart data:</strong> <a href="data.json">data.json</a>`);
  document.getElementById("data-links").innerHTML = parts.join(" &nbsp;·&nbsp; ");
}

/* ---------- boot ---------- */

/* The one place selection changes. The segmented control, the map markers and the
 * city rows are three views of the same state, so they all route through here and
 * re-read it rather than each tracking their own. */
function selectProfile(key) {
  if (key === current) return;
  current = key;
  syncSegmented();
  render();
}

function syncSegmented() {
  [...document.getElementById("segmented").children].forEach((c) =>
    c.setAttribute("aria-selected", String(c.dataset.key === current)));
}

function buildSegmented() {
  const seg = document.getElementById("segmented");
  seg.innerHTML = "";
  DATA.profiles.forEach((p, i) => {
    byKey[p.key] = p;
    const b = document.createElement("button");
    b.textContent = p.label;
    b.dataset.key = p.key;
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(i === 0));
    b.addEventListener("click", () => selectProfile(p.key));
    seg.appendChild(b);
  });
}

function setupTheme() {
  // ?theme=light|dark pins the palette for this load without touching the
  // stored preference. It exists so the screenshots in the README can be
  // regenerated headlessly against a known theme, and it doubles as a way to
  // link someone to the page as you are seeing it.
  const asked = new URLSearchParams(location.search).get("theme");
  const saved = asked === "dark" || asked === "light" ? asked : localStorage.getItem("driftloop-theme");
  if (saved === "dark" || saved === "light") document.documentElement.setAttribute("data-theme", saved);
  document.getElementById("theme").addEventListener("click", () => {
    const next = resolveTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("driftloop-theme", next);
    render();
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!document.documentElement.getAttribute("data-theme")) render();
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
  DRIFT_FEATURES = DATA.method?.drift_features || [];
  document.getElementById("built").textContent = `Snapshot · built ${DATA.built}`;
  // Static across profiles, so it is rendered once rather than per selection.
  renderMethod(DATA.method);
  renderSchedule(DATA.schedule);
  renderDataLinks(DATA);
  buildSegmented();
  if (DATA.profiles.length) {
    current = DATA.profiles[0].key;
    render();
  }
}

document.addEventListener("DOMContentLoaded", main);
