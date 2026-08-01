/* Client-side renderer for the drift-loop dashboard.
 * Fetches data.json (from scripts/build_site.py) and builds interactive Plotly
 * charts. Chart titles, legends, and descriptions live in the HTML card — the
 * Plotly canvas holds only the data — and everything is theme-aware (light/dark).
 * Palette follows the validated data-viz reference (series slots 1–4 + status). */

const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const FEATURES = ["temperature", "wind_speed", "humidity"];
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
    series: ["#2a78d6", "#008300", "#e87ba4", "#eda100"],
    good: "#0a9a2e", warn: "#c98500", crit: "#d03b3b",
    drift: "rgba(236,131,90,0.09)",
  },
  dark: {
    surface: "#17171a", ink: "#f4f4f2", ink2: "#b8b7b0", muted: "#8b8983",
    grid: "#29292c", axis: "#3a3a3e", border: "rgba(255,255,255,0.10)",
    series: ["#3987e5", "#22b45e", "#e07aa6", "#e0a53a"],
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

const lineT = (x, y, name, color, dash) => ({
  x, y, name, mode: "lines", type: "scatter",
  line: { color, width: 2.4, dash: dash || null, shape: "linear" },
  hovertemplate: "%{y:.3f}<extra>" + name + "</extra>",
});

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
function chartCard(jobs, title, desc, chips, traces, layout) {
  const card = document.createElement("section");
  card.className = "card";
  card.innerHTML =
    `<div class="card-head"><h3>${title}</h3><div class="legend">${legendHTML(chips)}</div></div>` +
    `<p class="desc">${desc}</p><div class="plot"></div>`;
  document.getElementById("charts").appendChild(card);
  jobs.push({ div: card.querySelector(".plot"), traces, layout });
}

function statTiles(stats, target) {
  const pal = P();
  const r2 = stats.latest_r2;
  const guide = target?.who_24h_guideline;
  const air = stats.latest_actual;
  // Lead with the physical quantity. Without it the page is all monitoring
  // machinery and never says what is predicted or how bad the air actually is.
  const airColor = air == null || guide == null ? pal.ink
    : air >= guide * 3 ? pal.crit : air >= guide ? pal.warn : pal.good;

  const tiles = [];
  if (air != null) {
    tiles.push({
      v: air, sub: target.units, color: airColor,
      k: `Latest ${target.name} reading${guide ? ` · WHO 24h guideline ${guide}` : ""}`,
    });
  }
  if (stats.latest_rmse != null) {
    tiles.push({
      v: `±${stats.latest_rmse}`, sub: target?.units,
      k: `Champion error${r2 == null ? "" : ` · R² ${r2.toFixed(2)}`}`,
    });
  }
  tiles.push(
    { v: stats.runs, k: "Monitoring runs" },
    { v: stats.retrains, k: "Retrains" },
    { v: stats.promotions, k: "Promotions" },
  );

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

/* Spin the globe to a new centre instead of cutting to it. The rotation carries
 * the information a cut throws away: how far apart two cities actually are, and
 * which way round the world you travelled to get there. */
let globeAnim = null;
// Bumped on every new spin. A spin chained onto a relayout promise can't be
// stopped with cancelAnimationFrame, so each frame checks it still owns the
// globe before doing anything.
let globeSpin = 0;
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

function spinGlobeTo(gd, target) {
  const spin = ++globeSpin;
  if (globeAnim) {
    cancelAnimationFrame(globeAnim);
    globeAnim = null;
  }
  // Read the *live* rotation, so interrupting a spin — or spinning after the user
  // has dragged the globe by hand — starts from where it actually is.
  const from = { ...(gd.layout?.geo?.projection?.rotation || { lon: 0, lat: 0 }) };
  // Normalise into (-180, 180] so we always take the short way round: Tokyo to
  // Los Angeles crosses the Pacific, it doesn't wind back across Eurasia.
  const dLon = ((target.lon - from.lon + 540) % 360) - 180;
  const dLat = target.lat - from.lat;
  if (Math.abs(dLon) < 0.01 && Math.abs(dLat) < 0.01) return;

  const settle = (lon, lat) =>
    Plotly.relayout(gd, { "geo.projection.rotation.lon": lon, "geo.projection.rotation.lat": lat });

  if (reducedMotion()) return void settle(target.lon, target.lat);

  // A neighbouring city shouldn't take as long as a half-world spin.
  const duration = 380 + 620 * Math.min(1, Math.hypot(dLon, dLat) / 180);
  // One clock for the baseline and for every reading of it. Mixing
  // performance.now() with the timestamp requestAnimationFrame hands the
  // callback is a bug: the two need not share a time origin, and when they
  // don't the easing goes negative and the globe creeps backwards.
  const t0 = performance.now();

  const frame = () => {
    if (spin !== globeSpin) return; // a newer spin owns the globe now
    const t = Math.min(1, Math.max(0, (performance.now() - t0) / duration));
    const e = easeInOut(t);
    // Wait for the projection to finish before asking for the next position.
    // Relayout on a geo subplot re-projects the whole topojson, which takes
    // longer than a frame; firing one per rAF queues them faster than they can
    // render and the backlog is what makes the spin tear. Pacing on the promise
    // renders as many intermediate positions as the browser can sustain, evenly
    // — and the easing stays time-based, so the duration is unchanged.
    settle(from.lon + dLon * e, from.lat + dLat * e)
      .then(() => {
        if (spin !== globeSpin || t >= 1) {
          globeAnim = null;
          return;
        }
        globeAnim = requestAnimationFrame(frame);
      })
      .catch(() => {
        globeAnim = null;
      });
  };
  globeAnim = requestAnimationFrame(frame);
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
    spinGlobeTo(gd, focus);
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
  if (globeAnim) { cancelAnimationFrame(globeAnim); globeAnim = null; }

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
  statTiles(p.stats, p.target);
  const charts = document.getElementById("charts");
  charts.innerHTML = "";
  const jobs = [];
  const xEnd = p.as_of[p.as_of.length - 1];
  const feat = FEATURES.filter((f) => p.psi[f]);
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
      `Measured hourly ${t.name} in ${t.units} over the most recent monitoring window, against ` +
      `what the serving champion forecast for those hours <em>a week before they happened</em>, ` +
      `from the weather forecast alone and no past ${t.name}. The gap between the two lines is ` +
      `the error every other chart summarises.`,
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

  // 1. Data drift — PSI per feature
  lay = plotBase(pal, "PSI");
  driftRegion(lay, pal, p.drift_date, xEnd);
  thresholdLine(lay, pal, PSI_SIGNIFICANT, "0.25 · significant");
  traces = feat.map((f, i) => lineT(p.as_of, p.psi[f], f, pal.series[i]));
  chartCard(jobs,
    "Data drift",
    "Each feature's recent distribution against the champion's training window (Population Stability Index). Above 0.25 counts as a meaningful shift.",
    feat.map((f, i) => ({ label: f, color: pal.series[i], kind: "line" })),
    traces, lay);

  // 2. Performance drift & retrains
  lay = plotBase(pal, "error ratio");
  driftRegion(lay, pal, p.drift_date, xEnd);
  thresholdLine(lay, pal, PERF_THRESHOLD, "1.25 · retrain trigger");
  traces = [lineT(p.as_of, p.perf_ratio, "perf ratio", pal.series[0])];
  const chips2 = [{ label: "champion error ÷ baseline", color: pal.series[0], kind: "line" }];
  if (p.retrain.as_of.length) {
    traces.push(eventT(pal, p.retrain.as_of, p.retrain.perf, "retrain triggered", pal.warn, "circle", 11));
    chips2.push({ label: "retrain", color: pal.warn, kind: "dot" });
  }
  chartCard(jobs,
    "Performance drift & retrains",
    "The champion's live error divided by its error at training time. Crossing 1.25, meaning 25% worse, triggers a retrain.",
    chips2, traces, lay);

  // 3. Champion vs. challenger on the held-out week
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
    "Champion vs. challenger",
    "When a retrain fires, both models are scored on a held-out week neither has seen. The challenger is promoted only if it wins by a margin.",
    chips3, traces, lay);

  // 4. Model coefficients
  lay = plotBase(pal, "coefficient", { hovermode: "closest" });
  let chips4 = [];
  // Two versions is the least that can show a coefficient moving, and plotting
  // one is worse than plotting none: a single point gives Plotly a zero-width
  // time axis, which it fills with millisecond ticks. Say "not yet" instead.
  const coefPts = p.coef?.train_end?.length ?? 0;
  if (coefPts > 1) {
    lay.shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 0, y1: 0, line: { color: pal.axis, width: 1 } });
    traces = FEATURES.map((f, i) => ({
      x: p.coef.train_end, y: p.coef[f], name: f, mode: "lines+markers", type: "scatter",
      line: { color: pal.series[i], width: 2.4 },
      marker: { size: 8, color: pal.series[i], line: { color: pal.surface, width: 2 } },
      hovertemplate: "%{y:.3f}<extra>" + f + "</extra>",
    }));
    chips4 = FEATURES.map((f, i) => ({ label: f, color: pal.series[i], kind: "dot" }));
  } else {
    traces = [];
    annotated(lay, pal, coefPts === 1
      ? "Only one model version so far, so there is no movement to trace yet."
      : "No model versions recorded yet.");
  }
  chartCard(jobs,
    "Model coefficients",
    "The Ridge model's learned slope per feature, across versions. A slope crossing zero means the real-world relationship has inverted, which is concept drift.",
    chips4, traces, lay);

  renderBenchmark(p);

  // All cards are in the DOM now and the grid has settled — plot at the real width.
  jobs.forEach((j) => Plotly.newPlot(j.div, j.traces, j.layout, CONFIG));
}

/* ---------- benchmark card ---------- */

const BENCH_LABEL = {
  champion_served: "Champion (loop)",
  champion_frozen: "Champion (never retrained)",
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
      "temperature, wind speed and humidity as forecast for the target hour. What the champion sees.",
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
    verdict.push(`<div><b>${ratio.toFixed(1)}×</b>` +
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
    "Everything above is produced by one loop of four steps, run once a week against a " +
    "small model. Nothing here is hand-tuned per city.";

  const steps = [
    ["Monitor", `Score the live champion on the last ${days(p.monitor_days)} of data.`],
    ["Detect", "Two independent signals: PSI on the feature distributions, and the champion's error against its error at training time."],
    ["Retrain", `If error crosses ${p.perf_drift_threshold}×, train a challenger on the last ${days(p.challenger_train_days)}.`],
    ["Promote", `Both models scored on a held-out ${days(p.holdout_days)} neither has seen. The challenger must win by ${pct(p.promotion_margin)} or it is rejected.`],
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
  const saved = localStorage.getItem("driftloop-theme");
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
  document.getElementById("built").textContent = `Snapshot · built ${DATA.built}`;
  // Static across profiles, so it is rendered once rather than per selection.
  renderMethod(DATA.method);
  renderDataLinks(DATA);
  buildSegmented();
  if (DATA.profiles.length) {
    current = DATA.profiles[0].key;
    render();
  }
}

document.addEventListener("DOMContentLoaded", main);
