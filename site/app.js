/* Client-side renderer for the drift-loop dashboard.
 * Fetches data.json (from scripts/build_site.py) and builds interactive Plotly
 * charts. Chart titles, legends, and descriptions live in the HTML card — the
 * Plotly canvas holds only the data — and everything is theme-aware (light/dark).
 * Palette follows the validated data-viz reference (series slots 1–4 + status). */

const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const FEATURES = ["temperature", "wind_speed", "humidity"];
const PSI_SIGNIFICANT = 0.25;
const PERF_THRESHOLD = 1.25;
const CONFIG = { displayModeBar: false, responsive: true };

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

function statTiles(stats) {
  const pal = P();
  const r2 = stats.latest_r2;
  const r2color = r2 == null ? pal.ink : r2 >= 0.5 ? pal.good : r2 >= 0.2 ? pal.warn : pal.crit;
  const tiles = [
    { v: stats.runs, k: "Monitoring runs" },
    { v: stats.retrains, k: "Retrains" },
    { v: stats.promotions, k: "Promotions" },
    { v: r2 == null ? "—" : r2.toFixed(2), k: "Latest champion R²", color: r2color },
  ];
  document.getElementById("tiles").innerHTML = tiles.map((t) =>
    `<div class="tile"><div class="tile-v"${t.color ? ` style="color:${t.color}"` : ""}>${t.v}</div>` +
    `<div class="tile-k">${t.k}</div></div>`).join("");
}

/* ---------- render ---------- */

function render() {
  const p = byKey[current];
  const pal = P();
  document.getElementById("story").textContent = p.story;
  statTiles(p.stats);
  const charts = document.getElementById("charts");
  charts.innerHTML = "";
  const jobs = [];
  const xEnd = p.as_of[p.as_of.length - 1];
  const feat = FEATURES.filter((f) => p.psi[f]);

  // 1. Data drift — PSI per feature
  let lay = plotBase(pal, "PSI");
  driftRegion(lay, pal, p.drift_date, xEnd);
  thresholdLine(lay, pal, PSI_SIGNIFICANT, "0.25 · significant");
  let traces = feat.map((f, i) => lineT(p.as_of, p.psi[f], f, pal.series[i]));
  chartCard(jobs,
    "Data drift",
    "Each feature's recent distribution vs. the champion's training window (Population Stability Index). Above 0.25 is a meaningful shift.",
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
    "The champion's live error divided by its error at training time. Crossing 1.25 (25% worse) triggers a retrain.",
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
    annotated(lay, pal, "No challenger trained yet — no retrain has fired.");
  }
  chartCard(jobs,
    "Champion vs. challenger",
    "When a retrain fires, both models are scored on a held-out week neither has seen. The challenger is promoted only if it wins by a margin.",
    chips3, traces, lay);

  // 4. Model coefficients
  lay = plotBase(pal, "coefficient", { hovermode: "closest" });
  let chips4 = [];
  if (p.coef) {
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
    annotated(lay, pal, "Only one model version so far.");
  }
  chartCard(jobs,
    "Model coefficients",
    "The Ridge model's learned slope per feature, across versions. A slope crossing zero is the real-world relationship inverting — concept drift.",
    chips4, traces, lay);

  // All cards are in the DOM now and the grid has settled — plot at the real width.
  jobs.forEach((j) => Plotly.newPlot(j.div, j.traces, j.layout, CONFIG));
}

function renderDataLinks(data) {
  const raw = data.raw_data;
  const parts = [];
  if (raw) parts.push(
    `<strong>Raw data:</strong> <a href="${raw.file}">${raw.rows.toLocaleString()} hourly Kraków observations</a> ` +
    `(${raw.start} → ${raw.end}, CSV)`);
  parts.push(`<strong>Chart data:</strong> <a href="data.json">data.json</a>`);
  document.getElementById("data-links").innerHTML = parts.join(" &nbsp;·&nbsp; ");
}

/* ---------- boot ---------- */

function buildSegmented() {
  const seg = document.getElementById("segmented");
  seg.innerHTML = "";
  DATA.profiles.forEach((p, i) => {
    byKey[p.key] = p;
    const b = document.createElement("button");
    b.textContent = p.label;
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(i === 0));
    b.addEventListener("click", () => {
      current = p.key;
      [...seg.children].forEach((c) => c.setAttribute("aria-selected", "false"));
      b.setAttribute("aria-selected", "true");
      render();
    });
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
  renderDataLinks(DATA);
  buildSegmented();
  if (DATA.profiles.length) {
    current = DATA.profiles[0].key;
    render();
  }
}

document.addEventListener("DOMContentLoaded", main);
