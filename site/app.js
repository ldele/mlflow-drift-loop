/* Client-side renderer for the drift-loop dashboard.
 * Fetches data.json (written by scripts/build_site.py) and builds interactive
 * Plotly charts. Palette + chart shapes mirror dashboard/theme.py so the static
 * site matches the Streamlit app. Chart titles + descriptions live in the HTML
 * card (below), not inside Plotly, so they stay crisp and readable. */

const PAL = {
  surface: "#ffffff", ink: "#1a1917", ink2: "#57544e", muted: "#8a867d",
  grid: "#e7e5de", line: "#d7d5cc",
  series: ["#2a78d6", "#008300", "#e87ba4", "#eda100"],
  good: "#0ca30c", warn: "#fab219", crit: "#d03b3b",
  drift: "rgba(236, 131, 90, 0.07)",
};
const FEATURES = ["temperature", "wind_speed", "humidity"];
const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const CONFIG = { displayModeBar: false, responsive: true };
const PSI_SIGNIFICANT = 0.25;
const PERF_THRESHOLD = 1.25;

function baseLayout(yTitle) {
  return {
    paper_bgcolor: PAL.surface, plot_bgcolor: PAL.surface,
    font: { family: FONT, size: 12, color: PAL.ink2 },
    margin: { l: 54, r: 18, t: 30, b: 38 }, height: 320, hovermode: "x unified",
    legend: { orientation: "h", yanchor: "bottom", y: 1.02, x: 0, font: { color: PAL.ink2 } },
    xaxis: { showgrid: false, linecolor: PAL.line, tickfont: { color: PAL.muted }, zeroline: false },
    yaxis: {
      title: { text: yTitle, font: { color: PAL.muted, size: 11 } },
      gridcolor: PAL.grid, linecolor: PAL.line, tickfont: { color: PAL.muted }, zeroline: false,
    },
    shapes: [], annotations: [],
  };
}

function driftRegion(layout, driftDate, xEnd) {
  if (!driftDate) return;
  layout.shapes.push({
    type: "rect", xref: "x", yref: "paper", x0: driftDate, x1: xEnd, y0: 0, y1: 1,
    fillcolor: PAL.drift, line: { width: 0 }, layer: "below",
  });
  layout.shapes.push({
    type: "line", xref: "x", yref: "paper", x0: driftDate, x1: driftDate, y0: 0, y1: 1,
    line: { color: PAL.muted, width: 1, dash: "dash" },
  });
  layout.annotations.push({
    x: driftDate, y: 1, xref: "x", yref: "paper", text: "regime shift", showarrow: false,
    font: { color: PAL.muted, size: 10 }, xanchor: "left", xshift: 4, yanchor: "top",
  });
}

const line = (x, y, name, color, dash) => ({
  x, y, name, mode: "lines", type: "scatter",
  line: { color, width: 2, dash: dash || null },
  hovertemplate: "%{y:.3f}<extra>" + name + "</extra>",
});

const threshold = (x, value, label) => ({
  x, y: x.map(() => value), name: label, mode: "lines",
  line: { color: PAL.muted, width: 1, dash: "dot" }, hoverinfo: "skip",
});

const events = (x, y, name, color, symbol) => ({
  x, y, name, mode: "markers", type: "scatter",
  marker: { color, size: 13, symbol: symbol || "star", line: { color: PAL.surface, width: 2 } },
  hovertemplate: "%{x|%Y-%m-%d}<extra>" + name + "</extra>",
});

function annotated(layout, text) {
  layout.annotations.push({
    text, showarrow: false, xref: "paper", yref: "paper", x: 0.5, y: 0.5,
    font: { color: PAL.muted, size: 13 },
  });
}

/* One <section class="card"> with a title, a description, and a Plotly plot. */
function chartCard(title, desc, traces, layout) {
  const card = document.createElement("section");
  card.className = "card";
  const h = document.createElement("h3");
  h.textContent = title;
  const p = document.createElement("p");
  p.className = "desc";
  p.textContent = desc;
  const plot = document.createElement("div");
  plot.className = "plot";
  card.append(h, p, plot);
  Plotly.newPlot(plot, traces, layout, CONFIG);
  return card;
}

function statTiles(stats) {
  const tiles = [
    ["Monitoring runs", stats.runs],
    ["Retrains", stats.retrains],
    ["Promotions", stats.promotions],
    ["Latest PSI", stats.latest_psi.toFixed(2)],
    ["Latest champion R²", stats.latest_r2 == null ? "—" : stats.latest_r2.toFixed(2)],
  ];
  const box = document.getElementById("tiles");
  box.innerHTML = "";
  for (const [k, v] of tiles) {
    const t = document.createElement("div");
    t.className = "tile";
    t.innerHTML = `<div class="tile-v">${v}</div><div class="tile-k">${k}</div>`;
    box.appendChild(t);
  }
}

function renderProfile(p) {
  document.getElementById("story").textContent = p.story;
  statTiles(p.stats);
  const charts = document.getElementById("charts");
  charts.innerHTML = "";
  const xEnd = p.as_of[p.as_of.length - 1];

  // 1. Data drift — PSI per feature
  let lay = baseLayout("PSI");
  driftRegion(lay, p.drift_date, xEnd);
  let traces = FEATURES.filter((f) => p.psi[f]).map((f, i) => line(p.as_of, p.psi[f], f, PAL.series[i]));
  traces.push(threshold(p.as_of, PSI_SIGNIFICANT, `significant (${PSI_SIGNIFICANT})`));
  charts.appendChild(chartCard(
    "Data drift",
    "How far each feature's recent readings have drifted from the champion's training " +
      "window, measured by PSI. The dotted line is the 0.25 “significant shift” mark.",
    traces, lay));

  // 2. Performance drift — champion RMSE ratio
  lay = baseLayout("ratio");
  driftRegion(lay, p.drift_date, xEnd);
  traces = [
    line(p.as_of, p.perf_ratio, "perf drift ratio", PAL.series[0]),
    threshold(p.as_of, PERF_THRESHOLD, `retrain trigger (${PERF_THRESHOLD})`),
  ];
  if (p.retrain.as_of.length) {
    traces.push(events(p.retrain.as_of, p.retrain.perf, "retrain triggered", PAL.warn, "circle"));
  }
  charts.appendChild(chartCard(
    "Performance drift & retrains",
    "The champion's error on fresh data ÷ its error at training time. Crossing 1.25 — the " +
      "model is 25% worse — triggers a retrain (marked).",
    traces, lay));

  // 3. Champion vs. challenger on the held-out window
  lay = baseLayout("RMSE");
  if (p.holdout.as_of.length) {
    driftRegion(lay, p.drift_date, xEnd);
    traces = [
      line(p.holdout.as_of, p.holdout.champion, "champion", PAL.series[0]),
      line(p.holdout.as_of, p.holdout.challenger, "challenger", PAL.series[1]),
    ];
    if (p.promoted.as_of.length) {
      traces.push(events(p.promoted.as_of, p.promoted.challenger, "promoted", PAL.good));
    }
  } else {
    traces = [];
    annotated(lay, "No challenger trained yet — no retrain has fired.");
  }
  charts.appendChild(chartCard(
    "Champion vs. challenger",
    "When a retrain fires, both models are scored on a held-out week neither has seen. The " +
      "challenger takes over only if it wins by a margin (★ = promoted).",
    traces, lay));

  // 4. Coefficient evolution
  lay = baseLayout("coefficient");
  lay.hovermode = "closest";
  if (p.coef) {
    traces = FEATURES.map((f, i) => ({
      x: p.coef.train_end, y: p.coef[f], name: f, mode: "lines+markers", type: "scatter",
      line: { color: PAL.series[i], width: 2 },
      marker: { size: 8, color: PAL.series[i], line: { color: PAL.surface, width: 2 } },
      hovertemplate: "%{y:.3f}<extra>" + f + "</extra>",
    }));
    lay.shapes.push({
      type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 0, y1: 0,
      line: { color: PAL.line, width: 1 },
    });
  } else {
    traces = [];
    annotated(lay, "Only one model version so far.");
  }
  charts.appendChild(chartCard(
    "Model coefficients",
    "The Ridge model's learned slope per feature across versions. A slope crossing zero is the " +
      "real-world relationship inverting — the concept drift each retrain has to chase.",
    traces, lay));
}

function renderDataLinks(data) {
  const raw = data.raw_data;
  const parts = [];
  if (raw) {
    parts.push(
      `<strong>Raw data:</strong> <a href="${raw.file}">${raw.rows.toLocaleString()} hourly ` +
        `Kraków observations</a> (${raw.start} → ${raw.end}, CSV)`);
  }
  parts.push(`<strong>Chart data:</strong> <a href="data.json">data.json</a>`);
  document.getElementById("data-links").innerHTML = parts.join(" &nbsp;·&nbsp; ");
}

async function main() {
  let data;
  try {
    const resp = await fetch("data.json", { cache: "no-cache" });
    if (!resp.ok) throw new Error(`data.json → HTTP ${resp.status}`);
    data = await resp.json();
  } catch (e) {
    document.getElementById("error").textContent =
      "Couldn't load data.json (" + e.message + "). If viewing locally, serve the folder over HTTP.";
    return;
  }

  document.getElementById("built").textContent = `Snapshot · built ${data.built}`;
  renderDataLinks(data);

  const sel = document.getElementById("profile");
  const byKey = {};
  for (const p of data.profiles) {
    byKey[p.key] = p;
    const opt = document.createElement("option");
    opt.value = p.key;
    opt.textContent = p.label;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", (e) => renderProfile(byKey[e.target.value]));
  if (data.profiles.length) renderProfile(data.profiles[0]);
}

document.addEventListener("DOMContentLoaded", main);
