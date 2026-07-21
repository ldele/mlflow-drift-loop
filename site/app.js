/* Client-side renderer for the drift-loop dashboard.
 * Fetches data.json (written by scripts/build_site.py) and builds interactive
 * Plotly charts. Palette + chart shapes mirror dashboard/theme.py so the static
 * site matches the Streamlit app. */

const PAL = {
  surface: "#fcfcfb", ink: "#0b0b0b", ink2: "#52514e", muted: "#898781",
  grid: "#e1e0d9", line: "#c3c2b7",
  series: ["#2a78d6", "#008300", "#e87ba4", "#eda100"],
  good: "#0ca30c", warn: "#fab219", crit: "#d03b3b",
  drift: "rgba(236, 131, 90, 0.07)",
};
const FEATURES = ["temperature", "wind_speed", "humidity"];
const FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif';
const CONFIG = { displayModeBar: false, responsive: true };
const PSI_SIGNIFICANT = 0.25;
const PERF_THRESHOLD = 1.25;

function baseLayout(title, yTitle) {
  return {
    title: { text: title, font: { size: 15, color: PAL.ink } },
    paper_bgcolor: PAL.surface, plot_bgcolor: PAL.surface,
    font: { family: FONT, size: 12, color: PAL.ink2 },
    margin: { l: 56, r: 24, t: 48, b: 40 }, height: 340, hovermode: "x unified",
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

/* One <div class="chart"> holding a Plotly plot. */
function chartDiv(traces, layout) {
  const wrap = document.createElement("div");
  wrap.className = "chart";
  const plot = document.createElement("div");
  wrap.appendChild(plot);
  Plotly.newPlot(plot, traces, layout, CONFIG);
  return wrap;
}

function statTiles(stats) {
  const tiles = [
    ["Scheduled runs", stats.runs],
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
  let lay = baseLayout("Data drift — PSI per feature vs. the champion's training window", "PSI");
  driftRegion(lay, p.drift_date, xEnd);
  let traces = FEATURES.filter((f) => p.psi[f]).map((f, i) => line(p.as_of, p.psi[f], f, PAL.series[i]));
  traces.push(threshold(p.as_of, PSI_SIGNIFICANT, `significant (${PSI_SIGNIFICANT})`));
  charts.appendChild(chartDiv(traces, lay));

  // 2. Performance drift — champion RMSE ratio
  lay = baseLayout("Performance drift — champion RMSE now ÷ RMSE at training time", "ratio");
  driftRegion(lay, p.drift_date, xEnd);
  traces = [
    line(p.as_of, p.perf_ratio, "perf drift ratio", PAL.series[0]),
    threshold(p.as_of, PERF_THRESHOLD, `retrain trigger (${PERF_THRESHOLD})`),
  ];
  if (p.retrain.as_of.length) {
    traces.push(events(p.retrain.as_of, p.retrain.perf, "retrain triggered", PAL.warn, "circle"));
  }
  charts.appendChild(chartDiv(traces, lay));

  // 3. Champion vs. challenger on the held-out window
  lay = baseLayout("Champion vs. challenger on the held-out window (RMSE)", "RMSE");
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
  charts.appendChild(chartDiv(traces, lay));

  // 4. Coefficient evolution
  lay = baseLayout("Coefficient evolution — the model's slopes across versions", "coefficient");
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
  charts.appendChild(chartDiv(traces, lay));
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

  document.getElementById("built").innerHTML =
    `Static snapshot · built ${data.built} · rebuilt weekly by GitHub Actions`;

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
