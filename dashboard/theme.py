"""One home for chart colors and the shared plotly layout.

Palette slots are assigned in fixed order and never cycled; thresholds and
status use the reserved status palette so they can never impersonate a series.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from driftloop.config import DRIFT_FEATURES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots, in documented order. Same hues and same order as the
# published site's light palette (site/app.js), so a feature keeps its colour
# whichever UI you are reading.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# A feature's colour, fixed by its position in the drift-feature list rather
# than hardcoded. This was a three-entry dict once, and it stayed that way when
# the model widened to six drift features, which crashed every chart that looked
# a newer feature up. Deriving it means the map cannot fall behind the config.
if len(SERIES) < len(DRIFT_FEATURES):  # pragma: no cover - guards a config edit
    raise RuntimeError(
        f"palette has {len(SERIES)} slots for {len(DRIFT_FEATURES)} drift features; "
        "add hues to SERIES (and to site/app.js) before adding features"
    )
FEATURE_COLOR = {feature: SERIES[i] for i, feature in enumerate(DRIFT_FEATURES)}

# Reserved status colors -- thresholds and decisions only, never a series.
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

# A faint wash to shade the post-drift era. Tinted from the "serious" status hue,
# kept well below any series so it reads as background, not data.
DRIFT_WASH = "rgba(236, 131, 90, 0.07)"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def base_figure(title: str | None, y_title: str, height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)) if title else None,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_SECONDARY),
        margin=dict(l=56, r=24, t=56 if title else 24, b=40),
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK_SECONDARY)),
    )
    fig.update_xaxes(
        showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED), title=None, zeroline=False
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(color=MUTED, size=11)),
        gridcolor=GRID,
        linecolor=AXIS,
        tickfont=dict(color=MUTED),
        zeroline=False,
    )
    return fig


def drift_region(fig: go.Figure, drift_date, x_end) -> None:
    """Shade the post-regime-shift era and mark the boundary. The single most
    orienting thing on every time-series here: 'everything right of this line is
    the new world the champion wasn't trained on'."""
    if drift_date is None:
        return
    drift_date = pd.Timestamp(drift_date)
    # Pass datetimes to plotly as ISO strings -- the most reliable form for
    # add_vrect/add_vline across versions on a date axis.
    fig.add_vrect(
        x0=drift_date.isoformat(),
        x1=pd.Timestamp(x_end).isoformat(),
        fillcolor=DRIFT_WASH,
        line_width=0,
        layer="below",
    )
    fig.add_vline(x=drift_date.isoformat(), line=dict(color=MUTED, width=1, dash="dash"))
    fig.add_annotation(
        x=drift_date,
        y=1.0,
        yref="paper",
        text="regime shift",
        showarrow=False,
        font=dict(color=MUTED, size=10),
        xanchor="left",
        xshift=4,
        yanchor="top",
    )


def line(fig: go.Figure, x, y, name: str, color: str, dash: str | None = None) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=name,
            mode="lines",
            line=dict(color=color, width=2, dash=dash),
            hovertemplate="%{y:.3f}<extra>" + name + "</extra>",
        )
    )


def threshold(fig: go.Figure, x, value: float, label: str) -> None:
    """A flat reference line in the muted ink, labelled at its right end."""
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[value] * len(x),
            name=label,
            mode="lines",
            line=dict(color=MUTED, width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=True,
        )
    )


def events(fig: go.Figure, x, y, name: str, color: str, symbol: str = "star") -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=name,
            mode="markers",
            marker=dict(color=color, size=13, symbol=symbol, line=dict(color=SURFACE, width=2)),
            hovertemplate="%{x|%Y-%m-%d}<extra>" + name + "</extra>",
        )
    )


def coef_small_multiples(
    versions: pd.DataFrame, features: list[str], ncols: int = 3
) -> go.Figure:
    """One small panel per feature: its coefficient across versions (x = train_end).

    Small multiples rather than one chart with a line per feature, because the
    coefficients are in *original* feature units and so are not comparable to
    each other. In Kraków precipitation moves across ~27 units while shortwave
    radiation moves across ~0.05 -- a factor of 500 -- so on one shared y-axis
    every small-unit feature is pinned flat against zero and the chart says
    nothing about four of the six.

    Giving each feature its own axis keeps the slope readable in the unit that
    makes it interpretable (PM2.5 per °C, per m/s, per hPa), and the thing this
    chart exists to show -- a slope crossing zero, meaning the real-world
    relationship inverted -- reads per panel instead of against a common scale.

    A 2px surface ring on the markers keeps overlapping points legible.
    """
    nrows = -(-len(features) // ncols)  # ceil
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        subplot_titles=list(features),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    for i, feature in enumerate(features):
        row, col = divmod(i, ncols)
        row, col = row + 1, col + 1
        color = FEATURE_COLOR.get(feature, SERIES[i % len(SERIES)])
        fig.add_trace(
            go.Scatter(
                x=versions["train_end"],
                y=versions[f"coef_{feature}"],
                name=feature,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color, line=dict(color=SURFACE, width=2)),
                customdata=versions["version"],
                hovertemplate="v%{customdata}: %{y:.3f}<extra>" + feature + "</extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        # Zero is the only line worth drawing here: crossing it is the event.
        fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=row, col=col)

    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_SECONDARY),
        margin=dict(l=48, r=24, t=40, b=36),
        height=190 * nrows + 60,
        hovermode="closest",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED, size=10), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED, size=10), zeroline=False)
    for annotation in fig.layout.annotations:  # the subplot titles
        annotation.font = dict(family=FONT, size=12, color=INK)
    return fig


def hist_overlay(edges: list[float], reference: list[int], current: list[int], color: str) -> go.Figure:
    """Reference vs current distribution of one feature, shared bins.

    Reference is a hollow outline (what the model knew); current is a filled bar
    in the feature's colour (what the world looks like now). Where they diverge
    is the drift the PSI number is summarising.
    """
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    width = [(edges[i + 1] - edges[i]) * 0.92 for i in range(len(edges) - 1)]
    ref = _normalize(reference)
    cur = _normalize(current)

    fig = base_figure(None, "share of rows", height=260)
    fig.update_layout(hovermode="closest", bargap=0)
    fig.add_trace(
        go.Bar(
            x=centers, y=ref, width=width, name="training window",
            marker=dict(color="rgba(0,0,0,0)", line=dict(color=MUTED, width=1.5)),
            hovertemplate="%{y:.1%}<extra>training</extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=centers, y=cur, width=width, name="latest window",
            marker=dict(color=color, line=dict(color=SURFACE, width=1)), opacity=0.75,
            hovertemplate="%{y:.1%}<extra>latest</extra>",
        )
    )
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=1.02, x=0))
    return fig


def scatter_fit(predicted, actual, color: str) -> go.Figure:
    """Predicted vs actual, with the y=x ideal line. Points hugging the diagonal
    = a good fit; a systematic lean off it = the model failing on this window."""
    lo = float(min(min(predicted), min(actual)))
    hi = float(max(max(predicted), max(actual)))
    fig = base_figure(None, "actual PM2.5", height=360)
    fig.update_layout(hovermode="closest")
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines", name="perfect",
            line=dict(color=MUTED, width=1, dash="dash"), hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predicted, y=actual, mode="markers", name="hours",
            marker=dict(color=color, size=6, opacity=0.5, line=dict(color=SURFACE, width=0.5)),
            hovertemplate="pred %{x:.1f} · actual %{y:.1f}<extra></extra>",
        )
    )
    fig.update_xaxes(title=dict(text="predicted PM2.5", font=dict(color=MUTED, size=11)))
    return fig


def residual_series(timestamp, residual, color: str) -> go.Figure:
    """Residual (actual - predicted) over the window. A flat cloud around zero is
    healthy; a drift away from zero means the model is biased on this window."""
    fig = base_figure(None, "residual", height=240)
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.add_trace(
        go.Scatter(
            x=timestamp, y=residual, mode="markers",
            marker=dict(color=color, size=5, opacity=0.5),
            hovertemplate="%{x|%b %d %H:%M}: %{y:.1f}<extra></extra>", name="residual",
        )
    )
    return fig


def _normalize(counts: list[int]) -> list[float]:
    total = sum(counts) or 1
    return [c / total for c in counts]
