"""
View 2 — Parallel coordinates.
Today's z-scored factor state (bold) + top-5 fingerprint-matched historical
days (faded), color-coded by stock performance in the following 10 days.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math_engine.pipeline import rolling_zscore, as_of


STOCK_LABELS = {
    "nifty50":    "NIFTY 50",
    "bank_nifty": "BANK NIFTY",
    "icici_bank": "ICICI Bank",
}


def _ret_color(fwd_ret: float | None) -> str:
    if fwd_ret is None:
        return "rgba(150,150,150,0.4)"
    return "rgba(34,139,34,0.35)" if fwd_ret > 0 else "rgba(200,30,30,0.35)"


def render_parallel_coords(
    factors_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    matches: list[dict],
    date,
    pca_window: int = 126,
    focus_stock: str = "nifty50",
    forward_days: int = 10,
) -> go.Figure:
    date = pd.Timestamp(date)

    # Today's z-scored factor state
    z_df  = rolling_zscore(as_of(factors_df, date), pca_window)
    if len(z_df) == 0 or z_df.iloc[-1].isna().all():
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for z-score at this date.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    today_z   = z_df.iloc[-1]
    all_factors = today_z.index.tolist()
    valid_cols  = [c for c in all_factors if pd.notna(today_z[c])]

    if not valid_cols:
        fig = go.Figure()
        fig.add_annotation(text="No valid z-scores for this date.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Axis positions (0 to 1, evenly spaced)
    n_axes   = len(valid_cols)
    x_pos    = list(range(n_axes))

    fig = go.Figure()

    # Historical fingerprint matches (faded polylines)
    for m in matches:
        match_date  = m["date"]
        fwd_ret_10d = m.get("per_stock_fwd", {}).get(focus_stock, {}).get(forward_days)
        color       = _ret_color(fwd_ret_10d)

        z_match = rolling_zscore(as_of(factors_df, match_date), pca_window)
        if len(z_match) == 0:
            continue
        match_z = z_match.iloc[-1]
        y_vals  = [float(match_z.get(c, np.nan)) for c in valid_cols]

        if all(np.isnan(v) for v in y_vals):
            continue

        sim = m.get("similarity", 0.0)
        fwd_label = f"{fwd_ret_10d*100:.2f}%" if fwd_ret_10d is not None else "N/A"

        fig.add_trace(go.Scatter(
            x=x_pos, y=y_vals,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            opacity=0.6,
            name=match_date.strftime("%Y-%m-%d"),
            hovertemplate=(
                f"Match: {match_date.strftime('%Y-%m-%d')}<br>"
                f"Cosine sim: {sim:.3f}<br>"
                f"{STOCK_LABELS.get(focus_stock, focus_stock)} +{forward_days}d: {fwd_label}"
                "<extra></extra>"
            ),
        ))

    # Today's state (bold)
    today_y = [float(today_z.get(c, np.nan)) for c in valid_cols]
    fig.add_trace(go.Scatter(
        x=x_pos, y=today_y,
        mode="lines+markers",
        line=dict(color="rgba(30,30,200,1.0)", width=3.5),
        marker=dict(size=8, color="rgba(30,30,200,1.0)", symbol="circle"),
        name=f"Today ({date.strftime('%Y-%m-%d')})",
        hovertemplate="<b>Today: %{x}</b><br>Z-score: %{y:.3f}<extra></extra>",
    ))

    # Axes as vertical dashed reference lines + labels
    for i, col in enumerate(valid_cols):
        fig.add_shape(
            type="line", x0=i, x1=i, y0=-3.5, y1=3.5,
            line=dict(color="lightgray", width=1),
        )

    # Zero line
    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.2)", width=1, dash="dash"))
    fig.add_hline(y=2, line=dict(color="rgba(200,30,30,0.3)", width=1, dash="dot"))
    fig.add_hline(y=-2, line=dict(color="rgba(34,139,34,0.3)", width=1, dash="dot"))

    stock_label = STOCK_LABELS.get(focus_stock, focus_stock)
    fig.update_layout(
        title=(
            f"Parallel Coordinates — {date.strftime('%Y-%m-%d')}<br>"
            f"<sup>Bold = today | Faded lines = top fingerprint matches | "
            f"Color = {stock_label} return {forward_days}d after match "
            f"(green=up, red=down)</sup>"
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=x_pos,
            ticktext=[c.replace("_", "<br>") for c in valid_cols],
            tickangle=0,
            tickfont=dict(size=10),
        ),
        yaxis=dict(title="Z-score", range=[-4, 4], zeroline=False),
        height=520,
        showlegend=True,
        legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
        margin=dict(t=80, b=120),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
