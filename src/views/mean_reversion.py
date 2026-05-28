"""
View 4 — Per-stock mean reversion.
Price chart (candles or line) with factor-implied mean, ±2σ bands, signal markers.
Right-panel marginal histogram + KDE of residuals.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

STOCK_LABELS = {
    "nifty50":    "NIFTY 50",
    "bank_nifty": "BANK NIFTY",
    "icici_bank": "ICICI Bank",
}


def render_mean_reversion(
    ohlcv_df: pd.DataFrame,         # OHLCV for the selected stock
    stock_returns: pd.Series,       # log-return series
    residuals: pd.Series,           # rolling OLS residuals
    mr_sigma: pd.Series,            # rolling 15-day std of residuals
    signal: pd.Series,              # bool series: |residual| > 2σ
    stock: str,
    date,
) -> go.Figure:
    date = pd.Timestamp(date)

    # Slice to selected date (no lookahead for display)
    ohlcv   = ohlcv_df[ohlcv_df.index <= date].copy()
    resid   = residuals[residuals.index <= date].copy()
    sigma   = mr_sigma[mr_sigma.index <= date].copy()
    sig_flags = signal[signal.index <= date].copy()

    close   = ohlcv["Close"].dropna()
    has_ohlc = all(c in ohlcv.columns for c in ["Open", "High", "Low", "Close"])

    if len(close) < 2 or resid.dropna().empty:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for this stock / date.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Factor-implied mean price = close × exp(-residual)
    # This removes the regression residual from each day's close, giving the
    # "fair value" price according to the factor model at that point in time.
    aligned_resid = resid.reindex(close.index)
    factor_mean   = close * np.exp(-aligned_resid)

    # ±2σ bands around factor mean (σ is in log-return space)
    aligned_sigma = sigma.reindex(close.index)
    upper_band    = factor_mean * np.exp(2.0 * aligned_sigma)
    lower_band    = factor_mean * np.exp(-2.0 * aligned_sigma)

    # Signal days
    signal_dates  = sig_flags[sig_flags == True].index
    signal_dates  = signal_dates[signal_dates.isin(close.index)]
    signal_prices = close.reindex(signal_dates)

    # Residual distribution (all history up to date)
    resid_vals = resid.dropna().values

    # ── Layout: price chart (left, wide) + marginal histogram (right, narrow) ─
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.80, 0.20],
        row_heights=[0.68, 0.32],
        shared_xaxes=False,
        shared_yaxes=False,
        specs=[
            [{"type": "xy"},               {"type": "xy", "rowspan": 2}],
            [{"type": "xy"},               None],
        ],
        vertical_spacing=0.04,
        horizontal_spacing=0.03,
    )

    # ── Price panel (row 1, col 1) ────────────────────────────────────────────
    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=ohlcv.index, open=ohlcv["Open"], high=ohlcv["High"],
            low=ohlcv["Low"],   close=ohlcv["Close"],
            name="Price", increasing_line_color="#26A69A",
            decreasing_line_color="#EF5350", showlegend=True,
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=close.index, y=close.values,
            mode="lines", name="Close",
            line=dict(color="#1565C0", width=1.5),
        ), row=1, col=1)

    # Factor-implied mean line
    fig.add_trace(go.Scatter(
        x=factor_mean.index, y=factor_mean.values,
        mode="lines", name="Factor-implied mean",
        line=dict(color="rgba(255,152,0,0.85)", width=2.0, dash="solid"),
    ), row=1, col=1)

    # ±2σ bands
    band_x = list(upper_band.index) + list(lower_band.index[::-1])
    band_y = list(upper_band.values) + list(lower_band.values[::-1])
    fig.add_trace(go.Scatter(
        x=band_x, y=band_y,
        fill="toself",
        fillcolor="rgba(255,152,0,0.12)",
        line=dict(color="rgba(255,152,0,0.0)"),
        name="±2σ band",
        hoverinfo="skip",
    ), row=1, col=1)

    # Signal markers on price chart
    if len(signal_dates) > 0:
        resid_at_signal = aligned_resid.reindex(signal_dates)
        marker_colors   = ["#D32F2F" if v > 0 else "#388E3C" for v in resid_at_signal]
        hover_texts     = [
            f"{d.date()}: resid={aligned_resid.get(d, np.nan):.4f}, "
            f"2σ={2*aligned_sigma.get(d, np.nan):.4f}"
            for d in signal_dates
        ]
        fig.add_trace(go.Scatter(
            x=signal_dates, y=signal_prices.values,
            mode="markers",
            name="Signal (|resid|>2σ)",
            marker=dict(size=8, color=marker_colors, symbol="triangle-up",
                        line=dict(width=1, color="white")),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)

    # ── Residual panel (row 2, col 1) ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=resid.index, y=resid.values,
        mode="lines", name="Residual",
        line=dict(color="rgba(100,100,200,0.8)", width=1.2),
        showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=sigma.index, y=(2.0 * sigma).values,
        mode="lines", name="+2σ",
        line=dict(color="rgba(200,30,30,0.6)", width=1.2, dash="dash"),
        showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=sigma.index, y=(-2.0 * sigma).values,
        mode="lines", name="−2σ",
        line=dict(color="rgba(34,139,34,0.6)", width=1.2, dash="dash"),
        showlegend=False,
    ), row=2, col=1)

    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.2)", width=1), row=2, col=1)

    # Signal markers on residual panel
    if len(signal_dates) > 0:
        resid_at_sig = resid.reindex(signal_dates)
        fig.add_trace(go.Scatter(
            x=signal_dates, y=resid_at_sig.values,
            mode="markers",
            marker=dict(size=7, color=marker_colors, symbol="circle",
                        line=dict(width=1, color="white")),
            showlegend=False,
            hoverinfo="skip",
        ), row=2, col=1)

    # ── Marginal histogram + KDE (col 2, spans both rows) ────────────────────
    if len(resid_vals) > 10:
        # Histogram (horizontal: residual value on y-axis)
        fig.add_trace(go.Histogram(
            y=resid_vals,
            nbinsy=50,
            orientation="h",
            name="Residual dist.",
            marker_color="rgba(100,100,200,0.4)",
            showlegend=False,
        ), row=1, col=2)

        # KDE overlay
        try:
            kde   = gaussian_kde(resid_vals, bw_method="scott")
            y_lin = np.linspace(resid_vals.min(), resid_vals.max(), 200)
            dens  = kde(y_lin)
            # Scale KDE to histogram counts
            bin_width  = (resid_vals.max() - resid_vals.min()) / 50
            kde_scaled = dens * len(resid_vals) * bin_width
            fig.add_trace(go.Scatter(
                x=kde_scaled, y=y_lin,
                mode="lines", name="KDE",
                line=dict(color="rgba(30,30,200,0.8)", width=2),
                showlegend=False,
            ), row=1, col=2)
        except Exception:
            pass

        # ±2σ reference lines on histogram
        sigma_global = float(np.std(resid_vals))
        for sign, color in [(1, "rgba(200,30,30,0.6)"), (-1, "rgba(34,139,34,0.6)")]:
            fig.add_hline(
                y=sign * 2.0 * sigma_global,
                line=dict(color=color, width=1, dash="dash"),
                row=1, col=2,
            )

    # ── Layout ────────────────────────────────────────────────────────────────
    stock_label = STOCK_LABELS.get(stock, stock)
    fig.update_layout(
        title=f"Mean Reversion — {stock_label} (as of {date.date()})",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, font=dict(size=11)),
        margin=dict(t=70, b=60, l=60, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_rangeslider_visible=False,
    )

    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (log-ret)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Residual", row=1, col=2)

    return fig
